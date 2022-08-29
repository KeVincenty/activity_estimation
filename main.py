import os
from modules.model import TextTransformer, FusionTransformer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.datasets import CharadesFeatures
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import yaml
from utils.solver import _optimizer, _lr_scheduler
from utils.text_prompt import *

class AttModule(nn.Module):
    def __init__(self, in_features, text_dim, emb_dim, n_head = 8, dropout = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, emb_dim)
        self.fc2 = nn.Linear(3*2*2*emb_dim, emb_dim)
        self.actv = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.attn = nn.MultiheadAttention(emb_dim, n_head, kdim=text_dim, vdim=text_dim)

    def forward(self, v, t):
        assert v.shape[0] == 1 and v.shape[1] == len(t)
        v = v.squeeze(0)
        v = self.dropout(self.actv(self.fc1(v)))
        v = v.flatten(start_dim=1)
        v = self.actv(self.fc2(v))
        sfeatures = []
        for idx, tt in enumerate(t):
            vv = v[idx].unsqueeze(0)
            attn_out, _ = self.attn(vv, tt, tt)
            sfeatures.append(attn_out)
        return v, torch.cat(sfeatures)

class SimLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.error_metric = nn.CrossEntropyLoss(size_average=True, reduce=True)

    def get_logits(self, ve, se):
        # normalized features
        ve = ve / ve.norm(dim=-1, keepdim=True)
        se = se / se.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_v = logit_scale * ve @ se.t()
        logits_i = logit_scale * se @ ve.t()   
        return logits_v, logits_i    

    def get_loss(self, logits_v, logits_i, label):
        loss_v = self.error_metric(logits_v, label)
        loss_i = self.error_metric(logits_i, label.t())
        return (loss_v + loss_i) / 2.

    def forward(self, ve, se, label):
        logits_v, logits_i = self.get_logits(ve, se)
        return self.get_loss(logits_v, logits_i, label)

def get_training_loss(script_emb_buffer, video_emb_buffer, criterion):
    assert len(script_emb_buffer) == len(video_emb_buffer)
    num_video = len(script_emb_buffer)
    se = torch.cat(script_emb_buffer)
    ve = torch.cat(video_emb_buffer)
    label = torch.eye(num_video).cuda()
    return criterion(ve, se, label)

@torch.no_grad()
def get_testing_meters(video_emb, script_tokens_all, label, text_encoder, criterion):
    script_tokens_chunks = script_tokens_all.chunk(100)
    logits_v_buffer = []
    logits_i_buffer = []
    for script_tokens in script_tokens_chunks:
        script_emb = text_encoder(script_tokens)
        logits_v, logits_i = criterion.get_logits(video_emb, script_emb)
        logits_v_buffer.append(logits_v)
        logits_i_buffer.append(logits_i)
    logits_v_all = torch.cat(logits_v_buffer, dim=-1)
    logits_i_all = torch.cat(logits_i_buffer, dim=0)
    label = label.squeeze(-1)
    loss = criterion.get_loss(logits_v_all, logits_i_all, label)
    _, pred_1 = logits_v_all.topk(1)
    _, pred_5 = logits_v_all.topk(5)
    pred_1 = pred_1.item() == label.nonzero().sum().item()
    pred_5 = label.nonzero().sum().item() in pred_5[0].tolist()
    return loss, pred_1, pred_5
    

def get_pretrain_model(path):
    try:
        model_state_dict = torch.jit.load(path, map_location="cpu").state_dict()
    except Exception:
        model_state_dict = torch.load(path, map_location="cpu")

    for k in list(model_state_dict.keys()):
        if k[:6] == "visual" or k in ["logit_scale", "input_resolution", "context_length", "vocab_size", "text_projection"]:
            del model_state_dict[k]
        elif k[:11] == "transformer":
            newk = k[12:]
            model_state_dict[newk] = model_state_dict.pop(k)

    return model_state_dict

def main():
    # setup config
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/default.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config["working_dir"] = os.path.join(config["output_dir"], config["exp_name"])
    os.makedirs(config["working_dir"], exist_ok=True)

    # build dataloader
    Charades_train = CharadesFeatures(mode="train")
    Charades_val = CharadesFeatures(mode="val")
    Charades_test = CharadesFeatures(mode="test")
    trainloader = DataLoader(Charades_train, batch_size = config["train"]["batch_size"], shuffle = config["train"]["shuffle"], num_workers = config["train"]["num_workers"])
    valloader = DataLoader(Charades_val, batch_size = 1, shuffle = False, num_workers = config["train"]["num_workers"])
    testloader = DataLoader(Charades_test, batch_size = 1, shuffle = False, num_workers = config["train"]["num_workers"])

    # build model
    text_encoder = TextTransformer(embed_dim=config["text_dim"])
    if config["pretrain"]:
        model_state_dict = get_pretrain_model(config["model_path"])
        text_encoder.load_state_dict(model_state_dict, strict=False)
    att_module = AttModule(config["visual_dim"], config["text_dim"], config["emb_dim"])
    fusion_module = FusionTransformer(clip_length=config["clip_length"], embed_dim=config["emb_dim"]*2)
    models = [text_encoder, att_module, fusion_module]

    print("-"*80)
    print("model details:")
    for model in models:
        for k, v in model.named_parameters():
            print('{}: {}'.format(k, v.requires_grad))
    print("-"*80)

    # training setting
    optimizer = _optimizer(config, models)
    lr_scheduler = _lr_scheduler(config, optimizer)
    criterion = SimLoss()
    
    #TODO: multigpu training
    # training
    models = train(trainloader, valloader, models, criterion, optimizer, lr_scheduler, config)

    # testing
    test(testloader, models, criterion, config)


def train(trainloader, valloader, models, criterion, optimizer, lr_scheduler, config):
    print("-"*80)
    print("training phase")
    text_encoder, att_module, fusion_module = models
    text_encoder = text_encoder.train().cuda()
    att_module = att_module.train().cuda()
    fusion_module = fusion_module.train().cuda()

    classes_dict = trainloader.dataset.classes
    script_tokens_all = trainloader.dataset.script_tokens.cuda()
    epochs = config["train"]["epochs"]
    val_freq = config["train"]["val_freq"]

    for epoch in range(epochs):
        print("-"*80)
        print("{}/{} training epochs".format(epoch, epochs))
        script_emb_buffer = []
        video_emb_buffer = []

        for step, (vfeatures, actions, label) in enumerate(tqdm(trainloader)):
            vfeatures = vfeatures.cuda()
            script_tokens = script_tokens_all[label.nonzero().sum().item()].unsqueeze(0)
            prompted_texts, _ = generate_prompted_text(actions, classes_dict)
            prompted_texts = [x.cuda() for x in prompted_texts]
            prompted_texts_emb = [text_encoder(text) for text in prompted_texts]
            script_emb = text_encoder(script_tokens)
            vfeatures, sfeatures = att_module(vfeatures, prompted_texts_emb)
            video_emb = fusion_module(vfeatures, sfeatures)

            script_emb_buffer.append(script_emb)
            video_emb_buffer.append(video_emb)

            if (step + 1) % config["train"]["gradient_acc"] == 0:
                loss = get_training_loss(script_emb_buffer, video_emb_buffer, criterion)
                print("Training loss: {:.3f}".format(loss.item()))
                #TODO: display training curve use tensorboard or wandb
                loss.backward()

                optimizer.step()
                lr_scheduler.step()

                script_emb_buffer.clear()
                video_emb_buffer.clear()

        lr_scheduler.step()

        if (epoch + 1) % val_freq == 0:
            print("-"*80)
            print("{}/{} val epochs".format(epoch // val_freq, epochs // val_freq))
            test(valloader, [text_encoder, att_module, fusion_module], criterion, config)
            torch.save({
        'epoch': epoch,
        'text_encoder_state_dict': text_encoder.state_dict(),
        'att_module_state_dict': att_module.state_dict(),
        'fusion_module_state_dict': fusion_module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(config["working_dir"], "epoch_{}.pt".format(epoch)))
            text_encoder = text_encoder.train()
            att_module = att_module.train()
            fusion_module = fusion_module.train()

    print("training end")
    print("-"*80)
    return [text_encoder, att_module, fusion_module]


@torch.no_grad()
def test(testloader, models, criterion, config):
    print("-"*80)
    print("testing phase")
    text_encoder, att_module, fusion_module = models
    text_encoder = text_encoder.eval().cuda()
    att_module = att_module.eval().cuda()
    fusion_module = fusion_module.eval().cuda()

    classes_dict = testloader.dataset.classes
    script_tokens_all = testloader.dataset.script_tokens.cuda()
    total_loss, top1_acc, top5_acc= 0., 0., 0.

    for step, (vfeatures, actions, label) in enumerate(tqdm(testloader)):
        vfeatures = vfeatures.cuda()
        label = label.cuda()
        prompted_texts, _ = generate_prompted_text(actions, classes_dict)
        prompted_texts = [x.cuda() for x in prompted_texts]
        prompted_texts_emb = [text_encoder(text) for text in prompted_texts]
        vfeatures, sfeatures = att_module(vfeatures, prompted_texts_emb)
        video_emb = fusion_module(vfeatures, sfeatures)

        loss, pred_1, pred_5 = get_testing_meters(video_emb, script_tokens_all, label, text_encoder, criterion)
        total_loss = (total_loss * step + loss.item()) / (step + 1)
        top1_acc = (top1_acc * step  + pred_1) / (step + 1)
        top5_acc = (top5_acc * step  + pred_5) / (step + 1)
        
    print("Testing: \n loss: {:.3f} \n top1_acc: {:.2f} \n top5_acc: {:.2f}".format(total_loss, top1_acc, top5_acc))
    #TODO: display testing curve use tensorboard or wandb

    print("testing end")
    print("-"*80)

if __name__ == '__main__':
    main()