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
import shutil
from utils.solver import *
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

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        if param.requires_grad:
            param_size += param.nelement() * param.element_size()
    return param_size / 1024**2

def main():
    # setup config
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/default.yaml')
    parser.add_argument('--exp_time', default='00000000_000000')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config["working_dir"] = os.path.join(config["output_dir"], config["exp_name"] + "_" + args.exp_time)
    os.makedirs(config["working_dir"], exist_ok=True)
    shutil.copy(args.config, config["working_dir"])

    #init wandb
    if config["wandb"]:
        wandb.init(project="activity_estimation", name='"activity_estimation"_{}_{}_{}'.format("Charades", config["exp_name"], args.exp_time))

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
    fusion_module = FusionTransformer(clip_length=config["clip_length"], embed_dim=config["emb_dim"], n_layers=config["fusion_layers"], vsfusion=config["vs_fusion"])
    models = [text_encoder, att_module, fusion_module]
    if config["wandb"]:
        wandb.watch(text_encoder, att_module, fusion_module)

    print("-"*80)
    print("model size")
    total_size = 0
    for model in models:
        model_size = get_model_size(model)
        total_size += model_size
        print("{}: {:.2f}MB".format(model.__class__.__name__, model_size))
    print("total size: {:.2f}MB".format(total_size))
    print("-"*80)

    # training setting
    n_trains = len(Charades_train)
    config["train"]["steps_per_epoch"] = n_trains // config["train"]["gradient_acc"]
    optimizer = build_optimizer(config, models)
    lr_scheduler = build_lr_scheduler(config, optimizer)
    criterion = SimLoss()
    
    #TODO: multigpu training
    # training
    models = train(trainloader, valloader, models, criterion, optimizer, lr_scheduler, config)

    # testing
    test("test", testloader, models, criterion, config)


def train(trainloader, valloader, models, criterion, optimizer, lr_scheduler, config):
    print("-"*80)
    print("training phase")
    text_encoder, att_module, fusion_module = models
    text_encoder = text_encoder.cuda()
    att_module = att_module.cuda()
    fusion_module = fusion_module.cuda()
    optimizer.zero_grad()

    classes_dict = trainloader.dataset.classes
    script_tokens_all = trainloader.dataset.script_tokens.cuda()
    epochs = config["train"]["epochs"]
    val_freq = config["train"]["val_freq"]

    for epoch in range(epochs):
        print("-"*80)
        print("{}/{} training epochs".format(epoch, epochs))
        text_encoder.train()
        att_module.train()
        fusion_module.train()
        script_emb_buffer = []
        video_emb_buffer = []
        torch.cuda.empty_cache()

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
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                print("Training step:{} lr: {} Training loss: {:.3f}".format(step+1, np.format_float_scientific(optimizer.param_groups[0]["lr"]), loss.item()))
                record = {"epoch": epoch, "training loss": loss.item(), "lr":  optimizer.param_groups[0]["lr"]}
                if config["wandb"]: 
                    wandb.log(record)

                script_emb_buffer.clear()
                video_emb_buffer.clear()
                torch.cuda.empty_cache()

        # validate
        if (epoch + 1) % val_freq == 0:
            print("-"*80)
            print("{}/{} val epochs".format(epoch // val_freq, epochs // val_freq))

            test("val", valloader, [text_encoder, att_module, fusion_module], criterion, config, cur_epoch = epoch)
            # save checkpoint
            torch.save({
        'epoch': epoch,
        'text_encoder_state_dict': text_encoder.state_dict(),
        'att_module_state_dict': att_module.state_dict(),
        'fusion_module_state_dict': fusion_module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(config["working_dir"], "epoch_{}.pt".format(epoch)))

        torch.cuda.empty_cache()

    print("training end")
    print("-"*80)
    return [text_encoder, att_module, fusion_module]


@torch.no_grad()
def test(mode, testloader, models, criterion, config, cur_epoch=999):
    assert mode in ["val", "test"]
    print("-"*80)
    print("{} phase".format(mode))
    text_encoder, att_module, fusion_module = models
    text_encoder = text_encoder.cuda().eval()
    att_module = att_module.cuda().eval()
    fusion_module = fusion_module.cuda().eval()

    classes_dict = testloader.dataset.classes
    script_tokens_all = testloader.dataset.script_tokens.cuda()
    loss_buffer, top1_acc_buffer, top5_acc_buffer = [], [], []

    for _, (vfeatures, actions, label) in enumerate(tqdm(testloader)):
        vfeatures = vfeatures.cuda()
        label = label.cuda()
        prompted_texts, _ = generate_prompted_text(actions, classes_dict)
        prompted_texts = [x.cuda() for x in prompted_texts]
        prompted_texts_emb = [text_encoder(text) for text in prompted_texts]
        vfeatures, sfeatures = att_module(vfeatures, prompted_texts_emb)
        video_emb = fusion_module(vfeatures, sfeatures)

        loss, pred_1, pred_5 = get_testing_meters(video_emb, script_tokens_all, label, text_encoder, criterion)
        loss_buffer.append(loss.item())
        top1_acc_buffer.append(pred_1)
        top5_acc_buffer.append(pred_5)

    total_loss = sum(loss_buffer) / len(loss_buffer)
    top1_acc = sum(top1_acc_buffer) / len(top1_acc_buffer)
    top5_acc = sum(top5_acc_buffer) / len(top5_acc_buffer)

    record = {"{} loss".format(mode): total_loss, "{} top1 acc".format(mode): top1_acc, "{} top5 acc".format(mode): top5_acc}
    if mode == "val":
        record["epoch"] = cur_epoch
    if config["wandb"]:
        wandb.log(record)

    print("{}: \n loss: {:.3f} \n top1_acc: {:.2f} \n top5_acc: {:.2f}".format(mode, total_loss, top1_acc, top5_acc))

    print("{} end".format(mode))
    print("-"*80)

if __name__ == '__main__':
    main()