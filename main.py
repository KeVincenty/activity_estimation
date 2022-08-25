import os
from turtle import forward
import clip
from clip.model import TextTransformer, FusionTransformer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import CharadesFeatures
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.text_prompt import *

class TextEncoder(nn.Module):
    def __init__(self, model):
        super(TextEncoder, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

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

class KLLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.error_metric = nn.KLDivLoss(size_average=True, reduce=True)

    def forward(self, ve, se, label):
        breakpoint()
        # normalized features
        ve = ve / ve.norm(dim=-1, keepdim=True)
        se = se / se.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_v = logit_scale * ve @ se.t()

        probs1 = F.log_softmax(logits_v, 1)
        loss = self.error_metric(probs1, label)
        return loss

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

    # build dataloader
    Charades_train = CharadesFeatures(mode="train")
    Charades_val = CharadesFeatures(mode="val")
    # Charades_test = CharadesFeatures(mode="test")
    trainloader = DataLoader(Charades_train, batch_size = config["train"]["batch_size"], shuffle = config["train"]["shuffle"], num_workers = config["train"]["num_workers"])
    valloader = DataLoader(Charades_val, batch_size = 1, shuffle = False, num_workers = config["train"]["num_workers"])
    # testloader = DataLoader(Charades_test, batch_size = 1, shuffle = False, num_workers = config["train"]["num_workers"])

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
    criterion = KLLoss()
    
    # training
    models = train(trainloader, valloader, models, criterion, optimizer, lr_scheduler, config)

    # testing
    # test(testloader, models, criterion, config)


def train(trainloader, valloader, models, criterion, optimizer, lr_scheduler, config):
    text_encoder, att_module, fusion_module = models
    text_encoder = text_encoder.train().cuda()
    att_module = att_module.train().cuda()
    fusion_module = fusion_module.train().cuda()

    classes_dict = trainloader.dataset.classes
    script_tokens = trainloader.dataset.script_tokens.cuda()
    epochs = config["train"]["epochs"]
    val_freq = config["train"]["val_freq"]

    for epoch in range(epochs):
        print("-"*80)
        print("{}/{} training epochs".format(epoch, epochs))
        for _, (vfeatures, actions, label) in enumerate(tqdm(trainloader)):
            vfeatures = vfeatures.cuda()
            prompted_texts, _ = generate_prompted_text(actions, classes_dict)
            prompted_texts = [x.cuda() for x in prompted_texts]
            prompted_texts_emb = [text_encoder(text) for text in prompted_texts]
            script_tokens_chunks = script_tokens.chunk(100)
            script_emd_chunks = []
            for script_tokens in script_tokens_chunks:
                script_emd_chunks.append(text_encoder(script_tokens))
            script_emb = torch.cat(script_emd_chunks)
            vfeatures, sfeatures = att_module(vfeatures, prompted_texts_emb)
            video_emb = fusion_module(vfeatures, sfeatures)

            loss = criterion(video_emb, script_emb, label)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

        print("Training loss: {}".format(loss.item()))

        if epoch % val_freq == 0:
            test()
            torch.save({
        'epoch': epoch,
        'text_encoder_state_dict': text_encoder.state_dict(),
        'att_module_state_dict': att_module.state_dict(),
        'fusion_module_state_dict': fusion_module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, config["working_dir"])

        print("-"*80)



def test(testloader, models, criterion, config):
    pass
            

if __name__ == '__main__':
    main()