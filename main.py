import os
from turtle import forward
import clip
from clip.model import TextTransformer, FusionTransformer
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
from utils.checkpoint import *

class TextEncoder(nn.Module):
    def __init__(self, model):
        super(TextEncoder, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class AttModule(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 2048)
        self.fc2 = nn.Linear(2*2*2048, 2048)
        self.attn = nn.MultiheadAttention(2048, 4, kdim=None, vdim=None)

    def forward(self, v, t):
        v = nn.Dropout(nn.ReLU(self.fc1(v)))
        v = v.flatten(start_dim=1)
        v = nn.ReLU(self.fc2(v))
        attn_out = self.attn(v, t, t)
        att_w = F.softmax(attn_out)
        return t @ att_w

class KLLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.error_metric = nn.KLDivLoss(size_average=True, reduce=True)

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

def get_pretrain_model(path):
    try:
        model_state_dict = torch.jit.load(path, map_location="cpu").state_dict()
    except Exception:
        model_state_dict = torch.load(path, map_location="cpu")

    for k in list(model_state_dict.keys()):
        if k[:6] == "visual" or k in ["logit_scale", "input_resolution", "context_length", "vocab_size"]:
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

    # build dataloader
    Charades_train = CharadesFeatures(mode="train")
    Charades_val = CharadesFeatures(mode="val")
    Charades_test = CharadesFeatures(mode="test")
    trainloader = DataLoader(Charades_train)
    valloader = DataLoader(Charades_val)
    testloader = DataLoader(Charades_test)

    # build model
    text_encoder = TextTransformer()
    if config["pretrain"]:
        model_state_dict = get_pretrain_model(config["model_path"])
        text_encoder.load_state_dict(model_state_dict)
    att_module = AttModule(config["feature_dim"])
    fusion_module = FusionTransformer(config["clip_length"])
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
    train(trainloader, valloader, models, criterion, optimizer, lr_scheduler, config)


def train(trainloader, valloader, models, criterion, optimizer, lr_scheduler, config):
    text_encoder, att_module, fusion_module = models
    text_encoder = text_encoder.train().cuda()
    att_module = att_module.train().cuda()
    fusion_module = fusion_module.train().cuda()

    classes_dict = trainloader.dataset.classes
    epochs = config["train"]["epochs"]
    val_freq = config["train"]["val_freq"]

    for epoch in range(epochs):
        print("-"*80)
        print("{}/{} training epochs".format(epoch, epochs))
        for _, (vfeatures, actions, script) in enumerate(tqdm(trainloader)):
            vfeatures = vfeatures.cuda()
            prompted_texts, script_token, _ = generate_prompted_text(actions, script, classes_dict)
            prompted_texts = [x.cuda() for x in prompted_texts]
            script_token = script_token.cuda()
            prompted_texts_emb = [text_encoder(text) for text in prompted_texts]
            script_emb = text_encoder(script_token)
            breakpoint()
            sfeatures = att_module(vfeatures, prompted_texts_emb)
            video_emb = fusion_module(vfeatures, sfeatures)

            loss = criterion(video_emb, script_emb)
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
    }, config["output_dir"])

        print("-"*80)



def test():
    pass
            

if __name__ == '__main__':
    main()