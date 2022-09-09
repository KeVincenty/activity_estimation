import os
from modules.model import TextTransformer, FusionTransformer
import numpy as np
import torch
from modules.datasets import CharadesFeatures
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
from utils.solver import *
from utils.text_prompt import *
from main import *

def inference():
    # setup config
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/default.yaml')
    parser.add_argument('--exp_time', default='00000000_000000')
    parser.add_argument('--model_path')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config["working_dir"] = os.path.join(config["output_dir"], "inference_" + config["exp_name"] + "_" + args.exp_time)
    print("experiment: {}".format(config["exp_name"]))
    print("setting: \n {}".format(config))

    # build dataloader
    Charades_test = CharadesFeatures(config,
                 root=config["root_dir"],
                 mode='test',
                 clip_len=config["data"]["clip_len"],
                 ol=None,
                 split_script=config["data"]["split_script"],
                 feature_dir=config["data"]["data_dir"],
                 label_dir=config["data"]["label_dir"],
                 class_dir=config["data"]["class_dir"])

    testloader = DataLoader(Charades_test, batch_size = 1, shuffle = False, num_workers = config["train"]["num_workers"])

    # build model
    text_encoder = TextTransformer(embed_dim=config["text_dim"])
    att_module = AttModule(config["visual_dim"], config["text_dim"], config["emb_dim"], plain_attn=config["plain_attn"])
    fusion_module = FusionTransformer(clip_length=config["clip_length"], embed_dim=config["emb_dim"], n_layers=config["fusion_layers"], batch_size=config["train"]["batch_size"]*config["train"]["gradient_acc"], vsfusion=config["vs_fusion"])
    model_state_dict = get_pretrain_model(args.model_path)
    text_encoder.load_state_dict(model_state_dict["text_encoder_state_dict"], strict=False)
    att_module.load_state_dict(model_state_dict["att_module_state_dict"], strict=False)
    fusion_module.load_state_dict(model_state_dict["fusion_module_state_dict"], strict=False)
    models = [text_encoder, att_module, fusion_module]

    print("-"*80)
    print("model size")
    total_size = 0
    for model in models:
        model_size = get_model_size(model)
        total_size += model_size
        print("{}: {:.2f}MB".format(model.__class__.__name__, model_size))
    print("total size: {:.2f}MB".format(total_size))
    print("-"*80)

    criterion = [SIMLoss(), MSELoss()]

    validate(testloader, models, criterion, config)

@torch.no_grad()
def validate(testloader, models, criterion, config):
    print("-"*80)
    mode = "test"
    text_encoder, att_module, fusion_module = models
    text_encoder = text_encoder.cuda().eval()
    att_module = att_module.cuda().eval()
    fusion_module = fusion_module.cuda().eval()

    classes_dict = testloader.dataset.classes
    script_tokens_all = testloader.dataset.script_pool
    activity_emb_buffer = []
    for _, script_tokens in enumerate(script_tokens_all):
        script_tokens = script_tokens.cuda()
        script_emb = text_encoder(script_tokens)
        if config["data"]["split_script"]:
            script_emb = fusion_module.fuse_script(script_emb)
        activity_emb_buffer.append(script_emb)
    activity_emb = torch.cat(activity_emb_buffer)
    del activity_emb_buffer
    loss_buffer, top1_acc_buffer, top5_acc_buffer = [], [], []

    for _, (vfeatures, actions, label, _) in enumerate(tqdm(testloader)):
        vfeatures = vfeatures.cuda()
        label = label.cuda()
        prompted_texts = generate_prompted_text(actions, classes_dict)
        prompted_texts = [x.cuda() for x in prompted_texts]
        prompted_texts_emb = [text_encoder(text) for text in prompted_texts]
        vfeatures, sfeatures = att_module(vfeatures, prompted_texts_emb)
        if config["avg_fusion"]:
            _, video_emb = fusion_module(sfeatures, vfeatures)
        else:
            video_emb, _ = fusion_module(sfeatures, vfeatures)

        loss, pred_1, pred_5 = get_testing_meters(video_emb, activity_emb, label, fusion_module, criterion, config)
        loss_buffer.append(loss.item())
        top1_acc_buffer.append(pred_1)
        top5_acc_buffer.append(pred_5)

    total_loss = sum(loss_buffer) / len(loss_buffer)
    top1_acc = sum(top1_acc_buffer) / len(top1_acc_buffer) * 100
    top5_acc = sum(top5_acc_buffer) / len(top5_acc_buffer) * 100

    print("{}: \n loss: {:.3f} \n top1_acc: {:.2f} \n top5_acc: {:.2f}".format(mode, total_loss, top1_acc, top5_acc))

    print("{} end".format(mode))
    print("-"*80)


if __name__ == '__main__':
    torch.manual_seed(2022)
    np.random.seed(2022)
    random.seed(2022)
    inference()