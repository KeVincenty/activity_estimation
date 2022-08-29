import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR


def build_optimizer(config, models):
    if config["train"]["optimizer"]["type"] == 'adam':
        optimizer = optim.Adam([{"params": model.parameters()} for model in models], lr=config["train"]["optimizer"]["lr"], betas=(0.9, 0.98), eps=1e-8,
                               weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        print('Using Adam as optimizer')
    elif config["train"]["optimizer"]["type"] == 'sgd':

        optimizer = optim.SGD([{"params": model.parameters()} for model in models],
                              lr=config["train"]["optimizer"]["lr"],
                              momentum=config["train"]["optimizer"]["momentum"],
                              weight_decay=config["train"]["optimizer"]["weight_decay"])
        print('Using SGD as optimizer')
    elif config["train"]["optimizer"]["type"] == 'adamw':

        optimizer = optim.AdamW([{"params": model.parameters()} for model in models],
                                betas=(0.9, 0.98), lr=config["train"]["optimizer"]["lr"], eps=1e-8,
                                weight_decay=config["train"]["optimizer"]["weight_decay"])  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        print('Using AdamW as optimizer')
    else:
        raise ValueError('Unknown optimizer: {}'.format(optimizer))

    for i, param_group in enumerate(optimizer.param_groups):
        print("learning rate is {} for {}".format(param_group['lr'], models[i].__class__.__name__))

    return optimizer

def build_lr_scheduler(config, optimizer):
    if config["train"]["lr_scheduler"]["type"] == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config["train"]["steps_per_epoch"]*config["train"]["epochs"],
            warmup_epochs=config["train"]["steps_per_epoch"]*config["train"]["lr_scheduler"]["lr_warmup_epoch"],
            warmup_lrs=1.e-8
        )
    elif config["train"]["lr_scheduler"]["type"] == 'multistep':
        if isinstance(config["train"]["lr_scheduler"]["lr_decay_epoch"], list):
            milestones = config["train"]["lr_scheduler"]["lr_decay_epoch"]
        elif isinstance(config["train"]["lr_scheduler"]["lr_decay_epoch"], int):
            milestones = [
                config["train"]["lr_scheduler"]["lr_decay_epoch"] * (i + 1)
                for i in range(config["train"]["epochs"] //
                               config["train"]["lr_scheduler"]["lr_decay_epoch"])]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config["train"]["lr_scheduler"]["lr_decay_epoch"])))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config["train"]["lr_scheduler"]["lr_warmup_epoch"]
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config["train"]["lr_scheduler"]["type"]))
    return lr_scheduler
