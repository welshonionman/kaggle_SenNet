import os

import torch
from torch.utils.data import DataLoader

import wandb
from src.model.loss import get_lossfn
from src.model.metrics import get_metrics
from src.dataset.common import get_dataset
from src.utils.common import SlackNotify
from src.model.model import build_model
from src.model.scheduler import get_scheduler


def init_dataset(fold, df, cfg):
    dataset = get_dataset(cfg)

    train_df = df[df[f"fold{fold}"] == "train"]
    valid_df = df[df[f"fold{fold}"] == "valid"]

    train_dataset = dataset(train_df, cfg, is_train=True)
    valid_dataset = dataset(valid_df, cfg, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.valid_batch_size, num_workers=cfg.num_workers, shuffle=False)

    return train_dataloader, valid_dataloader


def init_model(cfg):
    model = build_model(cfg.model_arch, cfg.backbone, cfg.in_chans, cfg.target_size)
    scaler = torch.cuda.amp.GradScaler()
    criterion = get_lossfn(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_scheduler(cfg, optimizer)
    metrics = get_metrics(cfg)

    return model, scaler, criterion, optimizer, scheduler, metrics


def init_exp(fold, cfg):
    TOKEN = os.environ["SLACK_TOKEN"]
    CHANNEL = os.environ["SLACK_CHANNEL"]
    USERID = os.environ["SLACK_USERID"]
    slacknotify = SlackNotify(TOKEN, CHANNEL, USERID, f"{cfg.exp_name}_fold{fold} {cfg.notes}")

    cfg_dict = {}
    for k, v in cfg.__dict__.items():
        if not k.startswith("__"):
            cfg_dict[k] = v

    if not cfg.debug:
        wandb.init(
            project=cfg.project,
            name=f"{cfg.exp_name}_fold{fold}",
            notes=cfg.notes,
            config=cfg_dict,
            dir="/kaggle",
            tags=[f"fold{fold}"]
            # reinit=True,
        )
    os.makedirs(f"./{cfg.exp_name}/", exist_ok=True)
    return slacknotify
