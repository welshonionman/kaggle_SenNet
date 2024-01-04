import torch
import torch.cuda.amp as amp
from tqdm import tqdm

import wandb


def train(model, train_dataloader, optimizer, criterion, scheduler, scaler, epoch, cfg):
    model.train()
    total_loss = 0.0
    pbar_train = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-0b}",
    )

    for i, (images, masks) in pbar_train:
        images, masks = images.cuda(), masks.cuda()
        optimizer.zero_grad()

        with amp.autocast():
            preds = model(images)
            loss = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item()

        lr = f"LR : {scheduler.get_lr()[0]:.2E}"
        gpu_mem = f"Mem : {torch.cuda.memory_reserved() / 1E9:.3g}GB"
        pbar_train.set_description(
            ("%10s  " * 3 + "%10s") % (f"Epoch {epoch}/{cfg.epochs}", gpu_mem, lr, f"Loss: {total_loss / (i + 1):.4f}"),
        )

    scheduler.step()

    if wandb.run:
        wandb.log({"epoch": epoch, "train_loss": loss})


def valid(model, valid_dataloader, criterion, epoch, cfg, log=True):
    model.eval()
    total_loss = 0.0
    pred_list, true_list = [], []
    pbar_val = tqdm(
        enumerate(valid_dataloader),
        total=len(valid_dataloader),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )

    for i, (images, masks) in pbar_val:
        images, masks = images.cuda(), masks.cuda()
        with torch.no_grad():
            preds = model(images)
            loss = criterion(preds, masks)
            total_loss += loss.item()
            preds = torch.sigmoid(preds)
            pred_list.append(preds.cpu().detach().to(torch.float16))
            true_list.append(masks.cpu().detach().to(torch.bool))

        loss_ = total_loss / (i + 1)
        pbar_val.set_description(
            ("%10s") % (f"Val Loss: {loss_:.4f}"),
        )

    if wandb.run and log:
        wandb.log({"epoch": epoch, "valid_loss": loss_})

    return loss_, pred_list, true_list
