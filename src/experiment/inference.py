import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.dataset.common import get_test_dataset
import torch.nn.functional as F


def add_pad(stack: torch.Tensor, pad: int):
    # stack=(C,H,W)
    # output=(C,H+2*pad,W+2*pad)
    mean_ = int(stack.to(torch.float32).mean())
    stack = torch.cat([stack, torch.ones([stack.shape[0], pad, stack.shape[2]], dtype=stack.dtype, device=stack.device) * mean_], dim=1)
    stack = torch.cat([stack, torch.ones([stack.shape[0], stack.shape[1], pad], dtype=stack.dtype, device=stack.device) * mean_], dim=2)
    stack = torch.cat([torch.ones([stack.shape[0], pad, stack.shape[2]], dtype=stack.dtype, device=stack.device) * mean_, stack], dim=1)
    stack = torch.cat([torch.ones([stack.shape[0], stack.shape[1], pad], dtype=stack.dtype, device=stack.device) * mean_, stack], dim=2)
    return stack


def shift_axis(tensor, axis):
    perm = [axis, (axis + 1) % 3, (axis + 2) % 3]  # 軸の順番をシフト
    tensor = tensor.permute(*perm)
    return tensor


def remove_pad(pred: torch.Tensor, pad: int):
    pred = pred[..., pad:-pad, pad:-pad]
    return pred


def cutout_chip(img, stack_shape, stride, img_size, edge):
    chip = []
    xy_indexs = []

    x1_list = np.arange(0, stack_shape[-2] + 1, stride)
    y1_list = np.arange(0, stack_shape[-1] + 1, stride)

    for y1 in y1_list:
        for x1 in x1_list:
            x2 = x1 + img_size
            y2 = y1 + img_size
            chip.append(img[..., x1:x2, y1:y2])
            xy_indexs.append([x1 + edge, x2 - edge, y1 + edge, y2 - edge])
    return chip, xy_indexs


def infer_each_z(model, img, stack_shape, cfg):
    img = img.to("cuda:0")
    img = add_pad(img[0], cfg.image_size // 2)[None]

    chip, xy_indexs = cutout_chip(img, stack_shape, cfg.stride, cfg.image_size, cfg.drop_egde_pixel)
    chip = torch.cat(chip)

    preds = model.forward(chip).to(device=0)
    preds = preds.unsqueeze(1)
    preds = remove_pad(preds, cfg.drop_egde_pixel)

    pred = torch.zeros_like(img[:, 0], dtype=torch.float16, device=img.device)
    count = torch.zeros_like(img[:, 0], dtype=torch.float16, device=img.device)
    for i, (x1, x2, y1, y2) in enumerate(xy_indexs):
        pred[..., x1:x2, y1:y2] += preds[i]
        count[..., x1:x2, y1:y2] += 1
    pred /= count
    pred = remove_pad(pred, cfg.image_size // 2)

    pred = (pred[0]).to(torch.float16).cpu()
    return pred


def inference_each_axis(model, stack_path, axis, save_path, cfg):
    stack = torch.tensor(np.load(stack_path))
    stack = shift_axis(stack, axis)

    preds = torch.zeros_like(stack, dtype=torch.float16)

    dataset = get_test_dataset(cfg)(stack, cfg.in_chans)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    for img, z_ in tqdm(dataloader):  # img=(1,C,H,W)
        pred = infer_each_z(model, img, stack.shape, cfg)
        preds[z_] = pred

    preds = shift_axis(preds, -axis)
    np.save(save_path, preds)


def inference(model, stack_path, save_path, cfg):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    for axis in range(3):
        kidney = stack_path.split("/")[-1].split(".")[0]

        save_path_each_axis = (f"./{save_dir}/{kidney}_{axis}.npy").replace("images", "preds")
        print(save_path_each_axis)

        if os.path.exists(save_path_each_axis):
            continue

        inference_each_axis(model, stack_path, axis, save_path_each_axis, cfg)

    preds = np.mean(np.stack([np.load((f"./{save_dir}/{kidney}_{axis}.npy").replace("images", "preds")) for axis in range(3)]), axis=0)
    np.save(save_path, preds)
    for axis in range(3):
        os.remove((f"./{save_dir}/{kidney}_{axis}.npy").replace("images", "preds"))
