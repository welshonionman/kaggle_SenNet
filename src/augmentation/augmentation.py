import random
import numpy as np
import torch
import torch.nn as nn


def cutmix(batch_images, batch_labels, p=0.4):
    def rand_bbox(size, lambda_):
        W = size[-2]
        H = size[-1]
        cut_rat = np.sqrt(1.0 - lambda_)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    if random.random() > p:
        beta = 1
        lambda_ = np.random.beta(beta, beta)
        rand_index = torch.randperm(batch_images.size()[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(batch_images.size(), lambda_)

        batch_images[:, :, bbx1:bbx2, bby1:bby2] = batch_images[rand_index, :, bbx1:bbx2, bby1:bby2]
        batch_labels[:, :, bbx1:bbx2, bby1:bby2] = batch_labels[rand_index, :, bbx1:bbx2, bby1:bby2]
    return batch_images, batch_labels


def tta_rotate(x: torch.Tensor, model: nn.Module):
    shape = x.shape
    x = [torch.rot90(x, k=i, dims=(-2, -1)) for i in [1]]
    x = torch.cat(x, dim=0)
    x = model(x)
    x = torch.sigmoid(x)
    x = x.reshape(1, shape[0], *shape[2:])
    x = [torch.rot90(x[0], k=-i, dims=(-2, -1)) for i in [1]]
    x = torch.stack(x, dim=0)
    return x.mean(0)
