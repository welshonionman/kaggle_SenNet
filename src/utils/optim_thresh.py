from tqdm import tqdm
import numpy as np
import torch


def batch_generator(data: torch.tensor, batch_size: torch.tensor):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def calc_score_batched(pred, true, metrics, thresh, batch_size):
    score = metrics(use_sigmoid=False)
    total = 0.0
    num_samples = len(pred)

    for pred_batch, true_batch in zip(batch_generator(pred, batch_size), batch_generator(true, batch_size)):
        pred_thresh = np.where(pred_batch > thresh, 1, 0)
        pred_thresh = torch.flatten(torch.from_numpy(pred_thresh))
        total += score(true_batch, pred_thresh).item()

    return total / (num_samples / batch_size)


def calc_optim_thresh(pred, true, metrics, cfg, batch_size=256 * 256 * 256):
    pred = torch.flatten(torch.cat(pred, axis=0)).to(torch.float16)
    true = torch.flatten(torch.cat(true, axis=0)).to(torch.bool)

    thresholds_to_test = [round(x * 0.01, 2) for x in cfg.thresholds_to_test]

    best_score = -1
    for thresh in tqdm(thresholds_to_test):
        score = calc_score_batched(pred, true, metrics, thresh, batch_size)
        if score > best_score:
            best_score = score
            best_thresh = thresh
    return best_score, best_thresh
