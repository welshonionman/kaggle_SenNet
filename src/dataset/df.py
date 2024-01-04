import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


def get_info(fname, info_type):
    infos = fname.split("_")
    for info in infos:
        if info.startswith(info_type):
            return int(info.replace(info_type, ""))


def get_label_acc(fname):
    infos = fname.split("_")
    for info in infos:
        if info in ["dense", "sparse"]:
            return info


def get_kidney(fname):
    return "_".join(fname.split("_")[-2])


def get_fold(x, dict):
    if x in dict["train"]:
        return "train"
    elif x in dict["valid"]:
        return "valid"
    else:
        return ""


def df_dataset(cfg):
    fold_dict = {
        "fold0": {
            "train": ["kidney_1_dense", "kidney_2"],
            "valid": ["kidney_3_dense"],
            "": ["kidney_3_sparse"],
        },
        "fold1": {
            "train": ["kidney_2", "kidney_3_dense", "kidney_3_sparse"],
            "valid": ["kidney_1_dense"],
        },
    }

    df = pd.DataFrame()

    df["image_path"] = sorted(glob(f"{cfg.dataset_path}/image/*/*.npy"))
    df["label_path"] = sorted(glob(f"{cfg.dataset_path}/label/*/*.npy"))

    df["fname"] = df["image_path"].str.split("/").str[-1].str.split(".").str[0]
    df["kidney"] = df["image_path"].str.split("/").str[-2]

    df["x"] = df["fname"].apply(get_info, info_type="x")
    df["y"] = df["fname"].apply(get_info, info_type="y")
    df["z"] = df["fname"].apply(get_info, info_type="z")

    df["std"] = df["fname"].apply(get_info, info_type="std")
    df["sum"] = df["fname"].apply(get_info, info_type="sum")

    df["fold0"] = df["kidney"].apply(get_fold, dict=fold_dict["fold0"])
    df["fold1"] = df["kidney"].apply(get_fold, dict=fold_dict["fold1"])
    return df


def check_dataset(df, dataset, cfg):
    fold = 0
    train_dataset = dataset(df[df[f"fold{fold}"] == "train"], cfg, is_train=True)
    valid_dataset = dataset(df[df[f"fold{fold}"] == "valid"], cfg, is_train=False)

    print("train_len         :", len(train_dataset))
    print("train_image_shape :", train_dataset.__getitem__(0)[0].shape)
    print("train_label_shape :", train_dataset.__getitem__(0)[1].shape)
    print("train_image_dtype :", train_dataset.__getitem__(0)[0].dtype)
    print("train_label_dtype :", train_dataset.__getitem__(0)[1].dtype)
    print()
    print("valid_len         :", len(valid_dataset))
    print("valid_image_shape :", valid_dataset.__getitem__(0)[0].shape)
    print("valid_label_shape :", valid_dataset.__getitem__(0)[1].shape)
    print("valid_image_dtype :", valid_dataset.__getitem__(0)[0].dtype)
    print("valid_label_dtype :", valid_dataset.__getitem__(0)[1].dtype)

    fig, ax = plt.subplots(10, 6, figsize=(20, 30))
    fig.tight_layout()
    ax = ax.flatten()
    for i, idx in enumerate(np.random.randint(0, len(train_dataset), 30)):
        item = train_dataset.__getitem__(idx)

        ax[2 * i].imshow(item[0][0, :, :], vmin=0, vmax=1)
        ax[2 * i].axis("off")
        ax[2 * i + 1].imshow(item[1][0, :, :], vmin=0, vmax=1)
        ax[2 * i + 1].axis("off")
