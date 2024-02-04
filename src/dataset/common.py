import math
import numpy as np
import cv2
import h5py
import albumentations as A
from torch.utils.data import Dataset
import torch
from src.dataset.gauss_label import make_soft_label, mix_transformed_proximity


class Base2dDataset(Dataset):
    def __init__(self, df, cfg, is_train=True, use_hdf=False):
        super(Dataset, self).__init__()
        self.df = df.reset_index(drop=True)
        # self.hdf = h5py.File(f"{cfg.dataset_path}/dataset.hdf5", mode="r")
        self.use_hdf = use_hdf

        if is_train:
            self.augmentation = A.Compose(cfg.train_aug)
        else:
            self.augmentation = A.Compose(cfg.valid_aug)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        # if self.use_hdf:
        #     image = np.array(self.hdf[self.df["image_path"][idx]]).astype(np.float32)
        #     label = np.array(self.hdf[self.df["label_path"][idx]]).astype(np.float32)
        # else:
        #     image = np.load(self.df["image_path"][idx]).astype(np.float32)
        #     label = np.load(self.df["label_path"][idx]).astype(np.float32)
        image = np.load(self.df["image_path"][idx]).astype(np.float32)
        label = np.load(self.df["label_path"][idx]).astype(np.float32)
        augmented = self.augmentation(image=image, mask=label)
        image = augmented["image"]
        label = augmented["mask"]
        return image, label


class LargeImageDataset(Dataset):
    def __init__(self, df, cfg, is_train=True, use_hdf=True):
        super(Dataset, self).__init__()
        self.df = df.reset_index(drop=True)
        # self.hdf = h5py.File(f"{cfg.dataset_path}/dataset.hdf5", mode="r")
        self.use_hdf = use_hdf
        self.cfg = cfg

        if is_train:
            self.augmentation = A.Compose(cfg.train_aug)
        else:
            self.augmentation = A.Compose(cfg.valid_aug)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        # if self.use_hdf:
        #     image = np.array(self.hdf[self.df["image_path"][idx]]).astype(np.float32)
        #     label = np.array(self.hdf[self.df["label_path"][idx]]).astype(np.float32)
        # else:
        #     image = np.load(self.df["image_path"][idx]).astype(np.float32)
        #     label = np.load(self.df["label_path"][idx]).astype(np.float32)
        image = np.load(self.df["image_path"][idx]).astype(np.float32)
        label = np.load(self.df["label_path"][idx]).astype(np.float32)
        image = cv2.resize(image, dsize=(self.cfg.image_size, self.cfg.image_size), interpolation=self.cfg.image_interp)
        image = np.expand_dims(image, 2)
        label = cv2.resize(label, dsize=(self.cfg.image_size, self.cfg.image_size), interpolation=self.cfg.label_interp)
        label = np.expand_dims(label, 2)
        augmented = self.augmentation(image=image, mask=label)
        image = augmented["image"]
        label = augmented["mask"]
        return image, label


class GaussLabelDataset(Dataset):
    def __init__(self, df, cfg, is_train=True, use_hdf=True):
        super(Dataset, self).__init__()
        self.df = df.reset_index(drop=True)
        # self.hdf = h5py.File(f"{cfg.dataset_path}/dataset.hdf5", mode="r")
        self.use_hdf = use_hdf
        self.cfg = cfg
        if is_train:
            self.augmentation = A.Compose(cfg.train_aug)
        else:
            self.augmentation = A.Compose(cfg.valid_aug)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        # if self.use_hdf:
        #     image = np.array(self.hdf[self.df["image_path"][idx]]).astype(np.float32)
        #     label = np.array(self.hdf[self.df["label_path"][idx]]).astype(np.float32)
        #     label
        # else:
        #     image = np.load(self.df["image_path"][idx]).astype(np.float32)
        #     label = np.load(self.df["label_path"][idx]).astype(np.float32)
        image = np.load(self.df["image_path"][idx]).astype(np.float32)
        label = np.load(self.df["label_path"][idx]).astype(np.float32)
        label = make_soft_label(label, self.cfg.max_distance, self.cfg.distance_from_edge)
        label = mix_transformed_proximity(label, self.cfg.sigma1, self.cfg.sigma2, self.cfg.weight)
        augmented = self.augmentation(image=image, mask=label)

        image = augmented["image"]
        label = augmented["mask"]
        return image, label


class BaseInferenceDataset(Dataset):
    def __init__(self, stack, in_chans):
        self.in_chans = in_chans
        self.target_chans = math.ceil(in_chans / 2)

        pad_top = torch.zeros(self.target_chans - 1, *stack.shape[1:], dtype=stack.dtype)
        pad_bottom = torch.zeros(self.in_chans - self.target_chans, *stack.shape[1:], dtype=stack.dtype)

        self.stack = torch.cat((pad_top, stack, pad_bottom), dim=0)

    def __len__(self):
        return self.stack.shape[0] - self.in_chans

    def __getitem__(self, z_):
        stack = self.stack[z_ : z_ + self.in_chans]
        return stack, z_


class LargeImageInferenceDataset(Dataset):
    def __init__(self, stack, in_chans):
        self.in_chans = in_chans
        self.target_chans = math.ceil(in_chans / 2)

        pad_top = torch.zeros(self.target_chans - 1, *stack.shape[1:], dtype=stack.dtype)
        pad_bottom = torch.zeros(self.in_chans - self.target_chans, *stack.shape[1:], dtype=stack.dtype)

        self.stack = torch.cat((pad_top, stack, pad_bottom), dim=0)

    def __len__(self):
        return self.stack.shape[0] - self.in_chans

    def __getitem__(self, z_):
        stack = self.stack[z_ : z_ + self.in_chans]
        return stack, z_


def get_train_dataset(cfg):
    if cfg.train_dataset == "Base2dDataset":
        dataset = Base2dDataset
    elif cfg.train_dataset == "GaussLabelDataset":
        dataset = GaussLabelDataset
    elif cfg.train_dataset == "LargeImageDataset":
        dataset = LargeImageDataset
    else:
        raise ValueError(f"Invalid dataset name: {cfg.train_dataset}")
    return dataset


def get_test_dataset(cfg):
    if cfg.test_dataset == "BaseInferenceDataset":
        dataset = BaseInferenceDataset
    elif cfg.test_dataset == "LargeImageInferenceDataset":
        dataset = LargeImageInferenceDataset
    else:
        raise ValueError(f"Invalid dataset name: {cfg.dataset}")
    return dataset
