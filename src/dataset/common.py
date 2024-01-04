import numpy as np
import h5py
import albumentations as A
from torch.utils.data import Dataset


class Base2dDataset(Dataset):
    def __init__(self, df, cfg, is_train=True):
        super(Dataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.hdf = h5py.File(f"{cfg.dataset_path}/dataset.hdf5", mode="r")

        if is_train:
            self.augmentation = A.Compose(cfg.train_aug)
        else:
            self.augmentation = A.Compose(cfg.valid_aug)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        image = np.array(self.hdf[self.df["image_path"][idx]]).astype(np.float32)
        label = np.array(self.hdf[self.df["label_path"][idx]]).astype(np.float32)

        augmented = self.augmentation(image=image, mask=label)

        image = augmented["image"]
        label = augmented["mask"]

        return image, label


def get_dataset(cfg):
    if cfg.dataset == "base2d":
        dataset = Base2dDataset
    return dataset
