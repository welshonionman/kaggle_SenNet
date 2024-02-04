import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast
import math


class CustomModel(nn.Module):
    def __init__(self, model_arch, backbone, in_chans, target_size, weight):
        super().__init__()
        self.model = smp.create_model(
            model_arch,
            encoder_name=backbone,
            encoder_weights=weight,
            in_channels=in_chans,
            classes=target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.model(image)
        return output


def build_model(model_arch, backbone, in_chans, target_size, weight="imagenet"):
    print("model_arch: ", model_arch)
    print("backbone: ", backbone)
    model = CustomModel(model_arch, backbone, in_chans, target_size, weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    return model


def save_model(model, cfg, filename, **kwargs):
    save_dict = {
        "model": model.state_dict(),
        "model_arch": cfg.model_arch,
        "backbone": cfg.backbone,
        "in_chans": cfg.in_chans,
        "target_size": cfg.target_size,
    }
    save_dict.update(kwargs)
    torch.save(save_dict, filename)


class CustomInferenceModel(nn.Module):
    def __init__(self, model_arch, backbone, in_chans, target_size, weight, cfg):
        super().__init__()

        self.model = smp.create_model(
            model_arch,
            encoder_name=backbone,
            encoder_weights=weight,
            in_channels=in_chans,
            classes=target_size,
            activation=None,
        )
        self.batch = cfg.valid_batch_size
        self.in_chans = in_chans
        self.target_chans = math.ceil(cfg.in_chans / 2) - 1

    def forward_(self, image):
        output = self.model(image)
        return output[:, self.target_chans]

    def forward(self, image):
        # image.shape=(batch,c,h,w)
        image = image.to(torch.float32)

        shape = image.shape
        image = [torch.rot90(image, k=i, dims=(-2, -1)) for i in range(4)]
        image = torch.cat(image, dim=0)
        with autocast():
            with torch.no_grad():
                image = [self.forward_(image[i * self.batch : (i + 1) * self.batch]) for i in range(image.shape[0] // self.batch + 1)]
                image = torch.cat(image, dim=0)
        image = image.sigmoid()
        image = image.reshape(4, shape[0], *shape[2:])
        image = [torch.rot90(image[i], k=-i, dims=(-2, -1)) for i in range(4)]
        image = torch.stack(image, dim=0).mean(0)

        return image


def load_inference_model(model_path, cfg):
    pth = torch.load(model_path)

    print("model_name", pth["model_arch"])
    print("backbone", pth["backbone"])
    model = CustomInferenceModel(pth["model_arch"], pth["backbone"], pth["in_chans"], pth["target_size"], None, cfg)
    model.load_state_dict(pth["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model
