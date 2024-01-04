import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


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


def load_model(model, ckp_path):
    ckp = torch.load(ckp_path)
    model.load_state_dict(ckp["model"])
    return model
