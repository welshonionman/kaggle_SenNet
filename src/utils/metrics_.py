import numpy as np

import torch
import torch.nn as nn

from .official_metric import create_table_neighbour_code_to_surface_area

device = torch.device("cuda")  # can be 'cpu'


def compute_area(y: list, unfold: nn.Unfold, area: torch.Tensor) -> torch.Tensor:
    """
    Args:
      y (list[Tensor]): A pair of consecutive slices of mask
      unfold: nn.Unfold(kernel_size=(2, 2), padding=1)
      area (Tensor): surface area for 256 patterns (256, )

    Returns:
      Surface area of surface in 2x2x2 cube
    """
    # Two layers of segmentation masks
    yy = torch.stack(y, dim=0).to(torch.float16).unsqueeze(0)
    # (batch_size=1, nch=2, H, W)
    # bit (0/1) but unfold requires float

    # unfold slides through the volume like a convolution
    # 2x2 kernel returns 8 values (2 channels * 2x2)
    cubes_float = unfold(yy).squeeze(0)  # (8, n_cubes)

    # Each of the 8 values are either 0 or 1
    # Convert those 8 bits to one uint8
    cubes_byte = torch.zeros(cubes_float.size(1), dtype=torch.int32, device=device)
    # indices are required to be int32 or long for area[cube_byte] below, not uint8
    # Can be int32 for torch 2.0.0, int32 raise IndexError in torch 1.13.1.

    for k in range(8):
        cubes_byte += cubes_float[k, :].to(torch.int32) << k

    # Use area lookup table: pattern index -> area [float]
    cubes_area = area[cubes_byte]

    return cubes_area


def compute_surface_dice_score_from_volume(pred: np.ndarray, label: np.ndarray) -> float:
    """
    Compute surface Dice score for one 3D volume

    pred (np.ndarray): 3D array, prediction masks
    label (np.ndarray): 3D array, ground truth masks
    """
    assert pred.shape == label.shape
    assert len(label.shape) == 3  # Ensure it's a 3D array

    n_slices, h, w = label.shape

    # Surface area lookup table: Tensor[float32] (256, )
    area = create_table_neighbour_code_to_surface_area((1, 1, 1))
    area = torch.from_numpy(area).to(device)  # torch.float32

    # Slide through the volume like a convolution
    unfold = torch.nn.Unfold(kernel_size=(2, 2), padding=1)

    # Padding before first slice
    y0 = y0_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

    num = 0  # numerator of surface Dice
    denom = 0  # denominator
    for i in range(n_slices + 1):
        # Load one slice
        if i < n_slices:
            y1 = torch.from_numpy(label[i]).to(device)
            y1_pred = torch.from_numpy(pred[i]).to(device)
        else:
            # Padding after the last slice
            y1 = y1_pred = torch.zeros((h, w), dtype=torch.uint8, device=device)

        # Compute the surface area between two slices (n_cubes,)
        area_pred = compute_area([y0_pred, y1_pred], unfold, area)
        area_true = compute_area([y0, y1], unfold, area)

        # True positive cube indices
        idx = torch.logical_and(area_pred > 0, area_true > 0)

        # Surface dice numerator and denominator
        num += area_pred[idx].sum() + area_true[idx].sum()
        denom += area_pred.sum() + area_true.sum()

        # Next slice
        y0 = y1
        y0_pred = y1_pred

    dice = num / denom.clamp(min=1e-8)
    return dice.item()
