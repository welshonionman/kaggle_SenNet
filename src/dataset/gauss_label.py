import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from scipy.stats import norm


def transform_proximity(proximity: np.ndarray, sigma: int) -> np.ndarray:
    pdf = np.where(proximity == 0, 0, norm.pdf(proximity, loc=1, scale=sigma))
    pdf_max = pdf.max() + 1e-8
    return pdf / pdf_max


def mix_transformed_proximity(proximity: np.ndarray, sigma1: int, sigma2: int, weight: float) -> np.ndarray:
    transformed1 = transform_proximity(proximity, sigma1)
    transformed2 = transform_proximity(proximity, sigma2)

    mixed = weight * transformed1 + (1 - weight) * transformed2
    return mixed


def make_soft_label(label: np.ndarray, max_distance: int = 20, distance_from_edge: int = 0) -> np.ndarray:
    # ラベルからの近さを0-1スケールで算出
    posmask = label.astype(np.float16)

    if posmask.any():
        edge_internal = distance(posmask) > distance_from_edge

        negmask = 1 - edge_internal

        label = (-distance(negmask) * negmask) + max_distance
        label = np.clip(label, 0, max_distance)
        label = label / max_distance  # normalize

    return label


# def make_soft_label(label: np.ndarray, max_distance: int = 20) -> np.ndarray:
#     # ラベルからの近さを0-1スケールで算出
#     proximity = np.zeros_like(label)
#     posmask = label.astype(np.float16)

#     if posmask.any():
#         negmask = 1 - posmask
#         proximity = (-distance(negmask) * negmask) + max_distance

#     proximity = (np.clip(proximity, 0, max_distance)) / max_distance  # normalize

#     return proximity
