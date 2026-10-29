"""
src/data/preprocess.py
----------------------
Image preprocessing and augmentation pipeline for paddy disease detection.

Provides:
    get_train_transforms()  — augmented transforms for training set
    get_val_transforms()    — deterministic transforms for validation/inference
    apply_clahe()           — optional CLAHE contrast enhancement (OpenCV)
    preprocess_single()     — preprocess a single PIL/numpy image for inference

All parameters default to values from config.yaml (passed as `config` dict).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transform Builders
# ---------------------------------------------------------------------------

def get_train_transforms(config: dict) -> T.Compose:
    """
    Build augmentation pipeline for training.

    Augmentations applied (all configurable via config.yaml):
        1. Resize to IMAGE_SIZE × IMAGE_SIZE
        2. RandomHorizontalFlip
        3. RandomRotation
        4. ColorJitter (brightness, contrast, saturation, hue)
        5. GaussianBlur (optional)
        6. ToTensor   (HWC [0,255] → CHW [0.0,1.0])
        7. Normalize  (ImageNet mean/std)

    Parameters
    ----------
    config : dict
        Parsed config.yaml dict. Uses ``config['data']`` section.
    """
    data_cfg = config["data"]
    aug_cfg = data_cfg.get("augmentation", {})

    image_size: int = data_cfg["image_size"]
    mean: Tuple[float, ...] = tuple(data_cfg["mean"])
    std: Tuple[float, ...] = tuple(data_cfg["std"])

    transforms_list = [
        T.Resize((image_size, image_size)),
    ]

    # RandomHorizontalFlip
    if aug_cfg.get("random_horizontal_flip", True):
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    # RandomVerticalFlip
    if aug_cfg.get("random_vertical_flip", False):
        transforms_list.append(T.RandomVerticalFlip(p=0.3))

    # RandomRotation
    rotation_deg = aug_cfg.get("random_rotation", 15)
    if rotation_deg > 0:
        transforms_list.append(
            T.RandomRotation(degrees=(-rotation_deg, rotation_deg))
        )

    # ColorJitter
    jitter_cfg = aug_cfg.get("color_jitter", {})
    if jitter_cfg:
        transforms_list.append(
            T.ColorJitter(
                brightness=jitter_cfg.get("brightness", 0.2),
                contrast=jitter_cfg.get("contrast", 0.2),
                saturation=jitter_cfg.get("saturation", 0.2),
                hue=jitter_cfg.get("hue", 0.05),
            )
        )

    # GaussianBlur
    if aug_cfg.get("gaussian_blur", True):
        kernel_size = aug_cfg.get("blur_kernel_size", 3)
        # Kernel size must be odd
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        transforms_list.append(T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)))

    # Final mandatory transforms
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    logger.debug("Train transforms built: %d steps", len(transforms_list))
    return T.Compose(transforms_list)


def get_val_transforms(config: dict) -> T.Compose:
    """
    Build deterministic transforms for validation and inference.

    Steps:
        1. Resize to IMAGE_SIZE × IMAGE_SIZE
        2. ToTensor
        3. Normalize (ImageNet mean/std)

    No augmentation is applied.
    """
    data_cfg = config["data"]
    image_size: int = data_cfg["image_size"]
    mean: Tuple[float, ...] = tuple(data_cfg["mean"])
    std: Tuple[float, ...] = tuple(data_cfg["std"])

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# CLAHE Enhancement (optional, OpenCV-based)
# ---------------------------------------------------------------------------

def apply_clahe(
    image: Union[Image.Image, np.ndarray],
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Useful for enhancing low-contrast field images. Applied per-channel in
    LAB colour space to avoid hue shifts.

    Parameters
    ----------
    image : PIL.Image or np.ndarray (RGB, uint8)
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid_size : tuple
        Size of grid for histogram equalization.

    Returns
    -------
    PIL.Image (RGB)
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed. Returning image unmodified.")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        np_img = np.array(image)
    else:
        np_img = image.copy()

    # BGR conversion for OpenCV
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    bgr_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb_enhanced = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb_enhanced)


# ---------------------------------------------------------------------------
# Single Image Preprocessor (for inference pipeline)
# ---------------------------------------------------------------------------

def preprocess_single(
    image: Union[str, Path, Image.Image, np.ndarray],
    config: dict,
    apply_clahe_flag: bool = False,
    return_batch: bool = True,
) -> torch.Tensor:
    """
    Preprocess a single image for model inference.

    Applies validation transforms (resize + normalize). Optionally applies
    CLAHE before normalization. Returns a tensor ready for the model.

    Parameters
    ----------
    image : str, Path, PIL.Image, or np.ndarray
        Input image in any supported format.
    config : dict
        Parsed config.yaml dict.
    apply_clahe_flag : bool
        Whether to apply CLAHE enhancement before transforms.
    return_batch : bool
        If True, returns shape [1, C, H, W] (batch of 1).
        If False, returns shape [C, H, W].

    Returns
    -------
    torch.Tensor
    """
    # Load image
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Optional CLAHE
    if apply_clahe_flag:
        pil_image = apply_clahe(pil_image)

    # Apply val transforms
    transform = get_val_transforms(config)
    tensor = transform(pil_image)  # [C, H, W]

    if return_batch:
        tensor = tensor.unsqueeze(0)  # [1, C, H, W]

    return tensor


# ---------------------------------------------------------------------------
# Utility: denormalize for visualization
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Reverse ImageNet normalization for display purposes.

    Parameters
    ----------
    tensor : torch.Tensor, shape [C, H, W] or [B, C, H, W]
    config : dict

    Returns
    -------
    torch.Tensor in [0, 1]
    """
    mean = torch.tensor(config["data"]["mean"], dtype=tensor.dtype)
    std = torch.tensor(config["data"]["std"], dtype=tensor.dtype)

    if tensor.ndim == 4:  # [B, C, H, W]
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:  # [C, H, W]
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

    return torch.clamp(tensor * std + mean, 0.0, 1.0)

