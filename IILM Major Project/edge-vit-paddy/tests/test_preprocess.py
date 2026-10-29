"""
tests/test_preprocess.py
------------------------
Unit tests for the preprocessing pipeline.

Run with:
    pytest tests/test_preprocess.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

# Minimal config for testing (mirrors config.yaml structure)
MOCK_CONFIG = {
    "classes": ["Leaf Blast", "Brown Spot", "Healthy"],
    "num_classes": 3,
    "data": {
        "image_size": 224,
        "train_split": 0.8,
        "val_split": 0.2,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "augmentation": {
            "random_horizontal_flip": True,
            "random_vertical_flip": False,
            "random_rotation": 15,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.05,
            },
            "gaussian_blur": True,
            "blur_kernel_size": 3,
            "clahe": False,
        },
    },
    "training": {
        "batch_size": 4,
        "num_workers": 0,
        "seed": 42,
    },
    "paths": {
        "train_dir": "data/train",
        "val_dir": "data/val",
    },
}


def _make_dummy_image(size: int = 256) -> Image.Image:
    """Create a random RGB PIL image for testing."""
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ── Transform tests ──────────────────────────────────────────────────────────

class TestTrainTransforms:
    def test_output_tensor_shape(self):
        from src.data.preprocess import get_train_transforms
        transform = get_train_transforms(MOCK_CONFIG)
        img = _make_dummy_image(300)
        out = transform(img)
        assert isinstance(out, torch.Tensor), "Output should be a Tensor"
        assert out.shape == (3, 224, 224), f"Expected (3,224,224), got {out.shape}"

    def test_output_dtype(self):
        from src.data.preprocess import get_train_transforms
        transform = get_train_transforms(MOCK_CONFIG)
        img = _make_dummy_image()
        out = transform(img)
        assert out.dtype == torch.float32

    def test_normalize_range(self):
        """After normalization the values should span beyond [0,1]."""
        from src.data.preprocess import get_train_transforms
        transform = get_train_transforms(MOCK_CONFIG)
        img = _make_dummy_image()
        out = transform(img)
        # Normalized values can be negative or >1 — just check it's finite
        assert torch.isfinite(out).all()


class TestValTransforms:
    def test_output_tensor_shape(self):
        from src.data.preprocess import get_val_transforms
        transform = get_val_transforms(MOCK_CONFIG)
        img = _make_dummy_image(400)
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_deterministic(self):
        """Applying val transforms twice to the same image yields the same tensor."""
        from src.data.preprocess import get_val_transforms
        transform = get_val_transforms(MOCK_CONFIG)
        img = _make_dummy_image()
        out1 = transform(img)
        out2 = transform(img)
        assert torch.equal(out1, out2), "Val transforms should be deterministic"


class TestPreprocessSingle:
    def test_pil_input_batch(self):
        from src.data.preprocess import preprocess_single
        img = _make_dummy_image()
        tensor = preprocess_single(img, MOCK_CONFIG, return_batch=True)
        assert tensor.shape == (1, 3, 224, 224)

    def test_pil_input_no_batch(self):
        from src.data.preprocess import preprocess_single
        img = _make_dummy_image()
        tensor = preprocess_single(img, MOCK_CONFIG, return_batch=False)
        assert tensor.shape == (3, 224, 224)

    def test_numpy_input(self):
        from src.data.preprocess import preprocess_single
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        tensor = preprocess_single(arr, MOCK_CONFIG, return_batch=True)
        assert tensor.shape == (1, 3, 224, 224)

    def test_invalid_input_raises(self):
        from src.data.preprocess import preprocess_single
        with pytest.raises(TypeError):
            preprocess_single(12345, MOCK_CONFIG)


class TestDenormalize:
    def test_denormalize_shape_preserved(self):
        from src.data.preprocess import denormalize, get_val_transforms
        transform = get_val_transforms(MOCK_CONFIG)
        img = _make_dummy_image()
        tensor = transform(img)
        dn = denormalize(tensor, MOCK_CONFIG)
        assert dn.shape == tensor.shape

    def test_denormalize_range(self):
        from src.data.preprocess import denormalize, get_val_transforms
        transform = get_val_transforms(MOCK_CONFIG)
        img = _make_dummy_image()
        tensor = transform(img)
        dn = denormalize(tensor, MOCK_CONFIG)
        assert dn.min() >= 0.0 and dn.max() <= 1.0, "Denormalized values should be in [0,1]"

