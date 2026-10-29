"""
tests/test_dataset.py
---------------------
Unit tests for PaddyDiseaseDataset.

Run with:
    pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

MOCK_CONFIG = {
    "classes": ["Leaf Blast", "Brown Spot", "Healthy"],
    "num_classes": 3,
    "data": {
        "image_size": 224,
        "train_split": 0.8,
        "val_split": 0.2,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "augmentation": {},
    },
    "training": {"batch_size": 4, "num_workers": 0, "seed": 42},
    "paths": {},
}


def _create_dummy_dataset(root: Path, classes: list, n_per_class: int = 5) -> None:
    """Create a minimal image dataset for testing."""
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(arr, "RGB")
            img.save(cls_dir / f"img_{i:03d}.jpg")


class TestPaddyDiseaseDataset:
    def test_len(self, tmp_path):
        from src.data.dataset import PaddyDiseaseDataset
        _create_dummy_dataset(tmp_path, ["Leaf Blast", "Brown Spot", "Healthy"], n_per_class=5)
        ds = PaddyDiseaseDataset(tmp_path, MOCK_CONFIG)
        assert len(ds) == 15

    def test_class_to_idx(self, tmp_path):
        from src.data.dataset import PaddyDiseaseDataset
        _create_dummy_dataset(tmp_path, ["Leaf Blast", "Brown Spot", "Healthy"])
        ds = PaddyDiseaseDataset(tmp_path, MOCK_CONFIG)
        assert ds.class_to_idx == {"Leaf Blast": 0, "Brown Spot": 1, "Healthy": 2}

    def test_idx_to_class(self, tmp_path):
        from src.data.dataset import PaddyDiseaseDataset
        _create_dummy_dataset(tmp_path, ["Leaf Blast", "Brown Spot", "Healthy"])
        ds = PaddyDiseaseDataset(tmp_path, MOCK_CONFIG)
        assert ds.idx_to_class[0] == "Leaf Blast"
        assert ds.idx_to_class[2] == "Healthy"

    def test_getitem_returns_pil_and_int(self, tmp_path):
        from src.data.dataset import PaddyDiseaseDataset
        _create_dummy_dataset(tmp_path, ["Leaf Blast", "Brown Spot", "Healthy"])
        ds = PaddyDiseaseDataset(tmp_path, MOCK_CONFIG)
        image, label = ds[0]
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)

    def test_getitem_with_transform(self, tmp_path):
        import torch
        import torchvision.transforms as T
        from src.data.dataset import PaddyDiseaseDataset
        _create_dummy_dataset(tmp_path, ["Leaf Blast", "Brown Spot", "Healthy"])
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        ds = PaddyDiseaseDataset(tmp_path, MOCK_CONFIG, transform=transform)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)

    def test_phase_a_filter(self, tmp_path):
        """Dataset should ignore folders not in the config class list."""
        from src.data.dataset import PaddyDiseaseDataset
        # Create extra class folder that should be ignored in Phase A
        _create_dummy_dataset(tmp_path, ["Leaf Blast", "Brown Spot", "Healthy", "Sheath Blight"])
        ds = PaddyDiseaseDataset(tmp_path, MOCK_CONFIG)
        # Only 3 × 5 = 15 samples; Sheath Blight is filtered
        assert len(ds) == 15

    def test_get_class_counts(self, tmp_path):
        from src.data.dataset import PaddyDiseaseDataset
        _create_dummy_dataset(tmp_path, ["Leaf Blast", "Brown Spot", "Healthy"], n_per_class=5)
        ds = PaddyDiseaseDataset(tmp_path, MOCK_CONFIG)
        counts = ds.get_class_counts()
        assert counts == {"Leaf Blast": 5, "Brown Spot": 5, "Healthy": 5}

    def test_get_class_weights_sum(self, tmp_path):
        from src.data.dataset import PaddyDiseaseDataset
        _create_dummy_dataset(tmp_path, ["Leaf Blast", "Brown Spot", "Healthy"], n_per_class=5)
        ds = PaddyDiseaseDataset(tmp_path, MOCK_CONFIG)
        weights = ds.get_class_weights()
        # For balanced dataset each weight = total / (n_classes * count_per_class) = 15 / (3*5) = 1.0
        assert all(abs(w - 1.0) < 1e-5 for w in weights), f"Expected uniform weights=1.0, got {weights}"

    def test_missing_class_dir_warning(self, tmp_path, caplog):
        import logging
        from src.data.dataset import PaddyDiseaseDataset
        # Only create 2 of 3 class dirs
        _create_dummy_dataset(tmp_path, ["Leaf Blast", "Brown Spot"])
        with caplog.at_level(logging.WARNING, logger="src.data.dataset"):
            ds = PaddyDiseaseDataset(tmp_path, MOCK_CONFIG)
        assert len(ds) == 10  # only 2 × 5

