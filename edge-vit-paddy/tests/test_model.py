"""
tests/test_model.py
-------------------
Unit tests for the Hybrid ResNet-18 + ViT model components.

Run with:
    pytest tests/test_model.py -v

Tests cover:
  - ResNetExtractor output shape
  - ViTEncoder output shape
  - HybridViTModel forward pass (shapes, dtype)
  - predict() and predict_single() outputs
  - ClassificationHead isolation
  - Phase A / Phase B config switching
  - Checkpoint save + load round-trip
  - Frozen layer count
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

# ── Shared minimal configs ────────────────────────────────────────────────

PHASE_A_CONFIG = {
    "project": {"name": "test", "phase": "phase_a"},
    "classes": ["Leaf Blast", "Brown Spot", "Healthy"],
    "num_classes": 3,
    "model": {
        "backbone": "resnet18",
        "pretrained": False,       # no network in tests
        "embed_dim": 64,           # small for speed
        "num_heads": 4,
        "num_transformer_layers": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "cls_token": True,
        "head_hidden_dim": 32,
        "head_dropout": 0.0,
        "frozen_layers": 0,
    },
    "data": {"image_size": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "app": {"model_version": "phase_a_v1"},
}

PHASE_B_CONFIG = {
    **PHASE_A_CONFIG,
    "project": {"name": "test", "phase": "phase_b"},
    "classes": ["Leaf Blast", "Brown Spot", "Healthy", "Sheath Blight", "Tungro"],
    "num_classes": 5,
}


def _dummy_batch(B: int = 2) -> torch.Tensor:
    return torch.randn(B, 3, 224, 224)


# ── ResNetExtractor ───────────────────────────────────────────────────────

class TestResNetExtractor:
    def test_output_shape(self):
        from src.models.resnet_extractor import ResNetExtractor
        model = ResNetExtractor(pretrained=False)
        x = _dummy_batch()
        out = model(x)
        assert out.shape == (2, 512, 7, 7), f"Expected (2,512,7,7), got {out.shape}"

    def test_out_channels_attr(self):
        from src.models.resnet_extractor import ResNetExtractor
        m = ResNetExtractor(pretrained=False)
        assert m.out_channels == 512
        assert m.out_spatial == 7

    def test_freeze_reduces_trainable(self):
        from src.models.resnet_extractor import ResNetExtractor
        m_full   = ResNetExtractor(pretrained=False, frozen_layers=0)
        m_frozen = ResNetExtractor(pretrained=False, frozen_layers=6)
        assert m_frozen.count_trainable_params() < m_full.count_trainable_params()

    def test_unfreeze_restores(self):
        from src.models.resnet_extractor import ResNetExtractor
        m = ResNetExtractor(pretrained=False, frozen_layers=6)
        before = m.count_trainable_params()
        m.unfreeze_all()
        after = m.count_trainable_params()
        assert after > before

    def test_single_image(self):
        from src.models.resnet_extractor import ResNetExtractor
        m = ResNetExtractor(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = m(x)
        assert out.shape == (1, 512, 7, 7)


# ── PatchEmbedding ────────────────────────────────────────────────────────

class TestPatchEmbedding:
    def test_output_shape(self):
        from src.models.vit_encoder import PatchEmbedding
        pe = PatchEmbedding(in_channels=512, num_patches=49, embed_dim=64)
        x = torch.randn(2, 512, 7, 7)
        out = pe(x)
        assert out.shape == (2, 49, 64)


# ── TransformerEncoderBlock ───────────────────────────────────────────────

class TestTransformerBlock:
    def test_output_shape_preserved(self):
        from src.models.vit_encoder import TransformerEncoderBlock
        block = TransformerEncoderBlock(embed_dim=64, num_heads=4, mlp_ratio=2.0)
        x = torch.randn(2, 50, 64)  # 50 tokens (49 patches + 1 CLS)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_not_zero(self):
        from src.models.vit_encoder import TransformerEncoderBlock
        block = TransformerEncoderBlock(embed_dim=64, num_heads=4)
        x = torch.randn(1, 10, 64)
        out = block(x)
        assert not torch.allclose(out, x), "Transformer block should modify input"


# ── ViTEncoder ────────────────────────────────────────────────────────────

class TestViTEncoder:
    def _make_encoder(self, use_cls=True):
        from src.models.vit_encoder import ViTEncoder
        return ViTEncoder(
            in_channels=512, num_patches=49, embed_dim=64,
            num_heads=4, num_layers=2, dropout=0.0, use_cls_token=use_cls
        )

    def test_output_shape_cls_token(self):
        enc = self._make_encoder(use_cls=True)
        x = torch.randn(2, 512, 7, 7)
        out = enc(x)
        assert out.shape == (2, 64), f"Expected (2,64), got {out.shape}"

    def test_output_shape_mean_pool(self):
        enc = self._make_encoder(use_cls=False)
        x = torch.randn(2, 512, 7, 7)
        out = enc(x)
        assert out.shape == (2, 64)

    def test_output_dtype(self):
        enc = self._make_encoder()
        x = torch.randn(1, 512, 7, 7)
        out = enc(x)
        assert out.dtype == torch.float32

    def test_batch_size_1(self):
        enc = self._make_encoder()
        x = torch.randn(1, 512, 7, 7)
        out = enc(x)
        assert out.shape == (1, 64)

    def test_large_batch(self):
        enc = self._make_encoder()
        x = torch.randn(8, 512, 7, 7)
        out = enc(x)
        assert out.shape == (8, 64)


# ── HybridViTModel ────────────────────────────────────────────────────────

class TestHybridViTModel:
    def _make_model(self, cfg=None):
        from src.models.hybrid_model import HybridViTModel
        return HybridViTModel(cfg or PHASE_A_CONFIG)

    def test_forward_shape(self):
        model = self._make_model()
        x = _dummy_batch(2)
        out = model(x)
        assert out.shape == (2, 3), f"Expected (2,3), got {out.shape}"

    def test_forward_dtype(self):
        model = self._make_model()
        out = model(_dummy_batch(1))
        assert out.dtype == torch.float32

    def test_no_softmax_in_logits(self):
        """Training logits should NOT be softmaxed (CrossEntropy handles it)."""
        model = self._make_model()
        logits = model(_dummy_batch(4))
        # Softmaxed values would all be in (0,1) and sum to 1 per row.
        # Raw logits can be negative or >1.
        row_sums = logits.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones(4)), \
            "Logits should not sum to 1 — softmax should NOT be applied in forward()"

    def test_predict_probabilities(self):
        model = self._make_model()
        probs, ids = model.predict(_dummy_batch(2))
        assert probs.shape == (2, 3)
        assert ids.shape == (2,)
        # Probabilities must sum to ~1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_predict_single_output_types(self):
        model = self._make_model()
        x = torch.randn(1, 3, 224, 224)
        cls_name, conf, scores = model.predict_single(x)
        assert isinstance(cls_name, str)
        assert cls_name in PHASE_A_CONFIG["classes"]
        assert 0.0 <= conf <= 1.0
        assert set(scores.keys()) == set(PHASE_A_CONFIG["classes"])

    def test_predict_single_3d_input(self):
        """Should handle (3, 224, 224) without explicit batch dim."""
        model = self._make_model()
        x = torch.randn(3, 224, 224)
        cls_name, conf, scores = model.predict_single(x)
        assert cls_name in PHASE_A_CONFIG["classes"]

    def test_phase_b_output_shape(self):
        """Switching to Phase B config → 5 output classes."""
        from src.models.hybrid_model import HybridViTModel
        model = HybridViTModel(PHASE_B_CONFIG)
        out = model(_dummy_batch(2))
        assert out.shape == (2, 5), f"Phase B: expected (2,5), got {out.shape}"

    def test_class_maps(self):
        model = self._make_model()
        assert model.class_to_idx["Leaf Blast"] == 0
        assert model.idx_to_class[2] == "Healthy"

    def test_trainable_params_positive(self):
        model = self._make_model()
        assert model.count_trainable_params() > 0

    def test_freeze_backbone(self):
        model = self._make_model()
        before = model.count_trainable_params()
        model.freeze_backbone(n_layers=8)
        after = model.count_trainable_params()
        assert after < before


# ── Checkpoint round-trip ─────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        from src.models.model_factory import save_checkpoint, load_checkpoint
        from src.models.hybrid_model import HybridViTModel

        model = HybridViTModel(PHASE_A_CONFIG)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt_path = tmp_path / "test_checkpoint.pth"

        save_checkpoint(model, opt, epoch=5, val_accuracy=0.91,
                        path=ckpt_path, config=PHASE_A_CONFIG)

        loaded_model, meta = load_checkpoint(ckpt_path, PHASE_A_CONFIG)

        assert meta["epoch"] == 5
        assert abs(meta["val_accuracy"] - 0.91) < 1e-6
        assert meta["num_classes"] == 3
        assert meta["class_labels"] == ["Leaf Blast", "Brown Spot", "Healthy"]

        # Weights should match
        for (n1, p1), (n2, p2) in zip(
            model.state_dict().items(), loaded_model.state_dict().items()
        ):
            assert torch.allclose(p1, p2), f"Mismatch in layer: {n1}"

    def test_load_wrong_classes_raises(self, tmp_path):
        from src.models.model_factory import save_checkpoint, load_checkpoint
        from src.models.hybrid_model import HybridViTModel

        model = HybridViTModel(PHASE_A_CONFIG)
        ckpt_path = tmp_path / "wrong_classes.pth"
        save_checkpoint(model, None, epoch=1, val_accuracy=0.5,
                        path=ckpt_path, config=PHASE_A_CONFIG)

        with pytest.raises(ValueError, match="Class mismatch"):
            load_checkpoint(ckpt_path, PHASE_B_CONFIG)


# ── Model Factory ─────────────────────────────────────────────────────────

class TestModelFactory:
    def test_build_model_returns_hybrid(self):
        from src.models.model_factory import build_model
        from src.models.hybrid_model import HybridViTModel
        model = build_model(PHASE_A_CONFIG, device=torch.device("cpu"))
        assert isinstance(model, HybridViTModel)

    def test_build_model_on_cpu(self):
        from src.models.model_factory import build_model
        model = build_model(PHASE_A_CONFIG, device=torch.device("cpu"))
        for p in model.parameters():
            assert p.device.type == "cpu"

