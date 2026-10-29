"""
scripts/verify_model.py
-----------------------
Verifies the Phase 2 model architecture end-to-end.

Run with:
    python scripts/verify_model.py --config config/config.yaml

Checks:
  1. Model builds from config
  2. Forward pass shape is correct
  3. Backbone → ViT → Head shapes at each stage
  4. predict_single() returns valid output
  5. Prints full parameter summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 2 model architecture")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")

    print("\n" + "=" * 60)
    print(" PHASE 2 — HYBRID MODEL ARCHITECTURE VERIFICATION")
    print("=" * 60)

    # ── Build model ───────────────────────────────────────────────────
    from src.models.model_factory import build_model, print_model_summary
    print("\n[1] Building HybridViTModel from config...")
    model = build_model(config, device=device)
    print("  ✓ Model built successfully")

    # ── Print summary ─────────────────────────────────────────────────
    print_model_summary(model)

    # ── Dummy forward pass ────────────────────────────────────────────
    print("[2] Running dummy forward pass (batch_size=2, 224×224×3)...")
    dummy_input = torch.randn(2, 3, 224, 224)

    # Backbone
    features = model.backbone(dummy_input)
    print(f"  backbone output   : {list(features.shape)}")
    assert features.shape == (2, 512, 7, 7), f"Unexpected backbone shape: {features.shape}"

    # ViT encoder
    global_feat = model.encoder(features)
    embed_dim = config["model"]["embed_dim"]
    print(f"  ViT encoder output: {list(global_feat.shape)}")
    assert global_feat.shape == (2, embed_dim), f"Unexpected encoder shape"

    # Classification head
    logits = model.head(global_feat)
    print(f"  head (logits)     : {list(logits.shape)}")
    assert logits.shape == (2, config["num_classes"]), f"Unexpected logit shape"

    # Full forward
    logits_full = model(dummy_input)
    print(f"  full forward      : {list(logits_full.shape)}")
    assert torch.allclose(logits, logits_full), "Full forward mismatch"
    print("  ✓ All shapes correct")

    # ── predict_single ────────────────────────────────────────────────
    print("\n[3] Testing predict_single()...")
    single = torch.randn(1, 3, 224, 224)
    cls_name, conf, scores = model.predict_single(single)
    print(f"  Predicted class : {cls_name}")
    print(f"  Confidence      : {conf:.4f}")
    print(f"  All scores      : {scores}")
    assert cls_name in config["classes"], "Predicted class not in class list"
    assert 0.0 <= conf <= 1.0
    print("  ✓ predict_single OK")

    print("\n" + "=" * 60)
    print(" ✅  Phase 2 model architecture OK — ready for Phase 3 training")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

