"""
Data pipeline verification script.
Run this to confirm your dataset is set up correctly before training.

Usage:
    python scripts/verify_data_pipeline.py --config config/config.yaml --data-dir data/train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 1 data pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", default=None, help="Directory to inspect (default: config train_dir)")
    parser.add_argument("--num-batches", type=int, default=2, help="Batches to iterate for timing")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 60)
    print(" PHASE 1 — DATA PIPELINE VERIFICATION")
    print("=" * 60)
    print(f"  Config   : {args.config}")
    print(f"  Classes  : {config['classes']}")
    print(f"  Phase    : {config['project']['phase']}")

    # ── Dataset ────────────────────────────────────────────────────────
    from src.data.dataset import PaddyDiseaseDataset
    from src.data.preprocess import get_train_transforms, get_val_transforms

    data_dir = args.data_dir or config["paths"]["train_dir"]
    print(f"\n[1] Loading dataset from: {data_dir}")

    transform = get_val_transforms(config)
    try:
        dataset = PaddyDiseaseDataset(data_dir, config, transform=transform)
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        sys.exit(1)

    print(dataset)
    counts = dataset.get_class_counts()
    weights = dataset.get_class_weights()

    print("\n[2] Class distribution:")
    for cls, cnt in counts.items():
        print(f"  {cls:<20}: {cnt:>5} samples | weight = {weights[dataset.class_to_idx[cls]]:.4f}")

    # ── DataLoader ─────────────────────────────────────────────────────
    print("\n[3] DataLoader batch test:")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    import time
    times = []
    for i, (images, labels) in enumerate(loader):
        t0 = time.time()
        if i >= args.num_batches:
            break
        times.append(time.time() - t0)
        print(f"  Batch {i+1}: images={images.shape}, labels={labels.tolist()}")
        print(f"    dtype={images.dtype}, min={images.min():.3f}, max={images.max():.3f}")

    # ── Sample grid plot ────────────────────────────────────────────────
    print("\n[4] Saving sample grid to outputs/plots/sample_grid.png ...")
    from src.data.preprocess import denormalize

    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    images, labels = next(iter(loader))
    images_denorm = denormalize(images, config)
    n = min(4, len(images))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, lbl in zip(axes, images_denorm[:n], labels[:n]):
        np_img = img.permute(1, 2, 0).numpy()
        ax.imshow(np_img.clip(0, 1))
        ax.set_title(dataset.idx_to_class[lbl.item()])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/plots/sample_grid.png", dpi=150)
    plt.close()
    print("  ✓ Saved: outputs/plots/sample_grid.png")

    print("\n" + "=" * 60)
    print(" ✅  Phase 1 pipeline OK — ready for Phase 2 model build")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

