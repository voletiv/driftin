#!/usr/bin/env python3
"""Plot loss curve from a training log CSV.

Expects CSV written like train_drift_multires_ddp.py / train_drift_ddp.py:
  header: step, loss, time_s, images_per_sec
  rows:   step (int), loss (float), time_s (float), images_per_sec (float)

Usage:
  python -m drifting_vs_diffusion.utils.plot_loss path/to/loss_log.csv
  python -m drifting_vs_diffusion.utils.plot_loss loss_log.csv -o loss_curve.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(log_path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    """Load loss_log.csv. Returns (steps, losses, time_s, images_per_sec)."""
    steps, losses, time_s, imgs_per_sec = [], [], [], []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            time_s.append(float(row["time_s"]))
            imgs_per_sec.append(float(row["images_per_sec"]))
    return steps, losses, time_s, imgs_per_sec


def plot_loss_curve(
    log_path: Path,
    output_path: Path | None = None,
    *,
    smooth_window: int = 20,
    no_smooth: bool = False,
) -> None:
    """Plot loss curve and save to output_path. Callable from training scripts."""
    log_path = Path(log_path).resolve()
    if not log_path.is_file():
        return
    steps, losses, time_s, imgs_per_sec = load_log(log_path)
    if not steps:
        return
    out_path = Path(output_path).resolve() if output_path else log_path.parent / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, alpha=0.4, color="C0", label="loss (raw)")
    if not no_smooth and len(losses) >= smooth_window:
        window = min(smooth_window, len(losses) // 2)
        kernel = np.ones(window) / window
        smoothed = np.convolve(losses, kernel, mode="valid")
        steps_smooth = steps[window - 1 :]
        ax.plot(steps_smooth, smoothed, color="C0", linewidth=2, label=f"loss (mean {window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(log_path.name)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss curve from loss_log.csv (train_drift_*_ddp.py format)."
    )
    parser.add_argument(
        "log",
        type=Path,
        help="Path to loss_log.csv",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output image path (default: same dir as log, name loss_curve.png)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable smoothing (default: show both raw and smoothed).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Window size for rolling mean (default: 20).",
    )
    args = parser.parse_args()
    log_path = args.log.resolve()
    if not log_path.is_file():
        raise SystemExit(f"File not found: {log_path}")
    if not load_log(log_path)[0]:
        raise SystemExit(f"No data rows in {log_path}")
    out_path = args.output or log_path.parent / "loss_curve.png"
    plot_loss_curve(log_path, out_path, smooth_window=args.smooth_window, no_smooth=args.no_smooth)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
