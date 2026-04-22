"""
correlation_analysis.py
=======================
Generates adjacent-pixel correlation graphs comparing original images
against their encrypted counterparts stored in FinalResults/encrypted/.

Outputs (saved to FinalResults/correlations/):
  - FinalResults/correlations/horizontal/<base>_H.png
      Original vs Encrypted scatter — Horizontal direction
  - FinalResults/correlations/vertical/<base>_V.png
      Original vs Encrypted scatter — Vertical direction
  - FinalResults/correlations/diagonal/<base>_D.png
      Original vs Encrypted scatter — Diagonal direction
  - FinalResults/correlations/correlation_summary.png
      Grouped bar chart: mean |r| across all images per direction
  - FinalResults/correlations/correlation_values.csv

Usage:
    python correlation_analysis.py
"""

import os
import math
import random

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless – no display needed
import matplotlib.pyplot as plt
import pandas as pd

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATASET_DIR   = "./Dataset"
ENCRYPTED_DIR = "./FinalResults/encrypted"
OUTPUT_DIR    = "./FinalResults/correlations"

# Direction sub-folders
DIR_FOLDERS = {
    "H": os.path.join(OUTPUT_DIR, "horizontal"),
    "V": os.path.join(OUTPUT_DIR, "vertical"),
    "D": os.path.join(OUTPUT_DIR, "diagonal"),
}

# Number of random pixel-pairs used for the scatter plots
N_SAMPLES = 3000

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_gray(path: str) -> np.ndarray | None:
    """Load an image as uint8 grayscale (2-D) array; return None on failure."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img  # None if load failed, else H x W uint8


def sample_adjacent_pairs(channel: np.ndarray, n: int, direction: str):
    """
    Sample n random adjacent pixel-pairs from a 2-D channel array.

    direction : 'H'  – horizontal  (x, x+1)
                'V'  – vertical    (y, y+1)
                'D'  – diagonal    (x+1, y+1)

    Returns (x_vals, y_vals) as float32 arrays of length ≤ n.
    """
    H, W = channel.shape
    rng = np.random.default_rng(seed=42)

    if direction == "H":
        rows = rng.integers(0, H,     size=n)
        cols = rng.integers(0, W - 1, size=n)
        x_vals = channel[rows, cols].astype(np.float32)
        y_vals = channel[rows, cols + 1].astype(np.float32)

    elif direction == "V":
        rows = rng.integers(0, H - 1, size=n)
        cols = rng.integers(0, W,     size=n)
        x_vals = channel[rows,     cols].astype(np.float32)
        y_vals = channel[rows + 1, cols].astype(np.float32)

    elif direction == "D":
        rows = rng.integers(0, H - 1, size=n)
        cols = rng.integers(0, W - 1, size=n)
        x_vals = channel[rows,     cols].astype(np.float32)
        y_vals = channel[rows + 1, cols + 1].astype(np.float32)

    else:
        raise ValueError(f"Unknown direction: {direction}")

    return x_vals, y_vals


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D arrays."""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def channel_correlation(img: np.ndarray, direction: str, n: int = N_SAMPLES) -> float:
    """
    Pearson correlation of adjacent pixel-pairs on a grayscale (2-D) image.
    """
    x, y = sample_adjacent_pairs(img, n, direction)
    return pearson_correlation(x, y)


# ──────────────────────────────────────────────
# Per-direction scatter plot  (Original vs Encrypted)
# ──────────────────────────────────────────────

DIRECTION_LABELS = {"H": "Horizontal", "V": "Vertical", "D": "Diagonal"}

# Scatter colour for grayscale plots
GRAY_ORIG_COLOR = "#E0C97F"   # warm sand  – original
GRAY_ENC_COLOR  = "#7FBBE0"   # cool blue  – encrypted


def _draw_scatter_ax(ax, img: np.ndarray, direction: str, label: str):
    """
    Draw grayscale adjacent-pixel scatter on *ax*.
    img must be a 2-D (H x W) uint8 grayscale array.
    Returns [r] as a single-element list for consistency with caller.
    """
    scatter_color = GRAY_ORIG_COLOR if label == "Original" else GRAY_ENC_COLOR

    x, y = sample_adjacent_pairs(img, N_SAMPLES, direction)
    r = pearson_correlation(x, y)

    ax.scatter(x, y, s=3, alpha=0.18, color=scatter_color,
               label=f"Gray: r={r:+.4f}", rasterized=True)

    d_label = DIRECTION_LABELS[direction]
    ax.set_title(
        f"{label}  [{d_label}]\nr = {r:+.4f}",
        color="#F0F0F0", fontsize=10, pad=7
    )
    ax.set_xlabel("Pixel(i)",   color="#AAAAAA", fontsize=9)
    ax.set_ylabel("Pixel(i+1)", color="#AAAAAA", fontsize=9)
    ax.set_xlim(-5, 260)
    ax.set_ylim(-5, 260)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    leg = ax.legend(fontsize=8, loc="upper left",
                    framealpha=0.35, labelcolor="white")
    leg.get_frame().set_facecolor("#1A1D27")
    return [r]


def plot_direction_graph(orig: np.ndarray, enc: np.ndarray,
                         direction: str, base_name: str, out_path: str):
    """
    Produce a single figure with two side-by-side scatter plots:
      Left  – Original  [direction]
      Right – Encrypted [direction]

    Includes grayscale scatter, Pearson r, and a delta annotation.
    """
    accent = {"H": "#E07B54", "V": "#5ABFA0", "D": "#7B6FD4"}[direction]
    d_label = DIRECTION_LABELS[direction]

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.5),
        facecolor="#0F1117",
        gridspec_kw={"wspace": 0.30}
    )

    for ax in axes:
        ax.set_facecolor("#1A1D27")

    orig_rs = _draw_scatter_ax(axes[0], orig, direction, "Original")
    enc_rs  = _draw_scatter_ax(axes[1], enc,  direction, "Encrypted")

    orig_mean = float(np.mean(orig_rs))
    enc_mean  = float(np.mean(enc_rs))
    delta     = orig_mean - enc_mean          # positive = good reduction

    # Accent border on both axes
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(1.4)

    # Centred super-title
    fig.suptitle(
        f"{d_label} Adjacent-Pixel Correlation  --  {base_name}\n"
        f"Correlation drop: {orig_mean:+.4f}  ->  {enc_mean:+.4f}  "
        f"(delta = {delta:+.4f})",
        color="#FFFFFF", fontsize=11, fontweight="bold"
    )

    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [{direction}] saved -> {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# Summary bar chart
# ──────────────────────────────────────────────

def plot_summary_bar(records: list[dict], out_path: str):
    """
    Grouped bar chart: mean |correlation| per direction for
    Original vs Encrypted, averaged across all processed images.

    records: list of dicts with keys
        orig_H, orig_V, orig_D, enc_H, enc_V, enc_D
    """
    df = pd.DataFrame(records)

    directions = ["H", "V", "D"]
    orig_means = [df[f"orig_{d}"].abs().mean() for d in directions]
    enc_means  = [df[f"enc_{d}"].abs().mean()  for d in directions]

    x      = np.arange(len(directions))
    width  = 0.30
    labels = [DIRECTION_LABELS[d] for d in directions]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#0F1117")
    ax.set_facecolor("#1A1D27")

    bars1 = ax.bar(x - width / 2, orig_means, width,
                   label="Original",  color="#E07B54", zorder=3)
    bars2 = ax.bar(x + width / 2, enc_means,  width,
                   label="Encrypted", color="#4A90D9", zorder=3)

    # Value annotations
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.4f}", ha="center", va="bottom",
                color="#DDDDDD", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#CCCCCC", fontsize=11)
    ax.set_ylabel("Mean |Pearson r| (avg over all images & R/G/B)",
                  color="#AAAAAA", fontsize=9)
    ax.set_ylim(0, min(1.05, max(orig_means + enc_means) + 0.12))
    ax.set_title("Adjacent-Pixel Correlation: Original vs Encrypted\n"
                 "(Lower is better for encrypted — indicates randomness)",
                 color="#FFFFFF", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.yaxis.grid(True, linestyle="--", alpha=0.25, color="#555566")
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=10, framealpha=0.4, labelcolor="white")
    legend.get_frame().set_facecolor("#1A1D27")

    # Annotation: ideal encrypted correlation ≈ 0
    ax.axhline(0, color="#AAAAAA", linewidth=0.7, linestyle=":")
    ax.text(x[-1] + width, 0.01, "Ideal enc ≈ 0",
            color="#AAAAAA", fontsize=8, va="bottom")

    n_images = len(df)
    ax.text(0.01, 0.97, f"n = {n_images} image(s)",
            transform=ax.transAxes, color="#999999",
            fontsize=8, va="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [saved] {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────

def export_csv(records: list[dict], out_path: str):
    df = pd.DataFrame(records)
    cols_order = [
        "filename",
        "orig_H", "orig_V", "orig_D",
        "enc_H",  "enc_V",  "enc_D",
    ]
    df = df[[c for c in cols_order if c in df.columns]]
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  [saved] {os.path.basename(out_path)}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # Create root + per-direction sub-folders
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for folder in DIR_FOLDERS.values():
        os.makedirs(folder, exist_ok=True)

    # Discover matched pairs: Dataset/<base>.jpeg  <->  FinalResults/encrypted/<base>_encrypted.png
    orig_files = [f for f in os.listdir(DATASET_DIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not orig_files:
        print(f"No images found in {DATASET_DIR}")
        return

    records = []

    for orig_fname in sorted(orig_files):
        base = os.path.splitext(orig_fname)[0]           # e.g. image_s3r2_kiit_132
        enc_fname = f"{base}_encrypted.png"
        orig_path = os.path.join(DATASET_DIR,   orig_fname)
        enc_path  = os.path.join(ENCRYPTED_DIR, enc_fname)

        if not os.path.exists(enc_path):
            print(f"[skip] No encrypted counterpart for {orig_fname}")
            continue

        print(f"Processing: {orig_fname}")

        orig = load_gray(orig_path)
        enc  = load_gray(enc_path)

        if orig is None or enc is None:
            print(f"  [skip] Could not load image(s)")
            continue

        # Resize encrypted to match original dimensions if needed (grayscale = H x W)
        if orig.shape != enc.shape:
            enc = cv2.resize(enc, (orig.shape[1], orig.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        # ── Compute Pearson r per direction (grayscale) ──
        orig_H = channel_correlation(orig, "H")
        orig_V = channel_correlation(orig, "V")
        orig_D = channel_correlation(orig, "D")
        enc_H  = channel_correlation(enc,  "H")
        enc_V  = channel_correlation(enc,  "V")
        enc_D  = channel_correlation(enc,  "D")

        print(f"  Original  -> H:{orig_H:+.4f}  V:{orig_V:+.4f}  D:{orig_D:+.4f}")
        print(f"  Encrypted -> H:{enc_H:+.4f}  V:{enc_V:+.4f}  D:{enc_D:+.4f}")

        # ── Separate graph per direction ──
        for d, suffix in [("H", "H"), ("V", "V"), ("D", "D")]:
            out_path = os.path.join(DIR_FOLDERS[d], f"{base}_{suffix}.png")
            plot_direction_graph(orig, enc, d, base, out_path)

        records.append({
            "filename": base,
            "orig_H": orig_H, "orig_V": orig_V, "orig_D": orig_D,
            "enc_H":  enc_H,  "enc_V":  enc_V,  "enc_D":  enc_D,
        })

    if not records:
        print("No valid pairs found. Exiting.")
        return

    # ── Summary bar chart (across all images) ──
    summary_path = os.path.join(OUTPUT_DIR, "correlation_summary.png")
    plot_summary_bar(records, summary_path)

    # ── CSV ──
    csv_path = os.path.join(OUTPUT_DIR, "correlation_values.csv")
    export_csv(records, csv_path)

    print(f"\nDone. All results saved to: {OUTPUT_DIR}")
    print(f"  horizontal/  ->  {len(records)} H graphs")
    print(f"  vertical/    ->  {len(records)} V graphs")
    print(f"  diagonal/    ->  {len(records)} D graphs")
    print(f"  summary + CSV in root correlations/ folder")


if __name__ == "__main__":
    main()
