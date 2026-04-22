# Capstone 2026 — GAN-Based Chaotic Image Encryption

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

A research implementation of a **Wasserstein GAN with Spectral Normalisation (WGAN-SN)** that generates high-entropy chaotic sequences for image encryption. The trained generator produces statistically independent multi-node chaotic sequences that serve as XOR key streams for encrypting surveillance images.

---

## Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Training Phase                                  │
│                                                                          │
│   Dataset (images)  →  ImageChaoticDataset  →  WGAN-SN Training loop   │
│                           (logistic / tent / cosine map seeds)           │
│                                    │                                     │
│                              LSTMGenerator                               │
│                         (4-branch SIREN + cumsum)                        │
│                                    │                                     │
│                          generator.pth  (saved weights)                  │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                         Inference / Encryption Phase                     │
│                                                                          │
│   inference.py  →  sequence_file.csv  →  encryption.py                  │
│                      (4-node chaotic sequences)     │                    │
│                                                     ▼                    │
│                                    scramble → diffuse(0) → diffuse(1)   │
│                                                     │                    │
│                                              FinalResults/encrypted/     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| GAN variant | WGAN-SN (Spectral Norm) | Avoids double backprop — enables CuDNN + LSTM together |
| Generator seed | SIREN `sin(30·x)` | High-frequency initial state → chaotic sensitivity |
| Generator output | `cumsum(ELU deltas)` | Mirrors real chaotic trajectory accumulation |
| Key derivation | SHAKE-256 (XOF) | Cryptographically uniform byte stream; removes bias |
| Diffusion | Double-pass XOR (salted) | Independent per-channel, per-pass key streams |
| Lipschitz constraint | Spectral norm on Critic FC | No GP → CuDNN fully enabled |

---

## Repository Structure

```
Capstone_2026/
├── src/
│   ├── models.py          # LSTMGenerator (4-branch) + LSTMCritic (dual-stream)
│   ├── dataset.py         # ImageChaoticDataset — chaotic map seeding from image corners
│   ├── chaos.py           # Logistic / tent / cosine map generators
│   └── utils.py           # set_seed()
│
├── main.py                # WGAN-SN training entry point
├── inference.py           # Generate chaotic sequences from a trained generator
├── encryption.py          # Bulk image encryption (scramble + double XOR diffusion)
├── decryption.py          # Image decryption (exact inverse of encryption)
├── outputParameters.py    # Encryption quality metrics (NPCR, UACI, PSNR, Entropy)
├── correlation_analysis.py# Adjacent-pixel correlation graphs (H / V / D)
├── check_uniformity.py    # Histogram uniformity checks
├── noise_androtate.py     # Noise + rotation augmentation utilities
├── Decryption_withnoise.py# Decryption robustness under additive noise
│
├── docs/
│   ├── WGAN_GP_to_WGAN_SN_Changes.md  # Full migration notes: WGAN-GP → WGAN-SN
│   └── Histogram_Flatness_Analysis.md  # Histogram uniformity analysis
│
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

> **Not included in this repository:**
> - `Dataset/` — proprietary KIIT-MiTA surveillance image subset (bring your own images)
> - `FinalResults/` — all output is regeneratable from the pipeline
> - `generator.pth` — model weights; train locally using `main.py`
> - `sequence_file.csv` — generated sequences; produce with `inference.py`

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ (with CUDA support recommended)
- See [`requirements.txt`](requirements.txt) for the full dependency list

### Supported Platforms

| Platform | Status |
|----------|--------|
| Linux (CUDA GPU) | ✅ Recommended |
| Windows (CUDA GPU) | ✅ Supported |
| macOS / CPU-only | ✅ Works (training is slow) |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/aadi101204/Capstone_2026.git
cd Capstone_2026

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\Activate.ps1      # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Prepare Dataset

Place your images (JPEG / PNG) under `Dataset/`. The loader supports both flat folders and `ImageFolder`-style subdirectories.

```
Dataset/
├── image_001.jpeg
├── image_002.jpeg
└── ...
```

### Step 2 — Train the WGAN-SN Generator

```bash
python main.py \
  --data_path ./Dataset \
  --image_size 8 \
  --noise_dim 64 \
  --hidden_dim 128 \
  --batch_size 16 \
  --epochs 20 \
  --n_critic 5 \
  --map_type logistic \
  --output sequence_file.csv
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `./Dataset` | Path to training images |
| `--image_size` | `8` | Resize dimension (H = W); sequence length = H×W×3 |
| `--noise_dim` | `64` | Generator noise input dimension |
| `--hidden_dim` | `128` | LSTM hidden state size |
| `--epochs` | `20` | Training epochs |
| `--n_critic` | `5` | Critic updates per generator update |
| `--map_type` | `logistic` | Chaotic map for dataset seeding: `logistic`, `tent`, `cosine` |

Saves `generator.pth` and `sequence_file.csv` on completion.

### Step 3 — Generate Sequences (Optional — inference only)

If you already have trained weights and want to generate new sequences without retraining:

```bash
python inference.py \
  --model_path generator.pth \
  --num_sequences 118 \
  --noise_dim 64 \
  --hidden_dim 128 \
  --seq_length 192 \
  --output sequence_file.csv
```

### Step 4 — Encrypt Images

```bash
python encryption.py
```

Edit the `__main__` block in `encryption.py` to set `dataset_dir`, `seq_file`, and `output_dir`. Results are written to `FinalResults/`:

```
FinalResults/
├── scrambled/        # Pixel-permuted intermediate
├── encrypted/        # Final encrypted images
├── histograms/       # Per-image RGB histograms
└── encryption_metrics.csv
```

### Step 5 — Decrypt Images

```bash
python decryption.py
```

Edit the `__main__` block with the path to an encrypted image and the same `sequence_file.csv` used for encryption. The decrypted image is saved as `decrypted_output.png`.

### Step 6 — Analyze Results

```bash
# Adjacent-pixel correlation analysis (H / V / D scatter plots + summary bar chart)
python correlation_analysis.py

# Encryption quality metrics (NPCR, UACI, PSNR, Entropy) — called internally by encryption.py
# or run standalone:
python outputParameters.py
```

---

## Encryption Quality Metrics

| Metric | Ideal | Description |
|--------|-------|-------------|
| NPCR | ≥ 99.60 % | Number of Pixels Change Rate |
| UACI (visual) | 25–30 % | Average intensity change (bounded by image content) |
| UACI (differential) | ~33.3 % | Avalanche effect — single-pixel flip → cipher divergence |
| Entropy | ~8.0 bits | Per-channel Shannon entropy of encrypted image |
| PSNR | Low (< 10 dB) | Low PSNR = high dissimilarity = good encryption |

---

## Architecture

### LSTMGenerator

Four independent branches, each processing a `noise_dim/4`-dimensional noise slice:

```
z_slice  →  sin(30 · Linear(z_slice))  →  h0, c0  (SIREN seeding)
         →  3-layer LSTM (batch_first)
         →  Linear → ELU → Linear  (delta head, ×0.1 scaling)
         →  cumsum(deltas)          (integration → chaotic trajectory)
         →  sigmoid                 (output to [0, 1])
```

Four branches are stacked → `(batch, 4, seq_length)` tensor.

### LSTMCritic

Dual-stream architecture with Spectral Normalisation:

```
seq  →  Raw LSTM (2-layer)    →  mean + max pooling  ──┐
     →  Diff LSTM (1-layer)   →  mean + max pooling  ──┼→ cat → SN-FC → SN-FC → score
```

See [`docs/WGAN_GP_to_WGAN_SN_Changes.md`](docs/WGAN_GP_to_WGAN_SN_Changes.md) for the full migration from WGAN-GP.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT License](LICENSE) © 2026 Aditya Arora
