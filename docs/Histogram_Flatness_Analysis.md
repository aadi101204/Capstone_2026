# Histogram Analysis: Encrypted Image Flatness & Root Cause Diagnosis

**Date:** 2026-02-21  
**Reference:** `Encrypted_Results/plots/` — 5 images from KIIT-MiTA test set  
**Metrics file:** `Encrypted_Results/encryption_metrics.csv`

---

## 1. What a "Good" Encrypted Histogram Should Look Like

In ideal image encryption, after applying a sufficiently random key stream, every pixel value (0–255) should appear with **equal probability**. The histogram of the encrypted image should therefore be completely **flat / uniform** — a rectangle spanning 0–255 with roughly constant height across all bins. This is the visual proof that:
- No statistical information about the original image survives
- The cipher is immune to frequency analysis attacks
- The key stream is effectively uniformly distributed

---

## 2. What We Actually Observe

Viewing all 5 analysis plots, the encrypted histograms share the same structural anomaly across every image:

| Image | Encrypted Histogram Shape | Key Anomaly |
|---|---|---|
| 1006 (artillery) | Dominant spike at 0–100, steep drop mid-range | Heavy left skew, sparse at 128–200 |
| 1018 (tanks) | Flat-ish 0–100, deep valley at 128, hump at 200–255 | **Clear bimodal gap around 128** |
| 1020 (convoy, grayscale) | High left region 0–128, steep drop then rise~200 | Strong U-shaped inflection |
| 1025 (armoured vehicle) | Dense 0–120 then empty 120–200, sporadic rise 200–255 | Severe bimodal gap at 128 |
| 1043 (aerial field) | Low 0–100, then climbing ramp 100–255 | **Right-heavy skew** — inverse of ideal |

**The consistent pattern: the encrypted histograms are NOT flat.** They exhibit either a left-dominant spike, a bimodal gap around the 128 midpoint, or a rising ramp — all of which indicate the key stream is not uniformly distributed in [0, 255].

---

## 3. Root Cause Analysis — The Code

The encryption pipeline in `encryption.py` has **three interacting problems** that produce the non-flat histograms:

### 3.1 ❌ Non-Uniform Key Quantisation in `diffuse_image()`

```python
# encryption.py — Line 53
seq_int = ((seq_repeated * 1e8) % 256).astype(np.uint8)
```

The WGAN generator outputs values in **[0, 1]** (after `torch.sigmoid()`). Multiplying by `1e8` and taking `% 256` attempts to extract randomness from the fractional digits of these floating-point values. However, this is flawed for two reasons:

**Problem A — Non-uniform sigmoid output:**  
The generator's `torch.sigmoid(cumsum(deltas))` output is statistically centred near **0.5**, meaning most values cluster in **[0.3, 0.7]**. When multiplied by `1e8`, these become integers in the range `[30,000,000 – 70,000,000]`. After `% 256`, the residue distribution of a non-uniform source modulo a power-of-two is **not uniform** — it inherits the unevenness of the original distribution. Values near 0.5 map to a specific subset of residues far more frequently than values near 0 or 1.

**Problem B — The key has correlation structure:**  
The WGAN-generated sequences use `torch.cumsum()` integration in the generator branches. Cumulative sums create **smoothly varying trajectories**, not independent random samples. The `% 256` operation on a smooth ramp produces a **sawtooth** pattern of residues, not white noise — certain residue ranges (particularly around 0 and 128) get over-represented due to where the sawtooth transitions happen.

The bimodal gap at ~128 visible in images 1018 and 1025 is a **direct fingerprint** of this sawtooth residue pattern: the sequence oscillates predictably through residues 0→255, creating near-equal probability up to the halfway point then a gap.

### 3.2 ❌ Same Key Applied to All 3 RGB Channels

```python
# encryption.py — Line 56
seq_int_expanded = np.repeat(seq_int[:, np.newaxis], C, axis=1)
diffused = np.bitwise_xor(flat, seq_int_expanded)
```

The scalar key stream `seq_int` is **replicated identically** across all 3 colour channels (R, G, B). This means:
- The XOR applied to the Red, Green, and Blue channels of every pixel is **the same byte**
- If the original image has correlated channels (which most natural images do), this correlation **survives encryption** — it is never broken
- In the histograms, all 3 coloured channels (R=red, G=green, B=blue bars) show nearly **identical shape**, which confirms this: they are shifted versions of the same distribution rather than three independently distributed flat channels

### 3.3 ❌ `np.roll` Does Not Create a True Second Key

```python
# encryption.py — Line 80
sequence2 = np.roll(sequence, shift=len(sequence)//3)
```

The second diffusion pass uses the same sequence, just cyclically shifted by ⅓ of its length. This is **not an independent second key** — it is the same key with temporal displacement. The two XOR passes therefore partially cancel each other out (XORing twice with correlated keys is weaker than XORing once with uncorrelated keys), and the second pass does not restore histogram flatness.

---

## 4. Metric Cross-Reference

Despite the non-flat histograms, the measured metrics look deceptively good:

| Metric | Range Observed | Ideal Value | Verdict |
|---|---|---|---|
| **NPCR** | 99.53% – 99.63% | ≥ 99.6% | ✅ Excellent — single-bit change flips ~all pixels |
| **UACI** | 26.1% – 28.4% | ~33.3% | ⚠️ Below ideal — should be ~33% for truly random XOR |
| **MSE** | 6165 – 7939 | High is better | ✅ Sufficient pixel distance |
| **PSNR** | 9.1 – 10.2 dB | < 10 dB | ✅ Low enough (visually degraded) |
| **Entropy** | 7.71 – 7.92 bits | 8.0 bits | ⚠️ Below ideal — 8 bits means perfect uniformity |

The **Entropy** values (7.71–7.92 vs ideal 8.0) directly corroborate the histogram observation — entropy of exactly 8 bits/pixel is achieved *only* when all 256 values appear with equal frequency (flat histogram). Values < 8 quantify the degree of non-uniformity visible in the plots.

The **UACI at 26–28% vs ideal 33.3%** is another confirmation: ideal UACI for a uniformly random XOR key is 33.3% (the expected average absolute difference between a pixel and its XOR with a uniform random byte). The deficit shows the XOR key is not uniform.

---

## 5. Summary of Issues and Root Causes

```
Non-flat encrypted histogram
        │
        ├── Key stream non-uniformity
        │       │
        │       ├── sigmoid(cumsum()) → values cluster near 0.5
        │       │   not uniform in [0,1]
        │       │
        │       └── (x * 1e8) % 256 on non-uniform float
        │           → biased residue distribution
        │
        ├── Same key applied to R, G, B
        │       → inter-channel correlation preserved
        │       → all 3 histogram shapes are identical
        │
        └── np.roll second key = correlated keys
                → second XOR pass does not add
                  independent randomness
```

---

## 6. Recommended Fixes

### Fix 1 — Uniform Key Byte Generation
Replace the flawed `(x * 1e8) % 256` with a hash-based extraction that guarantees uniformity:

```python
# Better: use each generator output value to seed a hash, guaranteeing uniform 0-255
import hashlib

def seq_to_uniform_bytes(sequence, n_bytes):
    """Hash each chaotic float to a uniform byte."""
    result = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        # Take the float's raw bits — much more uniform than modular arithmetic
        val = sequence[i % len(sequence)]
        h = hashlib.sha256(val.tobytes()).digest()
        result[i] = h[0]  # first byte of SHA-256 is uniformly distributed
    return result
```

Or more simply and efficiently — use the generator to produce 3x as many values and map channel keys independently:

```python
# Simple fix: derive 3 independent byte streams from 3 separate key slices
key_R = ((seq_slice_1 * 1e8).astype(np.int64) & 0xFF).astype(np.uint8)
key_G = ((seq_slice_2 * 1e8).astype(np.int64) & 0xFF).astype(np.uint8)
key_B = ((seq_slice_3 * 1e8).astype(np.int64) & 0xFF).astype(np.uint8)
```

### Fix 2 — Independent Per-Channel Keys
Use separate noise seeds or separate generator branches for R, G, B keys — the WGAN generator already produces 4 independent chaotic branches. Assign branches to (R, G, B, scramble) separately:

```python
# In diffuse_image: use 3 separate key columns instead of np.repeat
key_matrix = np.stack([key_R, key_G, key_B], axis=1)  # (N, 3)
diffused = np.bitwise_xor(flat, key_matrix)
```

### Fix 3 — True Second Key
Instead of `np.roll`, use a different row of the sequence CSV (the generator was trained to produce 118 different sequences for exactly this purpose):

```python
sequence2 = load_long_sequence(seq_file, row=1)  # use second sequence row, not a roll
```

---

*Analysis by: code inspection of `encryption.py` + visual inspection of all 5 plots in `Encrypted_Results/plots/`*
