import numpy as np
import cv2
import math


# ------------------------
# NPCR (Number of Pixels Change Rate)
# ------------------------
def npcr(img1, img2):
    assert img1.shape == img2.shape, "Images must be the same size"
    diff = np.not_equal(img1, img2).astype(np.float32)
    return np.sum(diff) / diff.size * 100


# ------------------------
# UACI (Unified Average Changing Intensity) -- visual
# ------------------------
def uaci(img1, img2):
    assert img1.shape == img2.shape, "Images must be the same size"
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    return np.mean(diff / 255.0) * 100


# ------------------------
# PSNR (Peak Signal-to-Noise Ratio)
# ------------------------
def psnr(img1, img2):
    assert img1.shape == img2.shape, "Images must be the same size"
    mse_val = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse_val == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse_val))


# ------------------------
# MSE (Mean Squared Error)
# ------------------------
def mse(img1, img2):
    assert img1.shape == img2.shape, "Images must be the same size"
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)


# ------------------------
# Entropy -- per-channel average (R, G, B)
# ------------------------
def entropy(img):
    """
    Shannon entropy as the average of per-channel (R, G, B) entropies.

    Why not grayscale:
        RGB->Gray (0.299R + 0.587G + 0.114B) averages three independent
        uniform distributions, concentrating the result near 128 via the
        Central Limit Theorem. This drops measured entropy to ~7.6 even
        when each channel is perfectly uniform (~8 bits). Measuring
        per-channel and averaging gives the correct value.

    Ideal: 8.0 bits per channel.
    """
    if img.ndim == 2:
        # Grayscale fallback
        values, counts = np.unique(img.flatten(), return_counts=True)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    channel_entropies = []
    for c in range(img.shape[2]):
        ch = img[:, :, c].flatten()
        values, counts = np.unique(ch, return_counts=True)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        channel_entropies.append(-np.sum(probs * np.log2(probs)))
    return float(np.mean(channel_entropies))


# ------------------------
# Differential UACI (cryptographic metric)
# ------------------------
def differential_uaci(encrypt_fn, image, n_trials=5):
    """
    Flip one random pixel by +1, re-encrypt, measure mean |cipher1 - cipher2|.

    Why this matters:
        Visual UACI = mean(|original - encrypted|) is bounded by the original
        image pixel distribution. For mid-grey surveillance images (pixels ~128)
        it is mathematically capped at ~25-28% regardless of key quality -- this
        is NOT a cipher weakness; it is a consequence of image content.

        Differential UACI tests single-bit-change avalanche effect, the true
        cryptographic diffusion quality. Ideal: ~33.3%.
    """
    results = []
    H, W, C = image.shape
    for _ in range(n_trials):
        r = np.random.randint(H)
        c_idx = np.random.randint(W)
        ch = np.random.randint(C)
        img2 = image.copy()
        img2[r, c_idx, ch] = (int(img2[r, c_idx, ch]) + 1) % 256

        enc1 = encrypt_fn(image)
        enc2 = encrypt_fn(img2)

        diff = np.abs(enc1.astype(np.float32) - enc2.astype(np.float32))
        results.append(np.mean(diff / 255.0) * 100)
    return float(np.mean(results))


# ------------------------
# Main Analysis Function
# ------------------------
def analyze_encryption_metrics(original_path, encrypted_path, encrypt_fn=None):
    img1 = cv2.imread(original_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(encrypted_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError("Could not load one or both images")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    npcr_val    = npcr(img1, img2)
    uaci_val    = uaci(img1, img2)
    psnr_val    = psnr(img1, img2)
    mse_val     = mse(img1, img2)
    entropy_val = entropy(img2)           # per-channel avg

    d_uaci_val = None
    if encrypt_fn is not None:
        d_uaci_val = differential_uaci(encrypt_fn, img1)

    print(" Encryption Quality Metrics")
    print("=" * 56)
    print(f"NPCR           : {npcr_val:.4f} %  (ideal >= 99.60 %)")
    print(f"UACI (visual)  : {uaci_val:.4f} %  (25-30 % is normal for mid-grey images)")
    print(f"MSE            : {mse_val:.4f}")
    print(f"PSNR           : {psnr_val:.4f} dB")
    print(f"Entropy        : {entropy_val:.4f} bits  (per-channel avg; ideal = 8.0)")
    if d_uaci_val is not None:
        print(f"UACI (diff.)   : {d_uaci_val:.4f} %  (cryptographic avalanche; ideal ~33.3 %)")

    result = {
        "NPCR":    npcr_val,
        "UACI":    uaci_val,
        "MSE":     mse_val,
        "PSNR":    psnr_val,
        "Entropy": entropy_val,
    }
    if d_uaci_val is not None:
        result["UACI_diff"] = d_uaci_val
    return result


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    orig = "./image.jpeg"
    enc  = "encrypted.png"
    try:
        results = analyze_encryption_metrics(orig, enc)
        print("\n Analysis complete!")
    except Exception as e:
        print(f" Error during analysis: {e}")
