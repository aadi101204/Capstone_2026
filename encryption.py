import os
import hashlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from outputParameters import analyze_encryption_metrics

# ------------------------
# Load chaotic sequences
# ------------------------
def load_long_sequence(file_path):
    """
    Load chaotic sequence file.
    Handles the new format with header and Sequence_ID index.
    """
    df = pd.read_csv(file_path, index_col=0)
    # The values are the chaotic sequence points
    sequence = df.values.flatten() 
    return sequence

# ------------------------
# Key Derivation via SHAKE-256
# ------------------------
def seq_to_key_bytes(sequence, n_bytes, salt=b""):
    """
    Derive n_bytes of cryptographically uniform key bytes from a chaotic float
    sequence using SHAKE-256 (an extendable-output hash function / XOF).

    The GAN sequence acts as a secret entropy seed. SHAKE-256 maps it to a
    perfectly uniform byte stream regardless of the GAN output distribution,
    fixing the bias introduced by the old (seq * 1e8) % 256 approach.

    salt: bytes — makes each derived stream independently keyed (per-channel,
    per-pass) without needing separate sequences.
    """
    raw = sequence.astype(np.float32).tobytes()
    xof = hashlib.shake_256(raw + salt)
    return np.frombuffer(xof.digest(n_bytes), dtype=np.uint8)

# ------------------------
# Image Scrambling (RGB)
# ------------------------
def scramble_image(image, sequence):
    H, W, C = image.shape
    flat = image.reshape(-1, C)  # shape: (H*W, 3)
    
    # Normalize sequence for sorting
    seq_min, seq_max = sequence.min(), sequence.max()
    if seq_max > seq_min:
        seq_norm = (sequence - seq_min) / (seq_max - seq_min)
    else:
        seq_norm = sequence
        
    needed = flat.shape[0]
    seq_repeated = np.tile(seq_norm, int(np.ceil(needed / len(seq_norm))))[:needed]
    
    indices = np.argsort(seq_repeated)  # permutation indices
    scrambled = flat[indices]
    scrambled_img = scrambled.reshape(H, W, C)
    return scrambled_img

# ------------------------
# Image Diffusion (XOR for RGB)
# ------------------------
def diffuse_image(image, sequence, pass_id=0):
    """
    XOR-diffuse image pixels using three independent per-channel key streams
    derived from the chaotic sequence via SHAKE-256.

    pass_id: int — changes the salt prefix so pass 0 and pass 1 produce
    completely uncorrelated key streams from the same sequence, replacing
    the old np.roll-based second key which was correlated.
    """
    H, W, C = image.shape
    flat = image.reshape(-1, C).astype(np.uint8)
    needed = flat.shape[0]

    # Salt prefix isolates each pass; channel suffix isolates R/G/B.
    # Result: 3 independent uniform byte streams — no shared key bytes across channels.
    pfx = f"pass{pass_id}_".encode()
    key_R = seq_to_key_bytes(sequence, needed, salt=pfx + b"R")
    key_G = seq_to_key_bytes(sequence, needed, salt=pfx + b"G")
    key_B = seq_to_key_bytes(sequence, needed, salt=pfx + b"B")

    key_matrix = np.stack([key_R, key_G, key_B], axis=1)  # (N, 3)
    diffused = np.bitwise_xor(flat, key_matrix)

    return diffused.reshape(H, W, C)

# ------------------------
# Full Encryption per Image
# ------------------------
def encrypt_image(image_path, seq_file, save_dir, filename_base):
    # load RGB image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Skipping: Could not load image at {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    
    # load chaotic key stream
    sequence = load_long_sequence(seq_file)
    
    # scrambling + double diffusion
    # pass_id=0 and pass_id=1 produce independent key streams via SHAKE-256 salts.
    # No np.roll needed — the salt change makes them cryptographically uncorrelated.
    scrambled = scramble_image(image, sequence)
    diffused1 = diffuse_image(scrambled, sequence, pass_id=0)
    encrypted = diffuse_image(diffused1, sequence, pass_id=1)

    # --- Output paths (three dedicated folders) ---
    scr_save_path  = os.path.join(save_dir, "scrambled",  f"{filename_base}_scrambled.png")
    enc_save_path  = os.path.join(save_dir, "encrypted",  f"{filename_base}_encrypted.png")
    hist_save_path = os.path.join(save_dir, "histograms", f"{filename_base}_histogram.png")

    # 1. Save scrambled image
    cv2.imwrite(scr_save_path, cv2.cvtColor(scrambled, cv2.COLOR_RGB2BGR))

    # 2. Save encrypted image
    cv2.imwrite(enc_save_path, cv2.cvtColor(encrypted, cv2.COLOR_RGB2BGR))

    # 3. Save histogram of encrypted image (R, G, B channels)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Encrypted Histogram — {filename_base}")
    ax.hist(encrypted[:, :, 0].flatten(), bins=256, color="red",   alpha=0.5, label="R")
    ax.hist(encrypted[:, :, 1].flatten(), bins=256, color="green", alpha=0.5, label="G")
    ax.hist(encrypted[:, :, 2].flatten(), bins=256, color="blue",  alpha=0.5, label="B")
    ax.set_xlim([0, 255])
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(hist_save_path, dpi=150)
    plt.close()
    
    # Metric Analysis
    try:
        # Build a stateless encrypt closure for differential UACI testing.
        # It takes a plain RGB ndarray and returns the fully encrypted RGB ndarray,
        # using the same sequence already loaded for this image.
        def _encrypt_fn(img_rgb):
            sc  = scramble_image(img_rgb, sequence)
            d1  = diffuse_image(sc, sequence, pass_id=0)
            return diffuse_image(d1, sequence, pass_id=1)

        metrics = analyze_encryption_metrics(image_path, enc_save_path, encrypt_fn=_encrypt_fn)

        metrics["Filename"] = filename_base
        return metrics
    except Exception as e:
        print(f"Error calculating metrics for {filename_base}: {e}")
        return {"Filename": filename_base, "Error": str(e)}

# ------------------------
# Bulk Export
# ------------------------
def run_bulk_encryption(dataset_path, seq_file, output_root, limit=10):
    os.makedirs(os.path.join(output_root, "scrambled"),  exist_ok=True)
    os.makedirs(os.path.join(output_root, "encrypted"),  exist_ok=True)
    os.makedirs(os.path.join(output_root, "histograms"), exist_ok=True)
    
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if limit:
        image_files = image_files[:limit]
        
    print(f"Starting bulk encryption for {len(image_files)} images...")
    results = []
    
    for img_name in image_files:
        path = os.path.join(dataset_path, img_name)
        base = os.path.splitext(img_name)[0]
        print(f"Processing {img_name}...")
        metric = encrypt_image(path, seq_file, output_root, base)
        if metric:
            results.append(metric)
    
    # Save CSV Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_root, "encryption_metrics.csv"), index=False)
    print(f"\nBulk encryption complete. Results saved to '{output_root}' folder.")

# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    seq_file = "sequence_file.csv"   
    # Dataset path from KIIT-MiTA
    dataset_dir = "./KIIT-MiTA/test/images/" 
    output_dir = "./Encrypted_Results"
    
    # Run through the folder (limit to 10 for quick verification, set to None for all)
    run_bulk_encryption(dataset_dir, seq_file, output_dir, limit=10)


