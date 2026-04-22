import numpy as np
import cv2
import hashlib
import matplotlib.pyplot as plt
import os
import pandas as pd

# ------------------------
# Load chaotic sequences
# ------------------------
def load_long_sequence(file_path):
    """
    Load chaotic sequence file.
    Handles the format with header and Sequence_ID index.
    """
    df = pd.read_csv(file_path, index_col=0)
    sequence = df.values.flatten()
    return sequence

# ------------------------
# Key Derivation via SHAKE-256  (mirrors encryption.py exactly)
# ------------------------
def seq_to_key_bytes(sequence, n_bytes, salt=b""):
    """
    Derive n_bytes of cryptographically uniform key bytes from a chaotic float
    sequence using SHAKE-256 (an extendable-output hash function / XOF).

    Must be called with the same (sequence, n_bytes, salt) tuple that was used
    during encryption to reproduce the identical key stream.
    """
    raw = sequence.astype(np.float32).tobytes()
    xof = hashlib.shake_256(raw + salt)
    return np.frombuffer(xof.digest(n_bytes), dtype=np.uint8)

# ------------------------
# Inverse Diffusion  (SHAKE-256 — mirrors encryption.py diffuse_image)
# XOR is self-inverse: applying the same key stream undoes the encryption.
# ------------------------
def inverse_diffuse(image, sequence, pass_id=0):
    """
    Undo one XOR-diffusion pass.

    pass_id must match the value used in the corresponding encryption pass:
      - To undo diffuse_image(..., pass_id=1) call inverse_diffuse(..., pass_id=1)
      - To undo diffuse_image(..., pass_id=0) call inverse_diffuse(..., pass_id=0)
    """
    H, W, C = image.shape
    flat = image.reshape(-1, C).astype(np.uint8)
    needed = flat.shape[0]

    pfx = f"pass{pass_id}_".encode()
    key_R = seq_to_key_bytes(sequence, needed, salt=pfx + b"R")
    key_G = seq_to_key_bytes(sequence, needed, salt=pfx + b"G")
    key_B = seq_to_key_bytes(sequence, needed, salt=pfx + b"B")

    key_matrix = np.stack([key_R, key_G, key_B], axis=1)  # (N, 3)
    recovered = np.bitwise_xor(flat, key_matrix)

    return recovered.reshape(H, W, C)

# ------------------------
# Inverse Scrambling (undo pixel permutation)
# ------------------------
def inverse_scramble(image, sequence):
    H, W, C = image.shape
    flat = image.reshape(-1, C)

    seq_min, seq_max = sequence.min(), sequence.max()
    if seq_max > seq_min:
        seq_norm = (sequence - seq_min) / (seq_max - seq_min)
    else:
        seq_norm = sequence

    needed = flat.shape[0]
    seq_repeated = np.tile(seq_norm, int(np.ceil(needed / len(seq_norm))))[:needed]

    indices = np.argsort(seq_repeated)   # forward permutation
    inv_indices = np.argsort(indices)    # inverse permutation

    unscrambled = flat[inv_indices]
    return unscrambled.reshape(H, W, C)

# ------------------------
# Full Decryption
# Encryption order: scramble → diffuse(pass_id=0) → diffuse(pass_id=1)
# Decryption order (exact reverse): undo pass_id=1 → undo pass_id=0 → unscramble
# ------------------------
def decrypt_image(enc_path, seq_file, save_path="decrypted.png"):
    encrypted = cv2.imread(enc_path, cv2.IMREAD_COLOR)
    if encrypted is None:
        raise ValueError(f"Could not load encrypted image at {enc_path}")
    encrypted = cv2.cvtColor(encrypted, cv2.COLOR_BGR2RGB)

    sequence = load_long_sequence(seq_file)

    # Reverse the two XOR passes in reverse order
    step1 = inverse_diffuse(encrypted, sequence, pass_id=1)
    step2 = inverse_diffuse(step1,     sequence, pass_id=0)
    decrypted = inverse_scramble(step2, sequence)

    cv2.imwrite(save_path, cv2.cvtColor(decrypted, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Encrypted Image")
    plt.imshow(encrypted)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Decrypted (Recovered) Image")
    plt.imshow(decrypted)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("decryption_result.png")
    plt.close()
    print(f"Decryption complete. Result saved to {save_path}")

    return decrypted

# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    seq_file = "sequence_file.csv"
    # Pick any encrypted image produced by encryption.py
    enc_image_path = "./FinalResults/encrypted/image_s3r2_kiit_132_encrypted.png"
    if os.path.exists(enc_image_path):
        decrypt_image(enc_image_path, seq_file, save_path="decrypted_output.png")
    else:
        print(
            f"Encrypted image not found at {enc_image_path}.\n"
            "Run encryption.py first to generate encrypted images."
        )
