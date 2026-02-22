import numpy as np
import cv2
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
    # The values are the chaotic sequence points
    sequence = df.values.flatten() 
    return sequence

# ------------------------
# Inverse Diffusion (XOR again with same sequence)
# ------------------------
def inverse_diffuse(image, sequence):
    H, W, C = image.shape
    flat = image.reshape(-1, C).astype(np.uint8)
    
    needed = flat.shape[0]
    seq_repeated = np.tile(sequence, int(np.ceil(needed / len(sequence))))[:needed]
    # Use high Precision multiplier to match encryption
    seq_int = ((seq_repeated * 1e8) % 256).astype(np.uint8)  
    
    # expand chaotic key across 3 channels
    seq_int_expanded = np.repeat(seq_int[:, np.newaxis], C, axis=1)
    recovered = np.bitwise_xor(flat, seq_int_expanded)
    
    return recovered.reshape(H, W, C)

# ------------------------
# Inverse Scrambling (undo permutation)
# ------------------------
def inverse_scramble(image, sequence):
    H, W, C = image.shape
    flat = image.reshape(-1, C)
    
    # Normalize sequence for sorting
    seq_min, seq_max = sequence.min(), sequence.max()
    if seq_max > seq_min:
        seq_norm = (sequence - seq_min) / (seq_max - seq_min)
    else:
        seq_norm = sequence
        
    needed = flat.shape[0]
    seq_repeated = np.tile(seq_norm, int(np.ceil(needed / len(seq_norm))))[:needed]
    
    indices = np.argsort(seq_repeated)  # forward permutation index
    inv_indices = np.argsort(indices)   # inverse permutation index
    
    unscrambled = flat[inv_indices]
    unscrambled_img = unscrambled.reshape(H, W, C)
    return unscrambled_img

# ------------------------
# Full Decryption
# ------------------------
def decrypt_image(enc_path, seq_file, save_path="decrypted.png"):
    # load encrypted RGB image
    encrypted = cv2.imread(enc_path, cv2.IMREAD_COLOR)
    if encrypted is None:
        raise ValueError(f"Could not load encrypted image at {enc_path}")
    encrypted = cv2.cvtColor(encrypted, cv2.COLOR_BGR2RGB)
    
    # load chaotic key stream
    sequence = load_long_sequence(seq_file)
    
    # Deriving the second key stream used in encryption (shifted version)
    sequence2 = np.roll(sequence, shift=len(sequence)//3) 

    # REVERSE STEPS: 
    # 1. Reverse Diffusion 2 (with shifted sequence)
    # 2. Reverse Diffusion 1 (with original sequence)
    # 3. Reverse Scrambling
    
    step1 = inverse_diffuse(encrypted, sequence2)
    step2 = inverse_diffuse(step1, sequence)
    decrypted = inverse_scramble(step2, sequence)
    
    # save result (convert back to BGR for OpenCV)
    cv2.imwrite(save_path, cv2.cvtColor(decrypted, cv2.COLOR_RGB2BGR))
    
    # plot results
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Encrypted Image")
    plt.imshow(encrypted)
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.title("Decrypted (Recovered) Image")
    plt.imshow(decrypted)
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("decryption_result.png")
    print(f"Decryption complete. Result saved to {save_path} and plot saved to decryption_result.png")
    # plt.show()
    
    return decrypted

# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    seq_file = "sequence_file.csv"   
    # Use one of the encrypted images from the results folder
    enc_image_path = "./Encrypted_Results/images/image_s3r2_kiit_1006_encrypted.png" 
    if os.path.exists(enc_image_path):
        decrypt_image(enc_image_path, seq_file, save_path="decrypted_1006.png")
    else:
        print(f"Encrypted image not found at {enc_image_path}. Run encryption first.")
