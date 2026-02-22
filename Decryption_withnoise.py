import numpy as np
import cv2
import matplotlib.pyplot as plt
from decryption import load_long_sequence, inverse_diffuse, inverse_scramble

def post_denoise(decrypted_img, method="median", ksize=3):
    """
    Denoise the *decrypted* image (not the ciphertext).
    method: "median" or "nlmeans"
    ksize: median kernel size (must be odd >=3)
    """
    if method == "median":
        # medianBlur supports multi-channel images
        k = max(3, (ksize // 2) * 2 + 1)
        return cv2.medianBlur(decrypted_img, k)
    elif method == "nlmeans":
        # parameters can be tuned
        return cv2.fastNlMeansDenoisingColored(decrypted_img, None, h=10, hColor=10,
                                               templateWindowSize=7, searchWindowSize=21)
    else:
        raise ValueError("Unknown denoise method")

def decrypt_noisy_image(noisy_enc_path, seq_file,
                        save_path="decrypted_from_noisy.png",
                        roll_shift=50,
                        denoise_method="median",
                        median_ksize=3):
    """
    1) Load noisy encrypted image
    2) Inverse diffuse with seq2 (rolled)
    3) Inverse diffuse with seq
    4) Inverse scramble
    5) Denoise resulting plaintext (median or nlmeans)
    """
    # load noisy encrypted image (BGR -> RGB)
    noisy_bgr = cv2.imread(noisy_enc_path, cv2.IMREAD_COLOR)
    if noisy_bgr is None:
        raise ValueError(f"Could not load noisy encrypted image at {noisy_enc_path}")
    noisy = cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2RGB)

    # load key stream
    sequence = load_long_sequence(seq_file)
    # debug info
    H, W, C = noisy.shape
    print(f"Loaded noisy image {noisy_enc_path} shape={noisy.shape}, sequence length={len(sequence)}")

    # Step A: undo the LAST diffusion first (the one using rolled/shifted sequence)
    seq2 = np.roll(sequence, roll_shift)
    step1 = inverse_diffuse(noisy, seq2)

    # Step B: undo the first diffusion
    step2 = inverse_diffuse(step1, sequence)

    # Step C: undo scrambling (permutation)
    decrypted_noisy = inverse_scramble(step2, sequence)

    # Step D: now denoise the decrypted image (plaintext)
    denoised = post_denoise(decrypted_noisy, method=denoise_method, ksize=median_ksize)

    # save result (convert back to BGR)
    cv2.imwrite(save_path, cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))
    print(f"Saved decrypted (post-denoised) image to {save_path}")

    # quick visualization
    plt.figure(figsize=(10,9))
    plt.subplot(2,2,1); plt.title("Noisy Encrypted"); plt.imshow(noisy); plt.axis("off")
    plt.subplot(2,2,2); plt.title("After inverse diffusions + unscramble (noisy plaintext)"); plt.imshow(decrypted_noisy); plt.axis("off")
    plt.subplot(2,2,3); plt.title("Post-denoised plaintext"); plt.imshow(denoised); plt.axis("off")
    plt.tight_layout()
    plt.show()

    return denoised

if __name__ == "__main__":
    seq_file = "wgangp_four_map_sequences_multichannel.csv"
    noisy_enc_path = "noisy_encrypted.png"   # your noisy cipher
    decrypt_noisy_image(noisy_enc_path, seq_file,
                        save_path="decrypted_from_noisy_postdenoise.png",
                        roll_shift=50,
                        denoise_method="median",
                        median_ksize=3)
