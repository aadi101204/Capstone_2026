import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------------
# Noise Functions
# ------------------------
def add_salt_pepper_noise(img, prob=0.02):
    noisy = np.copy(img)
    rnd = np.random.rand(*img.shape[:2])
    noisy[rnd < (prob / 2)] = 255
    noisy[rnd > 1 - (prob / 2)] = 0
    return noisy

def add_gaussian_noise(img, mean=0, sigma=15):
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy

# ------------------------
# Rotation
# ------------------------
def rotate_image(img, angle=90):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Only 90, 180, 270 supported")


def apply_noise_and_rotation(enc_path, save_path="cipher_noisy_rot.png",
                             angle=90, sp_prob=0.02, gauss_sigma=20):
    enc_bgr = cv2.imread(enc_path, cv2.IMREAD_COLOR)
    if enc_bgr is None:
        raise ValueError(f"Could not load encrypted image at {enc_path}")
    enc = cv2.cvtColor(enc_bgr, cv2.COLOR_BGR2RGB)

    rotated = rotate_image(enc, angle=angle)
    sp_noisy = add_salt_pepper_noise(rotated, prob=sp_prob)
    final_noisy = add_gaussian_noise(sp_noisy, sigma=gauss_sigma)

    cv2.imwrite(save_path, cv2.cvtColor(final_noisy, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1); plt.title("Encrypted"); plt.imshow(enc); plt.axis("off")
    plt.subplot(1,3,2); plt.title(f"Rotated {angle}° + S&P Noise"); plt.imshow(sp_noisy); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Final Noisy (with Gaussian)"); plt.imshow(final_noisy); plt.axis("off")
    plt.tight_layout(); plt.show()

    return final_noisy

# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    apply_noise_and_rotation("encrypted.png",
                             save_path="cipher_noisy_rot.png",
                             angle=90, sp_prob=0.02, gauss_sigma=20)
