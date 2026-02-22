import cv2
import numpy as np
import os

def check_histogram_uniformity(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    results = []
    # Check each channel
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        # Standard deviation of the histogram counts. 
        # Lower std_dev means more uniform.
        std_dev = np.std(hist)
        mean_val = np.mean(hist)
        # Normalized variation (Coefficient of Variation)
        variation = std_dev / mean_val if mean_val > 0 else 1.0
        results.append(variation)
        
    return np.mean(results)

output_root = "./Encrypted_Results/images"
encrypted_files = [f for f in os.listdir(output_root) if f.endswith("_encrypted.png")]

for f in encrypted_files:
    v = check_histogram_uniformity(os.path.join(output_root, f))
    print(f"{f} CV: {v:.4f}")
