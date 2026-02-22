import cv2
import torch
import numpy as np
import os
import pandas as pd
from src.models import LSTMGenerator
from encryption import scramble_image, diffuse_image

def generate_chaotic_sequence(generator, device, noise_dim=64):
    """Generates a chaotic sequence using the trained WGAN-SN generator."""
    with torch.no_grad():
        noise = torch.randn(1, noise_dim).to(device)
        sequence = generator(noise).detach().cpu().numpy().flatten()
    return sequence

def process_video(video_path, model_path="generator.pth", output_path="encrypted_video.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Generator
    generator = LSTMGenerator(noise_dim=64, hidden_dim=128, seq_length=192).to(device)
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()
        print(f"Loaded generator from {model_path}")
    else:
        print("Error: generator.pth not found!")
        return

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path} ({width}x{height} @ {fps}fps)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for the encryption functions
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Generate fresh sequence for each frame (or per video if preferred)
        sequence = generate_chaotic_sequence(generator, device)

        # Apply Encryption
        scrambled = scramble_image(frame_rgb, sequence)
        diffused1 = diffuse_image(scrambled, sequence)
        
        # Second key stream (shifted)
        sequence2 = np.roll(sequence, shift=len(sequence)//3) 
        encrypted = diffuse_image(diffused1, sequence2)

        # Convert back to BGR for saving
        encrypted_bgr = cv2.cvtColor(encrypted, cv2.COLOR_RGB2BGR)
        out.write(encrypted_bgr)

        # Optional: Show preview
        cv2.imshow('Encrypted Feed', encrypted_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Encrypted video saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video Feed Encryption")
    parser.add_argument("--input", type=str, required=True, help="Path to input mp4 file")
    parser.add_argument("--output", type=str, default="encrypted_output.mp4", help="Path to output mp4 file")
    
    args = parser.parse_args()
    process_video(args.input, output_path=args.output)
