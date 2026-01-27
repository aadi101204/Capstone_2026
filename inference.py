import argparse
import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
from src.models import LSTMGenerator
from src.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="Inference for Chaotic Sequence Generation")
    parser.add_argument("--model_path", type=str, default="generator.pth", help="Path to trained generator model")
    parser.add_argument("--image_path", type=str, help="Path to a specific image to use for baseline (optional)")
    parser.add_argument("--noise_dim", type=int, default=64, help="Noise dimension for generator")
    parser.add_argument("--hidden_dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--seq_length", type=int, default=192, help="Sequence length")
    parser.add_argument("--num_sequences", type=int, default=10, help="Number of sequences to generate")
    parser.add_argument("--output", type=str, default="generated_sequences.csv", help="Output CSV filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    G = LSTMGenerator(noise_dim=args.noise_dim, hidden_dim=args.hidden_dim, seq_length=args.seq_length).to(device)
    if os.path.exists(args.model_path):
        G.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: {args.model_path} not found. Using untrained model.")
    
    G.eval()
    
    all_sequences = []
    
    # If image_path is provided, we can use it to determine seq_length if not specified
    if args.image_path:
        img = Image.open(args.image_path)
        w, h = img.size
        # The current model expects seq_length to match what it was trained with.
        # But for inference, we might want to generate sequences of a specific length.
        print(f"Image {args.image_path} size: {w}x{h}")

    print(f"Generating {args.num_sequences} sequences...")
    for i in range(args.num_sequences):
        noise = torch.randn(1, args.noise_dim).to(device)
        # gen shape: (1, 4, L)
        seq = G(noise).detach().cpu().numpy().flatten()
        all_sequences.append(seq)
        
    columns = []
    # Assuming the sequence was trained with 4 nodes
    for n in range(4):
        # We need to know seq_length per node.
        # G(noise) returns (1, 4, L), so flattened seq has length 4*L
        L = len(seq) // 4
        columns.extend([f"Node{n}_Step{j}" for j in range(L)])

    df = pd.DataFrame(all_sequences, columns=columns)
    df.index.name = "Sequence_ID"
    df.to_csv(args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
