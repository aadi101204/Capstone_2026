import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import os

from src.utils import set_seed
from src.dataset import ImageChaoticDataset
from src.models import LSTMGenerator, LSTMCritic

def main():
    parser = argparse.ArgumentParser(description="WGAN-SN for Chaotic Sequence Generation")
    parser.add_argument("--data_path", type=str, default="./KIIT-MiTA/train/images", help="Path to images")
    parser.add_argument("--image_size", type=int, default=8, help="Image resize dimension")
    parser.add_argument("--noise_dim", type=int, default=64, help="Noise dimension for generator")
    parser.add_argument("--hidden_dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of critic updates per generator update")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="wgangp_four_node_sequences_logistic_22500.csv", help="Output filename")
    parser.add_argument("--map_type", type=str, default="logistic", choices=["logistic", "tent", "cosine"], help="Chaotic map type")
    
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Disable CuDNN to support double backpropagation for LSTMs (required for WGAN-GP Gradient Penalty)
    # Spectral Normalization (SN) handles stability, so we can re-enable CuDNN for speed
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = True
        print("Using WGAN-SN (CuDNN Enabled). Gradient Penalty removed to fix performance hangs.")

    image_size = (args.image_size, args.image_size)
    seq_length = args.image_size * args.image_size * 3 # H * W * C (assuming 3 channels)
    
    full_dataset = ImageChaoticDataset(
        dataset_path=args.data_path, 
        image_size=image_size, 
        seq_length=seq_length,
        map_type=args.map_type
    )
    
    dataset_size = len(full_dataset)
    print(f"Found {dataset_size} images in dataset at {args.data_path}")
    if dataset_size == 0:
        raise ValueError(f"No images found in {args.data_path}")
    
    dataset = Subset(full_dataset, range(dataset_size))
    dataloader = DataLoader(
        dataset, 
        batch_size=min(args.batch_size, dataset_size), 
        shuffle=True, 
        worker_init_fn=lambda worker_id: np.random.seed(args.seed)
    )

    G = LSTMGenerator(noise_dim=args.noise_dim, hidden_dim=args.hidden_dim, seq_length=seq_length).to(device)
    C = LSTMCritic(hidden_dim=args.hidden_dim).to(device)

    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

    fixed_noise = torch.randn(1, args.noise_dim).to(device)

    for epoch in range(args.epochs):
        for i, real_seq in enumerate(dataloader):
            real_seq = real_seq.to(device)
            batch_size = real_seq.size(0)

            # Train Critic
            for _ in range(args.n_critic):
                noise = torch.randn(batch_size, args.noise_dim).to(device)
                fake_seq = G(noise)

                outputs_real = C(real_seq)
                outputs_fake = C(fake_seq.detach())

                # Standard WGAN Loss (Stability provided by Spectral Norm in models.py)
                loss_C = -(torch.mean(outputs_real) - torch.mean(outputs_fake))

                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()

            # Train Generator
            noise = torch.randn(batch_size, args.noise_dim).to(device)
            fake_seq = G(noise)
            outputs_fake = C(fake_seq)

            loss_G = -torch.mean(outputs_fake)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss_C: {loss_C.item():.4f} Loss_G: {loss_G.item():.4f}")

    # Save model state
    torch.save(G.state_dict(), "generator.pth")
    print("Saved generator model to generator.pth")

    # Generate Sequences
    G.eval()
    num_sequences = 118
    all_sequences = []

    for i in range(num_sequences):
        noise = torch.randn(1, args.noise_dim).to(device)
        # gen shape: (1, 4, L) -> flatten to (4 * L)
        seq = G(noise).detach().cpu().numpy().flatten()
        all_sequences.append(seq)

    # Header for multi-node: Node0_Step0, ..., Node3_Step191
    columns = []
    for n in range(4):
        columns.extend([f"Node{n}_Step{j}" for j in range(seq_length)])
    
    df = pd.DataFrame(all_sequences, columns=columns)
    df.index.name = "Sequence_ID"
    df.to_csv(args.output)
    print(f"\nSaved {num_sequences} generated multi-node sequences to {args.output}")

if __name__ == "__main__":
    main()