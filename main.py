import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import os

from src.utils import set_seed, gradient_penalty
from src.dataset import ImageChaoticDataset
from src.models import LSTMGenerator, LSTMCritic

def main():
    parser = argparse.ArgumentParser(description="WGAN-GP for Chaotic Sequence Generation")
    parser.add_argument("--data_path", type=str, default="./KIIT-MiTA/train/images", help="Path to images")
    parser.add_argument("--image_size", type=int, default=8, help="Image resize dimension")
    parser.add_argument("--noise_dim", type=int, default=64, help="Noise dimension for generator")
    parser.add_argument("--hidden_dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--n_critic", type=int, default=3, help="Number of critic updates per generator update")
    parser.add_argument("--lambda_gp", type=int, default=10, help="Gradient penalty coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="wgangp_four_node_sequences_logistic_22500.csv", help="Output filename")
    parser.add_argument("--map_type", type=str, default="logistic", choices=["logistic", "tent", "cosine"], help="Chaotic map type")
    
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
                fake_seq = G(noise).unsqueeze(-1)

                outputs_real = C(real_seq)
                outputs_fake = C(fake_seq.detach())
                gp = gradient_penalty(C, real_seq, fake_seq.detach(), device)

                loss_C = -(torch.mean(outputs_real) - torch.mean(outputs_fake)) + args.lambda_gp * gp

                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()

            # Train Generator
            noise = torch.randn(batch_size, args.noise_dim).to(device)
            fake_seq = G(noise).unsqueeze(-1)
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
        seq = G(noise).detach().cpu().numpy().flatten()
        seq = (seq - seq.min()) / (seq.max() - seq.min() + 1e-8)
        all_sequences.append(seq)

    df = pd.DataFrame(all_sequences, columns=[f"Step_{i}" for i in range(seq_length)])
    df.index.name = "Sequence_ID"
    df.to_csv(args.output)
    print(f"\nSaved {num_sequences} generated sequences to {args.output}")

if __name__ == "__main__":
    main()