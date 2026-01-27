import torch
from src.models import LSTMGenerator, LSTMCritic
from src.dataset import ImageChaoticDataset
from torch.utils.data import DataLoader
import os

def test_multi_node_architecture():
    noise_dim = 64
    hidden_dim = 128
    seq_length = 192
    batch_size = 4

    device = torch.device("cpu")
    
    print("Testing Multi-Node Generator...")
    G = LSTMGenerator(noise_dim=noise_dim, hidden_dim=hidden_dim, seq_length=seq_length).to(device)
    noise = torch.randn(batch_size, noise_dim).to(device)
    gen_out = G(noise)
    print(f"Generator output shape: {gen_out.shape}")
    assert gen_out.shape == (batch_size, 4, seq_length), f"Expected {(batch_size, 4, seq_length)}, got {gen_out.shape}"

    print("Testing Multi-Node Critic...")
    C = LSTMCritic(hidden_dim=hidden_dim).to(device)
    # Critic expects (Batch, 4, Seq)
    crit_out = C(gen_out.detach())
    print(f"Critic output shape: {crit_out.shape}")
    assert crit_out.shape == (batch_size, 1), f"Expected {(batch_size, 1)}, got {crit_out.shape}"

    print("Final shape verification passed!")

if __name__ == "__main__":
    test_multi_node_architecture()
