import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import os
import pandas as pd

# ------------------------
# Reproducibility
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------
# Chaotic Sequence Generator
# ------------------------
def sequence(r, x0, length, map_type="logistic"):
    seq = np.zeros(length)
    seq[0] = x0
    for i in range(1, length):
        if map_type == "logistic":
            seq[i] = r * seq[i-1] * (1 - seq[i-1]) 
        elif map_type == "tent":
            seq[i] = r * seq[i-1] if seq[i-1] < 0.5 else r * (1 - seq[i-1])
        elif map_type == "cosine":
            seq[i] = r * (1 - np.cos(np.pi * seq[i-1])) / 2.0
        else:
            raise ValueError("Unknown map_type")
        seq[i] = np.clip(seq[i], 1e-6, 1-1e-6)
    return seq

# ------------------------
# Dataset with 4 chaotic sequences (corners)
# ------------------------
class ImageChaoticDataset(Dataset):
    def __init__(self, dataset_path, image_size=(8, 8), seq_length=192, r=3.567):
        self.seq_length = seq_length
        self.r = r
        self.image_size = image_size
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.images = datasets.ImageFolder(root=dataset_path, transform=transform)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, _ = self.images[idx]
        H, W = self.image_size
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        corners = [
            img[0, 0, 0],       
            img[0, 0, W-1],     
            img[0, H-1, 0],     
            img[0, H-1, W-1]    
        ]

        seqs = []
        for c in corners:
            x0 = np.clip(c.item(), 1e-6, 1-1e-6)
            seq_data = sequence(r=self.r, x0=x0, length=self.seq_length, map_type="logistic")
            seqs.append(torch.tensor(seq_data, dtype=torch.float32).unsqueeze(-1))

        seqs = torch.stack(seqs, dim=0)
        merged = seqs.mean(dim=0)  # (seq_length, 1)
        return merged

# ------------------------
# Generator (shared LSTM across 4 branches)
# ------------------------
class LSTMGenerator(nn.Module):
    def __init__(self, noise_dim=64, hidden_dim=128, seq_length=192):
        super(LSTMGenerator, self).__init__()
        self.seq_length = seq_length
        self.fc = nn.Linear(noise_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)  # shared
        self.bn = nn.BatchNorm1d(seq_length)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward_branch(self, z):
        x = self.fc(z).unsqueeze(1).repeat(1, self.seq_length, 1)
        lstm_out, _ = self.lstm(x)
        out = self.fc_out(lstm_out)
        return out.squeeze(-1)

    def forward(self, z):
        out1 = self.forward_branch(z)
        out2 = self.forward_branch(z)
        out3 = self.forward_branch(z)
        out4 = self.forward_branch(z)
        merged = (out1 + out2 + out3 + out4) / 4.0
        merged = torch.sigmoid(self.bn(merged))  # [0,1]
        return merged

# ------------------------
# Critic (unchanged WGAN-GP)
# ------------------------
class LSTMCritic(nn.Module):
    def __init__(self, hidden_dim=128):
        super(LSTMCritic, self).__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        lstm_out, _ = self.lstm(seq)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

# ------------------------
# Gradient Penalty
# ------------------------
def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    prob_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ------------------------
# Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_size = (8, 8)
seq_length = image_size[0] * image_size[1] * 3  # 192
noise_dim = 64

data_path = "./seg_test/seg_test"
full_dataset = ImageChaoticDataset(dataset_path=data_path, image_size=image_size, seq_length=seq_length)
dataset = Subset(full_dataset, range(200))  # smaller subset for faster run
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(42))

G = LSTMGenerator(noise_dim=noise_dim, hidden_dim=128, seq_length=seq_length).to(device)
C = LSTMCritic(hidden_dim=128).to(device)

opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_C = optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

# ------------------------
# Training Loop (WGAN-GP)
# ------------------------
epochs = 20
n_critic = 3
lambda_gp = 10
fixed_noise = torch.randn(1, noise_dim).to(device)

for epoch in range(epochs):
    for i, real_seq in enumerate(dataloader):
        real_seq = real_seq.to(device)
        batch_size = real_seq.size(0)

        # Train Critic
        for _ in range(n_critic):
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_seq = G(noise).unsqueeze(-1)

            outputs_real = C(real_seq)
            outputs_fake = C(fake_seq.detach())
            gp = gradient_penalty(C, real_seq, fake_seq.detach(), device)

            loss_C = -(torch.mean(outputs_real) - torch.mean(outputs_fake)) + lambda_gp * gp

            opt_C.zero_grad()
            loss_C.backward()
            opt_C.step()

        # Train Generator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_seq = G(noise).unsqueeze(-1)
        outputs_fake = C(fake_seq)

        loss_G = -torch.mean(outputs_fake)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss_C: {loss_C.item():.4f} Loss_G: {loss_G.item():.4f}")

# ------------------------
# Generate Sequence
# ------------------------
G.eval()
raw_generated_seq = G(fixed_noise).detach().cpu().numpy().flatten()
generated_seq = (raw_generated_seq - raw_generated_seq.min()) / (raw_generated_seq.max() - raw_generated_seq.min() + 1e-8)

print("\nGenerated Chaotic Sequence:\n", generated_seq)

# ------------------------
# Save Multiple Generated Sequences to CSV
# ------------------------
num_sequences = 118
all_sequences = []

for i in range(num_sequences):
    noise = torch.randn(1, noise_dim).to(device)
    seq = G(noise).detach().cpu().numpy().flatten()
    seq = (seq - seq.min()) / (seq.max() - seq.min() + 1e-8)
    all_sequences.append(seq)

df = pd.DataFrame(all_sequences, columns=[f"Step_{i}" for i in range(seq_length)])
df.index.name = "Sequence_ID"

df.to_csv("wgangp_four_node_sequences_logistic_22500.csv")
print(f"\nSaved {num_sequences} generated sequences to wgangp_four_node_sequences.csv")