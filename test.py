import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
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
# Select a specific image
# ------------------------
SELECTED_IMAGE_PATH = "./TrainSet/image_s3r2_kiit_221.jpeg" 

selected_img = Image.open(SELECTED_IMAGE_PATH).convert("RGB")
IMG_W, IMG_H = selected_img.size

TOTAL_PIXELS = IMG_W * IMG_H

print(f"Selected image: {SELECTED_IMAGE_PATH}")
print(f"Image size: {IMG_W} x {IMG_H}")
print(f"Total pixels (number of sequences): {TOTAL_PIXELS}")

# ------------------------
# Chaotic Sequence Generator (UNCHANGED)
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
# Dataset
# ------------------------
class ImageChaoticDataset(Dataset):
    def __init__(self, dataset_path, image_size, seq_length, r=3.567):
        self.seq_length = seq_length
        self.r = r
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.image_paths = [
            os.path.join(dataset_path, f)
            for f in sorted(os.listdir(dataset_path))
            if f.lower().endswith(exts)
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        H, W = self.image_size

        corners = [
            img[0, 0, 0],
            img[0, 0, W - 1],
            img[0, H - 1, 0],
            img[0, H - 1, W - 1]
        ]

        seqs = []
        for c in corners:
            x0 = np.clip(c.item(), 1e-6, 1 - 1e-6)
            seq_data = sequence(
                r=self.r,
                x0=x0,
                length=self.seq_length,
                map_type="logistic"
            )
            seqs.append(torch.tensor(seq_data, dtype=torch.float32).unsqueeze(-1))

        seqs = torch.stack(seqs, dim=0)
        merged = seqs.mean(dim=0)
        return merged

# ------------------------
# Generator
# ------------------------
class LSTMGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.fc = nn.Linear(noise_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(seq_length)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward_branch(self, z):
        x = self.fc(z).unsqueeze(1).repeat(1, self.seq_length, 1)
        out, _ = self.lstm(x)
        out = self.fc_out(out)
        return out.squeeze(-1)

    def forward(self, z):
        o1 = self.forward_branch(z)
        o2 = self.forward_branch(z)
        o3 = self.forward_branch(z)
        o4 = self.forward_branch(z)
        merged = (o1 + o2 + o3 + o4) / 4.0
        merged = torch.sigmoid(self.bn(merged))
        return merged

# ------------------------
# Critic
# ------------------------
class LSTMCritic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ------------------------
# Gradient Penalty
# ------------------------
def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    interp = epsilon * real + (1 - epsilon) * fake
    interp.requires_grad_(True)

    prob = critic(interp)
    grads = torch.autograd.grad(
        outputs=prob,
        inputs=interp,
        grad_outputs=torch.ones_like(prob),
        create_graph=True,
        retain_graph=True
    )[0]

    grads = grads.view(batch_size, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

# ------------------------
# Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = (IMG_H, IMG_W)
seq_length = TOTAL_PIXELS
noise_dim = 64

dataset = ImageChaoticDataset(
    dataset_path="./TrainSet",
    image_size=image_size,
    seq_length=seq_length
)

loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

G = LSTMGenerator(noise_dim, 128, seq_length).to(device)
C = LSTMCritic(128).to(device)

opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_C = optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

# ------------------------
# Training Loop
# ------------------------
epochs = 20
n_critic = 3
lambda_gp = 10

for epoch in range(epochs):
    for real in loader:
        real = real.to(device)

        for _ in range(n_critic):
            z = torch.randn(real.size(0), noise_dim).to(device)
            fake = G(z).unsqueeze(-1)

            loss_C = -(C(real).mean() - C(fake.detach()).mean())
            loss_C += lambda_gp * gradient_penalty(C, real, fake.detach(), device)

            opt_C.zero_grad()
            loss_C.backward()
            opt_C.step()

        z = torch.randn(real.size(0), noise_dim).to(device)
        fake = G(z).unsqueeze(-1)
        loss_G = -C(fake).mean()

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | C: {loss_C.item():.4f} | G: {loss_G.item():.4f}")

# ------------------------
# Generate EXACTLY pixel-count sequences
# ------------------------
G.eval()
sequences = []

for i in range(TOTAL_PIXELS):
    z = torch.randn(1, noise_dim).to(device)
    seq = G(z).detach().cpu().numpy().flatten()
    seq = (seq - seq.min()) / (seq.max() - seq.min() + 1e-8)
    sequences.append(seq)

df = pd.DataFrame(sequences, columns=[f"Step_{i}" for i in range(seq_length)])
df.index.name = "Sequence_ID"

img_name = os.path.splitext(os.path.basename(SELECTED_IMAGE_PATH))[0]
out_name = f"wgangp_sequences_{img_name}_{TOTAL_PIXELS}.csv"
df.to_csv(out_name)

print(f"\nSaved {TOTAL_PIXELS} sequences to {out_name}")
