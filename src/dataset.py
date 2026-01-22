import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .chaos import sequence

class ImageChaoticDataset(Dataset):
    def __init__(self, dataset_path, image_size=(8, 8), seq_length=192, r=3.567, map_type="logistic"):
        self.seq_length = seq_length
        self.r = r
        self.image_size = image_size
        self.map_type = map_type
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        
        # Detect directory structure
        if os.path.isdir(dataset_path):
            subdirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            if subdirs:
                # Assuming ImageFolder style
                self.image_paths = []
                for subdir in subdirs:
                    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                    files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.lower().endswith(exts)]
                    self.image_paths.extend(files)
            else:
                # Flat folder
                exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                files = [f for f in sorted(os.listdir(dataset_path)) if f.lower().endswith(exts)]
                self.image_paths = [os.path.join(dataset_path, f) for f in files]
        else:
            raise ValueError(f"Dataset path {dataset_path} is not a directory.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
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
            seq_data = sequence(r=self.r, x0=x0, length=self.seq_length, map_type=self.map_type)
            seqs.append(torch.tensor(seq_data, dtype=torch.float32).unsqueeze(-1))

        seqs = torch.stack(seqs, dim=0)
        merged = seqs.mean(dim=0)
        return merged
