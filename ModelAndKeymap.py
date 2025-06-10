#ModelAndKeymap.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class HandKeypointDataset(Dataset):
    def __init__(self, image_paths, points, transform=None):
        self.image_paths = image_paths
        self.points = points
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        point = torch.tensor(self.points[idx], dtype=torch.float32)
        return img, point

# Simple cnn for predicting 2D keypoints from 128x128 rgb images
class EchoNet(nn.Module):
    def __init__(self):
        super(EchoNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),    # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1), # 8 -> 4 (keeping channels same here)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 2048),  # 4096 input from conv, big dense layer
            nn.ReLU(),
            nn.Linear(2048, 42)  # 21 keypoints x 2 (x, y)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)