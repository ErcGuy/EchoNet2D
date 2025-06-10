#TrainEchonet.py

import os
import json
from glob import glob
from PIL import Image
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# Import dataset and model here
from FreiHANDLoader import load_freihand_dataset
from ModelAndKeymap import HandKeypointDataset, EchoNet  


# Instantiate the CNN model
model = EchoNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def train_echo_model(image_paths, keypoints, epochs=100, batch_size=32, lr=0.001):
    # Define image transformations (resize, tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Create dataset and dataloader using imported HandKeypointDataset
    dataset = HandKeypointDataset(image_paths, keypoints, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate model and move to device inside function (optional, you can move this outside)
    model = EchoNet()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_path = "echo_model.pth"
    epoch_path = "echo_epoch.txt"

    start_epoch = 0
    # Load checkpoint if available
    if os.path.exists(checkpoint_path) and os.path.exists(epoch_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        with open(epoch_path, "r") as f:
            start_epoch = int(f.read().strip())
        print(f"✅ Loaded checkpoint from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting fresh")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        start_epoch_time = time.time()
        print(f"\nStarting epoch {epoch+1}/{epochs}")

        for i, (imgs, targets) in enumerate(loader):
            batch_start = time.time()
            imgs, targets = imgs.to(device), targets.to(device)

            preds = model(imgs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            batch_time = time.time() - batch_start
            if i % 100 == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}, Time: {batch_time:.3f}s")
                euclidean = torch.sqrt(((preds - targets) ** 2).reshape(-1, 21, 2).sum(dim=2))
                print("Mean keypoint error:", euclidean.mean().item())
            

        epoch_time = time.time() - start_epoch_time
        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}, Epoch Time: {epoch_time:.1f}s")

        # Save checkpoint and current epoch number
        torch.save(model.state_dict(), checkpoint_path)
        with open(epoch_path, "w") as f:
            f.write(str(epoch + 1))
        print(f"✅ Checkpoint saved at epoch {epoch+1}")


# Load FreiHAND dataset (your external loader returns image paths and flattened 2D keypoints)
image_paths, flat_kpts = load_freihand_dataset(target_size=128)

# Entry point
if __name__ == "__main__":
    train_echo_model(image_paths, flat_kpts)