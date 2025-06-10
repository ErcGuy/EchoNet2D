#UseModel.py

#UI for loading an image and predicting keypoints using a pre-trained model

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from ModelAndKeymap import EchoNet
import matplotlib.pyplot as plt
import numpy as np

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once at start
model = EchoNet()
model.load_state_dict(torch.load("echo_model.pth", map_location=device))
model.to(device).eval()

def infer_keypoints(img_path):
    original_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = original_img.size

    # Resize using torchvision (matches test.py)
    img_resized = TF.resize(original_img, (128, 128))
    img_t = TF.to_tensor(img_resized)
    img_t = (img_t - 0.5) / 0.5
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)

    keypoints = output.cpu().numpy().reshape(-1, 2)

    return keypoints, original_img

def show_with_matplotlib(img, keypoints):
    w, h = img.size
    keypoints = np.array(keypoints)

    # Scale keypoints from 128 space back to original image size
    scaled = keypoints * [w / 128, h / 128]

    plt.imshow(img)
    plt.scatter(scaled[:, 0], scaled[:, 1], c='r', marker='x', label='Predicted')
    plt.title("Predicted Keypoints")
    plt.axis('off')
    plt.legend()
    plt.show()

def open_and_process():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return
    try:
        keypoints, original_img = infer_keypoints(path)
        show_with_matplotlib(original_img, keypoints)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image:\n{e}")

# Tkinter UI setup
root = tk.Tk()
root.title("EchoNet Keypoint Visualizer")

btn = tk.Button(root, text="Open Image and Predict", command=open_and_process)
btn.pack(pady=10)

label = tk.Label(root, text="(Prediction will open in matplotlib window)")
label.pack()

root.mainloop()