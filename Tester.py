#test.py

#Sanity check to see if UseModel.py and FreiHANDLoader.py are working correctly.
#FreiHAND dataset must be used for this test.

import torch
import numpy as np
import matplotlib.pyplot as plt
from FreiHANDLoader import load_freihand_dataset
from ModelAndKeymap import EchoNet
from PIL import Image
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EchoNet()
model.load_state_dict(torch.load("echo_model.pth", map_location=device))
model.to(device).eval()

# Load dataset (you can reduce target_size if needed)
image_paths, gt_keypoints = load_freihand_dataset(target_size=128)

# Use a subset to validate (e.g., first 20 images)
val_image_paths = image_paths[:20]
val_gt_keypoints = gt_keypoints[:20]

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = TF.to_tensor(TF.resize(img, (128, 128)))
    img_t = (img_t - 0.5) / 0.5  # normalize
    img_t = img_t.unsqueeze(0).to(device)
    return img_t

def infer_keypoints(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    return output.cpu().numpy().reshape(-1, 2)

def calculate_errors(pred_kpts, gt_kpts):
    pred = np.array(pred_kpts)
    gt = np.array(gt_kpts).reshape(-1, 2)
    mse = np.mean((pred - gt) ** 2)
    mae = np.mean(np.abs(pred - gt))
    return mse, mae

def visualize_predictions(img_path, pred_kpts, gt_kpts):
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    pred = np.array(pred_kpts)
    gt = np.array(gt_kpts).reshape(-1, 2)

    # Scale keypoints from [0,128] back to original image size
    w, h = img.size
    pred_scaled = pred * [w / 128, h / 128]
    gt_scaled = gt * [w / 128, h / 128]

    plt.scatter(pred_scaled[:, 0], pred_scaled[:, 1], c='r', label='Predicted', marker='x')
    plt.scatter(gt_scaled[:, 0], gt_scaled[:, 1], c='g', label='Ground Truth', marker='o')
    plt.legend()
    plt.title(f"Red: Predicted, Green: GT")
    plt.show()

def main():
    total_mse, total_mae = 0.0, 0.0
    for i, (img_path, gt_kpts) in enumerate(zip(val_image_paths, val_gt_keypoints)):
        img_tensor = preprocess_image(img_path)
        pred_kpts = infer_keypoints(img_tensor)

        mse, mae = calculate_errors(pred_kpts, gt_kpts)
        total_mse += mse
        total_mae += mae

        print(f"Sample {i+1}: MSE={mse:.4f}, MAE={mae:.4f}")

        # Visualize first 5 samples
        if i < 5:
            visualize_predictions(img_path, pred_kpts, gt_kpts)

    avg_mse = total_mse / len(val_image_paths)
    avg_mae = total_mae / len(val_image_paths)
    print(f"\nValidation on {len(val_image_paths)} samples:")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

if __name__ == "__main__":
    main()
