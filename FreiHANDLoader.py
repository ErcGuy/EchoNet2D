#FreiHANDLoader.py

import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm

def load_freihand_dataset(target_size=128):
    k_file = "dataset/FreiHAND_pub_v2/training_K.json"
    xyz_file = "dataset/FreiHAND_pub_v2/training_xyz.json"
    img_folder = "dataset/FreiHAND_pub_v2/training/rgb"

    with open(k_file, "r") as f:
        intrinsics = json.load(f)
    with open(xyz_file, "r") as f:
        xyz_kpts = json.load(f)

    img_paths = sorted(glob(os.path.join(img_folder, "*.jpg")))

    num_annotations = len(xyz_kpts)
    num_images = len(img_paths)

    # Some images are duplicated in FreiHAND — repeat keypoints to match
    repeat_factor = num_images // num_annotations
    assert num_images == num_annotations * repeat_factor, (
        f"Image count {num_images} not divisible by annotation count {num_annotations}"
    )

    xyz_kpts_repeated = xyz_kpts * repeat_factor
    intrinsics_repeated = intrinsics * repeat_factor

    assert len(img_paths) == len(xyz_kpts_repeated) == len(intrinsics_repeated)

    all_kpts_2d = []
    for K, pts3d in tqdm(zip(intrinsics_repeated, xyz_kpts_repeated), total=len(xyz_kpts_repeated)):
        K = np.array(K)
        pts3d = np.array(pts3d)  # shape: (21,3)

        pts2d = []
        for x, y, z in pts3d:
            if z == 0:
                pts2d.append([0.0, 0.0])
                continue
            # Project 3D to 2D using camera intrinsics
            u = (K[0][0] * x + K[0][2] * z) / z
            v = (K[1][1] * y + K[1][2] * z) / z
            pts2d.append([u, v])

        # Scale to match 128x128 model input (based on original size)
        orig_width = 224
        orig_height = 224
        pts2d_scaled = [[(u / orig_width) * target_size, (v / orig_height) * target_size] for u, v in pts2d]

        flat = [coord for pair in pts2d_scaled for coord in pair]
        all_kpts_2d.append(flat)

    return img_paths, all_kpts_2d