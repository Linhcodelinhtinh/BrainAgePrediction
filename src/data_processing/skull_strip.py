import os
import json
import ants
import random
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime

json_path = "mri_dataset.json"
original_raw_root = "data"     

input_root = "data_preprocessed/mni152_registration" 
output_root = "data_preprocessed/skull_stripping"
sample_root = "data_sample"

log_file = "logs/stage3.log"
HD_BET_DEVICE = "cuda" # Thay bằng "cpu" nếu không có GPU NVIDIA

os.makedirs(output_root, exist_ok=True)
os.makedirs(sample_root, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def log(msg):
    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)

with open(json_path, "r") as f:
    paths = json.load(f)

sample_file = "sample_paths.json"
if os.path.exists(sample_file):
    sample_paths = set(json.load(open(sample_file)))
else:
    sample_paths = set(random.sample(paths, min(3, len(paths))))
    json.dump(list(sample_paths), open(sample_file, "w"))

def save_slices_basic(img, save_path, title=""):
    """Lưu ảnh xám cơ bản"""
    img = img.numpy() if isinstance(img, ants.ANTsImage) else img
    z, y, x = img.shape[2]//2, img.shape[1]//2, img.shape[0]//2

    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    axes[0].imshow(np.rot90(img[:, :, z]), cmap="gray")
    axes[1].imshow(np.rot90(img[:, y, :]), cmap="gray")
    axes[2].imshow(np.rot90(img[x, :, :]), cmap="gray")
    axes[0].set_title("Axial"); axes[1].set_title("Coronal"); axes[2].set_title("Sagittal")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

log("BẮT ĐẦU STAGE 3: HD-BET SKULL STRIPPING...")

for idx, orig_path in enumerate(paths):
    try:
        rel_path = os.path.relpath(orig_path, original_raw_root) 
        input_path = os.path.join(input_root, rel_path) 
        save_path = os.path.join(output_root, rel_path) 

        if os.path.exists(save_path):
            continue
            
        if not os.path.exists(input_path):
            log(f"[{idx+1}/{len(paths)}] BỎ QUA: {input_path} không tồn tại")
            continue
            
        log(f"[{idx+1}/{len(paths)}] {input_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Chạy HD-BET bằng subprocess
        cmd = [
            "hd-bet",
            "-i", input_path,
            "-o", save_path,
            "-device", HD_BET_DEVICE,
            "--disable_tta",
            "--save_bet_mask"
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"HD-BET Error:\n{result.stderr}")

        # Visualize nếu nằm trong danh sách sample
        if orig_path in sample_paths:
            log(f"Saving sample: {orig_path}")

            base_name = os.path.basename(orig_path).replace(".nii.gz", "").replace(".nii", "")
            sample_dir = os.path.join(sample_root, base_name, "stage3_skull_strip")
            os.makedirs(sample_dir, exist_ok=True)

            img_bet = ants.image_read(save_path)
            save_slices_basic(img_bet, os.path.join(sample_dir, "0_hdbet_skull_stripped.png"), "HD-BET Skull Stripped")
            
            # Đọc file mask do HD-BET tạo ra
            mask_path = save_path.replace(".nii.gz", "_mask.nii.gz") if save_path.endswith(".nii.gz") else save_path.replace(".nii", "_mask.nii.gz")
            if os.path.exists(mask_path):
                img_mask = ants.image_read(mask_path)
                save_slices_basic(img_mask, os.path.join(sample_dir, "1_brain_mask.png"), "HD-BET Brain Mask")

    except Exception as e:
        log(f"FAILED: {orig_path} | Lỗi: {e}")

log("HOÀN THÀNH STAGE 3!")