import os
import json
import ants
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

json_path = "mri_dataset.json"
original_raw_root = "data"     

input_root = "data_preprocessed/skull_stripping" 
output_root = "data_preprocessed/final_ready"
sample_root = "data_sample"

log_file = "logs/stage4.log"

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

# Lấy lại đúng các subject đã được random từ trước
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

def pad_or_crop_to_target(image_array, target_shape=(130, 130, 130)):
    x, y, z = image_array.shape
    tx, ty, tz = target_shape

    dx = max(0, tx - x)
    dy = max(0, ty - y)
    dz = max(0, tz - z)

    pad_x = (dx // 2, dx - dx // 2)
    pad_y = (dy // 2, dy - dy // 2)
    pad_z = (dz // 2, dz - dz // 2)

    if any(p > 0 for p in pad_x + pad_y + pad_z):
        image_array = np.pad(image_array, (pad_x, pad_y, pad_z), mode='constant', constant_values=0)

    nx, ny, nz = image_array.shape
    start_x = (nx - tx) // 2
    start_y = (ny - ty) // 2
    start_z = (nz - tz) // 2

    return image_array[start_x:start_x+tx, start_y:start_y+ty, start_z:start_z+tz]

log("BẮT ĐẦU STAGE 4: RESAMPLING, RESIZING & NORMALIZATION...")

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

        # 1. Đọc ảnh
        img = ants.image_read(input_path)

        # 2. Resample về 1x1x1 mm³
        img_1mm = ants.resample_image(img, (1.0, 1.0, 1.0), use_voxels=False, interp_type=1)
        
        # 3. Ép shape về 130x130x130
        img_array = img_1mm.numpy()
        resized_array = pad_or_crop_to_target(img_array, target_shape=(130, 130, 130))

        # 4. Z-Score Normalization (Chỉ trên phần não)
        brain_mask = resized_array > 0
        if np.any(brain_mask):
            mean_val = resized_array[brain_mask].mean()
            std_val = resized_array[brain_mask].std()
            resized_array[brain_mask] = (resized_array[brain_mask] - mean_val) / (std_val + 1e-8)
        
        # 5. Lưu lại dạng ANTs image
        final_img = ants.from_numpy(resized_array, spacing=(1.0, 1.0, 1.0))
        ants.image_write(final_img, save_path)

        # Visualize nếu nằm trong danh sách sample
        if orig_path in sample_paths:
            log(f"Saving sample: {orig_path}")

            base_name = os.path.basename(orig_path).replace(".nii.gz", "").replace(".nii", "")
            sample_dir = os.path.join(sample_root, base_name, "stage4_final")
            os.makedirs(sample_dir, exist_ok=True)

            save_slices_basic(resized_array, os.path.join(sample_dir, "final_ready_130x130x130.png"), "Final: 1mm³ | 130³ | Z-Scored")

    except Exception as e:
        log(f"FAILED: {orig_path} | Lỗi: {e}")

log("HOÀN THÀNH STAGE 4!")