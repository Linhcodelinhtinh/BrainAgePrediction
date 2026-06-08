import os
import json
import ants
import random
import matplotlib.pyplot as plt
from datetime import datetime

# ===== CONFIG =====
json_path = "mri_dataset.json"
input_root = "data"
# Đổi tên thư mục output cho đúng với tác vụ
output_root = "data_preprocessed/bias_field_correction" 
sample_root = "data_sample"

log_file = "logs/stage1.log"

os.makedirs(output_root, exist_ok=True)
os.makedirs(sample_root, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ===== LOG =====
def log(msg):
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)

# ===== LOAD =====
with open(json_path, "r") as f:
    paths = json.load(f)

# 👉 chọn sample cố định (recommend giữ lại file này nếu muốn reproducible)
sample_file = "sample_paths.json"

if not os.path.exists(sample_file):
    sample_paths = set(random.sample(paths, 3))
    json.dump(list(sample_paths), open(sample_file, "w"))
else:
    sample_paths = set(json.load(open(sample_file)))

# ĐÃ XÓA: Template MNI vì không còn Registration

# ===== VISUALIZE =====
def save_slices(img, save_path, title=""):
    if isinstance(img, ants.ANTsImage):
        img = img.numpy()

    z = img.shape[2] // 2
    y = img.shape[1] // 2
    x = img.shape[0] // 2

    # Xoay ảnh 90 độ cho đúng chiều giải phẫu (dễ nhìn hơn)
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    axes[0].imshow(np.rot90(img[:, :, z]), cmap="gray")
    axes[1].imshow(np.rot90(img[:, y, :]), cmap="gray")
    axes[2].imshow(np.rot90(img[x, :, :]), cmap="gray")

    axes[0].set_title("Axial")
    axes[1].set_title("Coronal")
    axes[2].set_title("Sagittal")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ===== LOOP =====
for idx, path in enumerate(paths):
    try:
        rel_path = os.path.relpath(path, input_root)
        save_path = os.path.join(output_root, rel_path)
        
        # Bỏ qua nếu đã chạy rồi (Resume capability)
        if os.path.exists(save_path):
            continue
            
        log(f"[{idx+1}/{len(paths)}] {path}")

        # 1. READ
        img_ants = ants.image_read(path)

        # 2. N4 BIAS FIELD CORRECTION
        mask = ants.get_mask(img_ants)
        n4 = ants.n4_bias_field_correction(img_ants, mask=mask)
        
        # 3. SAVE PREPROCESSED (Lưu thẳng kết quả N4)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ants.image_write(n4, save_path)

        # 4. SAVE SAMPLE VISUALIZATION
        if path in sample_paths:
            log(f"Saving sample: {path}")

            base_name = os.path.basename(path).replace(".nii.gz", "")
            sample_dir = os.path.join(sample_root, base_name, "stage1_bfc")
            os.makedirs(sample_dir, exist_ok=True)

            save_slices(img_ants, os.path.join(sample_dir, "raw.png"), "Raw Image")
            save_slices(n4, os.path.join(sample_dir, "n4.png"), "N4 Bias Field Corrected")

    except Exception as e:
        log(f"FAILED: {path} | {e}")