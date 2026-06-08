import os
import json
import ants
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
json_path = "mri_dataset.json"
original_raw_root = "data"     

input_root = "data_preprocessed/bias_field_correction" 
output_root = "data_preprocessed/mni152_registration"
sample_root = "data_sample"

log_file = "logs/stage2.log"
template_path = "mni_template/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii"

os.makedirs(output_root, exist_ok=True)
os.makedirs(sample_root, exist_ok=True)
os.makedirs("logs", exist_ok=True)

if not os.path.exists(template_path):
    raise FileNotFoundError(f"Không tìm thấy template tại: {template_path}")

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

def save_overlay_slices(background, overlay, save_path, title="Overlay QC"):
    bg = background.numpy() if isinstance(background, ants.ANTsImage) else background
    ov = overlay.numpy() if isinstance(overlay, ants.ANTsImage) else overlay
    
    z, y, x = bg.shape[2]//2, bg.shape[1]//2, bg.shape[0]//2

    fig, axes = plt.subplots(1, 3, figsize=(12,4), facecolor='black')
    
    axes[0].imshow(np.rot90(bg[:, :, z]), cmap="gray")
    axes[0].imshow(np.rot90(ov[:, :, z]), cmap="hot", alpha=0.4) 
    
    axes[1].imshow(np.rot90(bg[:, y, :]), cmap="gray")
    axes[1].imshow(np.rot90(ov[:, y, :]), cmap="hot", alpha=0.4)
    
    axes[2].imshow(np.rot90(bg[x, :, :]), cmap="gray")
    axes[2].imshow(np.rot90(ov[x, :, :]), cmap="hot", alpha=0.4)

    for ax in axes: ax.axis('off') 
    fig.suptitle(title, color='white')
    plt.tight_layout()
    plt.savefig(save_path, facecolor='black', bbox_inches='tight')
    plt.close()

log("Loading MNI Template...")
template_img = ants.image_read(template_path)


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

        n4_img = ants.image_read(input_path)

        reg_result = ants.registration(
            fixed=template_img, 
            moving=n4_img, 
            type_of_transform='Affine'
        )
        
        registered_img = ants.apply_transforms(
            fixed=template_img,
            moving=n4_img,
            transformlist=reg_result['fwdtransforms'],
            interpolator='bSpline' 
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ants.image_write(registered_img, save_path)

        if orig_path in sample_paths:
            log(f"Saving sample: {orig_path}")

            base_name = os.path.basename(orig_path).replace(".nii.gz", "").replace(".nii", "")
            sample_dir = os.path.join(sample_root, base_name, "stage2_mni_registration")
            os.makedirs(sample_dir, exist_ok=True)

            save_slices_basic(template_img, os.path.join(sample_dir, "0_mni_template.png"), "MNI 152 Template")
            save_slices_basic(registered_img, os.path.join(sample_dir, "1_registered_mni.png"), "Registered (B-Spline)")
            
            save_overlay_slices(
                background=template_img, 
                overlay=registered_img, 
                save_path=os.path.join(sample_dir, "2_QC_overlay.png"), 
                title="QC: Registered Image (Hot) over Template (Gray)"
            )

    except Exception as e:
        log(f"FAILED: {orig_path} | Lỗi: {e}")

log("HOÀN THÀNH STAGE 2!")