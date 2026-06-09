# Brain Age Prediction from MRI scans using Federated Learning

Dự án nghiên cứu dự đoán tuổi não từ ảnh MRI sử dụng học liên hợp (Federated Learning) nhằm bảo mật dữ liệu y tế giữa các bệnh viện.

## 👥 Nhóm thực hiện (Group 05)
- Lưu Quang Linh (23020392)
- Nguyễn Thế Khôi (23020388)
- Phan Hoàng Dũng (23020346)
- Trần Khắc Phục Khánh (23020386)
- Nguyễn Đình Khải (23020384)
- Võ Duy Quang (23020414)

---

## 📂 Cấu trúc thư mục (Directory Structure)

```text
├── 📄 Brain_Age_Prediction_Project_Report.pdf  <-- Báo cáo chi tiết dự án
├── 📄 README.md                                 <-- Hướng dẫn này
├── 📄 requirements.txt                          <-- Thư viện yêu cầu
├── 📂 src/                                      <-- Mã nguồn tiền xử lý & mô hình
│   ├── 📂 data_processing/                      <-- Quy trình tiền xử lý ảnh MRI
│   │   ├── bias_field_correction.py             <-- Hiệu chỉnh nhiễu không đồng nhất (N4)
│   │   ├── mni152_registration.py               <-- Đăng ký không gian chuẩn MNI152
│   │   ├── skull_strip.py                       <-- Tách xương sọ (HD-BET)
│   │   └── final.py                             <-- Resampling (130x130x130) & chuẩn hóa
│   └── 📂 models/                               <-- Lưu trữ checkpoint của mô hình
│       ├── densenet_fedprox.pt
│       └── global_resnet18_fedprox.pth
├── 📂 notebooks/                                <-- Các file thử nghiệm chạy mô hình
│   ├── brainage-resnet18.ipynb
│   ├── brainage-densenet161.ipynb
│   ├── brainage-brainrotvit.ipynb
│   └── datasplit.ipynb
└── 📂 data/
    └── 📂 sample_data/                          <-- Chứa dữ liệu MRI thử nghiệm
```

---

## ⚙️ Quy trình xử lý & Mô hình

### 1. Tiền xử lý dữ liệu (MRI Preprocessing)
Chạy lần lượt qua 4 giai đoạn chuẩn hóa ảnh MRI thô:
1. **Bias Field Correction**: Dùng N4ITK loại bỏ nhiễu cường độ sáng không đều.
2. **MNI152 Registration**: Đăng ký Affine đưa ảnh về hệ tọa độ chuẩn.
3. **Skull Stripping**: Dùng mạng HD-BET (3D U-Net) loại bỏ hộp sọ.
4. **Resampling & Normalization**: Đưa ảnh về kích thước `130x130x130` (1mm³) và chuẩn hóa Z-score.

### 2. Các mô hình thử nghiệm (FedProx)
Nhóm triển khai học liên hợp với thuật toán **FedProx** để giải quyết dữ liệu không đồng nhất (Non-IID):
- **ResNet-18 (GroupNorm)**: Đạt kết quả ổn định nhất. (MAE: Centralized `1.83` | Federated `2.13`)
- **Tri-Planar DenseNet-161 + LoRA + Meta**: Sử dụng PEFT giảm 99% tham số huấn luyện, kết hợp thông tin giới tính. (MAE: `2.81`)
- **BrainRotViT**: Hybrid Vision Transformer kết hợp CNN regressor. (MAE: `5.20`)
