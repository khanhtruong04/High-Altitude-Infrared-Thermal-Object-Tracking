# High-Altitude Infrared Thermal Object Tracking for UAV

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)

Hệ thống theo dõi đối tượng dành cho UAV (Máy bay không người lái), được tối ưu hóa đặc biệt cho dữ liệu video nhiệt/hồng ngoại quay từ độ cao lớn. Dự án tích hợp **YOLOv8** để phát hiện đối tượng, **Global Motion Compensation (GMC)** để xử lý chuyển động nền của camera, và **Kalman Filters** để theo dõi quỹ đạo bền vững.
![Mô tả GIF](./asset/result_video.gif)

## Tính Năng Nổi Bật

* **Bù Trừ Chuyển Động Toàn Cục (GMC):** Sử dụng Sparse Optical Flow (Lucas-Kanade) và RANSAC để ổn định việc theo dõi khi camera rung lắc hoặc di chuyển nhanh.
* **Association:** Kết hợp **Kalman Filter** (ước lượng trạng thái) và **Thuật toán Hungarian** (liên kết dữ liệu dựa trên IoU) giúp giảm thiểu mất dấu.
* **Giao Diện Tương Tác (Web UI):** Giao diện trực quan sử dụng Gradio cho phép:
    * Quét video để xem các đối tượng phát hiện được.
    * **Click-to-Select:** Click chuột để chọn chính xác đối tượng cần theo dõi (hoặc để trống để theo dõi tất cả).
* **Đánh Giá Hiệu Suất:** Hỗ trợ tính toán các chỉ số chuẩn MOT (Multi-Object Tracking) như MOTA, MOTP, IDF1 thông qua thư viện `motmetrics`.
* **Tăng Tốc GPU:** Pipeline xử lý được tối ưu hóa để chạy trên GPU (CUDA).

## Cấu Trúc Dự Án

```text
uav_tracking_project/
├── models/                  # Thư mục các file trọng số (.pt)
├── src/
│   ├── tracker/             # Các thuật toán tracking cốt lõi
│   │   ├── gmc.py           # Logic bù trừ chuyển động (GMC)
│   │   ├── kalman.py        # Bộ lọc Kalman cho từng đối tượng
│   │   └── association.py   # Thuật toán Hungarian & tính IoU
│   ├── utils/               # Các tiện ích 
│   ├── processor.py         # Vòng lặp xử lý video chính
│   └── ui.py                # Thiết lập giao diện Gradio
├── main.py                  # File chạy chính
└── requirements.txt         # Các thư viện phụ thuộc

```
## Hướng Dẫn Cài Đặt
1. **Clone Repository:**
   ```bash
   git clone https://github.com/Astrq23/high-altitude-infrared-thermal-object-tracking.git
   cd high-altitude-infrared-thermal-object-tracking
   ```
2. **Tạo Môi Trường Ảo & Cài Đặt Phụ Thuộc:**
   ```bash
    python -m venv venv
    source venv/bin/activate  
    # Trên Windows: venv\Scripts\activate
    pip install -r requirements.txt
    # Cài đặt PyTorch hỗ trợ CUDA hoặc bỏ qua bước này
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Cài đặt các thư viện còn lại
    pip install -r requirements.txt
    ```
## Hướng dẫn Chạy
1. **Tải Mô Hình YOLOv8:**
   Tải các file trọng số mô hình: https://drive.google.com/drive/folders/1mcPH0b1xjLYx5SMhum8edtx-ZidA_tbn?usp=sharing
2. **Chạy Giao Diện Người Dùng:**
   ```bash
   python main.py
   ```
3. **Sử Dụng Giao Diện:**
   * Tải video nhiệt/hồng ngoại từ UAV.
    * Chọn mô hình YOLOv8 (EfficientNetB0 hoặc EfficientNetB3).
    * Click vào đối tượng cần theo dõi hoặc để trống để theo dõi tất cả.
    * Bấm "Start Processing" để bắt đầu theo dõi.

