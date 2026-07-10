# Đếm Phương Tiện Giao Thông Bằng YOLOv8 & ByteTrack
> Đồ án môn học: Cơ sở & Ứng dụng Trí tuệ Nhân tạo (CS & UD AI)

---

## 👥 Thành viên thực hiện
* **Giáo viên hướng dẫn:** PGS.TS Trương Ngọc Sơn
* **Sinh viên thực hiện (Nhóm 32):**
  1. Nguyễn Hoàng Sơn — MSSV: 23119102
  2. Đặng Thị Khánh Huyền — MSSV: 23119066
  3. Nguyễn Thành Trung — MSSV: 23119117

---

## 📌 Tổng quan dự án
Dự án tập trung vào việc phát triển hệ thống phát hiện, theo dõi (tracking) và đếm số lượng phương tiện giao thông theo thời gian thực từ video/camera giám sát. Hệ thống sử dụng:
* **YOLOv8n (Ultralytics):** Nhận diện các lớp phương tiện (`Car`, `Motorcycle`, `Bus`, `Truck`) với kích thước nhỏ gọn, tối ưu cho thiết bị Edge AI.
* **ByteTrack:** Duy trì ID đối tượng qua các frame liên tiếp để tránh đếm trùng và xử lý trường hợp bị che khuất.
* **Virtual Counting Line (Vạch đếm ảo):** Xác định thời điểm xe đi qua vạch để cập nhật thống kê.
* **Region of Interest (ROI):** Lọc bỏ các xe ở quá xa camera để giảm nhiễu phát hiện sai.
* **Quantization (Lượng tử hóa):** Đưa mô hình từ PyTorch (FP32) sang định dạng ONNX (FP16) để tăng tốc độ suy luận.

---

## 📁 Cấu trúc thư mục dự án
```
Finalllllllllllllllllllllll/
│
├── train.py               # Script huấn luyện YOLOv8n trên Kaggle (Online Augmentation)
├── export.py              # Script xuất/lượng tử hóa model sang ONNX (FP16)
├── detect_and_count.py    # Script chạy logic nhận diện, tracking và đếm xe qua vạch ảo
│
├── dataset.yaml           # File cấu hình dữ liệu thủ công
├── dataset_auto.yaml      # File cấu hình dữ liệu sinh ra tự động khi train trên Kaggle
├── yolov8n.pt             # Trọng số YOLOv8n gốc pre-trained
├── yolo26n.pt             # Trọng số YOLOv8n bản nén pre-trained
│
└── runs/                  # Thư mục chứa kết quả huấn luyện (đồ thị, confusion matrix, weights)
    └── detect/
        └── smart_parking/
            └── yolov8n_augmented/
                ├── weights/
                │   ├── best.pt    # Trọng số tốt nhất thu được sau khi train
                │   └── last.pt
                ├── confusion_matrix.png
                ├── results.png
                └── F1_curve.png
```

---

## 🛠️ Hướng dẫn cài đặt

### Yêu cầu hệ thống
* Python 3.8+
* GPU hỗ trợ CUDA (khuyên dùng để đạt hiệu năng tối ưu)

### Cài đặt thư viện phụ thuộc
Mở CMD/Terminal và chạy lệnh sau:
```bash
pip install ultralytics opencv-python onnx onnxsim pyyaml
```

---

## 🚀 Hướng dẫn chạy chương trình

### 1. Huấn luyện mô hình (`train.py`)
File `train.py` được thiết kế để chạy trên môi trường Kaggle Notebook với GPU. Script sẽ tự động tìm đường dẫn dataset, tạo file cấu hình `.yaml` và khởi động quá trình train với **Online Augmentation**:
```python
# Chạy trực tiếp trong Notebook:
python train.py
```
*Các tham số tăng cường dữ liệu được sử dụng:* Mosaic (1.0), Scale (0.5), Translate (0.1), Fliplr (0.5), HSV color adjustment.

### 2. Lượng tử hóa mô hình (`export.py`)
Để tối ưu hóa hiệu năng, giảm dung lượng và tăng tốc độ suy luận, chạy script sau để chuyển đổi trọng số từ định dạng `.pt` sang `.onnx` ở mức chính xác nửa thực (FP16):
```bash
python export.py
```
*Lưu ý:* Hãy đặt file `best.pt` trong cùng thư mục với `export.py` trước khi chạy.

### 3. Triển khai logic đếm xe (`detect_and_count.py`)
Chương trình sẽ đọc luồng video, áp dụng ByteTrack và đếm số lượng xe đi qua vạch đếm ảo:
```bash
python detect_and_count.py
```
*Trong file code, bạn có thể chỉnh sửa dict `CONFIG` để thay đổi:*
* `'model_path'`: Đường dẫn tới mô hình (`best.pt` hoặc `best.onnx`)
* `'video_source'`: File video đầu vào hoặc `0` để chạy trực tiếp từ Webcam.
* `'conf_thres'`: Ngưỡng lọc độ tin cậy của box (mặc định: `0.45`).

---

## 📊 Kết quả thực nghiệm (Kết quả từ Báo cáo)

### Đánh giá huấn luyện mô hình YOLOv8n
* **Precision:** ~0.91
* **Recall:** ~0.83
* **mAP50:** ~0.94
* **mAP50-95:** ~0.82
* **Độ chính xác theo lớp (mAP50):**
  * `Car`: 98.0%
  * `Motorcycle`: 96.7%
  * `Bus`: 92.4%
  * `Truck`: 82.4%

### So sánh hiệu năng Đếm phương tiện (FP32 vs FP16 ONNX)

| File Video | Định dạng Model | FPS | Xe máy | Ô tô | Xe buýt | Xe tải | Tổng |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **video_test** | **FP32 (.pt)** | **37.73** | **4** | **41** | **0** | **3** | **48** |
| video_test | FP16 (.onnx) | 26.67 | 1 | 42 | 0 | 3 | 46 |
| **video_demo** | **FP32 (.pt)** | **27.09** | **4** | **91** | **0** | **4** | **99** |
| video_demo | FP16 (.onnx) | 22.46 | 4 | 89 | 0 | 4 | 97 |

> 💡 **Nhận xét:** Trong môi trường thử nghiệm hiện tại, mô hình PyTorch FP32 chạy trực tiếp bằng CUDA mang lại tốc độ (FPS) và độ chính xác tối đa. Lượng tử hóa sang ONNX FP16 giúp tối ưu bộ nhớ hơn nhưng tốc độ thực tế sẽ được thể hiện rõ ràng nhất khi chạy trên các phần cứng Edge nhúng chuyên dụng (như NVIDIA Jetson Nano với TensorRT).
