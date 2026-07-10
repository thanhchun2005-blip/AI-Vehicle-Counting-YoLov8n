import os
import yaml
from ultralytics import YOLO

# BƯỚC 1: TỰ ĐỘNG DÒ TÌM ĐƯỜNG DẪN THỰC TẾ TRÊN KAGGLE
base_dir = '/kaggle/input'
exact_images_path = None
for root, dirs, files in os.walk(base_dir):
    if 'images' in dirs and 'labels' in dirs: 
        exact_images_path = os.path.join(root, 'images')
        break
if exact_images_path is None:
    print("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy thư mục 'images' nào! Hãy kiểm tra lại file Zip bạn up lên.")
else:
    print(f"TÌM THẤY RỒI! Đường dẫn chuẩn xác của thư mục ảnh là:\n {exact_images_path}")
    
    # BƯỚC 2: TẠO FILE YAML VỚI ĐƯỜNG DẪN VỪA TÌM ĐƯỢC
    data_config = {
        'train': exact_images_path,
        'val': exact_images_path, 
        'nc': 4,
        'names': ['car', 'motorcycle', 'bus', 'truck']
    }
    with open('/kaggle/working/dataset_auto.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print("✅ Đã tạo file cấu hình dataset_auto.yaml thành công!")
    
    # BƯỚC 3: HUẤN LUYỆN MÔ HÌNH
    # Load mô hình YOLO
    model = YOLO('yolov8n.pt') 
    print("🚀 Khởi động quá trình Training...")
    results = model.train(
        data='/kaggle/working/dataset_auto.yaml', # Trỏ vào file auto
        epochs=30,
        imgsz=640,
        batch=32,
        workers=4,
        cache=False,
        device=0,
        project='smart_parking',
        name='yolov8n_augmented',
        # Augmentation parameters
        mosaic=1.0, scale=0.5, translate=0.1, fliplr=0.5, 
        flipud=0.0, degrees=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
    )
