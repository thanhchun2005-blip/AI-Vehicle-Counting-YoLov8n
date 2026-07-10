from ultralytics import YOLO

# Sửa thành đường dẫn tương đối (vì file best.pt nằm cùng thư mục với export.py)
model_path = 'best.pt' 

# (Hoặc nếu muốn chắc ăn 100%, dùng đường dẫn tuyệt đối của Windows có chữ 'r' đằng trước)
# model_path = r'C:\Học Tập\CS & UD AI\results final\runs\detect\smart_parking\yolov8n_augmented\weights\best.pt'

model = YOLO(model_path)

success = model.export(
    format='onnx',       
    imgsz=640,           
    simplify=True,       
    dynamic=False,       
    half=True           
)

print("🎉 Quá trình xuất mô hình ONNX hoàn tất!")