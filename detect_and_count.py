import cv2
import time
import os
from ultralytics import YOLO

# 1. CẤU HÌNH THÔNG SỐ (CONFIGURATION)
CONFIG = {
    'model_path': 'best.pt',           # Đổi thành 'best.pt' hoặc 'best.onnx'
    'video_source': 'video_test.mp4',  # Đổi thành 0 nếu chạy webcam hoặc tên file video
    'target_classes': [0, 1, 2, 3],    # Các ID class cần theo dõi (0: Car, 1: Moto, 2: Bus, 3: Truck)
    'class_names': {0: 'Car', 1: 'Moto', 2: 'Bus', 3: 'Truck'},
    'conf_thres': 0.45,                # Ngưỡng tin cậy (Lọc rác)
    'iou_thres': 0.5,                  # Ngưỡng chống đè khung (NMS)
    'tracker_type': 'bytetrack.yaml'   # Thuật toán Tracking (ByteTrack)
}

# 2. HÀM HỖ TRỢ VẼ ĐỒ HOẠ (UI HELPER)
def draw_text_with_bg(img, text, position, font_scale=0.6, thickness=2, bg_color=(0,0,0), text_color=(255,255,255)):
    """Vẽ chữ có viền nền mờ phía sau để không bị lẫn vào xe"""
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x, y - text_h - 5), (x + text_w, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

# 3. KHỞI TẠO HỆ THỐNG
print(f"🚀 Đang khởi động AI với mô hình: {CONFIG['model_path']}")
model = YOLO(CONFIG['model_path'], task='detect')
cap = cv2.VideoCapture(CONFIG['video_source'])
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ================== OUTPUT VIDEO ==================
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 30
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/result_videodemo.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, video_fps, (W, H))

# CẤU HÌNH VÙNG ROI VÀ VẠCH ĐẾM ẢO
Y_LINE = int(H * 0.7)
LINE_START, LINE_END = (0, Y_LINE), (W, Y_LINE)
Y_IGNORE = int(H * 0.3)

prev_time = 0
frame_count = 0
total_processing_time = 0

track_history = {}              
crossed_ids = set()             
total_count = 0                 
class_counts = {name: 0 for name in CONFIG['class_names'].values()}

# 4. VÒNG LẶP XỬ LÝ CHÍNH
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Tính FPS
    curr_time = time.time()
    processing_time = curr_time - prev_time if prev_time > 0 else 0
    fps = int(1 / processing_time) if processing_time > 0 else 0
    prev_time = curr_time

    # Tính FPS trung bình để in ra terminal
    if processing_time > 0:
        frame_count += 1
        total_processing_time += processing_time

    # Tracking
    results = model.track(
        frame, 
        persist=True, 
        classes=CONFIG['target_classes'], 
        conf=CONFIG['conf_thres'], 
        iou=CONFIG['iou_thres'], 
        tracker=CONFIG['tracker_type'], 
        verbose=False
    )
    boxes = results[0].boxes
    line_color = (0, 0, 255) # Vạch mặc định màu đỏ

    # Vẽ ranh giới phân cách vùng bỏ qua
    cv2.line(frame, (0, Y_IGNORE), (W, Y_IGNORE), (100, 100, 100), 1, cv2.LINE_AA)

    # Xử lý nếu có xe xuất hiện
    if boxes is not None and boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        xyxy_coords = boxes.xyxy.int().cpu().tolist()
        
        for track_id, class_id, (x1, y1, x2, y2) in zip(track_ids, class_ids, xyxy_coords):
            cx = int((x1 + x2) / 2)
            cy = y2 

            # --- ÁP DỤNG KỸ THUẬT ROI ---
            # Nếu điểm tiếp xúc của xe vẫn nằm trong vùng 30% phía trên, bỏ qua không xử lý
            if cy < Y_IGNORE:
                continue            
            
            class_name = CONFIG['class_names'].get(class_id, 'Unknown')

            # Logic Đếm xe cắt vạch
            if track_id in track_history:
                prev_y = track_history[track_id]
                if (prev_y < Y_LINE <= cy) or (prev_y > Y_LINE >= cy):
                    if track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        total_count += 1
                        class_counts[class_name] += 1
                        line_color = (0, 255, 0) # Vạch nháy Xanh khi có xe qua
            
            track_history[track_id] = cy

            # Vẽ UI trực quan
            color = (int(track_id * 50 % 255), int(track_id * 100 % 255), int(track_id * 150 % 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            
            label = f"#{track_id} {class_name}"
            draw_text_with_bg(frame, label, (x1, y1 - 5), font_scale=0.5, bg_color=color)

    # Vẽ vạch đếm chính
    cv2.line(frame, LINE_START, LINE_END, line_color, 3)

    # 5. BẢNG HUD (THỐNG KÊ GÓC TRÁI)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame) # Độ mờ 60%

    texts = [f"FPS: {fps}", f"Tong Xe: {total_count}"]
    for name in CONFIG['class_names'].values():
        texts.append(f" - {name}: {class_counts[name]}")
    
    y_offset = 35
    for text in texts:
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

    # ================== OUTPUT VIDEO ==================
    out.write(frame)
    # ==================================================

    cv2.imshow("Smart Traffic Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
print(f"Video output được lưu tại: {output_path}")
print(f"FPS trung binh: {avg_fps:.2f}")
