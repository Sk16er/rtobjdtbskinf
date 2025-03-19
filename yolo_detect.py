from ultralytics import YOLO
import cv2

# Load YOLOv8 model (Use yolov8x for highest accuracy)
model = YOLO("yolov8x.pt")

def detect_objects(frame):
    results = model(frame, conf=0.98)  # Run inference with 98% confidence
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
    
    return frame
