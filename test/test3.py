from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
