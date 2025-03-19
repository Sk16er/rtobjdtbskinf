import webbrowser
import threading
from flask import Flask, render_template, Response
import cv2
import torch
from ultralytics import YOLO
from queue import Queue
from threading import Thread

app = Flask(__name__)

# Load YOLOv8 model (using the lightweight 'yolov8n.pt' for faster detection)
model = YOLO('yolov8n.pt')

# Check for GPU or fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using device: {device}")

# Enable half precision if using GPU
if device == 'cuda':
    model.half()

# Initialize camera
camera = cv2.VideoCapture(0)

# Queue for frames
frame_queue = Queue(maxsize=5)

def capture_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        # Put frame in queue
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            # Drop frame if queue is full
            pass

def process_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Perform object detection using YOLO
            results = model(frame, conf=0.5)
            annotated_frame = results[0].plot()
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    print("Starting the app... It will automatically open in your browser.")
    # Start frame capture thread
    threading.Thread(target=capture_frames, daemon=True).start()
    # Start the browser in a separate thread
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
