# this is optimes as fuck beacouse my app was crashing so..
from flask import Flask, render_template, Response
import cv2
import torch
from ultralytics import YOLO
from threading import Thread

app = Flask(__name__)

# Load a smaller YOLOv8 model for faster inference
model = YOLO('yolov8n.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Enable half precision for faster GPU processing
if device == 'cuda':
    model.half()

# Initialize variables
camera = cv2.VideoCapture(0)
frame_queue = []

def capture_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        # Resize for faster processing
        frame = cv2.resize(frame, (320, 240))

        frame_queue.append(frame)
        if len(frame_queue) > 3:  # Prevent memory overflow
            frame_queue.pop(0)

# Start the frame capture thread
Thread(target=capture_frames, daemon=True).start()

def generate_frames():
    while True:
        if frame_queue:
            frame = frame_queue.pop(0)

            # Perform detection with optimized parameters
            results = model(frame, conf=0.5)
            annotated_frame = results[0].plot()

            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting the app on:", device)
    app.run(debug=True)
