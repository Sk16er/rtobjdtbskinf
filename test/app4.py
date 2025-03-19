import webbrowser
import threading
from flask import Flask, render_template, Response
import cv2
import torch
from ultralytics import YOLO
from queue import Queue
from threading import Thread

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8x.pt')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    print("Starting the app... It will automatically open in your browser.")
    # Start frame capture thread
    threading.Thread(target=capture_frames, daemon=True).start()
    # Start the browser in a separate thread
    threading.Thread(target=open_browser).start()
    app.run(debug=True)

