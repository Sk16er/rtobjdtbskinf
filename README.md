# [RTOBJDTBSKINF](https://github.com/Sk16er/rtobjdtbskinf), with direct links to relevant files and resources.



# ![image](https://github.com/user-attachments/assets/289d73c6-9096-4f7c-843a-4b1951213cf9)



---

# Real-Time Object Detection with YOLOv8

This project demonstrates real-time object detection using the YOLOv8 model, implemented with Python and OpenCV. It showcases how to set up a Flask web application to stream live video with object detection overlays.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [YOLOv8 Model](#yolov8-model)
- [Acknowledgements](#acknowledgements)

## Overview

The objective of this project is to provide an easy-to-understand implementation of real-time object detection using the latest YOLOv8 model. The application captures live video from a webcam, processes each frame to detect objects, and streams the annotated video feed through a web interface.

## Features

- Real-time object detection using YOLOv8
- Web-based video streaming with detection overlays
- Automatic opening of the web browser to the streaming interface

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Sk16er/rtobjdtbskinf.git
   cd rtobjdtbskinf
   ```



2. **Create and Activate a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```



   *Note: Ensure that you have OpenCV, Flask, and the Ultralytics YOLO package installed.*

4. **Download the YOLOv8 Model Weights:**

   The project uses the YOLOv8 nano model (`yolov8n.pt`). Download it from the [Ultralytics YOLOv8 Models Page](https://docs.ultralytics.com/models/yolov8n/) and place it in the project directory.

## Usage

1. **Run the Application:**

   ```bash
   python app_auto_open.py
   ```



   This script will start the Flask web server and automatically open the default web browser to the streaming interface at `http://127.0.0.1:5000/`.

2. **Access the Video Stream:**

   Once the application is running, navigate to `http://127.0.0.1:5000/` in your web browser to view the live video stream with real-time object detection.

## Project Structure

The repository contains the following key files and directories:

- **[`app.py`](https://github.com/Sk16er/rtobjdtbskinf/blob/main/app.py)**: Main Flask application script that sets up the web server and video feed.

- **[`app_auto_open.py`](https://github.com/Sk16er/rtobjdtbskinf/blob/main/app_auto_open.py)**: Enhanced version of `app.py` that includes functionality to automatically open the web browser upon starting the application.

- **[`easy.py`](https://github.com/Sk16er/rtobjdtbskinf/blob/main/easy.py)**: Simplified script demonstrating basic usage of the YOLOv8 model for object detection.

- **[`yolo_detect.py`](https://github.com/Sk16er/rtobjdtbskinf/blob/main/yolo_detect.py)**: Script focused on object detection using YOLOv8 without the web streaming component.

- **[`templates/`](https://github.com/Sk16er/rtobjdtbskinf/tree/main/templates)**: Directory containing HTML templates for the Flask application.

  - **[`index.html`](https://github.com/Sk16er/rtobjdtbskinf/blob/main/templates/index.html)**: Main template for displaying the video stream.

- **[`test/`](https://github.com/Sk16er/rtobjdtbskinf/tree/main/test)**: Directory intended for test scripts or test data.

- **[`yolov8n.pt`](https://github.com/Sk16er/rtobjdtbskinf/blob/main/yolov8n.pt)**: YOLOv8 nano model weights file.

## YOLOv8 Model

YOLOv8 is the latest iteration in the YOLO (You Only Look Once) family of models, offering state-of-the-art performance in object detection tasks. The nano version (`yolov8n.pt`) is optimized for speed and efficiency, making it suitable for real-time applications. For more information, visit the [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8n/).

## Acknowledgements

- **Ultralytics**: For developing and maintaining the YOLO series of models.

- **Flask**: For providing a lightweight web framework to serve the video stream.

- **OpenCV**: For facilitating video capture and processing functionalities.

---
