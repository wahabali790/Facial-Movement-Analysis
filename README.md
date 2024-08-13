# Facial Movement Analysis

## Description
This Python code utilizes the MediaPipe model to analyze facial movements in real-time. When executed, the program captures the face via the camera. If the face is properly positioned in front of the camera, it will display a message indicating that the face position is correct. If the face is not properly aligned, it will prompt the user to adjust their face. If the face is not detected at all in front of the camera, it will display a message instructing the user to position their face in front of the camera.

## Installation
Ensure you have the following dependencies installed:
- python 3.9.13
```bash
pip install mediapipe==0.10.9 opencv-python==4.9.0.80 Flask==3.0.2
```
## Usage
Execute the following command to run the program:
```bash
python face_detection.py
```
## Flask Api
```bash
python Face_api.py
```
