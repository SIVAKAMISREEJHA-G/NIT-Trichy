from ultralytics import YOLO
import os
from flask import Flask, request, render_template

model_path = "models/yolov8_pest_model.pt"
model = YOLO(model_path)

def detect_pests(image_path):
    print(f"Running pest detection on: {image_path}")

    # Run YOLOv8 inference
    results = model(image_path)

    # Extract detected pests
    detected_pests = {}
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])  # Get confidence score as float
            label = model.names[class_id]  # Get class label

            if label in detected_pests:
                detected_pests[label].append(f"{confidence:.2%}")
            else:
                detected_pests[label] = [f"{confidence:.2%}"]

    print(f"Pest detection results: {detected_pests if detected_pests else 'No pests detected'}")

    return detected_pests if detected_pests else None  # Return None if no pests detected
