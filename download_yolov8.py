# Run this script to download YOLOv8n ONNX model
# Usage: python download_yolov8.py

from ultralytics import YOLO

# Download and export YOLOv8n to ONNX
print("Downloading YOLOv8n and exporting to ONNX...")
model = YOLO("yolov8n.pt")
# Export with opset 11 for maximum browser compatibility
model.export(format="onnx", imgsz=640, simplify=False, opset=11)
print("Done! File created: yolov8n.onnx")
print("The model is ready to use!")

