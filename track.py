from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official detection model

# Track with the model
# results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True) 
results = model.track(source="traffic.mp4", show=True, tracker="bytetrack.yaml") 
print(results)