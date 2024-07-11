from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="D:\Major project files\cattle-detection-1\data.yaml", epochs=10)  # train the model 

