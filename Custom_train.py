from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')  # YOLOv8 nano model for faster training

# Train the model on your custom dataset
model.train(
    data='/Users/nishantarora/Desktop/intern proj/custom_dataset.yaml',  # Path to your dataset YAML
    epochs=100,  # Increase the number of epochs
    imgsz=640,  # Image size
    batch=2,  # Batch size
    lr0=0.001,  # Learning rate
    name='custom_yolo_advance'  # Name for your training session
)

# Validate the model
results = model.val()

# Print validation results
print(results)
