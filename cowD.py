import streamlit as st
import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Initialize the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'best.pt')
model = YOLO(model_path)

# Threshold for detection confidence
threshold = 0.2

def detect_cows(image_path):
    frame = cv2.imread(image_path)
    results = model(frame)[0]
    cow_count = 0  # Initialize cow count
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold and results.names[int(class_id)] == "cow":  # Check if the detected object is a cow
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cow_count += 1  # Increment cow count
    return frame, cow_count

st.title("Cow Detection with YOLO")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Save the uploaded file to a temporary location
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.read())
        # Perform cow detection on the uploaded image
        detection_result, cow_count = detect_cows("temp_image.jpg")
        # Display the detection result
        st.image(detection_result, caption=f"Cow Detection Result (Count: {cow_count})", use_column_width=True)
        st.write(f"Number of cows detected: {cow_count}")
    except Exception as e:
        st.error(f"Error: {e}")
