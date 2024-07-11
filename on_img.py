import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Initialize the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'best.pt')
model = YOLO(model_path)

# Threshold for detection confidence
threshold = 0.2

# Load image from file
image_path = r"C:\Users\sree_\Downloads\cow-354428_1280.jpg"
# image_path = r"C:\Users\sree_\Downloads\dairy-cow-in-a-field.jpg"
# image_path = r"C:\Users\sree_\Downloads\Its-Cow-Appreciation-Day.jpg" #not success 
# image_path = r"C:\Users\sree_\Downloads\images.jpg"  #not success for multiple images
# image_path = r"C:\Users\sree_\Downloads\dairy-cow-in-a-field.jpg" #multiple image that was success. 
# image_path = r"C:\Users\sree_\Downloads\Cow-Names.jpg" #multiple image that was success. 
# image_path = r

frame = cv2.imread(image_path)

# Perform object detection
results = model(frame)[0]

# Create a Matplotlib figure
fig, ax = plt.subplots()

# Turn off axis
ax.axis('off')

# Iterate over each detected object
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        # Draw bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Display the resulting frame with detections
ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.show()

