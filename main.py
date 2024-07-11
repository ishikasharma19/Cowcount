import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from tempfile import NamedTemporaryFile
import os
import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import time  
import matplotlib.pyplot as plt



# Initialize Roboflow
rf = Roboflow(api_key="LIGj4r2IoZ23uKIKprtj")
project = rf.workspace().project("cow-disease-detection")
model = project.version(1).model

# Initialize the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'best.pt')
yolo_model = YOLO(model_path)

# Threshold for detection confidence
threshold = 0.2

def predict_image(image_path, is_url=False):
    if is_url:
        result = model.predict(image_path, hosted=True).json()
    else:
        result = model.predict(image_path).json()
    return result

def detect_cows(image_path):
    frame = cv2.imread(image_path)
    results = yolo_model(frame)[0]
    cow_count = 0  # Initialize cow count
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold and results.names[int(class_id)] == "cow":  # Check if the detected object is a cow
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cow_count += 1  # Increment cow count
    return frame, cow_count



def main():
    st.set_page_config(page_title="Cow and Lumpy Disease Detection", layout="wide", page_icon="🐄")  

    st.title("Welcome to Cattle Monitoring System")
    st.markdown("---")  

    st.sidebar.title("Choose Detection Type")
    
    option = st.sidebar.selectbox("", ["Select Detection Type", "Cow Detection", "Lumpy Disease Detection", "Webcam Detection"])

    # Create placeholder for launching message
    launch_message = st.empty()

    if option == "Cow Detection":
        # Display launching message
        launch_message.write("Launching Cow Detection... Please wait.")
        time.sleep(1)  # Add delay to simulate loading
        launch_message.empty()
        run_cow_detection()
          
    elif option == "Lumpy Disease Detection":
        # Display launching message
        launch_message.write("Launching Lumpy Disease Detection... Please wait.")
        time.sleep(1)  # Add delay to simulate loading
        launch_message.empty()
        run_lumpy_disease_detection()
          
    elif option == "Webcam Detection":
        # Display launching message
        launch_message.write("Launching Webcam Detection... Please wait.")
        time.sleep(1)  # Add delay to simulate loading
        launch_message.empty()
        run_webcam_detection()
         

def run_cow_detection():
    st.title("Cow Detection with YOLO")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Save the uploaded file to a temporary location
            with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
                image = Image.open(temp_path)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Display spinner while processing
                with st.spinner("Analyzing image... Please wait."):
                    detection_result, cow_count = detect_cows(temp_path)
                
                st.image(detection_result, caption=f"Cow Detection Result (Count: {cow_count})", use_column_width=True)
                st.write(f"Number of cows detected: {cow_count}")
                temp_file.close()  # Close the file handle
                os.unlink(temp_path)
                
                # Automatically scroll down to the output
                st.markdown("""<script>window.scrollTo(0, document.body.scrollHeight);</script>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

def run_lumpy_disease_detection():
    st.title("Lumpy Disease Detection")

    # Get user choice
    choice = st.radio("Choose an option", ("Local Image", "URL"))

    if choice == "Local Image":
        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if image_file is not None:
            with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(image_file.read())
                temp_path = temp_file.name
                image = Image.open(temp_path)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                result = predict_image(temp_path)
                temp_file.close()  # Close the file handle
                os.unlink(temp_path)
    elif choice == "URL":
        image_url = st.text_input("Enter Image URL")
        if image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Image from URL", use_column_width=True)
                result = predict_image(image_url, is_url=True)
            except Exception as e:
                st.error("Error: Unable to load image from URL.")

    if 'result' in locals():
        # Extract the predicted class
        predicted_class = result['predictions'][0]['predicted_classes'][0]
        st.write("\n")
        st.write("Predicted class:", predicted_class)
        
        # Check for lumpy disease and show emergency alert
        if predicted_class == "lumpycows":
            st.error("EMERGENCY ALERT: The predicted class is lumpy disease.")
            st.error("Please take the animal for immediate medical treatment to the nearest veterinarian.")
            st.write("\nFurther suggestions:")
            st.write("- Do not delay seeking medical help as lumpy disease can spread rapidly.")
            st.write("- Isolate the affected animal to prevent spreading the disease to others.")
            st.write("- Inform your veterinarian about any other animals that may have come in contact with the infected one.")
            st.write("- Follow your veterinarian's instructions for treatment and quarantine protocols.")

            # Automatically scroll down to the output
            st.markdown("""<script>window.scrollTo(0, document.body.scrollHeight);</script>""", unsafe_allow_html=True)
            
            
def run_webcam_detection():
    st.title("Webcam Detection")

    # Create a Matplotlib figure
    fig, ax = plt.subplots()

    # Turn off axis
    ax.axis('off')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Initialize the YOLO model
    model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'best.pt')
    model = YOLO(model_path)

    # Threshold for detection confidence
    threshold = 0.5

    while True:
        # Capture frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Perform object detection
        results = model(frame)[0]

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
        plt.pause(0.001)  # Pause for a short while to update the display

        # Clear the current plot to avoid overlap
        ax.cla()


    cap.release()
    plt.close()

if __name__ == "__main__":
    main()
