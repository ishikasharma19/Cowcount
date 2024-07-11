import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from tempfile import NamedTemporaryFile
import os
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="LIGj4r2IoZ23uKIKprtj")
project = rf.workspace().project("cow-disease-detection")
model = project.version(1).model

# Function to predict from a URL or an image
def predict_image(image_path, is_url=False):
    if is_url:
        result = model.predict(image_path, hosted=True).json()
    else:
        result = model.predict(image_path).json()
    return result

st.title("Cow Disease Detection")

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
