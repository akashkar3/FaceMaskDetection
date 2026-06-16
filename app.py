import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Set Streamlit page configuration
st.set_page_config(page_title="Face Mask Detection", layout="centered")

# Load the trained Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('face_mask_detector_model.h5')
    return model

model = load_model()

def predict_mask_status(image_data):
    # Preprocessing the image (consistent with training data preparation)
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((128, 128))  # Resize to model's input size
    image = image.convert("RGB")     # Ensure image is in RGB format
    image_np = np.array(image)
    image_scaled = image_np / 255.0  # Normalize pixel values
    image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3]) # Reshape for model input

    # Make prediction
    prediction = model.predict(image_reshaped)
    predicted_label = np.argmax(prediction)

    # Get confidence score for the predicted class
    confidence = prediction[0][predicted_label] * 100

    if predicted_label == 1: # Class 1 = Mask
        return "Mask Detected ✅", confidence
    else: # Class 0 = No Mask
        return "No Mask Detected ❌", confidence

# Streamlit app interface
st.title("😷 Face Mask Detector")
st.markdown("Upload an image to detect if the person is wearing a face mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    with st.spinner('Classifying...'):
        # Get raw image data for prediction
        uploaded_file.seek(0) # Reset file pointer to the beginning
        image_data = uploaded_file.read()

        # Make prediction and display result
        result_text, confidence_score = predict_mask_status(image_data)

        st.subheader("Prediction:")
        st.write(f"**{result_text}**")
        st.write(f"Confidence: **{confidence_score:.2f}%**")

st.markdown("""
--- 
*This app uses a Convolutional Neural Network (CNN) model trained on a dataset of masked and unmasked faces.*
""")
