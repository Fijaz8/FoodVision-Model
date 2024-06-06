import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from img_classification import load_and_prep_image,predict_data
# Load pre-trained model
model_1 = tf.keras.models.load_model('/content/simple_model.h5')

st.title("Your Streamlit App for Machine Learning Predictions")
st.write("Upload an image and get the predicted class.")

# Define class names
class_names = ['Biryani', 'Chole-Bhature', 'Jalebi', 'Kofta', 'Naan', 'Paneer-Tikka', 'Pani-Puri', 'Pav-Bhaji', 'Vadapav', 'Dabeli', 'Dal', 'Dhokla', 'Dosa', 'Kathi', 'Pakora']

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
      # Load and preprocess the image
      data = Image.open(uploaded_file)
      st.image(data, caption='Uploaded pic.', use_column_width=True)

      st.write("")
      st.write("Classifying...")

      img=load_and_prep_image(data)
      predicted_class = predict_data(model_1, img, class_name)


      # Display results
      st.success("Prediction successful!")
      st.write(f"Predicted Class: {predicted_class}")

    except Exception as e:
       st.error(f"An error occurred during prediction: {e}")
