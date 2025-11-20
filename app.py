import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("cats_dogs_model.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

st.title("Cat vs Dog Classifier 🐱🐶")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((180, 180))
    st.image(image)

    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = session.run([output_name], {input_name: img})[0][0]

    label = "Dog" if pred > 0.5 else "Cat"
    st.subheader(f"Prediction: **{label}**")
