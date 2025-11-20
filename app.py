import streamlit as st
import numpy as np
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.nn import TFLiteModel

# Load model
model = TFLiteModel("cats_dogs_model.tflite")

st.title("Cat vs Dog Classifier 🐱🐶")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((180, 180))
    st.image(image)

    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run model
    output = model(Tensor(img)).numpy()[0]

    label = "Dog" if output > 0.5 else "Cat"
    st.subheader(f"Prediction: **{label}**")
