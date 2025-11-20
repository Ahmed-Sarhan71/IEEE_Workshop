import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="cats_dogs_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Cat vs Dog Classifier 🐱🐶")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((180, 180))
    st.image(image)

    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    label = "Dog" if prediction > 0.5 else "Cat"
    st.subheader(f"Prediction: **{label}**")
