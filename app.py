import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Preprocess function
def preprocess(img):
    img = img.resize((180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("🐶🐱 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.success("Prediction: **Dog 🐶**")
    else:
        st.success("Prediction: **Cat 🐱**")
