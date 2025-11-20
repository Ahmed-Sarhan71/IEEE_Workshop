import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🐱🐶 Cat vs Dog Classifier (TFLite)")
st.write("Upload an image to classify")

# --------- Load the TFLite model ---------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="cats_dogs_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------- Preprocess ---------
def preprocess_image(image):
    image = image.resize((180, 180))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------- Predict ---------
def predict(img_array):
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return float(output[0][0])

# --------- UI ---------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = predict(img_array)

    if prediction > 0.5:
        st.success("Prediction: **Dog 🐶**")
    else:
        st.success("Prediction: **Cat 🐱**")
