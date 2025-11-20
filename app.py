import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="cats_dogs_model.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Cat vs Dog Classifier 🐱🐶")
st.write("Upload an image and the TFLite model will predict it!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize to model input size
    img = img.resize((180, 180))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Your model output might differ — adjust if needed
    class_names = ["Cat", "Dog"]
    pred_class = class_names[int(prediction[0] > 0.5)]
    confidence = float(prediction[0])

    st.write("### Prediction:", pred_class)
    st.write("### Confidence:", confidence)
