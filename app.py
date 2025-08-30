import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Set up the page
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the model will predict whether it indicates **Pneumonia** or **Normal** lungs.")

# Load your trained model
@st.cache_resource
def load_trained_model():
    return load_model("pneumonia_mobilenetv2_model.h5", compile=False)

model = load_trained_model()

# Upload image
uploaded_file = st.file_uploader("Choose a Chest X-ray Image (JPEG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "🟠 Pneumonia" if prediction > 0.5 else "🟢 Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader("Prediction:")
    st.markdown(f"**{result}**")
    st.write(f"Confidence: {confidence:.2%}")
