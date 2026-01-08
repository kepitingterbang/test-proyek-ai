import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image


# ===============================
# Load Model
# ===============================
def load_model():
    return MobileNetV2(weights="imagenet")


# ===============================
# Image Preprocessing
# ===============================
def preprocess_image(image: Image.Image):
    # 1. Paksa gambar ke RGB (PENTING)
    image = image.convert("RGB")

    # 2. PIL Image → NumPy array
    img = np.array(image)

    # 3. Resize ke 224x224
    img = cv2.resize(img, (224, 224))

    # 4. Tambah batch dimension → (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    # 5. Preprocess khusus MobileNetV2
    img = preprocess_input(img)

    return img


# ===============================
# Classification
# ===============================
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)

        # Debug (opsional)
        # st.write("Input shape:", processed_image.shape)

        predictions = model.predict(processed_image)
        decoded = decode_predictions(predictions, top=3)[0]

        return decoded

    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None


# ===============================
# Streamlit App
# ===============================
def main():
    st.set_page_config(
        page_title="Image Classifier",
        page_icon="🖼️",
        layout="centered"
    )

    st.title("Image Classifier using MobileNetV2")
    st.write("Upload an image to classify it using MobileNetV2 (ImageNet).")

    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_column_width=True
        )

        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Top Predictions:")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score * 100:.2f}%")


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    main()

# test
