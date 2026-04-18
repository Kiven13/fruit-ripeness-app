import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from datetime import datetime
from keras.models import load_model

# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("keras_model.h5", compile=False)

model = load_my_model()

labels = [line.strip() for line in open("labels.txt")]

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fruit Ripeness Detection", layout="centered")

st.title("Smart Multi-Fruit Ripeness Detection System")
st.write("Upload an image or use your camera to detect fruit type and ripeness.")

# -------------------------------
# PREPROCESSING
# -------------------------------
def preprocess(img):
    img = cv2.resize(img, (224, 224))

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0

    return img

# -------------------------------
# TEST-TIME AUGMENTATION (TTA)
# -------------------------------
def augment_image(img):
    return [
        img,
        cv2.convertScaleAbs(img, alpha=1.1, beta=5),
        cv2.convertScaleAbs(img, alpha=0.9, beta=-5),
        cv2.GaussianBlur(img, (3, 3), 0)
    ]

# -------------------------------
# PREDICTION
# -------------------------------
def predict_image(img):
    augmented = augment_image(img)
    predictions = []

    for im in augmented:
        processed = preprocess(im)
        processed = np.expand_dims(processed, axis=0)

        pred = model(processed, training=False).numpy()[0]
        predictions.append(pred)

    final_pred = np.mean(predictions, axis=0)

    class_id = np.argmax(final_pred)
    confidence = float(np.max(final_pred)) * 100

    return labels[class_id], confidence

# -------------------------------
# PARSE LABEL (FIXED)
# -------------------------------
def parse_label(label):
    label = label.lower()

    ripeness = "Unknown"

    if "overripe" in label:
        ripeness = "Overripe"
    elif "unripe" in label:
        ripeness = "Unripe"
    elif "ripe" in label:
        ripeness = "Ripe"

    fruit = label
    for word in ["overripe", "unripe", "ripe"]:
        fruit = fruit.replace(word, "")

    parts = fruit.split(" ", 1)
    if parts[0].isdigit() and len(parts) > 1:
        fruit = parts[1]

    fruit = fruit.strip().title()

    return fruit, ripeness

# -------------------------------
# RECOMMENDATION SYSTEM
# -------------------------------
def get_recommendation(ripeness):
    if ripeness == "Ripe":
        return "Ready to eat"
    elif ripeness == "Unripe":
        return "Wait 2–3 days"
    elif ripeness == "Overripe":
        return "Consume quickly or discard"
    else:
        return "Unknown condition"

# -------------------------------
# SAVE LOGS
# -------------------------------
def save_log(fruit, ripeness, confidence):
    data = {
        "time": [datetime.now()],
        "fruit": [fruit],
        "ripeness": [ripeness],
        "confidence": [confidence]
    }

    df = pd.DataFrame(data)

    try:
        old = pd.read_csv("logs.csv")
        df = pd.concat([old, df], ignore_index=True)
    except:
        pass

    df.to_csv("logs.csv", index=False)

# -------------------------------
# SIDEBAR
# -------------------------------
mode = st.sidebar.radio("Choose Mode", ["Upload Image", "Use Camera", "Analytics"])

# -------------------------------
# ANALYTICS PAGE
# -------------------------------
if mode == "Analytics":
    st.subheader("Prediction History")

    try:
        df = pd.read_csv("logs.csv")
        st.dataframe(df)

        st.subheader("Distribution")
        st.bar_chart(df["ripeness"].value_counts())

    except:
        st.warning("No data yet. Run predictions first.")

# -------------------------------
# IMAGE MODE
# -------------------------------
elif mode == "Upload Image":
    file = st.file_uploader("Upload Fruit Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", width=300)

        img = np.array(image.convert("RGB"))

        label, confidence = predict_image(img)
        fruit, ripeness = parse_label(label)

        if confidence < 75:
            st.error("Low confidence. Try better lighting or angle.")
        else:
            st.success(f"Fruit: {fruit}")
            st.info(f"Ripeness: {ripeness}")
            st.info(f"Confidence: {confidence:.2f}%")

            st.progress(int(confidence))
            st.warning(get_recommendation(ripeness))

            save_log(fruit, ripeness, confidence)

# -------------------------------
# CAMERA MODE
# -------------------------------
elif mode == "Use Camera":
    img_file = st.camera_input("Take a picture")

    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Captured Image", width=300)

        img = np.array(image.convert("RGB"))

        label, confidence = predict_image(img)
        fruit, ripeness = parse_label(label)

        if confidence < 75:
            st.error("Low confidence. Try again with better lighting.")
        else:
            st.success(f"Fruit: {fruit}")
            st.info(f"Ripeness: {ripeness}")
            st.info(f"Confidence: {confidence:.2f}%")

            st.progress(int(confidence))
            st.warning(get_recommendation(ripeness))

            save_log(fruit, ripeness, confidence)