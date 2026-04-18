import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from datetime import datetime
from keras.models import load_model
import time

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fruit Ripeness AI",
    page_icon="🍎",
    layout="centered"
)

# -------------------------------
# GLASSMORPHISM UI STYLE
# -------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #4ade80;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 20px;
}

/* Glass card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 20px;
    margin-top: 15px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Animated bar */
.bar-container {
    height: 18px;
    width: 100%;
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #22c55e, #86efac);
    width: 0%;
    transition: width 1s ease-in-out;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">🍎 Smart Fruit AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload or capture an image for detection</div>', unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("keras_model.h5", compile=False)

model = load_my_model()

labels = [line.strip() for line in open("labels.txt")]

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32)
    return (img / 127.5) - 1.0

# -------------------------------
# PREDICT
# -------------------------------
def predict_image(img):
    img = preprocess(img)
    img = np.expand_dims(img, axis=0)
    pred = model(img, training=False).numpy()[0]

    class_id = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    return labels[class_id], confidence

# -------------------------------
# PARSE LABEL
# -------------------------------
def parse_label(label):
    label = label.lower()

    if "overripe" in label:
        ripeness = "Overripe"
    elif "unripe" in label:
        ripeness = "Unripe"
    elif "ripe" in label:
        ripeness = "Ripe"
    else:
        ripeness = "Unknown"

    fruit = label.replace("overripe", "").replace("unripe", "").replace("ripe", "")
    fruit = fruit.strip().title()

    return fruit, ripeness

# -------------------------------
# RECOMMENDATION
# -------------------------------
def get_recommendation(ripeness):
    return {
        "Ripe": "🍽 Ready to eat",
        "Unripe": "⏳ Wait 2–3 days",
        "Overripe": "⚠ Consume quickly or discard"
    }.get(ripeness, "❓ Unknown")

# -------------------------------
# LOGGING
# -------------------------------
def save_log(fruit, ripeness, confidence):
    df = pd.DataFrame({
        "time": [datetime.now()],
        "fruit": [fruit],
        "ripeness": [ripeness],
        "confidence": [confidence]
    })

    try:
        old = pd.read_csv("logs.csv")
        df = pd.concat([old, df], ignore_index=True)
    except:
        pass

    df.to_csv("logs.csv", index=False)

# -------------------------------
# SIDEBAR
# -------------------------------
mode = st.sidebar.radio("📌 Mode", ["📤 Upload Image", "📷 Camera", "📊 Analytics"])

# -------------------------------
# UPLOAD MODE
# -------------------------------
if mode == "📤 Upload Image":

    file = st.file_uploader("Upload Fruit Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)
        st.image(image, use_container_width=True)

        img = np.array(image.convert("RGB"))
        label, confidence = predict_image(img)
        fruit, ripeness = parse_label(label)

        # ---------------- CARD UI ----------------
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("🔍 Prediction Result")
        st.write(f"🍎 Fruit: **{fruit}**")
        st.write(f"🍃 Ripeness: **{ripeness}**")
        st.write(f"🎯 Confidence: **{confidence:.2f}%**")

        # ---------------- ANIMATED BAR ----------------
        st.subheader("📊 Confidence Meter")
        bar = st.empty()
        label_box = st.empty()

        for i in range(int(confidence)):
            time.sleep(0.005)
            bar.progress(i + 1)
            label_box.markdown(f"**{i+1}% Confidence** 🔍")

        st.success(get_recommendation(ripeness))

        st.markdown('</div>', unsafe_allow_html=True)

        save_log(fruit, ripeness, confidence)

# -------------------------------
# CAMERA MODE
# -------------------------------
elif mode == "📷 Camera":

    img_file = st.camera_input("Capture Image")

    if img_file:
        image = Image.open(img_file)
        st.image(image, use_container_width=True)

        img = np.array(image.convert("RGB"))
        label, confidence = predict_image(img)
        fruit, ripeness = parse_label(label)

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("🔍 Prediction Result")
        st.write(f"🍎 Fruit: **{fruit}**")
        st.write(f"🍃 Ripeness: **{ripeness}**")
        st.write(f"🎯 Confidence: **{confidence:.2f}%**")

        st.subheader("📊 Confidence Meter")
        bar = st.empty()
        label_box = st.empty()

        for i in range(int(confidence)):
            time.sleep(0.005)
            bar.progress(i + 1)
            label_box.markdown(f"**{i+1}% Confidence** 🔍")

        st.success(get_recommendation(ripeness))

        st.markdown('</div>', unsafe_allow_html=True)

        save_log(fruit, ripeness, confidence)

# -------------------------------
# ANALYTICS
# -------------------------------
elif mode == "📊 Analytics":

    st.subheader("📊 Prediction History")

    try:
        df = pd.read_csv("logs.csv")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Ripeness Distribution")
        st.bar_chart(df["ripeness"].value_counts())

    except:
        st.warning("No data yet.")
