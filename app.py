import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from datetime import datetime
from keras.models import load_model
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fruit Ripeness AI",
    page_icon="🍎",
    layout="centered"
)

# -------------------------------
# GLASSMORPHISM UI
# -------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: 900;
    color: #4ade80;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 25px;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='title'>🍎 Fruit Ripeness AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload or capture fruit image for AI prediction</div>", unsafe_allow_html=True)

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

    img = img.astype(np.uint8)

    # CLAHE (lighting correction)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Normalize
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0

    return img

# -------------------------------
# PREDICTION
# -------------------------------
def predict_image(img):
    img = preprocess(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100

    return labels[class_id], confidence

# -------------------------------
# PARSE LABEL (FIXED FOR FRESH)
# -------------------------------
def parse_label(label):
    label = label.lower()

    label = ''.join(c for c in label if c.isalpha() or c.isspace())

    if "overripe" in label:
        ripeness = "Overripe"
    elif "unripe" in label:
        ripeness = "Unripe"
    elif "fresh" in label:
        ripeness = "Fresh"
    else:
        ripeness = "Unknown"

    fruit = label
    for w in ["overripe", "unripe", "fresh"]:
        fruit = fruit.replace(w, "")

    return fruit.strip().title(), ripeness

# -------------------------------
# RECOMMENDATION SYSTEM
# -------------------------------
def get_recommendation(fruit, ripeness):
    fruit = fruit.lower()

    data = {
        "banana": {
            "Unripe": "Keep at room temperature until yellow",
            "Fresh": "Best for eating or smoothies",
            "Ripe": "Best for eating or smoothies",
            "Overripe": "Perfect for baking"
        },
        "apple": {
            "Unripe": "Let it ripen at room temperature",
            "Fresh": "Best for fresh eating",
            "Ripe": "Best for fresh eating",
            "Overripe": "Use for juice or cooking"
        },
        "mango": {
            "Unripe": "Wait 2–3 days until soft",
            "Fresh": "Sweet and ready to eat",
            "Ripe": "Sweet and ready to eat",
            "Overripe": "Good for shakes"
        },
        "orange": {
            "Unripe": "Let ripen until fully orange",
            "Fresh": "Best for juice or eating",
            "Ripe": "Best for juice or eating",
            "Overripe": "Use immediately for juice"
        },
        "tomato": {
            "Unripe": "Keep until red and soft",
            "Fresh": "Best for salads",
            "Ripe": "Best for salads",
            "Overripe": "Use for sauces"
        }
    }

    storage = {
        "Fresh": "Store in fridge",
        "Ripe": "Store in fridge",
        "Unripe": "Keep at room temp",
        "Overripe": "Use immediately"
    }

    return data.get(fruit, {}).get(ripeness, "No recommendation"), storage.get(ripeness, "")

# -------------------------------
# LOGGING
# -------------------------------
def save_log(fruit, ripeness, confidence):
    df = pd.DataFrame([{
        "time": datetime.now(),
        "fruit": fruit,
        "ripeness": ripeness,
        "confidence": confidence
    }])

    file = "logs.csv"

    if os.path.exists(file):
        old = pd.read_csv(file)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(file, index=False)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Mode", ["Upload Image", "Camera", "Analytics"])

# -------------------------------
# ANALYTICS
# -------------------------------
if mode == "Analytics":
    st.header("📊 Analytics")

    if os.path.exists("logs.csv"):
        df = pd.read_csv("logs.csv")
        st.dataframe(df)

        st.subheader("Ripeness Chart")
        st.bar_chart(df["ripeness"].value_counts())
    else:
        st.warning("No data yet")

# -------------------------------
# UPLOAD MODE
# -------------------------------
elif mode == "Upload Image":
    st.header("📤 Upload Image")

    file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_container_width=True)

        img = np.array(image.convert("RGB"))
        label, confidence = predict_image(img)
        fruit, ripeness = parse_label(label)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.markdown("### 🔍 Prediction")
            st.write(f"🍎 Fruit: **{fruit}**")
            st.write(f"🍃 Ripeness: **{ripeness}**")

            color = (
                "lightgreen" if confidence > 90 else
                "orange" if confidence > 75 else
                "red"
            )

            st.markdown(
                f"**Confidence:** <span style='color:{color}'>{confidence:.2f}%</span>",
                unsafe_allow_html=True
            )

            # Animated confidence bar
            bar = st.empty()
            text = st.empty()

            for i in range(int(confidence)):
                bar.progress(i + 1)
                text.markdown(f"**Confidence: {i+1}% 🔍**")

            rec, storage = get_recommendation(fruit, ripeness)

            st.success(rec)
            st.warning(storage)

            st.markdown('</div>', unsafe_allow_html=True)

        save_log(fruit, ripeness, confidence)

# -------------------------------
# CAMERA MODE
# -------------------------------
elif mode == "Camera":
    st.header("📷 Camera")

    img_file = st.camera_input("Take picture")

    if img_file:
        image = Image.open(img_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_container_width=True)

        img = np.array(image.convert("RGB"))
        label, confidence = predict_image(img)
        fruit, ripeness = parse_label(label)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.markdown("### 🔍 Prediction")
            st.write(f"🍎 Fruit: **{fruit}**")
            st.write(f"🍃 Ripeness: **{ripeness}**")
            st.write(f"🎯 Confidence: {confidence:.2f}%")

            # Animated confidence bar
            bar = st.empty()
            text = st.empty()

            for i in range(int(confidence)):
                bar.progress(i + 1)
                text.markdown(f"**Confidence: {i+1}% 🔍**")

            rec, storage = get_recommendation(fruit, ripeness)

            st.success(rec)
            st.warning(storage)

            st.markdown('</div>', unsafe_allow_html=True)

        save_log(fruit, ripeness, confidence)
