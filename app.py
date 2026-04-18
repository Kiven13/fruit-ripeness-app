import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from datetime import datetime
from keras.models import load_model

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fruit Ripeness AI",
    page_icon="🍎",
    layout="centered"
)

# -------------------------------
# CUSTOM CSS (UI UPGRADE)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    color: #2c3e50;
}

.subtitle {
    text-align: center;
    color: #7f8c8d;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown("<div class='title'>🍎 Fruit Ripeness Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload or capture a fruit image to detect ripeness level</div>", unsafe_allow_html=True)

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

    img = cv2.GaussianBlur(img, (3, 3), 0)  # reduce noise

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img.astype(np.float32)

    # normalize properly
    img = img / 255.0

    return img

# -------------------------------
def predict_image(img):
    predictions = []

    for _ in range(3):  # run model 3 times
        processed = preprocess(img)
        processed = np.expand_dims(processed, axis=0)

        pred = model(processed, training=False).numpy()[0]
        predictions.append(pred)

    final_pred = np.mean(predictions, axis=0)

    class_id = np.argmax(final_pred)
    confidence = float(np.max(final_pred)) * 100

    return labels[class_id], confidence

# -------------------------------
def parse_label(label):
    label = label.lower().strip()

    # REMOVE numbers and symbols completely
    label = ''.join([c for c in label if c.isalpha() or c.isspace()])

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

    fruit = fruit.strip().title()

    return fruit, ripeness

# -------------------------------
def get_recommendation(fruit, ripeness):
    fruit = fruit.lower()

    recommendations = {
        "banana": {
            "Unripe": "Keep at room temperature until yellow",
            "Ripe": "Best for eating or smoothies",
            "Overripe": "Perfect for banana bread or baking"
        },
        "apple": {
            "Unripe": "Store at room temperature to ripen",
            "Ripe": "Best for fresh eating",
            "Overripe": "Use for juice or cooking"
        },
        "mango": {
            "Unripe": "Leave for 2–3 days until soft",
            "Ripe": "Sweet and ready to eat",
            "Overripe": "Use for shakes or desserts"
        }
    }

    storage_advice = {
        "Ripe": "Store in refrigerator to slow ripening",
        "Unripe": "Keep at room temperature",
        "Overripe": "Use immediately or freeze"
    }

    main_recommendation = recommendations.get(fruit, {}).get(
        ripeness,
        "No specific recommendation available"
    )

    storage = storage_advice.get(ripeness, "")

    return main_recommendation, storage

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
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Select Mode", ["Upload Image", "Camera", "Analytics"])

st.sidebar.markdown("---")
st.sidebar.info("AI model detects fruit ripeness using image classification.")

# -------------------------------
# ANALYTICS
# -------------------------------
if mode == "Analytics":
    st.header("📊 Prediction Analytics")

    try:
        df = pd.read_csv("logs.csv")
        st.dataframe(df)

        st.subheader("Ripeness Distribution")
        st.bar_chart(df["ripeness"].value_counts())

    except:
        st.warning("No data available yet.")

# -------------------------------
# UPLOAD MODE
# -------------------------------
elif mode == "Upload Image":
    st.header("📤 Upload Fruit Image")

    file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        img = np.array(image.convert("RGB"))
        label, confidence = predict_image(img)
        fruit, ripeness = parse_label(label)

        with col2:
            st.markdown("### 🧠 Prediction Result")

            if confidence < 60:
                st.error("❌ Low confidence - prediction may be unreliable")
            elif confidence < 80:
                st.warning("⚠️ Medium confidence - be careful")
            else:
                st.success("✅ High confidence prediction")

            st.metric("Fruit", fruit)
            st.metric("Ripeness", ripeness)
            st.metric("Confidence", f"{confidence:.2f}%")

            st.progress(int(confidence))

            recommendation, storage = get_recommendation(fruit, ripeness)

            st.info(f"💡 {recommendation}")
            st.warning(f"🧊 Storage: {storage}")

        save_log(fruit, ripeness, confidence)

# -------------------------------
# CAMERA MODE
# -------------------------------
elif mode == "Camera":
    st.header("📷 Capture Image")

    img_file = st.camera_input("Take a picture")

    if img_file:
        image = Image.open(img_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)

        img = np.array(image.convert("RGB"))
        label, confidence = predict_image(img)
        fruit, ripeness = parse_label(label)

        with col2:
            st.markdown("### 🧠 Prediction Result")

            st.metric("Fruit", fruit)
            st.metric("Ripeness", ripeness)
            st.metric("Confidence", f"{confidence:.2f}%")

            st.progress(int(confidence))

            recommendation, storage = get_recommendation(fruit, ripeness)

            st.info(f"💡 {recommendation}")
            st.warning(f"🧊 Storage: {storage}")

        save_log(fruit, ripeness, confidence)
