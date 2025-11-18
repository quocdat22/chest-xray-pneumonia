import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image as img_utils
from huggingface_hub import hf_hub_download

# Paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BASE_DIR / "images" / "x-ray-image-2b_full.jpg"
IMG_SIZE = (224, 224)

# Hugging Face model config
HF_REPO_ID = "wuxdas/pneumonia_cnn"
MODEL_FILENAME = "best_baseline_cnn.h5"
INFO_FILENAME = "model_info.json"


st.set_page_config(
    page_title="Pneumonia Detection with Grad-CAM", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .pneumonia-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    .normal-box {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
    }
    .prediction-label {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.9;
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .confidence-bar {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🫁 Pneumonia Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered Chest X-ray Analysis with Grad-CAM Visualization</p>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load trained model and metadata from Hugging Face."""
    try:
        # Download model from Hugging Face
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=str(BASE_DIR / ".cache")
        )
        model = keras.models.load_model(model_path)

        # Try to download model info
        try:
            info_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=INFO_FILENAME,
                cache_dir=str(BASE_DIR / ".cache")
            )
            with open(info_path, "r", encoding="utf-8") as f:
                model_info = json.load(f)
        except Exception:
            model_info = {}

    except Exception as e:
        st.error(f"❌ Failed to load model from Hugging Face: {str(e)}")
        st.stop()

    # Find last convolutional layer for Grad-CAM
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")

    return model, model_info, last_conv_layer_name


model, model_info, last_conv_layer_name = load_model_artifacts()


def load_image(source):
    """Load an image from path or file-like object, return PIL image and array."""
    if isinstance(source, (str, os.PathLike, Path)):
        pil_img = Image.open(source).convert("L")
    elif isinstance(source, io.BytesIO):
        pil_img = Image.open(source).convert("L")
    else:
        raise ValueError("Unsupported image source type")

    pil_img = pil_img.resize(IMG_SIZE)
    img_array = img_utils.img_to_array(pil_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return pil_img, img_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Replicate Grad-CAM implementation from notebook."""
    conv_layer_model = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(last_conv_layer_name).output,
    )

    with tf.GradientTape() as tape:
        conv_outputs = conv_layer_model(img_array)
        tape.watch(conv_outputs)

        conv_layer_index = None
        for i, layer in enumerate(model.layers):
            if layer.name == last_conv_layer_name:
                conv_layer_index = i
                break

        x = conv_outputs
        for layer in model.layers[conv_layer_index + 1 :]:
            x = layer(x)

        if pred_index is None:
            pred_index = tf.argmax(x[0])

        class_channel = x[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


def create_superimposed_image(img, heatmap, alpha=0.4):
    """Blend Grad-CAM heatmap with original image."""
    if isinstance(img, Image.Image):
        img = np.array(img)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = plt_colormap(heatmap_resized)
    heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def plt_colormap(heatmap):
    """Convert normalized heatmap to RGB using matplotlib colormap."""
    import matplotlib.pyplot as plt

    colored = plt.cm.jet(heatmap)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return colored


def predict_and_visualize(image_source):
    """Load image, predict pneumonia, generate Grad-CAM heatmap and overlay."""
    pil_img, img_array = load_image(image_source)
    raw_pred = model.predict(img_array, verbose=0)[0][0]
    predicted_class = "PNEUMONIA" if raw_pred > 0.5 else "NORMAL"
    confidence = raw_pred if raw_pred > 0.5 else (1 - raw_pred)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    overlay = create_superimposed_image(pil_img, heatmap)

    return {
        "pil_image": pil_img,
        "raw_prediction": raw_pred,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "heatmap": heatmap,
        "overlay": overlay,
    }


with st.sidebar:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Select chest X-ray image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image for pneumonia detection"
    )
    
    st.markdown("---")
    use_sample = st.checkbox("📁 Use Sample Image", value=True)
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    Results are for **reference only**. 
    
    Not a substitute for professional medical diagnosis.
    """)

    if uploaded_file:
        selected_source = io.BytesIO(uploaded_file.read())
        use_sample = False
    elif use_sample and DEFAULT_IMAGE.exists():
        selected_source = DEFAULT_IMAGE
    else:
        selected_source = None

if selected_source is None:
    st.info("👈 Please upload an image or enable the sample image option to get started.")
    st.stop()

with st.spinner("🔍 Analyzing chest X-ray..."):
    result = predict_and_visualize(selected_source)

# Prediction result with styled box
predicted_class = result["predicted_class"]
confidence = result["confidence"]
box_class = "pneumonia-box" if predicted_class == "PNEUMONIA" else "normal-box"
icon = "⚠️" if predicted_class == "PNEUMONIA" else "✅"

st.markdown(f"""
    <div class="prediction-box {box_class}">
        <div class="prediction-label">Diagnosis Result</div>
        <div class="prediction-value">{icon} {predicted_class}</div>
    </div>
""", unsafe_allow_html=True)

# Progress bar for confidence
st.markdown(f"#### 📊 Confidence Score: {confidence:.1%}")
st.progress(float(confidence))

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🖼️ Original X-ray Image")
    st.image(result["pil_image"], use_container_width=True, clamp=True)

with col2:
    st.markdown("### 🔥 Grad-CAM Heatmap")
    st.image(result["overlay"], use_container_width=True)
    st.caption("Red regions indicate areas with the highest attention from the model")

st.markdown("---")

# Additional metrics in expandable section
with st.expander("📈 Prediction Details"):
    metric_cols = st.columns(3)
    metric_cols[0].metric("Predicted Class", result["predicted_class"])
    metric_cols[1].metric("Confidence Score", f"{result['confidence']:.2%}")
    metric_cols[2].metric("Raw Probability", f"{result['raw_prediction']:.4f}")
    
    st.markdown("##### Explanation:")
    if predicted_class == "PNEUMONIA":
        st.warning(f"⚠️ The model detected signs of **pneumonia** with {confidence:.1%} confidence. The Grad-CAM visualization shows the lung regions that influenced this prediction.")
    else:
        st.success(f"✅ The model indicates **normal lungs** with {confidence:.1%} confidence. No significant abnormalities detected in the highlighted regions.")

#st.caption("Model and Grad-CAM workflow replicated from notebooks/Grad_CAM.ipynb")
