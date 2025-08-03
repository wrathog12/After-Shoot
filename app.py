import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch

# Import the model and processing function from your other file
from whit import UNET, process_image

# Use caching to load the model only once
@st.cache_resource
def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model

# --- Main App ---
st.title("AI Teeth Whitener")
st.markdown("An `After Shot` AI Assignment by **Abhishek Choudhary**")

MODEL_PATH = "unet_teeth_v1.pth"
model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload Your Tooth Picture", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    pil_image = Image.open(uploaded_file)
    original_image = np.array(pil_image)
    # Convert RGB (from PIL) to BGR (for OpenCV)
    if original_image.shape[2] == 4: # Handle RGBA
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
    else:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        
    st.image(original_image, caption="Original Uploaded Image", use_column_width=True, channels="BGR")

    st.header("Fine-Tune Whitening Effect")
    # Sliders for whitening adjustment
    saturation = st.slider("Adjust Whiteness (less yellow)", min_value=0, max_value=100, value=30, step=5)
    brightness = st.slider("Adjust Brightness", min_value=0, max_value=50, value=10, step=1)
    
    if st.button("Apply Whitening Filter"):
        with st.spinner("Processing..."):
            binary_mask, heatmap, blended, whitened = process_image(
                model=model,
                image_bgr=original_image,
                saturation_reduction=saturation,
                brightness_increase=brightness
            )

            st.header("Detection Output")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(binary_mask, caption="Binary Mask")
            with col2:
                st.image(heatmap, caption="Heatmap", channels="BGR")
            with col3:
                st.image(blended, caption="Blended Overlay", channels="BGR")
            
            st.header("Final Result")
            col_orig, col_whitened = st.columns(2)
            with col_orig:
                st.image(original_image, caption="Original Picture", channels="BGR")
            with col_whitened:
                st.image(whitened, caption="Whitened Picture", channels="BGR")