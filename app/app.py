# app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.generator import UNetGenerator
import numpy as np

# Load model
@st.cache_resource
def load_generator():
    model = UNetGenerator()
    model.load_state_dict(torch.load("best_generator.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess uploaded image
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img).unsqueeze(0)

# Convert output tensor to PIL image
def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5) + 0.5  # Unnormalize
    return transforms.ToPILImage()(tensor)

# Streamlit UI
st.set_page_config(page_title="Pix2Pix Comic Generator", layout="centered")
st.title("🎨 Real to Comic Face Translator")
st.markdown("Upload a **real face image** and get its **comic version** using Pix2Pix!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    if st.button("Generate Comic Version"):
        with st.spinner("Generating comic version..."):
            model = load_generator()
            input_tensor = transform_image(image)
            with torch.no_grad():
                output_tensor = model(input_tensor)
            comic_image = tensor_to_pil(output_tensor)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with col2:
            st.image(comic_image, caption="Comic Version", use_container_width=True)

        st.success("Done!")
        buf = comic_image.copy()
        st.download_button("📥 Download Comic Image", data=buf.tobytes(), file_name="comic_image.png", mime="image/png")
