import streamlit as st
from PIL import Image
import os
import random

st.set_page_config(page_title="GAN Style Transfer", layout="wide")

st.title("🎨 Artistic Style Transfer using GAN")

# ================= SIDEBAR =================
st.sidebar.title("Project Info")
st.sidebar.write("Model: GAN")
st.sidebar.write("Generator: CNN")
st.sidebar.write("Loss: Content + Style")

mode = st.sidebar.radio("Select Mode", ["Dataset Images", "Upload Images"])

# ================= FUNCTION =================
def get_random_image(folder):
    files = os.listdir(folder)
    img_name = random.choice(files)
    return os.path.join(folder, img_name)

# ================= DATASET MODE =================
if mode == "Dataset Images":

    st.subheader("Random Images from Dataset")

    try:
        content_path = get_random_image("dataset/trainA")
        style_path = get_random_image("dataset/trainB")

        content_img = Image.open(content_path)
        style_img = Image.open(style_path)

        col1, col2 = st.columns(2)

        with col1:
            st.image(content_img, caption="Content Image", use_container_width=True)

        with col2:
            st.image(style_img, caption="Style Image", use_container_width=True)

    except:
        st.error("Dataset folder not found")

# ================= UPLOAD MODE =================
else:

    content_file = st.file_uploader("Upload Content Image")
    style_file = st.file_uploader("Upload Style Image")

    if content_file and style_file:

        content_img = Image.open(content_file)
        style_img = Image.open(style_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(content_img, caption="Content Image", use_container_width=True)

        with col2:
            st.image(style_img, caption="Style Image", use_container_width=True)

    else:
        st.warning("Upload both images")

# ================= OUTPUT =================
st.subheader("Generated Image")

try:
    output_img = Image.open("result.png")
    st.image(output_img, use_container_width=True)
except:
    st.error("Run generate.py first")

# ================= DOWNLOAD =================
try:
    with open("result.png", "rb") as file:
        st.download_button("⬇ Download Output", file, "output.png")
except:
    pass
st.subheader("Loss Graph")

try:
    loss_img = Image.open("loss.png")
    st.image(loss_img, caption="Optimization Loss", use_container_width=True)
except:
    st.info("Run generate.py to see loss graph")