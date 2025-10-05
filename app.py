import streamlit as st
import os
import zipfile
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

# 🔓 Unzip LoRA weights if not already extracted
if not os.path.exists("instance_images_output"):
    with zipfile.ZipFile("kolam_lora.zip", 'r') as zip_ref:
        zip_ref.extractall("instance_images_output")

# 🚀 Load pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.load_lora_weights("instance_images_output/pytorch_lora_weights.safetensors")
pipe.to(device)

# 🎨 UI
st.title("🌀 Kolam Diffusion Generator")
st.markdown("Generate beautiful Kolam art using AI. Customize your prompt, choose a style, and download your creations.")

prompt = st.text_input("Prompt", "a Kolam art pattern")
style = st.selectbox("Kolam Style", ["Symmetrical", "Floral", "Geometric", "Minimal"])
num_images = st.slider("Number of Kolams", 1, 6, 3)
show_grid = st.checkbox("Show Grid Preview")
generate = st.button("Generate Kolams")

if generate:
    os.makedirs("kolam_temp", exist_ok=True)
    full_prompt = f"{prompt}, {style} Kolam art pattern"
    images = []
    paths = []

    with st.spinner("Generating Kolams..."):
        for i in range(num_images):
            image = pipe(full_prompt, guidance_scale=7.5).images[0]
            path = f"kolam_temp/kolam_{i}.png"
            image.save(path)
            images.append(image)
            paths.append(path)

    if show_grid:
        tensor_images = [T.ToTensor()(img) for img in images]
        grid = make_grid(tensor_images, nrow=3)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(grid.permute(1, 2, 0))
        ax.axis("off")
        ax.set_title("Kolam Grid Preview")
        st.pyplot(fig)

    st.markdown("### Download Kolams")
    for path in paths:
        with open(path, "rb") as file:
            st.download_button(label=f"Download {os.path.basename(path)}", data=file, file_name=os.path.basename(path))
