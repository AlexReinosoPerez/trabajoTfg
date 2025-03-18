import os
import requests
import torch
import streamlit as st
from torchvision import models, transforms
from PIL import Image

MODEL_URL = "https://huggingface.co/modelo_pytorch.pth/resolve/main/modelo_pytorch.pth"
MODEL_PATH = "modelo_pytorch.pth"

def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1000:
        os.remove(MODEL_PATH)  # Eliminar el archivo si es sospechosamente pequeño
        st.write("Archivo corrupto eliminado, descargando de nuevo...")

    if not os.path.exists(MODEL_PATH):
        st.write("Descargando el modelo desde Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.write("Modelo descargado exitosamente.")

@st.cache_resource()
def load_model():
    download_model()

    if os.path.getsize(MODEL_PATH) < 1000:  # Evitar cargar archivos vacíos
        raise RuntimeError("El archivo del modelo sigue estando corrupto o vacío.")

    modelo = models.resnet34(pretrained=False)
    modelo.fc = torch.nn.Linear(512, 10)  
    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True))
    modelo.eval()
    return modelo

modelo = load_model()
