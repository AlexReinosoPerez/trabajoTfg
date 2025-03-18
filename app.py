import os
import requests
import torch
import streamlit as st
from torchvision import models, transforms
from PIL import Image

# URL corregida para descargar el modelo
MODEL_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/modelo_pytorch.pth"
MODEL_PATH = "modelo_pytorch.pth"

def download_model():
    """ Descarga el modelo si no existe localmente o está corrupto. """
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1000:
        os.remove(MODEL_PATH)  # Eliminar archivo corrupto
        st.write("Archivo corrupto eliminado, descargando de nuevo...")

    if not os.path.exists(MODEL_PATH):
        st.write("Descargando el modelo desde Hugging Face...")

        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()  # Lanza un error si la descarga falla

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.getsize(MODEL_PATH) < 1000:
                raise RuntimeError("El archivo descargado parece estar corrupto o vacío.")

            st.write("Modelo descargado exitosamente.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error al descargar el modelo: {e}")
            raise RuntimeError("No se pudo descargar el modelo.")

@st.cache_resource()
def load_model():
    """ Carga el modelo en memoria después de descargarlo. """
    download_model()

    if os.path.getsize(MODEL_PATH) < 1000:
        raise RuntimeError("El archivo del modelo sigue estando corrupto o vacío.")

    modelo = models.resnet34(pretrained=False)
    modelo.fc = torch.nn.Linear(512, 10)  # Ajustar número de clases

    modelo.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    )

    modelo.eval()
    return modelo

modelo = load_model()
