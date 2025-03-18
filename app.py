import os
import requests
import torch
import streamlit as st
from torchvision import models, transforms
from PIL import Image

# URL del modelo en Hugging Face
MODEL_URL = "https://huggingface.co/modelo_pytorch.pth/resolve/main/modelo_pytorch.pth"
MODEL_PATH = "modelo_pytorch.pth"

# Descargar el modelo si no existe localmente
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        st.write("Descargando el modelo desde Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.write("Modelo descargado exitosamente.")

# Cargar el modelo en PyTorch
@st.cache_resource()
def load_model():
    download_model()

    # Verificar tamaño del archivo para evitar corrupción
    if os.path.getsize(MODEL_PATH) < 1000:  # Ajusta este valor según el tamaño real
        raise RuntimeError("El archivo del modelo parece estar corrupto o vacío.")

    modelo = models.resnet34(pretrained=False)  # Usa la misma arquitectura
    modelo.fc = torch.nn.Linear(512, 10)  # Ajusta el número de clases según tu modelo
    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True))
    modelo.eval()
    return modelo

# Cargar modelo
modelo = load_model()

# Transformaciones de preprocesamiento (iguales a las usadas en el entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Tamaño de entrada esperado por ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Interfaz de usuario en Streamlit
st.title("Clasificador de Imágenes")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesamiento de la imagen
    image_tensor = transform(image).unsqueeze(0)

    # Realizar predicción
    with torch.no_grad():
        output = modelo(image_tensor)
        _, predicted_class = output.max(1)

    st.write(f"Predicción: Clase {predicted_class.item()}")

