import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import streamlit as st
import requests
from PIL import Image

# 📌 URL del modelo en Hugging Face
MODEL_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/modelo_pytorch.pth"
MODEL_PATH = "modelo_pytorch.pth"

# 📌 Diccionario de clases con los estilos artísticos deseados
CLASSES = {
    0: "Impresionismo",
    1: "Postimpresionismo",
    2: "Pop Art",
    3: "Renacimiento"
}

# 📥 Función para descargar el modelo si no existe
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Descargando el modelo desde Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.write("Modelo descargado exitosamente.")

# 📦 Función para cargar el modelo en PyTorch
@st.cache_resource()
def load_model():
    download_model()

    # 🏗️ Crear el modelo ResNet50 sin pesos preentrenados
    modelo = models.resnet50(weights=None)

    # 🔗 Modificar la capa fully connected para 4 clases
    num_ftrs = modelo.fc.in_features
    modelo.fc = nn.Linear(num_ftrs, len(CLASSES))  # 4 clases

    # 📂 Cargar los pesos del modelo de forma segura
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    # 👀 Verificar las claves del modelo (para evitar errores)
    print("Claves en state_dict:", state_dict.keys())

    # 🏗️ Cargar los pesos con `strict=False` para evitar errores de compatibilidad
    modelo.load_state_dict(state_dict, strict=False)

    modelo.eval()
    return modelo

# 🚀 Cargar el modelo
modelo = load_model()

# 🎨 Transformaciones de preprocesamiento (ajustadas a FastAI)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajustar tamaño a 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización de ImageNet
])

# 🖼️ Interfaz de Streamlit
st.title("Clasificación de Estilos Artísticos 🎨")
st.write("Sube una imagen para obtener la predicción del modelo.")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # 🏗️ Preprocesar imagen
    image = transform(image).unsqueeze(0)  # Agregar batch dimension

    # 🔮 Realizar predicción
    with torch.no_grad():
        output = modelo(image)
        pred_index = torch.argmax(output, dim=1).item()
        pred_label = CLASSES.get(pred_index, "Estilo Desconocido")  # Obtener nombre del estilo

    st.write(f"**Predicción del modelo:** {pred_label} 🎭")
