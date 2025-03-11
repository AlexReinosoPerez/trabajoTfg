import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# 📌 Configuración para evitar problemas con asyncio en Streamlit Cloud
import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 📌 Clases del modelo (deben coincidir con el entrenamiento)
class_names = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"]

@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_model_quantized.pth")

    if not os.path.exists(model_path):
        github_url = "https://raw.githubusercontent.com/TU-USUARIO/TU-REPO/main/best_model_quantized.pth"
        try:
            st.write("📥 Descargando modelo desde GitHub...")
            response = requests.get(github_url, stream=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.write("✅ Modelo descargado correctamente desde GitHub.")
        except Exception as e:
            st.error(f"⚠️ No se pudo descargar desde GitHub: {e}")
            st.stop()

    # 📌 Definir la arquitectura del modelo asegurando que coincida con la original
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 4)  # Asegúrate de que el número de clases es correcto
    )

    # 📌 Intentar cargar los pesos correctamente
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        st.write("✅ Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

    return model

# 📌 Cargar el modelo
model = load_model()

# 📌 Transformaciones para preprocesar la imagen antes de la predicción
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 📌 Interfaz de Streamlit
st.title("🎨 Clasificación de Estilos Artísticos")

uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "png", "jpeg"])
image_url = st.text_input("🌐 O introduce una URL de imagen:")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

elif image_url:
    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(requests.get(image_url, stream=True).raw)
        st.image(image, caption="Imagen cargada desde URL", use_column_width=True)
    except Exception as e:
        st.error("❌ Error al cargar la imagen desde la URL.")

# 📌 Clasificar imagen cuando el usuario presiona el botón
if image and st.button("🎯 Clasificar Imagen"):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    st.write(f"### 🎨 Predicción: {predicted_class}")
