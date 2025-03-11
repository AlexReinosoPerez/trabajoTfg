import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import os
import gdown  # Para descargar desde Google Drive si es necesario

# 📌 Configuración para evitar problemas con asyncio en Streamlit Cloud
import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 📌 Clases del modelo (deben coincidir con el entrenamiento)
class_names = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"]

# 📌 Función para descargar y cargar el modelo
@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_model_quantized.pth")

    # 📌 Descargar el modelo si no está en local
    if not os.path.exists(model_path):
        # 🔹 Opción 1: Descargar desde GitHub
        github_url = "https://raw.githubusercontent.com/AlexReinosoPerez/trabajoTfg/main/best_model_quantized.pth"

        try:
            st.write("📥 Descargando modelo desde GitHub...")
            response = requests.get(github_url, stream=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.write("✅ Modelo descargado correctamente desde GitHub.")
        except Exception as e:
            st.error(f"⚠️ No se pudo descargar desde GitHub: {e}")

        # 🔹 Opción 2: Descargar desde Google Drive (Si GitHub no permite archivos grandes)
        if not os.path.exists(model_path):  # Si la descarga desde GitHub falla
            drive_url = "https://drive.google.com/uc?id=XXXXXXXXXXXXX"  # Reemplaza con tu ID de Google Drive
            try:
                st.write("📥 Descargando modelo desde Google Drive...")
                gdown.download(drive_url, model_path, quiet=False)
                st.write("✅ Modelo descargado correctamente desde Google Drive.")
            except Exception as e:
                st.error(f"⚠️ No se pudo descargar desde Google Drive: {e}")

    # 📌 Cargar el modelo
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, len(class_names))
    )

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
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
