import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# 📌 ID del archivo en Google Drive
DRIVE_FILE_ID = "1py5MYqmlgAvlLXtg39b-Slsdjwim5qdX"

# 📌 Función para descargar el modelo sin `gdown`
def download_from_google_drive(drive_id, destination):
    URL = f"https://drive.google.com/uc?export=download&id={drive_id}"
    session = requests.Session()
    response = session.get(URL, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# 📌 Descargar el modelo si no está en local
model_path = "best_model.pth"
if not os.path.exists(model_path):
    st.write("📥 Descargando modelo desde Google Drive...")
    download_from_google_drive(DRIVE_FILE_ID, model_path)
    st.write("✅ Modelo descargado correctamente.")

# 📌 Cargar el modelo
state_dict = torch.load(model_path, map_location=torch.device("cpu"))

# 📌 Definir la arquitectura correcta
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 4)
)

# 📌 Intentar cargar los pesos
model.load_state_dict(state_dict, strict=False)
model.eval()

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

    predicted_class = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"][predicted.item()]
    st.write(f"### 🎨 Predicción: {predicted_class}")
