import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import requests

# 📌 Configuración
HF_REPO_ID = "AlexReinoso/trabajoTFM"
MODEL_FILENAME = "best_model.pth"
class_names = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"]

# 📥 Descargar y cargar modelo con la arquitectura EXACTA original
@st.cache_resource
def load_model():
    st.write("📥 Descargando modelo desde Hugging Face...")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    st.write("✅ Modelo descargado correctamente.")

    # Estructura EXACTA del entrenamiento
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),  # <-- Dropout exactamente igual que en el entrenamiento (0.2)
        nn.Linear(num_features, len(class_names))
    )

    # Carga estricta para verificar coincidencia exacta
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"❌ Error crítico al cargar modelo: {e}")
        return None

    model.eval()
    return model

# 🚀 Cargar modelo
model = load_model()
if model is None:
    st.stop()

# 📌 Transformación DE VALIDACIÓN (exacta a la validación del entrenamiento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📌 Interfaz Streamlit
st.title("🎨 Clasificación de Estilos Artísticos")

uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "png", "jpeg"])
image_url = st.text_input("🌐 O introduce una URL de imagen:")

# Procesamiento imagen
image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen subida", use_column_width=True)
elif image_url:
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        image = Image.open(response.raw).convert('RGB')
        st.image(image, caption="Imagen cargada desde URL", use_column_width=True)
    except Exception as e:
        st.error(f"❌ Error al cargar la imagen desde URL: {e}")

# 📌 Clasificar imagen al pulsar botón
if image and st.button("🎯 Clasificar imagen"):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

        confidence, preds = torch.max(probs, 1)
        predicted_class = class_names[preds.item()]

    st.write(f"### 🎨 Predicción: {predicted_class} ({confidence.item()*100:.2f}% confianza)")
    st.write("🔍 Probabilidades completas:", {c: f"{p*100:.2f}%" for c, p in zip(class_names, probs[0])})

