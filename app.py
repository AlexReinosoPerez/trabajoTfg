import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import os

# 🔗 URL directa al modelo en Hugging Face (cambiado "blob" por "resolve")
MODEL_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/best_model.pth"
MODEL_PATH = "best_model.pth"

# 🖼️ Clases del modelo
CLASS_NAMES = ['Impresionismo', 'Pop Art', 'Post-Impresionismo', 'Renacimiento']

# 🔄 Transformaciones necesarias para la imagen de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    MODEL_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/best_model.pth"
    MODEL_PATH = "best_model.pth"

    if not os.path.exists(MODEL_PATH):
        print("⬇️ Descargando modelo desde Hugging Face...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("✅ Modelo descargado correctamente.")
        else:
            raise Exception(f"❌ Error al descargar el modelo: código {response.status_code}")

    # 🧠 Cargar la arquitectura y pesos del modelo
    from torchvision import models
    import torch.nn as nn

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 4)  # Cambiar si tus clases han cambiado
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model
def predict_image(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()
        return CLASS_NAMES[pred_index], probs[0][pred_index].item(), probs[0]

# 🖥️ Interfaz Streamlit
st.set_page_config(page_title="Clasificador de Estilos Artísticos", layout="centered")
st.title("🎨 Clasificador de Estilos Artísticos")
st.markdown("Sube una imagen de un cuadro para predecir su estilo artístico.")

uploaded_file = st.file_uploader("📤 Sube tu imagen aquí", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Imagen subida", use_column_width=True)

    model = load_model()

    st.write("⏳ Clasificando...")
    label, confidence, probs = predict_image(image, model)

    st.success(f"🎯 Predicción: **{label}** con una confianza de **{confidence * 100:.2f}%**")

    # 📊 Mostrar todas las probabilidades
    st.subheader("Distribución de Probabilidades:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"**{class_name}:** {probs[i].item() * 100:.2f}%")
