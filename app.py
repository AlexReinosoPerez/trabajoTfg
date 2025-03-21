import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# 🎯 Clases del modelo
CLASS_NAMES = ['Impresionismo', 'Pop Art', 'Post-Impresionismo', 'Renacimiento']

# 🧠 Cargar modelo
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# 🖼️ Preprocesamiento de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 🎨 Título de la app
st.title("🎨 Clasificador de Estilos Artísticos")
st.write("Sube una imagen de una obra de arte y el modelo te dirá su estilo.")

# 📤 Subir imagen
uploaded_file = st.file_uploader("📁 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Imagen subida", use_column_width=True)

    # 🔍 Inferencia
    img_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_prob, pred_class = torch.max(probabilities, dim=0)

    st.markdown("## 🧠 Predicción")
    st.write(f"🎯 **Estilo:** {CLASS_NAMES[pred_class.item()]}")
    st.write(f"📊 **Confianza:** {top_prob.item()*100:.2f}%")

    # 🔢 Mostrar todas las probabilidades
    st.markdown("### 📈 Distribución de predicciones")
    for idx, prob in enumerate(probabilities):
        st.write(f"{CLASS_NAMES[idx]}: {prob.item()*100:.2f}%")
