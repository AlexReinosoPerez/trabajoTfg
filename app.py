import streamlit as st
from fastai.vision.all import *
import requests
import os
import json

# 📁 Ruta de archivos
MODEL_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/best_model_fastai.pkl"
CLASSES_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/clases.json"

MODEL_PATH = "best_model_fastai.pkl"
CLASSES_PATH = "clases.json"

# 🔽 Descargar modelo si no está
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Descargando modelo..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.success("✅ Modelo descargado")

# 🔽 Descargar clases si no están
if not os.path.exists(CLASSES_PATH):
    with st.spinner("📥 Descargando clases..."):
        r = requests.get(CLASSES_URL)
        with open(CLASSES_PATH, "w", encoding="utf-8") as f:
            f.write(r.text)
        st.success("✅ Clases descargadas")

# 🧠 Cargar modelo
learn = load_learner(MODEL_PATH)

# 🎨 Título
st.title("🎨 Clasificador de Estilos Artísticos")

# 📤 Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = PILImage.create(uploaded_file)
    st.image(img, caption="🖼 Imagen cargada", use_column_width=True)

    # 🔍 Predicción
    pred_class, pred_idx, probs = learn.predict(img)

    # 📊 Mostrar resultado
    st.markdown(f"### 🎯 Predicción: `{pred_class}`")
    st.markdown(f"📈 Confianza: `{probs[pred_idx]:.2%}`")
