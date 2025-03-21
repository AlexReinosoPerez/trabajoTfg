import streamlit as st
from fastai.learner import load_learner
import os
import requests
import json

# 📁 Ruta local donde guardar el modelo
MODEL_PATH = "best_model_fastai.pkl"
MODEL_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/best_model_fastai.pkl"

# 🔁 Descargar modelo si no existe
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Descargando modelo desde Hugging Face..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    st.success("✅ Modelo descargado correctamente")

# 🔁 Descargar clases si no existen
if not os.path.exists("clases.json"):
    r = requests.get("https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/clases.json")
    with open("clases.json", "w", encoding="utf-8") as f:
        f.write(r.text)
    st.success("✅ Clases descargadas")

with open("clases.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

# 🧠 Cargar modelo FastAI
learn = load_learner(MODEL_PATH)

# 🖼️ Título
st.title("🎨 Clasificador de Estilos Artísticos")

# 📤 Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen subida", use_column_width=True)
    pred, pred_idx, probs = learn.predict(uploaded_file)
    st.markdown(f"### 🎯 Predicción: `{pred}`")
    st.markdown(f"Confianza: `{probs[pred_idx]*100:.2f}%`")
