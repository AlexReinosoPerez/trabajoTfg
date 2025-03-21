import streamlit as st
from fastai.vision.all import load_learner, PILImage
import json

# 📁 Cargar las clases
with open("clases.json", "r") as f:
    class_names = json.load(f)

# 🧠 Cargar el modelo entrenado en FastAI
learn = load_learner("best_model_fastai.pkl")

# 🎨 Título de la app
st.title("🎨 Clasificador de Estilos Artísticos")

# 📤 Subida de imagen
uploaded_file = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen
    st.image(uploaded_file, caption="Imagen subida", use_column_width=True)

    # Convertir a formato PIL
    img = PILImage.create(uploaded_file)

    # Predicción
    pred_class, pred_idx, probs = learn.predict(img)

    # Mostrar resultados
    st.markdown("### 🧠 Predicción:")
    st.write(f"**{pred_class}**")
    st.markdown("### 🔢 Probabilidades:")
    for i, p in enumerate(probs):
        st.write(f"{class_names[i]}: {p:.4f}")
