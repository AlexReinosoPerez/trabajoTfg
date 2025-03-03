import streamlit as st
import numpy as np
import tensorflow as tf
import requests
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo_resnet50.h5")

model = load_model()

# Definir clases de estilos artísticos
class_names = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"]

# Función para preprocesar imágenes
def preprocess_image(image):
    image = image.resize((224, 224))  # Redimensionar a 224x224 píxeles
    image = np.array(image) / 255.0   # Normalizar valores de píxeles
    image = np.expand_dims(image, axis=0)  # Expandir dimensiones para la predicción
    return image

# Función para realizar la predicción
def predict_style(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence, predictions

# Función para Grad-CAM (Explicabilidad del modelo)
def grad_cam(model, image, class_idx):
    grad_model = tf.keras.models.Model([model.input], [model.get_layer('conv5_block3_out').output, model.output])
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_output)
    heatmap = np.mean(grads, axis=-1)
    
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # Normalizar
    heatmap = cv2.resize(heatmap, (224, 224))  # Redimensionar
    
    return heatmap

# Configurar la interfaz de la aplicación en Streamlit
st.title("Clasificación de Estilos Artísticos")
st.write("Sube una imagen de una pintura y el modelo identificará su estilo artístico.")

# Opciones de carga de imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
image_url = st.text_input("O introduce una URL de imagen:")

# Cargar imagen desde dispositivo o URL
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

elif image_url:
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    st.image(image, caption="Imagen cargada desde URL", use_column_width=True)

# Clasificación de imagen
if image and st.button("Clasificar Imagen"):
    predicted_class, confidence, predictions = predict_style(image)
    
    st.write(f"### Predicción: {predicted_class}")
    st.write(f"**Confianza:** {confidence:.2%}")
    
    # Mostrar probabilidades de cada estilo artístico
    st.write("#### Probabilidades por clase:")
    for i, style in enumerate(class_names):
        st.write(f"{style}: {predictions[0][i]:.2%}")
    
    # Generar y mostrar Grad-CAM
    if st.checkbox("Mostrar Explicabilidad con Grad-CAM"):
        heatmap = grad_cam(model, preprocess_image(image), np.argmax(predictions))
        
        # Convertir a imagen
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image)
        ax.imshow(heatmap, alpha=0.5)
        ax.set_title("Mapa de activación (Grad-CAM)")
        ax.axis("off")
        st.pyplot(fig)
