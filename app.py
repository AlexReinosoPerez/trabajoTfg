import torch
import torchvision.models as models
import requests
import os

# URL del archivo en Hugging Face (cambia esto por la URL correcta)
MODEL_URL = "https://huggingface.co/modelo_pytorch.pth/resolve/main/modelo_pytorch.pth"
MODEL_PATH = "modelo_pytorch.pth"

# Descargar el modelo si no existe localmente
if not os.path.exists(MODEL_PATH):
    print(f"Descargando modelo desde {MODEL_URL}...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Modelo descargado correctamente.")

# Cargar el modelo
def load_model():
    modelo = models.resnet34(pretrained=False)  # Usa la arquitectura original del modelo
    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    modelo.eval()
    return modelo

modelo = load_model()


# Transformaciones de preprocesamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Clasificación de Obras de Arte")
st.write("Sube una imagen para clasificarla según su estilo artístico")

uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    imagen = Image.open(uploaded_file)
    st.image(imagen, caption="Imagen subida", use_column_width=True)
    
    # Preprocesar la imagen
    imagen = transform(imagen).unsqueeze(0)  # Añadir dimensión de batch
    
    # Hacer la predicción
    with torch.no_grad():
        salida = modelo(imagen)
    
    clase_predicha = torch.argmax(salida, dim=1).item()
    st.write(f"Predicción: Clase {clase_predicha}")
