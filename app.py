import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import requests
from huggingface_hub import hf_hub_download

HF_REPO_ID = "AlexReinoso/trabajoTFM"
MODEL_FILENAME = "best_model.pth"
class_names = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"]

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features

    # EXACTAMENTE como en el entrenamiento original
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, len(class_names))
    )

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

HF_REPO_ID = "AlexReinoso/trabajoTFM"
MODEL_FILENAME = "best_model.pth"
class_names = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"]

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("🎨 Clasificación de Estilos Artísticos")

uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen subida", use_column_width=True)

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    st.write(f"### 🎨 Predicción: {class_names[preds.item()]}")
