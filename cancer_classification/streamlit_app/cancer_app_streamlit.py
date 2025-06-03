import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from model import CombinedModel
from skimage.filters import threshold_otsu
from skimage import io
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalización manual
means = {
    'age': 58,
    'biopsy': 0.1,
    'invasive': 0.0,
    'BIRADS': 0.7,
    'area_segmentada_px': 94312,
    'otsu_threshold': 77
}
stds = {
    'age': 10,
    'biopsy': 0.3,
    'invasive': 0.2,
    'BIRADS': 0.6,
    'area_segmentada_px': 52269,
    'otsu_threshold': 51
}

def normalize(value, mean, std):
    return (value - mean) / std

# Calcular otsu y área
def calcular_otsu_area(image_path):
    try:
        image = io.imread(image_path, as_gray=True)
        thresh = threshold_otsu(image)
        binary_mask = image > thresh
        area = np.sum(binary_mask)
        return thresh, area
    except Exception:
        raise ValueError(f"No se pudo cargar la imagen desde: {image_path}")

# GradCAM para heatmap
vgg_model = VGG16(weights="imagenet")
grad_model = Model(inputs=vgg_model.inputs, outputs=[vgg_model.get_layer("block5_conv3").output, vgg_model.output])

def generate_heatmap(image_path):
    image_pil = Image.open(image_path).convert("RGB")  # Abrir correctamente como imagen
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)  # Convertir correctamente
    image_resized = cv2.resize(image_cv, (224, 224))
    img_array = np.expand_dims(image_resized, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(weights * conv_outputs[0], axis=-1)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (image_cv.shape[1], image_cv.shape[0]))
    heatmap = (heatmap * 255).astype("uint8")
    return heatmap


def apply_heatmap(image_path, heatmap, alpha=0.6):
    image_pil = Image.open(image_path).convert("RGB")   
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)  
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, image_cv, 1 - alpha, 0)
    return overlay


# Modelo combinado
num_features = 6
model = CombinedModel(num_features=num_features, num_classes=2).to(device)
model.load_state_dict(torch.load("cancer_classification_model_final.pth", map_location=device))
model.eval()

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.title("Clasificación de Mamografía con Red Combinada")

uploaded_file = st.file_uploader("Carga una imagen de mamografía", type=["jpg", "png"])
age = st.number_input("Edad", min_value=0)
biopsia = st.selectbox("¿Tuvo biopsia previa?", ["Sí", "No"])
invasive = st.selectbox("¿Lesión invasiva?", ["Sí", "No"])
birads = st.slider("BIRADS", min_value=0, max_value=2)

if uploaded_file is not None:
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(temp_path, caption="Imagen cargada", use_column_width=True)

    otsu_thresh, area_seg = calcular_otsu_area(temp_path)
    st.markdown(f"**Umbral Otsu:** {otsu_thresh}")
    st.markdown(f"**Área Segmentada (px):** {area_seg}")

    heatmap = generate_heatmap(temp_path)
    heatmap_img = apply_heatmap(temp_path, heatmap)
    st.image(heatmap_img, caption="Heatmap superpuesto", channels="BGR", use_column_width=True)

    if st.button("Predecir"):
        image_tensor = transform(Image.open(temp_path).convert("RGB")).unsqueeze(0).to(device)
        tabular = np.array([
            normalize(age, means["age"], stds["age"]),
            normalize(1 if biopsia == "Sí" else 0, means["biopsy"], stds["biopsy"]),
            1 if invasive == "Sí" else 0,
            normalize(birads, means["BIRADS"], stds["BIRADS"]),
            normalize(area_seg, means["area_segmentada_px"], stds["area_segmentada_px"]),
            normalize(otsu_thresh, means["otsu_threshold"], stds["otsu_threshold"])
        ], dtype=np.float32).reshape(1, -1)
        tabular_tensor = torch.tensor(tabular, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(image_tensor, tabular_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)

        st.markdown(f"**Probabilidad de Cáncer:** {probs[1]*100:.2f}%")
        if pred_class == 1:
            st.error("La imagen tiene alta probabilidad de presentar una lesión cancerosa. Se recomienda remitir.")
        else:
            st.success("Probabilidad baja de presentar una lesión cancerosa.")