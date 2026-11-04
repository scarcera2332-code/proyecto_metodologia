import streamlit as st
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import tempfile
import os

st.set_page_config(
    page_title="Detector de Minería Aluvial - Sentinel-2",
    page_icon="🛰️",
    layout="wide"
)

# 1. Cargar el modelo entrenado con caché para máxima velocidad
@st.cache_resource
def cargar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    ruta_pesos = os.path.join("models", "modelo_mineria_unet.pth")
    if not os.path.exists(ruta_pesos):
        # Si está en la misma carpeta raíz
        ruta_pesos = "modelo_mineria_unet.pth"
    
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
    modelo.to(device)
    modelo.eval()
    return modelo, device

st.title("🛰️ Detección de Minería Aluvial mediante Imágenes Sentinel-2")
st.markdown("Prototipo de segmentación semántica con arquitectura **U-Net (ResNet34)** para el monitoreo de deforestación y pozas mineras.")

modelo, device = cargar_modelo()

# 2. Barra lateral para controles
st.sidebar.header("Parámetros de Inferencia")
umbral = st.sidebar.slider("Umbral de confianza (Sigmoid)", 0.1, 0.9, 0.5, 0.05)
st.sidebar.markdown("""
**Detalles del Sensor:**
* Satélite: Sentinel-2 (Copernicus)
* Resolución espacial: 10 m / píxel
* Bandas procesadas: B4 (Rojo), B3 (Verde), B2 (Azul)
""")

# 3. Subir archivo satelital
archivo_subido = st.file_uploader("Selecciona un parche satelital Sentinel-2 (.tif)", type=["tif", "tiff"])

if archivo_subido is not None:
    # Guardar en archivo temporal para que rasterio pueda leerlo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
        tmp_file.write(archivo_subido.read())
        tmp_path = tmp_file.name

    try:
        with rasterio.open(tmp_path) as src:
            # Si tiene al menos 4 bandas, leemos B4, B3, B2 (índices 4, 3, 2 en rasterio)
            if src.count >= 4:
                img_data = src.read([4, 3, 2]).astype(np.float32)
            else:
                img_data = src.read([1, 2, 3]).astype(np.float32)
            
        # Normalizar para visualización y tensor
        rgb = np.zeros_like(img_data)
        for c in range(3):
            p2, p98 = np.percentile(img_data[c], (2, 98))
            rgb[c] = np.clip((img_data[c] - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Preparar tensor para inferencia
        tensor_img = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0).to(device)

        # Inferencia
        with torch.no_grad():
            pred = modelo(tensor_img)
            probabilidad = torch.sigmoid(pred).squeeze().cpu().numpy()
            mascara_pred = (probabilidad > umbral).astype(np.uint8)

        # Métricas de área estimada
        # Cada píxel de 10m x 10m = 100 m² = 0.01 hectáreas
        pixeles_afectados = int(mascara_pred.sum())
        hectareas = (pixeles_afectados * 100) / 10000.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Píxeles Clasificados como Minería", f"{pixeles_afectados:,} px")
        col2.metric("Área Estimada Afectada", f"{hectareas:.2f} ha")
        col3.metric("Porcentaje del Parche", f"{(mascara_pred.mean() * 100):.1f} %")

        st.markdown("---")

        # 4. Mostrar resultados gráficos
        c1, c2, c3 = st.columns(3)
        rgb_disp = rgb.transpose(1, 2, 0)

        with c1:
            st.subheader("1. Imagen Sentinel-2 (Color Real)")
            fig1, ax1 = plt.subplots()
            ax1.imshow(rgb_disp)
            ax1.axis("off")
            st.pyplot(fig1)

        with c2:
            st.subheader("2. Máscara de Segmentación")
            fig2, ax2 = plt.subplots()
            ax2.imshow(mascara_pred, cmap="gray")
            ax2.axis("off")
            st.pyplot(fig2)

        with c3:
            st.subheader("3. Detección Superpuesta")
            fig3, ax3 = plt.subplots()
            ax3.imshow(rgb_disp)
            ax3.imshow(mascara_pred, cmap="autumn", alpha=0.55)
            ax3.axis("off")
            st.pyplot(fig3)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)