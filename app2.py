import streamlit as st
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import segmentation_models_pytorch as smp
import tempfile
import os
import cv2

st.set_page_config(
    page_title="Detector de Minería Aluvial - Sentinel-2",
    page_icon="🛰️",
    layout="wide"
)

# ---------------------------------------------------------------------
# PALETA Y ESTILOS (solo interfaz)
# ---------------------------------------------------------------------
BG = "#10222A"        # agua profunda del río
PANEL = "#18313B"     # paneles
LINEA = "#2A4753"     # bordes
TEXTO = "#E9F0EC"
SUAVE = "#93AEAC"
ORO = "#F0B429"      # oro aluvial: color de la detección
SELVA = "#5FB38A"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Instrument Sans', sans-serif;
}}
.stApp {{
    background: radial-gradient(1200px 600px at 10% -10%, #1B3D47 0%, {BG} 55%);
    color: {TEXTO};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1250px; }}

h1, h2, h3, .titulo {{
    font-family: 'Bricolage Grotesque', sans-serif !important;
    color: {TEXTO};
    letter-spacing: -0.01em;
}}

/* Encabezado */
.hero {{
    display: flex; gap: 1.4rem; align-items: center;
    padding: 1.6rem 1.8rem;
    border: 1px solid {LINEA};
    border-radius: 18px;
    background: linear-gradient(120deg, {PANEL} 0%, #14292F 100%);
    margin-bottom: 1.4rem;
}}
.hero .icono {{
    font-size: 2.6rem; width: 74px; height: 74px; flex: none;
    display: grid; place-items: center;
    border-radius: 16px; background: {BG}; border: 1px solid {LINEA};
}}
.hero h1 {{
    margin: 0; font-size: 2.1rem; font-weight: 800; line-height: 1.1;
}}
.hero p {{ margin: .45rem 0 0; color: {SUAVE}; font-size: 1.02rem; max-width: 62ch; }}
.hero b {{ color: {ORO}; font-weight: 600; }}

/* Tarjetas de métricas */
.metrica {{
    background: {PANEL}; border: 1px solid {LINEA};
    border-left: 4px solid {ORO};
    border-radius: 14px; padding: 1.1rem 1.3rem;
}}
.metrica.verde {{ border-left-color: {SELVA}; }}
.metrica .et {{ color: {SUAVE}; font-size: .92rem; }}
.metrica .val {{
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 2.1rem; font-weight: 700; color: {TEXTO}; line-height: 1.2;
}}
.metrica .uni {{ font-size: 1rem; color: {SUAVE}; font-weight: 500; margin-left: .3rem; }}
.barra {{ height: 8px; border-radius: 8px; background: {BG}; margin-top: .6rem; overflow: hidden; }}
.barra > div {{ height: 100%; background: linear-gradient(90deg, {SELVA}, {ORO}); border-radius: 8px; }}

/* Tarjetas de imagen */
.panel-titulo {{
    font-family: 'Bricolage Grotesque', sans-serif;
    font-size: 1.12rem; font-weight: 700; color: {TEXTO};
    margin: 0 0 .1rem;
}}
.panel-sub {{ color: {SUAVE}; font-size: .88rem; margin-bottom: .6rem; }}

/* Leyenda */
.leyenda {{ display: flex; gap: 1.4rem; flex-wrap: wrap; color: {SUAVE}; font-size: .92rem; margin-top: .4rem; }}
.leyenda span i {{
    display: inline-block; width: 12px; height: 12px; border-radius: 3px;
    margin-right: .45rem; vertical-align: -1px;
}}

/* Estado vacío */
.vacio {{
    border: 1.5px dashed {LINEA}; border-radius: 16px;
    padding: 2.2rem; text-align: center; color: {SUAVE}; margin-top: 1rem;
    background: rgba(24,49,59,.45);
}}
.vacio .titulo {{ font-size: 1.25rem; font-weight: 700; color: {TEXTO}; margin-bottom: .3rem; }}
.pasos {{ display: flex; justify-content: center; gap: 2.2rem; flex-wrap: wrap; margin-top: 1.1rem; }}
.pasos div {{ max-width: 210px; font-size: .93rem; }}
.pasos strong {{ display: block; color: {ORO}; margin-bottom: .15rem; }}

/* Barra lateral */
section[data-testid="stSidebar"] {{
    background: #0C1A20; border-right: 1px solid {LINEA};
}}
section[data-testid="stSidebar"] h2 {{ font-size: 1.2rem; }}
.ficha {{
    background: {PANEL}; border: 1px solid {LINEA}; border-radius: 12px;
    padding: .9rem 1rem; font-size: .92rem; color: {SUAVE};
}}
.ficha div {{ display: flex; justify-content: space-between; gap: 1rem; padding: .22rem 0; }}
.ficha div + div {{ border-top: 1px solid {LINEA}; }}
.ficha b {{ color: {TEXTO}; font-weight: 600; text-align: right; }}

/* Cargador de archivos */
[data-testid="stFileUploader"] section {{
    background: {PANEL}; border: 1.5px dashed {ORO}; border-radius: 14px;
}}
[data-testid="stFileUploader"] button {{
    background: {ORO}; color: #10222A; border: 0; font-weight: 600; border-radius: 10px;
}}
[data-testid="stFileUploader"] button:hover {{ background: #FFD166; color: #10222A; }}

hr {{ border-color: {LINEA} !important; }}
:focus-visible {{ outline: 2px solid {ORO} !important; outline-offset: 2px; }}
</style>
""", unsafe_allow_html=True)


def tarjeta_metrica(etiqueta, valor, unidad="", verde=False, barra=None):
    clase = "metrica verde" if verde else "metrica"
    html_barra = f'<div class="barra"><div style="width:{min(max(barra, 0), 100):.1f}%"></div></div>' if barra is not None else ""
    return (
        f'<div class="{clase}"><div class="et">{etiqueta}</div>'
        f'<div class="val">{valor}<span class="uni">{unidad}</span></div>{html_barra}</div>'
    )


def figura_oscura():
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
    ax.axis("off")
    return fig, ax


def mostrar_panel(titulo, subtitulo, fig):
    st.markdown(f'<div class="panel-titulo">{titulo}</div><div class="panel-sub">{subtitulo}</div>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# 1. Cargar el modelo entrenado con caché
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
        ruta_pesos = "modelo_mineria_unet.pth"

    modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
    modelo.to(device)
    modelo.eval()
    return modelo, device


st.markdown("""
<div class="hero">
  <div class="icono">🛰️</div>
  <div>
    <h1>Detección de minería aluvial con Sentinel-2</h1>
    <p>Segmentación semántica con <b>U-Net (ResNet34)</b> para monitorear deforestación
    y pozas mineras a partir de parches satelitales.</p>
  </div>
</div>
""", unsafe_allow_html=True)

modelo, device = cargar_modelo()

# 2. Barra lateral para controles
st.sidebar.header("Parámetros de inferencia")
umbral = st.sidebar.slider("Umbral de confianza (Sigmoid)", 0.1, 0.9, 0.5, 0.05,
                           help="Más alto = detecta menos, pero con mayor certeza.")
st.sidebar.markdown(f"""
<div class="ficha">
<div><span>Satélite</span><b>Sentinel-2 (Copernicus)</b></div>
<div><span>Resolución</span><b>10 m / píxel</b></div>
<div><span>Bandas</span><b>B4 · B3 · B2</b></div>
<div><span>Procesamiento</span><b>{"GPU" if device.type == "cuda" else "CPU"}</b></div>
</div>
""", unsafe_allow_html=True)

# 3. Subir archivo satelital
archivo_subido = st.file_uploader("Selecciona un parche satelital Sentinel-2 (.tif, .tiff)", type=["tif", "tiff"])

if archivo_subido is None:
    st.markdown("""
    <div class="vacio">
      <div class="titulo">Sube un parche para empezar el análisis</div>
      Arrastra un archivo GeoTIFF sobre el cuadro de arriba.
      <div class="pasos">
        <div><strong>Carga</strong>Imagen RGB o multiespectral en .tif</div>
        <div><strong>Detección</strong>El modelo marca las zonas de minería</div>
        <div><strong>Resultado</strong>Hectáreas afectadas y mapa superpuesto</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

if archivo_subido is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
        tmp_file.write(archivo_subido.read())
        tmp_path = tmp_file.name

    try:
        with st.spinner("Analizando imagen satelital..."):
            with rasterio.open(tmp_path) as src:
                conteo = src.count
                if conteo == 3:
                    # True Color de Copernicus (RGB directo)
                    img_data = src.read([1, 2, 3]).astype(np.float32)
                elif conteo >= 4:
                    # Escena multiespectral (B4=Rojo, B3=Verde, B2=Azul)
                    img_data = src.read([4, 3, 2]).astype(np.float32)
                else:
                    banda = src.read(1).astype(np.float32)
                    img_data = np.stack([banda, banda, banda])

            # Normalizar para visualización y tensor
            rgb = np.zeros_like(img_data)
            for c in range(3):
                p2, p98 = np.percentile(img_data[c], (2, 98))
                rgb[c] = np.clip((img_data[c] - p2) / (p98 - p2 + 1e-8), 0, 1)

            # Ajuste de dimensiones para la U-Net (128x128)
            if rgb.shape[1] != 128 or rgb.shape[2] != 128:
                rgb_resized = np.zeros((3, 128, 128), dtype=np.float32)
                for c in range(3):
                    rgb_resized[c] = cv2.resize(rgb[c], (128, 128), interpolation=cv2.INTER_AREA)
                tensor_input = torch.tensor(rgb_resized, dtype=torch.float32).unsqueeze(0).to(device)
                rgb_to_display = rgb_resized.transpose(1, 2, 0)
            else:
                tensor_input = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0).to(device)
                rgb_to_display = rgb.transpose(1, 2, 0)

            # Inferencia
            with torch.no_grad():
                pred = modelo(tensor_input)
                probabilidad = torch.sigmoid(pred).squeeze().cpu().numpy()
                mascara_pred = (probabilidad > umbral).astype(np.uint8)

        # Métricas de área estimada
        # Cada píxel de 10m x 10m = 100 m² = 0.01 hectáreas
        pixeles_afectados = int(mascara_pred.sum())
        hectareas = (pixeles_afectados * 100) / 10000.0
        porcentaje = mascara_pred.mean() * 100

        col1, col2, col3 = st.columns(3)
        col1.markdown(tarjeta_metrica("Píxeles clasificados como minería", f"{pixeles_afectados:,}", "px"),
                      unsafe_allow_html=True)
        col2.markdown(tarjeta_metrica("Área estimada afectada", f"{hectareas:.2f}", "ha"),
                      unsafe_allow_html=True)
        col3.markdown(tarjeta_metrica("Porcentaje del parche", f"{porcentaje:.1f}", "%", verde=True, barra=porcentaje),
                      unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Mostrar resultados gráficos
        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            fig1, ax1 = figura_oscura()
            ax1.imshow(rgb_to_display)
            mostrar_panel("Imagen Sentinel-2", "Color real", fig1)

        with c2:
            fig2, ax2 = figura_oscura()
            ax2.imshow(mascara_pred, cmap=ListedColormap([BG, ORO]), vmin=0, vmax=1)
            mostrar_panel("Máscara de segmentación", "Zonas detectadas en dorado", fig2)

        with c3:
            fig3, ax3 = figura_oscura()
            ax3.imshow(rgb_to_display)
            capa = np.ma.masked_where(mascara_pred == 0, mascara_pred)
            ax3.imshow(capa, cmap=ListedColormap([ORO]), alpha=0.6)
            mostrar_panel("Detección superpuesta", "Minería sobre la imagen original", fig3)

        st.markdown(f"""
        <div class="leyenda">
          <span><i style="background:{ORO}"></i>Minería detectada</span>
          <span><i style="background:{LINEA}"></i>Sin minería</span>
          <span>Umbral aplicado: <b style="color:{TEXTO}">{umbral:.2f}</b></span>
        </div>
        """, unsafe_allow_html=True)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)