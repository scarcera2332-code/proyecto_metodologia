import streamlit as st
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import tempfile
import os
import cv2
import pandas as pd
import io
from datetime import datetime

st.set_page_config(
    page_title="Detector de Minería Aluvial - Sentinel-2",
    page_icon="🛰️",
    layout="wide"
)

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
    ruta_pesos = os.path.join("models", "modelo_mineria_unet_mejorado.pth")
    if not os.path.exists(ruta_pesos):
        ruta_pesos = "modelo_mineria_unet_mejorado.pth"
    
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
    modelo.to(device)
    modelo.eval()
    return modelo, device

# Inferencia por Ventanas Deslizantes (Sliding Window)
def predecir_imagen_completa(rgb_array, modelo, device, patch_size=128, batch_size=8):
    _, h, w = rgb_array.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size

    rgb_padded = np.pad(rgb_array, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    _, h_pad, w_pad = rgb_padded.shape

    prob_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    patches, coords = [], []

    for y in range(0, h_pad, patch_size):
        for x in range(0, w_pad, patch_size):
            patch = rgb_padded[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))

            if len(patches) == batch_size:
                tensor_batch = torch.tensor(np.array(patches), dtype=torch.float32).to(device)
                with torch.no_grad():
                    preds = torch.sigmoid(modelo(tensor_batch)).squeeze(1).cpu().numpy()
                for i, (py, px) in enumerate(coords):
                    prob_map[py:py+patch_size, px:px+patch_size] = preds[i]
                patches, coords = [], []

    if len(patches) > 0:
        tensor_batch = torch.tensor(np.array(patches), dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(modelo(tensor_batch)).squeeze(1).cpu().numpy()
        for i, (py, px) in enumerate(coords):
            prob_map[py:py+patch_size, px:px+patch_size] = preds[i]

    return prob_map[:h, :w]

# 2. Motor de Lógica Difusa
def trimf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-8), (c - x) / (c - b + 1e-8)))

def trapmf(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-8), 1), (d - x) / (d - c + 1e-8)))

def clasificador_difuso_severidad(probabilidad_map):
    kernel_size = 9
    densidad_local = cv2.blur(probabilidad_map, (kernel_size, kernel_size))

    # Fuzzificación
    p_baja = trapmf(probabilidad_map, -0.1, 0.0, 0.25, 0.45)
    p_media = trimf(probabilidad_map, 0.30, 0.55, 0.75)
    p_alta = trapmf(probabilidad_map, 0.60, 0.80, 1.0, 1.1)

    d_dispersa = trapmf(densidad_local, -0.1, 0.0, 0.2, 0.4)
    d_moderada = trimf(densidad_local, 0.25, 0.5, 0.7)
    d_densa = trapmf(densidad_local, 0.55, 0.75, 1.0, 1.1)

    # Reglas difusas
    peso_critico = np.maximum(np.minimum(p_alta, d_densa),
                              np.maximum(np.minimum(p_alta, d_moderada),
                                         np.minimum(p_media, d_densa)))
    peso_moderado = np.maximum(np.minimum(p_media, d_moderada),
                               np.minimum(p_alta, d_dispersa))
    peso_bajo = np.maximum(p_baja, np.minimum(p_media, d_dispersa))

    # Defuzzificación por centroide
    suma_pesos = peso_bajo + peso_moderado + peso_critico + 1e-8
    mapa_severidad = (peso_bajo * 10.0 + peso_moderado * 50.0 + peso_critico * 90.0) / suma_pesos

    return mapa_severidad, densidad_local

# 3. Generador de Reporte Técnico en PDF
def generar_reporte_pdf(nombre_archivo, res_m, umbral, ha_tot, ha_mod, ha_crit, pct, crs,
                        rgb_disp, masc_pred, mapa_sev, dens_map):
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150) # Formato A4
    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.08, right=0.92, hspace=0.35, wspace=0.2)

    # Título del Reporte
    fig.suptitle("REPORTE TÉCNICO DE MONITOREO AMBIENTAL SATELITAL\nDetección de Minería Aluvial mediante U-Net y Lógica Difusa", 
                 fontsize=13, fontweight='bold', y=0.97)

    # Tabla resumen de texto en la parte superior
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_text = (
        f"Fecha de Análisis: {fecha_actual}  |  Archivo: {nombre_archivo}\n"
        f"Resolución Espacial: {res_m:.2f} m/px  |  Sistema de Referencia (CRS): {crs}\n"
        f"Umbral CNN: {umbral:.2f}  |  Área Total Deforestada: {ha_tot:.2f} ha ({pct:.2f}% del parche)\n"
        f"Clasificación Difusa: [Alerta Moderada: {ha_mod:.2f} ha]  |  [Minería Crítica: {ha_crit:.2f} ha]"
    )
    plt.figtext(0.5, 0.88, info_text, ha="center", fontsize=8.5, 
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f0f2f6", "edgecolor": "#cccccc"})

    # Cuadrícula 2x2 para las 4 imágenes
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(rgb_disp)
    ax1.set_title("1. Imagen Sentinel-2 (Color Real)", fontsize=9, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(rgb_disp)
    ax2.imshow(masc_pred, cmap="autumn", alpha=0.5)
    ax2.set_title("2. Detección CNN (Superpuesta)", fontsize=9, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(mapa_sev, cmap="RdYlGn_r", vmin=0, vmax=100)
    ax3.set_title("3. Nivel de Severidad Difusa", fontsize=9, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(dens_map, cmap="viridis")
    ax4.set_title("4. Agrupación Espacial / Densidad", fontsize=9, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches='tight')
    plt.close(fig)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# -------------------------------------------------------------------------
# INTERFAZ PRINCIPAL
# -------------------------------------------------------------------------
st.title("🛰️ Aplicación de escritorio con redes neuronales convolucionales para la detección de áreas de deforestación por minería aluvial en imágenes satelitales Sentinel-2")
st.markdown("Sistema de teledetección con **U-Net (ResNet34)** e **Inferencia Difusa** para monitoreo ambiental con Sentinel-2.")

modelo, device = cargar_modelo()

# Barra lateral
st.sidebar.header("⚙️ Parámetros de Inferencia")
umbral = st.sidebar.slider("Umbral de confianza CNN (Sigmoid)", 0.1, 0.9, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("📐 Reglas del Sistema Difuso")
st.sidebar.markdown("""
* **Entrada 1:** Probabilidad de U-Net ($P_{CNN}$)
* **Entrada 2:** Densidad de agrupación espacial ($D_{local}$)
* **Salida de Severidad:**
  * 🟢 **Baja:** $< 35\%$
  * 🟡 **Alerta/Moderada:** $35\% - 65\%$
  * 🔴 **Crítica:** $> 65\%$
""")

# Subida del archivo principal
archivo_subido = st.file_uploader("Selecciona un parche satelital Sentinel-2 (.tif, .tiff)", type=["tif", "tiff"])

if archivo_subido is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
        tmp_file.write(archivo_subido.read())
        tmp_path = tmp_file.name

    try:
        with rasterio.open(tmp_path) as src:
            conteo = src.count
            res_x, res_y = src.res
            meta_original = src.meta.copy()
            crs_original = src.crs
            
            if conteo == 3:
                img_data = src.read([1, 2, 3]).astype(np.float32)
            elif conteo >= 4:
                img_data = src.read([4, 3, 2]).astype(np.float32)
            else:
                banda = src.read(1).astype(np.float32)
                img_data = np.stack([banda, banda, banda])

        # Normalización estiramiento 2% - 98%
        rgb = np.zeros_like(img_data)
        for c in range(3):
            p2, p98 = np.percentile(img_data[c], (2, 98))
            rgb[c] = np.clip((img_data[c] - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Inferencia CNN + Lógica Difusa
        with st.spinner("Procesando imagen satelital con U-Net y Motor Difuso..."):
            probabilidad = predecir_imagen_completa(rgb, modelo, device, patch_size=128)
            mascara_pred = (probabilidad > umbral).astype(np.uint8)
            mapa_severidad, densidad = clasificador_difuso_severidad(probabilidad)

        # Cálculo de áreas reales
        area_por_pixel_m2 = res_x * res_y if (res_x > 0 and res_y > 0 and res_x < 1000) else 100.0
        pixeles_afectados = int(mascara_pred.sum())
        hectareas_totales = (pixeles_afectados * area_por_pixel_m2) / 10000.0

        pix_moderados = int(np.sum((mapa_severidad >= 35.0) & (mapa_severidad < 65.0)))
        pix_criticos = int(np.sum(mapa_severidad >= 65.0))
        ha_moderadas = (pix_moderados * area_por_pixel_m2) / 10000.0
        ha_criticas = (pix_criticos * area_por_pixel_m2) / 10000.0

        rgb_to_display = rgb.transpose(1, 2, 0)

        # Métricas generales
        st.subheader("📊 Métricas de Detección y Severidad")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Área Total Detectada", f"{hectareas_totales:.2f} ha")
        m2.metric("Impacto Moderado (Alerta)", f"{ha_moderadas:.2f} ha")
        m3.metric("Impacto Crítico (Grave)", f"{ha_criticas:.2f} ha")
        m4.metric("% Parche Afectado", f"{(mascara_pred.mean() * 100):.1f} %")

        st.markdown("---")

        # Pestañas de la Aplicación
        tab1, tab2, tab3 = st.tabs([
            "🛰️ Resultados y Visualización", 
            "🔬 Validación Científica (Ground Truth)", 
            "💾 Exportación a SIG y Reportes"
        ])

        # TAB 1: VISUALIZACIÓN
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("1. Imagen Sentinel-2 (Color Real)")
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                ax1.imshow(rgb_to_display)
                ax1.axis("off")
                st.pyplot(fig1)

            with c2:
                st.subheader("2. Detección CNN (Superpuesta)")
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.imshow(rgb_to_display)
                ax2.imshow(mascara_pred, cmap="autumn", alpha=0.5)
                ax2.axis("off")
                st.pyplot(fig2)

            c3, c4 = st.columns(2)
            with c3:
                st.subheader("3. Mapa de Severidad Difusa")
                fig3, ax3 = plt.subplots(figsize=(6, 6))
                im_sev = ax3.imshow(mapa_severidad, cmap="RdYlGn_r", vmin=0, vmax=100)
                ax3.axis("off")
                plt.colorbar(im_sev, ax=ax3, fraction=0.046, pad=0.04, label="Nivel de Severidad (%)")
                st.pyplot(fig3)

            with c4:
                st.subheader("4. Agrupación Espacial / Densidad")
                fig4, ax4 = plt.subplots(figsize=(6, 6))
                im_dens = ax4.imshow(densidad, cmap="viridis")
                ax4.axis("off")
                plt.colorbar(im_dens, ax=ax4, fraction=0.046, pad=0.04, label="Densidad Local")
                st.pyplot(fig4)

        # TAB 2: VALIDACIÓN CIENTÍFICA (MÉTRICAS DE TESIS)
        with tab2:
            st.subheader("Validación Empírica frente a Máscara Real (Ground Truth)")
            st.info("Para evaluar el rendimiento experimental de la U-Net, sube la máscara binaria anotada manualmente.")

            archivo_gt = st.file_uploader("Subir máscara Ground Truth (.tif, .png, .jpg)", type=["tif", "tiff", "png", "jpg"])

            if archivo_gt is not None:
                gt_bytes = np.asarray(bytearray(archivo_gt.read()), dtype=np.uint8)
                gt_img = cv2.imdecode(gt_bytes, cv2.IMREAD_GRAYSCALE)

                if gt_img.shape != mascara_pred.shape:
                    gt_img = cv2.resize(gt_img, (mascara_pred.shape[1], mascara_pred.shape[0]), interpolation=cv2.INTER_NEAREST)

                gt_bin = (gt_img > 0).astype(np.uint8)

                tp = int(np.sum((mascara_pred == 1) & (gt_bin == 1)))
                fp = int(np.sum((mascara_pred == 1) & (gt_bin == 0)))
                fn = int(np.sum((mascara_pred == 0) & (gt_bin == 1)))
                tn = int(np.sum((mascara_pred == 0) & (gt_bin == 0)))

                iou = tp / (tp + fp + fn + 1e-8)
                dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)

                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                col_v1.metric("IoU (Jaccard)", f"{iou:.4f}")
                col_v2.metric("F1-Score (Dice)", f"{dice:.4f}")
                col_v3.metric("Precisión", f"{precision:.4f}")
                col_v4.metric("Sensibilidad (Recall)", f"{recall:.4f}")

                st.markdown("#### Matriz de Confusión a Nivel de Píxel")
                df_cm = pd.DataFrame(
                    [[tp, fp], [fn, tn]],
                    index=["Positivo Real", "Negativo Real"],
                    columns=["Positivo Predicho", "Negativo Predicho"]
                )
                st.table(df_cm)

        # TAB 3: EXPORTACIÓN
        with tab3:
            st.subheader("Exportar Capas y Reportes Técnicos")
            st.markdown("Descarga las capas georreferenciadas para **QGIS/ArcGIS**, los datos tabulados para **Excel**, o el **Reporte en PDF** listo para imprimir.")

            # 1. GeoTIFF con valor 255 (Blanco brillante en cualquier visor)
            meta_salida = meta_original.copy()
            meta_salida.update({
                "count": 1,
                "dtype": "uint8",
                "nodata": 0
            })

            with tempfile.NamedTemporaryFile(delete=False, suffix="_mascara.tif") as f_out:
                ruta_geotiff_salida = f_out.name

            with rasterio.open(ruta_geotiff_salida, 'w', **meta_salida) as dst:
                dst.write((mascara_pred * 255).astype(np.uint8), 1)

            with open(ruta_geotiff_salida, "rb") as f_tif:
                datos_geotiff = f_tif.read()

            if os.path.exists(ruta_geotiff_salida):
                os.remove(ruta_geotiff_salida)

            # 2. Reporte en CSV compatible al 100% con Excel en Español
            datos_reporte = {
                "Archivo": [archivo_subido.name],
                "Resolución Espacial (m)": [res_x],
                "Umbral Utilizado": [umbral],
                "Área Total Afectada (ha)": [round(hectareas_totales, 2)],
                "Impacto Moderado (ha)": [round(ha_moderadas, 2)],
                "Impacto Crítico (ha)": [round(ha_criticas, 2)],
                "Porcentaje Parche (%)": [round(mascara_pred.mean() * 100, 2)],
                "CRS": [str(crs_original)]
            }
            df_reporte = pd.DataFrame(datos_reporte)
            csv_buffer = io.StringIO()
            # sep=';' y encoding='utf-8-sig' arreglan las columnas y las tildes en Excel
            df_reporte.to_csv(csv_buffer, sep=';', index=False, encoding='utf-8-sig')

            # 3. Generar el PDF
            pdf_bytes = generar_reporte_pdf(
                archivo_subido.name, res_x, umbral, hectareas_totales,
                ha_moderadas, ha_criticas, mascara_pred.mean() * 100,
                str(crs_original), rgb_to_display, mascara_pred,
                mapa_severidad, densidad
            )

            # Botones de Descarga en 3 columnas
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                st.download_button(
                    label="📕 Descargar Reporte Completo (PDF)",
                    data=pdf_bytes,
                    file_name="reporte_mineria_aluvial.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with col_exp2:
                st.download_button(
                    label="📊 Descargar Datos para Excel (.csv)",
                    data=csv_buffer.getvalue().encode('utf-8-sig'),
                    file_name="datos_mineria_aluvial.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_exp3:
                st.download_button(
                    label="🗺️ Descargar Máscara GeoTIFF (.tif)",
                    data=datos_geotiff,
                    file_name="mascara_mineria_georreferenciada.tif",
                    mime="image/tiff",
                    use_container_width=True
                )

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)