from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    CLIMATE_FILE,
    FEATURES_B,
    MARKET_FILE,
    METADATA_FILE,
    MODEL_B_FILE,
    NINO_FILE,
    PRIMARY_COLOR,
    PROCESSED_FILE,
)
from src.data_processing import (
    consolidate_sources,
    load_processed,
    load_raw_sources,
    prediction_ready_rows,
    quality_report,
)
from src.prediction import predict_row, simple_explanation


st.set_page_config(
    page_title="Precios agrícolas de Piura",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_default_data(path: str) -> pd.DataFrame:
    return load_processed(path)


@st.cache_resource(show_spinner=False)
def load_default_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        :root {{ --institutional: {PRIMARY_COLOR}; }}
        .stApp {{ background: #FAFAFA; }}
        .block-container {{ max-width: 1240px; padding-top: 1.6rem; padding-bottom: 3rem; }}
        .stMarkdown p, .stMarkdown li {{ font-size: 1rem; line-height: 1.55; }}
        h1, h2, h3 {{ color: #263238; }}
        h1 {{ border-bottom: 4px solid var(--institutional); padding-bottom: .55rem; }}
        [data-testid="stSidebar"] {{ background: #FFF7F2; border-right: 1px solid #F1D5C3; }}
        [data-testid="stMetric"] {{
            background: white; border: 1px solid #ECECEC; border-radius: 12px;
            padding: .8rem 1rem; box-shadow: 0 2px 10px rgba(0,0,0,.035);
        }}
        [data-testid="stMetricValue"] {{ font-size: 2rem; }}
        .hero {{
            background: linear-gradient(120deg, #FFF4EC 0%, #FFFFFF 72%);
            border: 1px solid #F2D1BB; border-left: 7px solid var(--institutional);
            border-radius: 14px; padding: 1.25rem 1.4rem; margin-bottom: 1.2rem;
        }}
        .result-card {{
            border-radius: 14px; padding: 1.1rem 1.3rem; color: white;
            box-shadow: 0 3px 12px rgba(0,0,0,.10); margin: .5rem 0 1rem;
        }}
        .result-card h2 {{ color: white; margin: 0; }}
        .small-note {{ color: #5F6368; font-size: .92rem; }}
        div.stButton > button {{
            background: var(--institutional); color: white; border: none;
            border-radius: 9px; font-weight: 600;
        }}
        div.stButton > button:hover {{ background: #BF4F00; color: white; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<div class="small-note">{subtitle}</div>', unsafe_allow_html=True)


def result_card(label: str, product: str, price_type: str) -> None:
    colors = {"SUBE": "#C75000", "BAJA": "#1F6D8C", "ESTABLE": "#59636B"}
    descriptions = {
        "SUBE": "El modelo clasifica la siguiente variación disponible como aumento superior al 1 %.",
        "BAJA": "El modelo clasifica la siguiente variación disponible como disminución superior al 1 %.",
        "ESTABLE": "El modelo clasifica la siguiente variación dentro del intervalo de −1 % a +1 %.",
    }
    st.markdown(
        f"""
        <div class="result-card" style="background:{colors[label]}">
          <div style="font-size:.9rem;opacity:.9">{product} · {price_type.title()}</div>
          <h2>{label}</h2>
          <div>{descriptions[label]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def price_chart(data: pd.DataFrame, title: str):
    figure = px.line(
        data.sort_values("FECHA"),
        x="FECHA",
        y="PRECIO",
        color="TIPO_PRECIO",
        line_dash="UNIDAD_MEDIDA",
        markers=True,
        color_discrete_map={"MAYORISTA": PRIMARY_COLOR, "MINORISTA": "#1F6D8C"},
        labels={"FECHA": "Fecha", "PRECIO": "Precio registrado", "TIPO_PRECIO": "Tipo"},
        title=title,
    )
    figure.update_layout(hovermode="x unified", legend_title_text="Serie", height=430)
    return figure


def climate_chart(data: pd.DataFrame):
    climate = (
        data.sort_values("FECHA")
        .drop_duplicates("FECHA")
        [["FECHA", "LLUVIA_DIA_ANT", "TMAX_DIA_ANT", "TMIN_DIA_ANT"]]
    )
    figure = go.Figure()
    figure.add_bar(
        x=climate["FECHA"], y=climate["LLUVIA_DIA_ANT"], name="Lluvia (mm)",
        marker_color="#74A9CF", opacity=.55, yaxis="y2"
    )
    figure.add_scatter(
        x=climate["FECHA"], y=climate["TMAX_DIA_ANT"], name="Temperatura máxima",
        line={"color": PRIMARY_COLOR, "width": 2}
    )
    figure.add_scatter(
        x=climate["FECHA"], y=climate["TMIN_DIA_ANT"], name="Temperatura mínima",
        line={"color": "#2C6E91", "width": 2}
    )
    figure.update_layout(
        title="Clima conocido antes de cada registro",
        xaxis_title="Fecha del precio",
        yaxis={"title": "Temperatura (°C)"},
        yaxis2={"title": "Lluvia (mm)", "overlaying": "y", "side": "right", "showgrid": False},
        hovermode="x unified",
        height=420,
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def prediction_panel(data: pd.DataFrame, model, metadata: dict, key_prefix: str = "default") -> None:
    ready = prediction_ready_rows(data)
    if ready.empty:
        st.warning("No existen filas con historial y clima suficientes para generar una clasificación.")
        return

    products = sorted(ready["PRODUCTO"].unique())
    product = st.selectbox("Producto agrícola", products, key=f"{key_prefix}_product")
    product_rows = ready[ready["PRODUCTO"] == product]
    price_types = sorted(product_rows["TIPO_PRECIO"].unique())
    price_type = st.selectbox("Tipo de precio", price_types, key=f"{key_prefix}_type")
    type_rows = product_rows[product_rows["TIPO_PRECIO"] == price_type]
    units = sorted(type_rows["UNIDAD_MEDIDA"].unique())
    unit = st.selectbox("Unidad de medida", units, key=f"{key_prefix}_unit")
    selected = type_rows[type_rows["UNIDAD_MEDIDA"] == unit].sort_values("FECHA")
    latest = selected.tail(1)

    trained_products = set(metadata["entrenamiento_final"]["productos"])
    if product not in trained_products:
        st.error(
            "Este producto no estuvo presente durante el entrenamiento. Se muestran sus datos, "
            "pero no se genera una clasificación para evitar un resultado engañoso."
        )
        return

    label, support = predict_row(model, latest)
    row = latest.iloc[0]
    result_card(label, product, price_type)

    columns = st.columns(3)
    columns[0].metric("Último precio", f"S/ {row['PRECIO_ACTUAL']:.2f}")
    columns[1].metric("Fecha", row["FECHA"].strftime("%d/%m/%y"))
    columns[2].metric("Variación anterior", f"{row['VAR_PCT_ANT']:.1f} %")
    st.caption(f"Unidad de la serie: **{unit.title()}**")

    support_frame = pd.DataFrame(
        {"Clase": list(support.keys()), "Proporción de vecinos": list(support.values())}
    )
    support_figure = px.bar(
        support_frame,
        x="Clase",
        y="Proporción de vecinos",
        color="Clase",
        color_discrete_map={"BAJA": "#1F6D8C", "ESTABLE": "#747D83", "SUBE": PRIMARY_COLOR},
        text_auto=".1%",
        title="Distribución ponderada de los vecinos KNN",
    )
    support_figure.update_layout(showlegend=False, yaxis_tickformat=".0%", height=350)
    st.plotly_chart(support_figure, width="stretch")

    st.subheader("Explicación sencilla")
    for item in simple_explanation(row):
        st.markdown(f"- {item}")
    st.info(
        "La proporción de vecinos no es una probabilidad calibrada. La clasificación describe el "
        "siguiente registro disponible, cuyo número de días puede variar."
    )


inject_styles()

required_files = [PROCESSED_FILE, MODEL_B_FILE, METADATA_FILE]
missing_files = [str(path.relative_to(path.parents[1])) for path in required_files if not path.exists()]
if missing_files:
    st.error(
        "Faltan artefactos necesarios para iniciar la aplicación: " + ", ".join(missing_files)
        + ". Ejecute `python build_artifacts.py` desde la raíz del proyecto."
    )
    st.stop()

data = load_default_data(str(PROCESSED_FILE))
model = load_default_model(str(MODEL_B_FILE))
metadata = load_metadata(str(METADATA_FILE))

st.sidebar.markdown(f"## <span style='color:{PRIMARY_COLOR}'>Piura</span> · Precios agrícolas", unsafe_allow_html=True)
st.sidebar.caption("Prototipo académico con KNN")
page = st.sidebar.radio(
    "Navegación",
    ["Inicio", "Exploración histórica", "Tendencia estimada", "Cargar CSV", "Metodología"],
)
st.sidebar.divider()
st.sidebar.warning("La aplicación no recomienda comprar o vender. Sus resultados son experimentales.")

if page == "Inicio":
    page_header("Precios agrícolas del Mercado Modelo de Piura", "Consulta histórica y clasificación experimental con KNN")
    st.markdown(
        """
        <div class="hero">
        Esta aplicación permite explorar precios mayoristas y minoristas e identificar si el modelo
        clasifica la próxima variación disponible como <b>SUBE</b>, <b>BAJA</b> o <b>ESTABLE</b>.
        Combina historial de precios, lluvia, temperaturas e índice Niño 1+2.
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Productos", int(data["PRODUCTO"].nunique()))
    cols[1].metric("Registros consolidados", f"{len(data):,}")
    cols[2].metric("Primera fecha", data["FECHA"].min().strftime("%d/%m/%y"))
    cols[3].metric("Última fecha", data["FECHA"].max().strftime("%d/%m/%y"))

    st.subheader("¿Para qué sirve?")
    st.write(
        "Sirve como apoyo para consultar tendencias históricas y estudiar si las variables climáticas "
        "añaden información a la clasificación del siguiente movimiento de precio. No calcula la ganancia, "
        "la demanda ni qué producto conviene vender."
    )
    metrics = metadata["metricas_prueba"]["modelo_b_con_clima"]
    st.subheader("Desempeño observado en prueba")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")
    metric_cols[1].metric("F1 macro", f"{metrics['f1_macro']:.3f}")
    metric_cols[2].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.warning(metadata["advertencia"])

elif page == "Exploración histórica":
    page_header("Exploración histórica", "Compare precios mayoristas, minoristas y clima anterior a cada registro")
    product = st.selectbox("Seleccione un producto", sorted(data["PRODUCTO"].unique()))
    filtered = data[data["PRODUCTO"] == product].copy()
    minimum, maximum = filtered["FECHA"].min().date(), filtered["FECHA"].max().date()
    selected_dates = st.date_input(
        "Rango de fechas", value=(minimum, maximum), min_value=minimum, max_value=maximum
    )
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start, end = pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1])
        filtered = filtered[filtered["FECHA"].between(start, end)]
    st.plotly_chart(price_chart(filtered, f"Historial de {product}"), width="stretch")
    st.plotly_chart(climate_chart(filtered), width="stretch")
    with st.expander("Ver datos utilizados"):
        st.dataframe(
            filtered[["FECHA", "TIPO_PRECIO", "UNIDAD_MEDIDA", "PRECIO", "TMAX_DIA_ANT", "TMIN_DIA_ANT", "LLUVIA_DIA_ANT"]]
            .sort_values(["FECHA", "TIPO_PRECIO"]),
            width="stretch",
            hide_index=True,
        )

elif page == "Tendencia estimada":
    page_header("Tendencia estimada", "Clasificación del siguiente registro disponible usando el pipeline KNN evaluado")
    prediction_panel(data, model, metadata)
    st.warning(
        "La clasificación no es un precio futuro exacto ni una recomendación comercial. El desempeño "
        "de prueba fue modesto y la mejora climática observada fue pequeña."
    )

elif page == "Cargar CSV":
    page_header("Carga opcional de archivos", "Valide otros CSV sin reemplazar los datos originales del repositorio")
    st.write(
        "El archivo del mercado es obligatorio. Si no carga clima o Niño 1+2, se utilizarán los archivos "
        "incluidos en el proyecto. Solo se clasifican productos conocidos por el modelo."
    )
    market_upload = st.file_uploader("CSV del mercado", type="csv", key="market_upload")
    col1, col2 = st.columns(2)
    climate_upload = col1.file_uploader("CSV NASA POWER (opcional)", type="csv", key="climate_upload")
    nino_upload = col2.file_uploader("CSV Niño 1+2 (opcional)", type="csv", key="nino_upload")

    if st.button("Validar y consolidar", disabled=market_upload is None):
        try:
            market_source = market_upload
            climate_source = climate_upload if climate_upload is not None else CLIMATE_FILE
            nino_source = nino_upload if nino_upload is not None else NINO_FILE
            custom_market, custom_climate, custom_nino = load_raw_sources(
                market_source, climate_source, nino_source
            )
            custom = consolidate_sources(custom_market, custom_climate, custom_nino)
            st.session_state["custom_consolidated"] = custom
            st.success("Los archivos son compatibles y fueron consolidados.")
        except Exception as exc:
            st.session_state.pop("custom_consolidated", None)
            st.error(f"No se pudo procesar la carga: {exc}")

    if "custom_consolidated" in st.session_state:
        custom = st.session_state["custom_consolidated"]
        report = quality_report(custom)
        report_cols = st.columns(4)
        report_cols[0].metric("Filas", f"{report['filas_consolidadas']:,}")
        report_cols[1].metric("Modelables", f"{report['filas_modelables']:,}")
        report_cols[2].metric("Productos", report["productos"])
        report_cols[3].metric("Duplicados", report["duplicados_serie_fecha"])

        csv_bytes = custom.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
        st.download_button(
            "Descargar consolidado validado",
            data=csv_bytes,
            file_name="datos_consolidados_carga.csv",
            mime="text/csv",
        )
        st.subheader("Clasificación con los datos cargados")
        prediction_panel(custom, model, metadata, key_prefix="upload")

else:
    page_header("Metodología y limitaciones", "Cómo se consolidaron los datos y qué significa la salida")
    st.subheader("Flujo del proyecto")
    st.code(
        "CSV originales → limpieza → variables históricas → integración climática → "
        "división temporal → KNN → evaluación → aplicación",
        language=None,
    )
    st.subheader("Fuentes")
    st.markdown(
        "- [Mercado Modelo de Piura](https://www.datosabiertos.gob.pe/dataset/precios-mayorista-y-minorista-del-mercado-modelo-de-piuragobierno-regional-piura-grp-0)\n"
        "- [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)\n"
        "- [NOAA PSL: índices mensuales](https://psl.noaa.gov/data/timeseries/month/)"
    )
    st.subheader("Reglas temporales")
    st.markdown(
        "- El clima se utiliza hasta el día anterior al precio.\n"
        "- Niño 1+2 tiene un desfase conservador de un mes.\n"
        "- Entrenamiento, validación y prueba respetan la fecha de la etiqueta futura.\n"
        "- La clase corresponde al siguiente registro disponible, no a un horizonte fijo."
    )
    st.subheader("Limitaciones")
    st.markdown(
        "- El modelo no conoce ventas, demanda, márgenes, transporte ni volumen disponible.\n"
        "- La clase ESTABLE es mayoritaria.\n"
        "- Las métricas son modestas y la mejora climática fue pequeña.\n"
        "- Una asociación estadística no demuestra causalidad.\n"
        "- No debe utilizarse como recomendación financiera o comercial."
    )

