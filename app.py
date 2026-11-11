
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from pathlib import Path
from mlops_pipeline.ft_eng import preparar_datos


# ==========================================
# CONFIGURACIÓN DEL DASHBOARD
# ==========================================

st.set_page_config(
    page_title="Monitoreo de Data Drift",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# CARGA Y PREPARACIÓN DE LOS DATOS
# ==========================================

RUTA_DATOS = Path(__file__).parent / "Base_de_datos.csv"

@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)

datos = cargar_datos()

X_train, X_test, y_train, y_test = preparar_datos(datos)


# ==========================================
# RUTAS DE ARCHIVOS
# ==========================================

CARPETA_PROYECTO = Path(__file__).parent

RUTA_REPORTE = CARPETA_PROYECTO / "reporte_data_drift.csv"
RUTA_HISTORIAL = CARPETA_PROYECTO / "historial_monitoreo.csv"
RUTA_DATOS = CARPETA_PROYECTO / "Base_de_datos.csv"


# ==========================================
# CARGA DE RESULTADOS DEL MONITOREO
# ==========================================

@st.cache_data
def cargar_reporte():

    if RUTA_REPORTE.exists():
        return pd.read_csv(RUTA_REPORTE)

    return pd.DataFrame()


reporte = cargar_reporte()


# ==========================================
# CARGA DE DATOS PARA COMPARACIÓN
# ==========================================

@st.cache_data
def cargar_datos_comparacion():

    if not RUTA_DATOS.exists():
        return None, None

    df = pd.read_csv(RUTA_DATOS)

    X_train, X_test, y_train, y_test = preparar_datos(df)

    return X_train, X_test


X_train, X_test = cargar_datos_comparacion()


# ==========================================
# ENCABEZADO
# ==========================================

st.title("Dashboard de Monitoreo de Data Drift")

st.markdown(
    """
    Sistema de monitoreo para identificar cambios en la
    distribución de los datos utilizados en la clasificación
    del pago oportuno de préstamos.
    """
)

st.divider()


# ==========================================
# VALIDACIÓN DEL REPORTE
# ==========================================

if reporte.empty:

    st.warning(
        "No se encontró el reporte de Data Drift. "
        "Ejecuta primero el proceso de monitoreo."
    )

    st.stop()

else:

    st.success("Reporte de Data Drift cargado correctamente.")


# ==========================================
# INDICADORES PRINCIPALES (KPIs)
# ==========================================

st.subheader("Estado general del monitoreo")

total_variables = len(reporte)

riesgo_bajo = (reporte["Nivel_Riesgo"] == "Bajo").sum()
riesgo_medio = (reporte["Nivel_Riesgo"] == "Medio").sum()
riesgo_alto = (reporte["Nivel_Riesgo"] == "Alto").sum()


# Determinar estado general

if riesgo_alto > 0:
    estado_general = "Riesgo alto"

elif riesgo_medio > 0:
    estado_general = "Riesgo moderado"

else:
    estado_general = "Estable"


# Mostrar indicadores

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Variables monitoreadas", total_variables)

with col2:
    st.metric("Riesgo bajo", riesgo_bajo)

with col3:
    st.metric("Riesgo medio", riesgo_medio)

with col4:
    st.metric("Riesgo alto", riesgo_alto)

with col5:
    st.metric("Estado", estado_general)


# ==========================================
# HISTORIAL DE EJECUCIONES DEL MONITOREO
# ==========================================

st.subheader("Registrar ejecución del monitoreo")

if st.button("Registrar monitoreo actual"):

    nuevo_registro = pd.DataFrame([{
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Riesgo_bajo": int(riesgo_bajo),
        "Riesgo_medio": int(riesgo_medio),
        "Riesgo_alto": int(riesgo_alto),
        "PSI_promedio": float(reporte["PSI"].mean()),
        "PSI_maximo": float(reporte["PSI"].max()),
        "Estado_general": estado_general
    }])

    if RUTA_HISTORIAL.exists():
        historial = pd.read_csv(RUTA_HISTORIAL)
        historial = pd.concat(
            [historial, nuevo_registro],
            ignore_index=True
        )
    else:
        historial = nuevo_registro

    historial.to_csv(
        RUTA_HISTORIAL,
        index=False,
        encoding="utf-8-sig"
    )

    st.success("Ejecución registrada correctamente.")


# ==========================================
# ALERTAS Y RECOMENDACIONES AUTOMÁTICAS
# ==========================================

st.subheader("Alertas y recomendaciones del monitoreo")

variables_altas = reporte[
    reporte["Nivel_Riesgo"] == "Alto"
]

variables_medias = reporte[
    reporte["Nivel_Riesgo"] == "Medio"
]

variables_bajas = reporte[
    reporte["Nivel_Riesgo"] == "Bajo"
]

if not variables_altas.empty:

    st.error(
        f"ALERTA CRÍTICA: Se detectaron cambios importantes "
        f"en {len(variables_altas)} variable(s)."
    )

    st.write("**Variables que requieren atención inmediata:**")

    for _, fila in variables_altas.iterrows():
        st.write(
            f"- **{fila['Variable']}**: {fila['Alerta']}"
        )

    st.warning(
        "Recomendación: revisar la calidad de los datos, "
        "investigar las causas del cambio y evaluar un "
        "reentrenamiento del modelo antes de utilizarlo "
        "en nuevas predicciones."
    )

elif not variables_medias.empty:

    st.warning(
        f"ATENCIÓN: Se detectaron cambios moderados "
        f"en {len(variables_medias)} variable(s)."
    )

    st.write("**Variables que deben monitorearse:**")

    for _, fila in variables_medias.iterrows():
        st.write(
            f"- **{fila['Variable']}**: {fila['Alerta']}"
        )

    st.info(
        "Recomendación: revisar las variables identificadas, "
        "comparar sus distribuciones y realizar un nuevo "
        "monitoreo para comprobar si el cambio persiste."
    )

else:

    st.success(
        "No se detectaron desviaciones relevantes en las "
        "variables monitoreadas."
    )

    st.info(
        "Recomendación: continuar con el monitoreo periódico "
        "para verificar que las distribuciones se mantengan "
        "estables."
    )

st.divider()

# ==========================================
# ANÁLISIS TEMPORAL DEL DATA DRIFT
# ==========================================

st.divider()
st.subheader("Evolución temporal del monitoreo")

if RUTA_HISTORIAL.exists():

    historial = pd.read_csv(RUTA_HISTORIAL)

    if not historial.empty:

        historial["Fecha"] = pd.to_datetime(
            historial["Fecha"]
        )

        st.write(
            "Evolución de las variables según su nivel de riesgo "
            "en las diferentes ejecuciones registradas."
        )

        historial_grafica = historial.set_index("Fecha")

        st.line_chart(
            historial_grafica[
                ["Riesgo_bajo", "Riesgo_medio", "Riesgo_alto"]
            ]
        )

        st.write("**Historial de ejecuciones:**")

        st.dataframe(
            historial.sort_values(
                "Fecha",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.write("**Evolución del PSI promedio:**")

        st.line_chart(
            historial_grafica[["PSI_promedio", "PSI_maximo"]]
        )

    else:
        st.info("Todavía no hay ejecuciones registradas.")

else:
    st.info(
        "Aún no existe un historial. "
        "Registra la primera ejecución del monitoreo."
    )

# ==========================================
# TABLA DE RESULTADOS DEL MONITOREO
# ==========================================

st.subheader("Detalle de métricas por variable")

st.dataframe(
    reporte.sort_values(
        by="Nivel_Riesgo",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# DISTRIBUCIÓN DE NIVELES DE RIESGO
# ==========================================

st.divider()

st.subheader("Distribución de niveles de riesgo")

riesgos = pd.DataFrame({
    "Nivel de riesgo": ["Bajo", "Medio", "Alto"],
    "Cantidad": [riesgo_bajo, riesgo_medio, riesgo_alto]
})


col1, col2 = st.columns(2)

with col1:

    st.bar_chart(
        riesgos.set_index("Nivel de riesgo")
    )


with col2:

    st.write("**Resumen de variables monitoreadas**")

    st.dataframe(
        riesgos,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# COMPARACIÓN DE DISTRIBUCIONES
# ==========================================

st.divider()

st.subheader("Comparación de distribuciones")

st.write(
    """Compara la distribución de los datos de entrenamiento 
    con un conjunto de prueba utilizado como escenario de
    simulación de datos nuevos."""
)


# Validar disponibilidad de los datos

if X_train is None or X_test is None:

    st.error(
        "No se encontró Base_de_datos.csv. "
        "Verifica que el archivo esté en la carpeta del proyecto."
    )

else:

    # ==========================================
    # IDENTIFICACIÓN DE VARIABLES
    # ==========================================

    # Variables categóricas

    variables_categoricas = [
        columna
        for columna in [
            "tipo_credito",
            "tipo_laboral",
            "tendencia_ingresos"
        ]
        if columna in X_train.columns
        and columna in X_test.columns
    ]


    # Variables numéricas

    variables_numericas = [
        columna
        for columna in X_train.columns.intersection(X_test.columns)
        if pd.api.types.is_numeric_dtype(X_train[columna])
        and columna not in variables_categoricas
    ]


    # Unir variables disponibles

    variables_disponibles = (
        variables_numericas + variables_categoricas
    )


    # ==========================================
    # SELECCIÓN DE VARIABLE
    # ==========================================

    if variables_disponibles:

        variable_seleccionada = st.selectbox(
            "Selecciona una variable:",
            variables_disponibles
        )


        datos_historicos = (
            X_train[variable_seleccionada].dropna()
        )

        datos_actuales = (
            X_test[variable_seleccionada].dropna()
        )


        # ==========================================
        # GENERACIÓN DE GRÁFICAS
        # ==========================================

        if datos_historicos.empty or datos_actuales.empty:

            st.warning(
                "No hay suficientes datos para comparar "
                "esta variable."
            )

        else:

            fig, ax = plt.subplots(figsize=(10, 5))


            # Comparación de variables categóricas

            if variable_seleccionada in variables_categoricas:

                frecuencias_historicas = (
                    datos_historicos.astype(str)
                    .value_counts(normalize=True)
                )

                frecuencias_actuales = (
                    datos_actuales.astype(str)
                    .value_counts(normalize=True)
                )


                comparacion = pd.concat(
                    [
                        frecuencias_historicas.rename("Históricos"),
                        frecuencias_actuales.rename("Nuevos")
                    ],
                    axis=1
                ).fillna(0)


                comparacion.plot(
                    kind="bar",
                    ax=ax,
                    width=0.8
                )

                ax.set_ylabel("Frecuencia relativa")
                ax.set_xlabel("Categorías")
                ax.tick_params(
                    axis="x",
                    rotation=45
                )


            # Comparación de variables numéricas

            else:

                ax.hist(
                    datos_historicos,
                    bins=20,
                    alpha=0.6,
                    label="Datos históricos",
                    density=True
                )

                ax.hist(
                    datos_actuales,
                    bins=20,
                    alpha=0.6,
                    label="Datos nuevos",
                    density=True
                )

                ax.set_ylabel("Densidad")
                ax.set_xlabel(variable_seleccionada)


            # Configuración de la gráfica

            ax.set_title(
                f"Distribución de {variable_seleccionada}"
            )

            ax.legend()

            st.pyplot(fig)

            plt.close(fig)


    else:

        st.warning(
            "No hay variables disponibles para comparar."
        )
