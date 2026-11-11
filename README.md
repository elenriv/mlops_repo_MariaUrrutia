# Proyecto MLOps: Clasificación de pago oportuno de préstamos

## 1. Descripción del proyecto

Este proyecto implementa un flujo de trabajo de Machine Learning Operations (MLOps) para la clasificación del pago oportuno de préstamos. Su propósito es desarrollar modelos de aprendizaje automático que permitan identificar si un cliente cumple o no con el pago de sus obligaciones financieras dentro del tiempo establecido.

Además del entrenamiento y evaluación de modelos, el proyecto incorpora un proceso de monitoreo de Data Drift y un dashboard interactivo desarrollado con Streamlit para visualizar los resultados y las alertas generadas.

## 2. Objetivo

Desarrollar un pipeline de Machine Learning que permita:

- Cargar y explorar los datos.
- Realizar limpieza e ingeniería de características.
- Entrenar y evaluar diferentes modelos de clasificación.
- Seleccionar el modelo con mejor desempeño según los criterios establecidos.
- Detectar posibles cambios en la distribución de los datos mediante técnicas de Data Drift.
- Visualizar los resultados del monitoreo mediante un dashboard interactivo.

## 3. Dataset

El proyecto utiliza el archivo `Base_de_datos.csv`.

La variable objetivo es `Pago_atiempo`, que representa el cumplimiento del pago:

- `0`: El cliente no paga a tiempo.
- `1`: El cliente paga a tiempo.

El conjunto de datos presenta un desbalance entre las clases, por lo que se utilizan métricas como Balanced Accuracy, Recall y F1-score para complementar la evaluación del modelo.

## 4. Modelos de Machine Learning

Se implementaron y evaluaron los siguientes algoritmos de clasificación:

- Regresión Logística.
- K-Nearest Neighbors (KNN).
- Árbol de Decisión.
- Random Forest.

La selección del modelo considera el desempeño obtenido durante la validación cruzada, la estabilidad de los resultados y la eficiencia computacional.

## 5. Ingeniería de características

El módulo `mlops_pipeline/ft_eng.py` contiene los procesos de preparación y transformación de los datos.

Entre las actividades realizadas se encuentran:

- Preparación de variables numéricas y categóricas.
- Tratamiento de valores faltantes.
- Codificación de variables categóricas.
- Escalamiento de variables numéricas.
- Separación de los datos en conjuntos de entrenamiento y prueba.

## 6. Evaluación de modelos

El archivo `mlops_pipeline/model_evaluacion.py` contiene la implementación de los modelos y sus procesos de evaluación.

Se utilizan métricas como:

- Accuracy.
- Balanced Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.
- Validación cruzada.

Los resultados de la evaluación se almacenan en `resultados_modelos.csv`.

El modelo seleccionado se guarda en el archivo `mejor_modelo.pkl`.

## 7. Monitoreo de Data Drift

El proyecto incorpora un módulo de monitoreo ubicado en `mlops_pipeline/monitoring.py`.

Este módulo permite comparar conjuntos de datos para identificar posibles cambios en sus distribuciones mediante las siguientes técnicas estadísticas:

- **Population Stability Index (PSI):** permite medir cambios en la distribución de las variables.
- **Kolmogorov-Smirnov (KS):** permite comparar distribuciones de variables numéricas.
- **Jensen-Shannon (JS):** permite medir diferencias entre distribuciones.
- **Chi-cuadrado:** permite evaluar cambios en variables categóricas.

Los resultados se clasifican en niveles de riesgo:

- Bajo.
- Medio.
- Alto.

El reporte generado se almacena en `reporte_data_drift.csv`.

**Nota:** El monitoreo desarrollado corresponde a una simulación académica de comparación entre conjuntos de datos de referencia y comparación. No representa una conexión con datos reales de producción.

## 8. Dashboard interactivo

Se desarrolló un dashboard utilizando Streamlit, ubicado en el archivo `app.py`.

El dashboard permite:

- Visualizar indicadores generales del monitoreo.
- Consultar la cantidad de variables según su nivel de riesgo.
- Identificar alertas y recomendaciones.
- Consultar las métricas de Data Drift por variable.
- Comparar distribuciones de variables.
- Registrar ejecuciones de monitoreo.
- Visualizar el historial y la evolución de los resultados.

El historial de monitoreo se almacena en `historial_monitoreo.csv`.

## 9. Estructura del proyecto

```text
mlops_repo_MariaUrrutia/
│
├── mlops_pipeline/
│   ├── carga_datos.ipynb
│   ├── eda_analisis.ipynb
│   ├── ft_eng.py
│   ├── model_evaluacion.py
│   ├── deploy.py
│   └── monitoring.py
│
├── Base_de_datos.csv
├── main.py
├── app.py
├── config.json
├── requirements.txt
├── mejor_modelo.pkl
├── resultados_modelos.csv
├── reporte_data_drift.csv
├── historial_monitoreo.csv
│
├── comparacion_auc.png
├── comparacion_rendimiento.png
├── comparacion_tiempos.png
│
├── set_up.bat
└── README.md
```

## 10. Requisitos

Para ejecutar el proyecto se requiere:

- Python 3.
- pandas.
- NumPy.
- Matplotlib.
- Seaborn.
- Scikit-learn.
- Joblib.
- SciPy.
- Streamlit.
- Jupyter Notebook.

Las dependencias se encuentran registradas en `requirements.txt`.

## 11. Instalación y ejecución

### 11.1. Clonar el repositorio

```bash
git clone https://github.com/elenriv/mlops_repo_MariaUrrutia.git
```

Ingresar a la carpeta del proyecto:

```bash
cd mlops_repo_MariaUrrutia
```

### 11.2. Crear y activar el entorno virtual

```bash
python -m venv .venv
``

En Windows:

```bash
.venv\Scripts\activate
```

### 11.3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 11.4. Ejecutar el pipeline

```bash
python main.py
```

Este proceso realiza la carga de datos, preparación, evaluación y selección del modelo, además de generar los resultados correspondientes.

### 11.5. Ejecutar el dashboard

```bash
streamlit run app.py
```

Una vez iniciado, Streamlit permitirá acceder al dashboard desde el navegador.

## 12. Conclusiones

El proyecto integra diferentes etapas del ciclo de vida de un modelo de Machine Learning, desde la exploración y preparación de los datos hasta el entrenamiento, evaluación y monitoreo.

La incorporación de técnicas estadísticas de Data Drift y un dashboard interactivo permite visualizar posibles cambios en los datos y consultar los resultados del monitoreo.

De esta manera, se presenta una implementación académica de un flujo MLOps que integra componentes de análisis de datos, aprendizaje automático, evaluación y monitoreo.