# Desarrollo, evaluación y despliegue de modelos de Machine Learning

## 1. Descripción del proyecto

Este proyecto desarrolla un flujo de trabajo de Machine Learning para clasificar el pago oportuno de créditos.

Se realiza la preparación de datos, ingeniería de características, preprocesamiento, entrenamiento, evaluación y comparación de diferentes modelos de clasificación supervisada.

## 2. Objetivo

Construir y evaluar modelos capaces de clasificar si un cliente paga o no a tiempo sus obligaciones crediticias.

**Variable objetivo:** `Pago_atiempo`

- 0: No pagó a tiempo.
- 1: Pagó a tiempo.

## 3. Modelos evaluados

- Regresión Logística.
- K-Nearest Neighbors (KNN).
- Árbol de Decisión.
- Random Forest.

## 4. Ingeniería de características

Se generan variables derivadas y se preparan los conjuntos de entrenamiento y prueba.

Se utilizan pipelines de preprocesamiento con:

- SimpleImputer.
- OneHotEncoder.
- OrdinalEncoder.
- ColumnTransformer.
- StandardScaler.

## 5. Evaluación

Se utilizan métricas de clasificación como:

- Accuracy.
- Balanced Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.

También se realiza validación cruzada para analizar la consistencia de los modelos y se consideran los tiempos de entrenamiento y predicción.

## 6. Resultados

El modelo seleccionado mediante el criterio de evaluación implementado fue Regresión Logística.

Los resultados completos se encuentran en `resultados_modelos.csv`.

## 7. Archivos generados

- `resultados_modelos.csv`: resultados de la evaluación.
- `comparacion_rendimiento.png`: comparación del rendimiento.
- `comparacion_auc.png`: comparación de las métricas AUC.
- `comparacion_tiempos.png`: comparación de tiempos.
- `mejor_modelo.pkl`: modelo seleccionado y guardado.

## 8. Requisitos

- Python instalado.
- Git, para clonar el repositorio.
- Las dependencias especificadas en `requirements.txt`.

## 9. Instalación y ejecución

### 9.1. Clonar el repositorio

```bash
git clone url_repositorio
cd nombre_repositorio
```

### 9.2. Crear un entorno virtual

```bash
python -m venv .venv
```

### 9.3. Activar el entorno virtual

En Windows:

```powershell
.venv\Scripts\activate
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

### 9.4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 9.5. Ejecutar el proyecto

Desde la carpeta raíz del repositorio, ejecutar:

```bash
python main.py
```

Este comando inicia el flujo de trabajo, carga los datos, prepara las variables, entrena y evalúa los modelos, compara sus resultados y guarda los archivos generados.

## 10. Visualización de resultados

Al finalizar la ejecución:

- Los resultados de la evaluación se pueden consultar en `resultados_modelos.csv`.
- Las gráficas comparativas se generan como archivos PNG.
- El modelo seleccionado queda guardado en `mejor_modelo.pkl`.

Las gráficas también se muestran durante la ejecución del programa.
