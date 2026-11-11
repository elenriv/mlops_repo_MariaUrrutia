from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd
import time

def build_model(model, preprocessor):
    """
    Construye un pipeline que integra el preprocesamiento
    y el modelo de clasificación.
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline

#Configuración de modelos

MODELOS = {
    "Regresión Logística": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Árbol de Decisión": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

#Resumen de métricas de clasificación

def summarize_classification(y_true, y_pred, y_prob=None):
    """
    Calcula métricas para evaluar un modelo de clasificación.
    La clase 0 representa los créditos que no se pagaron a tiempo.
    """

    resultados = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (clase 0)": precision_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "Recall (clase 0)": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "F1 (clase 0)": f1_score(
            y_true, y_pred, pos_label=0, zero_division=0
        )
    }

    if y_prob is not None:
        resultados["ROC-AUC"] = roc_auc_score(
            (pd.Series(y_true) == 0).astype(int),
            y_prob
        )
        resultados["PR-AUC"] = average_precision_score(
            y_true,
            y_prob,
            pos_label=0
        )

    return resultados

#Entrenamiento y evaluación de modelos
def evaluar_modelo(model, preprocessor, X_train, X_test, y_train, y_test):

    pipeline = build_model(model, preprocessor)


    inicio_entrenamiento = time.perf_counter()
    pipeline.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio_entrenamiento

    inicio_prediccion = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    tiempo_prediccion = time.perf_counter() - inicio_prediccion

    y_prob = None

    if hasattr(pipeline, "predict_proba"):
        probabilidades = pipeline.predict_proba(X_test)
        indice_clase_0 = list(pipeline.classes_).index(0)
        y_prob = probabilidades[:, indice_clase_0]

    resultados = summarize_classification(
        y_test,
        y_pred,
        y_prob
    )

    resultados["Tiempo entrenamiento (s)"] = tiempo_entrenamiento
    resultados["Tiempo predicción (s)"] = tiempo_prediccion

    return pipeline, resultados


def comparar_modelos(
    modelos,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
    cv_splits=5
):
    resultados_modelos = []
    modelos_entrenados = {}

    validacion = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42
    )

    for nombre, modelo in modelos.items():

        print(f"\nEvaluando: {nombre}")


        pipeline_cv = build_model(modelo, preprocessor)

        puntuaciones = cross_validate(
            pipeline_cv,
            X_train,
            y_train,
            cv=validacion,
            scoring="balanced_accuracy",
            return_train_score=False
        )

        pipeline_entrenado, metricas = evaluar_modelo(
            modelo,
            preprocessor,
            X_train,
            X_test,
            y_train,
            y_test
        )

        metricas["Modelo"] = nombre
        metricas["CV Balanced Accuracy (media)"] = puntuaciones[
            "test_score"
        ].mean()
        metricas["CV Balanced Accuracy (desviación)"] = puntuaciones[
            "test_score"
        ].std()

        resultados_modelos.append(metricas)
        modelos_entrenados[nombre] = pipeline_entrenado

    tabla_resultados = pd.DataFrame(resultados_modelos)
    mejor_modelo, nombre_mejor, tabla_resultados = seleccionar_mejor_modelo(
        tabla_resultados,
        modelos_entrenados
    )
    guardar_resultados(tabla_resultados)
    graficar_resultados(tabla_resultados)
    guardar_mejor_modelo(mejor_modelo, nombre_mejor)
    return tabla_resultados, modelos_entrenados, mejor_modelo, nombre_mejor

def guardar_resultados(resultados, ruta="resultados_modelos.csv"):
    """Guarda la tabla comparativa de los modelos en formato CSV."""
    resultados.to_csv(
        ruta,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\nResultados guardados correctamente en:{ruta}")

def graficar_resultados(resultados):
    """Genera y guarda gráficos comparativos de los modelos."""
    metricas_rendimiento = [
        "Balanced Accuracy",
        "Precision (clase 0)",
        "Recall (clase 0)",
        "F1 (clase 0)"
    ]
    resultados.plot(
        x="Modelo",
        y=metricas_rendimiento,
        kind="bar",
        figsize=(12, 6)
    )
    plt.title("Comparación de rendimiento de los modelos")
    plt.ylabel("Puntuación")
    plt.xlabel("Modelo")
    plt.xticks(rotation=15)
    plt.ylim(0, 1)
    plt.legend(title="Métricas")
    plt.tight_layout()
    plt.savefig("comparacion_rendimiento.png", dpi=300)
    plt.show()
    plt.close()

    metricas_clasificacion = ["ROC-AUC", "PR-AUC"]
    resultados.plot(
        x="Modelo",
        y=metricas_clasificacion,
        kind="bar",
        figsize=(10, 6)
    )
    plt.title("Comparación de ROC-AUC y PR-AUC")
    plt.ylabel("Puntuación")
    plt.xlabel("Modelo")
    plt.xticks(rotation=15)
    plt.ylim(0, 1)
    plt.legend(title="Métricas")
    plt.tight_layout()
    plt.savefig("comparacion_auc.png", dpi=300)
    plt.show()
    plt.close()

    metricas_tiempo = [
        "Tiempo entrenamiento (s)",
        "Tiempo predicción (s)"
    ]
    resultados.plot(
        x="Modelo",
        y=metricas_tiempo,
        kind="bar",
        figsize=(10, 6)
    )
    plt.title("Comparación de tiempos de ejecución")
    plt.ylabel("Tiempo (segundos)")
    plt.xlabel("Modelo")
    plt.xticks(rotation=15)
    plt.legend(title="Tiempos")
    plt.tight_layout()
    plt.savefig("comparacion_tiempos.png", dpi=300)
    plt.show()
    plt.close()

    print("\nGráficos generados y guardados correctamente.")


    #Selección del mejor modelo

def seleccionar_mejor_modelo(resultados, modelos_entrenados):

    tabla = resultados.copy()


    tiempo_maximo = tabla["Tiempo entrenamiento (s)"].max()

    if tiempo_maximo > 0:
        eficiencia = 1 - (
            tabla["Tiempo entrenamiento (s)"] / tiempo_maximo
        )
    else:
        eficiencia = 1


    tabla["Puntuacion_seleccion"] = (
        0.50 * tabla["CV Balanced Accuracy (media)"]
        + 0.30 * (1 - tabla["CV Balanced Accuracy (desviación)"])
        + 0.20 * eficiencia
    )

    indice_mejor = tabla["Puntuacion_seleccion"].idxmax()

    nombre_mejor = tabla.loc[indice_mejor, "Modelo"]

    mejor_modelo = modelos_entrenados[nombre_mejor]

    print("\n======================================")
    print("MODELO SELECCIONADO")
    print("======================================")
    print(f"Modelo: {nombre_mejor}")
    print(
        f"Puntuación: "
        f"{tabla.loc[indice_mejor, 'Puntuacion_seleccion']:.4f}"
    )

    return mejor_modelo, nombre_mejor, tabla


#Guardar modelo seleccionado

def guardar_mejor_modelo(modelo, nombre, ruta= "mejor_modelo.pkl"):
    joblib.dump(modelo, ruta)
    print("\nModelo guardado correctamente.")
    print(f"Modelo seleccionado: {nombre}")
    print(f"Archivo generado: {ruta}")