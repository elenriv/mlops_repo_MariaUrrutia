
import pandas as pd

from mlops_pipeline.ft_eng import preparar_datos, preprocesador
from mlops_pipeline.model_evaluacion import MODELOS, comparar_modelos


def main():
    print("=" * 60)
    print("PROYECTO DE MACHINE LEARNING")
    print("Clasificación del pago oportuno de préstamos")
    print("=" * 60)

    print("\n1. Cargando conjunto de datos...")
    df = pd.read_csv("Base_de_datos.csv", encoding="utf-8-sig")
    print(f"Datos cargados: {df.shape[0]} registros y {df.shape[1]} columnas.")


    print("\n2. Preparando los datos...")
    X_train, X_test, y_train, y_test = preparar_datos(df)

    print(f"Datos de entrenamiento: {X_train.shape}")
    print(f"Datos de prueba: {X_test.shape}")

    print("\n3. Entrenando y evaluando modelos...")
    resultados, modelos_entrenados, mejor_modelo, nombre_mejor = comparar_modelos(
        MODELOS,
        preprocesador,
        X_train,
        X_test,
        y_train,
        y_test
    )
    
    print("\n4. Resultados de la evaluación:")
    print(resultados.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"MODELO FINAL SELECCIONADO: {nombre_mejor}")
    print("=" * 60)

    print("\nProceso finalizado.")


if __name__ == "__main__":
    main()
