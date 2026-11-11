import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


##Configuración de variables

TARGET = "Pago_atiempo"

CATEGORICAS_NOMINALES = [
    "tipo_credito",
    "tipo_laboral"
]
CATEGORICAS_ORDINALES = [
    "tendencia_ingresos"
]
VARIABLES_EXCLUIDAS = [
    "puntaje",
    "saldo_mora_codeudor",
    "fecha_prestamo"
]

#Clasificación de variables

NUMERICAS = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito"
]

NUMERICAS_DERIVADAS = [
    "cuota_sobre_salario",
    "capital_sobre_salario",
    "otros_prestamos_sobre_salario",
    "tiene_mora",
    "sin_datos_ingresos",
    "mes_prestamo"
]

NUMERICAS_COMPLETAS = NUMERICAS + NUMERICAS_DERIVADAS


# Ingeniería de características

def crear_atributos(df):

    """Genera características derivadas a partir de las variables originales"""
    df = df.copy()
    
    #Limpiar la variable ordinal
    categorias_validas = ["Decreciente", "Estable", "Creciente"]
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(categorias_validas),
        np.nan
    )


    df["cuota_sobre_salario"] = (
    df["cuota_pactada"] / df["salario_cliente"].replace(0, np.nan)
    )
    
    df["capital_sobre_salario"] = (
    df["capital_prestado"] / df["salario_cliente"].replace(0, np.nan)
    )
    df["otros_prestamos_sobre_salario"] = (
        df["total_otros_prestamos"] / df["salario_cliente"].replace(0, np.nan)
    )
    df["tiene_mora"] = (df["saldo_mora"] > 0).astype(int)

    df["sin_datos_ingresos"] = (
        df["promedio_ingresos_datacredito"].isna()
    ).astype(int)

    df["mes_prestamo"] = pd.to_datetime(
        df["fecha_prestamo"], errors="coerce"
    ).dt.month

    df = df.replace([np.inf, -np.inf], np.nan)

    return df

# Pipeline numérico

pipeline_numerico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler())
])

#Pipeline categórico nominal

pipeline_categorico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OneHotEncoder(handle_unknown="ignore"))
])

#Pipeline categórico ordinal

pipeline_ordinal = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OrdinalEncoder(
        categories=[["Decreciente", "Estable", "Creciente"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

#Integracipon de pipelines

preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", pipeline_numerico, NUMERICAS_COMPLETAS),
        ("categoricas", pipeline_categorico, CATEGORICAS_NOMINALES),
        ("ordinales", pipeline_ordinal, CATEGORICAS_ORDINALES)
    ],
    remainder="drop"
)

#Preparación y Separación de datos

def preparar_datos(df, test_size=0.2, random_state=42):

    df = crear_atributos(df)

    X = df.drop(columns=[TARGET] + VARIABLES_EXCLUIDAS)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test