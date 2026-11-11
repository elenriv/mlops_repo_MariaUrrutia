# Librerías y métricas

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
from mlops_pipeline.ft_eng import preparar_datos

def calcular_psi(datos_base, datos_nuevos, bins=10):
    """
    Calcula el Population Stability Index (PSI)
    para comparar dos distribuciones numéricas.
    """
    datos_base = pd.Series(datos_base).dropna()
    datos_nuevos = pd.Series(datos_nuevos).dropna()

    if datos_base.empty or datos_nuevos.empty:
        return np.nan

    cortes = np.unique(
        np.quantile(datos_base, np.linspace(0, 1, bins + 1))
    )

    if len(cortes) < 2:
        return 0.0

    cortes[0] = -np.inf
    cortes[-1] = np.inf

    frecuencias_base = pd.cut(
        datos_base, bins=cortes, include_lowest=True
    ).value_counts(normalize=True, sort=False)

    frecuencias_nuevas = pd.cut(
        datos_nuevos, bins=cortes, include_lowest=True
    ).value_counts(normalize=True, sort=False)

    epsilon = 0.0001

    proporciones_base = np.clip(frecuencias_base.values, epsilon, None)
    proporciones_nuevas = np.clip(frecuencias_nuevas.values, epsilon, None)

    psi = np.sum(
        (proporciones_nuevas - proporciones_base)
        * np.log(proporciones_nuevas / proporciones_base)
    )

    return float(psi)


def calcular_ks(datos_base, datos_nuevos):
    """Calcula la prueba Kolmogorov-Smirnov."""
    datos_base = pd.Series(datos_base).dropna()
    datos_nuevos = pd.Series(datos_nuevos).dropna()

    if datos_base.empty or datos_nuevos.empty:
        return np.nan, np.nan

    resultado = ks_2samp(datos_base, datos_nuevos)

    return float(resultado.statistic), float(resultado.pvalue)


def calcular_js(datos_base, datos_nuevos, bins=10):
    """Calcula la distancia Jensen-Shannon."""
    datos_base = pd.Series(datos_base).dropna()
    datos_nuevos = pd.Series(datos_nuevos).dropna()

    if datos_base.empty or datos_nuevos.empty:
        return np.nan

    minimo = min(datos_base.min(), datos_nuevos.min())
    maximo = max(datos_base.max(), datos_nuevos.max())

    if minimo == maximo:
        return 0.0

    cortes = np.linspace(minimo, maximo, bins + 1)

    frecuencias_base = np.histogram(datos_base, bins=cortes)[0]
    frecuencias_nuevas = np.histogram(datos_nuevos, bins=cortes)[0]

    epsilon = 0.0001

    p = frecuencias_base / frecuencias_base.sum()
    q = frecuencias_nuevas / frecuencias_nuevas.sum()

    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)

    p = p / p.sum()
    q = q / q.sum()

    return float(jensenshannon(p, q))


def calcular_js_categorico(datos_base, datos_nuevos):
    """Calcula la distancia Jensen-Shannon entre variables categóricas."""
    base = pd.Series(datos_base).fillna("Sin dato").astype(str)
    nuevos = pd.Series(datos_nuevos).fillna("Sin dato").astype(str)

    categorias = sorted(set(base.unique()) | set(nuevos.unique()))
    frecuencias_base = base.value_counts().reindex(categorias, fill_value=0)
    frecuencias_nuevas = nuevos.value_counts().reindex(categorias, fill_value=0)

    p = frecuencias_base.to_numpy(dtype=float)
    q = frecuencias_nuevas.to_numpy(dtype=float)
    p /= p.sum()
    q /= q.sum()

    return float(jensenshannon(p, q))


def calcular_psi_categorico(datos_base, datos_nuevos):
    """Calcula el PSI para variables categóricas."""
    base = pd.Series(datos_base).fillna("Sin dato").astype(str)
    nuevos = pd.Series(datos_nuevos).fillna("Sin dato").astype(str)

    categorias = sorted(set(base.unique()) | set(nuevos.unique()))
    frecuencias_base = base.value_counts(normalize=True).reindex(
        categorias, fill_value=0.0
    )
    frecuencias_nuevas = nuevos.value_counts(normalize=True).reindex(
        categorias, fill_value=0.0
    )

    epsilon = 1e-6
    p = np.clip(frecuencias_base.to_numpy(dtype=float), epsilon, None)
    q = np.clip(frecuencias_nuevas.to_numpy(dtype=float), epsilon, None)

    return float(np.sum((q - p) * np.log(q / p)))


def calcular_chi2(datos_base, datos_nuevos):
    """Calcula Chi-cuadrado para variables categóricas."""
    base = pd.Series(datos_base).fillna("Sin dato").astype(str)
    nuevos = pd.Series(datos_nuevos).fillna("Sin dato").astype(str)

    categorias = sorted(set(base.unique()) | set(nuevos.unique()))

    if len(categorias) < 2:
        return np.nan, np.nan

    frecuencias_base = base.value_counts().reindex(
        categorias, fill_value=0
    )

    frecuencias_nuevas = nuevos.value_counts().reindex(
        categorias, fill_value=0
    )

    tabla = pd.DataFrame({
        "base": frecuencias_base,
        "nuevos": frecuencias_nuevas
    })

    resultado = chi2_contingency(tabla)

    return float(resultado.statistic), float(resultado.pvalue)

#Función de detección de Data Drift

def detectar_data_drift(datos_base, datos_nuevos):
    """
    Compara las distribuciones de los datos históricos
    y los datos nuevos para detectar Data Drift.
    """

    resultados = []

    # Identificar variables comunes entre ambos conjuntos
    columnas_comunes = [
        columna for columna in datos_base.columns
        if columna in datos_nuevos.columns
    ]

    for columna in columnas_comunes:

        # Variables numéricas
        if (
            pd.api.types.is_numeric_dtype(datos_base[columna])
            and columna not in ["tipo_credito"]
        ):

            psi = calcular_psi(
                datos_base[columna],
                datos_nuevos[columna]
            )

            ks_stat, ks_pvalue = calcular_ks(
                datos_base[columna],
                datos_nuevos[columna]
            )

            js = calcular_js(
                datos_base[columna],
                datos_nuevos[columna]
            )

            resultados.append({
                "Variable": columna,
                "Tipo": "Numérica",
                "PSI": psi,
                "KS": ks_stat,
                "KS_pvalor": ks_pvalue,
                "Jensen_Shannon": js,
                "Chi2": np.nan,
                "Chi2_pvalor": np.nan
            })


        else:

            psi = calcular_psi_categorico(
                datos_base[columna],
                datos_nuevos[columna]
            )

            js = calcular_js_categorico(
                datos_base[columna],
                datos_nuevos[columna]
            )

            chi2_stat, chi2_pvalue = calcular_chi2(
                datos_base[columna],
                datos_nuevos[columna]
            )

            resultados.append({
                "Variable": columna,
                "Tipo": "Categórica",
                "PSI": psi,
                "KS": np.nan,
                "KS_pvalor": np.nan,
                "Jensen_Shannon": js,
                "Chi2": chi2_stat,
                "Chi2_pvalor": chi2_pvalue
            })

    reporte = pd.DataFrame(resultados)

    return reporte

#Clasificación del nivel de riesgo

def clasificar_riesgo(reporte):
    """
    Clasifica el nivel de Data Drift de cada variable
    y genera una alerta según los umbrales establecidos.
    """

    niveles = []
    alertas = []

    for _, fila in reporte.iterrows():

        indicadores_drift = []

        # Evaluar PSI
        if pd.notna(fila["PSI"]):
            if fila["PSI"] >= 0.25:
                indicadores_drift.append("Alto")
            elif fila["PSI"] >= 0.10:
                indicadores_drift.append("Medio")

        # Evaluar prueba KS
        if pd.notna(fila["KS_pvalor"]):
            if fila["KS_pvalor"] < 0.05:
                indicadores_drift.append("Detectado")

        # Evaluar Jensen-Shannon
        if pd.notna(fila["Jensen_Shannon"]):
            if fila["Jensen_Shannon"] >= 0.30:
                indicadores_drift.append("Alto")
            elif fila["Jensen_Shannon"] >= 0.10:
                indicadores_drift.append("Medio")

        # Evaluar Chi-cuadrado
        if pd.notna(fila["Chi2_pvalor"]):
            if fila["Chi2_pvalor"] < 0.05:
                indicadores_drift.append("Detectado")

        # Determinar nivel de riesgo
        if "Alto" in indicadores_drift:
            nivel = "Alto"
            alerta = "Revisar variable: cambio significativo en la distribución"

        elif "Medio" in indicadores_drift or "Detectado" in indicadores_drift:
            nivel = "Medio"
            alerta = "Monitorear variable: se detectaron cambios"

        else:
            nivel = "Bajo"
            alerta = "Sin cambios relevantes detectados"

        niveles.append(nivel)
        alertas.append(alerta)

    reporte = reporte.copy()
    reporte["Nivel_Riesgo"] = niveles
    reporte["Alerta"] = alertas

    return reporte

#Función principal de monitoreo

def ejecutar_monitoreo(datos_base, datos_nuevos, guardar=True):
    """
    Ejecuta el proceso completo de Data Drift.
    """

    print("INICIANDO MONITOREO DE DATA DRIFT")

    # Detectar cambios en las variables
    reporte = detectar_data_drift(
        datos_base,
        datos_nuevos
    )

    # Clasificar niveles de riesgo y generar alertas
    reporte = clasificar_riesgo(reporte)

    # Mostrar resultados
    print("\nRESULTADOS DEL MONITOREO")
    print(reporte.to_string(index=False))

    # Resumen de alertas
    print("\nRESUMEN DE RIESGOS")
    print(reporte["Nivel_Riesgo"].value_counts())

    # Guardar resultados
    if guardar:
        reporte.to_csv(
            "reporte_data_drift.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\nReporte guardado en: reporte_data_drift.csv")

    return reporte
