"""Evaluación sobre la malla completa a partir de las predicciones guardadas por semana."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import metricas
from .graficos import CLASES
from .tiempo import inicio_semana


def cargar_predicciones(carpeta):
    """Lee etiquetas y puntajes [n_semanas, n_celdas] (sin cargarlos enteros en memoria)."""
    carpeta = Path(carpeta)
    if not (carpeta / "y.npy").exists():
        raise FileNotFoundError(f"No hay predicciones en {carpeta}. Entrene el modelo: python -m incendios entrenar")
    return {
        "y": np.load(carpeta / "y.npy", mmap_mode="r"),
        "y_tol": np.load(carpeta / "y_tol.npy", mmap_mode="r"),
        "nuevo": np.load(carpeta / "nuevo.npy", mmap_mode="r"),
        "semanas": np.load(carpeta / "semanas.npy"),
        "pred": {ruta.stem[len("pred_"):]: np.load(ruta, mmap_mode="r")
                 for ruta in sorted(carpeta.glob("pred_*.npy"))},
    }


def filtrar(p, celdas=None, semanas=None):
    """Subconjunto de celdas (máscara) y/o de semanas (máscara)."""
    celdas = slice(None) if celdas is None else np.asarray(celdas)
    semanas = slice(None) if semanas is None else np.asarray(semanas)

    def corte(a):
        return np.asarray(a)[semanas][:, celdas]
    return {"y": corte(p["y"]), "y_tol": corte(p["y_tol"]), "nuevo": corte(p["nuevo"]),
            "semanas": p["semanas"][semanas],
            "pred": {n: corte(v) for n, v in p["pred"].items()}}


def _grupos(y):
    return np.repeat(np.arange(y.shape[0]), y.shape[1])


def metricas_modelo(p, nombre):
    """Todas las métricas de un modelo sobre el conjunto de predicciones dado."""
    y = np.asarray(p["y"])
    puntaje = np.asarray(p["pred"][nombre])
    grupos = _grupos(y)
    r = metricas.resumen(y.ravel(), puntaje.ravel(), grupos)
    # una celda señalada cuenta como acierto si hubo fuego en ella o en una vecina
    r["precision_1_tol"] = metricas.precision_top(np.asarray(p["y_tol"]).ravel(), puntaje.ravel(), grupos, 0.01)
    # focos nuevos: sin fuego a menos de ~2,75 km en las cuatro semanas previas
    y_nuevo = (y & np.asarray(p["nuevo"])).ravel()
    r["n_pos_nuevos"] = int(y_nuevo.sum())
    if r["n_pos_nuevos"]:
        captura = metricas.captura_por_semana(y_nuevo, puntaje.ravel(), grupos, fracciones=(0.01, 0.05, 0.10))
        r.update({f"captura_nuevos_{round(f * 100)}": c for f, c in captura.items()})
    else:
        r.update({f"captura_nuevos_{k}": float("nan") for k in (1, 5, 10)})
    return r


def tabla_modelos(p, nombres=None):
    nombres = list(nombres or p["pred"])
    tabla = pd.DataFrame({n: metricas_modelo(p, n) for n in nombres}).T
    tabla.index.name = "modelo"
    return tabla.sort_values("pr_auc", ascending=False)


def tabla_por_grupo(p, nombre, grupo_por_celda, nombres_grupo):
    """Métricas del modelo dentro de cada grupo de celdas (el ranking se rehace dentro del grupo)."""
    filas = {}
    for g, etiqueta in enumerate(nombres_grupo):
        sub = filtrar({**p, "pred": {nombre: p["pred"][nombre]}}, celdas=np.asarray(grupo_por_celda) == g)
        if sub["y"].sum() == 0:
            continue
        filas[etiqueta] = metricas_modelo(sub, nombre)
    tabla = pd.DataFrame(filas).T
    tabla.index.name = "grupo"
    return tabla


def calibracion(prob, y, bordes):
    """Probabilidad media predicha frente a frecuencia observada, por tramo de probabilidad."""
    prob, y = np.asarray(prob).ravel(), np.asarray(y).ravel()
    tramo = np.clip(np.searchsorted(bordes, prob, side="right") - 1, 0, len(bordes) - 2)
    n = np.bincount(tramo, minlength=len(bordes) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        predicha = np.bincount(tramo, weights=prob, minlength=len(bordes) - 1) / n
        observada = np.bincount(tramo, weights=y, minlength=len(bordes) - 1) / n
    return pd.DataFrame({"desde": bordes[:-1], "hasta": bordes[1:], "n": n,
                         "predicha": predicha, "observada": observada})


def resumen_calibracion(prob, y):
    """Nivel de unas probabilidades frente a lo observado: celdas con fuego esperadas y
    observadas, su razón (1 = nivel exacto) y la pérdida logarítmica media."""
    prob = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.clip(prob, 1e-9, 1.0 - 1e-9)
    observadas, esperadas = float(y.sum()), float(prob.sum())
    return {"observadas": int(observadas), "esperadas": esperadas,
            "razon": esperadas / observadas if observadas else float("nan"),
            "perdida_log": float(-(y * np.log(p) + (1.0 - y) * np.log1p(-p)).mean())}


def serie_semanal(prob, y, semanas):
    """Celdas con fuego observadas y esperadas (suma de probabilidades) en cada semana."""
    prob, y = np.asarray(prob), np.asarray(y)
    return pd.DataFrame({
        "semana": np.asarray(semanas),
        "inicio": [inicio_semana(s) for s in semanas],
        "observadas": y.sum(axis=1).astype(int),
        "esperadas": prob.sum(axis=1, dtype=np.float64),
    })


def clases_de_riesgo(prob, umbrales):
    """Clase 0 (muy bajo) a 4 (muy alto) según los umbrales de probabilidad. NaN -> 0."""
    prob = np.nan_to_num(np.asarray(prob, dtype=np.float64), nan=0.0)
    return np.searchsorted(np.asarray(umbrales, dtype=np.float64), prob, side="right").astype(np.uint8)


def tabla_clases(prob, y, umbrales):
    """Qué parte del territorio-tiempo y del fuego cae en cada clase, y cuánto arde cada una."""
    prob, y = np.asarray(prob).ravel(), np.asarray(y).ravel()
    clase = clases_de_riesgo(prob, umbrales)
    n = np.bincount(clase, minlength=5).astype(float)
    focos = np.bincount(clase, weights=y, minlength=5)
    with np.errstate(invalid="ignore", divide="ignore"):
        frecuencia = focos / n
    return pd.DataFrame({
        "clase": CLASES,
        "umbral_inferior": [0.0] + [float(u) for u in umbrales],
        "pct_celdas": 100.0 * n / n.sum(),
        "pct_focos": 100.0 * focos / focos.sum(),
        "frecuencia_observada": frecuencia,
    })


def importancia_permutacion(puntuar, X, y, grupos, repeticiones=3, semilla=42):
    """Cuánto cae el PR-AUC al barajar juntas las columnas de cada grupo.

    `puntuar`: función que recibe la matriz y devuelve un puntaje por fila.
    `grupos`: {nombre: [columnas]}. Barajar un grupo entero conserva la relación
    entre sus variables y rompe su relación con todo lo demás.
    """
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(semilla)
    base = average_precision_score(y, puntuar(X))
    filas = []
    trabajo = np.array(X, copy=True)
    for nombre, columnas in grupos.items():
        columnas = np.asarray(columnas)
        caidas = []
        for _ in range(repeticiones):
            orden = rng.permutation(len(y))
            trabajo[:, columnas] = X[orden][:, columnas]
            caidas.append(base - average_precision_score(y, puntuar(trabajo)))
        trabajo[:, columnas] = X[:, columnas]
        filas.append({"grupo": nombre, "caida_pr_auc": float(np.mean(caidas)), "desvio": float(np.std(caidas))})
    return pd.DataFrame(filas).sort_values("caida_pr_auc", ascending=False).reset_index(drop=True)


def formatear(valor, formato):
    """Número con formato en español: 'pct1' 12,3 %, 'pct2', 'pct0', 'entero' 1.234, 'dec3' 0,935."""
    if isinstance(valor, str):
        return valor
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    if formato == "entero":
        return f"{int(round(valor)):,}".replace(",", ".")
    if formato.startswith("pct"):
        return f"{100 * valor:.{int(formato[3:])}f} %".replace(".", ",")
    if formato.startswith("dec"):
        return f"{valor:.{int(formato[3:])}f}".replace(".", ",")
    return str(valor)


def tabla_md(tabla, formatos=None, encabezados=None):
    """Tabla en Markdown. `formatos`: {columna: formato}; las columnas numéricas van alineadas a la derecha."""
    formatos, encabezados = formatos or {}, encabezados or {}
    columnas = list(tabla.columns)
    lineas = ["| " + " | ".join(encabezados.get(c, c) for c in columnas) + " |",
              "|" + "|".join("---:" if c in formatos else "---" for c in columnas) + "|"]
    for fila in tabla.itertuples(index=False):
        celdas = [formatear(v, formatos[c]) if c in formatos else str(v) for c, v in zip(columnas, fila)]
        lineas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(lineas)
