"""Modelos de riesgo: fábrica de estimadores, entrenamiento sobre la muestra y línea base por provincia."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .flujo import columnas_de, matriz_provincia
from .muestreo import corregir_prior

NOMBRES = ("logistica", "bosque", "gbm")

# Punto de partida razonable; `python -m incendios modelar --busqueda` prueba alternativas.
PARAMETROS_GBM = {"learning_rate": 0.06, "max_leaf_nodes": 63, "min_samples_leaf": 100,
                  "l2_regularization": 1.0, "max_features": 0.8}
ITER_GBM = 100          # árboles por defecto si no se pide parada temprana
ITER_MAX = 600
PASO_PARADA = 25
PACIENCIA = 3

# Restricciones de monotonía del GBM. Solo en las variables que cambian de semana a
# semana para todo el departamento (clima antecedente y actividad departamental de
# fuego): con ~370 semanas de entrenamiento, un modelo libre puede "reconocer" semanas
# concretas por su clima y memorizar cuánto ardió. Obligarlo a que más sequía nunca baje
# el riesgo (y más lluvia nunca lo suba) evita eso y mejora la validación.
MONOTONIA = {
    "prec_7": -1, "prec_30": -1, "prec_90": -1, "dias_sin_lluvia": 1, "tmax_7": 1, "hrmin_7": -1,
    "viento_7": 1, "vpd_7": 1, "ffmc": 1, "dmc": 1, "dc": 1, "isi": 1, "bui": 1, "fwi": 1,
    "fwi_7": 1, "fwi_max7": 1, "suelo_sup_7": -1, "suelo_raiz_7": -1,
    "an_prec_90": -1, "an_tmax_7": 1, "an_hrmin_7": -1, "an_fwi_7": 1, "an_dc": 1,
    "rec_dep_1": 1, "rec_dep_4": 1, "rec_dep_anom": 1,
}


class ColumnasVacias(BaseEstimator, TransformerMixin):
    """Pone en cero las columnas que no tuvieron ningún dato al entrenar.

    Ocurre cuando falta una capa entera (por ejemplo, ninguna tesela de relieve).
    Una columna constante no aporta nada al modelo, y así ningún estimador falla.
    """

    def fit(self, X, y=None):
        self.vacias_ = np.flatnonzero(np.isnan(X).all(axis=0))
        return self

    def transform(self, X):
        if len(self.vacias_) == 0:
            return X
        X = np.array(X, copy=True)
        X[:, self.vacias_] = 0.0
        return X


def restricciones(variables):
    """Signo de la restricción de monotonía de cada variable (0 = libre)."""
    return [MONOTONIA.get(nombre, 0) for nombre in variables]


def crear(nombre, semilla=config.SEMILLA, rapido=False, n_iter=None, variables=None, monotono=True,
          **parametros):
    """Estimador de scikit-learn sin entrenar. `rapido` lo achica para pruebas.

    `variables`: nombres de las columnas, para aplicar al GBM las restricciones de monotonía.
    """
    if nombre == "logistica":
        final = [SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                 LogisticRegression(C=parametros.get("C", 1.0), max_iter=500,
                                    tol=1e-2 if rapido else 1e-4)]
    elif nombre == "bosque":
        base = {"n_estimators": 20 if rapido else 100, "min_samples_leaf": 5 if rapido else 50,
                "max_depth": 16, "max_features": "sqrt", "max_samples": 0.15, "n_jobs": -1}
        final = [RandomForestClassifier(**{**base, **parametros}, random_state=semilla)]
    elif nombre == "gbm":
        base = dict(PARAMETROS_GBM)
        if rapido:
            base.update(max_leaf_nodes=15, min_samples_leaf=20)
        base.update(parametros)
        if monotono and variables is not None and any(restricciones(variables)):
            base["monotonic_cst"] = restricciones(variables)
        iteraciones = n_iter or (60 if rapido else ITER_GBM)
        final = [HistGradientBoostingClassifier(**base, max_iter=iteraciones, early_stopping=False,
                                                random_state=semilla)]
    else:
        raise ValueError(f"Modelo desconocido: {nombre}. Opciones: {NOMBRES}")
    return make_pipeline(ColumnasVacias(), *final)


class ModeloRiesgo:
    """Estimador entrenado más lo necesario para devolver probabilidades reales por celda."""

    def __init__(self, nombre, regimen, estimador, columnas, variables, tasa_neg, calibrador=None, info=None):
        self.nombre = nombre
        self.regimen = regimen
        self.estimador = estimador
        self.columnas = columnas          # posición de sus variables en la matriz operativa
        self.variables = variables
        self.tasa_neg = tasa_neg          # fracción de negativos que se conservó al muestrear
        self.calibrador = calibrador
        self.info = info or {}

    def _entrada(self, X_operativo):
        if X_operativo.shape[1] == len(self.columnas):     # ya viene solo con sus variables
            return X_operativo
        return X_operativo[:, self.columnas]

    def puntaje(self, X_operativo):
        """Probabilidad corregida al prior real (sin calibrar)."""
        p = self.estimador.predict_proba(self._entrada(X_operativo))[:, 1]
        return corregir_prior(p, self.tasa_neg).astype(np.float32)

    def probabilidad(self, X_operativo):
        """Probabilidad calibrada de que la celda tenga al menos un foco en la semana."""
        p = self.puntaje(X_operativo)
        return p if self.calibrador is None else self.calibrador.aplicar(p).astype(np.float32)


def _filas(muestra, filas):
    return slice(None) if filas is None else filas


def _columnas(ctx, regimen, excluir):
    """Columnas (en la matriz operativa) y nombres de las variables del régimen, menos las excluidas."""
    variables = ctx.constructor.nombres(regimen)
    excluir = list(excluir or [])
    desconocidas = [n for n in excluir if n not in variables]
    if desconocidas:
        raise ValueError(f"Variables a excluir que no existen en el régimen {regimen}: {desconocidas}")
    conservar = [i for i, n in enumerate(variables) if n not in excluir]
    return columnas_de(ctx, regimen)[conservar], [variables[i] for i in conservar]


def entrenar(nombre, regimen, muestra, ctx, filas=None, rapido=False, semilla=config.SEMILLA,
             n_iter=None, excluir=None, **parametros):
    """Entrena un modelo sobre la muestra caso-control (o sobre las filas indicadas)."""
    columnas, variables = _columnas(ctx, regimen, excluir)
    sel = _filas(muestra, filas)
    X = muestra["X"][sel][:, columnas]
    y = muestra["y"][sel]
    estimador = crear(nombre, semilla=semilla, rapido=rapido, n_iter=n_iter, variables=variables, **parametros)
    estimador.fit(X, y)
    return ModeloRiesgo(nombre, regimen, estimador, columnas, variables,
                        muestra["tasa_neg"],
                        info={"filas": int(len(y)), "positivos": int(y.sum()), "parametros": parametros,
                              "n_iter": n_iter, "excluidas": list(excluir or []),
                              "huella": ctx.notas.get("huella")})


def iteraciones_con_parada(regimen, muestra, ctx, filas_entrena, filas_valida,
                           semilla=config.SEMILLA, verboso=True, excluir=None, **parametros):
    """Número de árboles del GBM elegido con un conjunto de validación POSTERIOR en el tiempo.

    (La parada temprana interna de scikit-learn parte los datos al azar, y con datos
    espacio-temporales eso deja pasar información del futuro.)
    """
    columnas, variables = _columnas(ctx, regimen, excluir)
    vacias = ColumnasVacias()
    X_e = vacias.fit_transform(muestra["X"][filas_entrena][:, columnas])
    X_v = vacias.transform(muestra["X"][filas_valida][:, columnas])
    y_e, y_v = muestra["y"][filas_entrena], muestra["y"][filas_valida]
    monotono = parametros.pop("monotono", True)
    opciones = {**PARAMETROS_GBM, **parametros}
    if monotono and any(restricciones(variables)):
        opciones["monotonic_cst"] = restricciones(variables)
    modelo = HistGradientBoostingClassifier(**opciones, max_iter=PASO_PARADA,
                                            warm_start=True, early_stopping=False, random_state=semilla)
    historial, mejor, mejor_iter, sin_mejora = [], np.inf, PASO_PARADA, 0
    for n in range(PASO_PARADA, ITER_MAX + 1, PASO_PARADA):
        modelo.set_params(max_iter=n)
        modelo.fit(X_e, y_e)
        p = modelo.predict_proba(X_v)[:, 1]
        perdida = float(log_loss(y_v, p))
        historial.append({"n_iter": n, "perdida": perdida,
                          "pr_auc_muestra": float(average_precision_score(y_v, p))})
        if verboso:
            print(f"    {n:5d} árboles  pérdida {perdida:.5f}  PR-AUC(muestra) {historial[-1]['pr_auc_muestra']:.4f}")
        if perdida < mejor - 1e-5:
            mejor, mejor_iter, sin_mejora = perdida, n, 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    return mejor_iter, historial


class ModeloProvincia:
    """Línea base: el enfoque anterior (clima de la provincia + época + provincia)."""

    nombre = "provincia"
    regimen = "provincia"

    def __init__(self, estimador, tasa_neg, calibrador=None):
        self.estimador = estimador
        self.tasa_neg = tasa_neg
        self.calibrador = calibrador

    def puntaje_semana(self, ctx, semana):
        p = self.estimador.predict_proba(matriz_provincia(ctx, semana))[:, 1]
        return corregir_prior(p, self.tasa_neg).astype(np.float32)


def entrenar_provincia(muestra, ctx, filas=None, rapido=False, semilla=config.SEMILLA):
    sel = _filas(muestra, filas)
    semanas, fila, y = muestra["semana"][sel], muestra["fila"][sel], muestra["y"][sel]
    X = np.empty((len(y), matriz_provincia(ctx, int(semanas[0]), fila[:1]).shape[1]), dtype=np.float32)
    for s in np.unique(semanas):
        de_la_semana = semanas == s
        X[de_la_semana] = matriz_provincia(ctx, int(s), fila[de_la_semana])
    estimador = make_pipeline(ColumnasVacias(), HistGradientBoostingClassifier(
        learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=50, max_iter=60 if rapido else 200,
        early_stopping=False, random_state=semilla))
    estimador.fit(X, y)
    return ModeloProvincia(estimador, muestra["tasa_neg"])
