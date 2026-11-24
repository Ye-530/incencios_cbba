"""Muestreo caso-control (todos los positivos, una muestra de negativos) y regreso a probabilidades reales."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def muestrear(y, n_neg, rng):
    """Índices de todos los positivos más `n_neg` negativos elegidos al azar sin reemplazo."""
    y = np.asarray(y)
    positivos = np.flatnonzero(y == 1)
    negativos = np.flatnonzero(y == 0)
    if n_neg < len(negativos):
        negativos = rng.choice(negativos, size=n_neg, replace=False)
    return np.sort(np.concatenate([positivos, negativos]))


def corregir_prior(p, tasa_neg):
    """Pasa la probabilidad aprendida en la muestra a la de la población.

    Si se conservaron todos los positivos y una fracción `tasa_neg` de los negativos,
    las odds reales son las de la muestra multiplicadas por `tasa_neg` (King y Zeng, 2001).
    """
    p = np.asarray(p, dtype=np.float64)
    return p * tasa_neg / (p * tasa_neg + (1.0 - p))


class Calibrador:
    """Regresión isotónica centrada: monótona y sin grandes escalones.

    La isotónica clásica devuelve tramos planos (muchos empates). Aquí cada tramo se
    reduce a su punto medio y se interpola entre tramos, con lo que el orden de las
    celdas se conserva casi siempre (Oron y Flournoy, 2017).
    """

    MAX_GRUPOS = 20_000

    def __init__(self):
        self._x = None
        self._y = None

    def ajustar(self, p, y):
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        orden = np.argsort(p, kind="stable")
        p, y = p[orden], y[orden]
        # agrupar en bloques de igual tamaño para que el ajuste sea rápido con millones de filas
        n_grupos = min(self.MAX_GRUPOS, len(p))
        cortes = np.linspace(0, len(p), n_grupos + 1).astype(np.int64)
        peso = np.diff(cortes).astype(np.float64)
        px = np.add.reduceat(p, cortes[:-1]) / peso
        py = np.add.reduceat(y, cortes[:-1]) / peso
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        ajuste = iso.fit(px, py, sample_weight=peso).predict(px)
        # un nudo por tramo plano, en el centro (ponderado) del tramo
        inicio = np.flatnonzero(np.r_[True, np.diff(ajuste) > 0])
        suma_peso = np.add.reduceat(peso, inicio)
        self._x = np.add.reduceat(px * peso, inicio) / suma_peso
        self._y = ajuste[inicio]
        return self

    def aplicar(self, p):
        if self._x is None:
            raise RuntimeError("El calibrador no está ajustado")
        return np.interp(np.asarray(p, dtype=np.float64), self._x, self._y)
