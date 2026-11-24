"""Métricas para mapas de riesgo: qué parte del fuego cae en la fracción de celdas señalada.

Todas ordenan las celdas DENTRO de cada semana (un mapa por semana). Los empates
se resuelven en valor esperado, como si se desempataran al azar: así un modelo
que da el mismo puntaje a toda una provincia no gana ni pierde por el orden de
las celdas.
"""
import numpy as np
from scipy.ndimage import maximum_filter
from sklearn.metrics import average_precision_score, roc_auc_score

FRACCIONES = (0.01, 0.02, 0.05, 0.10, 0.20)
_REJILLA = np.unique(np.concatenate([np.linspace(0, 0.02, 201), np.linspace(0.02, 0.2, 361),
                                     np.linspace(0.2, 1.0, 401)]))


def positivos_esperados_en_top(y, p, ks):
    """Positivos esperados entre las `k` celdas de mayor puntaje (k puede ser fraccionario)."""
    orden = np.argsort(-np.asarray(p), kind="stable")
    ps = np.asarray(p)[orden]
    acumulado = np.r_[0.0, np.cumsum(np.asarray(y)[orden], dtype=np.float64)]
    # dentro de un grupo de empate, los positivos se reparten de forma uniforme
    limites = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1]) + 1, len(ps)]
    return np.interp(ks, limites, acumulado[limites])


def _por_semana(y, p, semanas):
    y, p, semanas = np.asarray(y), np.asarray(p), np.asarray(semanas)
    orden = np.argsort(semanas, kind="stable")
    cortes = np.flatnonzero(np.r_[True, semanas[orden][1:] != semanas[orden][:-1], True])
    for a, b in zip(cortes[:-1], cortes[1:]):
        idx = orden[a:b]
        yield y[idx], p[idx]


def _acumular(y, p, semanas, fracciones, solo_con_fuego):
    fracciones = np.asarray(fracciones, dtype=np.float64)
    aciertos = np.zeros(len(fracciones))
    positivos = 0.0
    senaladas = np.zeros(len(fracciones))
    for ys, ps in _por_semana(y, p, semanas):
        n_pos = float(ys.sum())
        if solo_con_fuego and n_pos == 0:
            continue
        ks = fracciones * len(ys)
        aciertos += positivos_esperados_en_top(ys, ps, ks)
        senaladas += ks
        positivos += n_pos
    return aciertos, positivos, senaladas


def _primer_cruce(captura, fracciones, objetivo):
    """Menor fracción cuya captura alcanza el objetivo (la curva puede tener mesetas)."""
    i = int(np.searchsorted(captura, objetivo - 1e-12, side="left"))
    if i >= len(captura):
        return float("nan")
    if i == 0:
        return float(fracciones[0])
    c0, c1 = captura[i - 1], captura[i]
    peso = (objetivo - c0) / (c1 - c0) if c1 > c0 else 1.0
    return float(fracciones[i - 1] + peso * (fracciones[i] - fracciones[i - 1]))


def captura_por_semana(y, p, semanas, fracciones=FRACCIONES):
    """Fracción de las celdas con fuego que cae en el top de cada semana, sumando todas las semanas."""
    aciertos, positivos, _ = _acumular(y, p, semanas, fracciones, solo_con_fuego=True)
    return {float(f): float(a / positivos) if positivos else float("nan")
            for f, a in zip(fracciones, aciertos)}


def area_para_recall(y, p, semanas, recalls=(0.5, 0.8)):
    """Fracción mínima del territorio que hay que señalar cada semana para capturar ese recall."""
    aciertos, positivos, _ = _acumular(y, p, semanas, _REJILLA, solo_con_fuego=True)
    captura = aciertos / positivos
    return {float(r): _primer_cruce(captura, _REJILLA, r) for r in recalls}


def precision_top(y, p, semanas, fraccion):
    """De las celdas señaladas (top `fraccion` de cada semana, todas las semanas), cuántas ardieron."""
    aciertos, _, senaladas = _acumular(y, p, semanas, [fraccion], solo_con_fuego=False)
    return float(aciertos[0] / senaladas[0])


def dilatar_etiquetas(y_malla):
    """Marca también las 8 celdas vecinas de cada celda con fuego (tolerancia de una celda)."""
    return maximum_filter(np.asarray(y_malla, dtype=np.uint8), size=3, mode="constant", cval=0)


def resumen(y, p, semanas):
    y, p = np.asarray(y), np.asarray(p)
    captura = captura_por_semana(y, p, semanas)
    area = area_para_recall(y, p, semanas)
    r = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "tasa_base": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "area_50": area[0.5],
        "area_80": area[0.8],
        "precision_1": precision_top(y, p, semanas, 0.01),
    }
    for f, c in captura.items():
        r[f"captura_{round(f * 100)}"] = c
    return r


def curva_captura(y, p, semanas, fracciones):
    """Captura para cada fracción del territorio señalada (para dibujar la curva)."""
    aciertos, positivos, _ = _acumular(y, p, semanas, fracciones, solo_con_fuego=True)
    return aciertos / positivos
