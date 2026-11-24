import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from incendios import muestreo


def test_muestra_todos_los_positivos_y_n_negativos_sin_repetir():
    rng = np.random.default_rng(0)
    y = np.zeros(1000, dtype=np.uint8)
    y[[3, 500, 999]] = 1
    idx = muestreo.muestrear(y, 50, rng)
    assert len(idx) == 53 and len(np.unique(idx)) == 53
    assert {3, 500, 999} <= set(idx.tolist())
    assert (y[idx] == 0).sum() == 50
    assert np.array_equal(idx, np.sort(idx))


def test_la_misma_semilla_da_la_misma_muestra():
    y = np.zeros(1000, dtype=np.uint8)
    a = muestreo.muestrear(y, 50, np.random.default_rng(7))
    b = muestreo.muestrear(y, 50, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_si_se_piden_mas_negativos_de_los_que_hay_devuelve_todos():
    y = np.array([1, 0, 0, 1], dtype=np.uint8)
    assert muestreo.muestrear(y, 10, np.random.default_rng(0)).tolist() == [0, 1, 2, 3]


def test_corregir_prior_escala_las_odds_por_la_tasa_de_muestreo():
    p = muestreo.corregir_prior(np.array([0.0, 0.5, 1.0]), 0.01)
    assert p[0] == 0.0 and p[2] == 1.0
    assert p[1] == pytest.approx(0.01 / 1.01)


def test_corregir_prior_es_monotona():
    p = np.linspace(0, 1, 101)
    q = muestreo.corregir_prior(p, 0.03)
    assert (np.diff(q) > 0).all() and (q <= p + 1e-12).all()


def test_entrenar_en_muestra_y_corregir_recupera_la_tasa_real():
    rng = np.random.default_rng(1)
    n = 200_000
    y = (rng.random(n) < 0.01).astype(np.uint8)
    x = rng.normal(0, 1, n) + 2.0 * y
    idx = muestreo.muestrear(y, int(y.sum()), rng)                 # muestra 1:1
    tasa = int(y.sum()) / int((y == 0).sum())
    modelo = LogisticRegression().fit(x[idx, None], y[idx])
    p_muestra = modelo.predict_proba(x[:, None])[:, 1]
    assert p_muestra.mean() > 0.2                                   # sin corregir sobrestima muchísimo
    assert muestreo.corregir_prior(p_muestra, tasa).mean() == pytest.approx(0.01, rel=0.1)


def test_calibrador_ajusta_puntajes_mal_calibrados():
    rng = np.random.default_rng(2)
    p = rng.random(300_000)
    y = (rng.random(len(p)) < p ** 2).astype(np.uint8)              # la frecuencia real es p al cuadrado
    cal = muestreo.Calibrador().ajustar(p, y)
    q = cal.aplicar(p)
    assert q.min() >= 0.0 and q.max() <= 1.0
    for lo, hi in [(0.2, 0.3), (0.5, 0.6), (0.8, 0.9)]:
        tramo = (p >= lo) & (p < hi)
        assert q[tramo].mean() == pytest.approx(y[tramo].mean(), abs=0.02)
    orden = np.argsort(p)
    assert (np.diff(q[orden]) >= -1e-12).all()                      # no decrece
    medio = np.linspace(0.2, 0.8, 1000)
    assert len(np.unique(cal.aplicar(medio))) > 500                 # sin grandes escalones: conserva el orden


def test_calibrador_sin_ajustar_es_error():
    with pytest.raises(RuntimeError):
        muestreo.Calibrador().aplicar(np.array([0.1]))
