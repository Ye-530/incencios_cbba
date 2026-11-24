import numpy as np
import pytest

from incendios import metricas


def semanas_de(n_semanas, n_celdas):
    return np.repeat(np.arange(n_semanas), n_celdas)


def test_prediccion_perfecta_captura_todo_con_la_fraccion_igual_a_la_tasa_base():
    rng = np.random.default_rng(0)
    semanas = semanas_de(20, 1000)
    y = np.zeros(20_000, dtype=np.uint8)
    for s in range(20):
        y[s * 1000 + rng.choice(1000, 50, replace=False)] = 1      # 5 % cada semana
    c = metricas.captura_por_semana(y, y.astype(float), semanas, fracciones=(0.01, 0.05, 0.10))
    assert c[0.05] == pytest.approx(1.0)
    assert c[0.10] == pytest.approx(1.0)
    assert c[0.01] == pytest.approx(0.2)


def test_prediccion_al_azar_captura_mas_o_menos_la_fraccion():
    rng = np.random.default_rng(1)
    semanas = semanas_de(50, 2000)
    y = (rng.random(100_000) < 0.02).astype(np.uint8)
    c = metricas.captura_por_semana(y, rng.random(100_000), semanas, fracciones=(0.10, 0.50))
    assert c[0.10] == pytest.approx(0.10, abs=0.02)
    assert c[0.50] == pytest.approx(0.50, abs=0.03)


def test_puntajes_todos_iguales_dan_exactamente_la_fraccion():
    semanas = semanas_de(4, 500)
    y = np.zeros(2000, dtype=np.uint8)
    y[::7] = 1
    c = metricas.captura_por_semana(y, np.full(2000, 0.3), semanas, fracciones=(0.05, 0.20))
    assert c[0.05] == pytest.approx(0.05)
    assert c[0.20] == pytest.approx(0.20)


def test_el_ranking_es_dentro_de_cada_semana():
    # semana 0 tiene puntajes altos sin fuego; semana 1, puntajes bajos pero bien ordenados
    semanas = semanas_de(2, 100)
    p = np.r_[np.linspace(0.9, 1.0, 100), np.linspace(0.0, 0.1, 100)]
    y = np.zeros(200, dtype=np.uint8)
    y[195:] = 1                                   # los 5 más altos de la semana 1
    assert metricas.captura_por_semana(y, p, semanas, fracciones=(0.05,))[0.05] == pytest.approx(1.0)


def test_las_semanas_sin_fuego_no_afectan_la_captura():
    semanas = semanas_de(3, 100)
    p = np.tile(np.linspace(0, 1, 100), 3)
    y = np.zeros(300, dtype=np.uint8)
    y[95:100] = 1
    assert metricas.captura_por_semana(y, p, semanas, fracciones=(0.05,))[0.05] == pytest.approx(1.0)


def test_area_para_recall():
    rng = np.random.default_rng(3)
    semanas = semanas_de(20, 1000)
    y = np.zeros(20_000, dtype=np.uint8)
    for s in range(20):
        y[s * 1000 + rng.choice(1000, 50, replace=False)] = 1
    perfecta = metricas.area_para_recall(y, y.astype(float), semanas, recalls=(0.5, 1.0))
    assert perfecta[0.5] == pytest.approx(0.025, abs=0.001)
    assert perfecta[1.0] == pytest.approx(0.05, abs=0.001)
    iguales = metricas.area_para_recall(y, np.zeros(20_000), semanas, recalls=(0.5, 0.8))
    assert iguales[0.5] == pytest.approx(0.5, abs=0.002)
    assert iguales[0.8] == pytest.approx(0.8, abs=0.002)


def test_precision_en_el_top():
    semanas = semanas_de(2, 100)
    y = np.zeros(200, dtype=np.uint8)
    y[[0, 1, 100]] = 1
    p = y.astype(float)
    # top 2 % = 2 celdas por semana: semana 0 acierta 2 de 2; semana 1 acierta 1 de 2
    # (la segunda celda sale de un empate entre 99 celdas sin fuego)
    assert metricas.precision_top(y, p, semanas, 0.02) == pytest.approx(3 / 4)


def test_dilatar_etiquetas_marca_el_vecindario_de_3x3():
    m = np.zeros((6, 6), dtype=np.uint8)
    m[2, 2] = 1
    m[5, 5] = 1
    d = metricas.dilatar_etiquetas(m)
    assert d.dtype == np.uint8
    assert d[1:4, 1:4].all() and d[4:, 4:].all()
    assert d.sum() == 9 + 4


def test_resumen_incluye_las_metricas_principales():
    rng = np.random.default_rng(4)
    semanas = semanas_de(10, 500)
    y = (rng.random(5000) < 0.05).astype(np.uint8)
    p = np.clip(y * 0.5 + rng.random(5000) * 0.6, 0, 1)
    r = metricas.resumen(y, p, semanas)
    for clave in ("n", "n_pos", "tasa_base", "roc_auc", "pr_auc", "captura_1", "captura_5",
                  "captura_10", "area_50", "area_80", "precision_1"):
        assert clave in r
    assert r["n"] == 5000 and r["n_pos"] == int(y.sum())
    assert 0.5 < r["roc_auc"] <= 1.0
    assert r["pr_auc"] > r["tasa_base"]


def test_curva_de_captura_crece_de_cero_a_uno_y_coincide_con_captura_por_semana():
    rng = np.random.default_rng(5)
    semanas = semanas_de(8, 500)
    y = (rng.random(4000) < 0.04).astype(np.uint8)
    p = y * 0.3 + rng.random(4000)
    fracciones = np.array([0.0, 0.05, 0.2, 1.0])
    curva = metricas.curva_captura(y, p, semanas, fracciones)
    assert curva.shape == (4,)
    assert curva[0] == 0.0 and curva[-1] == pytest.approx(1.0)
    assert (np.diff(curva) >= 0).all()
    assert curva[1] == pytest.approx(metricas.captura_por_semana(y, p, semanas, fracciones=(0.05,))[0.05])
