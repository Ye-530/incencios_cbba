import numpy as np
import pytest

from incendios import fwi


def test_caso_de_referencia_de_van_wagner():
    # Primer día de la tabla de prueba de Van Wagner y Pickett (1985): abril, 46° N,
    # códigos previos 85 / 6 / 15.
    f = fwi.ffmc(17.0, 42.0, 25.0, 0.0, 85.0)
    p = fwi.dmc(17.0, 42.0, 0.0, 6.0, mes=4, lat=46.0)
    d = fwi.dc(17.0, 0.0, 15.0, mes=4, lat=46.0)
    r = fwi.isi(25.0, f)
    u = fwi.bui(p, d)
    s = fwi.fwi(r, u)
    assert f == pytest.approx(87.69, abs=0.01)
    assert p == pytest.approx(8.55, abs=0.01)
    assert d == pytest.approx(19.01, abs=0.01)
    assert r == pytest.approx(10.85, abs=0.01)
    assert u == pytest.approx(8.49, abs=0.01)
    assert s == pytest.approx(10.10, abs=0.01)


def test_una_lluvia_fuerte_baja_los_tres_codigos_de_humedad():
    # mismo día con y sin 20 mm de lluvia (el umbral no viene de la literatura: basta que baje claramente)
    assert fwi.ffmc(17.0, 42.0, 25.0, 20.0, 85.0) < fwi.ffmc(17.0, 42.0, 25.0, 0.0, 85.0) - 15.0
    assert fwi.dmc(17.0, 42.0, 20.0, 30.0, mes=9, lat=-17.0) < 30.0
    assert fwi.dc(17.0, 20.0, 300.0, mes=9, lat=-17.0) < 300.0


def test_lluvia_bajo_el_umbral_no_moja_las_capas_profundas():
    seco = fwi.dmc(20.0, 50.0, 0.0, 10.0, mes=6, lat=-17.0)
    assert fwi.dmc(20.0, 50.0, 1.5, 10.0, mes=6, lat=-17.0) == seco
    assert fwi.dc(20.0, 2.8, 100.0, mes=6, lat=-17.0) == fwi.dc(20.0, 0.0, 100.0, mes=6, lat=-17.0)


def test_factor_de_duracion_del_dia_para_cochabamba():
    # DMC: tabla de 20° S de cffdrs, junio = 7,8 h
    assert fwi.dmc(20.0, 50.0, 0.0, 10.0, mes=6, lat=-17.0) == pytest.approx(10.0 + 1.894 * 21.1 * 50 * 7.8e-4, rel=1e-9)
    # DC: entre 20° S y 20° N el factor es 1,4 todos los meses
    esperado = 100.0 + (0.36 * (20.0 + 2.8) + 1.4) / 2
    assert fwi.dc(20.0, 0.0, 100.0, mes=1, lat=-17.0) == pytest.approx(esperado, rel=1e-9)
    assert fwi.dc(20.0, 0.0, 100.0, mes=7, lat=-17.0) == pytest.approx(esperado, rel=1e-9)


def test_bui_es_cero_si_ambos_codigos_son_cero():
    assert fwi.bui(0.0, 0.0) == 0.0


def test_mas_viento_mas_propagacion():
    assert fwi.isi(40.0, 90.0) > fwi.isi(5.0, 90.0)


def test_la_serie_coincide_con_el_calculo_dia_a_dia():
    t = np.array([17.0, 20.0, 8.5])
    hr = np.array([42.0, 21.0, 40.0])
    v = np.array([25.0, 25.0, 17.0])
    ll = np.array([0.0, 2.4, 0.0])
    s = fwi.serie_fwi(t, hr, v, ll, meses=np.array([4, 4, 4]), lat=46.0)
    assert set(s) == {"ffmc", "dmc", "dc", "isi", "bui", "fwi"}
    assert s["ffmc"][0] == pytest.approx(87.69, abs=0.01)
    f2 = fwi.ffmc(20.0, 21.0, 25.0, 2.4, s["ffmc"][0])
    assert s["ffmc"][1] == pytest.approx(f2)
    assert s["dc"][2] == pytest.approx(fwi.dc(8.5, 0.0, s["dc"][1], mes=4, lat=46.0))
    assert all(len(v) == 3 for v in s.values())


def test_la_serie_no_produce_nan_ni_negativos_con_entradas_extremas():
    rng = np.random.default_rng(0)
    n = 2000
    t = rng.uniform(-10, 42, n)
    hr = rng.uniform(-5, 110, n)           # fuera de rango a propósito
    v = rng.uniform(-2, 80, n)
    ll = np.where(rng.random(n) < 0.2, rng.uniform(0, 120, n), 0.0)
    meses = rng.integers(1, 13, n)
    s = fwi.serie_fwi(t, hr, v, ll, meses, lat=-17.0)
    for nombre, valores in s.items():
        assert np.isfinite(valores).all(), nombre
        assert (valores >= 0).all(), nombre
    assert s["ffmc"].max() <= 101.0
