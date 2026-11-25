import numpy as np
import pandas as pd
import pytest

from incendios import clima, tiempo, variables
from incendios.malla import Malla
from incendios.rasters import CLASES_WORLDCOVER, VARS_TERRENO

NF, NC = 14, 16


def mundo(positivos, mascara=None, n_semanas=None, capas=None):
    """Mundo sintético pequeño: malla 14×16, clima de dos series 2012–2018."""
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=NC, nfilas=NF)
    rng = np.random.default_rng(0)
    est = {f"lc_{n}": rng.random((NF, NC)).astype(np.float32) for n in CLASES_WORLDCOVER.values()}
    total = sum(est.values())
    est = {k: v / total for k, v in est.items()}
    for n in VARS_TERRENO:
        est[n] = rng.random((NF, NC)).astype(np.float32)
    est["elev_media"][:, :3] = np.nan                       # zona sin relieve
    if mascara is None:
        mascara = np.ones((NF, NC), dtype=bool)
    fechas = pd.date_range("2012-01-01", "2018-12-31")
    partes = []
    for nombre, lat, lon, tmax in [("A", -17.02, -65.99, 22.0), ("B", -17.06, -65.93, 30.0)]:
        r = np.random.default_rng(len(nombre) + int(tmax))
        partes.append(pd.DataFrame({
            "provincia": nombre, "lat": lat, "lon": lon, "fecha": fechas,
            "temp_max": tmax + r.normal(0, 2, len(fechas)), "temp_min": 8.0,
            "humedad_rel": r.uniform(20, 80, len(fechas)), "viento_vel": 2.0,
            "precipitacion": np.where(r.random(len(fechas)) < 0.2, 5.0, 0.0),
            "humedad_suelo_sup": 0.4, "humedad_suelo_raiz": 0.5}))
    cs = clima.ClimaSemanal.desde_tabla(pd.concat(partes, ignore_index=True), anios_normales=(2012, 2014))
    lon_c, lat_c = malla.centros()
    LON, LAT = np.meshgrid(lon_c, lat_c)
    pesos = clima.pesos_idw(LON[mascara], LAT[mascara], cs.tabla.lon_c, cs.tabla.lat_c, k=2)
    if n_semanas is None:
        n_semanas = tiempo.semana_de_fecha("2018-12-30") + 1
    pos = pd.DataFrame(positivos, columns=["semana", "celda"])
    if capas is None:
        return variables.Constructor(malla, mascara, est, pos, cs, pesos, n_semanas)
    return variables.Constructor(malla, mascara, est, pos, cs, pesos, n_semanas, capas=capas)


def sem(anio, semana):
    return tiempo.semana_de_iso(anio, semana)


def celda(fila, col):
    return fila * NC + col


def columna(c, regimen, nombre, semana):
    return c.matriz(semana, regimen)[:, c.nombres(regimen).index(nombre)]


def test_los_nombres_coinciden_con_el_ancho_de_la_matriz():
    c = mundo([[sem(2016, 30), celda(5, 5)]])
    for regimen in variables.REGIMENES:
        m = c.matriz(sem(2017, 30), regimen)
        assert m.dtype == np.float32
        assert m.shape == (NF * NC, len(c.nombres(regimen)))
        assert len(set(c.nombres(regimen))) == len(c.nombres(regimen))


def test_estacional_es_subconjunto_de_operativo_y_sin_coordenadas():
    c = mundo([])
    est, ope = c.nombres("estacional"), c.nombres("operativo")
    assert set(est) < set(ope)
    assert ope[:len(est)] == est
    for nombre in ope:
        assert "lat" not in nombre and "lon" not in nombre and "prov" not in nombre and "muni" not in nombre


def test_los_grupos_reparten_todas_las_variables():
    c = mundo([])
    for regimen in variables.REGIMENES:
        grupos = c.grupos(regimen)
        juntas = [n for nombres in grupos.values() for n in nombres]
        assert sorted(juntas) == sorted(c.nombres(regimen))
    assert "fuego_reciente" in c.grupos("operativo") and "fuego_reciente" not in c.grupos("estacional")


def test_etiquetas_marcan_las_celdas_con_fuego_esa_semana():
    mascara = np.ones((NF, NC), dtype=bool)
    mascara[0, :] = False                                    # la primera fila no es del departamento
    t = sem(2017, 30)
    c = mundo([[t, celda(0, 2)], [t, celda(3, 4)], [t + 1, celda(6, 6)]], mascara=mascara)
    y = c.etiquetas(t)
    assert y.shape == (mascara.sum(),) and y.dtype == np.uint8
    assert y.sum() == 1
    assert y[np.flatnonzero(mascara.ravel()) == celda(3, 4)][0] == 1


def test_operativo_no_cambia_si_cambian_focos_de_la_semana_objetivo_o_posteriores():
    t = sem(2017, 36)
    base = [[sem(2016, 36), celda(4, 4)], [t - 1, celda(7, 7)], [t - 3, celda(2, 9)]]
    futuro = [[t, celda(f, col)] for f in range(NF) for col in range(NC)]
    futuro += [[t + 1, celda(5, 5)], [t + 20, celda(7, 7)]]
    a, b = mundo(base), mundo(base + futuro)
    for regimen in variables.REGIMENES:
        assert np.array_equal(a.matriz(t, regimen), b.matriz(t, regimen), equal_nan=True)


def test_estacional_no_cambia_con_ningun_foco_del_mismo_anio():
    t = sem(2017, 36)
    base = [[sem(2016, 36), celda(4, 4)], [sem(2015, 20), celda(9, 9)]]
    mismo_anio = [[sem(2017, s), celda(4, 4)] for s in range(1, 53)] + [[t - 1, celda(8, 8)]]
    a, b = mundo(base), mundo(base + mismo_anio)
    assert np.array_equal(a.matriz(t, "estacional"), b.matriz(t, "estacional"), equal_nan=True)
    assert not np.array_equal(a.matriz(t, "operativo"), b.matriz(t, "operativo"), equal_nan=True)


def test_un_foco_la_semana_anterior_sube_la_actividad_reciente():
    t = sem(2017, 36)
    c = mundo([[t - 1, celda(7, 7)]])
    rec_c1 = columna(c, "operativo", "rec_c1", t)
    assert rec_c1[celda(7, 7)] == 1.0 and rec_c1.sum() == 1.0
    v3 = columna(c, "operativo", "rec_v3_1", t).reshape(NF, NC)
    assert (v3[6:9, 6:9] == 1.0).all() and v3.sum() == 9.0
    assert columna(c, "operativo", "rec_c4", t)[celda(7, 7)] == 1.0
    assert columna(c, "operativo", "rec_c1", t + 1)[celda(7, 7)] == 0.0
    assert columna(c, "operativo", "rec_c4", t + 3)[celda(7, 7)] == 1.0
    assert columna(c, "operativo", "rec_c4", t + 4)[celda(7, 7)] == 0.0
    assert columna(c, "operativo", "rec_dep_1", t)[0] == 1.0


def test_recurrencia_estacional_cuenta_la_misma_epoca_de_anios_anteriores():
    c = mundo([[sem(2016, 36), celda(4, 4)], [sem(2016, 10), celda(9, 9)]])
    t = sem(2017, 36)
    est_c3 = columna(c, "estacional", "est_c3", t)
    assert est_c3[celda(4, 4)] == 1.0        # mismo momento del año anterior
    assert est_c3[celda(9, 9)] == 0.0        # fue en otra época
    his_c1 = columna(c, "estacional", "his_c1", t)
    assert his_c1[celda(4, 4)] == 1.0 and his_c1[celda(9, 9)] == 1.0
    assert columna(c, "estacional", "est_c3", sem(2017, 39))[celda(4, 4)] == 1.0   # ±3 semanas
    assert columna(c, "estacional", "est_c3", sem(2017, 40))[celda(4, 4)] == 0.0
    assert columna(c, "estacional", "est_c3", sem(2020 - 1, 36))[celda(4, 4)] == 1.0   # 2016 aún entra en 2019
    assert columna(c, "estacional", "his_anios3", t)[celda(4, 4)] == 1.0


def test_el_historial_de_tres_anios_olvida_lo_mas_antiguo_pero_el_decaimiento_no():
    c = mundo([[sem(2014, 36), celda(4, 4)]], n_semanas=tiempo.semana_de_fecha("2018-12-30") + 1)
    assert columna(c, "estacional", "his_c3", sem(2017, 36))[celda(4, 4)] == 1.0
    assert columna(c, "estacional", "his_c3", sem(2018, 36))[celda(4, 4)] == 0.0
    dec = columna(c, "estacional", "his_dec", sem(2018, 36))[celda(4, 4)]
    assert dec == pytest.approx(0.6 ** 3)


def test_la_semana_53_no_desfasa_la_recurrencia():
    # 2015 tiene 53 semanas ISO
    c = mundo([[sem(2015, 53), celda(4, 4)], [sem(2015, 30), celda(9, 9)]])
    assert columna(c, "estacional", "est_c3", sem(2016, 52))[celda(4, 4)] == 1.0
    assert columna(c, "estacional", "est_c3", sem(2016, 1))[celda(4, 4)] == 0.0
    # tras un año de 53 semanas, la semana 30 sigue alineada con la semana 30
    assert columna(c, "estacional", "est_c3", sem(2016, 30))[celda(9, 9)] == 1.0
    assert columna(c, "estacional", "est_c3", sem(2017, 30))[celda(9, 9)] == 1.0


def test_sin_tres_anios_de_historial_no_hay_matriz():
    c = mundo([])
    assert not c.tiene_historial(sem(2014, 30))
    assert c.tiene_historial(sem(2015, 1))
    with pytest.raises(ValueError, match="historial"):
        c.matriz(sem(2014, 30), "estacional")


def test_estacional_para_un_anio_sin_el_anterior_completo_es_error_salvo_que_se_fije_el_historial():
    c = mundo([[sem(2018, 36), celda(4, 4)]])     # datos hasta fin de 2018
    assert c.anio_ultimo_completo == 2018
    c.matriz(sem(2019, 36), "estacional")          # 2019 sí: el historial 2016–2018 está completo
    with pytest.raises(ValueError, match="historial"):
        c.matriz(sem(2021, 36), "estacional")
    m = c.matriz(sem(2021, 36), "estacional", anio_historial=2019)
    assert m[celda(4, 4), c.nombres("estacional").index("his_c1")] == 1.0


def test_operativo_sin_clima_observado_es_error():
    c = mundo([])
    with pytest.raises(ValueError):
        c.matriz(tiempo.semana_de_fecha("2019-03-04"), "operativo")


def test_las_estaticas_faltantes_siguen_siendo_nan():
    c = mundo([])
    elev = columna(c, "estacional", "elev_media", sem(2017, 30)).reshape(NF, NC)
    assert np.isnan(elev[:, :3]).all() and np.isfinite(elev[:, 3:]).all()


def test_el_clima_llega_interpolado_a_cada_celda():
    c = mundo([])
    t = sem(2017, 36)
    tmax = columna(c, "operativo", "tmax_7", t)
    assert np.isfinite(tmax).all()
    assert 20.0 < tmax.min() < tmax.max() < 32.0
    normal = columna(c, "estacional", "n_tmax_anual", t)
    assert normal.min() >= 21.0 and normal.max() <= 31.0


def test_la_anomalia_departamental_no_lee_semanas_fuera_del_registro():
    # Para las primeras semanas de 2015 la referencia de "tres años antes" cae en 2011,
    # antes del registro: no debe tomar valores del final del arreglo.
    t = sem(2015, 1)
    ultimas = [[tiempo.semana_de_fecha("2018-12-24"), celda(f, c)] for f in range(NF) for c in range(NC)]
    ultimas += [[tiempo.semana_de_fecha("2018-12-17"), celda(f, c)] for f in range(NF) for c in range(NC)]
    a, b = mundo([]), mundo(ultimas)
    assert np.array_equal(a.matriz(t, "operativo"), b.matriz(t, "operativo"), equal_nan=True)
    assert np.array_equal(a.matriz(t + 2, "operativo"), b.matriz(t + 2, "operativo"), equal_nan=True)


def test_un_hueco_de_clima_en_una_serie_no_contagia_a_las_celdas_de_la_otra():
    c = mundo([])
    t = sem(2017, 36)
    i = clima.VARS_ANTECEDENTE.index("suelo_sup_7")
    c.clima.valores[0, :, clima.VARS_CLIMA.index("suelo_sup_7")] = np.nan      # la serie 0 se queda sin dato
    c.pesos[:] = 0.0
    c.pesos[: c.n // 2, 0] = 1.0            # la primera mitad de las celdas solo ve la serie 0
    c.pesos[c.n // 2:, 0] = 0.3             # la otra mitad mezcla ambas
    c.pesos[c.n // 2:, 1] = 0.7
    suelo = columna(c, "operativo", "suelo_sup_7", t)
    assert np.isnan(suelo[: c.n // 2]).all()
    assert np.allclose(suelo[c.n // 2:], 0.4, atol=1e-5)     # se renormaliza con la serie que sí tiene dato


def test_malla_fuego_devuelve_la_presencia_de_fuego_en_toda_la_malla():
    mascara = np.ones((NF, NC), dtype=bool)
    mascara[0, :] = False
    t = sem(2017, 30)
    c = mundo([[t, celda(0, 2)], [t, celda(3, 4)], [t + 1, celda(6, 6)]], mascara=mascara)
    m = c.malla_fuego(t)
    assert m.shape == (NF, NC) and m.dtype == bool
    assert m[0, 2] and m[3, 4] and m.sum() == 2          # incluye celdas fuera del departamento
    assert not c.malla_fuego(10_000).any()                # semana sin datos


def test_el_constructor_da_lo_mismo_con_pesos_dispersos():
    from scipy import sparse
    c = mundo([[sem(2016, 30), celda(5, 5)]])
    s = sem(2017, 30)
    densa = c.matriz(s, "operativo")
    c.pesos = sparse.csr_matrix(c.pesos)
    assert np.allclose(c.matriz(s, "operativo"), densa, equal_nan=True, rtol=1e-5, atol=1e-4)   # orden de sumas


# ------------------------------------------------------------------ fuego de los últimos días
def mundo_con_dias(dias_celdas):
    """Mundo con focos por día: [(día desde el lunes de la semana 0, celda)]; las semanas salen de los días."""
    base = mundo([[d // 7, c] for d, c in dias_celdas])
    est = {f"lc_{n}": np.full((NF, NC), 0.1, np.float32) for n in CLASES_WORLDCOVER.values()}
    est.update({n: np.zeros((NF, NC), np.float32) for n in VARS_TERRENO})
    pos = pd.DataFrame({"semana": [d // 7 for d, _ in dias_celdas], "celda": [c for _, c in dias_celdas]},
                       columns=["semana", "celda"]).drop_duplicates()
    diarios = pd.DataFrame(dias_celdas, columns=["dia", "celda"])
    return variables.Constructor(base.malla, base.mascara, est, pos, base.clima, base.pesos, base.n_semanas,
                                 diarios=diarios)


def test_sin_focos_por_dia_no_hay_fuego_de_los_ultimos_dias():
    c = mundo([])
    assert "fuego_dias" not in c.grupos("operativo")
    assert not set(variables.FUEGO_DIAS) & set(c.nombres("operativo"))


def test_el_fuego_de_los_ultimos_dias_va_al_final_y_solo_en_el_operativo():
    c = mundo_con_dias([])
    ope, est = c.nombres("operativo"), c.nombres("estacional")
    assert ope[-len(variables.FUEGO_DIAS):] == variables.FUEGO_DIAS
    assert not set(variables.FUEGO_DIAS) & set(est) and ope[:len(est)] == est
    assert c.matriz(sem(2017, 36), "operativo").shape == (NF * NC, len(ope))
    assert c.matriz(sem(2017, 36), "estacional").shape == (NF * NC, len(est))


def test_cuenta_el_fuego_de_viernes_a_domingo_y_del_domingo():
    t = sem(2017, 36)
    lunes = 7 * t
    c = mundo_con_dias([(lunes - 1, celda(12, 14)), (lunes - 3, celda(1, 1)), (lunes - 4, celda(9, 9))])
    d3 = columna(c, "operativo", "rec_d3_c", t)
    assert d3[celda(12, 14)] == 1 and d3[celda(1, 1)] == 1 and d3[celda(9, 9)] == 0      # el jueves no entra
    domingo = columna(c, "operativo", "rec_d1_v11", t)
    assert domingo[celda(12, 14)] == 1 and domingo[celda(1, 1)] == 0                      # (1,1) ardió el viernes
    assert columna(c, "operativo", "rec_d3_v3", t)[celda(2, 2)] == 1                      # vecina del viernes
    assert columna(c, "operativo", "rec_d3_v11", t)[celda(6, 6)] == 1                     # a 5 celdas del viernes
    assert columna(c, "operativo", "rec_d3_v41", t)[celda(0, 0)] == 2                     # en ~20 km, las dos
    assert (columna(c, "operativo", "rec_dep_d3", t) == 2).all()
    assert columna(c, "operativo", "rec_d3_c", t + 1).sum() == 0                          # la semana siguiente, nada


def test_el_fuego_de_los_ultimos_dias_no_mira_el_lunes_ni_despues():
    t = sem(2017, 36)
    lunes = 7 * t
    pasado = [(lunes - 2, celda(4, 4)), (lunes - 10, celda(7, 7))]
    futuro = [(lunes + k, celda(f, 3)) for k in range(0, 9) for f in range(NF)]
    a, b = mundo_con_dias(pasado), mundo_con_dias(pasado + futuro)
    for regimen in variables.REGIMENES:
        assert np.array_equal(a.matriz(t, regimen), b.matriz(t, regimen), equal_nan=True)
