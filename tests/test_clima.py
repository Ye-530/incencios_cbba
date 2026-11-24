import numpy as np
import pandas as pd
import pytest

from incendios import clima, tiempo


def diario(provincia, lat, lon, inicio="2019-01-01", fin="2020-12-31", **cambios):
    fechas = pd.date_range(inicio, fin)
    n = len(fechas)
    base = dict(temp_max=22.0, temp_min=8.0, humedad_rel=50.0, viento_vel=2.0,
                precipitacion=1.0, humedad_suelo_sup=0.4, humedad_suelo_raiz=0.5)
    base.update(cambios)
    df = pd.DataFrame({"provincia": provincia, "lat": lat, "lon": lon, "fecha": fechas})
    for columna, valor in base.items():
        df[columna] = np.full(n, valor, dtype=float) if np.isscalar(valor) else valor
    return df


def test_limpiar_pone_en_cero_el_artefacto_de_lluvia_y_lo_informa():
    df = diario("Cercado", -17.43, -66.17, inicio="2020-05-01", fin="2020-07-31")
    df.loc[df.fecha == "2020-06-04", "precipitacion"] = 704.44
    df.loc[df.fecha == "2020-06-05", "precipitacion"] = 108.69
    df.loc[df.fecha == "2020-07-10", "precipitacion"] = 300.0   # otro valor imposible
    limpio, informe = clima.limpiar_clima(df)
    assert limpio.loc[limpio.fecha.isin(pd.to_datetime(["2020-06-04", "2020-06-05", "2020-07-10"])), "precipitacion"].tolist() == [0.0, 0.0, 0.0]
    assert informe["lluvia_artefacto"] == 2 and informe["lluvia_excesiva"] == 1
    assert limpio.precipitacion.max() == 1.0


def test_limpiar_enmascara_la_humedad_de_suelo_contaminada():
    df = diario("Cercado", -17.43, -66.17, inicio="2020-05-01", fin="2020-12-31")
    limpio, informe = clima.limpiar_clima(df)
    ventana = (limpio.fecha >= "2020-06-04") & (limpio.fecha <= "2020-10-31")
    assert limpio.loc[ventana, "humedad_suelo_sup"].isna().all()
    assert limpio.loc[ventana, "humedad_suelo_raiz"].isna().all()
    assert limpio.loc[~ventana, "humedad_suelo_sup"].notna().all()
    assert informe["suelo_enmascarado"] == int(ventana.sum())


def test_limpiar_trata_menos_999_y_dias_faltantes():
    df = diario("Cercado", -17.43, -66.17, inicio="2019-01-01", fin="2019-03-31")
    df.loc[df.fecha == "2019-02-10", ["temp_max", "precipitacion"]] = -999.0
    df = df[df.fecha != "2019-02-20"]                      # falta un día completo
    limpio, informe = clima.limpiar_clima(df)
    assert len(limpio) == 90
    assert limpio.temp_max.between(21.9, 22.1).all()       # interpolado, no -999
    assert (limpio.precipitacion >= 0).all()
    assert informe["dias_agregados"] == 1
    v = clima.variables_diarias(limpio.set_index("fecha"), lat=-17.4)
    assert np.isfinite(v["prec_30"].iloc[-1]) and v["prec_30"].iloc[-1] <= 30.0
    assert np.isfinite(v["fwi"]).all()


def test_provincias_con_la_misma_serie_cuentan_como_una():
    partes = [diario(p, lat, lon, fin="2019-03-31") for p, lat, lon in [
        ("Capinota", -17.7170, -66.2376), ("Cercado", -17.4251, -66.1687),
        ("GermanJordan", -17.5825, -65.9563), ("Quillacollo", -17.3174, -66.3280)]]
    partes.append(diario("Chapare", -16.7017, -65.6141, fin="2019-03-31", temp_max=30.0))
    series, tabla = clima.series_unicas(pd.concat(partes, ignore_index=True))
    assert len(tabla) == 2
    assert sorted(series.serie.unique().tolist()) == [0, 1]
    valle = tabla[tabla.provincias.str.contains("Cercado")].iloc[0]
    assert (valle.lat_c, valle.lon_c) == (-17.5, -66.25)       # centro de la celda MERRA-2
    assert valle.provincias.count(",") == 3
    chapare = tabla[tabla.provincias == "Chapare"].iloc[0]
    assert (chapare.lat_c, chapare.lon_c) == (-16.5, -65.625)
    assert len(series) == 2 * 90


def test_acumulados_y_dias_sin_lluvia():
    df = diario("Cercado", -17.43, -66.17, inicio="2019-01-01", fin="2019-04-30").set_index("fecha")
    df.loc["2019-04-11":, "precipitacion"] = 0.0
    v = clima.variables_diarias(df, lat=-17.4)
    assert v.loc["2019-04-10", "prec_30"] == pytest.approx(30.0)
    assert v.loc["2019-04-10", "prec_7"] == pytest.approx(7.0)
    assert v.loc["2019-04-10", "prec_90"] == pytest.approx(90.0)
    assert v.loc["2019-04-10", "dias_sin_lluvia"] == 0
    assert v.loc["2019-04-20", "dias_sin_lluvia"] == 10
    assert v.loc["2019-04-20", "prec_7"] == pytest.approx(0.0)
    assert list(v.columns) == clima.VARS_CLIMA


def test_humedad_minima_y_vpd_son_fisicamente_coherentes():
    df = diario("Cercado", -17.43, -66.17, fin="2019-02-28").set_index("fecha")
    v = clima.variables_diarias(df, lat=-17.4)
    assert 0 < v["hrmin_7"].iloc[-1] < 50.0        # a la hora de más calor la humedad es menor que la media
    assert v["vpd_7"].iloc[-1] > 0


def test_normales_de_una_serie_constante_son_constantes():
    df = diario("Cercado", -17.43, -66.17, inicio="2015-01-01", fin="2018-12-31", precipitacion=2.0).set_index("fecha")
    v = clima.variables_diarias(df, lat=-17.4)
    n = clima.normales(v, 2016, 2018)
    assert n.shape == (366, len(clima.VARS_NORMALES))
    assert list(n.columns) == clima.VARS_NORMALES
    assert np.allclose(n["n_prec_30"], 60.0)
    assert np.allclose(n["n_prec_anual"], 2.0 * 365.25, rtol=0.01)
    assert np.allclose(n["n_tmax_anual"], 22.0)


def test_pesos_idw_suman_uno_y_un_punto_sobre_la_serie_la_toma_entera():
    lon_s = np.array([-66.25, -65.625, -65.0, -66.875, -65.625])
    lat_s = np.array([-17.5, -17.5, -17.0, -17.5, -16.5])
    lon = np.array([-66.25, -65.9, -65.3])
    lat = np.array([-17.5, -17.2, -16.9])
    w = clima.pesos_idw(lon, lat, lon_s, lat_s, k=4)
    assert w.shape == (3, 5) and w.dtype == np.float32
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert w[0, 0] == pytest.approx(1.0)
    assert (w[1] > 0).sum() == 4                       # solo las 4 más cercanas
    assert w[1].argmax() in (0, 1)


def construir(df, anios=(2019, 2019)):
    return clima.ClimaSemanal.desde_tabla(df, anios_normales=anios)


def test_antecedente_no_depende_de_dias_desde_el_lunes_de_la_semana():
    rng = np.random.default_rng(1)
    fechas = pd.date_range("2019-01-01", "2020-12-31")
    lluvia = np.where(rng.random(len(fechas)) < 0.3, rng.uniform(0, 20, len(fechas)), 0.0)
    a = diario("Cercado", -17.4251, -66.1687, precipitacion=lluvia,
               temp_max=rng.uniform(15, 30, len(fechas)), humedad_rel=rng.uniform(20, 90, len(fechas)))
    b = a.copy()
    t = tiempo.semana_de_fecha("2020-09-16")
    lunes = pd.Timestamp(tiempo.inicio_semana(t))
    desde = b.fecha >= lunes
    b.loc[desde, "precipitacion"] = 55.0
    b.loc[desde, "temp_max"] = 40.0
    b.loc[desde, "humedad_rel"] = 5.0
    ca, cb = construir(a), construir(b)
    assert np.array_equal(ca.antecedente(t), cb.antecedente(t), equal_nan=True)
    assert not np.array_equal(ca.antecedente(t + 1), cb.antecedente(t + 1), equal_nan=True)
    assert ca.antecedente(t).shape == (1, len(clima.VARS_ANTECEDENTE))


def test_antecedente_usa_el_domingo_anterior():
    fechas = pd.date_range("2019-01-01", "2020-12-31")
    lluvia = np.zeros(len(fechas))
    t = tiempo.semana_de_fecha("2020-09-16")
    domingo = pd.Timestamp(tiempo.inicio_semana(t)) - pd.Timedelta(days=1)
    lluvia[fechas == domingo] = 12.0
    c = construir(diario("Cercado", -17.4251, -66.1687, precipitacion=lluvia))
    i = clima.VARS_ANTECEDENTE.index("prec_7")
    assert c.antecedente(t)[0, i] == pytest.approx(12.0)
    assert c.antecedente(t - 1)[0, i] == pytest.approx(0.0)


def test_normal_de_la_semana_y_disponibilidad():
    c = construir(diario("Cercado", -17.4251, -66.1687, precipitacion=2.0))
    t = tiempo.semana_de_fecha("2020-09-16")
    n = c.normal(t)
    assert n.shape == (1, len(clima.VARS_NORMALES))
    assert n[0, clima.VARS_NORMALES.index("n_prec_30")] == pytest.approx(60.0)
    assert c.disponible(t)
    assert not c.disponible(tiempo.semana_de_fecha("2021-03-01"))     # no hay datos del domingo anterior
    assert not c.disponible(tiempo.semana_de_fecha("2018-06-01"))
    # la normal sí existe para cualquier semana futura
    assert np.isfinite(c.normal(tiempo.semana_de_fecha("2031-09-17"))).all()


def test_las_anomalias_son_antecedente_menos_normal():
    c = construir(diario("Cercado", -17.4251, -66.1687, precipitacion=2.0))
    t = tiempo.semana_de_fecha("2020-09-16")
    ant = c.antecedente(t)[0]
    assert ant[clima.VARS_ANTECEDENTE.index("an_prec_90")] == pytest.approx(0.0, abs=1e-3)
    assert ant[clima.VARS_ANTECEDENTE.index("an_tmax_7")] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------- fuente ERA5-Land y retraso
def era5(lat, lon, **cambios):
    df = diario(f"era5_{lat}_{lon}", lat, lon, **cambios)
    df["fuente"] = "ERA5-Land"
    return df


def test_la_fuente_se_lee_de_la_columna_y_por_defecto_es_power():
    assert clima.fuente_de(diario("Cercado", -17.4, -66.2)) == "POWER"
    assert clima.fuente_de(era5(-17.4, -66.2)) == "ERA5-Land"


def test_era5_no_recibe_las_correcciones_propias_de_power():
    df = era5(-17.4, -66.2, inicio="2020-05-01", fin="2020-12-31")
    df.loc[df.fecha == "2020-06-04", "precipitacion"] = 40.0
    df.loc[df.fecha == "2020-07-10", "precipitacion"] = 180.0      # lluvia extrema posible en el Chapare
    limpio, informe = clima.limpiar_clima(df)
    assert limpio.loc[limpio.fecha == "2020-06-04", "precipitacion"].item() == 40.0
    assert limpio.loc[limpio.fecha == "2020-07-10", "precipitacion"].item() == 180.0
    assert limpio["humedad_suelo_sup"].notna().all()
    assert informe["lluvia_artefacto"] == informe["lluvia_excesiva"] == informe["suelo_enmascarado"] == 0


def test_las_series_de_era5_conservan_su_ubicacion_aunque_sean_iguales():
    df = pd.concat([era5(-17.4, -66.2, inicio="2020-01-01", fin="2020-03-01"),
                    era5(-17.3, -66.1, inicio="2020-01-01", fin="2020-03-01")], ignore_index=True)
    limpio, _ = clima.limpiar_clima(df)
    _, tabla = clima.series_unicas(limpio, fuente="ERA5-Land")
    assert sorted(zip(tabla.lat_c, tabla.lon_c)) == [(-17.4, -66.2), (-17.3, -66.1)]


def test_power_sigue_ajustando_a_la_malla_merra():
    limpio, _ = clima.limpiar_clima(diario("Cercado", -17.43, -66.17, inicio="2020-01-01", fin="2020-03-01"))
    _, tabla = clima.series_unicas(limpio)
    assert (tabla.lat_c[0], tabla.lon_c[0]) == (-17.5, -66.25)


def test_el_retraso_mueve_la_vispera_hacia_atras():
    fechas = pd.date_range("2019-01-01", "2020-12-31")
    t = tiempo.semana_de_fecha("2020-09-16")
    lunes = pd.Timestamp(tiempo.inicio_semana(t))
    lluvia = np.zeros(len(fechas))
    lluvia[fechas == lunes - pd.Timedelta(days=1)] = 30.0       # domingo: no se ve con 5 días de retraso
    lluvia[fechas == lunes - pd.Timedelta(days=6)] = 12.0       # martes anterior: último día que se ve
    c = clima.ClimaSemanal.desde_tabla(diario("Cercado", -17.4251, -66.1687, precipitacion=lluvia),
                                       anios_normales=(2019, 2019), retraso=5)
    i = clima.VARS_ANTECEDENTE.index("prec_7")
    assert c.retraso == 5
    assert c.antecedente(t)[0, i] == pytest.approx(12.0)
    assert c.antecedente(t + 1)[0, i] == pytest.approx(30.0)      # el domingo aparece una semana después


def test_el_retraso_por_defecto_depende_de_la_fuente():
    from incendios import config
    c_era5 = construir(era5(-17.4, -66.2))
    c_power = construir(diario("Cercado", -17.4, -66.2))
    assert c_era5.retraso == config.DIAS_RETRASO_CLIMA["ERA5-Land"] == 5
    assert c_power.retraso == config.DIAS_RETRASO_CLIMA["POWER"] == 0
    assert (c_era5.fuente, c_power.fuente) == ("ERA5-Land", "POWER")


def test_con_retraso_la_semana_queda_disponible_antes():
    c0 = clima.ClimaSemanal.desde_tabla(diario("Cercado", -17.4, -66.2), anios_normales=(2019, 2019), retraso=0)
    c5 = clima.ClimaSemanal.desde_tabla(diario("Cercado", -17.4, -66.2), anios_normales=(2019, 2019), retraso=5)
    t = tiempo.semana_de_fecha("2021-01-04")        # lunes; los datos llegan al 31-dic-2020
    assert not c0.disponible(t)
    assert c5.disponible(t)


def test_con_muchas_series_los_pesos_son_dispersos_e_iguales_a_los_densos():
    from scipy import sparse
    rng = np.random.default_rng(5)
    lon_s, lat_s = rng.uniform(-67, -64, 200), rng.uniform(-18.8, -15.7, 200)
    lon, lat = rng.uniform(-67, -64, 500), rng.uniform(-18.8, -15.7, 500)
    densos = clima.pesos_idw(lon, lat, lon_s, lat_s, k=4, disperso=False)
    dispersos = clima.pesos_idw(lon, lat, lon_s, lat_s, k=4)
    assert sparse.issparse(dispersos) and dispersos.dtype == np.float32
    assert dispersos.nnz == 4 * 500
    assert np.allclose(dispersos.toarray(), densos, atol=1e-6)
