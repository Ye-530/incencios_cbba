import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file

from incendios import clima, era5, tiempo
from incendios.malla import Malla, malla_cochabamba

LAT = np.array([-17.3, -17.4])
LON = np.array([-66.2, -66.1, -66.0])


def escribir_nc(ruta, variables, tiempos, nombre_tiempo="valid_time",
                unidades="seconds since 1970-01-01", lat=LAT, lon=LON, tipo="f", atributos=None):
    """NetCDF3 con dims (tiempo, latitude, longitude); `tiempos` en las unidades indicadas."""
    f = netcdf_file(str(ruta), "w")
    f.createDimension(nombre_tiempo, len(tiempos))
    f.createDimension("latitude", len(lat))
    f.createDimension("longitude", len(lon))
    t = f.createVariable(nombre_tiempo, "d", (nombre_tiempo,))
    t[:] = tiempos
    t.units = unidades
    a = f.createVariable("latitude", "d", ("latitude",))
    a[:] = lat
    o = f.createVariable("longitude", "d", ("longitude",))
    o[:] = lon
    for nombre, datos in variables.items():
        v = f.createVariable(nombre, tipo, (nombre_tiempo, "latitude", "longitude"))
        v[:] = datos
        for clave, valor in (atributos or {}).get(nombre, {}).items():
            setattr(v, clave, valor)
    f.close()
    return ruta


def segundos(fechas_horas):
    return ((pd.to_datetime(fechas_horas) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy()


def test_area_cubre_la_malla_con_margen_en_multiplos_de_un_decimo():
    m = malla_cochabamba()
    norte, oeste, sur, este = era5.area(m)
    assert norte >= m.norte + 0.1 and oeste <= m.oeste - 0.1
    assert sur <= m.norte - m.nfilas * m.celda - 0.1 and este >= m.oeste + m.ncols * m.celda + 0.1
    for v in (norte, oeste, sur, este):
        assert abs(v * 10 - round(v * 10)) < 1e-9


def test_la_solicitud_mensual_pide_cinco_horas_y_siete_variables():
    s = era5.solicitud(2020, 2, [-15.5, -67.2, -18.8, -64.0])
    assert s["year"] == "2020" and s["month"] == "02"
    assert s["day"] == [f"{d:02d}" for d in range(1, 30)]              # 2020 es bisiesto
    assert s["time"] == list(era5.HORAS_UTC) == ["00:00", "10:00", "13:00", "16:00", "19:00"]
    assert set(s["variable"]) == set(era5.VARIABLES) and len(era5.VARIABLES) == 7
    assert s["data_format"] == "netcdf" and s["download_format"] == "unarchived"
    assert s["area"] == [-15.5, -67.2, -18.8, -64.0]
    corta = era5.solicitud(2020, 2, [-15.5, -67.2, -18.8, -64.0], hasta=date(2020, 2, 10))
    assert corta["day"][-1] == "10"


def test_meses_entre_dos_fechas_y_ultimo_dia_publicado():
    assert era5.meses(date(2019, 11, 15), date(2020, 2, 3)) == [(2019, 11), (2019, 12), (2020, 1), (2020, 2)]
    assert era5.ultimo_dia_publicado(date(2026, 11, 18)) == date(2026, 11, 12)


def test_lee_netcdf_nuevo_con_tiempo_en_segundos(tmp_path):
    horas = pd.to_datetime(["2020-01-01 10:00", "2020-01-01 13:00"])
    datos = np.arange(12, dtype=np.float32).reshape(2, 2, 3) + 280.0
    ruta = escribir_nc(tmp_path / "a.nc", {"t2m": datos}, segundos(horas))
    cubos = era5.leer_netcdf(ruta)
    c = cubos["t2m"]
    assert list(pd.to_datetime(c["tiempo"])) == list(horas)
    assert np.allclose(c["lat"], LAT) and np.allclose(c["lon"], LON)
    assert c["datos"].shape == (2, 2, 3) and c["datos"].dtype == np.float32
    assert c["datos"][1, 0, 2] == pytest.approx(288.0)


def test_lee_un_zip_con_un_archivo_instantaneo_y_otro_acumulado(tmp_path):
    horas = segundos(["2020-01-02 00:00"])
    a = escribir_nc(tmp_path / "inst.nc", {"t2m": np.full((1, 2, 3), 290.0)}, horas)
    b = escribir_nc(tmp_path / "acum.nc", {"tp": np.full((1, 2, 3), 0.004)}, horas)
    ruta = tmp_path / "era5land_2020_01.nc"
    with zipfile.ZipFile(ruta, "w") as z:
        z.write(a, "data_stream-oper_stepType-instant.nc")
        z.write(b, "data_stream-oper_stepType-accum.nc")
    cubos = era5.leer_netcdf(ruta)
    assert set(cubos) == {"t2m", "tp"}
    assert cubos["tp"]["datos"][0, 0, 0] == pytest.approx(0.004)


def test_lee_el_formato_antiguo_empaquetado_con_valores_faltantes(tmp_path):
    horas = (pd.to_datetime(["2020-01-01 10:00"]) - pd.Timestamp("1900-01-01")) / pd.Timedelta(hours=1)
    crudo = np.array([[[100, 200, -32767], [0, 50, 10]]], dtype=np.int16)
    attrs = {"t2m": {"scale_factor": 0.01, "add_offset": 280.0, "_FillValue": np.int16(-32767)}}
    ruta = escribir_nc(tmp_path / "viejo.nc", {"t2m": crudo}, horas.to_numpy(), nombre_tiempo="time",
                       unidades="hours since 1900-01-01 00:00:00.0", tipo="h", atributos=attrs)
    c = era5.leer_netcdf(ruta)["t2m"]
    assert pd.Timestamp(c["tiempo"][0]) == pd.Timestamp("2020-01-01 10:00")
    assert c["datos"][0, 0, 0] == pytest.approx(281.0)
    assert c["datos"][0, 0, 1] == pytest.approx(282.0)
    assert np.isnan(c["datos"][0, 0, 2])


def test_unir_concatena_ordena_y_se_queda_con_el_ultimo_repetido():
    def cubo(horas, valor):
        t = pd.to_datetime(horas).to_numpy()
        return {"t2m": {"tiempo": t, "lat": LAT, "lon": LON,
                        "datos": np.full((len(t), 2, 3), valor, dtype=np.float32)}}
    unido = era5.unir([cubo(["2020-01-02 10:00"], 1.0), cubo(["2020-01-01 10:00", "2020-01-02 10:00"], 2.0)])
    c = unido["t2m"]
    assert list(pd.to_datetime(c["tiempo"])) == list(pd.to_datetime(["2020-01-01 10:00", "2020-01-02 10:00"]))
    assert c["datos"][:, 0, 0].tolist() == [2.0, 2.0]


# ---------------------------------------------------------------- de muestras horarias a días locales
def cubos_sinteticos(dias, lat=LAT, lon=LON, temp_local=None, tp_diaria_m=0.003, u=3.0, v=4.0, suelo=(0.30, 0.35)):
    """Muestras a 00, 10, 13, 16 y 19 UTC de los días UTC indicados (más el 00 UTC del día siguiente)."""
    dias = pd.to_datetime(dias)
    horas = [d + pd.Timedelta(hours=h) for d in dias for h in (0, 10, 13, 16, 19)]
    horas.append(dias[-1] + pd.Timedelta(days=1))
    t = pd.DatetimeIndex(horas)
    n = (len(t), len(lat), len(lon))
    local = t + pd.Timedelta(hours=era5.HUSO_HORAS)
    temp = np.array([temp_local(x) if temp_local else 15.0 for x in local], dtype=np.float32) + 273.15
    t2m = np.broadcast_to(temp[:, None, None], n).astype(np.float32)

    def cubo(datos):
        return {"tiempo": t.to_numpy(), "lat": np.asarray(lat), "lon": np.asarray(lon),
                "datos": np.ascontiguousarray(datos, dtype=np.float32)}
    tp = np.where((t.hour == 0)[:, None, None], tp_diaria_m, 0.001)
    return {"t2m": cubo(t2m), "d2m": cubo(t2m - 5.0), "u10": cubo(np.full(n, u)), "v10": cubo(np.full(n, v)),
            "swvl1": cubo(np.full(n, suelo[0])), "swvl2": cubo(np.full(n, suelo[1])),
            "tp": cubo(np.broadcast_to(tp, n))}


def test_diario_usa_la_fecha_local_y_las_cinco_muestras():
    por_hora = {6: 10.0, 9: 15.0, 12: 20.0, 15: 25.0, 20: 17.0}
    cubos = cubos_sinteticos(["2020-01-01", "2020-01-02"], temp_local=lambda x: por_hora[x.hour])
    cubos["d2m"]["datos"] = cubos["t2m"]["datos"].copy()               # aire saturado
    d = era5.diario(cubos)
    dia1 = d[(d.fecha == "2020-01-01") & (d.lat == -17.3) & (d.lon == -66.2)].iloc[0]
    assert dia1.temp_max == pytest.approx(25.0, abs=1e-3)
    assert dia1.temp_min == pytest.approx(10.0, abs=1e-3)
    assert dia1.humedad_rel == pytest.approx(100.0, abs=1e-3)
    assert dia1.viento_vel == pytest.approx(5.0 / clima.VIENTO_2M_A_10M, abs=1e-4)   # 10 m llevado a 2 m
    assert dia1.precipitacion == pytest.approx(3.0, abs=1e-4)          # 00 UTC del 2 de enero, en mm
    assert dia1.humedad_suelo_sup == pytest.approx(0.30) and dia1.humedad_suelo_raiz == pytest.approx(0.35)
    # el 00 UTC del 1 de enero es el 31 de diciembre a las 20 h: una sola muestra, ese día no sale
    assert sorted(d.fecha.unique()) == ["2020-01-01", "2020-01-02"]
    assert len(d) == 2 * len(LAT) * len(LON)


def test_diario_descarta_los_dias_sin_lluvia_publicada():
    cubos = cubos_sinteticos(["2020-01-01", "2020-01-02"])
    ultimo = cubos["tp"]["tiempo"][-1]
    for c in cubos.values():                                           # sin el 00 UTC del 3 de enero
        quedan = c["tiempo"] != ultimo
        c["tiempo"], c["datos"] = c["tiempo"][quedan], c["datos"][quedan]
    d = era5.diario(cubos)
    assert sorted(d.fecha.unique()) == ["2020-01-01"]


def test_puntos_utiles_cubren_el_departamento_mas_un_margen():
    malla = Malla(oeste=-66.30, norte=-17.20, celda=0.05, ncols=6, nfilas=4)     # 0,3° × 0,2°
    mascara = np.zeros(malla.forma, dtype=bool)
    mascara[1:3, 1:3] = True                                            # alrededor de (-17.3, -66.2)
    lat = np.round(np.arange(-17.0, -17.85, -0.1), 1)
    lon = np.round(np.arange(-66.6, -65.65, 0.1), 1)
    utiles = era5.puntos_utiles(lat, lon, malla, mascara, margen=1)
    filas, cols = np.nonzero(utiles)
    assert set(np.round(lat[filas], 1)) == {-17.2, -17.3, -17.4}
    assert set(np.round(lon[cols], 1)) == {-66.3, -66.2, -66.1}


def test_la_tabla_tiene_el_esquema_del_proyecto_y_la_entiende_el_clima(tmp_path):
    dias = pd.date_range("2019-01-01", "2020-12-31")
    rng = np.random.default_rng(3)
    lluvia = dict(zip(dias.normalize(), np.where(rng.random(len(dias)) < 0.3, 0.008, 0.0)))
    cubos = cubos_sinteticos(dias, temp_local=lambda x: 12.0 + 0.6 * (x.hour - 6))
    t = pd.DatetimeIndex(cubos["tp"]["tiempo"])
    cubos["tp"]["datos"] = np.broadcast_to(
        np.array([lluvia.get(x.normalize() - pd.Timedelta(days=1), 0.0) if x.hour == 0 else 0.0
                  for x in t], dtype=np.float32)[:, None, None], cubos["tp"]["datos"].shape).copy()
    tabla = era5.tabla_clima(era5.diario(cubos))
    assert list(tabla.columns) == ["provincia", "lat", "lon", "fecha", "anio", "mes", "dia", *clima.COLUMNAS_METEO,
                                   "fuente"]
    ruta = tmp_path / "clima_era5land_diario.csv.gz"
    tabla.to_csv(ruta, index=False)
    leida = clima.cargar_clima(ruta)
    assert clima.fuente_de(leida) == "ERA5-Land"
    c = clima.ClimaSemanal.desde_tabla(leida, anios_normales=(2019, 2019))
    assert len(c.tabla) == len(LAT) * len(LON)
    assert sorted(zip(c.tabla.lat_c, c.tabla.lon_c)) == sorted((a, o) for a in LAT for o in LON)
    assert c.retraso == 5
    ant = c.antecedente(tiempo.semana_de_fecha("2020-09-16"))
    assert ant.shape == (6, len(clima.VARS_ANTECEDENTE)) and np.isfinite(ant).all()
