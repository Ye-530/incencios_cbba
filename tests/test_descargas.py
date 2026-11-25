import numpy as np
import pandas as pd
import pytest

from incendios import descargas
from incendios.malla import Malla, malla_cochabamba


def mascara_rectangular(malla, oeste, sur, este, norte):
    lon, lat = malla.centros()
    LON, LAT = np.meshgrid(lon, lat)
    return (LON > oeste) & (LON < este) & (LAT > sur) & (LAT < norte)


def test_teselas_de_relieve_que_cubren_la_zona():
    m = malla_cochabamba()
    mascara = mascara_rectangular(m, -66.5, -17.9, -64.8, -16.2)
    assert descargas.teselas_dem(mascara, m) == [
        "Copernicus_DSM_COG_10_S17_00_W065_00_DEM", "Copernicus_DSM_COG_10_S17_00_W066_00_DEM",
        "Copernicus_DSM_COG_10_S17_00_W067_00_DEM", "Copernicus_DSM_COG_10_S18_00_W065_00_DEM",
        "Copernicus_DSM_COG_10_S18_00_W066_00_DEM", "Copernicus_DSM_COG_10_S18_00_W067_00_DEM"]


def test_una_tesela_con_apenas_un_borde_de_celdas_no_se_pide():
    m = malla_cochabamba()
    mascara = mascara_rectangular(m, -66.5, -17.9, -64.8, -16.2)
    lon, _ = m.centros()
    # una sola columna de celdas al oeste de -67 (franja de ~500 m), solo 10 celdas
    mascara[100:110, np.flatnonzero(lon < -67.0)[-1]] = True
    assert not any("W068" in n for n in descargas.teselas_dem(mascara, m))
    assert any("W068" in n for n in descargas.teselas_dem(mascara, m, min_celdas=5))


def test_teselas_de_cobertura_son_de_tres_grados_y_se_nombran_por_la_esquina_suroeste():
    m = malla_cochabamba()
    mascara = mascara_rectangular(m, -66.5, -18.6, -64.8, -16.2)
    assert descargas.teselas_worldcover(mascara, m) == [
        "ESA_WorldCover_10m_2021_v200_S18W066_Map", "ESA_WorldCover_10m_2021_v200_S18W069_Map",
        "ESA_WorldCover_10m_2021_v200_S21W066_Map", "ESA_WorldCover_10m_2021_v200_S21W069_Map"]


def test_los_nombres_cambian_de_hemisferio_correctamente():
    m = Malla(oeste=20.0, norte=11.0, celda=0.005, ncols=400, nfilas=400)       # 20–22° E, 9–11° N
    nombres = descargas.teselas_dem(np.ones(m.forma, bool), m)
    assert nombres == ["Copernicus_DSM_COG_10_N09_00_E020_00_DEM", "Copernicus_DSM_COG_10_N09_00_E021_00_DEM",
                       "Copernicus_DSM_COG_10_N10_00_E020_00_DEM", "Copernicus_DSM_COG_10_N10_00_E021_00_DEM"]
    assert descargas.teselas_worldcover(np.ones(m.forma, bool), m) == [
        "ESA_WorldCover_10m_2021_v200_N09E018_Map", "ESA_WorldCover_10m_2021_v200_N09E021_Map"]


def test_urls_de_las_teselas():
    assert descargas.url_dem("Copernicus_DSM_COG_10_S19_00_W065_00_DEM") == (
        "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S19_00_W065_00_DEM/"
        "Copernicus_DSM_COG_10_S19_00_W065_00_DEM.tif")
    assert descargas.url_worldcover("ESA_WorldCover_10m_2021_v200_S21W066_Map") == (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        "ESA_WorldCover_10m_2021_v200_S21W066_Map.tif")


def test_faltantes_son_las_teselas_que_no_estan_en_la_carpeta(tmp_path):
    (tmp_path / "a.tif").write_bytes(b"x")
    assert descargas.faltantes(["a", "b"], tmp_path) == ["b"]
    assert descargas.faltantes(["a", "b"], tmp_path / "no_existe") == ["a", "b"]


def test_celdas_del_reanalisis_que_tocan_el_departamento():
    m = malla_cochabamba()
    # una zona pequeña dentro de la celda centrada en (-17.5, -66.25)
    mascara = mascara_rectangular(m, -66.4, -17.6, -66.1, -17.4)
    assert descargas.celdas_merra(mascara, m) == [(-17.5, -66.25)]
    # una zona que cruza a la celda vecina del este
    mascara = mascara_rectangular(m, -66.4, -17.6, -65.8, -17.4)
    assert descargas.celdas_merra(mascara, m) == [(-17.5, -66.25), (-17.5, -65.625)]


def test_url_de_power_pide_los_parametros_del_proyecto():
    url = descargas.url_power(-17.5, -66.25, "2012-01-01", "2026-11-17")
    assert url.startswith("https://power.larc.nasa.gov/api/temporal/daily/point?")
    assert "latitude=-17.5" in url and "longitude=-66.25" in url
    assert "start=20120101" in url and "end=20261117" in url
    for parametro in ("T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR", "GWETTOP", "GWETROOT"):
        assert parametro in url


def test_tabla_de_power_tiene_el_mismo_esquema_que_el_archivo_original():
    respuesta = {"properties": {"parameter": {
        "T2M_MAX": {"20240101": 22.5, "20240102": -999.0}, "T2M_MIN": {"20240101": 8.0, "20240102": 7.5},
        "RH2M": {"20240101": 60.0, "20240102": 61.0}, "WS2M": {"20240101": 2.0, "20240102": 2.1},
        "PRECTOTCORR": {"20240101": 1.2, "20240102": 0.0}, "GWETTOP": {"20240101": 0.5, "20240102": 0.5},
        "GWETROOT": {"20240101": 0.55, "20240102": 0.55}}}}
    t = descargas.tabla_power(respuesta, "celda_-17.5_-66.25", -17.5, -66.25)
    assert list(t.columns) == ["provincia", "lat", "lon", "fecha", "anio", "mes", "dia", "temp_max", "temp_min",
                               "humedad_rel", "viento_vel", "precipitacion", "humedad_suelo_sup", "humedad_suelo_raiz"]
    assert t["fecha"].tolist() == ["2024-01-01", "2024-01-02"]
    assert t["temp_max"].iloc[0] == 22.5 and np.isnan(t["temp_max"].iloc[1])        # -999 = sin dato
    assert t["provincia"].unique().tolist() == ["celda_-17.5_-66.25"]


def test_url_de_firms_para_un_recuadro_y_el_maximo_de_dias():
    maximo = descargas.MAX_DIAS_FIRMS
    url = descargas.url_firms("CLAVE", "VIIRS_NOAA20_SP", (-69.7, -23.0, -57.4, -9.6), "2024-09-01", maximo)
    assert url == ("https://firms.modaps.eosdis.nasa.gov/api/area/csv/CLAVE/VIIRS_NOAA20_SP/"
                   f"-69.7,-23.0,-57.4,-9.6/{maximo}/2024-09-01")
    with pytest.raises(ValueError, match=str(maximo)):
        descargas.url_firms("CLAVE", "VIIRS_NOAA20_SP", (-69.7, -23.0, -57.4, -9.6), "2024-09-01", maximo + 1)
    with pytest.raises(ValueError):
        descargas.url_firms("CLAVE", "VIIRS_NOAA20_SP", (-69.7, -23.0, -57.4, -9.6), "2024-09-01", 0)


def test_tramos_cubren_el_periodo_sin_solaparse_ni_pasar_el_maximo():
    tramos = descargas.tramos("2024-01-01", "2024-01-25")            # 25 días
    assert all(1 <= n <= descargas.MAX_DIAS_FIRMS for _, n in tramos)
    assert sum(n for _, n in tramos) == 25
    assert tramos[0] == ("2024-01-01", descargas.MAX_DIAS_FIRMS)
    inicios = [pd.Timestamp(f) for f, _ in tramos]
    assert all(b == a + pd.Timedelta(days=n) for (a, b), (_, n) in zip(zip(inicios, inicios[1:]), tramos))
    assert descargas.tramos("2024-01-01", "2024-01-25", dias=10) == [("2024-01-01", 10), ("2024-01-11", 10), ("2024-01-21", 5)]
    assert descargas.tramos("2024-03-05", "2024-03-05") == [("2024-03-05", 1)]
