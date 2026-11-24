import numpy as np
import pandas as pd
import pytest

from incendios import focos
from incendios.malla import Malla
from incendios.tiempo import semana_de_fecha


@pytest.fixture
def malla():
    return Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=10, nfilas=10)


def detecciones(filas):
    columnas = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type", "daynight"]
    return pd.DataFrame(filas, columns=columnas)


def test_filtrar_vegetacion_quita_fuentes_estaticas_y_baja_confianza():
    df = pd.DataFrame({"type": [0, 0, 2, 0, 3], "confidence": ["n", "h", "n", "l", "h"]})
    assert focos.filtrar_vegetacion(df).index.tolist() == [0, 1]


def conteo(inicio, fin, ceros):
    s = pd.Series(100, index=pd.date_range(inicio, fin))
    s[pd.to_datetime(ceros)] = 0
    return s


def test_racha_de_dos_dias_sin_detecciones_es_apagon():
    s = conteo("2022-07-01", "2022-08-31", ["2022-07-27", "2022-07-28"])
    assert focos.dias_sin_satelite(s).strftime("%Y-%m-%d").tolist() == ["2022-07-27", "2022-07-28"]


def test_dia_aislado_sin_detecciones_en_epoca_de_lluvias_no_es_apagon():
    s = conteo("2021-01-01", "2021-03-31", ["2021-02-10"])
    assert len(focos.dias_sin_satelite(s)) == 0


def test_dia_aislado_sin_detecciones_en_temporada_de_fuego_si_es_apagon():
    s = conteo("2024-09-01", "2024-09-30", ["2024-09-26"])
    assert focos.dias_sin_satelite(s).strftime("%Y-%m-%d").tolist() == ["2024-09-26"]


def test_dos_detecciones_en_la_misma_celda_y_semana_dan_una_fila(malla):
    df = detecciones([
        [-17.0101, -65.9901, "2024-09-16", 1730, "n", 5.0, 0, "D"],
        [-17.0102, -65.9902, "2024-09-18", 1740, "h", 9.0, 0, "D"],
        [-17.0101, -65.9901, "2024-09-25", 1730, "n", 2.0, 0, "D"],
    ])
    df["fecha"] = pd.to_datetime(df["acq_date"])
    pos = focos.positivos_semanales(df, malla)
    assert list(pos.columns) == ["semana", "celda", "n_det", "frp_max"]
    assert len(pos) == 2
    s = semana_de_fecha("2024-09-16")
    fila = pos[pos.semana == s].iloc[0]
    assert fila.celda == 2 * 10 + 1 and fila.n_det == 2 and fila.frp_max == 9.0
    assert pos[pos.semana == s + 1].iloc[0].n_det == 1


def test_focos_fuera_del_recuadro_no_generan_positivos(malla):
    df = detecciones([
        [-17.0101, -65.9901, "2024-09-16", 1730, "n", 5.0, 0, "D"],
        [-17.0601, -65.9901, "2024-09-16", 1730, "n", 5.0, 0, "D"],   # al sur de la malla
        [-17.0101, -66.0001, "2024-09-16", 1730, "n", 5.0, 0, "D"],   # al oeste
        [-17.0101, -65.9499, "2024-09-16", 1730, "n", 5.0, 0, "D"],   # apenas pasado el borde este
    ])
    df["fecha"] = pd.to_datetime(df["acq_date"])
    pos = focos.positivos_semanales(df, malla)
    assert len(pos) == 1 and (pos.celda >= 0).all() and (pos.celda < 100).all()


def test_dias_observados_descuenta_apagones_y_dias_fuera_del_registro():
    # registro del lunes 2012-01-09 (semana 1) al domingo 2012-01-29 (semana 3)
    s = conteo("2012-01-09", "2012-01-29", ["2012-01-17", "2012-01-18", "2012-01-19"])
    obs = focos.dias_observados(s, n_semanas=5)
    assert obs.tolist() == [0, 7, 4, 7, 0]


def test_cargar_focos_lee_todos_los_anios_y_recorta_al_recuadro(tmp_path, malla):
    cab = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type\n"
    (tmp_path / "viirs-snpp_2023_Bolivia.csv").write_text(
        cab + "-17.0101,-65.9901,330,0.4,0.4,2023-09-01,1730,N,VIIRS,n,2,290,5.0,D,0\n"
            + "-12.0,-63.0,330,0.4,0.4,2023-09-03,1730,N,VIIRS,n,2,290,5.0,D,0\n")
    (tmp_path / "viirs-snpp_2024_Bolivia.csv").write_text(
        cab + "-17.0201,-65.9801,330,0.4,0.4,2024-01-02,1730,N,VIIRS,h,2,290,7.5,D,2\n")
    df, diario = focos.cargar_focos(tmp_path, malla)
    assert len(df) == 2
    assert str(df.fecha.dtype).startswith("datetime64")
    assert df.celda.tolist() == [2 * 10 + 1, 4 * 10 + 3]
    assert diario.index[0] == pd.Timestamp("2023-09-01") and diario.index[-1] == pd.Timestamp("2024-01-02")
    assert diario["2023-09-01"] == 1 and diario["2023-09-02"] == 0 and diario["2023-09-03"] == 1
    assert len(diario) == (pd.Timestamp("2024-01-02") - pd.Timestamp("2023-09-01")).days + 1


def test_cargar_focos_sin_archivos_es_error_claro(tmp_path, malla):
    with pytest.raises(FileNotFoundError, match="viirs-snpp"):
        focos.cargar_focos(tmp_path, malla)


def test_cargar_focos_de_otro_satelite_con_otro_patron(tmp_path, malla):
    cab = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type\n"
    (tmp_path / "viirs-snpp_2023_Bolivia.csv").write_text(
        cab + "-17.0101,-65.9901,330,0.4,0.4,2023-09-01,1730,N,VIIRS,n,2,290,5.0,D,0\n")
    (tmp_path / "viirs-noaa20_2023_Bolivia.csv").write_text(
        cab + "-17.0201,-65.9801,330,0.4,0.4,2023-09-02,1730,N20,VIIRS,n,2,290,7.5,D,0\n"
            + "-17.0301,-65.9701,330,0.4,0.4,2023-09-03,1730,N20,VIIRS,n,2,290,7.5,D,0\n")
    por_defecto, _ = focos.cargar_focos(tmp_path, malla)
    assert len(por_defecto) == 1                                        # el patrón por defecto es S-NPP
    noaa, diario = focos.cargar_focos(tmp_path, malla, patron="viirs-noaa20_*_Bolivia.csv")
    assert len(noaa) == 2 and diario.index[0] == pd.Timestamp("2023-09-02")
    with pytest.raises(FileNotFoundError, match="viirs-noaa21"):
        focos.cargar_focos(tmp_path, malla, patron="viirs-noaa21_*_Bolivia.csv")


def test_el_patron_por_defecto_viene_de_la_configuracion():
    from incendios import config
    assert config.PATRON_FOCOS == "viirs-snpp_*_Bolivia.csv"


CABECERA = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,"
            "version,bright_ti5,frp,daynight,type\n")


def fila_csv(fecha, satelite="N", lat=-17.0101, lon=-65.9901):
    return f"{lat},{lon},330,0.4,0.4,{fecha},1730,{satelite},VIIRS,n,2,290,5.0,D,0\n"


def test_los_focos_se_encadenan_snpp_hasta_su_cierre_y_despues_noaa20(tmp_path, malla):
    (tmp_path / "viirs-snpp_2026_Bolivia.csv").write_text(
        CABECERA + fila_csv("2026-10-31") + fila_csv("2026-11-01") + fila_csv("2026-11-02"))   # el 2-nov va a medias
    (tmp_path / "viirs-noaa20_2026_Bolivia.csv").write_text(
        CABECERA + fila_csv("2026-11-01", "N20") + fila_csv("2026-11-02", "N20") + fila_csv("2026-11-05", "N20"))
    df, diario = focos.cargar_focos(tmp_path, malla)
    assert df["acq_date"].tolist() == ["2026-10-31", "2026-11-01", "2026-11-02", "2026-11-05"]
    assert df["fuente"].tolist() == ["S-NPP", "S-NPP", "NOAA-20", "NOAA-20"]
    assert diario.index[-1] == pd.Timestamp("2026-11-05")
    assert diario["2026-11-01"] == 1 and diario["2026-11-02"] == 1 and diario["2026-11-03"] == 0


def test_sin_archivos_noaa20_los_focos_terminan_con_snpp(tmp_path, malla):
    (tmp_path / "viirs-snpp_2026_Bolivia.csv").write_text(CABECERA + fila_csv("2026-10-30") + fila_csv("2026-11-02"))
    df, diario = focos.cargar_focos(tmp_path, malla)
    assert df["acq_date"].tolist() == ["2026-10-30"] and diario.index[-1] == pd.Timestamp("2026-10-30")


def test_la_cadena_de_satelites_viene_de_la_configuracion():
    from incendios import config
    snpp, noaa20 = config.FUENTES_FOCOS
    assert snpp["patron"] == config.PATRON_FOCOS and snpp["hasta"] == "2026-11-01"
    assert noaa20["patron"] == "viirs-noaa20_*_Bolivia.csv" and noaa20["desde"] == "2026-11-02"


def _semanales(celda, n_semanas, daynight):
    lunes = pd.date_range("2020-01-06", periods=n_semanas, freq="7D")
    return pd.DataFrame({"celda": celda, "fecha": lunes, "daynight": daynight})


def test_fuentes_persistentes_marca_la_celda_nocturna_en_muchas_semanas():
    df = pd.concat([
        _semanales(7, 12, "N"),      # 12 semanas distintas, siempre de noche: horno o antorcha
        _semanales(8, 12, "D"),      # 12 semanas de día: fuego de vegetación recurrente
        _semanales(9, 3, "N"),       # de noche, pero pocas semanas
    ], ignore_index=True)
    t = focos.fuentes_persistentes(df)
    assert t["celda"].tolist() == [7]
    assert t["semanas"].tolist() == [12]
    assert t["detecciones"].tolist() == [12]
    assert t["nocturnas"].tolist() == [1.0]


def test_fuentes_persistentes_cuenta_semanas_y_no_detecciones():
    # 30 detecciones nocturnas en una sola semana son un incendio grande, no una fuente fija
    df = pd.DataFrame({"celda": 5, "fecha": pd.to_datetime(["2021-09-06"] * 30), "daynight": "N"})
    assert focos.fuentes_persistentes(df).empty


def test_fuentes_persistentes_exige_que_casi_todo_sea_nocturno():
    df = _semanales(3, 10, ["N"] * 9 + ["D"])            # 90 % nocturnas
    assert focos.fuentes_persistentes(df)["celda"].tolist() == [3]
    df = _semanales(3, 10, ["N"] * 7 + ["D"] * 3)        # 70 %
    assert focos.fuentes_persistentes(df).empty


def test_fuentes_persistentes_sin_columna_de_dia_o_noche_no_marca_nada():
    df = _semanales(3, 12, "N").drop(columns="daynight")
    assert focos.fuentes_persistentes(df).empty


def test_quitar_fuentes_elimina_solo_las_detecciones_de_esas_celdas():
    df = pd.concat([_semanales(7, 12, "N"), _semanales(8, 12, "D")], ignore_index=True)
    limpio = focos.quitar_fuentes(df, focos.fuentes_persistentes(df))
    assert set(limpio["celda"]) == {8} and len(limpio) == 12


def test_los_focos_se_leen_de_varias_carpetas_y_un_archivo_repetido_cuenta_una_vez(tmp_path, malla):
    originales, bajados = tmp_path / "raw", tmp_path / "external"
    originales.mkdir()
    bajados.mkdir()
    (originales / "viirs-snpp_2025_Bolivia.csv").write_text(CABECERA + fila_csv("2025-09-01"))
    (originales / "viirs-snpp_2026_Bolivia.csv").write_text(CABECERA + fila_csv("2026-03-01"))
    # el archivo de 2026 se volvió a bajar con más días: manda el bajado
    (bajados / "viirs-snpp_2026_Bolivia.csv").write_text(CABECERA + fila_csv("2026-03-01") + fila_csv("2026-03-02"))
    (bajados / "viirs-noaa20_2026_Bolivia.csv").write_text(CABECERA + fila_csv("2026-11-03", "N20"))
    df, _ = focos.cargar_focos([originales, bajados], malla)
    assert df["acq_date"].tolist() == ["2025-09-01", "2026-03-01", "2026-03-02", "2026-11-03"]
    with pytest.raises(FileNotFoundError, match="raw"):
        focos.cargar_focos([tmp_path / "raw" / "nada", bajados / "nada"], malla)
