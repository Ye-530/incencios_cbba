"""Piezas de la actualización (sin red): qué meses de clima pedir, armar la tabla diaria y las semanas."""
import os
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from incendios import actualizacion, era5, tiempo
from incendios.malla import Malla
from tests.test_era5 import LAT, LON, cubos_sinteticos, escribir_nc, segundos

# celdas de 0,05° alrededor de los puntos ERA5 de las pruebas: lat -17,3/-17,4 y lon -66,2/-66,1/-66,0
MALLA = Malla(oeste=-66.25, norte=-17.25, celda=0.05, ncols=5, nfilas=4)
MASCARA = np.ones(MALLA.forma, dtype=bool)


def test_los_meses_pendientes_son_los_que_faltan_y_los_ultimos(tmp_path):
    for anio, mes in era5.meses(date(2012, 1, 1), date(2026, 8, 31)):
        if (anio, mes) != (2015, 3):
            era5.archivo_mes(tmp_path, anio, mes).write_bytes(b"x")
    pendientes = actualizacion.meses_pendientes(tmp_path, date(2026, 11, 18))
    assert [(a, m) for a, m, _ in pendientes] == [(2015, 3), (2026, 9), (2026, 10), (2026, 11)]
    assert pendientes[0][2] is None and pendientes[-1][2] == date(2026, 11, 12)   # el mes en curso, hasta lo publicado


def test_un_mes_bajado_en_dos_mitades_cuenta_como_bajado(tmp_path):
    (tmp_path / "era5land_2012_01_a.nc").write_bytes(b"x")
    (tmp_path / "era5land_2012_01_b.nc").write_bytes(b"x")
    pendientes = actualizacion.meses_pendientes(tmp_path, date(2012, 9, 20), desde=date(2012, 1, 1))
    assert (2012, 1) not in [(a, m) for a, m, _ in pendientes]
    assert actualizacion.archivos_del_mes(tmp_path, 2012, 1)[0].name == "era5land_2012_01_a.nc"


def escribir_mes(carpeta, anio, mes, dias, sin_ultimo=False):
    cubos = cubos_sinteticos(dias)
    corte = slice(None, -1) if sin_ultimo else slice(None)
    escribir_nc(era5.archivo_mes(carpeta, anio, mes), {n: c["datos"][corte] for n, c in cubos.items()},
                segundos(cubos["t2m"]["tiempo"][corte]))


@pytest.fixture
def meses(tmp_path):
    carpeta = tmp_path / "era5land"
    carpeta.mkdir()
    escribir_mes(carpeta, 2019, 12, pd.date_range("2019-12-01", "2019-12-31"), sin_ultimo=True)  # sin 00 UTC del 1-ene
    escribir_mes(carpeta, 2020, 1, pd.date_range("2020-01-01", "2020-01-10"))
    return carpeta


def test_los_meses_se_vuelven_la_tabla_diaria_con_un_cache_por_anio(meses, tmp_path):
    destino = tmp_path / "clima_era5land_diario.csv.gz"
    actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino)
    tabla = pd.read_csv(destino)
    assert tabla["fecha"].min() == "2019-12-01" and tabla["fecha"].max() == "2020-01-10"
    assert "2019-12-31" in set(tabla["fecha"])              # su lluvia está en el archivo de enero
    assert tabla.groupby("fecha").size().eq(len(LAT) * len(LON)).all()
    assert (tabla["fuente"] == "ERA5-Land").all()
    assert sorted(p.name for p in (meses / "diario").glob("*.csv.gz")) == [
        "era5land_diario_2019.csv.gz", "era5land_diario_2020.csv.gz"]


def test_solo_entran_los_puntos_que_sirven_al_departamento(meses, tmp_path):
    mascara = np.zeros(MALLA.forma, dtype=bool)
    mascara[0, 0] = True                                   # una celda junto a (-17,3, -66,2)
    destino = tmp_path / "clima.csv.gz"
    actualizacion.armar_clima_era5(meses, MALLA, mascara, destino, margen=0)
    tabla = pd.read_csv(destino)
    assert sorted(set(zip(tabla["lat"], tabla["lon"]))) == [(-17.3, -66.2)]


def test_el_cache_de_un_anio_se_rehace_solo_si_cambia_alguno_de_sus_meses(meses, tmp_path):
    destino = tmp_path / "clima_era5land_diario.csv.gz"
    actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino)
    cache19, cache20 = (meses / "diario" / f"era5land_diario_{a}.csv.gz" for a in (2019, 2020))
    assert (meses / "diario" / "era5land_diario_2019.firma.json").exists()
    viejo = time.time() - 1000
    for c in (cache19, cache20):
        os.utime(c, (viejo, viejo))
    for p in meses.glob("*.nc"):                    # archivos más nuevos que el caché, pero con los mismos datos
        os.utime(p, None)                           # (como tras copiar la carpeta o un git checkout)
    actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino)
    assert cache19.stat().st_mtime == pytest.approx(viejo) and cache20.stat().st_mtime == pytest.approx(viejo)
    escribir_mes(meses, 2020, 1, pd.date_range("2020-01-01", "2020-01-12"))     # enero se volvió a bajar, con más días
    actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino)
    assert cache19.stat().st_mtime > viejo + 600 and cache20.stat().st_mtime > viejo + 600
    assert pd.read_csv(destino)["fecha"].max() == "2020-01-12"


def test_un_cache_sin_firma_vale_si_es_mas_nuevo_que_sus_meses_y_desde_ahi_queda_firmado(meses, tmp_path):
    destino = tmp_path / "clima_era5land_diario.csv.gz"
    actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino)
    firma = meses / "diario" / "era5land_diario_2019.firma.json"
    firma.unlink()                                   # como los cachés armados con versiones anteriores
    actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino)
    assert firma.exists()


def test_la_tabla_escrita_es_identica_byte_a_byte_si_los_datos_no_cambian(meses, tmp_path):
    destino = tmp_path / "clima.csv.gz"
    actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino)
    antes = destino.read_bytes()
    time.sleep(1.1)                                  # gzip guarda la hora si no se le dice lo contrario
    actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino)
    assert destino.read_bytes() == antes
    import gzip
    assert b"\r\n" not in gzip.decompress(antes)


def test_sin_archivos_mensuales_es_un_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="clima_era5"):
        actualizacion.armar_clima_era5(tmp_path, MALLA, MASCARA, tmp_path / "x.csv.gz")


def test_la_semana_de_un_dia_empieza_el_lunes():
    assert actualizacion.lunes_de(date(2026, 11, 18)) == date(2026, 11, 16)
    assert actualizacion.lunes_de(date(2026, 11, 16)) == date(2026, 11, 16)


def test_semanas_con_datos_para_el_modo_operativo(ctx_sintetico):
    c = ctx_sintetico
    semanas = actualizacion.semanas_operativas(c, 2019)
    assert semanas[0] == tiempo.semana_de_iso(2019, 1)
    assert all(c.clima.disponible(s) and s <= c.n_semanas for s in semanas)
    assert not c.clima.disponible(semanas[-1] + 1) or semanas[-1] + 1 > c.n_semanas
    assert actualizacion.semana_completa(c, tiempo.semana_de_iso(2019, 50))
    assert not actualizacion.semana_completa(c, c.n_semanas)


def test_no_publica_una_tabla_que_empieza_tarde_o_tiene_huecos(meses, tmp_path):
    destino = tmp_path / "clima_era5land_diario.csv.gz"
    with pytest.raises(ValueError, match="2019-12-01"):
        actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino, exigir_desde=date(2012, 1, 1))
    assert not destino.exists()
    actualizacion.armar_clima_era5(meses, MALLA, MASCARA, destino, exigir_desde=date(2019, 12, 1))
    assert destino.exists()
    # un mes que falta en el medio
    escribir_mes(meses, 2020, 3, pd.date_range("2020-03-01", "2020-03-05"))
    with pytest.raises(ValueError, match="2020-01-11"):
        actualizacion.armar_clima_era5(meses, MALLA, MASCARA, tmp_path / "otro.csv.gz")


def resumen_eval(pr_auc, captura_5):
    fila = {"roc_auc": 0.9, "pr_auc": pr_auc, "captura_1": 0.2, "captura_5": captura_5, "captura_10": 0.7,
            "area_50": 0.04, "area_80": 0.15}
    return {"modelos_prueba": {"gbm_operativo": fila, "gbm_estacional": dict(fila, pr_auc=pr_auc / 2)},
            "modelos_validacion": {"gbm_operativo": fila, "gbm_estacional": fila}}


def test_comparar_dos_evaluaciones():
    t = actualizacion.comparar_evaluaciones(resumen_eval(0.10, 0.50), resumen_eval(0.12, 0.55))
    fila = t[(t["periodo"] == "prueba") & (t["modelo"] == "gbm_operativo") & (t["metrica"] == "pr_auc")].iloc[0]
    assert fila["antes"] == 0.10 and fila["despues"] == 0.12 and fila["cambio"] == pytest.approx(0.02)
    assert set(t["metrica"]) == {"roc_auc", "pr_auc", "captura_1", "captura_5", "captura_10", "area_50", "area_80"}
    # el área necesaria mejora cuando baja: la columna "mejor" lo tiene en cuenta
    area = t[(t["metrica"] == "area_50") & (t["periodo"] == "prueba")].iloc[0]
    assert area["mejor"] == "igual"
    assert fila["mejor"] == "después"
