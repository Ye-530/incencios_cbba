"""El paso de descarga con servicios falsos: lo que hace con lo que respondería cada servidor."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from incendios import credenciales, era5
from incendios.pasos import descargar as dd
from incendios.rutas import Rutas
from tests.test_actualizacion import MALLA, MASCARA
from tests.test_era5 import cubos_sinteticos, escribir_nc, segundos



class CdsFalso:
    """Responde como el CDS con datos sintéticos; `negar(pedido)` devuelve el error a levantar, si hay."""

    def __init__(self, negar=lambda pedido: None):
        self.negar = negar
        self.pedidos = []

    def retrieve(self, dataset, pedido, destino):
        assert dataset == era5.DATASET
        self.pedidos.append(pedido)
        error = self.negar(pedido)
        if error:
            raise Exception(error)
        dias = pd.to_datetime([f"{pedido['year']}-{pedido['month']}-{d}" for d in pedido["day"]])
        cubos = cubos_sinteticos(dias)
        escribir_nc(Path(destino), {n: c["datos"] for n, c in cubos.items()}, segundos(cubos["t2m"]["tiempo"]))


@pytest.fixture
def cred():
    c = credenciales.cargar(Path("no-existe"), entorno={})
    c["copernicus_cds"]["key"] = "clave-de-prueba"
    return c


def preparar(monkeypatch, cliente):
    monkeypatch.setattr(dd, "_cliente_cds", lambda cred: cliente)
    monkeypatch.setattr(dd, "falta_paquete", lambda modulo, paquete: False)


def test_baja_los_meses_y_arma_el_clima_era5(tmp_path, monkeypatch, cred):
    cliente = CdsFalso()
    preparar(monkeypatch, cliente)
    ok = dd.descargar_clima_era5(Rutas.en(tmp_path), MALLA, MASCARA, cred, paralelo=2, hoy=date(2026, 11, 18),
                                 desde=date(2026, 8, 1))
    assert ok
    tabla = pd.read_csv(Rutas.en(tmp_path).clima_era5)
    assert tabla["fecha"].min() == "2026-08-01" and tabla["fecha"].max() == "2026-11-12"
    assert sorted((p["year"], p["month"]) for p in cliente.pedidos) == [
        ("2026", "08"), ("2026", "09"), ("2026", "10"), ("2026", "11")]
    # segunda vez: solo los últimos tres meses (ERA5T se corrige)
    cliente.pedidos.clear()
    dd.descargar_clima_era5(Rutas.en(tmp_path), MALLA, MASCARA, cred, paralelo=1, hoy=date(2026, 11, 18), desde=date(2026, 8, 1))
    assert sorted(p["month"] for p in cliente.pedidos) == ["09", "10", "11"]


def test_un_mes_que_el_cds_no_acepta_entero_se_pide_en_dos_mitades(tmp_path, monkeypatch, cred):
    cliente = CdsFalso(negar=lambda p: "cost limits exceeded" if p["month"] == "09" and len(p["day"]) > 20 else None)
    preparar(monkeypatch, cliente)
    assert dd.descargar_clima_era5(Rutas.en(tmp_path), MALLA, MASCARA, cred, paralelo=1, hoy=date(2026, 11, 18),
                                   desde=date(2026, 8, 1))
    carpeta = Rutas.en(tmp_path).era5_meses
    assert [p.name for p in sorted(carpeta.glob("era5land_2026_09*.nc"))] == [
        "era5land_2026_09_a.nc", "era5land_2026_09_b.nc"]
    tabla = pd.read_csv(Rutas.en(tmp_path).clima_era5)
    assert pd.date_range("2026-08-01", "2026-11-12").strftime("%Y-%m-%d").isin(tabla["fecha"]).all()


def test_si_los_ultimos_dias_no_estan_publicados_pide_menos(tmp_path, monkeypatch, cred):
    cliente = CdsFalso(negar=lambda p: "None of the data you have requested is available yet"
                       if p["month"] == "11" and p["day"][-1] > "10" else None)
    preparar(monkeypatch, cliente)
    dd.descargar_clima_era5(Rutas.en(tmp_path), MALLA, MASCARA, cred, paralelo=1, hoy=date(2026, 11, 18), desde=date(2026, 8, 1))
    assert pd.read_csv(Rutas.en(tmp_path).clima_era5)["fecha"].max() == "2026-11-10"


def test_un_mes_que_falla_no_publica_un_clima_con_huecos(tmp_path, monkeypatch, cred):
    cliente = CdsFalso(negar=lambda p: "servidor caído" if p["month"] == "09" else None)
    preparar(monkeypatch, cliente)
    ok = dd.descargar_clima_era5(Rutas.en(tmp_path), MALLA, MASCARA, cred, paralelo=1, hoy=date(2026, 11, 18),
                                 desde=date(2026, 8, 1))
    assert not ok and not (Rutas.en(tmp_path).clima_era5).exists()


def test_sin_clave_de_copernicus_no_pide_nada(tmp_path, monkeypatch, cred, capsys):
    cred["copernicus_cds"]["key"] = ""
    cliente = CdsFalso()
    preparar(monkeypatch, cliente)
    assert not dd.descargar_clima_era5(Rutas.en(tmp_path), MALLA, MASCARA, cred, hoy=date(2026, 11, 18))
    assert cliente.pedidos == [] and "copernicus_cds.key" in capsys.readouterr().out


def test_los_focos_del_anio_en_curso_se_completan_sin_repetir(tmp_path, monkeypatch):
    def serie(clave, disponible, prefijo, desde, hasta):
        dias = pd.date_range(desde, hasta).strftime("%Y-%m-%d")
        return pd.DataFrame({"latitude": -17.0, "longitude": -65.0, "acq_date": dias, "acq_time": 1730,
                             "confidence": "n", "frp": 5.0, "type": 0, "daynight": "D"})
    pedidos = []
    monkeypatch.setattr(dd, "serie_firms", lambda *a: pedidos.append(a[3:]) or serie(*a))
    destino = tmp_path / "viirs-noaa20_2026_Bolivia.csv"
    dd.completar_archivo_focos(destino, "k", {}, "VIIRS_NOAA20", date(2026, 11, 2), date(2026, 11, 10))
    nuevas = dd.completar_archivo_focos(destino, "k", {}, "VIIRS_NOAA20", date(2026, 11, 2), date(2026, 11, 16))
    assert pedidos[-1] == (date(2026, 11, 8), date(2026, 11, 16))          # repasa los dos últimos días
    tabla = pd.read_csv(destino)
    assert nuevas == 9 and tabla["acq_date"].tolist() == list(pd.date_range("2026-11-02", "2026-11-16").strftime("%Y-%m-%d"))
    assert list(tabla.columns) == dd.COLUMNAS_FOCOS
    assert dd.completar_archivo_focos(destino, "k", {}, "VIIRS_NOAA20", date(2026, 11, 2), date(2026, 11, 16)) == 0


def test_si_falta_un_paquete_dice_como_instalarlo_con_el_proyecto(capsys):
    assert dd.falta_paquete("paquete_que_no_existe_xyz", "cdsapi") is True
    assert 'python -m pip install -e ".[descargas]"' in capsys.readouterr().out
    assert dd.falta_paquete("json", "json") is False


def test_descargar_sin_opciones_muestra_el_diagnostico_y_como_bajar(monkeypatch, capsys):
    llamadas = []
    monkeypatch.setattr(dd, "malla_cochabamba", lambda: "malla")
    monkeypatch.setattr(dd, "leer_credenciales", lambda: "cred")
    monkeypatch.setattr(dd, "diagnostico", lambda rutas, malla, cred: llamadas.append((malla, cred)) or 0)
    assert dd.main([]) == 0
    assert llamadas == [("malla", "cred")]
    assert "python -m incendios descargar --que mejorados" in capsys.readouterr().out
