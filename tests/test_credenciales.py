import json
from pathlib import Path

import pytest

from incendios import credenciales


def _escribir(carpeta, datos):
    (carpeta / "credenciales.json").write_text(json.dumps(datos), encoding="utf-8")


def test_cargar_lee_el_archivo_y_completa_lo_que_falta(tmp_path):
    _escribir(tmp_path, {"copernicus_cds": {"key": " abc-123 "}})
    c = credenciales.cargar(tmp_path, entorno={})
    assert c["copernicus_cds"]["key"] == "abc-123"
    assert c["copernicus_cds"]["url"] == "https://cds.climate.copernicus.eu/api"
    assert c["earth_engine"]["proyecto"] == ""
    assert c["firms"]["map_key"] == ""
    assert credenciales.faltantes(c) == ["earth_engine", "firms"]


def test_sin_archivo_faltan_todas(tmp_path):
    c = credenciales.cargar(tmp_path, entorno={})
    assert credenciales.faltantes(c) == ["copernicus_cds", "earth_engine", "firms"]


def test_las_variables_de_entorno_tienen_prioridad(tmp_path):
    _escribir(tmp_path, {"firms": {"map_key": "del-archivo"}})
    c = credenciales.cargar(tmp_path, entorno={"FIRMS_MAP_KEY": "del-entorno", "EE_PROYECTO": "mi-proyecto",
                                               "CDSAPI_KEY": "k"})
    assert c["firms"]["map_key"] == "del-entorno"
    assert c["earth_engine"]["proyecto"] == "mi-proyecto"
    assert c["copernicus_cds"]["key"] == "k"


def test_ignora_comentarios_y_servicios_desconocidos(tmp_path):
    _escribir(tmp_path, {"_LEAME": "texto", "otro": {"x": 1}, "earth_engine": {"proyecto": "p-1"}})
    c = credenciales.cargar(tmp_path, entorno={})
    assert set(c) == {"copernicus_cds", "earth_engine", "firms"}
    assert c["earth_engine"]["proyecto"] == "p-1"


def test_json_mal_formado_da_un_error_que_nombra_el_archivo(tmp_path):
    (tmp_path / "credenciales.json").write_text("{ no es json", encoding="utf-8")
    with pytest.raises(ValueError, match="credenciales.json"):
        credenciales.cargar(tmp_path, entorno={})


def test_ocultar_no_muestra_la_clave_completa():
    assert credenciales.ocultar("1234abcd-0000-1111-2222-333344445555") == "1234…5555"
    assert credenciales.ocultar("corta") == "…"
    assert credenciales.ocultar("") == "(vacía)"


def test_la_plantilla_no_trae_claves_y_tiene_todos_los_servicios():
    raiz = Path(__file__).resolve().parents[1]
    plantilla = json.loads((raiz / "credenciales.ejemplo.json").read_text(encoding="utf-8"))
    assert {"copernicus_cds", "earth_engine", "firms"} <= set(plantilla)
    c = credenciales.cargar(raiz, entorno={}, archivo="credenciales.ejemplo.json")
    assert credenciales.faltantes(c) == ["copernicus_cds", "earth_engine", "firms"]


def test_el_archivo_real_esta_fuera_de_git():
    raiz = Path(__file__).resolve().parents[1]
    ignorados = (raiz / ".gitignore").read_text(encoding="utf-8").split()
    assert "credenciales.json" in ignorados
