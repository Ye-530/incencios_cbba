"""Claves de los servicios de datos, leídas de un archivo privado que nunca va al código.

El archivo `credenciales.json` vive en la carpeta del proyecto y está fuera de git. Cada
campo puede reemplazarse con una variable de entorno (útil en una tarea programada):

    CDSAPI_URL, CDSAPI_KEY   Copernicus Climate Data Store (clima ERA5-Land)
    EE_PROYECTO              ID del proyecto de Google Cloud registrado en Earth Engine
    FIRMS_MAP_KEY            clave de NASA FIRMS (focos de calor)
"""
import copy
import json
import os
from pathlib import Path

ARCHIVO = "credenciales.json"
URL_CDS = "https://cds.climate.copernicus.eu/api"
DEFECTO = {
    "copernicus_cds": {"url": URL_CDS, "key": ""},
    "earth_engine": {"proyecto": ""},
    "firms": {"map_key": ""},
}
_ENTORNO = {
    "CDSAPI_URL": ("copernicus_cds", "url"),
    "CDSAPI_KEY": ("copernicus_cds", "key"),
    "EE_PROYECTO": ("earth_engine", "proyecto"),
    "FIRMS_MAP_KEY": ("firms", "map_key"),
}
_CAMPO_PRINCIPAL = {"copernicus_cds": "key", "earth_engine": "proyecto", "firms": "map_key"}
NOMBRE_SERVICIO = {"copernicus_cds": "Copernicus (clima ERA5-Land)",
                   "earth_engine": "Google Earth Engine (capas anuales)",
                   "firms": "NASA FIRMS (focos de calor)"}


def dir_proyecto():
    return Path(__file__).resolve().parents[1]


def cargar(carpeta=None, entorno=None, archivo=ARCHIVO):
    """Credenciales completas: lo que haya en el archivo, con las variables de entorno encima."""
    carpeta = Path(carpeta) if carpeta else dir_proyecto()
    entorno = os.environ if entorno is None else entorno
    datos = copy.deepcopy(DEFECTO)
    ruta = carpeta / archivo
    if ruta.exists():
        try:
            leido = json.loads(ruta.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{ruta} no es un JSON válido (línea {error.lineno}, columna {error.colno}): "
                             "revise comas, llaves y comillas dobles") from error
        for servicio, valores in leido.items():
            if servicio in datos and isinstance(valores, dict):
                datos[servicio].update({campo: str(valor).strip() for campo, valor in valores.items()
                                        if valor is not None})
    for variable, (servicio, campo) in _ENTORNO.items():
        if entorno.get(variable):
            datos[servicio][campo] = str(entorno[variable]).strip()
    return datos


def faltantes(credenciales):
    """Servicios cuyo dato principal (clave o proyecto) está vacío."""
    return [servicio for servicio, campo in _CAMPO_PRINCIPAL.items() if not credenciales[servicio].get(campo)]


def ocultar(valor):
    """Para mostrar una clave en pantalla sin revelarla."""
    valor = str(valor or "")
    if not valor:
        return "(vacía)"
    if len(valor) < 12:
        return "…"
    return f"{valor[:4]}…{valor[-4:]}"
