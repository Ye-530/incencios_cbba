"""Qué datos faltan y de dónde se bajan (sin hacer la descarga: eso lo hace el script).

Las funciones de aquí no tocan la red; solo calculan nombres de teselas, direcciones
y convierten respuestas al formato del proyecto, para poder probarlas.
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .clima import MERRA_DLAT, MERRA_DLON

PARAMETROS_POWER = ("T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR", "GWETTOP", "GWETROOT")
_COLUMNA_POWER = {"T2M_MAX": "temp_max", "T2M_MIN": "temp_min", "RH2M": "humedad_rel", "WS2M": "viento_vel",
                  "PRECTOTCORR": "precipitacion", "GWETTOP": "humedad_suelo_sup", "GWETROOT": "humedad_suelo_raiz"}
FUENTES_FIRMS = ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
# Días por consulta a la API de área de FIRMS. La documentación ha dicho 10; se usa 5 para quedar del
# lado seguro con los productos de archivo (SP): solo cuesta el doble de pedidos, lejos del límite de
# 5.000 cada 10 minutos. Si el servidor rechaza los pedidos, bájelo; si acepta más, se puede subir.
MAX_DIAS_FIRMS = 5


def _conteo_por_tesela(mascara, malla, grados):
    """Celdas del departamento dentro de cada tesela de `grados` × `grados` (por su esquina suroeste)."""
    lon, lat = malla.centros()
    LON, LAT = np.meshgrid(lon, lat)
    sur = (np.floor(LAT[mascara] / grados) * grados).astype(int)
    oeste = (np.floor(LON[mascara] / grados) * grados).astype(int)
    esquinas, cuenta = np.unique(np.column_stack([sur, oeste]), axis=0, return_counts=True)
    return [(int(s), int(o), int(n)) for (s, o), n in zip(esquinas, cuenta)]


def _ns(lat, ancho):
    return f"{'S' if lat < 0 else 'N'}{abs(lat):0{ancho}d}"


def _eo(lon, ancho):
    return f"{'W' if lon < 0 else 'E'}{abs(lon):0{ancho}d}"


def teselas_dem(mascara, malla, min_celdas=20):
    """Teselas Copernicus GLO-30 (1° × 1°) que contienen al menos `min_celdas` celdas del departamento."""
    return sorted(f"Copernicus_DSM_COG_10_{_ns(s, 2)}_00_{_eo(o, 3)}_00_DEM"
                  for s, o, n in _conteo_por_tesela(mascara, malla, 1) if n >= min_celdas)


def teselas_worldcover(mascara, malla, min_celdas=20):
    """Teselas ESA WorldCover 2021 (3° × 3°) necesarias."""
    return sorted(f"ESA_WorldCover_10m_2021_v200_{_ns(s, 2)}{_eo(o, 3)}_Map"
                  for s, o, n in _conteo_por_tesela(mascara, malla, 3) if n >= min_celdas)


def url_dem(nombre):
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{nombre}/{nombre}.tif"


def url_worldcover(nombre):
    return f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{nombre}.tif"


def faltantes(nombres, carpeta):
    """Las teselas de la lista que todavía no están en la carpeta."""
    carpeta = Path(carpeta)
    return [n for n in nombres if not (carpeta / f"{n}.tif").exists()]


def celdas_merra(mascara, malla):
    """Centros (lat, lon) de las celdas del reanálisis MERRA-2 que tocan el departamento."""
    lon, lat = malla.centros()
    LON, LAT = np.meshgrid(lon, lat)
    lat_c = np.floor(LAT[mascara] / MERRA_DLAT + 0.5) * MERRA_DLAT
    lon_c = np.floor(LON[mascara] / MERRA_DLON + 0.5) * MERRA_DLON
    return [(float(a), float(b)) for a, b in np.unique(np.column_stack([lat_c, lon_c]), axis=0)]


def url_power(lat, lon, inicio, fin):
    """Consulta diaria a NASA POWER para un punto."""
    f0, f1 = pd.Timestamp(inicio), pd.Timestamp(fin)
    return ("https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters={','.join(PARAMETROS_POWER)}&community=AG&longitude={lon}&latitude={lat}"
            f"&start={f0:%Y%m%d}&end={f1:%Y%m%d}&format=JSON")


def tabla_power(respuesta, nombre, lat, lon):
    """Convierte la respuesta JSON de POWER al esquema del archivo de clima del proyecto."""
    parametros = respuesta["properties"]["parameter"]
    dias = sorted(parametros["T2M_MAX"])
    fechas = pd.to_datetime(dias, format="%Y%m%d")
    tabla = pd.DataFrame({"provincia": nombre, "lat": lat, "lon": lon, "fecha": fechas.strftime("%Y-%m-%d"),
                          "anio": fechas.year, "mes": fechas.month, "dia": fechas.day})
    for parametro, columna in _COLUMNA_POWER.items():
        valores = np.array([parametros[parametro].get(d, np.nan) for d in dias], dtype=float)
        tabla[columna] = np.where(valores <= -900.0, np.nan, valores)     # -999 = sin dato
    return tabla


def tramos(inicio, fin, dias=MAX_DIAS_FIRMS):
    """Parte un periodo en tramos consecutivos de hasta `dias` días: [(fecha_inicio, n_dias), ...]."""
    f, fin = pd.Timestamp(inicio).date(), pd.Timestamp(fin).date()
    salida = []
    while f <= fin:
        n = min(dias, (fin - f).days + 1)
        salida.append((str(f), n))
        f = f + timedelta(days=n)
    return salida


def url_firms(clave, fuente, recuadro, fecha, dias):
    """API de área de NASA FIRMS: detecciones en un recuadro (oeste, sur, este, norte) durante `dias` días."""
    if not 1 <= dias <= MAX_DIAS_FIRMS:
        raise ValueError(f"FIRMS admite entre 1 y {MAX_DIAS_FIRMS} días por consulta, no {dias}")
    area = ",".join(str(v) for v in recuadro)
    return f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{clave}/{fuente}/{area}/{dias}/{fecha}"
