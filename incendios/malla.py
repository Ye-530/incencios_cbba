"""Malla regular lat/lon y rasterización de polígonos GeoJSON sobre ella."""
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from . import config

KM_POR_GRADO_LAT = 110.574
KM_POR_GRADO_LON_ECUADOR = 111.320


@dataclass(frozen=True)
class Malla:
    oeste: float
    norte: float
    celda: float
    ncols: int
    nfilas: int

    @property
    def forma(self):
        return (self.nfilas, self.ncols)

    def fila_col(self, lon, lat):
        """Índices (fila, columna) de cada punto; -1 en ambos si cae fuera."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        cols = np.floor((lon - self.oeste) / self.celda).astype(np.int64)
        filas = np.floor((self.norte - lat) / self.celda).astype(np.int64)
        fuera = (cols < 0) | (cols >= self.ncols) | (filas < 0) | (filas >= self.nfilas)
        filas[fuera] = -1
        cols[fuera] = -1
        return filas, cols

    def id_celda(self, lon, lat):
        filas, cols = self.fila_col(lon, lat)
        ids = filas * self.ncols + cols
        ids[filas < 0] = -1
        return ids

    def centros(self):
        lon = self.oeste + (np.arange(self.ncols) + 0.5) * self.celda
        lat = self.norte - (np.arange(self.nfilas) + 0.5) * self.celda
        return lon, lat

    def area_km2(self):
        _, lat = self.centros()
        alto = self.celda * KM_POR_GRADO_LAT
        ancho = self.celda * KM_POR_GRADO_LON_ECUADOR * np.cos(np.deg2rad(lat))
        return np.repeat((alto * ancho)[:, None], self.ncols, axis=1)


def malla_cochabamba():
    return Malla(config.OESTE, config.NORTE, config.CELDA, config.NCOLS, config.NFILAS)


def _poligonos(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    raise ValueError(f"Geometría no soportada: {geom['type']}")


def rasterizar_geometria(malla, geom):
    """Máscara de las celdas cuyo centro cae dentro del polígono (los anillos interiores son huecos)."""
    lon, lat = malla.centros()
    LON, LAT = np.meshgrid(lon, lat)
    puntos = np.column_stack([LON.ravel(), LAT.ravel()])
    dentro = np.zeros(len(puntos), dtype=bool)
    for anillos in _poligonos(geom):
        parte = Path(np.asarray(anillos[0], dtype=float)[:, :2]).contains_points(puntos)
        for hueco in anillos[1:]:
            parte &= ~Path(np.asarray(hueco, dtype=float)[:, :2]).contains_points(puntos)
        dentro |= parte
    return dentro.reshape(malla.forma)


def rasterizar_capas(malla, geojson, campo):
    """Id de entidad por celda (-1 si ninguna) y la lista de nombres en el orden de los ids."""
    ids = np.full(malla.forma, -1, dtype=np.int16)
    nombres = []
    for i, entidad in enumerate(geojson["features"]):
        nombres.append(entidad["properties"][campo])
        ids[rasterizar_geometria(malla, entidad["geometry"])] = i
    return ids, nombres
