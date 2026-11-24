"""Escritura de GeoTIFF (EPSG:4326) con Pillow, para abrir los mapas en QGIS o ArcGIS."""
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

SIN_DATO = -9999.0

_ESCALA, _ANCLA, _CLAVES, _GDAL_SIN_DATO = 33550, 33922, 34735, 42113
_DOBLE, _CORTO, _ASCII = 12, 3, 2
# GeoKeyDirectory: modelo geográfico (2), píxel como área (1), WGS 84 (4326), grados (9102)
_CLAVES_WGS84 = (1, 1, 0, 4,
                 1024, 0, 1, 2,
                 1025, 0, 1, 1,
                 2048, 0, 1, 4326,
                 2054, 0, 1, 9102)


def escribir_geotiff(ruta, arr, malla, sin_dato=None):
    """Guarda un arreglo de la malla como GeoTIFF. Los NaN se escriben como SIN_DATO."""
    arr = np.asarray(arr)
    if arr.shape != malla.forma:
        raise ValueError(f"El arreglo tiene forma {arr.shape} y la malla {malla.forma}")
    if arr.dtype.kind == "f":
        sin_dato = SIN_DATO if sin_dato is None else sin_dato
        arr = np.where(np.isfinite(arr), arr, sin_dato).astype(np.float32)
    else:
        arr = arr.astype(np.uint8)
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[_ESCALA] = (float(malla.celda), float(malla.celda), 0.0)
    ifd.tagtype[_ESCALA] = _DOBLE
    ifd[_ANCLA] = (0.0, 0.0, 0.0, float(malla.oeste), float(malla.norte), 0.0)
    ifd.tagtype[_ANCLA] = _DOBLE
    ifd[_CLAVES] = _CLAVES_WGS84
    ifd.tagtype[_CLAVES] = _CORTO
    if sin_dato is not None:
        ifd[_GDAL_SIN_DATO] = f"{sin_dato:g}"
        ifd.tagtype[_GDAL_SIN_DATO] = _ASCII
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(ruta), format="TIFF", tiffinfo=ifd, compression="tiff_deflate")
    return ruta
