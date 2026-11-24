"""Lectura de GeoTIFF (cobertura de suelo y relieve) y agregación a la malla.

Solo usa Pillow: no hace falta GDAL ni rasterio. Las teselas deben estar en
EPSG:4326 y alineadas con la malla (WorldCover y Copernicus DEM lo están).
"""
from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image

from .malla import KM_POR_GRADO_LAT, KM_POR_GRADO_LON_ECUADOR

CLASES_WORLDCOVER = {
    10: "arbol", 20: "arbusto", 30: "pasto", 40: "cultivo", 50: "urbano",
    60: "desnudo", 70: "nieve", 80: "agua", 90: "humedal",
}
VARS_TERRENO = ("elev_media", "elev_rango", "pendiente", "nortitud", "estitud")

ETIQUETA_ESCALA = 33550   # ModelPixelScaleTag
ETIQUETA_ANCLA = 33922    # ModelTiepointTag

Ventana = namedtuple("Ventana", "fila col nfilas ncols pix_fila pix_col factor")


def leer_geotiff(ruta):
    """Devuelve (arreglo, oeste, norte, tamaño de píxel en grados)."""
    limite = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # las teselas de 10 m tienen 1.296 millones de píxeles
    try:
        with Image.open(str(ruta)) as im:
            escala = im.tag_v2.get(ETIQUETA_ESCALA)
            ancla = im.tag_v2.get(ETIQUETA_ANCLA)
            if escala is None or ancla is None:
                raise ValueError(f"{ruta} no tiene georreferencia GeoTIFF")
            arr = np.asarray(im)
    finally:
        Image.MAX_IMAGE_PIXELS = limite
    pix = float(escala[0])
    if abs(float(escala[1]) - pix) > 1e-12:
        raise ValueError(f"{ruta}: píxeles no cuadrados")
    i, j, _, x, y, _ = (float(v) for v in ancla[:6])
    return arr, x - i * pix, y + j * pix, pix


def ventana_en_malla(malla, oeste, norte, pix, forma):
    """Parte de una tesela que cae sobre la malla, o None si no la toca."""
    factor_real = malla.celda / pix
    factor = int(round(factor_real))
    desfase_col = (oeste - malla.oeste) / malla.celda
    desfase_fila = (malla.norte - norte) / malla.celda
    dc, df = int(round(desfase_col)), int(round(desfase_fila))
    if (factor < 1 or abs(factor_real - factor) > 1e-6
            or abs(desfase_col - dc) > 1e-6 or abs(desfase_fila - df) > 1e-6):
        raise ValueError(
            f"Tesela no alineada con la malla (origen {oeste}, {norte}; píxel {pix})")
    ncols_t, nfilas_t = forma[1] // factor, forma[0] // factor
    c0, c1 = max(0, dc), min(malla.ncols, dc + ncols_t)
    f0, f1 = max(0, df), min(malla.nfilas, df + nfilas_t)
    if c0 >= c1 or f0 >= f1:
        return None
    return Ventana(f0, c0, f1 - f0, c1 - c0, (f0 - df) * factor, (c0 - dc) * factor, factor)


def fracciones_por_bloque(arr, factor, clases):
    """Fracción de píxeles de cada clase en bloques de factor × factor."""
    nf, nc = arr.shape[0] // factor, arr.shape[1] // factor
    salida = np.empty((nf, nc, len(clases)), dtype=np.float32)
    paso = max(1, 50_000_000 // max(1, arr.shape[1] * factor))  # filas de bloques por tira
    for f0 in range(0, nf, paso):
        f1 = min(nf, f0 + paso)
        tira = arr[f0 * factor:f1 * factor, :nc * factor]
        for k, clase in enumerate(clases):
            iguales = (tira == clase).reshape(f1 - f0, factor, nc, factor)
            salida[f0:f1, :, k] = iguales.mean(axis=(1, 3))
    return salida


def terreno_por_bloque(dem, factor, pix, lat_norte):
    """Elevación, relieve local, pendiente (grados) y orientación por bloque.

    `nortitud` y `estitud` son las componentes norte y este de la dirección hacia
    la que baja la ladera, multiplicadas por el seno de la pendiente: valen 0 en
    terreno plano. En el hemisferio sur, nortitud > 0 es ladera soleada.
    """
    dem = np.asarray(dem, dtype=np.float32)
    nf, nc = dem.shape[0] // factor, dem.shape[1] // factor
    lat = lat_norte - (np.arange(dem.shape[0]) + 0.5) * pix
    metros_fila = pix * KM_POR_GRADO_LAT * 1000.0
    metros_col = (pix * KM_POR_GRADO_LON_ECUADOR * 1000.0 * np.cos(np.deg2rad(lat)))[:, None]
    dz_fila, dz_col = np.gradient(dem)
    dz_norte = -dz_fila / metros_fila          # las filas crecen hacia el sur
    dz_este = dz_col / metros_col
    gradiente = np.hypot(dz_norte, dz_este)
    norma = np.sqrt(1.0 + gradiente ** 2)

    def bloques(a):
        return a[:nf * factor, :nc * factor].reshape(nf, factor, nc, factor)

    b = bloques(dem)
    return {
        "elev_media": b.mean(axis=(1, 3)).astype(np.float32),
        "elev_rango": (b.max(axis=(1, 3)) - b.min(axis=(1, 3))).astype(np.float32),
        "pendiente": bloques(np.degrees(np.arctan(gradiente))).mean(axis=(1, 3)).astype(np.float32),
        "nortitud": bloques(-dz_norte / norma).mean(axis=(1, 3)).astype(np.float32),
        "estitud": bloques(-dz_este / norma).mean(axis=(1, 3)).astype(np.float32),
    }


def archivos_de(carpetas, patron):
    """Archivos que cumplen el patrón en una o varias carpetas. Si el mismo nombre está en más de
    una, cuenta el de la última (lo bajado reemplaza a lo original)."""
    if isinstance(carpetas, (str, Path)):
        carpetas = [carpetas]
    por_nombre = {}
    for carpeta in map(Path, carpetas):
        if carpeta.is_dir():
            for ruta in carpeta.glob(patron):
                por_nombre[ruta.name] = ruta
    return [por_nombre[n] for n in sorted(por_nombre)]


def _teselas(carpetas):
    return archivos_de(carpetas, "*.tif")


def variables_estaticas(malla, dir_cobertura, dir_terreno):
    """Variables fijas por celda. NaN donde ninguna tesela cubre la celda."""
    nombres_lc = [f"lc_{n}" for n in CLASES_WORLDCOVER.values()]
    est = {n: np.full(malla.forma, np.nan, dtype=np.float32) for n in (*nombres_lc, *VARS_TERRENO)}

    for ruta in _teselas(dir_cobertura):
        arr, oeste, norte, pix = leer_geotiff(ruta)
        v = ventana_en_malla(malla, oeste, norte, pix, arr.shape)
        if v is None:
            continue
        recorte = arr[v.pix_fila:v.pix_fila + v.nfilas * v.factor,
                      v.pix_col:v.pix_col + v.ncols * v.factor]
        fracciones = fracciones_por_bloque(recorte, v.factor, list(CLASES_WORLDCOVER))
        for k, nombre in enumerate(nombres_lc):
            est[nombre][v.fila:v.fila + v.nfilas, v.col:v.col + v.ncols] = fracciones[:, :, k]
        del arr, recorte

    for ruta in _teselas(dir_terreno):
        dem, oeste, norte, pix = leer_geotiff(ruta)
        v = ventana_en_malla(malla, oeste, norte, pix, dem.shape)
        if v is None:
            continue
        terreno = terreno_por_bloque(dem, v.factor, pix, norte)
        bf, bc = v.pix_fila // v.factor, v.pix_col // v.factor
        for nombre in VARS_TERRENO:
            est[nombre][v.fila:v.fila + v.nfilas, v.col:v.col + v.ncols] = \
                terreno[nombre][bf:bf + v.nfilas, bc:bc + v.ncols]
    return est
