"""Utilidades compartidas por las pruebas."""
import pytest

from incendios.rutas import Rutas
from PIL import Image, TiffImagePlugin


def escribir_tif(ruta, arr, oeste, norte, pix):
    """GeoTIFF mínimo (EPSG:4326) con las dos etiquetas que lee el proyecto."""
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (float(pix), float(pix), 0.0)
    ifd.tagtype[33550] = 12
    ifd[33922] = (0.0, 0.0, 0.0, float(oeste), float(norte), 0.0)
    ifd.tagtype[33922] = 12
    Image.fromarray(arr).save(str(ruta), format="TIFF", tiffinfo=ifd)


@pytest.fixture
def tif():
    return escribir_tif


@pytest.fixture(scope="session")
def ctx_sintetico(tmp_path_factory):
    from incendios import flujo
    from tests import mundo
    raiz = mundo.crear(tmp_path_factory.mktemp("datos"))
    rutas = Rutas(datos=raiz / "data", salida=tmp_path_factory.mktemp("salidas"))
    return flujo.cargar_todo(rutas, malla=mundo.MALLA, verboso=False)


@pytest.fixture(scope="session")
def muestra_sintetica(ctx_sintetico):
    from incendios import flujo
    c = ctx_sintetico
    return flujo.construir_muestra(c, c.semanas_utiles(2015, 2017), n_neg=300, semilla=1, verboso=False)


@pytest.fixture(scope="session")
def ctx_con_capas(tmp_path_factory):
    """El mismo mundo sintético, con capas extra en data/capas."""
    from incendios import flujo
    from tests import mundo
    raiz = mundo.crear(tmp_path_factory.mktemp("datos_capas"))
    mundo.crear_capas(raiz)
    rutas = Rutas(datos=raiz / "data", salida=tmp_path_factory.mktemp("salidas_capas"))
    return flujo.cargar_todo(rutas, malla=mundo.MALLA, verboso=False)


@pytest.fixture(scope="session")
def muestra_con_capas(ctx_con_capas):
    from incendios import flujo
    c = ctx_con_capas
    return flujo.construir_muestra(c, c.semanas_utiles(2015, 2017), n_neg=300, semilla=1, verboso=False)
