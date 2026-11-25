"""Capas anuales de Google Earth Engine llevadas a la malla de 500 m del proyecto.

Earth Engine hace el trabajo pesado: agrega los píxeles de 30 m a 500 m sobre la malla del
proyecto y devuelve arreglos chicos con `computePixels`. Así no se bajan imágenes de gigas.

    quemas           Área quemada MODIS MCD64A1 (500 m, mensual desde 2000): fracción de la celda
                     quemada en cada año.
    perdida_bosque   Global Forest Change de Hansen (30 m): fracción de la celda que perdió bosque
                     cada año (2001–2024) y cobertura de árboles en 2000.
    cobertura        MapBiomas Bolivia (30 m, anual): fracción de la celda en cada tipo de
                     cobertura y la conversión de vegetación natural a uso agropecuario en 3 años.
    poblacion        WorldPop R2025A (100 m, 2015–2030): personas por celda.

La conexión a Earth Engine vive en `ClienteEE`; el resto se prueba con un cliente falso.
"""
from pathlib import Path

import numpy as np

from . import capas

LIMITE_BYTES = 24_000_000          # Earth Engine acepta hasta 48 MB por solicitud: queda margen
MAX_PIXELES_POR_CELDA = 1024       # píxeles de entrada que se promedian por celda (30 m → 500 m: ~400)

# Clases de MapBiomas Bolivia (leyenda del catálogo de Earth Engine, colección 1985–2024) agrupadas.
CLASES_MAPBIOMAS = {
    "bosque": [1, 3, 4, 6],
    "natural": [10, 11, 12, 13, 66, 81, 82],
    "pasto": [15],
    "agricultura": [14, 18, 39, 72],
    "mosaico": [21],
    "no_vegetado": [22, 23, 24, 25, 29, 30, 61, 68],
    "agua": [26, 31, 33, 34],
}
NO_OBSERVADO = 27
NATURAL = CLASES_MAPBIOMAS["bosque"] + CLASES_MAPBIOMAS["natural"]
AGROPECUARIO = CLASES_MAPBIOMAS["pasto"] + CLASES_MAPBIOMAS["agricultura"] + CLASES_MAPBIOMAS["mosaico"]

CAPAS = {
    "quemas": {
        "activo": "MODIS/061/MCD64A1",
        "descripcion": "Área quemada MODIS MCD64A1 (500 m): fracción de la celda quemada en el año",
        "anuales": ["quema"], "estaticas": [], "desde": 2001, "bandas_por_pedido": 8,
    },
    "perdida_bosque": {
        "activo": "UMD/hansen/global_forest_change_2025_v1_13",
        "descripcion": "Hansen Global Forest Change v1.13 (30 m): fracción de la celda que perdió bosque en el año",
        "anuales": ["perdida"], "estaticas": ["arboles_2000"], "desde": 2001, "bandas_por_pedido": 6,
    },
    "cobertura": {
        "activo": "projects/mapbiomas-public/assets/bolivia/lulc/v1",
        "descripcion": "MapBiomas Bolivia (30 m): fracción de la celda en cada cobertura y conversión a uso agropecuario",
        "anuales": list(CLASES_MAPBIOMAS) + ["conversion3"], "estaticas": [], "desde": 2010,
        "bandas_por_pedido": 4,
    },
    "poblacion": {
        "activo": "WorldPop/GP/100m/R2025A/v1/pop",
        "descripcion": "WorldPop R2025A (100 m): personas por celda",
        "anuales": ["poblacion"], "estaticas": [], "desde": 2015, "bandas_por_pedido": 8,
    },
}
ANIOS_HANSEN = (2001, 2025)


def transformacion(malla):
    return [malla.celda, 0, malla.oeste, 0, -malla.celda, malla.norte]


def grilla(malla):
    """Malla de salida para computePixels: exactamente la malla del proyecto."""
    return {"dimensions": {"width": malla.ncols, "height": malla.nfilas},
            "affineTransform": {"scaleX": malla.celda, "shearX": 0, "translateX": malla.oeste,
                                "shearY": 0, "scaleY": -malla.celda, "translateY": malla.norte},
            "crsCode": "EPSG:4326"}


def trozos(lista, n):
    return [list(lista[i:i + n]) for i in range(0, len(lista), n)]


def bandas_por_pedido(malla, deseadas):
    return max(1, min(int(deseadas), LIMITE_BYTES // (malla.ncols * malla.nfilas * 4)))


def a_diccionario(resultado, nombres):
    """Resultado de computePixels (arreglo estructurado o común) → {banda: float32 2D}, -1 = sin dato."""
    arr = np.asarray(resultado)
    if arr.dtype.names:
        salida = {n: np.array(arr[n], dtype=np.float32) for n in nombres}
    elif arr.ndim == 2 and len(nombres) == 1:
        salida = {nombres[0]: arr.astype(np.float32)}
    elif arr.ndim == 3 and arr.shape[-1] == len(nombres):
        salida = {n: arr[..., i].astype(np.float32) for i, n in enumerate(nombres)}
    else:
        raise ValueError(f"Se pidieron {len(nombres)} bandas y llegó un arreglo de forma {arr.shape}")
    for v in salida.values():
        v[v < -0.5] = np.nan
    return salida


def descargar(nombre, malla, carpeta, cliente, verboso=False):
    """Baja (o completa) una capa y la guarda en `carpeta/<nombre>.npz`. Devuelve la ruta.

    Solo pide los años que todavía no están en el archivo, y guarda después de cada año: si
    se corta, la próxima vez sigue desde ahí.
    """
    definicion = CAPAS[nombre]
    ruta = Path(carpeta) / f"{nombre}.npz"
    previa = capas.cargar(ruta, malla) if ruta.exists() else None
    anios_disp = sorted(a for a in cliente.anios(nombre) if a >= definicion["desde"])
    tenidos = set(previa.anios) if previa else set()
    anuales = {v: dict(zip(previa.anios, previa.anuales[v])) if previa else {} for v in definicion["anuales"]}
    estaticas = dict(previa.estaticas) if previa else {}
    meta = {"activo": definicion["activo"], "descripcion": definicion["descripcion"]}
    por_pedido = bandas_por_pedido(malla, definicion["bandas_por_pedido"])

    def pedir(bandas):
        datos = {}
        for trozo in trozos(bandas, por_pedido):
            datos.update(cliente.pixeles(nombre, trozo, malla))
        return datos

    def guardar_todo():
        anios = sorted(set.intersection(*[set(d) for d in anuales.values()])) if anuales else []
        capas.guardar(ruta, nombre, malla, anios,
                      {v: np.stack([anuales[v][a] for a in anios]) if anios else
                       np.zeros((0,) + malla.forma, np.float32) for v in anuales},
                      estaticas, meta)

    faltan_fijas = [e for e in definicion["estaticas"] if e not in estaticas]
    if faltan_fijas:
        estaticas.update(pedir(faltan_fijas))
    nuevos = [a for a in anios_disp if a not in tenidos]
    for anio in nuevos:
        if verboso:
            print(f"    {nombre} {anio}...", flush=True)
        datos = pedir([f"{v}_{anio}" for v in definicion["anuales"]])
        for v in definicion["anuales"]:
            anuales[v][anio] = datos[f"{v}_{anio}"]
        guardar_todo()
    if not nuevos and (faltan_fijas or not ruta.exists()):
        guardar_todo()
    return ruta


class ClienteEE:
    """Conexión real a Earth Engine. Necesita `pip install earthengine-api`, haber ejecutado una vez
    `earthengine authenticate` y el ID de un proyecto de Google Cloud registrado en Earth Engine."""

    def __init__(self, proyecto, autenticar=False):
        try:
            import ee
        except ImportError as error:
            raise ImportError("Falta el paquete de Earth Engine: pip install earthengine-api") from error
        if not proyecto:
            raise ValueError("Falta el ID del proyecto de Earth Engine en credenciales.json (earth_engine.proyecto)")
        if autenticar:
            ee.Authenticate()            # abre el navegador para iniciar sesión con la cuenta de Google
        ee.Initialize(project=proyecto)
        self.ee = ee

    # ------------------------------------------------------------------ años disponibles
    def anios(self, nombre):
        ee = self.ee
        activo = CAPAS[nombre]["activo"]
        if nombre == "quemas":
            col = ee.ImageCollection(activo)
            ultima = ee.Date(col.aggregate_max("system:time_start"))
            anio, mes = ultima.get("year").getInfo(), ultima.get("month").getInfo()
            return list(range(2001, anio + 1 if mes == 12 else anio))       # solo años completos
        if nombre == "perdida_bosque":
            return list(range(ANIOS_HANSEN[0], ANIOS_HANSEN[1] + 1))
        if nombre == "cobertura":
            anios = ee.ImageCollection(activo).aggregate_array(self._anio_mapbiomas()).distinct().getInfo()
            return sorted(int(a) for a in anios)
        if nombre == "poblacion":
            col = ee.ImageCollection(activo).filter(ee.Filter.eq("country", "BOL"))
            return sorted(int(a) for a in col.aggregate_array("year").distinct().getInfo())
        raise KeyError(nombre)

    # ------------------------------------------------------------------ imágenes por banda
    def _a_malla(self, imagen, malla, reductor):
        return (imagen.toFloat()
                .reduceResolution(reducer=reductor, maxPixels=MAX_PIXELES_POR_CELDA)
                .reproject(crs="EPSG:4326", crsTransform=transformacion(malla))
                .unmask(-1))

    def _anio_mapbiomas(self):
        """El catálogo lista la propiedad del año como `año` y su ejemplo filtra por `year`: se mira cuál hay."""
        if not hasattr(self, "_prop_anio"):
            nombres = self.ee.ImageCollection(CAPAS["cobertura"]["activo"]).first().propertyNames().getInfo()
            self._prop_anio = "year" if "year" in nombres else "año"
        return self._prop_anio

    def _mapbiomas(self, anio):
        ee = self.ee
        img = (ee.ImageCollection(CAPAS["cobertura"]["activo"]).filter(ee.Filter.eq(self._anio_mapbiomas(), anio))
               .first().select("classification"))
        return img.updateMask(img.neq(NO_OBSERVADO).And(img.gt(0)))

    def _imagen(self, nombre, banda, malla):
        ee = self.ee
        media = ee.Reducer.mean()
        if nombre == "quemas":
            anio = int(banda.rsplit("_", 1)[1])
            col = (ee.ImageCollection(CAPAS["quemas"]["activo"])
                   .filterDate(f"{anio}-01-01", f"{anio + 1}-01-01").select("BurnDate"))
            # un compuesto pierde la proyección de origen: se le devuelve la de MODIS
            # unmask(0): píxeles sin quema deben ser 0, no sin-dato (evita NaN en 3 variables)
            quemado = col.max().gt(0).unmask(0).setDefaultProjection(col.first().projection())
            return self._a_malla(quemado, malla, media)
        if nombre == "perdida_bosque":
            gfc = ee.Image(CAPAS["perdida_bosque"]["activo"])
            tierra = gfc.select("datamask").eq(1)
            if banda == "arboles_2000":
                return self._a_malla(gfc.select("treecover2000").divide(100).updateMask(tierra), malla, media)
            anio = int(banda.rsplit("_", 1)[1])
            # unmask(0): píxeles de tierra sin pérdida ese año → 0, no sin-dato (evita fuga de futuro)
            return self._a_malla(gfc.select("lossyear").eq(anio - 2000).updateMask(tierra).unmask(0), malla, media)
        if nombre == "cobertura":
            variable, anio = banda.rsplit("_", 1)
            anio = int(anio)
            if variable == "conversion3":
                antes = self._mapbiomas(anio - 3).remap(NATURAL, [1] * len(NATURAL), 0)
                ahora = self._mapbiomas(anio).remap(AGROPECUARIO, [1] * len(AGROPECUARIO), 0)
                return self._a_malla(antes.And(ahora), malla, media)
            codigos = CLASES_MAPBIOMAS[variable]
            return self._a_malla(self._mapbiomas(anio).remap(codigos, [1] * len(codigos), 0), malla, media)
        if nombre == "poblacion":
            anio = int(banda.rsplit("_", 1)[1])
            img = (ee.ImageCollection(CAPAS["poblacion"]["activo"]).filter(ee.Filter.eq("country", "BOL"))
                   .filter(ee.Filter.eq("year", anio)).first().select("population"))
            return self._a_malla(img.unmask(0), malla, ee.Reducer.sum())   # suma: personas por celda; unmask(0): sin población = 0
        raise KeyError(nombre)

    def pixeles(self, nombre, bandas, malla):
        ee = self.ee
        imagen = ee.Image.cat([self._imagen(nombre, b, malla).rename(b) for b in bandas])
        resultado = ee.data.computePixels({"expression": imagen, "fileFormat": "NUMPY_NDARRAY",
                                           "grid": grilla(malla)})
        return a_diccionario(resultado, bandas)
