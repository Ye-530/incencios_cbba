"""Caminos de OpenStreetMap llevados a la malla: distancia al camino más cercano y densidad.

Casi todo incendio de vegetación lo enciende una persona, y la gente llega por caminos y
sendas. Se bajan con la API Overpass (gratuita, sin cuenta) en teselas de medio grado, se
miden los kilómetros de camino en cada celda y de ahí salen tres variables fijas:

    dist_via_km     distancia (km) a la vía para vehículos más cercana
    dist_huella_km  distancia (km) a la huella o senda más cercana (track, path)
    vias_km_km2     kilómetros de vías y huellas por km² en un entorno de ~5,5 km

OpenStreetMap es el estado actual: un camino abierto en 2023 aparece también al entrenar
con 2016 (ver docs/decisiones.md).
"""
import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from . import capas
from .malla import KM_POR_GRADO_LAT, KM_POR_GRADO_LON_ECUADOR

URL_OVERPASS = "https://overpass-api.de/api/interpreter"
CLASES_VIA = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
              "living_street", "service", "road", "motorway_link", "trunk_link", "primary_link",
              "secondary_link", "tertiary_link")
CLASES_HUELLA = ("track", "path", "bridleway", "footway")
VARIABLES = ("dist_via_km", "dist_huella_km", "vias_km_km2")
LADO_DENSIDAD = 11          # celdas: ~5,5 km
PASO_TESELA = 0.5           # grados
SUBDIVISION = 4             # cada tramo se parte en pedazos de 1/4 de celda o menos


def tipo_de(clase):
    if clase in CLASES_VIA:
        return "via"
    if clase in CLASES_HUELLA:
        return "huella"
    return None


def teselas(malla, paso=PASO_TESELA):
    """Recuadros (sur, oeste, norte, este) que cubren la malla, de `paso` grados como máximo."""
    oeste, norte = malla.oeste, malla.norte
    este, sur = oeste + malla.ncols * malla.celda, norte - malla.nfilas * malla.celda
    salida = []
    s = np.floor(sur / paso) * paso
    while s < norte - 1e-9:
        o = np.floor(oeste / paso) * paso
        while o < este - 1e-9:
            recuadro = (max(s, sur), max(o, oeste), min(s + paso, norte), min(o + paso, este))
            if recuadro[2] > recuadro[0] and recuadro[3] > recuadro[1]:
                salida.append(tuple(round(float(v), 6) for v in recuadro))
            o += paso
        s += paso
    return salida


def consulta(recuadro, timeout=180):
    """Consulta Overpass QL: todas las vías y huellas del recuadro, con su geometría."""
    sur, oeste, norte, este = recuadro
    clases = "|".join(CLASES_VIA + CLASES_HUELLA)
    return (f"[out:json][timeout:{timeout}];\n"
            f'way["highway"~"^({clases})$"]({sur:.4f},{oeste:.4f},{norte:.4f},{este:.4f});\n'
            f"out geom;")


def leer(respuestas):
    """Respuestas de Overpass (una por tesela) → líneas sin repetir: {id, tipo, lon, lat}."""
    vistas, lineas = set(), []
    for respuesta in respuestas:
        for e in respuesta.get("elements", []):
            if e.get("type") != "way" or e.get("id") in vistas:
                continue
            tipo = tipo_de(e.get("tags", {}).get("highway"))
            puntos = [p for p in (e.get("geometry") or []) if p]
            if tipo is None or len(puntos) < 2:
                continue
            vistas.add(e["id"])
            lineas.append({"id": e["id"], "tipo": tipo,
                           "lon": np.array([p["lon"] for p in puntos], dtype=np.float64),
                           "lat": np.array([p["lat"] for p in puntos], dtype=np.float64)})
    return lineas


def longitud_por_celda(malla, lineas, tipo):
    """Kilómetros de líneas de ese tipo dentro de cada celda [nfilas, ncols]."""
    km = np.zeros(malla.nfilas * malla.ncols, dtype=np.float64)
    sel = [l for l in lineas if l["tipo"] == tipo]
    if sel:
        lon0 = np.concatenate([l["lon"][:-1] for l in sel])
        lat0 = np.concatenate([l["lat"][:-1] for l in sel])
        dlon = np.concatenate([np.diff(l["lon"]) for l in sel])
        dlat = np.concatenate([np.diff(l["lat"]) for l in sel])
        largo_km = np.hypot(dlon * KM_POR_GRADO_LON_ECUADOR * np.cos(np.deg2rad(lat0 + dlat / 2)),
                            dlat * KM_POR_GRADO_LAT)
        paso = malla.celda / SUBDIVISION
        n = np.maximum(1, np.ceil(np.hypot(dlon, dlat) / paso - 1e-6)).astype(np.int64)
        tramo = np.repeat(np.arange(len(n)), n)
        k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        t = (k + 0.5) / n[tramo]
        ids = malla.id_celda(lon0[tramo] + t * dlon[tramo], lat0[tramo] + t * dlat[tramo])
        dentro = ids >= 0
        km += np.bincount(ids[dentro], weights=(largo_km / n)[tramo][dentro], minlength=km.size)
    return km.reshape(malla.forma).astype(np.float32)


def variables_caminos(malla, km_via, km_huella):
    lat_media = malla.norte - malla.nfilas * malla.celda / 2.0
    paso_km = (malla.celda * KM_POR_GRADO_LAT,
               malla.celda * KM_POR_GRADO_LON_ECUADOR * np.cos(np.deg2rad(lat_media)))

    def distancia(km):
        fuente = np.nan_to_num(km) > 0
        if not fuente.any():
            return np.full(malla.forma, np.nan, dtype=np.float32)
        return distance_transform_edt(~fuente, sampling=paso_km).astype(np.float32)

    total = np.nan_to_num(km_via).astype(np.float64) + np.nan_to_num(km_huella)
    suma_km = uniform_filter(total, size=LADO_DENSIDAD, mode="constant", cval=0.0)
    suma_area = uniform_filter(malla.area_km2(), size=LADO_DENSIDAD, mode="constant", cval=0.0)
    densidad = suma_km / suma_area
    densidad[suma_km < 1e-9] = 0.0                  # restos de redondeo del filtro
    return {"dist_via_km": distancia(km_via), "dist_huella_km": distancia(km_huella),
            "vias_km_km2": densidad.astype(np.float32)}


def guardar(carpeta, malla, variables, meta):
    return capas.guardar(Path(carpeta) / "caminos.npz", "caminos", malla, anios=[], anuales={},
                         estaticas=variables, meta=meta)


def descargar(malla, carpeta, pedir, verboso=False):
    """Baja las teselas que falten (cada una queda guardada en carpeta/osm/) y arma la capa.

    `pedir(consulta) -> dict` hace la llamada a Overpass; así esto se prueba sin internet.
    """
    carpeta = Path(carpeta)
    crudos = carpeta / "osm"
    crudos.mkdir(parents=True, exist_ok=True)
    respuestas = []
    lista = teselas(malla)
    for k, recuadro in enumerate(lista, start=1):
        archivo = crudos / "caminos_{:.3f}_{:.3f}.json.gz".format(recuadro[0], recuadro[1])
        if archivo.exists():
            with gzip.open(archivo, "rt", encoding="utf-8") as f:
                respuestas.append(json.load(f))
            continue
        if verboso:
            print(f"    tesela {k}/{len(lista)} {recuadro}...", flush=True)
        respuesta = pedir(consulta(recuadro))
        temporal = archivo.with_name(archivo.name + ".parcial")
        with gzip.open(temporal, "wt", encoding="utf-8") as f:
            json.dump(respuesta, f)
        temporal.replace(archivo)
        respuestas.append(respuesta)
    lineas = leer(respuestas)
    variables = variables_caminos(malla, longitud_por_celda(malla, lineas, "via"),
                                  longitud_por_celda(malla, lineas, "huella"))
    meta = {"fuente": "OpenStreetMap (API Overpass)", "bajado": str(date.today()),
            "vias": sum(l["tipo"] == "via" for l in lineas),
            "huellas": sum(l["tipo"] == "huella" for l in lineas)}
    return guardar(carpeta, malla, variables, meta)
