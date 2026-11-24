"""Fabrica una carpeta de datos sintética con la misma estructura que la del usuario."""
import json

import numpy as np
import pandas as pd

from incendios.malla import Malla
from tests.conftest import escribir_tif

MALLA = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=40, nfilas=40)
ANIOS = range(2012, 2020)
APAGON = pd.date_range("2016-08-01", "2016-08-10")       # diez días sin satélite
CHIMENEA = (5, 30)                                       # (fila, columna) de una fuente de calor fija
FRP_CHIMENEA = 99.0


def _cuadro(x0, y0, x1, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]


def crear(raiz, semilla=0):
    rng = np.random.default_rng(semilla)
    m = MALLA
    este, sur = m.oeste + m.ncols * m.celda, m.norte - m.nfilas * m.celda
    medio = m.oeste + 20 * m.celda
    geo = raiz / "data" / "raw" / "geo"
    geo.mkdir(parents=True)

    def guardar(nombre, entidades):
        (geo / nombre).write_text(json.dumps({"type": "FeatureCollection", "features": entidades}), encoding="utf-8")

    def entidad(props, x0, y0, x1, y1):
        return {"type": "Feature", "properties": props,
                "geometry": {"type": "Polygon", "coordinates": _cuadro(x0, y0, x1, y1)}}

    # el departamento deja fuera las dos filas del norte
    norte_depto = m.norte - 2 * m.celda
    guardar("cochabamba_limite.geojson", [entidad({"NAME_1": "Cochabamba"}, m.oeste, sur, este, norte_depto)])
    guardar("cochabamba_provincias.geojson", [
        entidad({"NAME_2": "Cercado"}, m.oeste, sur, medio, norte_depto),
        entidad({"NAME_2": "Chapare"}, medio, sur, este, norte_depto)])
    mitad_sur = m.norte - 20 * m.celda
    guardar("cochabamba_municipios.geojson", [
        entidad({"NAME_3": "Cochabamba", "NAME_2": "Cercado"}, m.oeste, sur, medio, norte_depto),
        entidad({"NAME_3": "VillaTunari", "NAME_2": "Chapare"}, medio, mitad_sur, este, norte_depto),
        entidad({"NAME_3": "Totora", "NAME_2": "Chapare"}, medio, sur, este, mitad_sur)])

    # cobertura: oeste pasto, este bosque, una franja de cultivo; relieve solo en la mitad oeste
    (raiz / "data" / "raw" / "landcover").mkdir(parents=True)
    (raiz / "data" / "raw" / "terrain").mkdir(parents=True)
    lc = np.full((40 * 4, 40 * 4), 30, dtype=np.uint8)
    lc[:, 80:] = 10
    lc[60:80, :] = 40
    escribir_tif(raiz / "data" / "raw" / "landcover" / "lc.tif", lc, m.oeste, m.norte, m.celda / 4)
    filas = np.arange(40 * 2, dtype=np.float32)[:, None]
    dem = (2500.0 + 5.0 * filas) * np.ones((40 * 2, 20 * 2), dtype=np.float32)
    escribir_tif(raiz / "data" / "raw" / "terrain" / "dem.tif", dem, m.oeste, m.norte, m.celda / 2)

    # clima: dos provincias, lluvia de noviembre a abril
    fechas = pd.date_range(f"{ANIOS[0]}-01-01", f"{ANIOS[-1]}-12-31")
    (raiz / "data" / "raw" / "weather").mkdir(parents=True)
    partes = []
    for nombre, lat, lon, tmax in [("Cercado", -17.05, -65.95, 22.0), ("Chapare", -17.05, -65.85, 30.0)]:
        humedo = (fechas.month >= 11) | (fechas.month <= 4)
        lluvia = np.where(humedo & (rng.random(len(fechas)) < 0.5), rng.uniform(1, 20, len(fechas)), 0.0)
        partes.append(pd.DataFrame({
            "provincia": nombre, "lat": lat, "lon": lon, "fecha": fechas.strftime("%Y-%m-%d"),
            "anio": fechas.year, "mes": fechas.month, "dia": fechas.day,
            "temp_max": tmax + rng.normal(0, 2, len(fechas)), "temp_min": tmax - 14.0,
            "humedad_rel": np.where(humedo, 75.0, 40.0) + rng.normal(0, 5, len(fechas)),
            "viento_vel": 2.0, "precipitacion": lluvia,
            "humedad_suelo_sup": 0.4, "humedad_suelo_raiz": 0.5}))
    pd.concat(partes).to_csv(raiz / "data" / "raw" / "weather" / "clima_suelo_diario_cochabamba_2012_2025.csv", index=False)

    # focos: temporada de agosto a octubre; un "punto caliente" recurrente en el pasto del sur
    (raiz / "data" / "raw" / "focos").mkdir(parents=True)
    lon_c, lat_c = m.centros()
    LON, LAT = np.meshgrid(lon_c, lat_c)
    fila, col = np.indices(m.forma)
    propension = np.full(m.forma, 0.0005)
    propension[(fila >= 25) & (fila < 35) & (col >= 5) & (col < 15)] = 0.05     # punto caliente
    propension[60 // 4:80 // 4, :] = 0.004                                        # franja de cultivo
    for anio in ANIOS:
        filas_csv = []
        for dia in pd.date_range(f"{anio}-01-01", f"{anio}-12-31"):
            if dia in APAGON:
                continue
            # siempre hay fuego en algún lugar de Bolivia, lejos de la malla
            filas_csv.append((-12.0, -63.0, dia, "n", 5.0, 0))
            if dia.dayofyear % 21 == 0:      # la chimenea: cada tres semanas, todo el año, de noche
                filas_csv.append((lat_c[CHIMENEA[0]], lon_c[CHIMENEA[1]], dia, "n", FRP_CHIMENEA, 0))
            if dia.month in (8, 9, 10):
                arde = rng.random(m.forma) < propension
                for la, lo in zip(LAT[arde], LON[arde]):
                    filas_csv.append((la + rng.uniform(-0.001, 0.001), lo + rng.uniform(-0.001, 0.001),
                                      dia, rng.choice(["n", "h", "l"], p=[0.85, 0.1, 0.05]), 8.0, 0))
        df = pd.DataFrame(filas_csv, columns=["latitude", "longitude", "acq_date", "confidence", "frp", "type"])
        df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")
        df["bright_ti4"], df["scan"], df["track"], df["acq_time"] = 330.0, 0.4, 0.4, 1730
        df["satellite"], df["instrument"], df["version"], df["bright_ti5"] = "N", "VIIRS", 2, 290.0
        df["daynight"] = np.where(df["frp"] == FRP_CHIMENEA, "N", "D")
        df.to_csv(raiz / "data" / "raw" / "focos" / f"viirs-snpp_{anio}_Bolivia.csv", index=False)
    return raiz


def crear_capas(raiz, semilla=1):
    """Capas extra sintéticas en data/capas (lo que dejarían Earth Engine y OpenStreetMap)."""
    from incendios import capas
    rng = np.random.default_rng(semilla)
    m = MALLA
    carpeta = raiz / "data" / "external" / "capas"
    fila, col = np.indices(m.forma)
    caliente = (fila >= 25) & (fila < 35) & (col >= 5) & (col < 15)
    este = col >= 20

    def cubo(anios, f):
        return np.stack([np.asarray(f(a), dtype=np.float32) for a in anios])

    anios = list(range(2001, 2020))
    capas.guardar(carpeta / "quemas.npz", "quemas", m, anios,
                  {"quema": cubo(anios, lambda a: np.where(caliente, rng.uniform(0, 0.6, m.forma), 0.0))},
                  {}, {"activo": "sintetico/MCD64A1"})
    capas.guardar(carpeta / "perdida_bosque.npz", "perdida_bosque", m, anios,
                  {"perdida": cubo(anios, lambda a: np.where(este, rng.uniform(0, 0.02, m.forma), 0.0))},
                  {"arboles_2000": np.where(este, 0.8, 0.1).astype(np.float32)}, {"activo": "sintetico/hansen"})
    anios_mb = list(range(2010, 2020))
    fracciones = {"bosque": este * 0.9, "natural": este * 0.1, "pasto": ~este * 0.7, "agricultura": ~este * 0.2,
                  "mosaico": ~este * 0.1, "no_vegetado": np.zeros(m.forma), "agua": np.zeros(m.forma)}
    anuales_mb = {v: cubo(anios_mb, lambda a, f=f: f) for v, f in fracciones.items()}
    anuales_mb["conversion3"] = cubo(anios_mb, lambda a: np.where(este & (col < 24), rng.uniform(0, 0.1, m.forma), 0.0))
    capas.guardar(carpeta / "cobertura.npz", "cobertura", m, anios_mb, anuales_mb, {}, {"activo": "sintetico/mapbiomas"})
    anios_pob = list(range(2015, 2031))
    capas.guardar(carpeta / "poblacion.npz", "poblacion", m, anios_pob,
                  {"poblacion": cubo(anios_pob, lambda a: 200.0 * np.exp(-col / 6.0) * (1 + 0.01 * (a - 2015)))},
                  {}, {"activo": "sintetico/worldpop"})
    capas.guardar(carpeta / "caminos.npz", "caminos", m, [], {},
                  {"dist_via_km": (np.abs(col - 20) * 0.5).astype(np.float32),
                   "dist_huella_km": (np.abs(fila - 30) * 0.5).astype(np.float32),
                   "vias_km_km2": (1.0 / (1.0 + np.abs(col - 20))).astype(np.float32)},
                  {"fuente": "sintetico/OpenStreetMap"})
    return carpeta
