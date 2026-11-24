"""Clima ERA5-Land del Climate Data Store de Copernicus: solicitudes, lectura de NetCDF y tabla diaria.

ERA5-Land tiene celdas de 0,1° (unos 11 km). Para no bajar las 24 horas de cada día se piden
cinco horas UTC (00, 10, 13, 16 y 19), que en Bolivia (UTC-4) son las 20, 06, 09, 12 y 15 h:
la mínima del amanecer, la máxima de la tarde y el mediodía del índice FWI. La lluvia viene
acumulada desde las 00 UTC, así que el valor de las 00 UTC del día D+1 es la lluvia del día D.

El resultado es una tabla con el mismo esquema del archivo de clima original (una fila por
punto y día), más la columna `fuente`, para que el resto del proyecto la use sin cambios.
Las funciones de aquí no tocan la red: las solicitudes las hace el script de descarga.
"""
import calendar
import io
import re
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .clima import COLUMNAS_METEO, FUENTE_ERA5, VIENTO_2M_A_10M, _presion_saturacion

DATASET = "reanalysis-era5-land"
VARIABLES = ("2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind",
             "10m_v_component_of_wind", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
             "total_precipitation")
HORAS_UTC = ("00:00", "10:00", "13:00", "16:00", "19:00")
HUSO_HORAS = -4             # Bolivia, sin horario de verano
LATENCIA_DIAS = 6           # ERA5-Land se publica con unos 5 días de retraso
PASO = 0.1
MIN_MUESTRAS = 3            # muestras de temperatura que necesita un día para entrar en la tabla
_NOMBRES_TIEMPO = ("valid_time", "time")
_NOMBRES_LAT = ("latitude", "lat")
_NOMBRES_LON = ("longitude", "lon")
_UNIDADES_TIEMPO = {"seconds": "s", "second": "s", "minutes": "min", "minute": "min",
                    "hours": "h", "hour": "h", "days": "D", "day": "D"}


# ---------------------------------------------------------------- solicitudes
def area(malla, margen=0.2):
    """Recuadro [norte, oeste, sur, este] que cubre la malla más un margen, en múltiplos de 0,1°."""
    sur = malla.norte - malla.nfilas * malla.celda
    este = malla.oeste + malla.ncols * malla.celda
    redondo = lambda v, f: round(f(v / PASO) * PASO, 1)          # noqa: E731
    return [redondo(malla.norte + margen, np.ceil), redondo(malla.oeste - margen, np.floor),
            redondo(sur - margen, np.floor), redondo(este + margen, np.ceil)]


def solicitud(anio, mes, recuadro, hasta=None):
    """Pedido de un mes al CDS. `hasta` limita los días (para el mes en curso)."""
    ultimo = calendar.monthrange(anio, mes)[1]
    if hasta is not None and (hasta.year, hasta.month) == (anio, mes):
        ultimo = min(ultimo, hasta.day)
    return {
        "variable": list(VARIABLES),
        "year": str(anio),
        "month": f"{mes:02d}",
        "day": [f"{d:02d}" for d in range(1, ultimo + 1)],
        "time": list(HORAS_UTC),
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": list(recuadro),
    }


def meses(desde, hasta):
    """Meses (año, mes) entre dos fechas, ambos incluidos."""
    salida, a, m = [], desde.year, desde.month
    while (a, m) <= (hasta.year, hasta.month):
        salida.append((a, m))
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)
    return salida


def ultimo_dia_publicado(hoy):
    return hoy - timedelta(days=LATENCIA_DIAS)


def archivo_mes(carpeta, anio, mes):
    return Path(carpeta) / f"era5land_{anio}_{mes:02d}.nc"


# ---------------------------------------------------------------- lectura de NetCDF
def _contenidos(ruta):
    """(nombre, bytes) de cada NetCDF del archivo; el CDS a veces entrega un ZIP con varios."""
    datos = Path(ruta).read_bytes()
    if datos[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(datos)) as z:
            return [(n, z.read(n)) for n in z.namelist() if n.lower().endswith(".nc")]
    return [(Path(ruta).name, datos)]


def _texto(valor):
    return valor.decode("utf-8", "replace") if isinstance(valor, bytes) else str(valor)


def _variables_crudas(nombre, datos):
    """{variable: (dimensiones, atributos, arreglo crudo)} sin aplicar escala ni valores faltantes."""
    if datos[:4] == b"\x89HDF":                     # NetCDF4 (formato del CDS actual)
        try:
            import netCDF4
        except ImportError as error:
            raise ImportError(f"{nombre} es NetCDF4: instale el paquete netCDF4 (pip install netCDF4)") from error
        with netCDF4.Dataset("en_memoria", memory=datos) as ds:
            ds.set_auto_maskandscale(False)
            return {n: (tuple(v.dimensions), {k: v.getncattr(k) for k in v.ncattrs()}, np.asarray(v[:]))
                    for n, v in ds.variables.items()}
    from scipy.io import netcdf_file                 # NetCDF3 (formato antiguo)
    with netcdf_file(io.BytesIO(datos), mmap=False, maskandscale=False) as ds:
        return {n: (tuple(v.dimensions), dict(v._attributes), np.array(v.data))
                for n, v in ds.variables.items()}


def _decodificar(crudo, atributos):
    datos = crudo.astype(np.float64)
    for clave in ("_FillValue", "missing_value"):
        if clave in atributos:
            relleno = np.asarray(atributos[clave]).astype(np.float64)
            if np.isfinite(relleno).all():
                datos[np.isin(crudo, np.asarray(atributos[clave]))] = np.nan
    if "scale_factor" in atributos:
        datos = datos * float(np.asarray(atributos["scale_factor"]))
    if "add_offset" in atributos:
        datos = datos + float(np.asarray(atributos["add_offset"]))
    return datos


def _tiempos(valores, unidades):
    m = re.match(r"\s*(\w+)\s+since\s+(.+)", _texto(unidades))
    if not m or m.group(1).lower() not in _UNIDADES_TIEMPO:
        raise ValueError(f"Unidades de tiempo no reconocidas: {unidades!r}")
    origen = pd.Timestamp(m.group(2).strip().replace("T", " ").split(".")[0])
    delta = pd.to_timedelta(np.asarray(valores, dtype=np.float64), unit=_UNIDADES_TIEMPO[m.group(1).lower()])
    return (origen + delta).round("s").to_numpy(dtype="datetime64[ns]")


def _primero(variables, candidatos, que):
    for n in candidatos:
        if n in variables:
            return n
    raise ValueError(f"El NetCDF no trae {que} (se buscó {', '.join(candidatos)})")


def leer_netcdf(ruta):
    """Cubos {nombre_corto: {tiempo, lat, lon, datos[t, lat, lon] float32}} de un archivo del CDS."""
    cubos = {}
    for nombre, datos in _contenidos(ruta):
        variables = _variables_crudas(nombre, datos)
        nt = _primero(variables, _NOMBRES_TIEMPO, "tiempo")
        nla = _primero(variables, _NOMBRES_LAT, "latitud")
        nlo = _primero(variables, _NOMBRES_LON, "longitud")
        tiempo = _tiempos(variables[nt][2], variables[nt][1].get("units", ""))
        lat = variables[nla][2].astype(np.float64)
        lon = variables[nlo][2].astype(np.float64)
        for n, (dims, atributos, crudo) in variables.items():
            if n in (nt, nla, nlo) or not {nt, nla, nlo} <= set(dims):
                continue
            arr = _decodificar(crudo, atributos)
            for eje in reversed(range(arr.ndim)):          # dimensiones extra (number, expver)
                if dims[eje] not in (nt, nla, nlo):
                    with np.errstate(invalid="ignore"):
                        arr = np.nanmean(arr, axis=eje) if arr.shape[eje] > 1 else np.take(arr, 0, axis=eje)
            restantes = [d for d in dims if d in (nt, nla, nlo)]
            arr = np.transpose(arr, [restantes.index(d) for d in (nt, nla, nlo)])
            cubos[n] = {"tiempo": tiempo, "lat": lat, "lon": lon, "datos": arr.astype(np.float32)}
    return cubos


def unir(lista):
    """Junta cubos de varios archivos: concatena en el tiempo, ordena y deja el último repetido."""
    unidos = {}
    for cubos in lista:
        for n, c in cubos.items():
            unidos.setdefault(n, []).append(c)
    salida = {}
    for n, partes in unidos.items():
        base = partes[0]
        for p in partes[1:]:
            if p["datos"].shape[1:] != base["datos"].shape[1:] or not (
                    np.allclose(p["lat"], base["lat"]) and np.allclose(p["lon"], base["lon"])):
                raise ValueError(f"Los archivos de {n} no tienen la misma malla: bájelos con el mismo recuadro")
        tiempo = np.concatenate([p["tiempo"] for p in partes])
        datos = np.concatenate([p["datos"] for p in partes])
        # el último archivo leído manda (ERA5T se corrige): orden estable invertido
        inverso = np.arange(len(tiempo))[::-1]
        _, primeros = np.unique(tiempo[inverso], return_index=True)
        quedan = np.sort(inverso[primeros])
        orden = quedan[np.argsort(tiempo[quedan], kind="stable")]
        salida[n] = {"tiempo": tiempo[orden], "lat": base["lat"], "lon": base["lon"], "datos": datos[orden]}
    return salida


# ---------------------------------------------------------------- días locales
def _por_dia(fechas, datos, funcion):
    """Aplica `funcion` (nanmax, nanmin, nanmean) a las muestras de cada fecha: ([fechas], [n, lat, lon])."""
    unicas, inverso = np.unique(fechas, return_inverse=True)
    salida = np.full((len(unicas),) + datos.shape[1:], np.nan, dtype=np.float32)
    cuenta = np.zeros(len(unicas), dtype=np.int32)
    with np.errstate(invalid="ignore"):
        for i in range(len(unicas)):
            sel = inverso == i
            cuenta[i] = int(sel.sum())
            salida[i] = funcion(datos[sel], axis=0)
    return unicas, salida, cuenta


def diario(cubos, huso=HUSO_HORAS):
    """Tabla larga [fecha, lat, lon, variables del proyecto] por fecha local.

    Temperatura máxima y mínima: extremos de las muestras del día. Humedad relativa: media de
    las muestras (desde temperatura y punto de rocío). Viento: media de la rapidez a 10 m,
    guardada como su equivalente a 2 m, que es la altura del viento de POWER que espera el
    proyecto (el cálculo del FWI la vuelve a llevar a 10 m). Lluvia: el valor de las 00 UTC
    del día siguiente, en mm. Los días sin lluvia publicada o con menos de MIN_MUESTRAS
    muestras de temperatura no salen.
    """
    t2m = cubos["t2m"]
    local = (pd.DatetimeIndex(t2m["tiempo"]) + pd.Timedelta(hours=huso)).normalize().to_numpy()
    temp = t2m["datos"] - 273.15
    fechas, tmax, cuenta = _por_dia(local, temp, np.nanmax)
    _, tmin, _ = _por_dia(local, temp, np.nanmin)

    vacio = np.full(temp.shape[1:], np.nan, dtype=np.float32)

    def media_de(nombre, transformar=None):
        """{fecha: media del día}; las fechas que falten en esta variable quedan sin dato."""
        c = cubos[nombre]
        x = c["datos"] if transformar is None else transformar(c)
        loc = (pd.DatetimeIndex(c["tiempo"]) + pd.Timedelta(hours=huso)).normalize().to_numpy()
        f, m, _ = _por_dia(loc, x, np.nanmean)
        return dict(zip(f, m))

    rocio = cubos["d2m"]["datos"] - 273.15
    hr = np.clip(100.0 * _presion_saturacion(rocio) / _presion_saturacion(temp), 0.0, 100.0)
    _, hr_media, _ = _por_dia(local, hr.astype(np.float32), np.nanmean)
    viento = media_de("u10", lambda c: np.hypot(c["datos"], cubos["v10"]["datos"]))
    suelo1 = media_de("swvl1")
    suelo2 = media_de("swvl2")

    tp = cubos["tp"]
    t_tp = pd.DatetimeIndex(tp["tiempo"])
    a_las_cero = t_tp.hour == 0
    dia_lluvia = (t_tp[a_las_cero].normalize() - pd.Timedelta(days=1)).to_numpy()
    lluvia = pd.Series(range(int(a_las_cero.sum())), index=dia_lluvia)
    lluvia_mm = np.clip(tp["datos"][a_las_cero] * 1000.0, 0.0, None)

    lat, lon = t2m["lat"], t2m["lon"]
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    filas = []
    for i, f in enumerate(fechas):
        if cuenta[i] < MIN_MUESTRAS or f not in lluvia.index:
            continue
        j = lluvia.loc[f]
        j = int(j.iloc[-1]) if isinstance(j, pd.Series) else int(j)
        bloque = {
            "temp_max": tmax[i], "temp_min": tmin[i], "humedad_rel": hr_media[i],
            "viento_vel": viento.get(f, vacio) / VIENTO_2M_A_10M,
            "precipitacion": lluvia_mm[j],
            "humedad_suelo_sup": suelo1.get(f, vacio), "humedad_suelo_raiz": suelo2.get(f, vacio),
        }
        parte = pd.DataFrame({"fecha": pd.Timestamp(f).strftime("%Y-%m-%d"),
                              "lat": np.round(LAT.ravel(), 4), "lon": np.round(LON.ravel(), 4)})
        for columna, valores in bloque.items():
            parte[columna] = np.asarray(valores, dtype=np.float64).ravel()
        filas.append(parte)
    if not filas:
        return pd.DataFrame(columns=["fecha", "lat", "lon", *COLUMNAS_METEO])
    return pd.concat(filas, ignore_index=True)[["fecha", "lat", "lon", *COLUMNAS_METEO]]


# ---------------------------------------------------------------- puntos y tabla final
def puntos_utiles(lat, lon, malla, mascara, margen=1):
    """Puntos ERA5 [lat, lon] más cercanos a alguna celda del departamento, más `margen` puntos alrededor."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    lon_c, lat_c = malla.centros()
    filas, cols = np.nonzero(mascara)
    i = np.abs(lat_c[np.unique(filas)][:, None] - lat[None, :]).argmin(axis=1)
    j = np.abs(lon_c[np.unique(cols)][:, None] - lon[None, :]).argmin(axis=1)
    # celda por celda, para no marcar esquinas vacías del recuadro
    i_de_fila = dict(zip(np.unique(filas), i))
    j_de_col = dict(zip(np.unique(cols), j))
    utiles = np.zeros((len(lat), len(lon)), dtype=bool)
    utiles[[i_de_fila[f] for f in filas], [j_de_col[c] for c in cols]] = True
    if margen:
        from scipy.ndimage import binary_dilation
        utiles = binary_dilation(utiles, structure=np.ones((2 * margen + 1, 2 * margen + 1), bool))
    return utiles


def tabla_clima(dias):
    """Tabla diaria con el esquema del archivo de clima del proyecto y `fuente` = ERA5-Land."""
    t = dias.copy()
    fechas = pd.to_datetime(t["fecha"])
    t.insert(0, "provincia", [f"era5_{a:.2f}_{o:.2f}" for a, o in zip(t["lat"], t["lon"])])
    t["fecha"] = fechas.dt.strftime("%Y-%m-%d")
    t.insert(4, "anio", fechas.dt.year)
    t.insert(5, "mes", fechas.dt.month)
    t.insert(6, "dia", fechas.dt.day)
    t[COLUMNAS_METEO] = t[COLUMNAS_METEO].astype(float).round(3)
    t["fuente"] = FUENTE_ERA5
    return t[["provincia", "lat", "lon", "fecha", "anio", "mes", "dia", *COLUMNAS_METEO, "fuente"]]
