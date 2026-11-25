"""Piezas de la actualización de datos que no tocan la red (las descargas las hace incendios/pasos/descargar.py).

- Qué meses de ERA5-Land hay que pedir al CDS.
- De los NetCDF mensuales a la tabla diaria del proyecto, con un caché por año.
- Qué semanas se pueden predecir en modo operativo con los datos que hay.
"""
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import era5
from .tiempo import info_semana, semana_de_iso

INICIO_CLIMA = date(2012, 1, 1)
MESES_A_REFRESCAR = 3       # ERA5T (los últimos ~3 meses) se corrige; el mes en curso va creciendo


def archivos_del_mes(carpeta, anio, mes):
    """NetCDF de ese mes: uno, o dos mitades (_a, _b) si el CDS no aceptó el mes entero."""
    return sorted(Path(carpeta).glob(f"era5land_{anio}_{mes:02d}*.nc"))


def meses_pendientes(carpeta, hoy, desde=INICIO_CLIMA, refrescar=MESES_A_REFRESCAR):
    """(año, mes, hasta) a pedir: los que faltan y los últimos `refrescar`. `hasta` solo en el mes en curso."""
    ultimo = era5.ultimo_dia_publicado(hoy)
    todos = era5.meses(desde, ultimo)
    recientes = set(todos[-refrescar:]) if refrescar else set()
    return [(a, m, ultimo if (a, m) == (ultimo.year, ultimo.month) else None)
            for a, m in todos if (a, m) in recientes or not archivos_del_mes(carpeta, a, m)]


def _diario_del_anio(entradas, anio, malla, mascara, margen):
    cubos = era5.unir([era5.leer_netcdf(p) for p in entradas])
    t2m = cubos["t2m"]
    utiles = era5.puntos_utiles(t2m["lat"], t2m["lon"], malla, mascara, margen=margen)
    i, j = np.nonzero(utiles)
    claves = set(zip(np.round(t2m["lat"][i], 4), np.round(t2m["lon"][j], 4)))
    dias = era5.diario(cubos)
    dias = dias[dias["fecha"].str.startswith(f"{anio}-")]
    sirve = [(a, o) in claves for a, o in zip(dias["lat"], dias["lon"])]
    dias = dias[np.asarray(sirve, dtype=bool) & dias["temp_max"].notna().to_numpy()]
    return era5.tabla_clima(dias)


def _escribir(tabla, destino):
    """CSV comprimido idéntico byte a byte en cualquier sistema: fin de línea \\n y gzip sin fecha.
    Así, rehacer un archivo con los mismos datos no aparece como cambio en git."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".parcial.csv.gz")
    tabla.to_csv(temporal, index=False, lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    temporal.replace(destino)


def _firma(entradas):
    return {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in entradas}


def _ruta_firma(cache):
    return cache.with_name(cache.name.replace(".csv.gz", ".firma.json"))


def cache_vigente(cache, entradas):
    """¿El diario de un año se armó con estos mismos archivos mensuales? Se compara el contenido, no la
    fecha: copiar la carpeta o hacer `git checkout` cambia las fechas sin cambiar los datos."""
    cache = Path(cache)
    if not cache.exists():
        return False
    ruta = _ruta_firma(cache)
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8")) == _firma(entradas)
    # cachés de antes de la firma: decide la fecha, y desde ahora queda la firma
    vigente = cache.stat().st_mtime >= max(p.stat().st_mtime for p in entradas)
    if vigente:
        ruta.write_text(json.dumps(_firma(entradas), indent=1), encoding="utf-8")
    return vigente


def armar_clima_era5(carpeta_meses, malla, mascara, destino, margen=1, exigir_desde=None, verboso=False):
    """Tabla diaria de ERA5-Land (esquema del proyecto) a partir de los NetCDF mensuales.

    Cada año se guarda en carpeta_meses/diario/ y solo se recalcula si cambió el contenido de alguno
    de sus meses o del enero siguiente (la lluvia y la noche del 31 de diciembre están en ese archivo).
    Si faltan días, o la serie empieza después de `exigir_desde`, no escribe nada: un archivo
    de clima con huecos pasaría al modelo como datos faltantes sin que nadie lo note.
    Devuelve la ruta del archivo final y la última fecha.
    """
    carpeta_meses = Path(carpeta_meses)
    archivos = sorted(carpeta_meses.glob("era5land_*.nc"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos era5land_AAAA_MM.nc en {carpeta_meses}: baje primero "
                                f"el clima con: python -m incendios descargar --que clima_era5")
    cache = carpeta_meses / "diario"
    partes = []
    for anio in sorted({int(p.name.split("_")[1]) for p in archivos}):
        entradas = [p for m in range(1, 13) for p in archivos_del_mes(carpeta_meses, anio, m)]
        if not entradas:
            continue                     # solo está el enero de un año siguiente
        entradas += archivos_del_mes(carpeta_meses, anio + 1, 1)
        ruta = cache / f"era5land_diario_{anio}.csv.gz"
        if not cache_vigente(ruta, entradas):
            if verboso:
                print(f"    días de {anio}...", flush=True)
            _escribir(_diario_del_anio(entradas, anio, malla, mascara, margen), ruta)
            _ruta_firma(ruta).write_text(json.dumps(_firma(entradas), indent=1), encoding="utf-8")
        partes.append(pd.read_csv(ruta))
    tabla = pd.concat(partes, ignore_index=True)
    completo = tabla.groupby("provincia")["fecha"].max().min()       # el último día que tienen todos
    tabla = tabla[tabla["fecha"] <= completo]
    fechas = pd.DatetimeIndex(pd.to_datetime(tabla["fecha"].unique())).sort_values()
    if exigir_desde is not None and fechas[0].date() > exigir_desde:
        raise ValueError(f"El clima ERA5-Land empieza el {fechas[0].date()} y para entrenar debe empezar el "
                         f"{exigir_desde}: faltan meses por bajar (vuelva a ejecutar --que clima_era5).")
    faltan = pd.date_range(fechas[0], fechas[-1]).difference(fechas)
    if len(faltan):
        raise ValueError(f"Faltan {len(faltan)} días de clima ERA5-Land, el primero {faltan[0].date()}: "
                         f"vuelva a ejecutar --que clima_era5 para bajar los meses que fallaron.")
    _escribir(tabla, destino)
    return Path(destino), completo


# ---------------------------------------------------------------------- semanas
def lunes_de(dia):
    return dia - timedelta(days=dia.weekday())


def semanas_operativas(ctx, desde_anio):
    """Semanas desde el año indicado que se pueden predecir en modo operativo con los datos que hay."""
    primera = max(4, semana_de_iso(desde_anio, 1))
    return [s for s in range(primera, ctx.n_semanas + 1) if ctx.clima.disponible(s)]


def semana_completa(ctx, semana):
    """¿Ya pasó el domingo de esa semana y hay focos hasta ese día?"""
    return str(info_semana(semana)["fin"]) <= ctx.notas["ultima_fecha_focos"] and semana < ctx.n_semanas


# ---------------------------------------------------------------------- antes y después
METRICAS_COMPARACION = ("roc_auc", "pr_auc", "captura_1", "captura_5", "captura_10", "area_50", "area_80")
MENOR_ES_MEJOR = {"area_50", "area_80"}


def comparar_evaluaciones(antes, despues):
    """Tabla larga con las métricas de dos `evaluacion_resumen.json` (de `evaluar`) de los modelos comunes."""
    filas = []
    for periodo, clave in (("prueba", "modelos_prueba"), ("validacion", "modelos_validacion")):
        a, d = antes.get(clave, {}), despues.get(clave, {})
        for modelo in [m for m in d if m in a]:
            for metrica in METRICAS_COMPARACION:
                if metrica not in a[modelo] or metrica not in d[modelo]:
                    continue
                va, vd = float(a[modelo][metrica]), float(d[modelo][metrica])
                cambio = vd - va
                if abs(cambio) < 1e-6:
                    mejor = "igual"
                else:
                    mejor = "después" if (cambio < 0) == (metrica in MENOR_ES_MEJOR) else "antes"
                filas.append({"periodo": periodo, "modelo": modelo, "metrica": metrica, "antes": va,
                              "despues": vd, "cambio": cambio, "mejor": mejor})
    return pd.DataFrame(filas, columns=["periodo", "modelo", "metrica", "antes", "despues", "cambio", "mejor"])
