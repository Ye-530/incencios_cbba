"""Focos de calor VIIRS (NASA FIRMS): carga, filtro, apagones del satélite y positivos semanales."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .rasters import archivos_de
from .tiempo import semana_de_fecha

COLUMNAS = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type", "daynight"]
MESES_TEMPORADA = (7, 8, 9, 10, 11)


def cargar_focos(carpetas, malla, patron=None):
    """Lee los CSV anuales de Bolivia: por defecto la cadena de satélites de config.FUENTES_FOCOS
    (S-NPP hasta su cierre y NOAA-20 después); con `patron`, solo esos archivos.

    `carpetas` es una carpeta o varias (los originales y los bajados); un archivo con el mismo
    nombre en dos carpetas cuenta una vez, el de la última. Devuelve las detecciones dentro del
    recuadro de la malla (con `fecha`, `celda` y `fuente`) y el conteo diario nacional, que sirve
    para reconocer los días sin satélite.
    """
    fuentes = config.FUENTES_FOCOS if patron is None else (
        {"nombre": patron.split("_")[0], "patron": patron},)
    partes, conteos = [], []
    for k, fuente in enumerate(fuentes):
        archivos = archivos_de(carpetas, fuente["patron"])
        if not archivos and k == 0:
            donde = [str(c) for c in ([carpetas] if isinstance(carpetas, (str, Path)) else carpetas)]
            raise FileNotFoundError(f"No hay archivos {fuente['patron']} en {' ni en '.join(donde)}")
        for ruta in archivos:
            df = pd.read_csv(ruta, usecols=COLUMNAS)
            fecha = df["acq_date"].astype(str)
            dentro_de_fechas = np.ones(len(df), dtype=bool)
            if fuente.get("desde"):
                dentro_de_fechas &= (fecha >= fuente["desde"]).to_numpy()
            if fuente.get("hasta"):
                dentro_de_fechas &= (fecha <= fuente["hasta"]).to_numpy()
            df = df[dentro_de_fechas]
            conteos.append(df["acq_date"].value_counts())
            celda = malla.id_celda(df["longitude"].to_numpy(), df["latitude"].to_numpy())
            df = df[celda >= 0].copy()
            df["celda"] = celda[celda >= 0]
            df["fuente"] = fuente["nombre"]
            partes.append(df)
    focos = pd.concat(partes, ignore_index=True)
    focos["fecha"] = pd.to_datetime(focos["acq_date"])
    diario = pd.concat(conteos).groupby(level=0).sum()
    diario.index = pd.to_datetime(diario.index)
    diario = diario.sort_index()
    diario = diario.reindex(pd.date_range(diario.index[0], diario.index[-1]), fill_value=0)
    return focos, diario


def filtrar_vegetacion(df):
    """Fuego de vegetación (type 0) con confianza nominal o alta."""
    return df[(df["type"] == 0) & df["confidence"].isin(["n", "h"])]


def fuentes_persistentes(df, min_semanas=None, min_nocturnas=None):
    """Celdas con una fuente de calor fija y no con fuego de vegetación.

    Un horno industrial o una antorcha de gas se detectan en muchas semanas distintas,
    todo el año y casi siempre en la pasada nocturna; FIRMS no siempre las marca como
    fuente estática. Devuelve una fila por celda: semanas con detección, detecciones y
    fracción nocturna, de más a menos semanas.
    """
    min_semanas = config.FUENTE_MIN_SEMANAS if min_semanas is None else min_semanas
    min_nocturnas = config.FUENTE_MIN_NOCTURNAS if min_nocturnas is None else min_nocturnas
    columnas = ["celda", "semanas", "detecciones", "nocturnas"]
    if "daynight" not in df or len(df) == 0:
        return pd.DataFrame(columns=columnas)
    tabla = pd.DataFrame({
        "celda": df["celda"].to_numpy(),
        "semana": np.asarray(semana_de_fecha(df["fecha"])),
        "noche": df["daynight"].astype(str).str.upper().eq("N").to_numpy(),
    })
    por_celda = (tabla.groupby("celda")
                 .agg(semanas=("semana", "nunique"), detecciones=("noche", "size"), nocturnas=("noche", "mean"))
                 .reset_index())
    fijas = por_celda[(por_celda["semanas"] >= min_semanas) & (por_celda["nocturnas"] >= min_nocturnas)]
    return fijas.sort_values(["semanas", "celda"], ascending=[False, True]).reset_index(drop=True)[columnas]


def quitar_fuentes(df, fuentes):
    """Detecciones que no caen en las celdas de fuentes de calor fijas."""
    return df[~df["celda"].isin(fuentes["celda"])]


def dias_sin_satelite(conteo_diario):
    """Días en que el satélite no observó, deducidos del conteo nacional.

    Regla: dos o más días seguidos sin ninguna detección en todo el país, o un
    día aislado sin detecciones entre julio y noviembre (en plena temporada
    siempre hay fuego en algún lugar de Bolivia).
    """
    cero = (conteo_diario == 0).to_numpy()
    racha = np.zeros(len(cero), dtype=int)
    grupo = np.cumsum(np.r_[True, cero[1:] != cero[:-1]])
    largos = pd.Series(cero).groupby(grupo).transform("size").to_numpy()
    racha[cero] = largos[cero]
    en_temporada = conteo_diario.index.month.isin(MESES_TEMPORADA)
    return conteo_diario.index[cero & ((racha >= 2) | en_temporada)]


def dias_observados(conteo_diario, n_semanas):
    """Días con observación del satélite en cada semana (0 a 7)."""
    observado = pd.Series(True, index=conteo_diario.index)
    observado[dias_sin_satelite(conteo_diario)] = False
    semanas = semana_de_fecha(observado.index)
    dias = np.zeros(n_semanas, dtype=np.int8)
    validas = (semanas >= 0) & (semanas < n_semanas)
    np.add.at(dias, semanas[validas], observado.to_numpy()[validas].astype(np.int8))
    return dias


def positivos_semanales(df, malla):
    """Una fila por (semana, celda) con fuego: número de detecciones y FRP máxima."""
    if "celda" in df:
        celda = df["celda"].to_numpy()
    else:
        celda = malla.id_celda(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    tabla = pd.DataFrame({
        "semana": semana_de_fecha(df["fecha"]),
        "celda": celda,
        "frp": df["frp"].to_numpy(),
    })
    tabla = tabla[tabla["celda"] >= 0]
    pos = (tabla.groupby(["semana", "celda"], sort=True)
           .agg(n_det=("frp", "size"), frp_max=("frp", "max"))
           .reset_index())
    return pos.astype({"semana": np.int32, "celda": np.int64, "n_det": np.int32, "frp_max": np.float32})
