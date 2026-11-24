"""Calendario de semanas ISO (lunes a domingo) con un índice entero continuo."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

EPOCA = date(2012, 1, 2)  # lunes de la semana 0


def semana_de_fecha(fechas):
    """Índice de semana de una fecha (escalar) o de una serie de fechas (arreglo de enteros)."""
    if isinstance(fechas, (pd.Series, pd.DatetimeIndex, np.ndarray, list, tuple)):
        dias = (pd.DatetimeIndex(pd.to_datetime(fechas)).normalize() - pd.Timestamp(EPOCA)).days
        return np.floor_divide(np.asarray(dias), 7)
    dias = (pd.Timestamp(fechas).normalize() - pd.Timestamp(EPOCA)).days
    return int(dias // 7)


def inicio_semana(idx):
    return EPOCA + timedelta(days=7 * int(idx))


def info_semana(idx):
    inicio = inicio_semana(idx)
    jueves = inicio + timedelta(days=3)
    anio, semana, _ = jueves.isocalendar()
    return {
        "anio": anio,
        "semana_anio": semana,
        "inicio": inicio,
        "fin": inicio + timedelta(days=6),
        "dia_anio": jueves.timetuple().tm_yday,
    }


def semana_de_iso(anio, semana):
    try:
        lunes = date.fromisocalendar(int(anio), int(semana), 1)
    except ValueError as error:
        raise ValueError(f"El año ISO {anio} no tiene semana {semana}") from error
    return semana_de_fecha(lunes)


def semanas_de_anio(anio):
    primera = semana_de_iso(anio, 1)
    siguiente = semana_de_iso(anio + 1, 1)
    return list(range(primera, siguiente))
