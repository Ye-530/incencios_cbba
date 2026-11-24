"""Índice Meteorológico de Incendios del sistema canadiense (FWI).

Ecuaciones de Van Wagner (1987), "Development and structure of the Canadian
Forest Fire Weather Index System". Los factores de duración del día por latitud
son los del paquete cffdrs (Natural Resources Canada), que permiten usar el
índice fuera de Canadá.

Entradas diarias: temperatura (°C), humedad relativa (%), viento a 10 m (km/h)
y lluvia de 24 h (mm), idealmente del mediodía/primeras horas de la tarde.
"""
from math import exp, log, sqrt

import numpy as np

# Duración efectiva del día para el DMC, enero a diciembre
_DMC_DIA_46N = (6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0)
_DMC_DIA_20N = (7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8)
_DMC_DIA_20S = (10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2)
_DMC_DIA_40S = (11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8)
_DMC_DIA_ECUADOR = 9.0
# Factor de duración del día para el DC
_DC_DIA_NORTE = (-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6)
_DC_DIA_SUR = (6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8)
_DC_DIA_TROPICO = 1.4


def _dia_dmc(mes, lat):
    if lat > 30:
        return _DMC_DIA_46N[mes - 1]
    if lat > 10:
        return _DMC_DIA_20N[mes - 1]
    if lat > -10:
        return _DMC_DIA_ECUADOR
    if lat > -30:
        return _DMC_DIA_20S[mes - 1]
    return _DMC_DIA_40S[mes - 1]


def _dia_dc(mes, lat):
    if lat > 20:
        return _DC_DIA_NORTE[mes - 1]
    if lat > -20:
        return _DC_DIA_TROPICO
    return _DC_DIA_SUR[mes - 1]


def ffmc(t, hr, viento_kmh, lluvia, previo):
    """Código de humedad del combustible fino (hojarasca superficial). 0 a 101."""
    hr = min(max(hr, 0.0), 100.0)
    viento_kmh = max(viento_kmh, 0.0)
    mo = 147.2 * (101.0 - previo) / (59.5 + previo)
    if lluvia > 0.5:
        rf = lluvia - 0.5
        mr = mo + 42.5 * rf * exp(-100.0 / (251.0 - mo)) * (1.0 - exp(-6.93 / rf))
        if mo > 150.0:
            mr += 0.0015 * (mo - 150.0) ** 2 * sqrt(rf)
        mo = min(mr, 250.0)
    ed = 0.942 * hr ** 0.679 + 11.0 * exp((hr - 100.0) / 10.0) \
        + 0.18 * (21.1 - t) * (1.0 - exp(-0.115 * hr))
    if mo > ed:
        ko = 0.424 * (1.0 - (hr / 100.0) ** 1.7) \
            + 0.0694 * sqrt(viento_kmh) * (1.0 - (hr / 100.0) ** 8)
        m = ed + (mo - ed) * 10.0 ** (-ko * 0.581 * exp(0.0365 * t))
    else:
        ew = 0.618 * hr ** 0.753 + 10.0 * exp((hr - 100.0) / 10.0) \
            + 0.18 * (21.1 - t) * (1.0 - exp(-0.115 * hr))
        if mo < ew:
            k1 = 0.424 * (1.0 - ((100.0 - hr) / 100.0) ** 1.7) \
                + 0.0694 * sqrt(viento_kmh) * (1.0 - ((100.0 - hr) / 100.0) ** 8)
            m = ew - (ew - mo) * 10.0 ** (-k1 * 0.581 * exp(0.0365 * t))
        else:
            m = mo
    return min(max(59.5 * (250.0 - m) / (147.2 + m), 0.0), 101.0)


def dmc(t, hr, lluvia, previo, mes, lat):
    """Código de humedad del mantillo (capa orgánica poco profunda)."""
    hr = min(max(hr, 0.0), 100.0)
    t = max(t, -1.1)
    secado = 1.894 * (t + 1.1) * (100.0 - hr) * _dia_dmc(mes, lat) * 1e-4
    if lluvia > 1.5:
        rw = 0.92 * lluvia - 1.27
        wmi = 20.0 + 280.0 / exp(0.023 * previo)
        if previo <= 33.0:
            b = 100.0 / (0.5 + 0.3 * previo)
        elif previo <= 65.0:
            b = 14.0 - 1.3 * log(previo)
        else:
            b = 6.2 * log(previo) - 17.2
        wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
        previo = max(43.43 * (5.6348 - log(wmr - 20.0)), 0.0)
    return max(previo + secado, 0.0)


def dc(t, lluvia, previo, mes, lat):
    """Código de sequía (capa orgánica profunda; memoria de meses)."""
    t = max(t, -2.8)
    evap = max((0.36 * (t + 2.8) + _dia_dc(mes, lat)) / 2.0, 0.0)
    if lluvia > 2.8:
        rw = 0.83 * lluvia - 1.27
        smi = 800.0 * exp(-previo / 400.0)
        previo = max(previo - 400.0 * log(1.0 + 3.937 * rw / smi), 0.0)
    return max(previo + evap, 0.0)


def isi(viento_kmh, ffmc_hoy):
    """Índice de propagación inicial."""
    m = 147.2 * (101.0 - ffmc_hoy) / (59.5 + ffmc_hoy)
    f_humedad = 91.9 * exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * exp(0.05039 * max(viento_kmh, 0.0)) * f_humedad


def bui(dmc_hoy, dc_hoy):
    """Índice de combustible disponible."""
    if dmc_hoy <= 0.0 and dc_hoy <= 0.0:
        return 0.0
    if dmc_hoy <= 0.4 * dc_hoy:
        u = 0.8 * dmc_hoy * dc_hoy / (dmc_hoy + 0.4 * dc_hoy)
    else:
        u = dmc_hoy - (1.0 - 0.8 * dc_hoy / (dmc_hoy + 0.4 * dc_hoy)) \
            * (0.92 + (0.0114 * dmc_hoy) ** 1.7)
    return max(u, 0.0)


def fwi(isi_hoy, bui_hoy):
    """Índice meteorológico de incendios (intensidad potencial del frente)."""
    if bui_hoy <= 80.0:
        fd = 0.626 * bui_hoy ** 0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * exp(-0.023 * bui_hoy))
    b = 0.1 * isi_hoy * fd
    return exp(2.72 * (0.434 * log(b)) ** 0.647) if b > 1.0 else b


def serie_fwi(t, hr, viento_kmh, lluvia, meses, lat, inicio=(85.0, 6.0, 15.0)):
    """Calcula el sistema día por día a lo largo de una serie continua.

    `inicio` son los códigos (FFMC, DMC, DC) del día anterior al primero.
    """
    t = np.asarray(t, dtype=float)
    hr = np.clip(np.asarray(hr, dtype=float), 0.0, 100.0)
    viento_kmh = np.maximum(np.asarray(viento_kmh, dtype=float), 0.0)
    lluvia = np.maximum(np.asarray(lluvia, dtype=float), 0.0)
    n = len(t)
    salida = {k: np.empty(n) for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}
    f, p, d = inicio
    for i in range(n):
        mes = int(meses[i])
        f = ffmc(t[i], hr[i], viento_kmh[i], lluvia[i], f)
        p = dmc(t[i], hr[i], lluvia[i], p, mes, lat)
        d = dc(t[i], lluvia[i], d, mes, lat)
        r = isi(viento_kmh[i], f)
        u = bui(p, d)
        salida["ffmc"][i], salida["dmc"][i], salida["dc"][i] = f, p, d
        salida["isi"][i], salida["bui"][i], salida["fwi"][i] = r, u, fwi(r, u)
    return salida
