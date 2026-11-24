"""Clima diario (NASA POWER o ERA5-Land): limpieza, series, variables de peligro y su paso a la malla."""
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config
from .fwi import serie_fwi
from .malla import KM_POR_GRADO_LAT, KM_POR_GRADO_LON_ECUADOR
from .tiempo import info_semana, inicio_semana

# POWER entrega la meteorología en la malla de MERRA-2: 0,5° de latitud × 0,625° de longitud.
MERRA_DLAT, MERRA_DLON = 0.5, 0.625
FUENTE_POWER, FUENTE_ERA5 = "POWER", "ERA5-Land"

COLUMNAS_METEO = ["temp_max", "temp_min", "humedad_rel", "viento_vel", "precipitacion",
                  "humedad_suelo_sup", "humedad_suelo_raiz"]
_CONTINUAS = ["temp_max", "temp_min", "humedad_rel", "viento_vel"]
_SUELO = ["humedad_suelo_sup", "humedad_suelo_raiz"]

VARS_CLIMA = ["prec_7", "prec_30", "prec_90", "dias_sin_lluvia", "tmax_7", "hrmin_7", "viento_7",
              "vpd_7", "ffmc", "dmc", "dc", "isi", "bui", "fwi", "fwi_7", "fwi_max7",
              "suelo_sup_7", "suelo_raiz_7"]
_BASE_NORMAL = {"n_prec_30": "prec_30", "n_prec_90": "prec_90", "n_tmax_7": "tmax_7",
                "n_hrmin_7": "hrmin_7", "n_vpd_7": "vpd_7", "n_viento_7": "viento_7",
                "n_fwi_7": "fwi_7", "n_dc": "dc"}
VARS_NORMALES = list(_BASE_NORMAL) + ["n_prec_anual", "n_tmax_anual"]
_BASE_ANOMALIA = {"an_prec_90": "prec_90", "an_tmax_7": "tmax_7", "an_hrmin_7": "hrmin_7",
                  "an_fwi_7": "fwi_7", "an_dc": "dc"}
VARS_ANOMALIA = list(_BASE_ANOMALIA)
VARS_ANTECEDENTE = VARS_CLIMA + VARS_ANOMALIA

DIAS_SIN_LLUVIA_MAX = 180
LLUVIA_QUE_CUENTA = 1.0      # mm; menos que esto es un día seco
VIENTO_2M_A_10M = 1.0 / 0.748  # perfil logarítmico FAO-56


def fuente_de(df):
    """Fuente del archivo de clima: la columna `fuente` si existe; si no, POWER (el archivo original)."""
    if "fuente" not in df.columns:
        return FUENTE_POWER
    valores = [str(v) for v in pd.unique(df["fuente"].dropna())]
    if len(valores) != 1:
        raise ValueError(f"El archivo de clima mezcla fuentes: {valores}")
    return valores[0]


def cargar_clima(ruta):
    df = pd.read_csv(ruta)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def limpiar_clima(df):
    """Completa días faltantes y, solo en POWER, corrige sus problemas conocidos. Devuelve también un informe.

    Las correcciones de POWER (lluvia imposible del 4 y 5 de junio de 2020, tope de lluvia diaria y
    humedad de suelo contaminada después) no se aplican a otras fuentes: en ERA5-Land borrarían
    lluvias reales.
    """
    power = fuente_de(df) == FUENTE_POWER
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    fechas_artefacto = pd.to_datetime(list(config.FECHAS_LLUVIA_ARTEFACTO))
    suelo_ini, suelo_fin = (pd.Timestamp(f) for f in config.VENTANA_SUELO_CONTAMINADO)
    informe = {"lluvia_artefacto": 0, "lluvia_excesiva": 0, "suelo_enmascarado": 0,
               "faltantes": 0, "dias_agregados": 0}
    partes = []
    for provincia, g in df.groupby("provincia", sort=False):
        g = g.set_index("fecha").sort_index()
        completo = pd.date_range(g.index[0], g.index[-1])
        informe["dias_agregados"] += len(completo) - len(g)
        g = g.reindex(completo)
        met = g[COLUMNAS_METEO].where(g[COLUMNAS_METEO] > -900.0)      # -999 = sin dato
        informe["faltantes"] += int(met[_CONTINUAS + ["precipitacion"]].isna().sum().sum())

        artefacto = met.index.isin(fechas_artefacto) & power
        excesiva = (met["precipitacion"] > config.LLUVIA_MAXIMA_CREIBLE) & ~artefacto & power
        informe["lluvia_artefacto"] += int(artefacto.sum())
        informe["lluvia_excesiva"] += int(excesiva.sum())
        met.loc[artefacto | excesiva, "precipitacion"] = 0.0
        met["precipitacion"] = met["precipitacion"].fillna(0.0).clip(lower=0.0)
        met[_CONTINUAS] = met[_CONTINUAS].interpolate(limit_direction="both")

        contaminado = (met.index >= suelo_ini) & (met.index <= suelo_fin) & power
        informe["suelo_enmascarado"] += int(contaminado.sum())
        met.loc[contaminado, _SUELO] = np.nan

        met.insert(0, "provincia", provincia)
        met.insert(1, "lat", g["lat"].ffill().bfill())
        met.insert(2, "lon", g["lon"].ffill().bfill())
        partes.append(met.rename_axis("fecha").reset_index())
    return pd.concat(partes, ignore_index=True), informe


def _centro_merra(lat, lon):
    return (float(np.floor(lat / MERRA_DLAT + 0.5) * MERRA_DLAT),
            float(np.floor(lon / MERRA_DLON + 0.5) * MERRA_DLON))


def series_unicas(df, fuente=FUENTE_POWER):
    """Agrupa las provincias que comparten exactamente la misma serie (misma celda del reanálisis).

    En POWER, cada serie se ubica en el centro de su celda MERRA-2. En las demás fuentes cada
    punto ya es el centro de su celda y conserva su ubicación aunque coincida con otro.
    """
    power = fuente == FUENTE_POWER
    grupos = {}
    for provincia, g in df.groupby("provincia", sort=False):
        g = g.sort_values("fecha").reset_index(drop=True)
        firma = (g["fecha"].iloc[0], len(g),
                 int(pd.util.hash_pandas_object(g[["temp_max", "precipitacion", "humedad_rel"]], index=False).sum()))
        if not power:
            firma += (round(float(g["lat"].iloc[0]), 4), round(float(g["lon"].iloc[0]), 4))
        grupos.setdefault(firma, []).append((provincia, g))
    series, filas = [], []
    for serie, miembros in enumerate(grupos.values()):
        nombre, g = miembros[0]
        if power:
            lat_c, lon_c = _centro_merra(g["lat"].iloc[0], g["lon"].iloc[0])
        else:
            lat_c, lon_c = round(float(g["lat"].iloc[0]), 4), round(float(g["lon"].iloc[0]), 4)
        filas.append({"serie": serie, "lat_c": lat_c, "lon_c": lon_c,
                      "provincias": ", ".join(p for p, _ in miembros)})
        parte = g[["fecha"] + COLUMNAS_METEO].copy()
        parte.insert(0, "serie", serie)
        series.append(parte)
    return pd.concat(series, ignore_index=True), pd.DataFrame(filas)


def _presion_saturacion(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))  # kPa (Tetens)


def variables_diarias(d, lat):
    """Variables de peligro de cada día (cada valor resume ese día y los anteriores).

    `d`: tabla diaria de una serie, indexada por fecha, ya limpia.
    """
    tmax, tmin = d["temp_max"], d["temp_min"]
    lluvia = d["precipitacion"]
    es_max = _presion_saturacion(tmax)
    ea = d["humedad_rel"] / 100.0 * (es_max + _presion_saturacion(tmin)) / 2.0
    hr_min = (100.0 * ea / es_max).clip(0.0, 100.0)       # humedad a la hora de más calor
    vpd = (es_max - ea).clip(lower=0.0)
    viento = d["viento_vel"] * 3.6 * VIENTO_2M_A_10M         # km/h a 10 m
    f = serie_fwi(tmax.to_numpy(), hr_min.to_numpy(), viento.to_numpy(), lluvia.to_numpy(),
                  d.index.month.to_numpy(), lat)

    seco = lluvia < LLUVIA_QUE_CUENTA
    racha = seco.groupby((~seco).cumsum()).cumsum().clip(upper=DIAS_SIN_LLUVIA_MAX)

    def media7(s):
        return s.rolling(7, min_periods=1).mean()

    v = pd.DataFrame(index=d.index)
    v["prec_7"] = lluvia.rolling(7, min_periods=1).sum()
    v["prec_30"] = lluvia.rolling(30, min_periods=1).sum()
    v["prec_90"] = lluvia.rolling(90, min_periods=1).sum()
    v["dias_sin_lluvia"] = racha.astype(float)
    v["tmax_7"] = media7(tmax)
    v["hrmin_7"] = media7(hr_min)
    v["viento_7"] = media7(viento)
    v["vpd_7"] = media7(vpd)
    for nombre in ("ffmc", "dmc", "dc", "isi", "bui", "fwi"):
        v[nombre] = f[nombre]
    v["fwi_7"] = media7(v["fwi"])
    v["fwi_max7"] = v["fwi"].rolling(7, min_periods=1).max()
    v["suelo_sup_7"] = media7(d["humedad_suelo_sup"])
    v["suelo_raiz_7"] = media7(d["humedad_suelo_raiz"])
    return v[VARS_CLIMA]


def normales(v, anio_ini, anio_fin):
    """Valor normal por día del año (1 a 366), suavizado con media móvil circular de 15 días."""
    sel = v[(v.index.year >= anio_ini) & (v.index.year <= anio_fin)]
    if sel.empty:
        raise ValueError(f"No hay clima entre {anio_ini} y {anio_fin} para calcular normales")
    base = sel[list(_BASE_NORMAL.values())]
    por_dia = (base.groupby(sel.index.dayofyear).mean()
               .reindex(range(1, 367)).interpolate(limit_direction="both"))
    extendido = pd.concat([por_dia.iloc[-15:], por_dia, por_dia.iloc[:15]])
    suave = extendido.rolling(15, center=True, min_periods=1).mean().iloc[15:-15]
    suave.columns = list(_BASE_NORMAL)
    suave["n_prec_anual"] = float(sel["prec_7"].mean() / 7.0 * 365.25)
    suave["n_tmax_anual"] = float(sel["tmax_7"].mean())
    suave.index.name = "dia_anio"
    return suave[VARS_NORMALES]


SERIES_PARA_DISPERSO = 64


def _pesos_idw_dispersos(lon, lat, lon_s, lat_s, k, potencia):
    """Los mismos pesos, en matriz CSR: con cientos de series la versión densa ocuparía gigas."""
    from scipy import sparse
    from scipy.spatial import cKDTree
    # candidatas con una proyección aproximada; la elección final usa la distancia exacta de la
    # versión densa (coseno de la latitud de cada celda), para que ambas den las mismas vecinas
    escala = np.cos(np.deg2rad(np.mean(lat))) * KM_POR_GRADO_LON_ECUADOR
    arbol = cKDTree(np.column_stack([lon_s * escala, lat_s * KM_POR_GRADO_LAT]))
    n_cand = min(len(lon_s), k + 8)
    _, cand = arbol.query(np.column_stack([lon * escala, lat * KM_POR_GRADO_LAT]), k=n_cand)
    cand = cand.reshape(len(lon), n_cand)
    dx = (lon[:, None] - lon_s[cand]) * np.cos(np.deg2rad(lat))[:, None] * KM_POR_GRADO_LON_ECUADOR
    dy = (lat[:, None] - lat_s[cand]) * KM_POR_GRADO_LAT
    dist = np.hypot(dx, dy)
    mejores = np.argpartition(dist, k - 1, axis=1)[:, :k]
    cerca = np.take_along_axis(cand, mejores, axis=1)
    w = 1.0 / np.maximum(np.take_along_axis(dist, mejores, axis=1), 1e-6) ** potencia
    w /= w.sum(axis=1, keepdims=True)
    filas = np.repeat(np.arange(len(lon)), k)
    return sparse.csr_matrix((w.ravel().astype(np.float32), (filas, cerca.ravel())),
                             shape=(len(lon), len(lon_s)), dtype=np.float32)


def pesos_idw(lon, lat, lon_s, lat_s, k=4, potencia=2.0, disperso=None):
    """Pesos de interpolación por inverso de la distancia hacia las k series más cercanas.

    Devuelve una matriz [celdas, series]: densa con pocas series (POWER) y dispersa con muchas
    (ERA5-Land), salvo que `disperso` diga otra cosa.
    """
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    lon_s, lat_s = np.asarray(lon_s, float), np.asarray(lat_s, float)
    if disperso is None:
        disperso = len(lon_s) > SERIES_PARA_DISPERSO
    if disperso:
        return _pesos_idw_dispersos(lon, lat, lon_s, lat_s, min(k, len(lon_s)), potencia)
    dx = (lon[:, None] - lon_s[None, :]) * np.cos(np.deg2rad(lat))[:, None] * KM_POR_GRADO_LON_ECUADOR
    dy = (lat[:, None] - lat_s[None, :]) * KM_POR_GRADO_LAT
    dist = np.hypot(dx, dy)
    k = min(k, len(lon_s))
    cerca = np.argpartition(dist, k - 1, axis=1)[:, :k]
    w = 1.0 / np.maximum(np.take_along_axis(dist, cerca, axis=1), 1e-6) ** potencia
    w /= w.sum(axis=1, keepdims=True)
    pesos = np.zeros(dist.shape, dtype=np.float32)
    np.put_along_axis(pesos, cerca, w.astype(np.float32), axis=1)
    return pesos


class ClimaSemanal:
    """Clima por serie listo para consultar por semana, sin mirar hacia adelante."""

    def __init__(self, tabla, fecha0, valores, normal, informe, retraso=0, fuente=FUENTE_POWER):
        self.tabla = tabla          # serie, lat_c, lon_c, provincias
        self.retraso = int(retraso)  # días entre el último clima usado y el domingo previo
        self.fuente = fuente
        self.fecha0 = fecha0        # primer día de `valores`
        self.valores = valores      # [n_series, n_dias, len(VARS_CLIMA)]
        self._normal = normal       # [n_series, 366, len(VARS_NORMALES)]
        self.informe = informe
        self._i_anom = [VARS_CLIMA.index(b) for b in _BASE_ANOMALIA.values()]
        self._i_norm = [VARS_NORMALES.index("n_" + b) for b in _BASE_ANOMALIA.values()]

    @classmethod
    def desde_tabla(cls, df, anios_normales=(2012, 2021), retraso=None):
        fuente = fuente_de(df)
        if retraso is None:
            retraso = config.DIAS_RETRASO_CLIMA.get(fuente, 0)
        limpio, informe = limpiar_clima(df)
        series, tabla = series_unicas(limpio, fuente)
        fecha0, fecha1 = series["fecha"].min(), series["fecha"].max()
        dias = pd.date_range(fecha0, fecha1)
        valores = np.full((len(tabla), len(dias), len(VARS_CLIMA)), np.nan, dtype=np.float32)
        normal = np.empty((len(tabla), 366, len(VARS_NORMALES)), dtype=np.float32)
        for fila in tabla.itertuples():
            d = series[series["serie"] == fila.serie].set_index("fecha")[COLUMNAS_METEO]
            v = variables_diarias(d, fila.lat_c)
            valores[fila.serie] = v.reindex(dias).to_numpy(dtype=np.float32)
            normal[fila.serie] = normales(v, *anios_normales).to_numpy(dtype=np.float32)
        return cls(tabla, fecha0.date(), valores, normal, informe, retraso=retraso, fuente=fuente)

    @classmethod
    def desde_csv(cls, ruta, anios_normales=(2012, 2021), retraso=None):
        return cls.desde_tabla(cargar_clima(ruta), anios_normales, retraso)

    def _vispera(self, semana):
        """Último día de clima que puede usar la semana: el domingo previo menos el retraso."""
        return inicio_semana(semana) - timedelta(days=1 + self.retraso)

    def disponible(self, semana):
        """¿Hay clima observado hasta la víspera (el domingo previo menos el retraso)?"""
        dia = (self._vispera(semana) - self.fecha0).days
        return 0 <= dia < self.valores.shape[1]

    def antecedente(self, semana):
        """Estado del clima al domingo anterior: [n_series, len(VARS_ANTECEDENTE)]."""
        if not self.disponible(semana):
            raise ValueError(f"No hay clima observado hasta el {self._vispera(semana)}")
        vispera = self._vispera(semana)
        actual = self.valores[:, (vispera - self.fecha0).days, :]
        normal = self._normal[:, vispera.timetuple().tm_yday - 1, :]
        anomalia = actual[:, self._i_anom] - normal[:, self._i_norm]
        return np.concatenate([actual, anomalia], axis=1)

    def normal(self, semana):
        """Clima normal de esa época del año: [n_series, len(VARS_NORMALES)]. Existe para cualquier fecha."""
        return self._normal[:, info_semana(semana)["dia_anio"] - 1, :]
