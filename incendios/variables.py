"""Variables explicativas por (celda, semana), calculadas solo con información del pasado.

Dos regímenes:
- 'estacional': lo que se conoce con meses de anticipación (sirve para cualquier fecha futura).
- 'operativo' : además, el clima hasta la víspera y el fuego de las semanas previas.

Regla de oro: nada de la semana `t` depende de focos de semanas >= t ni de clima
desde el lunes de `t`. En régimen estacional tampoco de focos del mismo año ISO. El fuego de los
últimos días usa solo días anteriores al lunes de `t`.
"""
import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt, uniform_filter

from .clima import VARS_ANTECEDENTE, VARS_NORMALES
from .malla import KM_POR_GRADO_LAT, KM_POR_GRADO_LON_ECUADOR
from .rasters import CLASES_WORLDCOVER
from .tiempo import info_semana, semana_de_iso, semanas_de_anio

REGIMENES = ("estacional", "operativo")

ANIOS_HISTORIAL = 3        # ventana fija de años anteriores
DECAIMIENTO = 0.6          # peso de cada año adicional hacia atrás
VENTANA_EPOCA = 3          # ± semanas alrededor de la misma época del año
VECINDARIOS = (3, 11, 41)  # lado en celdas: ~1,5 km, ~5,5 km, ~20 km

COBERTURA = ["lc_arbol", "lc_arbusto", "lc_pasto", "lc_cultivo", "lc_urbano", "lc_desnudo",
             "lc_agua", "lc_humedal", "lc_combustible", "lc_diversidad",
             "ent_arbol", "ent_pasto", "ent_cultivo", "ent_urbano",
             "dist_urbano", "dist_cultivo", "dist_agua"]
TERRENO = ["elev_media", "elev_rango", "pendiente", "nortitud", "estitud"]
HISTORIAL = ["his_c1", "his_c3", "his_anios3", "his_dec",
             "his_v3_1", "his_v3_3", "his_v3_dec",
             "his_v11_1", "his_v11_3", "his_v11_dec", "his_v41_3",
             "est_c3", "est_v3_3", "est_v11_3", "est_v11_dec"]
EPOCA = ["doy_sen", "doy_cos", "semana_anio"]
RECIENTE = ["rec_c1", "rec_c4", "rec_v3_1", "rec_v3_4", "rec_v11_1", "rec_v11_4",
            "rec_v41_1", "rec_v41_4", "rec_dep_1", "rec_dep_4", "rec_dep_anom",
            "tem_c", "tem_v11"]

_GRUPOS = {
    "cobertura": COBERTURA,
    "terreno": TERRENO,
    "historial": HISTORIAL,
    "epoca": EPOCA,
    "clima_normal": list(VARS_NORMALES),
    "fuego_reciente": RECIENTE,
    "clima_antecedente": list(VARS_ANTECEDENTE),
}
# Fuego de los últimos días antes del lunes (viernes a domingo, y el domingo solo), en la celda, en
# sus entornos y en todo el departamento. Solo modo operativo y solo si el constructor recibe los
# focos por día; va al final de la matriz, así los modelos entrenados sin él siguen leyendo sus
# columnas en el mismo lugar.
FUEGO_DIAS = ["rec_d3_c", "rec_d3_v3", "rec_d3_v11", "rec_d3_v41", "rec_d1_v11", "rec_dep_d3"]
DIAS_RECIENTES = 3

_GRUPOS_POR_REGIMEN = {
    "estacional": ["cobertura", "terreno", "historial", "epoca", "clima_normal"],
    "operativo": ["cobertura", "terreno", "historial", "epoca", "clima_normal",
                  "fuego_reciente", "clima_antecedente"],
}


def _suma_vecindad(a, lado):
    """Suma de `a` en la ventana cuadrada de `lado` celdas centrada en cada celda."""
    return uniform_filter(a.astype(np.float32), size=lado, mode="constant", cval=0.0) * (lado * lado)


def _media_vecindad_con_huecos(a, lado):
    valido = np.isfinite(a)
    suma = _suma_vecindad(np.where(valido, a, 0.0), lado)
    cuenta = _suma_vecindad(valido, lado)
    with np.errstate(invalid="ignore", divide="ignore"):
        media = suma / cuenta
    media[~valido] = np.nan
    return media.astype(np.float32)


def derivadas_estaticas(estaticas, malla):
    """Añade a las fracciones de cobertura: combustible, diversidad, entorno y distancias."""
    d = dict(estaticas)
    clases = [f"lc_{n}" for n in CLASES_WORLDCOVER.values()]
    con_dato = np.isfinite(d["lc_arbol"])
    d["lc_combustible"] = (d["lc_arbol"] + d["lc_arbusto"] + d["lc_pasto"]
                           + d["lc_cultivo"] + d["lc_humedal"]).astype(np.float32)
    fr = np.stack([d[c] for c in clases])
    with np.errstate(invalid="ignore", divide="ignore"):
        d["lc_diversidad"] = (-np.where(fr > 0, fr * np.log(fr), 0.0).sum(axis=0)).astype(np.float32)
    d["lc_diversidad"][~con_dato] = np.nan
    for nombre in ("arbol", "pasto", "cultivo", "urbano"):
        d[f"ent_{nombre}"] = _media_vecindad_con_huecos(d[f"lc_{nombre}"], 11)

    lat_media = malla.norte - malla.nfilas * malla.celda / 2.0
    paso_km = (malla.celda * KM_POR_GRADO_LAT,
               malla.celda * KM_POR_GRADO_LON_ECUADOR * np.cos(np.deg2rad(lat_media)))
    for nombre, umbral in (("urbano", 0.2), ("cultivo", 0.2), ("agua", 0.5)):
        fuente = np.nan_to_num(d[f"lc_{nombre}"], nan=0.0) >= umbral
        if fuente.any():
            dist = distance_transform_edt(~fuente, sampling=paso_km).astype(np.float32)
        else:
            dist = np.full(malla.forma, np.nan, dtype=np.float32)
        dist[~con_dato] = np.nan
        d[f"dist_{nombre}"] = dist
    return d


class Constructor:
    """Arma la matriz de variables de una semana para todas las celdas del departamento."""

    def __init__(self, malla, mascara, estaticas, positivos, clima, pesos, n_semanas, capas=None, diarios=None):
        """`positivos`: tabla (semana, celda) con fuego. `diarios`: tabla (dia, celda), con `dia` contado
        desde el lunes de la semana 0; si se da, el régimen operativo suma el fuego de los últimos días."""
        self.malla = malla
        # capas extra (capas.CapasExtra): sus grupos van al final de ambos regímenes, así los
        # modelos entrenados sin ellas siguen encontrando sus columnas en el mismo lugar
        self.capas = capas
        self._cache_capas = {}
        self.mascara = mascara
        self.idx = np.flatnonzero(mascara.ravel())      # id de celda de cada fila de la matriz
        self.n = len(self.idx)
        self.clima = clima
        # [n, n_series]: densa con pocas series de clima, dispersa (CSR) con ERA5-Land
        self.pesos = pesos.astype(np.float32) if sparse.issparse(pesos) else np.asarray(pesos, dtype=np.float32)
        self.n_semanas = int(n_semanas)

        derivadas = derivadas_estaticas(estaticas, malla)
        self._estaticas = np.column_stack(
            [derivadas[nombre].ravel()[self.idx] for nombre in COBERTURA + TERRENO]).astype(np.float32)

        # focos: lista de celdas por semana (toda la malla, también fuera del departamento)
        pos = positivos[(positivos["semana"] >= 0) & (positivos["semana"] < self.n_semanas)]
        orden = np.argsort(pos["semana"].to_numpy(), kind="stable")
        semanas = pos["semana"].to_numpy()[orden]
        celdas = pos["celda"].to_numpy()[orden].astype(np.int64)
        cortes = np.searchsorted(semanas, np.arange(self.n_semanas + 1))
        self._celdas = [celdas[cortes[s]:cortes[s + 1]] for s in range(self.n_semanas)]

        dentro = mascara.ravel()
        self._n_depto = np.array([int(dentro[c].sum()) for c in self._celdas], dtype=np.float32)

        # focos por día: lista de celdas por día (toda la malla)
        self._celdas_dia = None
        if diarios is not None:
            n_dias = 7 * self.n_semanas
            d = diarios[(diarios["dia"] >= 0) & (diarios["dia"] < n_dias)]
            orden_d = np.argsort(d["dia"].to_numpy(), kind="stable")
            dias = d["dia"].to_numpy()[orden_d]
            celdas_d = d["celda"].to_numpy()[orden_d].astype(np.int64)
            cortes_d = np.searchsorted(dias, np.arange(n_dias + 1))
            self._celdas_dia = [celdas_d[cortes_d[i]:cortes_d[i + 1]] for i in range(n_dias)]

        self.anio_inicial = info_semana(0)["anio"]
        self.anio_ultimo_completo = max(
            anio for anio in range(self.anio_inicial, info_semana(self.n_semanas - 1)["anio"] + 1)
            if semanas_de_anio(anio)[-1] <= self.n_semanas - 1)
        self._cache_historial = {}

    # ------------------------------------------------------------------ utilidades
    def grupos(self, regimen):
        grupos = {g: list(_GRUPOS[g]) for g in _GRUPOS_POR_REGIMEN[regimen]}
        if self.capas is not None:
            grupos.update(self.capas.grupos())
        if regimen == "operativo" and self._celdas_dia is not None:
            grupos["fuego_dias"] = list(FUEGO_DIAS)
        return grupos

    def nombres(self, regimen):
        return [n for variables in self.grupos(regimen).values() for n in variables]

    def _malla_de(self, semanas):
        """Número de semanas con fuego por celda, sumando las semanas indicadas."""
        cuenta = np.zeros(self.malla.nfilas * self.malla.ncols, dtype=np.float32)
        for s in semanas:
            if 0 <= s < self.n_semanas and len(self._celdas[s]):
                cuenta[self._celdas[s]] += 1.0
        return cuenta.reshape(self.malla.forma)

    def malla_fuego(self, semana):
        """Presencia de fuego esa semana en toda la malla (también fuera del departamento)."""
        return self._malla_de([semana]) > 0

    def _depto(self, malla_2d):
        return malla_2d.ravel()[self.idx]

    def _vecindad(self, malla_2d, lado, entera=True):
        suma = _suma_vecindad(malla_2d, lado)
        return self._depto(np.rint(suma) if entera else suma)

    def _a_celdas(self, por_serie):
        """Interpola valores por serie [n_series, p] a las celdas, ignorando series sin dato."""
        valido = np.isfinite(por_serie)
        if valido.all():
            return np.asarray(self.pesos @ por_serie, dtype=np.float32)
        suma = np.asarray(self.pesos @ np.where(valido, por_serie, 0.0).astype(np.float32))
        peso = np.asarray(self.pesos @ valido.astype(np.float32))
        with np.errstate(invalid="ignore", divide="ignore"):
            salida = suma / peso
        salida[peso <= 0] = np.nan
        return salida

    def tiene_historial(self, semana):
        return info_semana(semana)["anio"] >= self.anio_inicial + ANIOS_HISTORIAL

    def etiquetas(self, semana):
        y = np.zeros(self.malla.nfilas * self.malla.ncols, dtype=np.uint8)
        if 0 <= semana < self.n_semanas:
            y[self._celdas[semana]] = 1
        return y[self.idx]

    # ------------------------------------------------------------------ bloques
    def _capas(self, anio):
        """Variables de las capas extra para el año objetivo (las reglas de año están en CapasExtra)."""
        if anio not in self._cache_capas:
            mapas = self.capas.mapas(anio)
            self._cache_capas = {anio: np.column_stack([m.ravel()[self.idx] for m in mapas]).astype(np.float32)}
        return self._cache_capas[anio]

    def _historial(self, anio):
        """Variables anuales: dependen solo de años ISO anteriores a `anio`."""
        if anio in self._cache_historial:
            return self._cache_historial[anio]
        anuales = {a: self._malla_de(semanas_de_anio(a)) for a in range(self.anio_inicial, anio)}
        ultimo = anuales[anio - 1]
        tres = sum(anuales[a] for a in range(anio - ANIOS_HISTORIAL, anio))
        anios3 = sum((anuales[a] > 0).astype(np.float32) for a in range(anio - ANIOS_HISTORIAL, anio))
        dec = sum(DECAIMIENTO ** (anio - 1 - a) * m for a, m in anuales.items())
        bloque = np.column_stack([
            self._depto(ultimo), self._depto(tres), self._depto(anios3), self._depto(dec),
            self._vecindad(ultimo, 3), self._vecindad(tres, 3), self._vecindad(dec, 3, entera=False),
            self._vecindad(ultimo, 11), self._vecindad(tres, 11), self._vecindad(dec, 11, entera=False),
            self._vecindad(tres, 41),
        ]).astype(np.float32)
        self._cache_historial = {anio: bloque}      # un solo año en memoria
        return bloque

    def _misma_epoca(self, anio, semana_anio):
        """Semanas del año `anio` a ±VENTANA_EPOCA de `semana_anio` (sin cruzar de año)."""
        del_anio = semanas_de_anio(anio)
        primera = max(1, semana_anio - VENTANA_EPOCA)
        ultima = min(len(del_anio), semana_anio + VENTANA_EPOCA)
        return del_anio[primera - 1:ultima]

    def _recurrencia(self, anio, semana_anio):
        tres = np.zeros(self.malla.forma, dtype=np.float32)
        dec = np.zeros(self.malla.forma, dtype=np.float32)
        for a in range(self.anio_inicial, anio):
            m = self._malla_de(self._misma_epoca(a, semana_anio))
            dec += DECAIMIENTO ** (anio - 1 - a) * m
            if a >= anio - ANIOS_HISTORIAL:
                tres += m
        return np.column_stack([
            self._depto(tres), self._vecindad(tres, 3), self._vecindad(tres, 11),
            self._vecindad(dec, 11, entera=False),
        ]).astype(np.float32)

    def _epoca(self, info):
        angulo = 2.0 * np.pi * info["dia_anio"] / 365.25
        fila = np.array([np.sin(angulo), np.cos(angulo), info["semana_anio"]], dtype=np.float32)
        return np.broadcast_to(fila, (self.n, 3))

    def _reciente(self, semana, info):
        una = self._malla_de([semana - 1])
        cuatro = self._malla_de(range(semana - 4, semana))
        dep_1 = self._n_depto[semana - 1]
        dep_4 = float(self._n_depto[semana - 4:semana].sum())
        # mismas cuatro semanas (por número ISO) en los tres años anteriores
        referencia = []
        for atras in range(1, ANIOS_HISTORIAL + 1):
            indices = []
            for s in range(semana - 4, semana):
                i = info_semana(s)
                n_sem = len(semanas_de_anio(i["anio"] - atras))
                indices.append(semana_de_iso(i["anio"] - atras, min(i["semana_anio"], n_sem)))
            if min(indices) >= 0:          # ese año de referencia existe en el registro
                referencia.append(float(self._n_depto[indices].sum()))
        anomalia = np.log((dep_4 + 10.0) / (np.mean(referencia) + 10.0)) if referencia else 0.0
        temporada = self._malla_de(range(semana_de_iso(info["anio"], 1), semana))
        bloque = np.column_stack([
            self._depto(una), self._depto(cuatro),
            self._vecindad(una, 3), self._vecindad(cuatro, 3),
            self._vecindad(una, 11), self._vecindad(cuatro, 11),
            self._vecindad(una, 41), self._vecindad(cuatro, 41),
            np.full(self.n, dep_1), np.full(self.n, dep_4), np.full(self.n, anomalia),
            self._depto(temporada), self._vecindad(temporada, 11),
        ]).astype(np.float32)
        return bloque

    def _malla_dias(self, dias):
        """1 en las celdas con fuego en alguno de esos días (índices de día desde la semana 0)."""
        presencia = np.zeros(self.malla.nfilas * self.malla.ncols, dtype=np.float32)
        for i in dias:
            if 0 <= i < len(self._celdas_dia) and len(self._celdas_dia[i]):
                presencia[self._celdas_dia[i]] = 1.0
        return presencia.reshape(self.malla.forma)

    def _dias(self, semana):
        """Fuego de viernes a domingo de la semana anterior y del domingo solo: días < lunes de `semana`."""
        lunes = 7 * semana
        ultimos = self._malla_dias(range(lunes - DIAS_RECIENTES, lunes))
        domingo = self._malla_dias([lunes - 1])
        en_depto = float(self._depto(ultimos).sum())
        return np.column_stack([
            self._depto(ultimos), self._vecindad(ultimos, 3), self._vecindad(ultimos, 11),
            self._vecindad(ultimos, 41), self._vecindad(domingo, 11), np.full(self.n, en_depto),
        ]).astype(np.float32)

    # ------------------------------------------------------------------ matriz
    def matriz(self, semana, regimen, anio_historial=None):
        """Variables de todas las celdas del departamento para la semana pedida: float32 [n, p].

        `anio_historial` permite, en régimen estacional, pedir un año para el que el
        historial todavía no está cerrado usando el de un año anterior (queda a cargo
        de quien llama avisar que el historial es viejo).
        """
        if regimen not in REGIMENES:
            raise ValueError(f"Régimen desconocido: {regimen}")
        info = info_semana(semana)
        anio = info["anio"] if anio_historial is None else int(anio_historial)
        if anio < self.anio_inicial + ANIOS_HISTORIAL:
            raise ValueError(
                f"Sin historial suficiente: se necesitan {ANIOS_HISTORIAL} años de focos antes de {anio}")
        if anio - 1 > self.anio_ultimo_completo:
            raise ValueError(
                f"El historial de focos llega hasta {self.anio_ultimo_completo}; "
                f"no alcanza para {anio}. Actualice los focos o fije anio_historial="
                f"{self.anio_ultimo_completo + 1}")
        bloques = [
            self._estaticas,
            self._historial(anio),
            self._recurrencia(anio, info["semana_anio"]),
            self._epoca(info),
            self._a_celdas(self.clima.normal(semana)),
        ]
        if regimen == "operativo":
            if anio_historial is not None:
                raise ValueError("El régimen operativo no admite un historial desplazado")
            if not (4 <= semana <= self.n_semanas):
                raise ValueError(
                    f"El régimen operativo necesita focos de las 4 semanas previas a la semana {semana}")
            bloques += [self._reciente(semana, info), self._a_celdas(self.clima.antecedente(semana))]
        if self.capas is not None:
            bloques.append(self._capas(info["anio"]))      # año real: CapasExtra avisa si es viejo
        if regimen == "operativo" and self._celdas_dia is not None:
            bloques.append(self._dias(semana))
        return np.concatenate(bloques, axis=1).astype(np.float32, copy=False)
