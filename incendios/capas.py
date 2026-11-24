"""Capas extra (Earth Engine, caminos) en la malla del proyecto: un archivo .npz por capa.

Cada archivo guarda variables anuales [años, filas, columnas], variables fijas [filas,
columnas], los años, la malla con que se hizo y de dónde salió (meta). Las variables que el
modelo construye con ellas están más abajo, en `CapasExtra`.
"""
import json
from pathlib import Path

import numpy as np

from .variables import _media_vecindad_con_huecos, _suma_vecindad

_ANUAL, _FIJA = "a__", "e__"


def _malla_dict(malla):
    return {"oeste": float(malla.oeste), "norte": float(malla.norte), "celda": float(malla.celda),
            "ncols": int(malla.ncols), "nfilas": int(malla.nfilas)}


class Capa:
    def __init__(self, nombre, anios, anuales, estaticas, meta):
        self.nombre = nombre
        self.anios = [int(a) for a in anios]
        self.anuales = anuales          # {variable: [años, filas, columnas]}
        self.estaticas = estaticas      # {variable: [filas, columnas]}
        self.meta = meta

    def anual(self, variable, anio):
        return self.anuales[variable][self.anios.index(int(anio))]

    def estatica(self, variable):
        return self.estaticas[variable]


def guardar(ruta, nombre, malla, anios, anuales, estaticas, meta=None):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {}, nombre=nombre, malla=_malla_dict(malla))
    arreglos = {"_meta": np.array(json.dumps(meta, ensure_ascii=False)),
                "_anios": np.asarray(sorted(int(a) for a in anios), dtype=np.int32)}
    for variable, arr in anuales.items():
        arreglos[_ANUAL + variable] = np.asarray(arr, dtype=np.float32)
    for variable, arr in estaticas.items():
        arreglos[_FIJA + variable] = np.asarray(arr, dtype=np.float32)
    temporal = ruta.with_name(ruta.stem + ".parcial.npz")
    np.savez_compressed(temporal, **arreglos)
    temporal.replace(ruta)
    return ruta


def cargar(ruta, malla):
    with np.load(ruta, allow_pickle=False) as z:
        meta = json.loads(str(z["_meta"]))
        if meta.get("malla") != _malla_dict(malla):
            raise ValueError(f"{Path(ruta).name} se hizo para otra malla ({meta.get('malla')}); vuelva a bajarla")
        anuales = {k[len(_ANUAL):]: z[k] for k in z.files if k.startswith(_ANUAL)}
        estaticas = {k[len(_FIJA):]: z[k] for k in z.files if k.startswith(_FIJA)}
        return Capa(meta.get("nombre", Path(ruta).stem), z["_anios"].tolist(), anuales, estaticas, meta)


# ---------------------------------------------------------------------- variables para el modelo
# grupo de variables -> capa de la que sale
ORIGEN = {"quemas_previas": "quemas", "deforestacion": "perdida_bosque", "uso_suelo": "cobertura",
          "poblacion": "poblacion", "caminos": "caminos"}
GRUPOS_CAPAS = {
    "quemas_previas": ["q_ult", "q_3", "q_desde", "q_v11_3"],
    "deforestacion": ["per_ult", "per_3", "per_acum", "per_v11_3", "arb_2000"],
    "uso_suelo": ["mb_bosque", "mb_natural", "mb_pasto", "mb_agricultura", "mb_mosaico", "mb_no_vegetado",
                  "mb_conversion3", "mb_v11_conversion3", "mb_v11_agro"],
    "poblacion": ["pob_log", "pob_v11_log"],
    "caminos": ["dist_via_km", "dist_huella_km", "vias_km_km2"],
}
# Año de la capa que se usa para el año objetivo Y: Y - desfase. Quemas, pérdida de bosque y
# cobertura de un año se publican meses después de terminado; la población es una estimación.
DESFASE = {"quemas": 1, "perdida_bosque": 1, "cobertura": 1, "poblacion": 0}
UMBRAL_QUEMA = 0.1              # fracción de la celda quemada para contar el año como "ardió"
TOPE_ANIOS_SIN_QUEMA = 15
LADO_ENTORNO = 11               # celdas: ~5,5 km


class CapasExtra:
    """Variables por año objetivo a partir de las capas de data/capas.

    Regla contra la fuga: para el año Y solo entran quemas, pérdida de bosque y cobertura de
    Y-1 o antes, y población de Y. Si una capa todavía no llega a ese año se usa su último año
    (como si el año objetivo fuera el siguiente a ese) y `avisos` lo dice.
    """

    def __init__(self, capas_por_nombre, malla):
        self.malla = malla
        self.capas = {n: capas_por_nombre[n] for n in ORIGEN.values() if n in capas_por_nombre}
        self._cache = {}

    @classmethod
    def desde_carpeta(cls, carpeta, malla):
        """Las capas conocidas que haya en la carpeta; None si no hay ninguna."""
        encontradas = {}
        for nombre in ORIGEN.values():
            ruta = Path(carpeta) / f"{nombre}.npz"
            if ruta.exists():
                encontradas[nombre] = cargar(ruta, malla)
        return cls(encontradas, malla) if encontradas else None

    def grupos(self):
        return {g: list(v) for g, v in GRUPOS_CAPAS.items() if ORIGEN[g] in self.capas}

    def nombres(self):
        return [n for v in self.grupos().values() for n in v]

    def resumen(self):
        return {nombre: {"anios": [c.anios[0], c.anios[-1]] if c.anios else "fija",
                         "fuente": c.meta.get("activo") or c.meta.get("fuente", "")}
                for nombre, c in self.capas.items()}

    def anio_dato(self, nombre, anio):
        """Año de la capa que corresponde al año objetivo; None si no hay ninguno utilizable."""
        capa = self.capas[nombre]
        ideal = int(anio) - DESFASE[nombre]
        previos = [a for a in capa.anios if a <= ideal]
        if previos:
            return max(previos)
        if nombre == "poblacion" and capa.anios:        # antes de la serie: la primera estimación
            return min(capa.anios)
        return None

    def avisos(self, anio):
        salida = []
        for nombre in self.capas:
            if nombre not in DESFASE:
                continue
            ideal, usado = int(anio) - DESFASE[nombre], self.anio_dato(nombre, anio)
            if usado is None:
                salida.append(f"La capa {nombre} no tiene datos de {ideal} ni anteriores: sus variables "
                              f"quedan vacías para {anio}.")
            elif usado < ideal:
                salida.append(f"La capa {nombre} llega hasta {usado} y para {anio} lo ideal sería {ideal}: "
                              f"actualícela con: python -m incendios descargar --que capas")
        return salida

    def mapas(self, anio):
        """Mapas 2D de todas las variables para el año objetivo, en el orden de `nombres()`."""
        anio = int(anio)
        if anio not in self._cache:
            salida = []
            for grupo in self.grupos():
                salida += [np.asarray(m, dtype=np.float32) for m in getattr(self, "_" + grupo)(anio)]
            self._cache = {anio: salida}            # un solo año en memoria
        return self._cache[anio]

    # -------------------------------------------------------------- un método por grupo
    def _vacios(self, k):
        return [np.full(self.malla.forma, np.nan, dtype=np.float32) for _ in range(k)]

    def _serie(self, nombre, variable, hasta):
        capa = self.capas[nombre]
        return {a: capa.anual(variable, a).astype(np.float32) for a in capa.anios if a <= hasta}

    @staticmethod
    def _ultimos_tres(serie, hasta):
        return sum(serie[a] for a in range(hasta - 2, hasta + 1) if a in serie)

    def _quemas_previas(self, anio):
        ultimo = self.anio_dato("quemas", anio)
        if ultimo is None:
            return self._vacios(4)
        serie = self._serie("quemas", "quema", ultimo)
        tres = self._ultimos_tres(serie, ultimo)
        desde = np.full(self.malla.forma, TOPE_ANIOS_SIN_QUEMA, dtype=np.float32)
        for a in sorted(serie):                          # el año más reciente queda al final
            desde[np.nan_to_num(serie[a]) >= UMBRAL_QUEMA] = min(TOPE_ANIOS_SIN_QUEMA, ultimo + 1 - a)
        desde[np.isnan(serie[ultimo])] = np.nan
        return [serie[ultimo], tres, desde, _media_vecindad_con_huecos(tres, LADO_ENTORNO)]

    def _deforestacion(self, anio):
        capa = self.capas["perdida_bosque"]
        arboles = capa.estaticas.get("arboles_2000", self._vacios(1)[0])
        ultimo = self.anio_dato("perdida_bosque", anio)
        if ultimo is None:
            return self._vacios(4) + [arboles]
        serie = self._serie("perdida_bosque", "perdida", ultimo)
        tres = self._ultimos_tres(serie, ultimo)
        return [serie[ultimo], tres, sum(serie.values()), _media_vecindad_con_huecos(tres, LADO_ENTORNO), arboles]

    def _uso_suelo(self, anio):
        ultimo = self.anio_dato("cobertura", anio)
        if ultimo is None:
            return self._vacios(9)
        capa = self.capas["cobertura"]
        fr = {v: capa.anual(v, ultimo).astype(np.float32)
              for v in ("bosque", "natural", "pasto", "agricultura", "mosaico", "no_vegetado", "conversion3")}
        agro = fr["pasto"] + fr["agricultura"] + fr["mosaico"]
        return list(fr.values()) + [_media_vecindad_con_huecos(fr["conversion3"], LADO_ENTORNO),
                                    _media_vecindad_con_huecos(agro, LADO_ENTORNO)]

    def _poblacion(self, anio):
        cual = self.anio_dato("poblacion", anio)
        if cual is None:
            return self._vacios(2)
        personas = self.capas["poblacion"].anual("poblacion", cual).astype(np.float32)
        entorno = np.log1p(_suma_vecindad(np.nan_to_num(personas), LADO_ENTORNO))
        entorno[np.isnan(personas)] = np.nan
        return [np.log1p(personas), entorno]

    def _caminos(self, anio):
        return [self.capas["caminos"].estatica(v) for v in GRUPOS_CAPAS["caminos"]]
