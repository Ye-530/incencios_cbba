"""Flujo de trabajo: cargar los datos, armar la muestra de entrenamiento y puntuar la malla completa."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import config, focos as mod_focos
from .capas import CapasExtra
from .clima import ClimaSemanal, pesos_idw
from .malla import malla_cochabamba, rasterizar_capas, rasterizar_geometria
from .muestreo import muestrear
from .rasters import archivos_de, variables_estaticas
from .rutas import Rutas
from .tiempo import info_semana, semana_de_fecha, semanas_de_anio
from .variables import Constructor

# Archivos de clima en orden de preferencia, según INCENDIOS_CLIMA
_CLIMA_POR_FUENTE = {"auto": ("clima_era5", "clima_power_nativo", "clima_power"),
                     "era5": ("clima_era5",),
                     "power": ("clima_power_nativo", "clima_power")}
# Partes de la huella que, si cambian, obligan a reentrenar (focos más recientes no).
CLAVES_HUELLA_MODELO = ("clima", "teselas", "capas")


def _de_huella(huella, clave):
    return huella.get(clave, [] if clave == "capas" else None)       # huellas viejas no tenían capas


def huellas_distintas(una, otra):
    return any(_de_huella(una, k) != _de_huella(otra, k) for k in CLAVES_HUELLA_MODELO)


def datos_cambiaron(carpeta, huella):
    """¿Los datos de ahora son otros que los anotados en `carpeta` al entrenar? (sin anotación: sí)."""
    ruta = Path(carpeta) / "huella.json"
    if not ruta.exists():
        return True
    return huellas_distintas(json.loads(ruta.read_text(encoding="utf-8")), huella)


def anotar_huella(carpeta, huella):
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    (Path(carpeta) / "huella.json").write_text(json.dumps(huella, indent=1, ensure_ascii=False), encoding="utf-8")


def capas_activadas():
    """Las capas extra se usan si existen, salvo INCENDIOS_CAPAS=0 (para comparar con y sin ellas)."""
    return os.environ.get("INCENDIOS_CAPAS", "1").strip().lower() not in ("0", "no", "false")


def archivo_clima(rutas):
    """Archivo de clima a usar: ERA5-Land si ya se descargó; si no, POWER en malla nativa; si no, el original.

    La variable de entorno INCENDIOS_CLIMA=power o =era5 fuerza una fuente (para comparar modelos).
    """
    forzada = os.environ.get("INCENDIOS_CLIMA", "auto").strip().lower() or "auto"
    if forzada not in _CLIMA_POR_FUENTE:
        raise ValueError(f"INCENDIOS_CLIMA debe ser auto, power o era5, no {forzada!r}")
    for nombre in _CLIMA_POR_FUENTE[forzada]:
        if getattr(rutas, nombre).exists():
            return getattr(rutas, nombre)
    cual = {"era5": "ERA5-Land ", "power": "POWER "}.get(forzada, "")
    raise FileNotFoundError(f"No hay archivo de clima {cual}en {rutas.clima_power.parent} "
                            f"ni en {rutas.clima_era5.parent}")


@dataclass
class Contexto:
    """Todo lo necesario para construir variables y mapas, ya cargado."""
    malla: object
    mascara: np.ndarray           # celdas del departamento [nfilas, ncols]
    idx: np.ndarray               # id de celda de cada fila de las matrices
    lon: np.ndarray               # centro de cada celda del departamento
    lat: np.ndarray
    municipio_id: np.ndarray
    municipios: list
    provincia_id: np.ndarray
    provincias: list
    provincia_de_municipio: list
    zona_id: np.ndarray
    zonas: list
    constructor: Constructor
    clima: ClimaSemanal
    dias_obs: np.ndarray          # días con satélite por semana
    conteo_diario: object         # detecciones diarias en todo el país
    focos: object                 # detecciones de vegetación en el recuadro, sin fuentes de calor fijas
    n_semanas: int
    rutas: Rutas
    notas: dict = field(default_factory=dict)
    fuentes: object = None        # celdas retiradas por ser fuentes de calor fijas (tabla)

    @property
    def n(self):
        return len(self.idx)

    def semana_valida(self, semana):
        """Semana con satélite suficiente, ella y la anterior (la anterior alimenta el fuego reciente)."""
        return (1 <= semana < self.n_semanas
                and self.dias_obs[semana] >= config.MIN_DIAS_OBSERVADOS
                and self.dias_obs[semana - 1] >= config.MIN_DIAS_OBSERVADOS)

    def semanas_utiles(self, anio_ini, anio_fin):
        """Semanas de esos años ISO que sirven para entrenar o evaluar."""
        semanas = [s for anio in range(anio_ini, anio_fin + 1) for s in semanas_de_anio(anio)]
        return [s for s in semanas if self.semana_valida(s) and self.constructor.tiene_historial(s)]

    def a_malla(self, valores, relleno=np.nan):
        """Pasa un vector por celda del departamento a un arreglo 2D de la malla."""
        salida = np.full(self.malla.nfilas * self.malla.ncols, relleno, dtype=np.asarray(valores).dtype
                         if not np.isnan(relleno) else np.float32)
        salida[self.idx] = valores
        return salida.reshape(self.malla.forma)


def _leer_geojson(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _rellenar_sin_asignar(ids, mascara):
    """A las celdas del departamento sin entidad les da la de la celda asignada más cercana."""
    falta = mascara & (ids < 0)
    if not falta.any():
        return ids
    _, (fi, co) = distance_transform_edt(ids < 0, return_indices=True)
    ids = ids.copy()
    ids[falta] = ids[fi[falta], co[falta]]
    return ids


def _firma_teselas(rutas):
    usados = archivos_de(rutas.cobertura, "*.tif") + archivos_de(rutas.relieve, "*.tif")
    return json.dumps([[p.name, p.stat().st_size] for p in usados])


def cargar_estaticas(malla, rutas, verboso=True):
    """Variables estáticas, con caché en data/interim que se invalida si cambian las teselas."""
    cache = rutas.estaticas
    firma = _firma_teselas(rutas) + json.dumps(list(malla.forma))
    if cache.exists():
        with np.load(cache, allow_pickle=False) as datos:
            if str(datos["firma"]) == firma:
                return {k: datos[k] for k in datos.files if k != "firma"}
    if verboso:
        print("  agregando rásteres de cobertura y relieve a la malla (tarda ~20 s)...")
    est = variables_estaticas(malla, rutas.cobertura, rutas.relieve)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, firma=np.array(firma), **est)
    return est


def cargar_todo(rutas=None, malla=None, verboso=True):
    """Lee los datos y deja listo el constructor de variables."""
    rutas = rutas or Rutas.del_proyecto()
    malla = malla or malla_cochabamba()
    geo = rutas.geo
    if not (geo / "cochabamba_limite.geojson").exists():
        raise FileNotFoundError(
            f"No se encuentra {geo / 'cochabamba_limite.geojson'}. Los datos originales (límites, focos, "
            f"relieve, cobertura y clima) van en {rutas.raw}: vea la sección Datos del README.")
    ruta_clima = archivo_clima(rutas)

    if verboso:
        print("Cargando límites, rásteres, focos y clima...")
    limite = _leer_geojson(geo / "cochabamba_limite.geojson")
    mascara = np.zeros(malla.forma, dtype=bool)
    for entidad in limite["features"]:
        mascara |= rasterizar_geometria(malla, entidad["geometry"])
    gj_muni = _leer_geojson(geo / "cochabamba_municipios.geojson")
    muni, municipios = rasterizar_capas(malla, gj_muni, "NAME_3")
    prov, provincias = rasterizar_capas(malla, _leer_geojson(geo / "cochabamba_provincias.geojson"), "NAME_2")
    muni, prov = _rellenar_sin_asignar(muni, mascara), _rellenar_sin_asignar(prov, mascara)
    provincia_de_municipio = [e["properties"]["NAME_2"] for e in gj_muni["features"]]

    idx = np.flatnonzero(mascara.ravel())
    lon_c, lat_c = malla.centros()
    LON, LAT = np.meshgrid(lon_c, lat_c)
    zonas = list(config.ZONAS) + [config.ZONA_RESTO]
    zona_de_muni = np.full(len(municipios), len(zonas) - 1, dtype=np.int8)
    for z, nombres in enumerate(config.ZONAS.values()):
        for nombre in nombres:
            if nombre in municipios:
                zona_de_muni[municipios.index(nombre)] = z
    municipio_id = muni.ravel()[idx]

    estaticas = cargar_estaticas(malla, rutas, verboso)
    detecciones, conteo = mod_focos.cargar_focos(rutas.focos, malla)
    vegetacion = mod_focos.filtrar_vegetacion(detecciones)
    # hornos, chimeneas y antorchas de gas que FIRMS entrega como "vegetación": fuera
    fuentes = mod_focos.fuentes_persistentes(vegetacion)
    vegetacion = mod_focos.quitar_fuentes(vegetacion, fuentes)
    fila_f, col_f = np.divmod(fuentes["celda"].to_numpy(dtype=np.int64), malla.ncols)
    fuentes = fuentes.assign(
        latitud=lat_c[fila_f], longitud=lon_c[col_f], en_departamento=mascara[fila_f, col_f],
        municipio=[municipios[m] if m >= 0 else "" for m in muni[fila_f, col_f]])
    positivos = mod_focos.positivos_semanales(vegetacion, malla)
    n_semanas = int(semana_de_fecha(conteo.index[-1])) + 1
    dias_obs = mod_focos.dias_observados(conteo, n_semanas)

    clima = ClimaSemanal.desde_csv(ruta_clima)
    pesos = pesos_idw(LON.ravel()[idx], LAT.ravel()[idx], clima.tabla["lon_c"], clima.tabla["lat_c"])
    extra = CapasExtra.desde_carpeta(rutas.capas, malla) if capas_activadas() else None
    constructor = Constructor(malla, mascara, estaticas, positivos, clima, pesos, n_semanas, capas=extra)

    notas = {
        "celdas": int(len(idx)),
        "detecciones_recuadro": int(len(detecciones)),
        "detecciones_vegetacion": int(len(vegetacion)),
        "fuentes_persistentes": int(len(fuentes)),
        "detecciones_en_fuentes": int(fuentes["detecciones"].sum()),
        "series_clima": int(len(clima.tabla)),
        "fuente_clima": clima.fuente,
        "retraso_clima": clima.retraso,
        "limpieza_clima": clima.informe,
        "cobertura_dem": float(np.isfinite(estaticas["elev_media"].ravel()[idx]).mean()),
        "cobertura_landcover": float(np.isfinite(estaticas["lc_arbol"].ravel()[idx]).mean()),
        "ultima_fecha_focos": str(conteo.index[-1].date()),
        "focos_por_satelite": {f: [str(g.min().date()), str(g.max().date())]
                               for f, g in detecciones.groupby("fuente", sort=False)["fecha"]},
        "capas": extra.resumen() if extra else {},
        # con qué datos se construyó todo: si cambian, hay que volver a entrenar
        "huella": {
            "clima": ruta_clima.name,
            "retraso_clima": clima.retraso,
            "teselas": sorted(nombre for nombre, _ in json.loads(_firma_teselas(rutas))),
            "focos_hasta": str(conteo.index[-1].date()),
            "capas": sorted(extra.capas) if extra else [],
        },
    }
    if verboso:
        print(f"  {notas['celdas']:,} celdas; {notas['detecciones_vegetacion']:,} detecciones de vegetación; "
              f"{notas['series_clima']} series de clima; relieve en {notas['cobertura_dem']:.0%} "
              f"y cobertura en {notas['cobertura_landcover']:.0%} de las celdas")
        if extra:
            print("  capas extra: " + ", ".join(
                f"{n} ({r['anios'][0]}–{r['anios'][1]})" if r["anios"] != "fija" else n
                for n, r in notas["capas"].items()))
        if len(fuentes):
            print(f"  se retiraron {len(fuentes)} celdas con fuentes de calor fijas "
                  f"({notas['detecciones_en_fuentes']} detecciones nocturnas todo el año)")
    return Contexto(
        malla=malla, mascara=mascara, idx=idx, lon=LON.ravel()[idx], lat=LAT.ravel()[idx],
        municipio_id=municipio_id, municipios=municipios,
        provincia_id=prov.ravel()[idx], provincias=provincias,
        provincia_de_municipio=provincia_de_municipio,
        zona_id=zona_de_muni[municipio_id], zonas=zonas,
        constructor=constructor, clima=clima, dias_obs=dias_obs, conteo_diario=conteo,
        focos=vegetacion, n_semanas=n_semanas, rutas=rutas,
        notas=notas, fuentes=fuentes)


# ---------------------------------------------------------------------- muestra de entrenamiento
def construir_muestra(ctx, semanas, n_neg, semilla=config.SEMILLA, verboso=True):
    """Muestra caso-control: por semana, todas las celdas con fuego y `n_neg` sin fuego.

    Devuelve la matriz en régimen operativo (el estacional son sus primeras columnas).
    """
    rng = np.random.default_rng(semilla)
    c = ctx.constructor
    i_cerca = c.nombres("operativo").index("rec_v11_4")
    X, y, sem, fila, nuevo = [], [], [], [], []
    negativos_totales = negativos_muestra = 0
    for k, s in enumerate(semanas):
        etiquetas = c.etiquetas(s)
        elegidas = muestrear(etiquetas, n_neg, rng)
        matriz = c.matriz(s, "operativo")[elegidas]
        X.append(matriz)
        y.append(etiquetas[elegidas])
        sem.append(np.full(len(elegidas), s, dtype=np.int32))
        fila.append(elegidas.astype(np.int32))
        nuevo.append(matriz[:, i_cerca] == 0)
        negativos_totales += int((etiquetas == 0).sum())
        negativos_muestra += int((etiquetas[elegidas] == 0).sum())
        if verboso and (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(semanas)} semanas")
    return {
        "X": np.concatenate(X), "y": np.concatenate(y), "semana": np.concatenate(sem),
        "fila": np.concatenate(fila), "nuevo": np.concatenate(nuevo),
        "anio": np.array([info_semana(s)["anio"] for s in np.concatenate(sem)], dtype=np.int16),
        "tasa_neg": negativos_muestra / max(1, negativos_totales),
    }


def columnas_de(ctx, regimen):
    """Posición, dentro de la matriz operativa, de las variables de un régimen."""
    operativo = ctx.constructor.nombres("operativo")
    return np.array([operativo.index(n) for n in ctx.constructor.nombres(regimen)])


# ---------------------------------------------------------------------- línea base "provincia"
def serie_de_provincia(ctx):
    """Índice de la serie de clima que corresponde a cada provincia.

    Por nombre cuando el archivo de clima viene por provincia; si no (clima en malla
    nativa), la serie cuyo centro queda más cerca del centro de la provincia.
    """
    def limpio(nombre):
        return "".join(ch for ch in nombre.lower() if ch.isalnum())
    tabla = ctx.clima.tabla
    por_nombre = {}
    for fila in tabla.itertuples():
        for nombre in fila.provincias.split(","):
            por_nombre[limpio(nombre)] = fila.serie
    if all(limpio(p) in por_nombre for p in ctx.provincias):
        return np.array([por_nombre[limpio(p)] for p in ctx.provincias])
    series = []
    for p in range(len(ctx.provincias)):
        lon, lat = ctx.lon[ctx.provincia_id == p].mean(), ctx.lat[ctx.provincia_id == p].mean()
        distancia = np.hypot((tabla["lon_c"] - lon) * np.cos(np.deg2rad(lat)), tabla["lat_c"] - lat)
        series.append(int(tabla["serie"].iloc[int(np.argmin(distancia))]))
    return np.array(series)


def matriz_provincia(ctx, semana, filas=None):
    """Variables del enfoque anterior: clima de la provincia, época del año y provincia.

    Todas las celdas de una provincia reciben exactamente los mismos valores.
    """
    serie = serie_de_provincia(ctx)
    antecedente = ctx.clima.antecedente(semana)[serie]           # [n_prov, vars]
    info = info_semana(semana)
    angulo = 2.0 * np.pi * info["dia_anio"] / 365.25
    epoca = np.array([np.sin(angulo), np.cos(angulo), info["semana_anio"]], dtype=np.float32)
    por_provincia = np.hstack([antecedente, np.tile(epoca, (len(serie), 1)),
                               np.eye(len(serie), dtype=np.float32)]).astype(np.float32)
    prov = ctx.provincia_id if filas is None else ctx.provincia_id[filas]
    return por_provincia[prov]


# ---------------------------------------------------------------------- puntuar la malla completa
def puntuar_malla(ctx, semanas, puntuadores, carpeta, verboso=True):
    """Calcula el puntaje de todas las celdas para cada semana y cada modelo.

    `puntuadores`: {nombre: función(ctx, semana, X_operativo) -> vector de n puntajes}.
    Escribe en `carpeta` un .npy por modelo [n_semanas, n] más las etiquetas y metadatos.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    c = ctx.constructor
    i_cerca = c.nombres("operativo").index("rec_v11_4")
    forma = (len(semanas), ctx.n)
    np.save(carpeta / "semanas.npy", np.asarray(semanas, dtype=np.int32))
    y = np.lib.format.open_memmap(carpeta / "y.npy", mode="w+", dtype=np.uint8, shape=forma)
    y_tol = np.lib.format.open_memmap(carpeta / "y_tol.npy", mode="w+", dtype=np.uint8, shape=forma)
    nuevo = np.lib.format.open_memmap(carpeta / "nuevo.npy", mode="w+", dtype=np.bool_, shape=forma)
    pred = {nombre: np.lib.format.open_memmap(carpeta / f"pred_{nombre}.npy", mode="w+",
                                              dtype=np.float32, shape=forma)
            for nombre in puntuadores}
    from .metricas import dilatar_etiquetas
    for k, s in enumerate(semanas):
        X = c.matriz(s, "operativo")
        y[k] = c.etiquetas(s)
        y_tol[k] = dilatar_etiquetas(c.malla_fuego(s)).ravel()[ctx.idx]
        nuevo[k] = X[:, i_cerca] == 0
        for nombre, funcion in puntuadores.items():
            pred[nombre][k] = funcion(ctx, s, X)
        if verboso and (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(semanas)} semanas puntuadas")
    for arreglo in (y, y_tol, nuevo, *pred.values()):
        arreglo.flush()
    return carpeta


# ---------------------------------------------------------------------- guardar y cargar la muestra
def guardar_muestra(muestra, ctx, carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    np.save(carpeta / "muestra_X.npy", muestra["X"])
    np.savez(carpeta / "muestra_meta.npz", **{k: muestra[k] for k in ("y", "semana", "fila", "nuevo", "anio")})
    c = ctx.constructor
    info = {
        "tasa_neg": muestra["tasa_neg"],
        "filas": int(len(muestra["y"])),
        "positivos": int(muestra["y"].sum()),
        "semanas": int(len(np.unique(muestra["semana"]))),
        "variables": {r: c.nombres(r) for r in ("estacional", "operativo")},
        "grupos": {r: c.grupos(r) for r in ("estacional", "operativo")},
        "notas": ctx.notas,
    }
    (carpeta / "muestra_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    return info


def cargar_muestra(carpeta, ctx=None):
    """Lee la muestra de `preparar`. Con `ctx`, verifica que se armó con las mismas variables."""
    carpeta = Path(carpeta)
    if not (carpeta / "muestra_X.npy").exists():
        raise FileNotFoundError(
            f"No hay muestra en {carpeta}. Ejecute primero: python -m incendios preparar")
    info = json.loads((carpeta / "muestra_info.json").read_text(encoding="utf-8"))
    if ctx is not None and info["variables"]["operativo"] != ctx.constructor.nombres("operativo"):
        antes, ahora = set(info["variables"]["operativo"]), set(ctx.constructor.nombres("operativo"))
        raise ValueError(
            f"La muestra de {carpeta} se armó con otras variables (nuevas: {sorted(ahora - antes)[:6]}, "
            f"que ya no están: {sorted(antes - ahora)[:6]}). Cambiaron las capas o el clima: vuelva a "
            f"ejecutar: python -m incendios preparar")
    muestra = {"X": np.load(carpeta / "muestra_X.npy"), "tasa_neg": info["tasa_neg"]}
    with np.load(carpeta / "muestra_meta.npz") as meta:
        muestra.update({k: meta[k] for k in meta.files})
    return muestra, info
