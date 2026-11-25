"""Descarga los datos que mejoran el modelo. Se ejecuta en una PC con internet.

Las claves se leen de credenciales.json, en la carpeta del proyecto (si no existe, copie
credenciales.ejemplo.json con ese nombre y complételo). Nunca se escriben en otro lado.

    copernicus_cds.key      clima ERA5-Land          https://cds.climate.copernicus.eu (su perfil, "API Token")
    earth_engine.proyecto   capas de Earth Engine    ID de su proyecto de Google Cloud registrado en Earth Engine
    firms.map_key           focos NASA FIRMS         https://firms.modaps.eosdis.nasa.gov/api/map_key/

    python -m incendios diagnostico                     qué claves, paquetes y datos hay
    python -m incendios descargar                       lo mismo, y cómo seguir
    python -m incendios descargar --que mejorados       todo lo de abajo menos teselas y clima
    python -m incendios descargar --que clima_era5      clima ERA5-Land de 0,1° desde 2012
    python -m incendios descargar --que capas           quemas, deforestación, uso del suelo y población
    python -m incendios descargar --que caminos         caminos y sendas de OpenStreetMap (sin cuenta)
    python -m incendios descargar --que focos           focos VIIRS: S-NPP hasta su cierre y NOAA-20
    python -m incendios descargar --que teselas         relieve y cobertura que falten (sin cuenta)
    python -m incendios descargar --que clima           clima NASA POWER en su malla nativa (sin cuenta)

Todo lo que se baja va a data/external/; data/raw/ (los datos originales) no se toca.
Paquetes extra (una vez, desde la carpeta del proyecto): python -m pip install -e ".[descargas]"
La primera vez, el clima ERA5-Land tarda horas (son ~180 pedidos mensuales que el servidor de
Copernicus pone en cola); si se corta, vuelva a ejecutarlo: sigue desde donde quedó.
Earth Engine pide iniciar sesión una vez: agregue --autenticar la primera vez.

Después de bajar datos nuevos, reentrene con `python -m incendios entrenar`, o deje que lo haga
`python -m incendios semana`, que reentrena solo cuando hace falta.

AVISO: esta parte no pudo ejecutarse donde se construyó el proyecto (no tenía internet). Lo que
no toca la red está probado, incluido lo que se hace con las respuestas de cada servidor.
"""
import argparse
import importlib.util
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path


import numpy as np
import pandas as pd

from incendios import actualizacion, caminos, capas, capas_ee, config, credenciales, descargas, era5
from incendios.rasters import archivos_de
from incendios.rutas import Rutas
from incendios.malla import malla_cochabamba, rasterizar_geometria

AGENTE = {"User-Agent": "IncendiosCochabamba/2.0 (investigacion forestal; oliverzy530@gmail.com)"}
RECUADRO_BOLIVIA = (-69.7, -23.0, -57.4, -9.6)       # oeste, sur, este, norte
COLUMNAS_FOCOS = ["latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time",
                  "satellite", "instrument", "confidence", "version", "bright_ti5", "frp", "daynight", "type"]
SERVIDORES_OVERPASS = (caminos.URL_OVERPASS, "https://overpass.kumi.systems/api/interpreter")
FIN_SNPP = date(2026, 11, 2)                          # NASA dejó de producir S-NPP ese día a las 13 UTC
OPCIONES = ("diagnostico", "clima_era5", "capas", "caminos", "focos", "mejorados", "teselas", "clima", "todo")


# ---------------------------------------------------------------------- utilidades
def bajar(url, destino=None, intentos=3, espera=5, timeout=180, datos=None):
    """Descarga una dirección (POST si hay `datos`). Devuelve los bytes, o escribe en `destino`."""
    for intento in range(1, intentos + 1):
        try:
            solicitud = urllib.request.Request(url, headers=AGENTE, data=datos)
            with urllib.request.urlopen(solicitud, timeout=timeout) as respuesta:
                contenido = respuesta.read()
            if destino is None:
                return contenido
            temporal = Path(str(destino) + ".parcial")
            temporal.write_bytes(contenido)
            temporal.replace(destino)              # nunca queda un archivo a medias con el nombre final
            return contenido
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if intento == intentos:
                raise RuntimeError(f"No se pudo descargar {url}: {error}") from error
            time.sleep(espera * intento)


def mascara_departamento(rutas, malla):
    ruta = rutas.geo / "cochabamba_limite.geojson"
    if not ruta.exists():
        raise FileNotFoundError(f"No se encuentra {ruta}: los datos originales van en {rutas.raw} "
                                f"(vea la sección Datos del README).")
    limite = json.loads(ruta.read_text(encoding="utf-8"))
    mascara = np.zeros(malla.forma, dtype=bool)
    for entidad in limite["features"]:
        mascara |= rasterizar_geometria(malla, entidad["geometry"])
    return mascara


def leer_credenciales():
    cred = credenciales.cargar(credenciales.dir_proyecto())
    return cred


def falta_paquete(modulo, paquete):
    if importlib.util.find_spec(modulo) is None:
        print(f'  Falta el paquete {paquete}: python -m pip install -e ".[descargas]" '
              "(desde la carpeta del proyecto)")
        return True
    return False


# ---------------------------------------------------------------------- relieve, cobertura y POWER
def descargar_teselas(rutas, malla, mascara):
    trabajos = [(descargas.teselas_dem(mascara, malla), rutas.relieve, descargas.url_dem, "relieve"),
                (descargas.teselas_worldcover(mascara, malla), rutas.cobertura, descargas.url_worldcover, "cobertura")]
    for nombres, (original, carpeta), url_de, etiqueta in trabajos:
        carpeta.mkdir(parents=True, exist_ok=True)
        pendientes = [n for n in descargas.faltantes(nombres, carpeta) if n in descargas.faltantes(nombres, original)]
        print(f"Teselas de {etiqueta}: {len(nombres)} necesarias, {len(pendientes)} por descargar")
        for nombre in pendientes:
            try:
                datos = bajar(url_de(nombre), carpeta / f"{nombre}.tif", timeout=600)
                print(f"  {nombre}.tif  ({len(datos) / 1e6:.0f} MB)")
            except RuntimeError as error:
                print(f"  NO se pudo: {error}")


def descargar_clima_power(rutas, malla, mascara):
    celdas = descargas.celdas_merra(mascara, malla)
    fin = date.today() - timedelta(days=3)          # POWER publica con unos días de retraso
    print(f"Clima NASA POWER en malla nativa: {len(celdas)} celdas, 2012-01-01 a {fin}")
    partes = []
    for k, (lat, lon) in enumerate(celdas, start=1):
        try:
            respuesta = json.loads(bajar(descargas.url_power(lat, lon, "2012-01-01", fin)).decode("utf-8"))
            partes.append(descargas.tabla_power(respuesta, f"celda_{lat}_{lon}", lat, lon))
            print(f"  [{k}/{len(celdas)}] celda {lat}, {lon}")
        except (RuntimeError, KeyError, ValueError) as error:
            print(f"  [{k}/{len(celdas)}] celda {lat}, {lon}: NO se pudo ({error})")
        time.sleep(0.5)
    if not partes:
        print("  No se obtuvo ninguna celda; no se escribe nada.")
        return
    tabla = pd.concat(partes, ignore_index=True)
    # los últimos días pueden venir sin dato: se recorta al último día completo para todas las celdas
    completo = tabla.dropna(subset=["temp_max", "precipitacion"]).groupby("provincia")["fecha"].max().min()
    tabla = tabla[tabla["fecha"] <= completo]
    destino = rutas.clima_power_nativo
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False)
    print(f"  -> {destino}  ({len(tabla):,} filas, hasta {completo})")


# ---------------------------------------------------------------------- ERA5-Land (Copernicus CDS)
def _cliente_cds(cred):
    import cdsapi
    url, clave = cred["copernicus_cds"]["url"], cred["copernicus_cds"]["key"]
    try:
        return cdsapi.Client(url=url, key=clave, quiet=True, progress=False)
    except TypeError:                                   # versiones que no aceptan quiet/progress
        return cdsapi.Client(url=url, key=clave)


def _demasiado_grande(error):
    texto = str(error).lower()
    return any(p in texto for p in ("cost", "too large", "limit", "demasiado"))


def bajar_mes_era5(cred, anio, mes, hasta, recuadro, carpeta):
    """Un mes de ERA5-Land. Si el CDS no acepta el mes entero, lo pide en dos mitades; si los
    últimos días del mes en curso aún no están publicados, prueba con hasta tres días menos."""
    cliente = _cliente_cds(cred)
    destino = era5.archivo_mes(carpeta, anio, mes)

    def pedir(pedido, ruta):
        temporal = ruta.with_name(ruta.stem + ".parcial.nc")
        cliente.retrieve(era5.DATASET, pedido, str(temporal))
        temporal.replace(ruta)

    for atras in range(0, 4 if hasta else 1):
        corte = hasta - timedelta(days=atras) if hasta else None
        if corte is not None and (corte.year, corte.month) != (anio, mes):
            break
        pedido = era5.solicitud(anio, mes, recuadro, hasta=corte)
        try:
            pedir(pedido, destino)
            for mitad in actualizacion.archivos_del_mes(carpeta, anio, mes):
                if mitad != destino:
                    mitad.unlink()                      # el mes entero reemplaza a las mitades viejas
            return f"{anio}-{mes:02d}" + (f" (hasta el {corte.day})" if corte else "")
        except Exception as error:  # noqa: BLE001 - el CDS levanta excepciones de varios tipos
            if _demasiado_grande(error):
                dias = pedido["day"]
                for sufijo, parte in (("a", dias[:len(dias) // 2]), ("b", dias[len(dias) // 2:])):
                    pedir(dict(pedido, day=parte), destino.with_name(f"{destino.stem}_{sufijo}.nc"))
                if destino.exists():
                    destino.unlink()
                return f"{anio}-{mes:02d} (en dos mitades)"
            if corte is None or atras == 3:
                raise
    raise RuntimeError(f"ERA5-Land de {anio}-{mes:02d} todavía no está publicado")


def descargar_clima_era5(rutas, malla, mascara, cred, paralelo=4, hoy=None, desde=actualizacion.INICIO_CLIMA):
    """Baja los meses que faltan (y los últimos, que ERA5 corrige) y arma la tabla diaria."""
    print("Clima ERA5-Land (Copernicus CDS)")
    if not cred["copernicus_cds"]["key"]:
        print("  Falta la clave de Copernicus en credenciales.json (copernicus_cds.key).")
        return False
    if falta_paquete("cdsapi", "cdsapi") or falta_paquete("netCDF4", "netCDF4"):
        return False
    carpeta = rutas.era5_meses
    carpeta.mkdir(parents=True, exist_ok=True)
    recuadro = era5.area(malla)
    pendientes = actualizacion.meses_pendientes(carpeta, hoy or date.today(), desde=desde)
    print(f"  {len(pendientes)} meses por pedir (recuadro N, O, S, E = {recuadro}); "
          f"{paralelo} pedidos a la vez")
    fallidos = []
    with ThreadPoolExecutor(max_workers=max(1, paralelo)) as grupo:
        trabajos = {grupo.submit(bajar_mes_era5, cred, a, m, h, recuadro, carpeta): (a, m) for a, m, h in pendientes}
        for k, trabajo in enumerate(as_completed(trabajos), start=1):
            a, m = trabajos[trabajo]
            try:
                print(f"  [{k}/{len(pendientes)}] {trabajo.result()}", flush=True)
            except Exception as error:  # noqa: BLE001
                fallidos.append((a, m))
                print(f"  [{k}/{len(pendientes)}] {a}-{m:02d}: NO se pudo ({str(error)[:200]})", flush=True)
    if fallidos:
        print(f"  {len(fallidos)} meses fallaron; vuelva a ejecutar para completarlos.")
    print("  Armando la tabla diaria...")
    try:
        ruta, ultimo = actualizacion.armar_clima_era5(carpeta, malla, mascara, rutas.clima_era5,
                                                      exigir_desde=desde, verboso=True)
    except (ValueError, FileNotFoundError) as error:
        print(f"  Todavía no se usa ERA5-Land: {error}")
        return False
    print(f"  -> {ruta}  (hasta {ultimo})")
    return not fallidos


# ---------------------------------------------------------------------- capas de Earth Engine
def descargar_capas(rutas, malla, cred, autenticar=False):
    print("Capas de Google Earth Engine (quemas, pérdida de bosque, uso del suelo, población)")
    proyecto = cred["earth_engine"]["proyecto"]
    if not proyecto:
        print("  Falta el ID del proyecto en credenciales.json (earth_engine.proyecto). Es el ID del proyecto de\n"
              "  Google Cloud que registró en https://code.earthengine.google.com/register (no la clave AIza...).")
        return False
    if falta_paquete("ee", "earthengine-api"):
        return False
    try:
        cliente = capas_ee.ClienteEE(proyecto, autenticar=autenticar)
    except Exception as error:  # noqa: BLE001
        print(f"  No se pudo conectar a Earth Engine: {str(error)[:300]}\n"
              "  Si es la primera vez, ejecute este script con --autenticar (o, en una terminal,\n"
              "  'earthengine authenticate') y verifique que el proyecto esté registrado en Earth Engine.")
        return False
    carpeta = rutas.capas
    todo_bien = True
    for nombre in capas_ee.CAPAS:
        try:
            ruta = capas_ee.descargar(nombre, malla, carpeta, cliente, verboso=True)
            c = capas.cargar(ruta, malla)
            print(f"  {nombre}: {c.anios[0]}–{c.anios[-1]}  -> {ruta}")
        except Exception as error:  # noqa: BLE001 - que una capa que falle no impida las demás
            todo_bien = False
            print(f"  {nombre}: NO se pudo ({str(error)[:300]})")
    return todo_bien


# ---------------------------------------------------------------------- caminos (OpenStreetMap)
def pedir_overpass(consulta, intentos=6):
    datos = urllib.parse.urlencode({"data": consulta}).encode("utf-8")
    for intento in range(intentos):
        servidor = SERVIDORES_OVERPASS[intento % len(SERVIDORES_OVERPASS)]
        try:
            return json.loads(bajar(servidor, datos=datos, intentos=1, timeout=400).decode("utf-8"))
        except (RuntimeError, ValueError) as error:
            if intento == intentos - 1:
                raise
            print(f"      Overpass ocupado ({str(error)[:120]}); reintento en {30 * (intento + 1)} s")
            time.sleep(30 * (intento + 1))


def descargar_caminos(rutas, malla):
    print("Caminos de OpenStreetMap (API Overpass)")
    try:
        ruta = caminos.descargar(malla, rutas.capas, pedir_overpass, verboso=True)
    except Exception as error:  # noqa: BLE001
        print(f"  NO se pudo: {str(error)[:300]}. Vuelva a ejecutar: lo bajado queda guardado.")
        return False
    meta = capas.cargar(ruta, malla).meta
    print(f"  {meta['vias']:,} vías y {meta['huellas']:,} huellas  -> {ruta}")
    return True


# ---------------------------------------------------------------------- focos (NASA FIRMS)
def disponibilidad_firms(clave):
    """Fechas disponibles de cada fuente según FIRMS: {fuente: (desde, hasta)}."""
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{clave}/ALL"
    tabla = pd.read_csv(io.BytesIO(bajar(url)))
    return {fila.data_id: (fila.min_date, fila.max_date) for fila in tabla.itertuples()}


def bajar_periodo(clave, fuente, desde, hasta):
    partes = []
    for fecha, dias in descargas.tramos(desde, hasta):
        datos = bajar(descargas.url_firms(clave, fuente, RECUADRO_BOLIVIA, fecha, dias))
        if datos.strip() and not datos.lstrip().lower().startswith(b"invalid"):
            tabla = pd.read_csv(io.BytesIO(datos))
            if len(tabla):
                partes.append(tabla)
        time.sleep(0.3)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def serie_firms(clave, disponible, prefijo, desde, hasta):
    """Une el producto estándar (SP) con el casi-real (NRT) para los días que el SP aún no cubre."""
    partes = []
    sp, nrt = f"{prefijo}_SP", f"{prefijo}_NRT"
    corte = pd.Timestamp(desde) - pd.Timedelta(days=1)
    if sp in disponible:
        fin_sp = min(pd.Timestamp(hasta), pd.Timestamp(disponible[sp][1]))
        if fin_sp >= pd.Timestamp(desde):
            partes.append(bajar_periodo(clave, sp, desde, fin_sp.date()))
            corte = fin_sp
    if nrt in disponible:
        inicio_nrt = max(corte + pd.Timedelta(days=1), pd.Timestamp(disponible[nrt][0]))
        fin_nrt = min(pd.Timestamp(hasta), pd.Timestamp(disponible[nrt][1]))
        if fin_nrt >= inicio_nrt:
            casi_real = bajar_periodo(clave, nrt, inicio_nrt.date(), fin_nrt.date())
            if len(casi_real):
                casi_real["type"] = 0      # el producto NRT no trae `type`; las fuentes fijas se filtran luego
                partes.append(casi_real)
    partes = [p for p in partes if len(p)]
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=COLUMNAS_FOCOS)


def completar_archivo_focos(destino, clave, disponible, prefijo, desde, hasta):
    """Agrega al CSV anual lo que le falta (repasa los dos últimos días, que el NRT completa)."""
    previo = pd.read_csv(destino) if destino.exists() else None
    if previo is not None and len(previo):
        ultimo = pd.to_datetime(previo["acq_date"]).max().date()
        if ultimo >= hasta:
            return 0
        desde = max(desde, ultimo - timedelta(days=2))
        previo = previo[pd.to_datetime(previo["acq_date"]).dt.date < desde]
    nuevo = serie_firms(clave, disponible, prefijo, desde, hasta)
    tabla = pd.concat([p for p in (previo, nuevo) if p is not None and len(p)] or [nuevo], ignore_index=True)
    tabla = tabla.reindex(columns=COLUMNAS_FOCOS).drop_duplicates()
    temporal = destino.with_name(destino.name + ".parcial")
    tabla.to_csv(temporal, index=False)
    temporal.replace(destino)
    return len(nuevo)


def descargar_focos(rutas, clave, historico=False, hoy=None):
    print("Focos de calor VIIRS (NASA FIRMS)")
    if not clave:
        print("  Falta la MAP_KEY de FIRMS en credenciales.json (firms.map_key), o use --clave.")
        return False
    hoy = hoy or date.today()
    carpeta = rutas.focos_bajados
    carpeta.mkdir(parents=True, exist_ok=True)
    disponible = disponibilidad_firms(clave)
    print("  fuentes disponibles:", {k: v for k, v in disponible.items() if "VIIRS" in k})
    trabajos = [("VIIRS_SNPP", carpeta / "viirs-snpp_2026_Bolivia.csv", date(2026, 1, 1), min(hoy, FIN_SNPP))]
    for anio in range(2018 if historico else 2026, hoy.year + 1):
        trabajos.append(("VIIRS_NOAA20", carpeta / f"viirs-noaa20_{anio}_Bolivia.csv",
                         date(2018, 4, 1) if anio == 2018 else date(anio, 1, 1), min(hoy, date(anio, 12, 31))))
    for prefijo, destino, desde, hasta in trabajos:
        if hasta < desde:
            continue
        nuevas = completar_archivo_focos(destino, clave, disponible, prefijo, desde, hasta)
        print(f"  {destino.name}: {nuevas:,} detecciones nuevas")
    return True


# ---------------------------------------------------------------------- diagnóstico
def diagnostico(rutas, malla, cred):
    print("== Claves (credenciales.json) ==")
    print(f"  Copernicus CDS : {credenciales.ocultar(cred['copernicus_cds']['key'])}")
    print(f"  Earth Engine   : {cred['earth_engine']['proyecto'] or '(vacía)'}")
    print(f"  NASA FIRMS     : {credenciales.ocultar(cred['firms']['map_key'])}")
    print("== Paquetes ==")
    for modulo, paquete in (("cdsapi", "cdsapi"), ("netCDF4", "netCDF4"), ("ee", "earthengine-api")):
        print(f"  {paquete:16s} {'instalado' if importlib.util.find_spec(modulo) else 'FALTA'}")

    print(f"== Datos originales, en {rutas.raw} ==")
    falta_base = []
    limite = rutas.geo / "cochabamba_limite.geojson"
    print(f"  límites        : {'bien' if limite.exists() else 'NO ESTÁN'}")
    if not limite.exists():
        falta_base.append("geo")
    snpp = archivos_de(rutas.focos, config.FUENTES_FOCOS[0]["patron"])
    if snpp:
        ultima = pd.read_csv(snpp[-1], usecols=["acq_date"])["acq_date"].max()
        print(f"  focos S-NPP    : {len(snpp)} archivos, último día {ultima}")
    else:
        print("  focos S-NPP    : NO ESTÁN")
        falta_base.append("focos")
    print(f"  relieve        : {len(archivos_de(rutas.relieve, '*.tif'))} teselas")
    print(f"  cobertura      : {len(archivos_de(rutas.cobertura, '*.tif'))} teselas")
    print(f"  clima POWER    : {'bien' if rutas.clima_power.exists() else 'no está'}")
    if falta_base:
        print(f"  -> Sin estos datos el modelo no puede correr. Copie el contenido del zip original en {rutas.raw}\n"
              f"     (límites en geo, focos en focos, relieve en terrain, cobertura en landcover, clima en weather).")

    print(f"== Datos que se bajan, en {rutas.external} ==")
    meses = sorted({p.name[9:16] for p in rutas.era5_meses.glob("era5land_*.nc")}) if rutas.era5_meses.exists() else []
    pendientes = actualizacion.meses_pendientes(rutas.era5_meses, date.today(), refrescar=0)
    print(f"  ERA5-Land      : {len(meses)} meses bajados" + (f" ({meses[0]} a {meses[-1]})" if meses else "")
          + (f", {len(pendientes)} por bajar" if pendientes else ""))
    print(f"  tabla ERA5     : {'lista (el modelo la usa)' if rutas.clima_era5.exists() else 'todavía no (el modelo usa NASA POWER)'}")
    for nombre in list(capas_ee.CAPAS) + ["caminos"]:
        ruta = rutas.capas / f"{nombre}.npz"
        if ruta.exists():
            c = capas.cargar(ruta, malla)
            print(f"  capa {nombre:15s}: " + (f"{c.anios[0]}–{c.anios[-1]}" if c.anios else "lista"))
        else:
            print(f"  capa {nombre:15s}: todavía no")
    for fuente in config.FUENTES_FOCOS[1:]:
        bajados = archivos_de(rutas.focos_bajados, fuente["patron"])
        print(f"  focos {fuente['nombre']:9s}: " + (f"{len(bajados)} archivos" if bajados else "todavía no"))
    print("  (lo que dice «todavía no» lo trae: python -m incendios descargar --que mejorados)")
    return 1 if falta_base else 0


# ---------------------------------------------------------------------- principal
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--que", choices=OPCIONES[1:], help="qué descargar")
    ap.add_argument("--diagnostico", action="store_true", help="mostrar claves, paquetes y datos disponibles")
    ap.add_argument("--clave", help="MAP_KEY de NASA FIRMS (si no está en credenciales.json)")
    ap.add_argument("--autenticar", action="store_true", help="iniciar sesión en Earth Engine (la primera vez)")
    ap.add_argument("--paralelo", type=int, default=4, help="pedidos simultáneos al CDS (ERA5-Land)")
    ap.add_argument("--noaa20-historico", action="store_true", help="bajar también NOAA-20 de 2018 a 2025")
    args = ap.parse_args(argv)
    rutas = Rutas.del_proyecto()
    malla = malla_cochabamba()
    cred = leer_credenciales()
    if args.diagnostico or not args.que:
        estado = diagnostico(rutas, malla, cred)
        if not args.que:
            if not args.diagnostico:   # sin opciones: el diagnóstico y cómo bajar
                print("\nPara bajar los datos: python -m incendios descargar --que mejorados "
                      "(la primera vez, agregue --autenticar).\nTodas las opciones: python -m incendios descargar -h")
            return estado
    que = args.que
    mascara = mascara_departamento(rutas, malla)
    if que in ("teselas", "todo"):
        descargar_teselas(rutas, malla, mascara)
    if que in ("clima", "todo"):
        descargar_clima_power(rutas, malla, mascara)
    if que in ("clima_era5", "mejorados", "todo"):
        descargar_clima_era5(rutas, malla, mascara, cred, paralelo=args.paralelo)
    if que in ("capas", "mejorados", "todo"):
        descargar_capas(rutas, malla, cred, autenticar=args.autenticar)
    if que in ("caminos", "mejorados", "todo"):
        descargar_caminos(rutas, malla)
    if que in ("focos", "mejorados", "todo"):
        try:
            descargar_focos(rutas, args.clave or cred["firms"]["map_key"], historico=args.noaa20_historico)
        except Exception as error:  # noqa: BLE001 - la API puede cambiar: que lo demás no se pierda
            print(f"  No se pudieron bajar los focos por la API ({str(error)[:200]}).\n"
                  "  Alternativa manual: https://firms.modaps.eosdis.nasa.gov/download/ (archivo por país y año,\n"
                  "  VIIRS NOAA-20, CSV), guardado en data/external/focos/ como viirs-noaa20_AAAA_Bolivia.csv")
    print("\nListo. Para que el modelo use lo nuevo: python -m incendios entrenar "
          "(o python -m incendios semana, que reentrena solo si hace falta).")


if __name__ == "__main__":
    raise SystemExit(main())
