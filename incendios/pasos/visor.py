"""CRISP-DM fase 6 — Despliegue: visor web autónomo del mapa de riesgo.

Genera reports/visor_riesgo.html (un solo archivo, sin servidor ni internet para los datos):
  · modo operativo, 2024–2025: lo que el modelo habría dicho cada semana, con los focos que hubo;
  · modo estacional, 2024–2025: ídem;
  · modo estacional, año siguiente y subsiguiente al último con datos: calendario de riesgo;
  · modo operativo después de la prueba (seguimiento): lo que el modelo final dijo cada semana
    con los datos de ese momento, y los focos que hubo cuando la semana ya pasó.

    python -m incendios visor
    python -m incendios visor --solo-contenido pagina.html   además, la página sin la cabecera
"""
import argparse
import base64
import io
import json
from datetime import date
from pathlib import Path


import numpy as np
from PIL import Image

from incendios import actualizacion, config, evaluacion as ev, flujo, prediccion
from incendios.tiempo import info_semana, inicio_semana, semanas_de_anio

PLANTILLA = Path(__file__).resolve().parents[1] / "visor_plantilla.html"
MARCA = "/*__DATOS__*/"
CABECERA = ('<!doctype html>\n<html lang="es">\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
N_TOP = 10


def png_b64(arr):
    """Arreglo uint8 2D -> PNG en escala de grises, en base64."""
    memoria = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)).save(memoria, format="PNG", optimize=True)
    return base64.b64encode(memoria.getvalue()).decode("ascii")


def a_rejilla(ctx, valores, relleno=255):
    salida = np.full(ctx.malla.nfilas * ctx.malla.ncols, relleno, dtype=np.uint8)
    salida[ctx.idx] = valores
    return salida.reshape(ctx.malla.forma)


def entrada_semana(ctx, semana, prob, observadas=None, aviso=None):
    info = info_semana(semana)
    clase = ev.clases_de_riesgo(prob, config.UMBRALES_CLASES)
    orden = np.argsort(-prob, kind="stable")[:N_TOP]
    entrada = {
        "ini": str(info["inicio"]), "fin": str(info["fin"]), "anio": info["anio"], "sem": info["semana_anio"],
        "png": png_b64(a_rejilla(ctx, clase)),
        "esp": round(float(prob.sum()), 1),
        "cls": np.bincount(clase, minlength=5).tolist(),
        "top": [[round(float(ctx.lat[i]), 4), round(float(ctx.lon[i]), 4), round(float(prob[i]), 4),
                 int(ctx.municipio_id[i])] for i in orden],
    }
    if observadas is not None:
        celdas = ctx.idx[np.flatnonzero(observadas)]
        entrada["obs"] = np.diff(celdas, prepend=0).tolist()      # codificación por diferencias
        entrada["aciertos"] = np.bincount(clase[np.asarray(observadas, bool)], minlength=5).tolist()
    if aviso:
        entrada["aviso"] = aviso
    return entrada


def contornos(geojson, malla, decimales=1):
    """Polígonos en coordenadas de la rejilla (columna, fila), redondeadas para ahorrar espacio."""
    salida = []
    for entidad in geojson["features"]:
        geom = entidad["geometry"]
        poligonos = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        anillos = []
        for poligono in poligonos:
            for anillo in poligono:
                a = np.asarray(anillo, dtype=float)
                x = np.round((a[:, 0] - malla.oeste) / malla.celda, decimales)
                y = np.round((malla.norte - a[:, 1]) / malla.celda, decimales)
                puntos = np.column_stack([x, y])
                conservar = np.r_[True, np.any(np.diff(puntos, axis=0) != 0, axis=1)]
                anillos.append(puntos[conservar].ravel().tolist())
        salida.append(anillos)
    return salida


def opciones(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo-contenido", type=Path, metavar="RUTA",
                    help="guardar además la página sin la cabecera del documento, para plataformas que ponen la suya")
    return ap.parse_args(argv)


def main(argv=None):
    args = opciones(argv)
    ctx = flujo.cargar_todo()
    geo = ctx.rutas.geo
    municipios_gj = json.loads((geo / "cochabamba_municipios.geojson").read_text(encoding="utf-8"))
    limite_gj = json.loads((geo / "cochabamba_limite.geojson").read_text(encoding="utf-8"))
    est = flujo.cargar_estaticas(ctx.malla, ctx.rutas, verboso=False)
    tablas = ctx.rutas.tablas
    resumen = json.loads((tablas / "evaluacion_resumen.json").read_text(encoding="utf-8"))

    # contexto para orientarse: zonas urbanas y agua (de WorldCover)
    contexto = np.zeros(ctx.malla.forma, dtype=np.uint8)
    contexto[np.nan_to_num(est["lc_urbano"]) >= 0.4] = 1
    contexto[np.nan_to_num(est["lc_agua"]) >= 0.5] = 2

    series = {}
    # 1) años de prueba: predicciones ya calculadas con modelos que no vieron esos años
    if not flujo.predicciones_vigentes(ctx.rutas.predicciones / "prueba", ctx.notas["huella"],
                                       ("gbm_operativo", "gbm_estacional")):
        print("  AVISO: las predicciones de prueba son de otro modelo o de otros datos; para que el visor "
              "muestre el modelo actual en 2024–2025, ejecute antes: python -m incendios modelar")
    pru = ev.cargar_predicciones(ctx.rutas.predicciones / "prueba")
    y = np.asarray(pru["y"])
    for regimen in ("operativo", "estacional"):
        prob = np.asarray(pru["pred"][f"gbm_{regimen}"])
        for k, semana in enumerate(pru["semanas"]):
            anio = info_semana(int(semana))["anio"]
            series.setdefault(f"{regimen}_{anio}", []).append(
                entrada_semana(ctx, int(semana), prob[k], observadas=y[k]))
        print(f"  {regimen}: {len(pru['semanas'])} semanas de prueba")

    # 2) calendario estacional de los años que vienen, con el modelo final
    modelo = prediccion.cargar_modelo(ctx.rutas.modelos, "estacional")
    ultimo = ctx.constructor.anio_ultimo_completo
    futuros = [ultimo + 1, ultimo + 2]
    for anio in futuros:
        for semana in semanas_de_anio(anio):
            r = prediccion.predecir(ctx, str(inicio_semana(semana)), "estacional", modelo=modelo)
            series.setdefault(f"estacional_{anio}", []).append(
                entrada_semana(ctx, semana, r["prob"], aviso=r["avisos"][0] if r["avisos"] else None))
        print(f"  estacional {anio}: calendario de {len(series[f'estacional_{anio}'])} semanas")

    # 3) seguimiento: el modelo operativo final, semana a semana, después de los años de prueba
    seguimiento = []
    try:
        modelo_op = prediccion.cargar_modelo(ctx.rutas.modelos, "operativo")
    except (FileNotFoundError, RuntimeError) as error:
        modelo_op = None
        print(f"  seguimiento: sin modelo operativo ({error})")
    if modelo_op is not None:
        ultima_de_prueba = int(np.max(pru["semanas"]))
        for semana in [s for s in actualizacion.semanas_operativas(ctx, config.ANIOS_PRUEBA[1] + 1)
                       if s > ultima_de_prueba]:
            r = prediccion.predecir(ctx, str(inicio_semana(semana)), "operativo", modelo=modelo_op)
            observadas = ctx.constructor.etiquetas(semana) if actualizacion.semana_completa(ctx, semana) else None
            anio = info_semana(semana)["anio"]
            series.setdefault(f"operativo_{anio}", []).append(entrada_semana(
                ctx, semana, r["prob"], observadas=observadas, aviso=" ".join(r["avisos"]) or None))
            if anio not in seguimiento:
                seguimiento.append(anio)
        if seguimiento:
            print(f"  operativo {seguimiento}: seguimiento de "
                  f"{sum(len(series[f'operativo_{a}']) for a in seguimiento)} semanas")

    m = ctx.malla
    t_pru = resumen["modelos_prueba"]
    datos = {
        "generado": str(date.today()),
        "malla": {"oeste": m.oeste, "norte": m.norte, "celda": m.celda, "ncols": m.ncols, "nfilas": m.nfilas},
        "umbrales": list(config.UMBRALES_CLASES),
        "municipios": ctx.municipios,
        "provincia_de_municipio": ctx.provincia_de_municipio,
        "raster_municipio": png_b64(a_rejilla(ctx, ctx.municipio_id.astype(np.uint8))),
        "raster_contexto": png_b64(contexto),
        "contorno_municipios": contornos(municipios_gj, m),
        "contorno_departamento": contornos(limite_gj, m),
        "series": series,
        "anios_prueba": list(config.ANIOS_PRUEBA),
        "anios_futuros": futuros,
        "anios_seguimiento": seguimiento,
        "ultimo_anio_con_focos": ultimo,
        "ultima_fecha_focos": ctx.notas["ultima_fecha_focos"],
        "fuentes_fijas": int(ctx.fuentes["en_departamento"].sum()),
        "desempeno": {
            nombre: {k: t_pru[nombre][k] for k in ("captura_1", "captura_5", "captura_10", "area_50", "area_80",
                                                   "roc_auc", "pr_auc")}
            for nombre in ("gbm_operativo", "gbm_estacional", "provincia", "recurrencia") if nombre in t_pru},
        "clases_prueba": resumen["clases"],
    }
    html = PLANTILLA.read_text(encoding="utf-8")
    if MARCA not in html:
        raise RuntimeError(f"La plantilla no contiene la marca {MARCA}")
    carga = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    contenido = html.replace(MARCA, carga)
    destino = ctx.rutas.visor
    destino.write_text(CABECERA + contenido, encoding="utf-8")
    print(f"Visor: {destino}  ({destino.stat().st_size / 1e6:.1f} MB, {sum(len(v) for v in series.values())} mapas semanales)")
    if args.solo_contenido:
        args.solo_contenido.write_text(contenido, encoding="utf-8")
        print(f"Contenido sin cabecera: {args.solo_contenido}")


if __name__ == "__main__":
    main()
