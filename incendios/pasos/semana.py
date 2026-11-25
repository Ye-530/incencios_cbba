"""Actualización semanal: baja lo nuevo, reentrena si cambiaron las fuentes y predice la semana.

    python -m incendios semana                     lo de cada lunes
    python -m incendios semana --sin-descargar     solo predecir con lo que ya hay
    python -m incendios semana --fecha 2026-11-23  la semana que contiene esa fecha
    python -m incendios semana --no-reentrenar     avisar en vez de reentrenar
    python -m incendios semana --registro          además copia todo a reports/semanal/registro.txt

Pasos
  1. Descargas, según las claves de credenciales.json (lo que falle se avisa y se sigue):
     focos nuevos de NASA FIRMS, clima ERA5-Land de los últimos meses (sin clave de Copernicus,
     clima NASA POWER) y, si salió un año nuevo, capas de Earth Engine. Los caminos se bajan una
     sola vez con `python -m incendios descargar --que caminos`.
  2. Si los datos ya no son los del entrenamiento (por ejemplo, la primera vez que hay ERA5-Land
     o capas), guarda la evaluación anterior en reports/anteriores/AAAA-MM-DD/, reentrena
     (preparar, modelar con ablación y evaluar: puede tardar más de una hora) y escribe
     reports/comparacion.md.
  3. Predice la semana (lunes a domingo): modo operativo si los datos llegan hasta la víspera;
     si no, estacional. Deja en reports/semanal/AAAA-MM-DD/ el GeoTIFF, el PNG, el CSV con las
     100 celdas de mayor riesgo y un resumen.
  4. Regenera el visor (reports/visor_riesgo.html) con el seguimiento semanal del año.

Programarlo cada lunes: python -m incendios programar
"""
import argparse
import contextlib
import json
import shutil
import sys
import time
from datetime import date, datetime

from incendios import actualizacion, flujo, prediccion
from incendios.malla import malla_cochabamba
from incendios.pasos import comparar, descargar as dd, evaluar, modelar, preparar, visor
from incendios.rutas import Rutas


class _Doble:
    """Escribe en la consola y en un archivo a la vez."""

    def __init__(self, consola, archivo):
        self.consola, self.archivo = consola, archivo

    def write(self, texto):
        self.consola.write(texto)
        self.archivo.write(texto)
        return len(texto)

    def flush(self):
        self.consola.flush()
        self.archivo.flush()


@contextlib.contextmanager
def registrar_en(ruta):
    """Copia todo lo que se muestra (y los errores) a un archivo, sin dejar de mostrarlo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "a", encoding="utf-8") as archivo:
        salida, errores = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Doble(salida, archivo), _Doble(errores, archivo)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = salida, errores


def paso(nombre, funcion, argumentos=()):
    print(f"\n===== {nombre} {' '.join(argumentos)}".rstrip(), flush=True)
    funcion(list(argumentos))


def descargar(rutas, cred, hoy):
    malla = malla_cochabamba()
    mascara = dd.mascara_departamento(rutas, malla)
    hechos = {}
    tareas = [("focos", lambda: dd.descargar_focos(rutas, cred["firms"]["map_key"], hoy=hoy), cred["firms"]["map_key"])]
    if cred["copernicus_cds"]["key"]:
        tareas.append(("clima ERA5-Land", lambda: dd.descargar_clima_era5(rutas, malla, mascara, cred, hoy=hoy), True))
    else:   # sin clave de Copernicus, al menos el clima NASA POWER al día (no pide cuenta)
        tareas.append(("clima NASA POWER", lambda: dd.descargar_clima_power(rutas, malla, mascara) or True, True))
    tareas.append(("capas de Earth Engine", lambda: dd.descargar_capas(rutas, malla, cred),
                   cred["earth_engine"]["proyecto"]))
    for nombre, tarea, hay_clave in tareas:
        if not hay_clave:
            print(f"\n{nombre}: sin clave en credenciales.json, no se actualiza.")
            hechos[nombre] = "sin clave"
            continue
        print()
        try:
            hechos[nombre] = "bien" if tarea() else "incompleto"
        except Exception as error:  # noqa: BLE001 - que una fuente caída no frene la predicción
            print(f"  {nombre}: falló ({str(error)[:300]})")
            hechos[nombre] = "falló"
    return hechos


def motivo_para_reentrenar(ctx):
    """None si el modelo final se entrenó con estos mismos datos; si no, el porqué."""
    try:
        modelo = prediccion.cargar_modelo(ctx.rutas.modelos, "operativo")
    except (FileNotFoundError, RuntimeError) as error:
        return str(error)
    entrenado = getattr(modelo, "info", {}).get("huella")
    if not entrenado:
        return "el modelo no dice con qué datos se entrenó"
    cambios = flujo.claves_distintas(entrenado, ctx.notas["huella"])
    if cambios:
        return f"cambiaron {', '.join(cambios)} respecto del entrenamiento"
    return None


def reentrenar(ctx, motivo, hoy):
    rutas = ctx.rutas
    print(f"\nHay que reentrenar: {motivo}.")
    anterior = rutas.anteriores / str(hoy)
    if (rutas.tablas / "evaluacion_resumen.json").exists():
        shutil.copytree(rutas.tablas, anterior / "tablas", dirs_exist_ok=True)
        if rutas.resultados.exists():
            shutil.copy2(rutas.resultados, anterior / rutas.resultados.name)
        print(f"  la evaluación anterior queda en {anterior}")
    t0 = time.time()
    paso("preparar", preparar.main)
    paso("modelar", modelar.main, ["--ablacion"])
    paso("evaluar", evaluar.main)
    print(f"\nReentrenado en {(time.time() - t0) / 60:.0f} min")
    if (anterior / "tablas" / "evaluacion_resumen.json").exists():
        _, ruta = comparar.comparar(anterior, rutas.informes, titulo=f"Antes y después del {hoy}: {motivo}")
        print(f"  comparación con el modelo anterior: {ruta}")


def predecir_semana(ctx, lunes):
    try:
        r = prediccion.predecir(ctx, str(lunes), "operativo")
    except (ValueError, FileNotFoundError) as error:
        print(f"  modo operativo no disponible ({error}); se usa el estacional.")
        r = prediccion.predecir(ctx, str(lunes), "estacional")
    archivos = prediccion.guardar_prediccion(ctx, r, ctx.rutas.semanal / str(lunes))
    info = r["info"]
    print(f"\nRiesgo {r['modo']}, semana del {info['inicio']} al {info['fin']}: "
          f"{r['prob'].sum():.0f} celdas esperadas con fuego de {ctx.n:,}")
    for aviso in r["avisos"]:
        print(f"  AVISO: {aviso}")
    print(prediccion.tabla_top(ctx, r["prob"], 10).to_string(index=False))
    return r, archivos


def regenerar_visor(ctx):
    """El visor necesita las predicciones de los años de prueba que deja `modelar` (no vienen en el zip)."""
    if not (ctx.rutas.predicciones / "prueba" / "y.npy").exists():
        print(f"\nEl visor no se regeneró: faltan las predicciones de prueba en {ctx.rutas.predicciones} "
              "(no vienen en el zip; se crean al entrenar). El mapa de la semana ya quedó guardado.")
        return
    try:
        paso("visor", visor.main)
    except Exception as error:  # noqa: BLE001 - la predicción de la semana ya está guardada
        print(f"\nEl visor no se pudo regenerar: {str(error)[:300]}")


def actualizar(args):
    hoy = date.today()
    lunes = actualizacion.lunes_de(date.fromisoformat(args.fecha) if args.fecha else hoy)
    rutas = Rutas.del_proyecto()
    print(f"Actualización semanal — {datetime.now():%Y-%m-%d %H:%M} — semana del {lunes}")

    hechos = {} if args.sin_descargar else descargar(rutas, dd.leer_credenciales(), hoy)

    ctx = flujo.cargar_todo(rutas)
    motivo = motivo_para_reentrenar(ctx)
    if motivo and args.no_reentrenar:
        print(f"\nAVISO: {motivo}. Reentrene con: python -m incendios entrenar (o quite --no-reentrenar).")
    elif motivo:
        reentrenar(ctx, motivo, hoy)

    r, archivos = predecir_semana(ctx, lunes)
    registro = rutas.semanal / "registro.jsonl"
    registro.parent.mkdir(parents=True, exist_ok=True)
    with open(registro, "a", encoding="utf-8") as f:
        f.write(json.dumps({"cuando": datetime.now().isoformat(timespec="minutes"), "semana": str(lunes),
                            "modo": r["modo"], "descargas": hechos,
                            "reentrenado": bool(motivo and not args.no_reentrenar),
                            "celdas_esperadas": round(float(r["prob"].sum()), 1), "avisos": r["avisos"],
                            "mapa": str(archivos["mapa_png"])}, ensure_ascii=False) + "\n")
    if not args.sin_visor:
        regenerar_visor(ctx)
    print(f"\nListo: {archivos['mapa_png'].parent}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fecha", help="un día de la semana a predecir (AAAA-MM-DD); por defecto, hoy")
    ap.add_argument("--sin-descargar", action="store_true", help="no bajar nada: usar los datos que ya hay")
    ap.add_argument("--no-reentrenar", action="store_true", help="si los datos cambiaron, solo avisar")
    ap.add_argument("--sin-visor", action="store_true", help="no regenerar el visor")
    ap.add_argument("--registro", action="store_true", help="copiar todo a reports/semanal/registro.txt")
    args = ap.parse_args(argv)
    if args.registro:
        with registrar_en(Rutas.del_proyecto().semanal / "registro.txt"):
            print(f"\n######## {datetime.now():%Y-%m-%d %H:%M}")
            actualizar(args)
    else:
        actualizar(args)


if __name__ == "__main__":
    main()
