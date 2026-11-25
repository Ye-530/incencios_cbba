"""CRISP-DM fase 6 — Despliegue: de una fecha al mapa de riesgo.

    python -m incendios predecir --fecha 2027-09-15                  # modo estacional (cualquier fecha)
    python -m incendios predecir --fecha 2025-10-15 --modo operativo # necesita datos hasta la víspera
    python -m incendios predecir --anio 2027                         # calendario: las 52 semanas del año

Cada predicción deja en reports/mapas/: GeoTIFF de probabilidad y de clase (para QGIS),
PNG, CSV con las 100 celdas de mayor riesgo (latitud, longitud, municipio) y un resumen.
"""
import argparse


import pandas as pd

from incendios import flujo, prediccion
from incendios.tiempo import inicio_semana, semanas_de_anio


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fecha", help="cualquier día de la semana a predecir (AAAA-MM-DD)")
    ap.add_argument("--anio", type=int, help="predecir todas las semanas de ese año (solo modo estacional)")
    ap.add_argument("--modo", choices=("estacional", "operativo"), default="estacional")
    ap.add_argument("--top", type=int, default=100, help="cuántas celdas de mayor riesgo listar")
    args = ap.parse_args(argv)
    if bool(args.fecha) == bool(args.anio):
        ap.error("indique --fecha o --anio (uno de los dos)")
    if args.anio and args.modo != "estacional":
        ap.error("el calendario anual solo existe en modo estacional")

    ctx = flujo.cargar_todo()
    modelo = prediccion.cargar_modelo(ctx.rutas.modelos, args.modo)
    carpeta = ctx.rutas.mapas

    if args.fecha:
        r = prediccion.predecir(ctx, args.fecha, args.modo, modelo=modelo)
        rutas = prediccion.guardar_prediccion(ctx, r, carpeta, n_top=args.top)
        info = r["info"]
        print(f"\nRiesgo {args.modo}, semana del {info['inicio']} al {info['fin']} "
              f"(semana {info['semana_anio']} de {info['anio']})")
        for aviso in r["avisos"]:
            print(f"  AVISO: {aviso}")
        print(f"  celdas esperadas con fuego: {r['prob'].sum():.0f} de {ctx.n:,}")
        print("\nLas 10 celdas de mayor riesgo:")
        print(prediccion.tabla_top(ctx, r["prob"], 10).to_string(index=False))
        print("\nArchivos:")
        for ruta in rutas.values():
            print(f"  {ruta}")
        return

    filas = []
    carpeta = carpeta / f"calendario_{args.anio}"
    for semana in semanas_de_anio(args.anio):
        r = prediccion.predecir(ctx, str(inicio_semana(semana)), "estacional", modelo=modelo)
        prediccion.guardar_prediccion(ctx, r, carpeta, n_top=args.top)
        filas.append({"semana_iso": r["info"]["semana_anio"], "inicio": r["info"]["inicio"],
                      "fin": r["info"]["fin"], "celdas_esperadas_con_fuego": round(float(r["prob"].sum()), 1),
                      "celdas_alto_o_muy_alto": int((r["clase"] >= 3).sum()),
                      "probabilidad_maxima": round(float(r["prob"].max()), 4)})
        avisos = r["avisos"]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta / f"calendario_{args.anio}.csv", index=False, encoding="utf-8-sig")
    for aviso in avisos:
        print(f"AVISO: {aviso}")
    print(tabla.to_string(index=False))
    print(f"\nCalendario en {carpeta}")


if __name__ == "__main__":
    main()
