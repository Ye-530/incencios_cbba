"""CRISP-DM fase 3 — Preparación de los datos.

Construye la malla de 500 m, las variables y la muestra caso-control de 2015 a 2025.

    python -m incendios preparar
"""
import argparse
import sys
import time
from pathlib import Path


from incendios import config, flujo


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    inicio = time.time()
    ctx = flujo.cargar_todo()
    semanas = ctx.semanas_utiles(config.ANIOS_ENTRENA[0], config.ANIOS_PRUEBA[1])
    print(f"Armando la muestra: {len(semanas)} semanas, todos los positivos + "
          f"{config.NEG_POR_SEMANA:,} negativos por semana...")
    muestra = flujo.construir_muestra(ctx, semanas, config.NEG_POR_SEMANA)
    info = flujo.guardar_muestra(muestra, ctx, ctx.rutas.muestra)
    print(f"Muestra: {info['filas']:,} filas ({info['positivos']:,} con fuego), "
          f"{len(info['variables']['operativo'])} variables, tasa de negativos {info['tasa_neg']:.4f}")
    print(f"Guardada en {ctx.rutas.muestra}  ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()
