"""Entrena de punta a punta: preparar → modelar → evaluar → visor.

    python -m incendios entrenar               lo habitual
    python -m incendios entrenar --ablacion    además mide qué aporta cada grupo de variables
    python -m incendios entrenar --busqueda    además compite variantes de variables y parámetros
    python -m incendios entrenar --rehacer     no reutiliza nada de corridas anteriores
"""
import argparse
import time

from incendios.pasos import evaluar, modelar, preparar, visor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablacion", action="store_true", help="medir qué aporta cada grupo de variables")
    ap.add_argument("--busqueda", action="store_true", help="competir variantes de variables y parámetros")
    ap.add_argument("--rehacer", action="store_true", help="no reutilizar modelos ni predicciones anteriores")
    ap.add_argument("--sin-visor", action="store_true", help="no regenerar el visor al final")
    args = ap.parse_args(argv)
    inicio = time.time()
    opciones = [f"--{n}" for n in ("ablacion", "busqueda", "rehacer") if getattr(args, n)]
    for nombre, paso, argumentos in (("preparar", preparar, []), ("modelar", modelar, opciones),
                                     ("evaluar", evaluar, []), ("visor", visor, [])):
        if nombre == "visor" and args.sin_visor:
            continue
        print(f"\n===== {nombre} {' '.join(argumentos)}".rstrip(), flush=True)
        paso.main(argumentos)
    print(f"\nEntrenamiento completo en {(time.time() - inicio) / 60:.0f} min")


if __name__ == "__main__":
    main()
