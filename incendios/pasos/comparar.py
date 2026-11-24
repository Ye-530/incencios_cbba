"""Compara dos evaluaciones (de `evaluar`): antes y después de cambiar los datos o el modelo.

    python -m incendios comparar --antes reports/anteriores/2026-11-23
    python -m incendios comparar --antes corrida_base/reports --despues reports

Cada carpeta es una carpeta de informes (reports), con tablas/evaluacion_resumen.json. Escribe la
tabla en <despues>/comparacion.md y comparacion.csv. Para comparar a mano con y sin las mejoras:
    INCENDIOS_CLIMA=power INCENDIOS_CAPAS=0 INCENDIOS_SALIDAS=corrida_base  python -m incendios entrenar
    y luego, sin esas variables, python -m incendios entrenar.
"""
import argparse
import json
import sys
from pathlib import Path


from incendios import actualizacion, config
from incendios.rutas import Rutas

NOMBRES = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "captura_1": "Captura 1 %", "captura_5": "Captura 5 %",
           "captura_10": "Captura 10 %", "area_50": "Área para 50 %", "area_80": "Área para 80 %"}
MODELOS = {"gbm_operativo": "Operativo", "gbm_estacional": "Estacional", "provincia": "Por provincia",
           "recurrencia": "Recurrencia"}


def resumen(carpeta):
    ruta = Path(carpeta) / "tablas" / "evaluacion_resumen.json"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}: esa carpeta no tiene una evaluación de `python -m incendios evaluar`")
    return json.loads(ruta.read_text(encoding="utf-8"))


def escribir(tabla, carpeta, titulo):
    carpeta = Path(carpeta)
    tabla.to_csv(carpeta / "comparacion.csv", index=False, encoding="utf-8-sig")
    lineas = [f"# {titulo}", "",
              "Mismos años de prueba y validación; cambia lo que se indica. Las capturas y los AUC mejoran si "
              "suben; el área necesaria, si baja.", ""]
    for periodo in ("prueba", "validacion"):
        parte = tabla[(tabla["periodo"] == periodo) & tabla["modelo"].isin(["gbm_operativo", "gbm_estacional"])]
        if parte.empty:
            continue
        anios = config.ANIOS_PRUEBA if periodo == "prueba" else config.ANIOS_VALIDA
        lineas += [f"## {'Prueba' if periodo == 'prueba' else 'Validación'} {anios[0]}–{anios[1]}", "",
                   "| Modelo | Métrica | Antes | Después | Cambio | Mejor |", "|---|---|---:|---:|---:|---|"]
        for f in parte.itertuples():
            lineas.append(f"| {MODELOS.get(f.modelo, f.modelo)} | {NOMBRES.get(f.metrica, f.metrica)} | "
                          f"{f.antes:.4f} | {f.despues:.4f} | {f.cambio:+.4f} | {f.mejor} |")
        lineas.append("")
    (carpeta / "comparacion.md").write_text("\n".join(lineas), encoding="utf-8")
    return carpeta / "comparacion.md"


def comparar(antes, despues, titulo="Antes y después"):
    tabla = actualizacion.comparar_evaluaciones(resumen(antes), resumen(despues))
    return tabla, escribir(tabla, despues, titulo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--antes", required=True, help="carpeta de informes de la corrida anterior")
    ap.add_argument("--despues", default=str(Rutas.del_proyecto().informes), help="carpeta de informes nueva")
    args = ap.parse_args(argv)
    tabla, ruta = comparar(args.antes, args.despues)
    principal = tabla[(tabla["periodo"] == "prueba") & tabla["modelo"].isin(["gbm_operativo", "gbm_estacional"])]
    print(principal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nTabla completa en {ruta}")


if __name__ == "__main__":
    main()
