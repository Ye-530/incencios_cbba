"""CRISP-DM fase 4 — Modelado.

  A. Selección: entrena con 2015–2021 y compara algoritmos sobre 2022–2023 (malla completa).
  B. Prueba: reentrena lo elegido con 2015–2023 y puntúa 2024–2025, que nunca se usó.
  C. Final: reentrena con todo (2015–2025) y guarda los modelos para predecir.
En B y C el GBM es el promedio de un modelo por semilla (config.SEMILLAS_GBM); en A, uno solo, igual
que la ablación, para que sus comparaciones sean parejas.

    python -m incendios modelar              # usa los parámetros ya elegidos si existen
    python -m incendios modelar --busqueda   # además compite variantes (variables y parámetros)
    python -m incendios modelar --ablacion   # además mide qué aporta cada grupo de variables
    python -m incendios modelar --rehacer    # no reutiliza nada de corridas anteriores

Si cambiaron el clima, el relieve, la cobertura o las capas extra desde la última corrida
(lo dice models/huella.json), se reentrena todo aunque no se pida --rehacer.
"""
import argparse
import json
import time


import joblib
import numpy as np

from incendios import clima, config, flujo, modelos, prediccion
from incendios.muestreo import Calibrador
from incendios.variables import REGIMENES

# Del clima antecedente, el conjunto "compacto" conserva seis indicadores de sequía y peligro.
CLIMA_COMPACTO = ("fwi_7", "dias_sin_lluvia", "prec_30", "an_prec_90", "an_tmax_7", "dc")
CLIMA_RESTO = [v for v in clima.VARS_ANTECEDENTE if v not in CLIMA_COMPACTO]

# Variantes que compite la búsqueda (régimen operativo). Todas llevan restricciones de
# monotonía en el clima salvo la que dice lo contrario.
CANDIDATOS = {
    "clima_compacto": {"excluir": CLIMA_RESTO},
    "clima_completo": {},
    "clima_completo_sin_monotonia": {"monotono": False},
    "sin_clima": {"excluir": list(clima.VARS_ANTECEDENTE)},
    "compacto_hojas_31": {"excluir": CLIMA_RESTO, "max_leaf_nodes": 31},
    "compacto_lento": {"excluir": CLIMA_RESTO, "learning_rate": 0.03},
    "compacto_regularizado": {"excluir": CLIMA_RESTO, "min_samples_leaf": 500, "l2_regularization": 5.0},
}
PREDETERMINADO = "clima_compacto"
FRACCION_BUSQUEDA = 0.5      # la búsqueda usa la mitad de las filas de entrenamiento


def reloj(t0):
    return f"{time.time() - t0:.0f} s"


def en_anios(muestra, rango):
    return (muestra["anio"] >= rango[0]) & (muestra["anio"] <= rango[1])


def puntuador_recurrencia(ctx):
    """Línea base sin aprendizaje: ¿ardió aquí en esta época en los últimos años?"""
    nombres = ctx.constructor.nombres("operativo")
    i_celda, i_vecinas, i_zona = (nombres.index(n) for n in ("est_c3", "est_v3_3", "est_v11_dec"))

    def puntaje(ctx, semana, X):
        return (10000.0 * X[:, i_celda] + 100.0 * X[:, i_vecinas] + X[:, i_zona]).astype(np.float32)
    return puntaje


def separar(candidato):
    """Parte un candidato en (hiperparámetros del GBM, variables a excluir)."""
    parametros = {k: v for k, v in candidato.items() if k != "excluir"}
    return parametros, list(candidato.get("excluir", []))


def elegir_parametros(ctx, muestra, entrena, valida, carpeta, buscar, rehacer):
    """Variante y número de árboles del GBM, elegidos con validación posterior en el tiempo."""
    ruta = carpeta / "seleccion.json"
    if ruta.exists() and not rehacer and not buscar:
        return json.loads(ruta.read_text(encoding="utf-8"))
    seleccion = {"variante": PREDETERMINADO, "busqueda": []}
    if buscar:
        rng = np.random.default_rng(config.SEMILLA)
        parte = entrena & (rng.random(len(entrena)) < FRACCION_BUSQUEDA)
        print(f"Búsqueda de variantes (régimen operativo, {parte.sum():,} filas de entrenamiento):")
        for nombre, candidato in CANDIDATOS.items():
            t0 = time.time()
            parametros, excluir = separar(candidato)
            n_iter, historial = modelos.iteraciones_con_parada(
                "operativo", muestra, ctx, parte, valida, verboso=False, excluir=excluir, **parametros)
            mejor = min(historial, key=lambda h: h["perdida"])
            seleccion["busqueda"].append({"variante": nombre, "n_iter": n_iter, "perdida": mejor["perdida"],
                                          "pr_auc_muestra": mejor["pr_auc_muestra"],
                                          "segundos": round(time.time() - t0)})
            print(f"  {nombre:30s} pérdida {mejor['perdida']:.5f}  PR-AUC(muestra) {mejor['pr_auc_muestra']:.4f}"
                  f"  árboles {n_iter:4d}  ({reloj(t0)})", flush=True)
        seleccion["variante"] = min(seleccion["busqueda"], key=lambda b: b["perdida"])["variante"]
        print(f"  elegida: {seleccion['variante']}")
    seleccion["parametros"], seleccion["excluir"] = separar(CANDIDATOS[seleccion["variante"]])
    for regimen in REGIMENES:
        t0 = time.time()
        print(f"Número de árboles, régimen {regimen}:")
        n_iter, historial = modelos.iteraciones_con_parada(
            regimen, muestra, ctx, entrena, valida,
            excluir=excluir_en(ctx, regimen, seleccion["excluir"]), **seleccion["parametros"])
        seleccion[regimen] = {"n_iter": n_iter, "historial": historial}
        print(f"  -> {n_iter} árboles ({reloj(t0)})", flush=True)
    ruta.write_text(json.dumps(seleccion, indent=1), encoding="utf-8")
    return seleccion


def modelos_finales_desactualizados(ctx):
    """¿Algún modelo final se entrenó con otras variables que las que arma el código ahora?"""
    for regimen in REGIMENES:
        ruta = ctx.rutas.modelos / f"final_{regimen}.joblib"
        if not ruta.exists():
            continue
        try:
            modelo = joblib.load(ruta)
        except Exception:  # noqa: BLE001 - un modelo que no abre también hay que rehacerlo
            return True
        if modelos.variables_cambiaron(modelo, ctx):
            return True
    return False


def excluir_en(ctx, regimen, excluir):
    """De la lista a excluir, solo las variables que existen en ese régimen."""
    presentes = set(ctx.constructor.nombres(regimen))
    return [v for v in excluir if v in presentes]


def entrenar_o_cargar(ruta, rehacer, fabrica):
    if ruta.exists() and not rehacer:
        return joblib.load(ruta)
    t0 = time.time()
    modelo = fabrica()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta, compress=3)
    print(f"  {ruta.stem}: entrenado en {reloj(t0)}")
    return modelo


def fase(ctx, muestra, filas, anios_eval, algoritmos, seleccion, nombre, rehacer, semillas=None):
    """Entrena con `filas`, puntúa toda la malla en `anios_eval` y devuelve los modelos.

    `semillas`: las del GBM (varias = promedio de un modelo por semilla); los otros algoritmos usan una.
    """
    dir_modelos = ctx.rutas.modelos / nombre
    dir_pred = ctx.rutas.predicciones / nombre
    print(f"\n== Fase {nombre}: {int(filas.sum()):,} filas de entrenamiento "
          f"({int(muestra['y'][filas].sum()):,} con fuego) ==")
    entrenados = {}
    for regimen in REGIMENES:
        for algoritmo in algoritmos:
            extra = {"excluir": excluir_en(ctx, regimen, seleccion["excluir"])}
            if algoritmo == "gbm":
                extra.update(seleccion["parametros"], n_iter=seleccion[regimen]["n_iter"], semillas=semillas)
            entrenados[f"{algoritmo}_{regimen}"] = entrenar_o_cargar(
                dir_modelos / f"{algoritmo}_{regimen}.joblib", rehacer,
                lambda a=algoritmo, r=regimen, e=extra: modelos.entrenar(a, r, muestra, ctx, filas=filas, **e))
    provincia = entrenar_o_cargar(dir_modelos / "provincia.joblib", rehacer,
                                  lambda: modelos.entrenar_provincia(muestra, ctx, filas=filas))
    semanas = ctx.semanas_utiles(*anios_eval)
    if not rehacer and flujo.predicciones_vigentes(dir_pred, ctx.notas["huella"],
                                                    (*entrenados, "provincia", "recurrencia")):
        print(f"  predicciones de {nombre} ya calculadas")
        return entrenados, provincia
    puntuadores = {n: (lambda c, s, X, m=m: m.puntaje(X)) for n, m in entrenados.items()}
    puntuadores["provincia"] = lambda c, s, X: provincia.puntaje_semana(c, s)
    puntuadores["recurrencia"] = puntuador_recurrencia(ctx)
    t0 = time.time()
    print(f"  puntuando la malla completa en {len(semanas)} semanas de {anios_eval[0]}–{anios_eval[1]}...")
    flujo.puntuar_malla(ctx, semanas, puntuadores, dir_pred)
    flujo.anotar_huella(dir_pred, ctx.notas["huella"])
    print(f"  listo ({reloj(t0)})")
    return entrenados, provincia


def ablacion(ctx, muestra, filas, seleccion, rehacer):
    """Reentrena el modelo operativo quitando un grupo de variables cada vez (sobre validación)."""
    dir_modelos = ctx.rutas.modelos / "ablacion"
    dir_pred = ctx.rutas.predicciones / "ablacion"
    grupos = ctx.constructor.grupos("operativo")
    ya_fuera = set(seleccion["excluir"])
    print("\n== Ablación: el modelo operativo sin cada grupo de variables ==")
    entrenados = {}
    for grupo in grupos:                     # también las capas extra, si las hay
        excluir = sorted(ya_fuera | set(grupos[grupo]))
        entrenados[f"sin_{grupo}"] = entrenar_o_cargar(
            dir_modelos / f"sin_{grupo}.joblib", rehacer,
            lambda e=excluir: modelos.entrenar("gbm", "operativo", muestra, ctx, filas=filas, excluir=e,
                                               n_iter=seleccion["operativo"]["n_iter"], **seleccion["parametros"]))
    if not rehacer and flujo.predicciones_vigentes(dir_pred, ctx.notas["huella"], entrenados):
        print("  predicciones de la ablación ya calculadas")
        return
    semanas = ctx.semanas_utiles(*config.ANIOS_VALIDA)
    flujo.puntuar_malla(ctx, semanas, {n: (lambda c, s, X, m=m: m.puntaje(X)) for n, m in entrenados.items()},
                        dir_pred)
    flujo.anotar_huella(dir_pred, ctx.notas["huella"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--busqueda", action="store_true", help="probar variantes del GBM (variables y parámetros)")
    ap.add_argument("--ablacion", action="store_true", help="medir qué aporta cada grupo de variables")
    ap.add_argument("--rehacer", action="store_true", help="no reutilizar modelos ni predicciones previas")
    args = ap.parse_args(argv)

    inicio = time.time()
    ctx = flujo.cargar_todo()
    muestra, _ = flujo.cargar_muestra(ctx.rutas.muestra, ctx)
    entrena, valida, prueba = (en_anios(muestra, r) for r in
                               (config.ANIOS_ENTRENA, config.ANIOS_VALIDA, config.ANIOS_PRUEBA))
    dir_modelos = ctx.rutas.modelos
    dir_modelos.mkdir(parents=True, exist_ok=True)
    rehacer = args.rehacer
    if not rehacer and any(dir_modelos.rglob("*.joblib")) and (
            flujo.datos_cambiaron(dir_modelos, ctx.notas["huella"]) or modelos_finales_desactualizados(ctx)):
        print("Los datos o las variables cambiaron desde la última corrida (clima, relieve, cobertura, capas "
              "extra, satélites o código de las variables): se reentrena todo.")
        rehacer = True

    seleccion = elegir_parametros(ctx, muestra, entrena, valida, dir_modelos, args.busqueda, rehacer)

    # A. comparar algoritmos en validación
    fase(ctx, muestra, entrena, config.ANIOS_VALIDA, modelos.NOMBRES, seleccion, "validacion", rehacer)
    if args.ablacion:
        ablacion(ctx, muestra, entrena, seleccion, rehacer)
    # B. el algoritmo elegido (GBM, promedio de config.SEMILLAS_GBM) y las líneas base, sobre los años de prueba
    fase(ctx, muestra, entrena | valida, config.ANIOS_PRUEBA, ("gbm",), seleccion, "prueba", rehacer,
         semillas=config.SEMILLAS_GBM)

    # C. modelos finales con todos los años. Su probabilidad es la del modelo corregida por el
    # muestreo, sin recalibrar: la recalibración isotónica ajustada con los dos años de validación
    # copiaba el nivel de esos años y empeoró en la prueba (`evaluar` mide las dos opciones).
    print("\n== Fase final: modelos para predecir ==")
    todo = entrena | valida | prueba
    carpeta_val = ctx.rutas.predicciones / "validacion"
    for regimen in REGIMENES:
        joblib.dump(Calibrador().ajustar(np.load(carpeta_val / f"pred_gbm_{regimen}.npy").ravel(),
                                         np.load(carpeta_val / "y.npy").ravel()),
                    dir_modelos / f"calibrador_validacion_{regimen}.joblib")

        def fabrica(r=regimen):
            m = modelos.entrenar("gbm", r, muestra, ctx, filas=todo, n_iter=seleccion[r]["n_iter"],
                                 excluir=excluir_en(ctx, r, seleccion["excluir"]), semillas=config.SEMILLAS_GBM,
                                 **seleccion["parametros"])
            m.info["anios"] = [config.ANIOS_ENTRENA[0], config.ANIOS_PRUEBA[1]]
            return m
        entrenar_o_cargar(dir_modelos / f"final_{regimen}.joblib", rehacer, fabrica)
    prediccion.guardar_versiones(ctx.rutas.modelos)
    flujo.anotar_huella(dir_modelos, ctx.notas["huella"])
    print(f"\nModelos en {dir_modelos}  (total {reloj(inicio)})")


if __name__ == "__main__":
    main()
