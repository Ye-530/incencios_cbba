"""CRISP-DM fase 5 — Evaluación.

Mide los modelos sobre la malla completa (todas las celdas, prior real):
validación 2022–2023 (comparar algoritmos) y prueba 2024–2025 (resultado final).

    python -m incendios evaluar
"""
import argparse
import json
import re
import sys
from pathlib import Path


import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as efectos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from incendios import config, evaluacion as ev, flujo, graficos as g, metricas
from incendios.tiempo import info_semana, inicio_semana, semana_de_fecha

ETIQUETAS = {
    "gbm_operativo": "Operativo (500 m)",
    "gbm_estacional": "Estacional (500 m)",
    "recurrencia": "Recurrencia histórica",
    "provincia": "Por provincia (enfoque anterior)",
    "bosque_operativo": "Bosque aleatorio, operativo",
    "bosque_estacional": "Bosque aleatorio, estacional",
    "logistica_operativo": "Regresión logística, operativo",
    "logistica_estacional": "Regresión logística, estacional",
}
COLOR = {"gbm_operativo": g.AZUL, "gbm_estacional": g.NARANJA, "recurrencia": g.AGUA, "provincia": g.AMARILLO}
PRINCIPALES = ["gbm_operativo", "gbm_estacional", "recurrencia", "provincia"]
BORDES_CALIBRACION = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]


def pct(x, d=1):
    return f"{100 * x:.{d}f} %".replace(".", ",")


# --------------------------------------------------------------------------- tablas
def tablas_de_modelos(val, pru, tablas):
    t_val = ev.tabla_modelos(val)
    t_pru = ev.tabla_modelos(pru)
    for t, nombre in ((t_val, "modelos_validacion"), (t_pru, "modelos_prueba")):
        t.insert(0, "descripcion", [ETIQUETAS.get(n, n) for n in t.index])
        t.to_csv(tablas / f"{nombre}.csv")
    return t_val, t_pru


def probabilidades(pru):
    """Probabilidad por celda en los años de prueba: la del modelo, corregida por el muestreo."""
    return {"operativo": np.asarray(pru["pred"]["gbm_operativo"]),
            "estacional": np.asarray(pru["pred"]["gbm_estacional"]),
            "provincia": np.asarray(pru["pred"]["provincia"])}


USADA = "Del modelo, corregida por el muestreo (la que se usa)"
DESCARTADA = f"Recalibrada con isotónica sobre {config.ANIOS_VALIDA[0]}–{config.ANIOS_VALIDA[1]} (descartada)"


def comparar_calibracion(ctx, pru, val, tablas):
    """Nivel de las probabilidades frente a lo observado, por conjunto de años, y qué habría
    pasado en la prueba al recalibrarlas con los dos años de validación."""
    filas = []
    for conjunto, p in (("validación", val), ("prueba", pru)):
        y = np.asarray(p["y"])
        for regimen in ("operativo", "estacional"):
            crudo = np.asarray(p["pred"][f"gbm_{regimen}"])
            filas.append({"conjunto": conjunto, "modo": regimen, "probabilidad": USADA,
                          **ev.resumen_calibracion(crudo, y)})
            ruta = ctx.rutas.modelos / f"calibrador_validacion_{regimen}.joblib"
            if conjunto == "prueba" and ruta.exists():
                filas.append({"conjunto": conjunto, "modo": regimen, "probabilidad": DESCARTADA,
                              **ev.resumen_calibracion(joblib.load(ruta).aplicar(crudo), y)})
        filas.append({"conjunto": conjunto, "modo": "referencia", "probabilidad": "Tasa base constante",
                      **ev.resumen_calibracion(np.full(y.shape, y.mean()), y)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(tablas / "calibracion_nivel.csv", index=False)
    return tabla


def desgloses(ctx, pru, tablas):
    """Resultado de prueba por zona, temporada, año y disponibilidad de relieve."""
    meses = np.array([(inicio_semana(s).month) for s in pru["semanas"]])
    anios = np.array([info_semana(s)["anio"] for s in pru["semanas"]])
    est = flujo.cargar_estaticas(ctx.malla, ctx.rutas, verboso=False)
    con_relieve = np.isfinite(est["elev_media"].ravel()[ctx.idx]).astype(int)
    partes = []
    for modelo in ("gbm_operativo", "gbm_estacional", "provincia", "recurrencia"):
        solo = {**pru, "pred": {modelo: pru["pred"][modelo]}}
        t = ev.tabla_por_grupo(solo, modelo, ctx.zona_id, ctx.zonas)
        t.insert(0, "desglose", "zona")
        partes.append(t.assign(modelo=modelo))
        t = ev.tabla_por_grupo(solo, modelo, con_relieve, ["Sin relieve (falta tesela)", "Con relieve"])
        t.insert(0, "desglose", "relieve")
        partes.append(t.assign(modelo=modelo))
        for etiqueta, mascara in (("Temporada (jul–nov)", (meses >= 7) & (meses <= 11)),
                                  ("Resto del año", (meses < 7) | (meses > 11)),
                                  *((str(a), anios == a) for a in np.unique(anios))):
            sub = ev.filtrar(solo, semanas=mascara)
            if sub["y"].sum() == 0:
                continue
            fila = pd.DataFrame({etiqueta: ev.metricas_modelo(sub, modelo)}).T
            fila.index.name = "grupo"
            fila.insert(0, "desglose", "temporada" if "año" in etiqueta or "Temporada" in etiqueta else "anio")
            partes.append(fila.assign(modelo=modelo))
    tabla = pd.concat(partes).reset_index().set_index(["modelo", "desglose", "grupo"])
    tabla.to_csv(tablas / "desgloses_prueba.csv")
    return tabla


def importancias(ctx, tablas):
    """Importancia por permutación en la muestra de prueba, con los modelos entrenados hasta 2023."""
    muestra, info = flujo.cargar_muestra(ctx.rutas.muestra, ctx)
    prueba = (muestra["anio"] >= config.ANIOS_PRUEBA[0]) & (muestra["anio"] <= config.ANIOS_PRUEBA[1])
    X, y = muestra["X"][prueba], muestra["y"][prueba]
    salida = {}
    for regimen in ("operativo", "estacional"):
        modelo = joblib.load(ctx.rutas.modelos / "prueba" / f"gbm_{regimen}.joblib")
        nombres = info["variables"]["operativo"]
        grupos = {gr: [nombres.index(v) for v in variables] for gr, variables in info["grupos"][regimen].items()}
        por_grupo = ev.importancia_permutacion(modelo.puntaje, X, y, grupos, repeticiones=3)
        por_variable = ev.importancia_permutacion(
            modelo.puntaje, X, y, {v: [nombres.index(v)] for v in modelo.variables}, repeticiones=1)
        por_grupo.to_csv(tablas / f"importancia_grupos_{regimen}.csv", index=False)
        por_variable.rename(columns={"grupo": "variable"}).to_csv(
            tablas / f"importancia_variables_{regimen}.csv", index=False)
        salida[regimen] = (por_grupo, por_variable)
    return salida


def tabla_ablacion(ctx, val, tablas):
    """Qué se pierde en validación al quitar cada grupo de variables del modelo operativo."""
    carpeta = ctx.rutas.predicciones / "ablacion"
    if not (carpeta / "y.npy").exists():
        return None
    abl = ev.cargar_predicciones(carpeta)
    tabla = ev.tabla_modelos(abl)
    completo = ev.metricas_modelo({**val, "pred": {"gbm_operativo": val["pred"]["gbm_operativo"]}}, "gbm_operativo")
    tabla.loc["completo"] = completo
    for columna in ("pr_auc", "captura_5", "captura_10", "roc_auc"):
        tabla[f"cambio_{columna}"] = tabla[columna] - completo[columna]
    tabla = tabla.sort_values("cambio_pr_auc")
    tabla.to_csv(tablas / "ablacion_validacion.csv")
    return tabla


COLUMNAS_MD = {"roc_auc": "dec3", "pr_auc": "dec3", "captura_1": "pct1", "captura_5": "pct1", "captura_10": "pct1",
               "area_50": "pct1", "area_80": "pct1"}
TITULOS_MD = {"descripcion": "Modelo", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "captura_1": "Captura 1 %",
              "captura_5": "Captura 5 %", "captura_10": "Captura 10 %", "area_50": "Área para 50 %",
              "area_80": "Área para 80 %", "grupo": "Grupo", "tasa_base": "Tasa base", "n_pos": "Celdas con fuego",
              "precision_1": "Precisión top 1 %", "precision_1_tol": "Precisión top 1 % (±1 celda)",
              "captura_nuevos_5": "Focos nuevos, captura 5 %", "captura_nuevos_10": "Focos nuevos, captura 10 %"}


def escribir_resultados(ctx, t_val, t_pru, clases, desg, imp, ablacion_t, series, nivel, ruta):
    """Resumen legible de todos los resultados, generado solo a partir de las tablas."""
    def tabla(t, columnas, formatos):
        return ev.tabla_md(t[columnas].reset_index(drop=True), formatos, TITULOS_MD)

    seleccion = json.loads((ctx.rutas.modelos / "seleccion.json").read_text(encoding="utf-8"))
    cols = ["descripcion", *COLUMNAS_MD]
    partes = [
        "# Resultados", "",
        "Generado por `python -m incendios evaluar`. Todas las métricas se calculan sobre la malla completa "
        f"({ev.formatear(ctx.n, 'entero')} celdas por semana, prior real).", "",
        "- **Captura X %**: de las celdas que tuvieron fuego, qué parte cayó en el X % de celdas de mayor riesgo de su semana.",
        "- **Área para 50 % / 80 %**: qué parte del territorio hay que señalar cada semana para capturar esa parte del fuego.",
        "- **PR-AUC y ROC-AUC**: ordenan todas las celdas-semana a la vez (miden también el acierto en el tiempo).", "",
        f"## Validación {config.ANIOS_VALIDA[0]}–{config.ANIOS_VALIDA[1]}: comparación de algoritmos", "",
        tabla(t_val, cols, COLUMNAS_MD), "",
        f"## Prueba {config.ANIOS_PRUEBA[0]}–{config.ANIOS_PRUEBA[1]}: resultado final", "",
        tabla(t_pru, cols, COLUMNAS_MD), "",
        "### Precisión y focos nuevos (prueba)", "",
        tabla(t_pru, ["descripcion", "precision_1", "precision_1_tol", "captura_nuevos_5", "captura_nuevos_10"],
              {"precision_1": "pct2", "precision_1_tol": "pct2", "captura_nuevos_5": "pct1", "captura_nuevos_10": "pct1"}), "",
        "Foco nuevo: celda con fuego sin ningún foco a menos de ~2,75 km en las cuatro semanas anteriores.", "",
    ]
    for regimen, t in clases.items():
        partes += [f"### Clases de riesgo, modo {regimen} (prueba)", "",
                   ev.tabla_md(t, {"umbral_inferior": "pct2", "pct_celdas": "dec3", "pct_focos": "dec1",
                                   "frecuencia_observada": "pct3"},
                               {"clase": "Clase", "umbral_inferior": "Probabilidad desde",
                                "pct_celdas": "% de celdas-semana", "pct_focos": "% del fuego",
                                "frecuencia_observada": "Frecuencia observada"}), ""]
    partes += ["## Nivel de las probabilidades", "",
               "Razón entre las celdas con fuego esperadas (suma de probabilidades) y las observadas: 1 es el "
               "nivel exacto, más de 1 es sobrestimar. Menor pérdida logarítmica es mejor.", "",
               ev.tabla_md(nivel, {"observadas": "entero", "esperadas": "entero", "razon": "dec2",
                                   "perdida_log": "dec5"},
                           {"conjunto": "Años", "modo": "Modo", "probabilidad": "Probabilidad",
                            "observadas": "Observadas", "esperadas": "Esperadas", "razon": "Esperadas / observadas",
                            "perdida_log": "Pérdida logarítmica"}), ""]
    partes += ["## Prueba por zona, temporada y año", ""]
    for modelo in ("gbm_operativo", "gbm_estacional"):
        t = desg.loc[modelo].reset_index()
        partes += [f"### {ETIQUETAS[modelo]}", "",
                   ev.tabla_md(t[["grupo", "n_pos", "tasa_base", "roc_auc", "captura_5", "captura_10", "area_50"]],
                               {"n_pos": "entero", "tasa_base": "pct3", "roc_auc": "dec3", "captura_5": "pct1",
                                "captura_10": "pct1", "area_50": "pct1"}, TITULOS_MD), ""]
    if ablacion_t is not None:
        t = ablacion_t.reset_index().rename(columns={"modelo": "variante"})
        partes += [f"## Ablación (validación): el modelo operativo sin cada grupo de variables", "",
                   ev.tabla_md(t[["variante", "pr_auc", "cambio_pr_auc", "captura_5", "cambio_captura_5", "captura_10"]],
                               {"pr_auc": "dec3", "cambio_pr_auc": "dec3", "captura_5": "pct1",
                                "cambio_captura_5": "pct1", "captura_10": "pct1"},
                               {"variante": "Variante", "pr_auc": "PR-AUC", "cambio_pr_auc": "Cambio PR-AUC",
                                "captura_5": "Captura 5 %", "cambio_captura_5": "Cambio captura 5 %",
                                "captura_10": "Captura 10 %"}), ""]
    partes += ["## Importancia por permutación (muestra de prueba)", ""]
    for regimen in imp:
        t = imp[regimen][0].assign(grupo=lambda d: [NOMBRE_GRUPO.get(n, n) for n in d["grupo"]])
        partes += [f"### Modo {regimen}", "",
                   ev.tabla_md(t[["grupo", "caida_pr_auc"]], {"caida_pr_auc": "dec3"},
                               {"grupo": "Grupo de variables", "caida_pr_auc": "Caída del PR-AUC al barajarlo"}), ""]
    if seleccion.get("busqueda"):
        t = pd.DataFrame(seleccion["busqueda"])
        partes += ["## Búsqueda de variantes del GBM (validación, muestra)", "",
                   ev.tabla_md(t[["variante", "perdida", "pr_auc_muestra", "n_iter"]],
                               {"perdida": "dec5", "pr_auc_muestra": "dec3", "n_iter": "entero"},
                               {"variante": "Variante", "perdida": "Pérdida logarítmica", "pr_auc_muestra": "PR-AUC (muestra)",
                                "n_iter": "Árboles"}), "",
                   f"Elegida: **{seleccion['variante']}** (menor pérdida). Árboles finales: operativo "
                   f"{seleccion['operativo']['n_iter']}, estacional {seleccion['estacional']['n_iter']}.", ""]
    correlacion = {r: float(np.corrcoef(series[r]["observadas"], series[r]["esperadas"])[0, 1]) for r in series}
    partes += ["## Acierto en el tiempo (prueba)", "",
               "Correlación semanal entre celdas con fuego observadas y esperadas: "
               + "; ".join(f"modo {r} {ev.formatear(c, 'dec2')}" for r, c in correlacion.items()) + ".", ""]
    ruta.write_text("\n".join(partes), encoding="utf-8")


# --------------------------------------------------------------------------- figuras
def figura_captura(pru, ruta):
    y = np.asarray(pru["y"]).ravel()
    grupos = np.repeat(np.arange(pru["y"].shape[0]), pru["y"].shape[1])
    fracciones = np.linspace(0, 0.30, 121)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(100 * fracciones, 100 * fracciones, color=g.GRIS_CONTEXTO, linewidth=1.2)
    ax.text(29.6, 27.0, "Al azar", color=g.TENUE, fontsize=8.5, ha="right", va="top")
    finales = []
    for nombre in PRINCIPALES:
        curva = metricas.curva_captura(y, np.asarray(pru["pred"][nombre]).ravel(), grupos, fracciones)
        ax.plot(100 * fracciones, 100 * curva, color=COLOR[nombre], label=ETIQUETAS[nombre])
        ax.plot([30], [100 * curva[-1]], marker="o", markersize=6, color=COLOR[nombre],
                markeredgecolor=g.SUPERFICIE, markeredgewidth=1.5, clip_on=False, zorder=5)
        finales.append((100 * curva[-1], nombre))
    # rótulos al final de cada línea, separados lo justo para no pisarse
    finales.sort()
    posiciones, minimo = [], 5.0
    for valor, _ in finales:
        posiciones.append(valor if not posiciones else max(valor, posiciones[-1] + minimo))
    for (valor, nombre), pos in zip(finales, posiciones):
        ax.annotate(f"{ETIQUETAS[nombre]}  {valor:.0f} %", xy=(30, valor), xytext=(31.2, pos),
                    fontsize=8.5, color=g.TINTA, va="center",
                    arrowprops={"arrowstyle": "-", "color": g.EJE, "linewidth": 0.7} if abs(pos - valor) > 1 else None)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Territorio señalado cada semana (% de las celdas, de mayor a menor riesgo)")
    ax.set_ylabel("Celdas con fuego capturadas (%)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    g.titulos(ax, "Con el mismo territorio señalado, el modelo de 500 m captura mucho más fuego",
              "Prueba 2024–2025, malla completa; el ranking se hace dentro de cada semana")
    fig.subplots_adjust(right=0.70)
    return g.guardar(fig, ruta)


def figura_area(t_pru, ruta):
    modelos = [m for m in PRINCIPALES if m in t_pru.index]
    fig, ejes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    for ax, columna, objetivo in zip(ejes, ("area_50", "area_80"), ("la mitad", "el 80 %")):
        valores = [100 * t_pru.loc[m, columna] for m in modelos]
        barras = ax.barh(range(len(modelos)), valores, height=0.45, color=g.AZUL)
        for barra, valor in zip(barras, valores):
            ax.text(valor + 1.0, barra.get_y() + barra.get_height() / 2, f"{valor:.1f} %".replace(".", ","),
                    va="center", fontsize=8.5, color=g.TINTA)
        ax.set_yticks(range(len(modelos)), [ETIQUETAS[m] for m in modelos])
        ax.invert_yaxis()
        ax.set_xlim(0, max(valores) * 1.25)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"Territorio a señalar para capturar {objetivo} del fuego (%)")
    g.titulos(ejes[0], "Cuánto territorio hay que marcar en rojo para no perder el fuego",
              "Prueba 2024–2025; menos es mejor")
    return g.guardar(fig, ruta)


def figura_serie(series, ruta):
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    base = series["operativo"]
    fechas = pd.to_datetime(base["inicio"])
    ax.plot(fechas, base["observadas"], color=g.TINTA_2, linewidth=1.6, label="Observado")
    ax.plot(fechas, base["esperadas"], color=g.AZUL, label="Esperado, modo operativo")
    ax.plot(fechas, series["estacional"]["esperadas"], color=g.NARANJA, label="Esperado, modo estacional")
    ax.set_ylabel("Celdas de 500 m con fuego en la semana")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    g.titulos(ax, "El modo operativo sigue el pulso de cada temporada; el estacional da el calendario promedio",
              "Prueba 2024–2025: celdas con fuego observadas y esperadas (suma de probabilidades) por semana")
    return g.guardar(fig, ruta)


def figura_calibracion(tablas_cal, nivel, ruta):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.plot([1e-6, 1], [1e-6, 1], color=g.GRIS_CONTEXTO, linewidth=1.2)
    ax.text(0.045, 0.062, "Calibración perfecta", color=g.TENUE, fontsize=8.5, rotation=45,
            rotation_mode="anchor", transform_rotates_text=True, ha="center", va="bottom")
    for regimen, color in (("operativo", g.AZUL), ("estacional", g.NARANJA)):
        t = tablas_cal[regimen]
        t = t[(t["n"] >= 200) & (t["observada"] > 0)]
        ax.plot(t["predicha"], t["observada"], color=color, marker="o", markersize=6,
                markeredgecolor=g.SUPERFICIE, markeredgewidth=1.5, label=f"Modo {regimen}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5e-6, 0.3)
    ax.set_ylim(5e-6, 0.3)
    ax.set_xlabel("Probabilidad predicha (media del tramo)")
    ax.set_ylabel("Frecuencia observada de fuego")
    ax.legend(loc="upper left")
    usada = nivel[(nivel["conjunto"] == "prueba") & (nivel["probabilidad"] == USADA)].set_index("modo")["razon"]
    g.titulos(ax, "Lo predicho y lo observado van juntos; el nivel se pasa un poco",
              f"Prueba {config.ANIOS_PRUEBA[0]}–{config.ANIOS_PRUEBA[1]}. Fuego esperado ÷ observado: "
              f"operativo {ev.formatear(usada['operativo'], 'dec2')}; "
              f"estacional {ev.formatear(usada['estacional'], 'dec2')}")
    return g.guardar(fig, ruta)


NOMBRE_GRUPO = {"cobertura": "Cobertura de suelo", "terreno": "Relieve", "historial": "Historial de fuego",
                "epoca": "Época del año", "clima_normal": "Clima normal", "fuego_reciente": "Fuego reciente",
                "clima_antecedente": "Clima antecedente", "quemas_previas": "Quemas previas (MODIS)",
                "deforestacion": "Deforestación (Hansen)", "uso_suelo": "Uso del suelo (MapBiomas)",
                "poblacion": "Población (WorldPop)", "caminos": "Caminos (OpenStreetMap)"}


def figura_importancia(imp, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, regimen in zip(ejes, ("operativo", "estacional")):
        t = imp[regimen][0].sort_values("caida_pr_auc", ascending=False)
        barras = ax.barh(range(len(t)), t["caida_pr_auc"], height=0.45, color=g.AZUL)
        for barra, valor in zip(barras, t["caida_pr_auc"]):
            ax.text(max(valor, 0) + t["caida_pr_auc"].max() * 0.03, barra.get_y() + barra.get_height() / 2,
                    f"{valor:.3f}".replace(".", ","), va="center", fontsize=8.5, color=g.TINTA)
        ax.set_yticks(range(len(t)), [NOMBRE_GRUPO.get(n, n) for n in t["grupo"]])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(min(0, t["caida_pr_auc"].min()), t["caida_pr_auc"].max() * 1.22)
        ax.set_xlabel(f"Caída del PR-AUC al barajar el grupo · modo {regimen}")
    primero = imp["operativo"][0].sort_values("caida_pr_auc", ascending=False)["grupo"].iloc[0]
    g.titulos(ejes[0], f"Lo que más pesa en el modo operativo: {NOMBRE_GRUPO.get(primero, primero).lower()}",
              "Importancia por permutación en la muestra de prueba 2024–2025")
    fig.subplots_adjust(wspace=0.55)
    return g.guardar(fig, ruta)


def _semana_k(pru, semana):
    return int(np.flatnonzero(pru["semanas"] == semana)[0])


def figura_antes_despues(ctx, pru, cal, semana, limite, ruta):
    """El mismo mapa con el enfoque por provincia y con el modelo de 500 m, y los focos que hubo."""
    k = _semana_k(pru, semana)
    info = info_semana(semana)
    y = np.asarray(pru["y"][k]).astype(bool)
    fig, ejes = plt.subplots(1, 2, figsize=(12.0, 6.4))
    for ax, nombre, titulo in ((ejes[0], "provincia", "Por provincia (enfoque anterior)"),
                               (ejes[1], "operativo", "Modelo de 500 m, modo operativo")):
        clases = ev.clases_de_riesgo(cal[nombre][k], config.UMBRALES_CLASES)
        g.mapa_clases(ax, ctx.malla, ctx.a_malla(clases, relleno=0).astype(np.uint8), ctx.mascara)
        g.contornos(ax, limite, color=g.TINTA_2, ancho=0.8)
        ax.scatter(ctx.lon[y], ctx.lat[y], s=7, color=g.AZUL, edgecolor=g.SUPERFICIE, linewidth=0.3, zorder=5)
        alto = clases >= 3
        ax.set_title(f"{titulo}\nAlto o muy alto: {pct(alto.mean())} del territorio, "
                     f"con {pct(y[alto].sum() / max(1, y.sum()), 0)} de los focos",
                     loc="left", fontsize=9.5, fontweight="normal", color=g.TINTA_2)
    leyenda = g.leyenda_clases(ejes[0])
    ejes[0].add_artist(leyenda)
    ejes[1].legend(handles=[g.marca_focos()], loc="lower left", fontsize=8)
    fig.suptitle(f"Semana del {info['inicio']:%d/%m/%Y} al {info['fin']:%d/%m/%Y}: riesgo previsto y focos observados",
                 x=0.125, ha="left", fontsize=11.5, fontweight="bold", color=g.TINTA, y=0.98)
    return g.guardar(fig, ruta)


MARCO_LEYENDA = {"frameon": True, "facecolor": g.SUPERFICIE, "framealpha": 0.94, "edgecolor": g.EJE}


def nombre_legible(nombre):
    """'VillaTunari' -> 'Villa Tunari' (los nombres de GADM vienen sin espacios)."""
    return re.sub(r"(?<=[a-záéíóúñ])(?=[A-ZÁÉÍÓÚÑ])", " ", nombre)


def etiquetar_municipios(ax, ctx, recorte, minimo=250):
    """Escribe el nombre de cada municipio en el centro de su parte visible, para orientarse."""
    visible = (ctx.lon >= recorte[0]) & (ctx.lon <= recorte[1]) & (ctx.lat >= recorte[2]) & (ctx.lat <= recorte[3])
    for m, nombre in enumerate(ctx.municipios):
        parte = visible & (ctx.municipio_id == m)
        if parte.sum() >= minimo:
            ax.text(ctx.lon[parte].mean(), ctx.lat[parte].mean(), nombre_legible(nombre), fontsize=7.5,
                    color=g.TINTA, ha="center", va="center", zorder=6,
                    path_effects=[efectos.withStroke(linewidth=2.4, foreground=g.SUPERFICIE)])


def figura_acercamiento(ctx, pru, cal, semana, limite, municipios_gj, recorte, nombre_zona, ruta):
    k = _semana_k(pru, semana)
    info = info_semana(semana)
    y = np.asarray(pru["y"][k]).astype(bool)
    clases = ev.clases_de_riesgo(cal["operativo"][k], config.UMBRALES_CLASES)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    g.mapa_clases(ax, ctx.malla, ctx.a_malla(clases, relleno=0).astype(np.uint8), ctx.mascara, recorte=recorte)
    g.contornos(ax, municipios_gj, color=g.TENUE, ancho=0.6)
    g.contornos(ax, limite, color=g.TINTA_2, ancho=0.9)
    dentro = y & (ctx.lon >= recorte[0]) & (ctx.lon <= recorte[1]) & (ctx.lat >= recorte[2]) & (ctx.lat <= recorte[3])
    muchos = int(dentro.sum()) > 80           # con muchos focos, puntos más chicos para no tapar el mapa
    ax.scatter(ctx.lon[dentro], ctx.lat[dentro], s=11 if muchos else 26, color=g.AZUL, edgecolor=g.SUPERFICIE,
               linewidth=0.6 if muchos else 1.0, zorder=5)
    etiquetar_municipios(ax, ctx, recorte)
    leyenda = g.leyenda_clases(ax, loc="upper right", **MARCO_LEYENDA)
    ax.add_artist(leyenda)
    ax.legend(handles=[g.marca_focos()], loc="lower right", fontsize=8, **MARCO_LEYENDA)
    g.titulos(ax, f"{nombre_zona}: cada cuadro es una celda de ~500 m",
              f"Modo operativo, semana del {info['inicio']:%d/%m/%Y} al {info['fin']:%d/%m/%Y}; "
              f"{int(dentro.sum())} celdas con fuego observado en el recuadro")
    return g.guardar(fig, ruta)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    g.estilo()
    ctx = flujo.cargar_todo()
    tablas, figuras = ctx.rutas.tablas, ctx.rutas.figuras
    tablas.mkdir(parents=True, exist_ok=True)
    geo = ctx.rutas.geo
    limite = json.loads((geo / "cochabamba_limite.geojson").read_text(encoding="utf-8"))
    municipios_gj = json.loads((geo / "cochabamba_municipios.geojson").read_text(encoding="utf-8"))
    val = ev.cargar_predicciones(ctx.rutas.predicciones / "validacion")
    pru = ev.cargar_predicciones(ctx.rutas.predicciones / "prueba")

    print("Tablas de modelos...")
    t_val, t_pru = tablas_de_modelos(val, pru, tablas)
    columnas = ["roc_auc", "pr_auc", "captura_1", "captura_5", "captura_10", "area_50", "area_80", "precision_1"]
    print("\nValidación 2022–2023:\n", t_val[columnas].astype(float).round(4).to_string())
    print("\nPrueba 2024–2025:\n", t_pru[columnas].astype(float).round(4).to_string())

    print("\nCalibración, serie semanal y clases...")
    cal = probabilidades(pru)
    y_pru = np.asarray(pru["y"])
    nivel = comparar_calibracion(ctx, pru, val, tablas)
    print("\nNivel de las probabilidades:\n", nivel.round(5).to_string(index=False))
    tablas_cal, series, clases = {}, {}, {}
    for regimen in ("operativo", "estacional"):
        tablas_cal[regimen] = ev.calibracion(cal[regimen], y_pru, BORDES_CALIBRACION)
        tablas_cal[regimen].to_csv(tablas / f"calibracion_{regimen}.csv", index=False)
        series[regimen] = ev.serie_semanal(cal[regimen], y_pru, pru["semanas"])
        series[regimen].to_csv(tablas / f"serie_semanal_{regimen}.csv", index=False)
        clases[regimen] = ev.tabla_clases(cal[regimen], y_pru, config.UMBRALES_CLASES)
        clases[regimen].to_csv(tablas / f"clases_{regimen}.csv", index=False)
        print(f"\nClases, modo {regimen}:\n", clases[regimen].round(5).to_string(index=False))

    print("\nDesgloses por zona, temporada y año...")
    desg = desgloses(ctx, pru, tablas)
    print("Importancia por permutación...")
    imp = importancias(ctx, tablas)
    ablacion_t = tabla_ablacion(ctx, val, tablas)
    escribir_resultados(ctx, t_val, t_pru, clases, desg, imp, ablacion_t, series, nivel,
                        ctx.rutas.resultados)

    print("Figuras...")
    figura_captura(pru, figuras / "eval_curva_captura.png")
    figura_area(t_pru, figuras / "eval_area_necesaria.png")
    figura_serie(series, figuras / "eval_serie_semanal.png")
    figura_calibracion(tablas_cal, nivel, figuras / "eval_calibracion.png")
    figura_importancia(imp, figuras / "eval_importancia.png")
    # semanas de ejemplo: la de más fuego de cada año de prueba
    anios = np.array([info_semana(s)["anio"] for s in pru["semanas"]])
    ejemplos = {}
    for anio in np.unique(anios):
        k = np.flatnonzero(anios == anio)[np.argmax(y_pru[anios == anio].sum(axis=1))]
        ejemplos[int(anio)] = int(pru["semanas"][k])
        figura_antes_despues(ctx, pru, cal, ejemplos[int(anio)], limite,
                             figuras / f"mapa_antes_despues_{anio}.png")
    semana_zoom = ejemplos[max(ejemplos)]
    figura_acercamiento(ctx, pru, cal, semana_zoom, limite, municipios_gj, (-65.75, -64.85, -17.25, -16.65),
                        "Trópico (Villa Tunari, Shinahota, Chimoré)", figuras / "mapa_zoom_tropico.png")
    # región metropolitana: la semana de prueba con más fuego en esa zona
    metro = ctx.zona_id == ctx.zonas.index("Región Metropolitana")
    k_metro = int(np.argmax(y_pru[:, metro].sum(axis=1)))
    figura_acercamiento(ctx, pru, cal, int(pru["semanas"][k_metro]), limite, municipios_gj,
                        (-66.55, -65.85, -17.60, -17.15), "Región Metropolitana y Parque Tunari",
                        figuras / "mapa_zoom_metropolitana.png")

    resumen = {
        "validacion": {"anios": list(config.ANIOS_VALIDA), "semanas": int(len(val["semanas"])),
                       "celdas_con_fuego": int(np.asarray(val["y"]).sum())},
        "prueba": {"anios": list(config.ANIOS_PRUEBA), "semanas": int(len(pru["semanas"])),
                   "celdas_con_fuego": int(y_pru.sum()), "tasa_base": float(y_pru.mean())},
        "modelos_prueba": json.loads(t_pru.drop(columns="descripcion").astype(float).round(5).to_json(orient="index")),
        "modelos_validacion": json.loads(t_val.drop(columns="descripcion").astype(float).round(5).to_json(orient="index")),
        "clases": {r: json.loads(t.round(6).to_json(orient="records")) for r, t in clases.items()},
        "nivel_probabilidades": json.loads(nivel.round(6).to_json(orient="records")),
        "semanas_ejemplo": {str(a): str(inicio_semana(s)) for a, s in ejemplos.items()},
        "semana_zoom_metropolitana": str(inicio_semana(int(pru["semanas"][k_metro]))),
        "correlacion_semanal_observado_esperado": {
            r: float(np.corrcoef(series[r]["observadas"], series[r]["esperadas"])[0, 1]) for r in series},
        "importancia_grupos": {r: json.loads(imp[r][0].round(5).to_json(orient="records")) for r in imp},
        "ablacion": None if ablacion_t is None else json.loads(
            ablacion_t.astype(float).round(5).to_json(orient="index")),
        "desgloses": {modelo: json.loads(desg.loc[modelo].reset_index().drop(columns="desglose")
                                         .set_index("grupo").astype(float).round(5).to_json(orient="index"))
                      for modelo in ("gbm_operativo", "gbm_estacional", "provincia", "recurrencia")},
    }
    (tablas / "evaluacion_resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nTablas en {tablas} y figuras en {figuras}")


if __name__ == "__main__":
    main()
