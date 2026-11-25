"""Despliegue: de una fecha a un mapa de riesgo y a la lista de lugares con más riesgo."""
import json
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .flujo import huellas_distintas
from .evaluacion import clases_de_riesgo
from .geotiff import escribir_geotiff
from .graficos import CLASES
from .tiempo import info_semana, semana_de_fecha
from .variables import ANIOS_HISTORIAL, REGIMENES


def guardar_versiones(dir_modelos):
    """Deja constancia de con qué versiones se entrenó (los modelos guardados dependen de ellas)."""
    import platform

    import sklearn
    carpeta = Path(dir_modelos)
    carpeta.mkdir(parents=True, exist_ok=True)
    versiones = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    ruta = carpeta / "versiones.json"
    ruta.write_text(json.dumps(versiones, indent=1), encoding="utf-8")
    return ruta


def _versiones_de_entrenamiento(dir_modelos):
    ruta = Path(dir_modelos) / "versiones.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def cargar_modelo(dir_modelos, modo):
    """Abre el modelo final de ese modo. Un modelo guardado con otra versión de scikit-learn
    puede no abrir, o abrir con resultados distintos: en ese caso hay que reentrenar."""
    import warnings

    import sklearn
    ruta = Path(dir_modelos) / f"final_{modo}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Entrene el modelo: python -m incendios entrenar")
    entrenado = _versiones_de_entrenamiento(dir_modelos).get("scikit-learn")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # el aviso de versión se da abajo, en español
            modelo = joblib.load(ruta)
    except Exception as error:
        raise RuntimeError(
            f"No se pudo abrir {ruta.name} ({type(error).__name__}). Se guardó con scikit-learn "
            f"{entrenado or 'desconocido'} y aquí hay {sklearn.__version__}. Vuelva a entrenar con "
            f"python -m incendios entrenar, o instale esa versión.") from error
    modelo.aviso_version = None
    if entrenado and entrenado != sklearn.__version__:
        modelo.aviso_version = (f"El modelo se guardó con scikit-learn {entrenado} y aquí hay "
                                f"{sklearn.__version__}: si los mapas se ven raros, reentrene con "
                                f"python -m incendios entrenar.")
    return modelo


def predecir(ctx, fecha, modo="estacional", modelo=None):
    """Riesgo de la semana (lunes a domingo) que contiene `fecha`.

    modo 'estacional': cualquier fecha; usa historial de focos hasta el año anterior.
    modo 'operativo' : solo si hay focos y clima hasta la víspera de esa semana.
    Devuelve probabilidad y clase por celda del departamento, y avisos si hubo que
    usar información menos reciente de lo ideal.
    """
    if modo not in REGIMENES:
        raise ValueError(f"El modo debe ser uno de {REGIMENES}, no '{modo}'")
    modelo = modelo or cargar_modelo(ctx.rutas.modelos, modo)
    if modelo.regimen != modo:
        raise ValueError(f"El modelo es de modo '{modelo.regimen}' y se pidió modo '{modo}'")
    c = ctx.constructor
    semana = semana_de_fecha(fecha)
    info = info_semana(semana)
    avisos = []
    if getattr(modelo, "aviso_version", None):
        avisos.append(modelo.aviso_version)
    entrenado_con, ahora = getattr(modelo, "info", {}).get("huella"), ctx.notas.get("huella")
    if entrenado_con and ahora and huellas_distintas(entrenado_con, ahora):
        avisos.append("El modelo se entrenó con otros archivos de clima, relieve, cobertura, capas extra, variables o "
                      "satélites que los que hay ahora: vuelva a entrenar (python -m incendios entrenar) para que modelo "
                      "y datos coincidan.")
    # las columnas se eligen por nombre: sirve para modelos entrenados con o sin capas extra
    nombres = c.nombres(modo)
    variables = list(getattr(modelo, "variables", None) or nombres)
    faltan = [v for v in variables if v not in nombres]
    if faltan:
        raise ValueError(
            f"El modelo usa {len(faltan)} variables que ahora no existen ({', '.join(faltan[:6])}). Se entrenó "
            f"con capas extra (data/capas) que faltan o están apagadas con INCENDIOS_CAPAS=0: devuélvalas o "
            f"vuelva a entrenar con: python -m incendios entrenar")
    if c.capas is not None and set(variables) & set(c.capas.nombres()):
        avisos += c.capas.avisos(info["anio"])
    primer_anio = c.anio_inicial + ANIOS_HISTORIAL
    if info["anio"] < primer_anio:
        raise ValueError(f"Solo se puede predecir desde {primer_anio}: antes no hay "
                         f"{ANIOS_HISTORIAL} años de focos de historial")
    if modo == "operativo":
        ultimo = ctx.notas["ultima_fecha_focos"]
        if semana > ctx.n_semanas or not ctx.clima.disponible(semana):
            raise ValueError(
                f"El modo operativo necesita focos y clima hasta la víspera del {info['inicio']}, "
                f"y los datos llegan hasta {ultimo}. Actualice los datos o use el modo estacional.")
        if ctx.dias_obs[semana - 1] < config.MIN_DIAS_OBSERVADOS:
            avisos.append(f"La semana anterior tuvo solo {int(ctx.dias_obs[semana - 1])} días de satélite: "
                          f"la actividad reciente de fuego está subestimada.")
        vispera = info["inicio"] - timedelta(days=1)
        if str(ultimo) < str(vispera):
            avisos.append(f"Los focos llegan hasta el {ultimo} y la semana empieza el {info['inicio']}: la "
                          f"actividad reciente de fuego está incompleta. Baje los focos nuevos "
                          f"(python -m incendios descargar --que focos) para una predicción mejor.")
        X = c.matriz(semana, "operativo")
    elif info["anio"] - 1 > c.anio_ultimo_completo:
        avisos.append(f"Aún no hay focos de {info['anio'] - 1}: se usó el historial hasta "
                      f"{c.anio_ultimo_completo}. Actualice los focos para una predicción mejor.")
        X = c.matriz(semana, "estacional", anio_historial=c.anio_ultimo_completo + 1)
    else:
        X = c.matriz(semana, "estacional")
    prob = modelo.probabilidad(X[:, [nombres.index(v) for v in variables]])
    return {"semana": semana, "info": info, "modo": modo, "prob": prob,
            "clase": clases_de_riesgo(prob, config.UMBRALES_CLASES), "avisos": avisos}


def tabla_top(ctx, prob, n=100):
    """Las `n` celdas de mayor riesgo, con coordenadas del centro y municipio."""
    orden = np.argsort(-prob, kind="stable")[:n]
    clase = clases_de_riesgo(prob[orden], config.UMBRALES_CLASES)
    return pd.DataFrame({
        "puesto": np.arange(1, len(orden) + 1),
        "latitud": np.round(ctx.lat[orden], 5),
        "longitud": np.round(ctx.lon[orden], 5),
        "probabilidad": prob[orden].astype(float),
        "clase": [CLASES[k] for k in clase],
        "municipio": [ctx.municipios[i] for i in ctx.municipio_id[orden]],
        "provincia": [ctx.provincia_de_municipio[i] for i in ctx.municipio_id[orden]],
    })


def guardar_prediccion(ctx, resultado, carpeta, n_top=100):
    """Escribe GeoTIFF de probabilidad y de clase, CSV con las celdas de más riesgo, PNG y resumen."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import graficos as g

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    info, modo = resultado["info"], resultado["modo"]
    base = f"riesgo_{modo}_{info['inicio']}"
    prob_2d = ctx.a_malla(resultado["prob"])
    clase_2d = np.full(ctx.malla.nfilas * ctx.malla.ncols, 255, dtype=np.uint8)
    clase_2d[ctx.idx] = resultado["clase"]
    clase_2d = clase_2d.reshape(ctx.malla.forma)
    rutas = {
        "probabilidad_tif": escribir_geotiff(carpeta / f"{base}_probabilidad.tif", prob_2d, ctx.malla),
        "clase_tif": escribir_geotiff(carpeta / f"{base}_clase.tif", clase_2d, ctx.malla, sin_dato=255),
        "top_csv": carpeta / f"{base}_top{n_top}.csv",
        "mapa_png": carpeta / f"{base}.png",
        "resumen_json": carpeta / f"{base}.json",
    }
    tabla_top(ctx, resultado["prob"], n_top).to_csv(rutas["top_csv"], index=False, encoding="utf-8-sig")

    g.estilo()
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    g.mapa_clases(ax, ctx.malla, np.where(clase_2d == 255, 0, clase_2d), ctx.mascara)
    g.leyenda_clases(ax)
    g.titulos(ax, f"Riesgo de incendio, semana del {info['inicio']:%d/%m/%Y} al {info['fin']:%d/%m/%Y}",
              f"Modo {modo} · probabilidad semanal de foco por celda de ~500 m")
    g.guardar(fig, rutas["mapa_png"])

    cuenta = np.bincount(resultado["clase"], minlength=5)
    resumen = {
        "semana_inicio": str(info["inicio"]), "semana_fin": str(info["fin"]),
        "anio_iso": info["anio"], "semana_iso": info["semana_anio"], "modo": modo,
        "umbrales": list(config.UMBRALES_CLASES),
        "celdas_por_clase": {CLASES[k]: int(cuenta[k]) for k in range(5)},
        "celdas_esperadas_con_fuego": float(resultado["prob"].sum()),
        "probabilidad_maxima": float(resultado["prob"].max()),
        "avisos": resultado["avisos"],
    }
    rutas["resumen_json"].write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    return rutas
