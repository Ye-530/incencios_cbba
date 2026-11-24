"""Parámetros del proyecto. Todo lo que se decide una sola vez vive aquí."""
from .rutas import RAIZ  # noqa: F401  (dónde está cada carpeta: incendios/rutas.py)

# Archivos anuales de focos (NASA FIRMS). El modelo se entrenó con VIIRS de S-NPP; NASA dejó de
# producirlo el 2-nov-2026 a las 13:00 UTC (ese día ya no tiene la pasada de la tarde). Desde ahí se
# encadena VIIRS de NOAA-20: mismo instrumento, misma resolución (375 m) y órbita parecida.
PATRON_FOCOS = "viirs-snpp_*_Bolivia.csv"
FUENTES_FOCOS = (
    {"nombre": "S-NPP", "patron": PATRON_FOCOS, "hasta": "2026-11-01"},
    {"nombre": "NOAA-20", "patron": "viirs-noaa20_*_Bolivia.csv", "desde": "2026-11-02"},
)

# Malla regular en grados, alineada con los píxeles de WorldCover (10 m) y del DEM (30 m).
CELDA = 0.005          # ~555 m norte-sur, ~530 m este-oeste a 17° S
OESTE = -67.010
NORTE = -15.775
NCOLS = 556
NFILAS = 582

# Partición temporal por año ISO.
ANIOS_ENTRENA = (2015, 2021)
ANIOS_VALIDA = (2022, 2023)
ANIOS_PRUEBA = (2024, 2025)

NEG_POR_SEMANA = 4000   # negativos muestreados por semana (caso-control)
SEMILLA = 42
MIN_DIAS_OBSERVADOS = 4  # una semana con menos días de satélite no se usa para entrenar ni evaluar

# Calidad del clima NASA POWER (ver informe, fase 2).
# Días de retraso con que se publica cada fuente de clima. El modo operativo usa el clima hasta
# el lunes - 1 - retraso, igual al entrenar que al predecir: así la predicción se puede hacer el
# mismo lunes con lo que ya está publicado. POWER queda en 0, como se entrenaron los modelos.
DIAS_RETRASO_CLIMA = {"POWER": 0, "ERA5-Land": 5}

FECHAS_LLUVIA_ARTEFACTO = ("2020-06-04", "2020-06-05")
LLUVIA_MAXIMA_CREIBLE = 150.0                      # mm/día para una celda de ~55 km
VENTANA_SUELO_CONTAMINADO = ("2020-06-04", "2020-10-31")

# Fuente de calor fija (horno, chimenea, antorcha de gas): celda con detecciones en al menos
# FUENTE_MIN_SEMANAS semanas distintas, de las cuales FUENTE_MIN_NOCTURNAS o más son nocturnas.
# El fuego de vegetación se detecta sobre todo de día (79 % de las detecciones); con estos
# umbrales, que una celda de vegetación caiga aquí por azar tiene probabilidad ~1 en 10.000.
FUENTE_MIN_SEMANAS = 10
FUENTE_MIN_NOCTURNAS = 0.8

# Clases de riesgo: probabilidad semanal de al menos un foco en la celda (~0,29 km²).
# Muy bajo < 0,03 % <= Bajo < 0,1 % <= Moderado < 0,3 % <= Alto < 1 % <= Muy alto
# Cada clase arde unas tres veces más que la anterior. La tasa media es ~0,05 % y casi
# ninguna celda supera el 5 % en una semana, así que "Muy alto" empieza en 1 %.
UMBRALES_CLASES = (0.0003, 0.001, 0.003, 0.01)

# Zonas para reportar resultados (nombres de municipio como vienen en GADM).
ZONAS = {
    "Trópico": ["VillaTunari", "Chimoré", "PuertoVillarroel", "EntreRíos", "Shinahota"],
    "Región Metropolitana": ["Cochabamba", "Sacaba", "Quillacollo", "Colcapirhua",
                             "Tiquipaya", "Vinto", "Sipesipe"],
}
ZONA_RESTO = "Valles, Cono Sur y Andina"
