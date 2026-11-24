# Riesgo de incendios en Cochabamba a 500 m — Diseño

Fecha: 2026-11-18 · Metodología: CRISP-DM

## Intención y porqué

Un sistema de machine learning que, dada una **fecha** (año y día), entregue el
**mapa de riesgo de incendio del departamento de Cochabamba para esa semana** con
detalle fino (celdas de ~500 m), y las coordenadas de los puntos de mayor riesgo.
El intento anterior predecía por provincia y día: el resultado era "un punto rojo
grande que cubre media Cochabamba". El proyecto es para un diplomado, así que el
proceso (CRISP-DM) y su documentación importan tanto como el modelo.

## Objetivos

- Unidad espacial: celda de 0,005° (≈ 555 m × 530 m); 185.072 celdas en el departamento.
- Unidad temporal: semana ISO (lunes a domingo). Una fecha se asigna a su semana.
- Dos horizontes:
  - **Estacional**: cualquier fecha futura. Usa solo lo que se conoce con meses de
    anticipación (terreno, cobertura, historial de fuego hasta el año anterior,
    época del año, clima normal).
  - **Operativo**: la semana siguiente. Suma clima antecedente (hasta el día
    anterior) y actividad de fuego de las semanas previas.
- Evaluación honesta: partición temporal (entrenar 2015–2021, validar 2022–2023,
  probar 2024–2025) y comparación contra líneas base, incluida la de "provincia"
  que reproduce el enfoque anterior.
- Reproducible en una PC común: Python + numpy, pandas, scipy, scikit-learn,
  matplotlib, Pillow. Sin GDAL, rasterio ni geopandas.

## No-objetivos

- No se descargan datos nuevos (el entorno de construcción no tiene red). Se
  entrega un script para que el usuario los baje y el flujo los use al re-ejecutar.
- No se predice propagación ni área quemada; solo ocurrencia de fuego detectable por satélite.
- No se distingue incendio forestal de quema agrícola (chaqueo): VIIRS no lo separa.

## Anti-objetivos (fallaría aunque "funcione")

- Mapas en bloques por provincia o por celda de clima (el "punto rojo grande").
- Métricas infladas por fuga de información del futuro o por evaluar con muestras balanceadas.
- Afirmar precisión que los datos no sostienen (clima de ~55 km presentado como local).

## Restricciones encontradas en los datos

- Clima NASA POWER: 16 puntos pero 9 series distintas (celda MERRA-2 de 0,5° × 0,625°).
  Artefacto de lluvia el 4–5 de junio de 2020 (hasta 808 mm/día) que contamina la
  humedad de suelo hasta ~octubre de 2020.
- DEM: 4 teselas (lon −67 a −65, lat −18 a −16) = 68 % del área, 57 % de los focos.
- WorldCover 2021: falta lat < −18 (86 % del área, 99 % de los focos). Es una sola
  fecha: para años ≤ 2021 puede reflejar quemas pasadas; la prueba (2024–2025) es posterior.
- NDVI: promedio por provincia, solo 2022–2026. No se usa en el modelo.
- VIIRS S-NPP: días sin satélite (27-jul a 10-ago 2022; varios en 2024). Las
  semanas con menos de 4 días observados se excluyen de entrenar y evaluar.

## Enfoques considerados

1. **Clasificador por provincia y día** (el anterior). Simple, pero su resolución
   espacial es la provincia. Descartado como solución; se conserva como línea base.
2. **Clasificación celda × semana con muestreo caso-control** (elegido). Cada celda
   de 500 m y semana es una observación; se usan todos los positivos y una muestra
   uniforme de negativos, corrigiendo después la probabilidad al prior real. Permite
   evaluar sobre la malla completa y da mapas finos. Costo: desbalance extremo
   (0,05 % de positivos), que se maneja con el muestreo y con métricas adecuadas.
3. **Procesos puntuales / densidad kernel espacio-temporal** o **redes convolucionales
   sobre imágenes**. Más potentes en teoría, pero exigen más datos finos de los que
   hay (clima de 55 km) y cómputo que no se justifica; menos explicables ante un tribunal.

Decisión: enfoque 2 con Gradient Boosting por histogramas (scikit-learn), con
regresión logística y bosque aleatorio como comparación.

## Diseño

**Variable objetivo.** `y = 1` si la celda tuvo ≥ 1 detección VIIRS S-NPP de
vegetación (`type == 0`) con confianza nominal o alta en la semana.

**Grupos de variables** (todas calculadas solo con información anterior a la semana objetivo):

| Grupo | Contenido | Estacional | Operativo |
|---|---|---|---|
| Estáticas | fracciones de cobertura WorldCover, diversidad, entorno de 2,5 km, distancia a zona urbana/cultivo/agua, elevación, pendiente, rugosidad, orientación | sí | sí |
| Historial anual | semanas con fuego en la celda y en vecindarios de 1,5 / 5,5 / 20 km en los 1 y 3 años ISO anteriores, suma con decaimiento, recurrencia en ±3 semanas de la misma época | sí | sí |
| Época del año | seno/coseno del día del año, semana ISO | sí | sí |
| Clima normal | normales por semana del año (de 2012–2021) interpoladas a la celda | sí | sí |
| Fuego reciente | actividad en la celda y vecindarios en la semana anterior y las 4 anteriores; actividad departamental | no | sí |
| Clima antecedente | FWI canadiense y sus códigos, lluvia 7/30/90 días, días sin lluvia, temperatura, humedad, viento, VPD y anomalías, al día anterior al inicio de la semana | no | sí |

No se usan latitud, longitud ni unidad administrativa como variables (evitan mapas en bloques).

**Clima a la celda.** Las 9 series únicas se ubican en el centro de su celda
MERRA-2 y se interpolan con IDW (4 vecinas, potencia 2) para no dejar costuras.

**Muestreo.** Por semana: todos los positivos + 4.000 negativos al azar. La
probabilidad se corrige con `odds_real = odds_muestra × tasa_de_muestreo` y luego
se calibra con regresión isotónica sobre validación (malla completa).

**Modelos.** Línea base provincia, línea base recurrencia histórica, regresión
logística, bosque aleatorio, HistGradientBoosting (principal) en ambos regímenes.

**Métricas** (sobre la malla completa, prior real): PR-AUC, ROC-AUC, captura de
focos en el 1/2/5/10 % de celdas de mayor riesgo de cada semana, área necesaria
para capturar 50 % y 80 % de los focos, precisión con tolerancia de una celda,
calibración, y desglose por zona, por temporada y para focos nuevos.

**Despliegue.** `predecir(fecha, modo)` → GeoTIFF + PNG + CSV con las celdas de
mayor riesgo (lat, lon, municipio). Visor HTML autónomo para elegir semana.

## Decisiones tomadas por el constructor (no consultadas)

- Tamaño de celda 0,005°, semana ISO, 3 años de historial, 4.000 negativos por semana.
- Zonas de reporte: Trópico (Villa Tunari, Chimoré, Puerto Villarroel, Entre Ríos,
  Shinahota), Región Metropolitana (Cochabamba, Sacaba, Quillacollo, Colcapirhua,
  Tiquipaya, Vinto, Sipe Sipe) y Resto.
- Lluvia de 2020-06-04/05 → 0 mm; humedad de suelo de 2020-06-04 a 2020-10-31 → sin dato.

## Qué queda para quien lo ejecute

Bajar las teselas faltantes y los datos mejores con `scripts/descargar_datos_mejorados.py`
y volver a correr los pasos 02 a 05.
