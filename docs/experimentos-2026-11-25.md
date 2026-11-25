# Experimentos para mejorar el modelo (25 de noviembre de 2026)

Se probó qué cambios mejoran de verdad el modelo. Todas las variantes se midieron sobre la malla
completa, ranking dentro de cada semana, igual que `RESULTADOS.md`. Para elegir se usó la validación
(entrenando 2015–2021 y midiendo 2022–2023). La candidata final se confirmó en un segundo período
(entrenando 2015–2019 y midiendo 2020–2021) y con dos semillas. Los años de prueba 2024–2025 se usaron
una sola vez, al final, con lo ya elegido.

## Lo primero: cuánto se mueven las métricas por azar

Con dos años de validación, la captura al 5 % se mueve hasta **1,3 puntos** con solo cambiar la
semilla del GBM (validación 2020–2021: 43,1 % y 44,4 %; validación 2022–2023: 38,7 % y 38,1 %).
Remuestreando semanas, el intervalo al 95 % de una diferencia entre dos variantes es de unos
**±3 puntos**: dos semanas de mucho fuego pesan más que todo el resto. Por eso una variante solo
cuenta como mejora si gana en los dos períodos y con las dos semillas.

## Resultados

### Se integró: fuego de los últimos días

Variables nuevas del modo operativo: celdas con fuego de viernes a domingo de la semana anterior
(en la celda, en 3×3, 11×11 y 41×41 celdas), con fuego el domingo (11×11) y total del departamento
de viernes a domingo. El fuego de los últimos días anticipa la semana mejor que el de toda la semana.

| Período de validación | Semilla | PR-AUC | Captura 1 % | Captura 5 % | Captura 10 % | Área para 50 % |
|---|---|---|---|---|---|---|
| 2022–2023 | 42 | 0,0157 → 0,0163 | 13,8 → 15,7 | 38,7 → 39,7 | 57,2 → 58,5 | 7,8 → 7,5 |
| 2022–2023 | 7 | 0,0161 → 0,0158 | 13,6 → 14,3 | 38,1 → 39,0 | 56,7 → 58,8 | 8,0 → 7,6 |
| 2020–2021 | 42 | 0,0130 → 0,0149 | 14,0 → 14,9 | 43,1 → 44,3 | 64,9 → 65,5 | 6,4 → 6,2 |
| 2020–2021 | 7 | 0,0134 → 0,0151 | 14,1 → 14,8 | 44,4 → 44,0 | 65,4 → 66,0 | 6,2 → 6,2 |

Mejora la captura al 1 % y al 10 % en los cuatro casos, y la del 5 % y el PR-AUC en tres de cuatro.

### Se integró: promedio de tres semillas

Promediar la probabilidad de un GBM por semilla dio, en los cuatro casos, un resultado igual o mejor
que la mejor de las dos semillas sola, siempre en PR-AUC y captura al 10 %. Con las dos mejoras juntas
(promedio de dos semillas), frente al modelo anterior:

| Período | Captura 5 % | Captura 10 % | PR-AUC |
|---|---|---|---|
| 2022–2023 | 38,7 → 40,1 | 57,2 → 59,4 | 0,0157 → 0,0165 |
| 2020–2021 | 43,1 → 44,3 | 64,9 → 66,4 | 0,0130 → 0,0156 |

Los modelos de prueba y los finales promedian tres semillas (`config.SEMILLAS_GBM`). La fase de
validación y la ablación siguen con una, para que sus comparaciones sean parejas y no triplicar su tiempo.

### Se descartó

| Idea | Resultado en validación 2022–2023 (captura 5 %, PR-AUC) |
|---|---|
| Ubicación (latitud y longitud de la celda) e historial extra (años desde el último fuego, años con fuego, máximo anual) | estacional 33,0 → 31,9 y 0,0115 → 0,0107; en el operativo, junto con la intensidad, 38,7 → 37,1 |
| Intensidad del fuego reciente (detecciones, FRP, semanas desde el último fuego) | 38,7 → 38,2 y 0,0157 → 0,0147 |
| Días de satélite de la semana anterior | tenía una fuga: depende del lunes de la semana predicha (lo encontró la revisión independiente) |
| Suavizado espacial de las probabilidades | igual o peor (3×3 con peso 0,5: 38,5; con peso 1: 37,5) |
| Mezcla GBM + bosque aleatorio | 39,1 y 0,0163: menos que el fuego de los últimos días y suma un modelo de 4 MB |
| Unión de satélites S-NPP + NOAA-20 como focos y como objetivo | contra los focos S-NPP: 38,2 y 0,0136; contra la unión, igual que el modelo S-NPP |

La ubicación y el historial que crece con el registro (desde 2012) son además riesgosos: con las mismas
celdas en entrenamiento y validación, la validación temporal no puede detectar que el modelo memorice
lugares, y "3 años con fuego" significa cosas distintas en 2015 y en 2026.

## Satélites: el cambio de S-NPP a NOAA-20

NASA dejó de producir S-NPP el 2 de noviembre de 2026. Con la serie NOAA-20 bajada de FIRMS desde abril
de 2018 se midió en 2024–2025 qué pasa cuando el modelo recibe focos NOAA-20 y se lo juzga contra focos NOAA-20:

| Modelo | PR-AUC | Captura 5 % | Captura 10 % | Esperadas / observadas |
|---|---|---|---|---|
| Entrenado con S-NPP (el de siempre) | 0,0145 | 39,4 % | 63,5 % | 1,13 |
| Reentrenado con NOAA-20 desde abril de 2018 | 0,0156 | 40,0 % | 63,8 % | 1,17 |

El modelo entrenado con S-NPP sigue funcionando con NOAA-20: el cambio de satélite no lo rompe.
Reentrenar con NOAA-20 gana poco, así que se mantiene la cadena actual (`config.FUENTES_FOCOS`).

Un dato que explica por qué acertar la celda exacta es tan difícil: en 2024–2025, de las 8.197
celdas-semana donde S-NPP vio fuego, NOAA-20 vio fuego en la misma celda y la misma semana solo en el
**30 %**, aunque en total ve parecido (8.902). Los dos instrumentos son iguales y pasan con unos 50
minutos de diferencia: buena parte de qué celda "tiene fuego" depende de qué satélite pasó mientras
el fuego estaba activo.

## Cómo se hizo

Las variantes se entrenaron sobre una misma muestra caso-control (las mismas filas para todas) con el
mismo número de árboles, y se puntuó la malla completa de cada semana de validación. Las diferencias
por semana se remuestrearon (4.000 réplicas) para el intervalo de confianza. La revisión de fugas la
hicieron tres revisores independientes (fugas, errores numéricos, uso operativo); cada hallazgo lo
verificaron dos escépticos que intentaron refutarlo.
