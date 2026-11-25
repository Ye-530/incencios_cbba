# Resultados

Generado por `python -m incendios evaluar`. Todas las métricas se calculan sobre la malla completa (185.072 celdas por semana, prior real).

- **Captura X %**: de las celdas que tuvieron fuego, qué parte cayó en el X % de celdas de mayor riesgo de su semana.
- **Área para 50 % / 80 %**: qué parte del territorio hay que señalar cada semana para capturar esa parte del fuego.
- **PR-AUC y ROC-AUC**: ordenan todas las celdas-semana a la vez (miden también el acierto en el tiempo).

## Validación 2022–2023: comparación de algoritmos

| Modelo | ROC-AUC | PR-AUC | Captura 1 % | Captura 5 % | Captura 10 % | Área para 50 % | Área para 80 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Operativo (500 m) | 0,935 | 0,016 | 13,8 % | 38,7 % | 57,2 % | 7,8 % | 19,5 % |
| Bosque aleatorio, operativo | 0,938 | 0,014 | 13,8 % | 39,5 % | 59,6 % | 7,5 % | 18,7 % |
| Regresión logística, operativo | 0,921 | 0,012 | 11,7 % | 35,0 % | 56,3 % | 8,3 % | 21,3 % |
| Estacional (500 m) | 0,924 | 0,012 | 10,7 % | 33,0 % | 52,5 % | 9,4 % | 21,9 % |
| Bosque aleatorio, estacional | 0,926 | 0,011 | 10,4 % | 33,9 % | 53,5 % | 9,0 % | 22,2 % |
| Regresión logística, estacional | 0,912 | 0,011 | 10,0 % | 30,3 % | 51,1 % | 9,7 % | 25,2 % |
| Recurrencia histórica | 0,890 | 0,008 | 7,9 % | 25,7 % | 42,6 % | 12,5 % | 29,0 % |
| Por provincia (enfoque anterior) | 0,846 | 0,003 | 1,5 % | 7,5 % | 16,3 % | 29,1 % | 51,6 % |

## Prueba 2024–2025: resultado final

| Modelo | ROC-AUC | PR-AUC | Captura 1 % | Captura 5 % | Captura 10 % | Área para 50 % | Área para 80 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Operativo (500 m) | 0,949 | 0,012 | 12,2 % | 39,2 % | 62,0 % | 7,1 % | 16,0 % |
| Estacional (500 m) | 0,941 | 0,010 | 7,9 % | 32,8 % | 55,9 % | 8,6 % | 17,7 % |
| Recurrencia histórica | 0,913 | 0,006 | 6,2 % | 27,3 % | 48,2 % | 10,4 % | 22,3 % |
| Por provincia (enfoque anterior) | 0,883 | 0,004 | 1,7 % | 8,5 % | 17,2 % | 29,4 % | 46,8 % |

### Precisión y focos nuevos (prueba)

| Modelo | Precisión top 1 % | Precisión top 1 % (±1 celda) | Focos nuevos, captura 5 % | Focos nuevos, captura 10 % |
|---|---:|---:|---:|---:|
| Operativo (500 m) | 0,55 % | 2,51 % | 19,8 % | 38,5 % |
| Estacional (500 m) | 0,36 % | 1,67 % | 25,5 % | 42,6 % |
| Recurrencia histórica | 0,28 % | 1,31 % | 23,8 % | 38,1 % |
| Por provincia (enfoque anterior) | 0,08 % | 0,49 % | 6,6 % | 13,3 % |

Foco nuevo: celda con fuego sin ningún foco a menos de ~2,75 km en las cuatro semanas anteriores.

### Clases de riesgo, modo operativo (prueba)

| Clase | Probabilidad desde | % de celdas-semana | % del fuego | Frecuencia observada |
|---|---:|---:|---:|---:|
| Muy bajo | 0,00 % | 80,270 | 6,3 | 0,004 % |
| Bajo | 0,03 % | 11,546 | 10,0 | 0,039 % |
| Moderado | 0,10 % | 4,217 | 12,3 | 0,132 % |
| Alto | 0,30 % | 2,854 | 32,8 | 0,519 % |
| Muy alto | 1,00 % | 1,114 | 38,6 | 1,565 % |

### Clases de riesgo, modo estacional (prueba)

| Clase | Probabilidad desde | % de celdas-semana | % del fuego | Frecuencia observada |
|---|---:|---:|---:|---:|
| Muy bajo | 0,00 % | 75,572 | 5,5 | 0,003 % |
| Bajo | 0,03 % | 14,585 | 10,6 | 0,033 % |
| Moderado | 0,10 % | 5,240 | 11,9 | 0,102 % |
| Alto | 0,30 % | 3,208 | 35,2 | 0,496 % |
| Muy alto | 1,00 % | 1,394 | 36,9 | 1,196 % |

## Nivel de las probabilidades

Razón entre las celdas con fuego esperadas (suma de probabilidades) y las observadas: 1 es el nivel exacto, más de 1 es sobrestimar. Menor pérdida logarítmica es mejor.

| Años | Modo | Probabilidad | Observadas | Esperadas | Esperadas / observadas | Pérdida logarítmica |
|---|---|---|---:|---:|---:|---:|
| validación | operativo | Del modelo, corregida por el muestreo (la que se usa) | 13.410 | 10.983 | 0,82 | 0,00462 |
| validación | estacional | Del modelo, corregida por el muestreo (la que se usa) | 13.410 | 9.743 | 0,73 | 0,00478 |
| validación | referencia | Tasa base constante | 13.410 | 13.410 | 1,00 | 0,00591 |
| prueba | operativo | Del modelo, corregida por el muestreo (la que se usa) | 8.197 | 9.567 | 1,17 | 0,00295 |
| prueba | operativo | Recalibrada con isotónica sobre 2022–2023 (descartada) | 8.197 | 12.294 | 1,50 | 0,00301 |
| prueba | estacional | Del modelo, corregida por el muestreo (la que se usa) | 8.197 | 11.538 | 1,41 | 0,00305 |
| prueba | estacional | Recalibrada con isotónica sobre 2022–2023 (descartada) | 8.197 | 16.130 | 1,97 | 0,00319 |
| prueba | referencia | Tasa base constante | 8.197 | 8.197 | 1,00 | 0,00393 |

## Prueba por zona, temporada y año

### Operativo (500 m)

| Grupo | Celdas con fuego | Tasa base | ROC-AUC | Captura 5 % | Captura 10 % | Área para 50 % |
|---|---:|---:|---:|---:|---:|---:|
| Trópico | 6.565 | 0,105 % | 0,939 | 22,8 % | 37,0 % | 15,7 % |
| Región Metropolitana | 236 | 0,026 % | 0,810 | 13,1 % | 20,3 % | 30,8 % |
| Valles, Cono Sur y Andina | 1.396 | 0,013 % | 0,904 | 44,6 % | 56,2 % | 7,2 % |
| Con relieve | 8.197 | 0,045 % | 0,949 | 39,2 % | 62,0 % | 7,1 % |
| Temporada (jul–nov) | 7.782 | 0,108 % | 0,914 | 39,6 % | 62,7 % | 6,9 % |
| Resto del año | 415 | 0,004 % | 0,840 | 32,0 % | 48,7 % | 10,6 % |
| 2024 | 4.525 | 0,053 % | 0,949 | 38,9 % | 59,3 % | 7,4 % |
| 2025 | 3.672 | 0,038 % | 0,947 | 39,6 % | 65,2 % | 6,7 % |

### Estacional (500 m)

| Grupo | Celdas con fuego | Tasa base | ROC-AUC | Captura 5 % | Captura 10 % | Área para 50 % |
|---|---:|---:|---:|---:|---:|---:|
| Trópico | 6.565 | 0,105 % | 0,929 | 15,9 % | 29,0 % | 19,5 % |
| Región Metropolitana | 236 | 0,026 % | 0,802 | 9,3 % | 15,3 % | 34,5 % |
| Valles, Cono Sur y Andina | 1.396 | 0,013 % | 0,893 | 38,8 % | 52,4 % | 8,9 % |
| Con relieve | 8.197 | 0,045 % | 0,941 | 32,8 % | 55,9 % | 8,6 % |
| Temporada (jul–nov) | 7.782 | 0,108 % | 0,901 | 33,1 % | 56,6 % | 8,5 % |
| Resto del año | 415 | 0,004 % | 0,821 | 27,0 % | 42,9 % | 12,3 % |
| 2024 | 4.525 | 0,053 % | 0,939 | 28,4 % | 50,0 % | 10,0 % |
| 2025 | 3.672 | 0,038 % | 0,944 | 38,1 % | 63,2 % | 7,0 % |

## Importancia por permutación (muestra de prueba)

### Modo operativo

| Grupo de variables | Caída del PR-AUC al barajarlo |
|---|---:|
| Fuego reciente | 0,142 |
| Historial de fuego | 0,068 |
| Época del año | 0,061 |
| Cobertura de suelo | 0,026 |
| Clima normal | 0,015 |
| Deforestación (Hansen) | 0,014 |
| Uso del suelo (MapBiomas) | 0,009 |
| Población (WorldPop) | 0,008 |
| Clima antecedente | 0,005 |
| Relieve | 0,001 |
| Caminos (OpenStreetMap) | -0,000 |
| Quemas previas (MODIS) | -0,009 |

### Modo estacional

| Grupo de variables | Caída del PR-AUC al barajarlo |
|---|---:|
| Historial de fuego | 0,183 |
| Época del año | 0,115 |
| Deforestación (Hansen) | 0,049 |
| Cobertura de suelo | 0,006 |
| Relieve | 0,003 |
| Uso del suelo (MapBiomas) | 0,002 |
| Caminos (OpenStreetMap) | 0,001 |
| Clima normal | 0,000 |
| Población (WorldPop) | -0,002 |
| Quemas previas (MODIS) | -0,011 |

## Acierto en el tiempo (prueba)

Correlación semanal entre celdas con fuego observadas y esperadas: modo operativo 0,73; modo estacional 0,72.
