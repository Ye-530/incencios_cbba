# Resultados

Generado por `python -m incendios evaluar`. Todas las métricas se calculan sobre la malla completa (185.072 celdas por semana, prior real).

- **Captura X %**: de las celdas que tuvieron fuego, qué parte cayó en el X % de celdas de mayor riesgo de su semana.
- **Área para 50 % / 80 %**: qué parte del territorio hay que señalar cada semana para capturar esa parte del fuego.
- **PR-AUC y ROC-AUC**: ordenan todas las celdas-semana a la vez (miden también el acierto en el tiempo).

## Validación 2022–2023: comparación de algoritmos

| Modelo | ROC-AUC | PR-AUC | Captura 1 % | Captura 5 % | Captura 10 % | Área para 50 % | Área para 80 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Operativo (500 m) | 0,940 | 0,017 | 15,2 % | 41,2 % | 60,7 % | 7,1 % | 18,0 % |
| Bosque aleatorio, operativo | 0,939 | 0,014 | 14,4 % | 40,8 % | 60,7 % | 7,1 % | 18,5 % |
| Regresión logística, operativo | 0,925 | 0,013 | 12,8 % | 37,2 % | 57,4 % | 7,9 % | 19,6 % |
| Estacional (500 m) | 0,931 | 0,012 | 10,4 % | 35,0 % | 55,1 % | 8,6 % | 19,8 % |
| Bosque aleatorio, estacional | 0,931 | 0,012 | 11,0 % | 34,9 % | 54,2 % | 8,7 % | 21,4 % |
| Regresión logística, estacional | 0,919 | 0,011 | 10,3 % | 32,0 % | 52,5 % | 9,4 % | 21,6 % |
| Recurrencia histórica | 0,890 | 0,008 | 7,9 % | 25,7 % | 42,6 % | 12,5 % | 29,0 % |
| Por provincia (enfoque anterior) | 0,846 | 0,003 | 1,5 % | 7,5 % | 16,3 % | 29,1 % | 51,6 % |

## Prueba 2024–2025: resultado final

| Modelo | ROC-AUC | PR-AUC | Captura 1 % | Captura 5 % | Captura 10 % | Área para 50 % | Área para 80 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Operativo (500 m) | 0,948 | 0,012 | 12,5 % | 39,5 % | 63,2 % | 7,0 % | 16,0 % |
| Estacional (500 m) | 0,942 | 0,009 | 9,2 % | 35,4 % | 59,3 % | 7,8 % | 17,1 % |
| Recurrencia histórica | 0,913 | 0,006 | 6,2 % | 27,3 % | 48,2 % | 10,4 % | 22,3 % |
| Por provincia (enfoque anterior) | 0,883 | 0,004 | 1,7 % | 8,5 % | 17,2 % | 29,4 % | 46,8 % |

### Precisión y focos nuevos (prueba)

| Modelo | Precisión top 1 % | Precisión top 1 % (±1 celda) | Focos nuevos, captura 5 % | Focos nuevos, captura 10 % |
|---|---:|---:|---:|---:|
| Operativo (500 m) | 0,57 % | 2,20 % | 23,0 % | 44,2 % |
| Estacional (500 m) | 0,42 % | 1,61 % | 30,7 % | 47,4 % |
| Recurrencia histórica | 0,28 % | 1,31 % | 23,8 % | 38,1 % |
| Por provincia (enfoque anterior) | 0,08 % | 0,49 % | 6,6 % | 13,3 % |

Foco nuevo: celda con fuego sin ningún foco a menos de ~2,75 km en las cuatro semanas anteriores.

### Clases de riesgo, modo operativo (prueba)

| Clase | Probabilidad desde | % de celdas-semana | % del fuego | Frecuencia observada |
|---|---:|---:|---:|---:|
| Muy bajo | 0,00 % | 82,028 | 7,1 | 0,004 % |
| Bajo | 0,03 % | 11,174 | 12,5 | 0,051 % |
| Moderado | 0,10 % | 3,958 | 18,9 | 0,216 % |
| Alto | 0,30 % | 2,346 | 42,1 | 0,811 % |
| Muy alto | 1,00 % | 0,494 | 19,4 | 1,776 % |

### Clases de riesgo, modo estacional (prueba)

| Clase | Probabilidad desde | % de celdas-semana | % del fuego | Frecuencia observada |
|---|---:|---:|---:|---:|
| Muy bajo | 0,00 % | 77,145 | 6,3 | 0,004 % |
| Bajo | 0,03 % | 14,528 | 12,5 | 0,039 % |
| Moderado | 0,10 % | 4,924 | 15,7 | 0,145 % |
| Alto | 0,30 % | 2,788 | 45,9 | 0,744 % |
| Muy alto | 1,00 % | 0,616 | 19,5 | 1,429 % |

## Nivel de las probabilidades

Razón entre las celdas con fuego esperadas (suma de probabilidades) y las observadas: 1 es el nivel exacto, más de 1 es sobrestimar. Menor pérdida logarítmica es mejor.

| Años | Modo | Probabilidad | Observadas | Esperadas | Esperadas / observadas | Pérdida logarítmica |
|---|---|---|---:|---:|---:|---:|
| validación | operativo | Del modelo, corregida por el muestreo (la que se usa) | 13.410 | 9.344 | 0,70 | 0,00459 |
| validación | estacional | Del modelo, corregida por el muestreo (la que se usa) | 13.410 | 8.203 | 0,61 | 0,00476 |
| validación | referencia | Tasa base constante | 13.410 | 13.410 | 1,00 | 0,00591 |
| prueba | operativo | Del modelo, corregida por el muestreo (la que se usa) | 8.197 | 7.160 | 0,87 | 0,00298 |
| prueba | operativo | Recalibrada con isotónica sobre 2022–2023 (descartada) | 8.197 | 10.954 | 1,34 | 0,00299 |
| prueba | estacional | Del modelo, corregida por el muestreo (la que se usa) | 8.197 | 8.751 | 1,07 | 0,00305 |
| prueba | estacional | Recalibrada con isotónica sobre 2022–2023 (descartada) | 8.197 | 14.653 | 1,79 | 0,00314 |
| prueba | referencia | Tasa base constante | 8.197 | 8.197 | 1,00 | 0,00393 |

## Prueba por zona, temporada y año

### Operativo (500 m)

| Grupo | Celdas con fuego | Tasa base | ROC-AUC | Captura 5 % | Captura 10 % | Área para 50 % |
|---|---:|---:|---:|---:|---:|---:|
| Trópico | 6.565 | 0,105 % | 0,937 | 21,4 % | 35,8 % | 15,9 % |
| Región Metropolitana | 236 | 0,026 % | 0,815 | 16,9 % | 23,7 % | 31,2 % |
| Valles, Cono Sur y Andina | 1.396 | 0,013 % | 0,904 | 44,8 % | 57,1 % | 7,1 % |
| Con relieve | 8.197 | 0,045 % | 0,948 | 39,5 % | 63,2 % | 7,0 % |
| Temporada (jul–nov) | 7.782 | 0,108 % | 0,912 | 39,9 % | 64,0 % | 6,9 % |
| Resto del año | 415 | 0,004 % | 0,843 | 33,0 % | 49,2 % | 10,4 % |
| 2024 | 4.525 | 0,053 % | 0,951 | 41,8 % | 63,9 % | 6,6 % |
| 2025 | 3.672 | 0,038 % | 0,944 | 36,7 % | 62,4 % | 7,4 % |

### Estacional (500 m)

| Grupo | Celdas con fuego | Tasa base | ROC-AUC | Captura 5 % | Captura 10 % | Área para 50 % |
|---|---:|---:|---:|---:|---:|---:|
| Trópico | 6.565 | 0,105 % | 0,930 | 16,8 % | 31,1 % | 17,7 % |
| Región Metropolitana | 236 | 0,026 % | 0,804 | 10,6 % | 15,3 % | 31,0 % |
| Valles, Cono Sur y Andina | 1.396 | 0,013 % | 0,894 | 39,3 % | 51,8 % | 9,1 % |
| Con relieve | 8.197 | 0,045 % | 0,942 | 35,4 % | 59,3 % | 7,8 % |
| Temporada (jul–nov) | 7.782 | 0,108 % | 0,903 | 35,8 % | 60,4 % | 7,6 % |
| Resto del año | 415 | 0,004 % | 0,821 | 27,2 % | 40,0 % | 12,9 % |
| 2024 | 4.525 | 0,053 % | 0,943 | 35,5 % | 59,0 % | 7,8 % |
| 2025 | 3.672 | 0,038 % | 0,940 | 35,2 % | 59,7 % | 7,9 % |

## Importancia por permutación (muestra de prueba)

### Modo operativo

| Grupo de variables | Caída del PR-AUC al barajarlo |
|---|---:|
| Fuego reciente | 0,141 |
| Historial de fuego | 0,080 |
| Época del año | 0,033 |
| Deforestación (Hansen) | 0,023 |
| Cobertura de suelo | 0,023 |
| Clima normal | 0,021 |
| Uso del suelo (MapBiomas) | 0,012 |
| Clima antecedente | 0,009 |
| Población (WorldPop) | 0,001 |
| Relieve | -0,002 |
| Quemas previas (MODIS) | -0,005 |

### Modo estacional

| Grupo de variables | Caída del PR-AUC al barajarlo |
|---|---:|
| Historial de fuego | 0,177 |
| Época del año | 0,073 |
| Deforestación (Hansen) | 0,032 |
| Cobertura de suelo | 0,019 |
| Relieve | 0,008 |
| Uso del suelo (MapBiomas) | 0,004 |
| Población (WorldPop) | -0,000 |
| Clima normal | -0,005 |
| Quemas previas (MODIS) | -0,006 |

## Acierto en el tiempo (prueba)

Correlación semanal entre celdas con fuego observadas y esperadas: modo operativo 0,70; modo estacional 0,66.
