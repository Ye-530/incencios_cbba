# Resultados

Generado por `python -m incendios evaluar`. Todas las métricas se calculan sobre la malla completa (185.072 celdas por semana, prior real).

- **Captura X %**: de las celdas que tuvieron fuego, qué parte cayó en el X % de celdas de mayor riesgo de su semana.
- **Área para 50 % / 80 %**: qué parte del territorio hay que señalar cada semana para capturar esa parte del fuego.
- **PR-AUC y ROC-AUC**: ordenan todas las celdas-semana a la vez (miden también el acierto en el tiempo).

## Validación 2022–2023: comparación de algoritmos

| Modelo | ROC-AUC | PR-AUC | Captura 1 % | Captura 5 % | Captura 10 % | Área para 50 % | Área para 80 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Operativo (500 m) | 0,936 | 0,017 | 14,1 % | 39,5 % | 58,7 % | 7,5 % | 19,2 % |
| Regresión logística, operativo | 0,922 | 0,016 | 13,2 % | 36,1 % | 57,2 % | 8,1 % | 20,9 % |
| Bosque aleatorio, operativo | 0,939 | 0,016 | 14,6 % | 40,8 % | 61,1 % | 7,0 % | 18,9 % |
| Estacional (500 m) | 0,924 | 0,012 | 10,7 % | 33,0 % | 52,5 % | 9,4 % | 21,9 % |
| Bosque aleatorio, estacional | 0,926 | 0,011 | 10,4 % | 33,9 % | 53,5 % | 9,0 % | 22,2 % |
| Regresión logística, estacional | 0,912 | 0,011 | 10,0 % | 30,3 % | 51,1 % | 9,7 % | 25,2 % |
| Recurrencia histórica | 0,890 | 0,008 | 7,9 % | 25,7 % | 42,6 % | 12,5 % | 29,0 % |
| Por provincia (enfoque anterior) | 0,846 | 0,003 | 1,5 % | 7,5 % | 16,3 % | 29,1 % | 51,6 % |

## Prueba 2024–2025: resultado final

| Modelo | ROC-AUC | PR-AUC | Captura 1 % | Captura 5 % | Captura 10 % | Área para 50 % | Área para 80 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Operativo (500 m) | 0,950 | 0,021 | 14,7 % | 40,6 % | 64,0 % | 6,8 % | 15,8 % |
| Estacional (500 m) | 0,942 | 0,010 | 8,6 % | 32,8 % | 55,9 % | 8,5 % | 17,8 % |
| Recurrencia histórica | 0,913 | 0,006 | 6,2 % | 27,3 % | 48,2 % | 10,4 % | 22,3 % |
| Por provincia (enfoque anterior) | 0,883 | 0,004 | 1,7 % | 8,5 % | 17,2 % | 29,4 % | 46,8 % |

### Precisión y focos nuevos (prueba)

| Modelo | Precisión top 1 % | Precisión top 1 % (±1 celda) | Focos nuevos, captura 5 % | Focos nuevos, captura 10 % |
|---|---:|---:|---:|---:|
| Operativo (500 m) | 0,66 % | 2,74 % | 21,9 % | 42,6 % |
| Estacional (500 m) | 0,39 % | 1,77 % | 26,4 % | 43,4 % |
| Recurrencia histórica | 0,28 % | 1,31 % | 23,8 % | 38,1 % |
| Por provincia (enfoque anterior) | 0,08 % | 0,49 % | 6,6 % | 13,3 % |

Foco nuevo: celda con fuego sin ningún foco a menos de ~2,75 km en las cuatro semanas anteriores.

### Clases de riesgo, modo operativo (prueba)

| Clase | Probabilidad desde | % de celdas-semana | % del fuego | Frecuencia observada |
|---|---:|---:|---:|---:|
| Muy bajo | 0,00 % | 80,166 | 6,4 | 0,004 % |
| Bajo | 0,03 % | 11,768 | 9,8 | 0,038 % |
| Moderado | 0,10 % | 4,206 | 12,3 | 0,133 % |
| Alto | 0,30 % | 2,768 | 32,0 | 0,523 % |
| Muy alto | 1,00 % | 1,092 | 39,5 | 1,633 % |

### Clases de riesgo, modo estacional (prueba)

| Clase | Probabilidad desde | % de celdas-semana | % del fuego | Frecuencia observada |
|---|---:|---:|---:|---:|
| Muy bajo | 0,00 % | 75,516 | 5,5 | 0,003 % |
| Bajo | 0,03 % | 14,540 | 10,4 | 0,032 % |
| Moderado | 0,10 % | 5,369 | 12,2 | 0,103 % |
| Alto | 0,30 % | 3,215 | 34,8 | 0,489 % |
| Muy alto | 1,00 % | 1,360 | 37,2 | 1,235 % |

## Nivel de las probabilidades

Razón entre las celdas con fuego esperadas (suma de probabilidades) y las observadas: 1 es el nivel exacto, más de 1 es sobrestimar. Menor pérdida logarítmica es mejor.

| Años | Modo | Probabilidad | Observadas | Esperadas | Esperadas / observadas | Pérdida logarítmica |
|---|---|---|---:|---:|---:|---:|
| validación | operativo | Del modelo, corregida por el muestreo (la que se usa) | 13.410 | 10.852 | 0,81 | 0,00459 |
| validación | estacional | Del modelo, corregida por el muestreo (la que se usa) | 13.410 | 9.743 | 0,73 | 0,00478 |
| validación | referencia | Tasa base constante | 13.410 | 13.410 | 1,00 | 0,00591 |
| prueba | operativo | Del modelo, corregida por el muestreo (la que se usa) | 8.197 | 9.376 | 1,14 | 0,00291 |
| prueba | operativo | Recalibrada con isotónica sobre 2022–2023 (descartada) | 8.197 | 12.070 | 1,47 | 0,00297 |
| prueba | estacional | Del modelo, corregida por el muestreo (la que se usa) | 8.197 | 11.357 | 1,39 | 0,00304 |
| prueba | estacional | Recalibrada con isotónica sobre 2022–2023 (descartada) | 8.197 | 16.062 | 1,96 | 0,00318 |
| prueba | referencia | Tasa base constante | 8.197 | 8.197 | 1,00 | 0,00393 |

## Prueba por zona, temporada y año

### Operativo (500 m)

| Grupo | Celdas con fuego | Tasa base | ROC-AUC | Captura 5 % | Captura 10 % | Área para 50 % |
|---|---:|---:|---:|---:|---:|---:|
| Trópico | 6.565 | 0,105 % | 0,942 | 24,2 % | 37,3 % | 15,3 % |
| Región Metropolitana | 236 | 0,026 % | 0,811 | 14,4 % | 24,6 % | 31,8 % |
| Valles, Cono Sur y Andina | 1.396 | 0,013 % | 0,906 | 45,3 % | 56,9 % | 6,7 % |
| Con relieve | 8.197 | 0,045 % | 0,950 | 40,6 % | 64,0 % | 6,8 % |
| Temporada (jul–nov) | 7.782 | 0,108 % | 0,917 | 41,0 % | 64,7 % | 6,7 % |
| Resto del año | 415 | 0,004 % | 0,847 | 34,0 % | 50,6 % | 9,9 % |
| 2024 | 4.525 | 0,053 % | 0,951 | 40,5 % | 62,0 % | 7,0 % |
| 2025 | 3.672 | 0,038 % | 0,948 | 40,8 % | 66,5 % | 6,6 % |

### Estacional (500 m)

| Grupo | Celdas con fuego | Tasa base | ROC-AUC | Captura 5 % | Captura 10 % | Área para 50 % |
|---|---:|---:|---:|---:|---:|---:|
| Trópico | 6.565 | 0,105 % | 0,930 | 16,0 % | 29,0 % | 19,4 % |
| Región Metropolitana | 236 | 0,026 % | 0,802 | 9,7 % | 14,4 % | 33,5 % |
| Valles, Cono Sur y Andina | 1.396 | 0,013 % | 0,894 | 40,3 % | 52,7 % | 8,8 % |
| Con relieve | 8.197 | 0,045 % | 0,942 | 32,8 % | 55,9 % | 8,5 % |
| Temporada (jul–nov) | 7.782 | 0,108 % | 0,902 | 33,1 % | 56,7 % | 8,3 % |
| Resto del año | 415 | 0,004 % | 0,823 | 26,5 % | 41,0 % | 12,4 % |
| 2024 | 4.525 | 0,053 % | 0,940 | 28,0 % | 50,0 % | 10,0 % |
| 2025 | 3.672 | 0,038 % | 0,944 | 38,7 % | 63,2 % | 7,1 % |

## Ablación (validación): el modelo operativo sin cada grupo de variables

| Variante | PR-AUC | Cambio PR-AUC | Captura 5 % | Cambio captura 5 % | Captura 10 % |
|---|---:|---:|---:|---:|---:|
| sin_quemas_previas | 0,015 | -0,002 | 38,4 % | -1,1 % | 58,3 % |
| sin_uso_suelo | 0,015 | -0,001 | 39,4 % | -0,1 % | 58,6 % |
| sin_fuego_reciente | 0,016 | -0,001 | 37,5 % | -2,0 % | 57,1 % |
| sin_fuego_dias | 0,016 | -0,001 | 38,7 % | -0,8 % | 57,2 % |
| sin_caminos | 0,016 | -0,001 | 39,5 % | -0,0 % | 58,8 % |
| sin_clima_normal | 0,016 | -0,001 | 40,7 % | 1,1 % | 59,6 % |
| sin_terreno | 0,016 | -0,001 | 38,7 % | -0,9 % | 58,4 % |
| sin_poblacion | 0,017 | -0,000 | 39,1 % | -0,5 % | 59,0 % |
| completo | 0,017 | 0,000 | 39,5 % | 0,0 % | 58,7 % |
| sin_historial | 0,018 | 0,001 | 40,1 % | 0,6 % | 59,3 % |
| sin_deforestacion | 0,018 | 0,001 | 39,0 % | -0,5 % | 59,3 % |
| sin_cobertura | 0,018 | 0,001 | 39,4 % | -0,1 % | 58,6 % |
| sin_epoca | 0,018 | 0,001 | 38,9 % | -0,6 % | 58,0 % |
| sin_clima_antecedente | 0,019 | 0,002 | 39,5 % | 0,0 % | 58,7 % |

## Importancia por permutación (muestra de prueba)

### Modo operativo

| Grupo de variables | Caída del PR-AUC al barajarlo |
|---|---:|
| Fuego reciente | 0,121 |
| Historial de fuego | 0,082 |
| Fuego de los últimos días | 0,069 |
| Época del año | 0,059 |
| Cobertura de suelo | 0,027 |
| Clima normal | 0,023 |
| Deforestación (Hansen) | 0,020 |
| Uso del suelo (MapBiomas) | 0,006 |
| Clima antecedente | 0,004 |
| Población (WorldPop) | 0,002 |
| Caminos (OpenStreetMap) | 0,001 |
| Relieve | 0,001 |
| Quemas previas (MODIS) | -0,002 |

### Modo estacional

| Grupo de variables | Caída del PR-AUC al barajarlo |
|---|---:|
| Historial de fuego | 0,191 |
| Época del año | 0,101 |
| Deforestación (Hansen) | 0,038 |
| Clima normal | 0,017 |
| Cobertura de suelo | 0,012 |
| Caminos (OpenStreetMap) | 0,002 |
| Uso del suelo (MapBiomas) | 0,001 |
| Relieve | 0,001 |
| Población (WorldPop) | -0,002 |
| Quemas previas (MODIS) | -0,009 |

## Acierto en el tiempo (prueba)

Correlación semanal entre celdas con fuego observadas y esperadas: modo operativo 0,75; modo estacional 0,73.
