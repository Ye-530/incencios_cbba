# Riesgo de incendios en Cochabamba a 500 m — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Paquete Python `incendios` + scripts CRISP-DM que producen un mapa semanal de riesgo de incendio a 500 m para Cochabamba, en modos estacional y operativo, evaluado en 2024–2025.

**Architecture:** Malla regular lat/lon de 0,005° alineada con los rásteres. Cada (celda, semana ISO) es una fila; las variables se calculan por semana para toda la malla y se muestrean (caso-control) para entrenar. La evaluación recorre la malla completa semana a semana.

**Tech Stack:** Python 3.10+, numpy, pandas, scipy, scikit-learn (HistGradientBoosting), matplotlib, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-11-18-incendios-cochabamba-design.md`

## Global Constraints

- Dependencias permitidas: numpy, pandas, scipy, scikit-learn, matplotlib, Pillow, joblib. Nada de GDAL/rasterio/geopandas/shapely/lightgbm.
- Malla: `CELDA=0.005`, `OESTE=-67.010`, `NORTE=-15.775`, `NCOLS=556`, `NFILAS=582`. Id de celda = `fila*NCOLS+col`.
- Semana = semana ISO; índice 0 = lunes 2012-01-02.
- Partición temporal por año ISO: entrenar 2015–2021, validar 2022–2023, probar 2024–2025.
- Ninguna variable de la semana `t` puede depender de focos de semanas `≥ t` ni de clima de días `≥` lunes de `t`. En régimen estacional tampoco de focos del mismo año ISO.
- Datos de entrada en `INCENDIOS_DATOS` (por defecto, la carpeta que contiene al proyecto) con `data/` y `descargas/` tal como vienen en el zip del usuario. Salidas en `salidas/`.
- Nombres de código, mensajes y columnas en español; sin latitud/longitud/unidad administrativa como variables del modelo.
- Semilla aleatoria única: 42.

## Review Focus

1. Falta una tesela de DEM o de cobertura → las variables de esa zona quedan en NaN y todo sigue corriendo (prueba en Tarea 3).
2. Focos fuera del recuadro de la malla o justo en el borde → se descartan, nunca dan índice negativo ni "dan la vuelta" (Tarea 1 y 4).
3. Clima con días faltantes, −999 o el artefacto de lluvia → se limpia y los acumulados no propagan NaN (Tarea 6).
4. Semanas con apagón del satélite y años con semana 53 (2015, 2020) → se excluyen o se indexan bien, sin desfasar la recurrencia estacional (Tareas 2, 4 y 7).
5. Fecha pedida fuera del rango con datos (p. ej. 2030, o antes de 2015) → error claro o historial más reciente con aviso; nunca un mapa silenciosamente vacío (Tarea 11).

---

### Task 1: Malla y rasterización de polígonos

**Files:** Create `incendios/__init__.py`, `incendios/config.py`, `incendios/malla.py`, `tests/test_malla.py`, `requirements.txt`, `pytest.ini`

**Interfaces — Produces:**
- `config`: `DIR_DATOS`, `DIR_SALIDAS`, `CELDA`, `OESTE`, `NORTE`, `NCOLS`, `NFILAS`, `ANIOS_ENTRENA=(2015,2021)`, `ANIOS_VALIDA=(2022,2023)`, `ANIOS_PRUEBA=(2024,2025)`, `NEG_POR_SEMANA=6000`, `SEMILLA=42`
- `Malla(oeste, norte, celda, ncols, nfilas)` (dataclass congelada) con `fila_col(lon, lat) -> (filas, cols)` (−1 fuera), `id_celda(lon, lat) -> ids` (−1 fuera), `centros() -> (lon[ncols], lat[nfilas])`, `area_km2() -> (nfilas, ncols)`, `forma`
- `malla_cochabamba() -> Malla`
- `rasterizar_geometria(malla, geom: dict) -> bool[nfilas, ncols]`
- `rasterizar_capas(malla, geojson: dict, campo: str) -> (int16[nfilas, ncols], list[str])`

- [ ] Pruebas: punto en la esquina NO cae en (0,0); punto en el borde este/sur y fuera → −1; polígono con hueco; multipolígono; dos entidades con nombres; área de celda a 17° S ≈ 0,294 km².
- [ ] Ver fallar, implementar con `matplotlib.path.Path`, ver pasar, commit.

### Task 2: Semanas ISO

**Files:** Create `incendios/tiempo.py`, `tests/test_tiempo.py`

**Interfaces — Produces:** `EPOCA=date(2012,1,2)`; `semana_de_fecha(fechas) -> int | ndarray`; `inicio_semana(idx) -> date`; `info_semana(idx) -> dict(anio, semana_anio, inicio, fin, dia_anio)` (año y semana ISO; `dia_anio` del jueves); `semanas_de_anio(anio) -> list[int]`; `semana_de_iso(anio, semana) -> int`

- [ ] Pruebas: 2012-01-02→0, 2012-01-08→0, 2012-01-09→1; 2020-12-31 es (2020, 53); 2021-01-03 sigue en (2020, 53); `semanas_de_anio(2020)` tiene 53 y `2021` 52; ida y vuelta `semana_de_iso`.
- [ ] Ver fallar, implementar, ver pasar, commit.

### Task 3: Rásteres → variables estáticas

**Files:** Create `incendios/rasters.py`, `tests/test_rasters.py`

**Interfaces — Consumes:** `Malla`. **Produces:**
- `leer_geotiff(ruta) -> (arr, oeste, norte, pix)` (Pillow; etiquetas 33550 y 33922)
- `fracciones_por_bloque(arr, factor, clases) -> float32[nf, nc, len(clases)]`
- `terreno_por_bloque(dem, factor, pix, lat_norte) -> dict(elev_media, elev_rango, pendiente, nortitud, estitud)`
- `pegar_en_malla(malla, destino, bloque, oeste, norte)` (lanza `ValueError` si la tesela no está alineada)
- `variables_estaticas(malla, dir_cobertura, dir_terreno) -> dict[str, float32[nfilas, ncols]]` con NaN donde no hay tesela
- `CLASES_WORLDCOVER = {10:'arbol',20:'arbusto',30:'pasto',40:'cultivo',50:'urbano',60:'desnudo',70:'nieve',80:'agua',90:'humedal'}`

- [ ] Pruebas con GeoTIFF sintéticos escritos por la propia prueba: fracciones exactas por bloque; plano inclinado hacia el norte da `nortitud>0` y pendiente esperada; tesela desalineada → `ValueError`; carpeta con una sola tesela → NaN fuera de ella y valores dentro (Review Focus 1).
- [ ] Ver fallar, implementar, ver pasar, commit.

### Task 4: Focos, apagones y positivos semanales

**Files:** Create `incendios/focos.py`, `tests/test_focos.py`

**Interfaces — Produces:**
- `cargar_focos(dir_descargas, malla) -> (df_recuadro, conteo_diario_nacional: Series)`
- `filtrar_vegetacion(df) -> df` (`type==0` y `confidence in {'n','h'}`)
- `dias_sin_satelite(conteo_diario: Series) -> DatetimeIndex` (rachas ≥ 2 días en cero, o día aislado en cero de julio a noviembre)
- `positivos_semanales(df, malla) -> DataFrame[semana, celda, n_det, frp_max]`
- `dias_observados(dias_sin, n_semanas) -> int8[n_semanas]`

- [ ] Pruebas: filtro deja fuera `type 2` y confianza `l`; racha de 2 días se marca, día aislado de febrero no, día aislado de septiembre sí; dos detecciones misma celda y semana → una fila con `n_det=2`; foco fuera del recuadro se descarta (Review Focus 2); semana con 3 días sin satélite → 4 observados (Review Focus 4).
- [ ] Ver fallar, implementar, ver pasar, commit.

### Task 5: Índice meteorológico de incendios (FWI)

**Files:** Create `incendios/fwi.py`, `tests/test_fwi.py`

**Interfaces — Produces:** `ffmc(t, hr, viento_kmh, lluvia, previo)`, `dmc(t, hr, lluvia, previo, mes, lat)`, `dc(t, lluvia, previo, mes, lat)`, `isi(viento_kmh, ffmc)`, `bui(dmc, dc)`, `fwi(isi, bui)`, `serie_fwi(t, hr, viento_kmh, lluvia, meses, lat, inicio=(85.0, 6.0, 15.0)) -> dict de arrays`

- [ ] Pruebas con el caso de referencia de Van Wagner (T=17, HR=42, viento=25, lluvia=0, abril, lat 46° N, previos 85/6/15 → 87,69 / 8,55 / 19,01 / 10,85 / 8,49 / 10,10); lluvia fuerte baja FFMC; para lat −17° el factor de día de DC es 1,4 y el de DMC en junio es 7,8; la serie nunca produce NaN ni negativos.
- [ ] Ver fallar, implementar (ecuaciones de Van Wagner 1987 con factores de latitud de cffdrs), ver pasar, commit.

### Task 6: Clima

**Files:** Create `incendios/clima.py`, `tests/test_clima.py`

**Interfaces — Consumes:** `fwi.serie_fwi`. **Produces:**
- `cargar_clima(ruta) -> DataFrame`
- `limpiar_clima(df) -> (df, informe: dict)` (lluvia de fechas artefacto y > 150 mm → 0 y se cuenta; humedad de suelo en ventana contaminada → NaN; −999 → NaN; huecos de días → reindexado e interpolado)
- `series_unicas(df) -> (diario: DataFrame[serie, fecha, ...], tabla: DataFrame[serie, lat_c, lon_c, provincias])`
- `variables_diarias(diario_serie, lat) -> DataFrame` indexado por fecha con las columnas de `VARS_CLIMA`
- `VARS_CLIMA: list[str]`, `VARS_NORMALES: list[str]`
- `normales(vars_diarias, anio_ini, anio_fin) -> DataFrame[dia_anio]` (media móvil circular de 15 días)
- `pesos_idw(lon, lat, lon_s, lat_s, k=4, potencia=2.0) -> float32[n, n_series]`
- `class ClimaSemanal`: `antecedente(semana) -> float32[n_series, len(VARS_CLIMA)]` (valor del día anterior al lunes), `normal(semana) -> float32[n_series, len(VARS_NORMALES)]`

- [ ] Pruebas: el artefacto se pone en 0 y se informa; `−999` no llega a los acumulados (Review Focus 3); 4 provincias con la misma serie → 1 serie; `antecedente(t)` no cambia si se alteran los datos desde el lunes de `t`; pesos IDW suman 1 y un punto sobre una serie recibe peso ≈ 1; `prec_30` de 30 días con 1 mm = 30.
- [ ] Ver fallar, implementar, ver pasar, commit.

### Task 7: Variables por semana (sin fuga)

**Files:** Create `incendios/variables.py`, `tests/test_variables.py`

**Interfaces — Consumes:** `Malla`, positivos semanales, estáticas, `ClimaSemanal`, pesos IDW. **Produces:**
- `class Constructor(malla, mascara, estaticas, positivos, clima, pesos, n_semanas)`:
  `nombres(regimen) -> list[str]`, `grupos(regimen) -> dict[str, list[str]]`, `etiquetas(semana) -> uint8[n]`, `matriz(semana, regimen) -> float32[n, p]`, `tiene_historial(semana) -> bool` (`n` = celdas del departamento, en orden de `np.flatnonzero(mascara)`)
- `derivadas_estaticas(estaticas, mascara) -> dict` (diversidad, combustible, entorno 11×11, distancias)
- `REGIMENES = ('estacional', 'operativo')`

- [ ] Pruebas de fuga: cambiar focos de semanas ≥ t no altera `matriz(t,'operativo')`; cambiar cualquier foco del mismo año ISO no altera `matriz(t,'estacional')`; un foco en t−1 sí sube `rec_c1`; un foco el año anterior en la misma época sube `est_c3`; semana 53 no desfasa (Review Focus 4); nombres de `estacional` son subconjunto de `operativo` y no contienen lat/lon.
- [ ] Ver fallar, implementar con `scipy.ndimage.uniform_filter`, ver pasar, commit.

### Task 8: Muestreo y corrección del prior

**Files:** Create `incendios/muestreo.py`, `tests/test_muestreo.py`

**Interfaces — Produces:** `muestrear(y, n_neg, rng) -> índices` (todos los positivos + `n_neg` negativos sin reemplazo); `corregir_prior(p, tasa_neg) -> p_real`; `class Calibrador` (isotónica) con `ajustar(p, y)`, `aplicar(p)`

- [ ] Pruebas: incluye todos los positivos y exactamente `n_neg` negativos; con datos sintéticos de prior 1 %, entrenar en muestra 1:1 y corregir devuelve media de probabilidad ≈ 0,01; `corregir_prior` es monótona y deja 0 y 1 fijos.
- [ ] Ver fallar, implementar, ver pasar, commit.

### Task 9: Métricas

**Files:** Create `incendios/metricas.py`, `tests/test_metricas.py`

**Interfaces — Produces:** `captura_por_semana(y, p, semanas, fracciones) -> dict[frac, float]` (focos capturados en el top de cada semana, agregados); `area_para_recall(y, p, semanas, recalls) -> dict`; `precision_top(y, p, semanas, fraccion) -> float`; `dilatar_etiquetas(y_malla) -> y` (3×3); `resumen(y, p, semanas) -> dict` (roc_auc, pr_auc, tasa_base, capturas, áreas)

- [ ] Pruebas: predicción perfecta → captura 100 % con la fracción igual a la tasa base; aleatoria → captura ≈ fracción; semanas sin positivos no cuentan; empates se resuelven de forma estable.
- [ ] Ver fallar, implementar, ver pasar, commit.

### Task 10: Scripts CRISP-DM 01–04 (auditoría, preparación, entrenamiento, evaluación)

**Files:** Create `scripts/01_auditoria_datos.py`, `scripts/02_preparar_datos.py`, `scripts/03_entrenar.py`, `scripts/04_evaluar.py`, `incendios/modelos.py`, `incendios/flujo.py`, `incendios/graficos.py`, `tests/test_flujo.py`

**Interfaces — Produces:** `flujo.cargar_todo(dir_datos, dir_salidas) -> Contexto` (malla, máscara, municipios, constructor, días observados); `modelos.crear(nombre) -> estimador`; `modelos.NOMBRES`; artefactos en `salidas/`: `preparado/*.npz`, `modelos/*.joblib`, `tablas/*.csv`, `figuras/*.png`

- [ ] Prueba de humo `test_flujo.py`: mundo sintético pequeño (malla 40×40, 3 años) recorre preparar → entrenar → evaluar sin errores y el modelo supera al azar.
- [ ] Ejecutar 01–04 con los datos reales; revisar tablas y figuras; commit.

### Task 11: Predicción por fecha y GeoTIFF

**Files:** Create `incendios/geotiff.py`, `incendios/prediccion.py`, `scripts/05_predecir.py`, `tests/test_prediccion.py`

**Interfaces — Produces:** `escribir_geotiff(ruta, arr, malla)`; `clases_de_riesgo(prob, umbrales) -> uint8`; `tabla_top(prob, malla, municipios, n) -> DataFrame[lat, lon, probabilidad, clase, municipio, provincia]`; `predecir(ctx, fecha, modo) -> dict`

- [ ] Pruebas: el GeoTIFF escrito se vuelve a leer con `leer_geotiff` y conserva origen, tamaño de píxel y valores; `clases_de_riesgo` respeta umbrales; fecha anterior a 2015 → `ValueError` con mensaje claro; modo operativo para una semana sin datos previos → `ValueError`; modo estacional para año sin historial cerrado usa el último disponible y lo avisa (Review Focus 5).
- [ ] Ver fallar, implementar, ver pasar, generar mapas de ejemplo, commit.

### Task 12: Visor interactivo

**Files:** Create `scripts/06_generar_visor.py`, `incendios/visor_plantilla.html`

- [ ] Genera `salidas/visor_riesgo.html` autónomo (PNG por semana incrustados, límites municipales, focos observados). Verificar con Playwright que carga, cambia de semana y muestra el detalle de celda.

### Task 13: Datos mejores, informe y paquete

**Files:** Create `scripts/descargar_datos_mejorados.py`, `README.md`, `docs/informe_crisp_dm.md`

- [ ] Script de descarga (teselas faltantes, VIIRS NOAA-20/21, POWER en malla nativa, guía Earth Engine). Solo sintaxis verificable aquí; se marca como no ejecutado.
- [ ] Informe por fases CRISP-DM con las cifras de `salidas/tablas`. Zip final.
