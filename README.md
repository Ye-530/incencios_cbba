# Riesgo de incendios en Cochabamba a 500 m

Sistema de machine learning que, dada una fecha, entrega el mapa de riesgo de incendio del
departamento de Cochabamba para esa semana, en celdas de unos 500 m, siguiendo CRISP-DM.

- **Modo estacional:** cualquier fecha futura. Usa relieve, cobertura, historial de focos hasta el
  año anterior y época del año.
- **Modo operativo:** la semana que empieza. Suma el clima y los focos hasta el domingo previo,
  incluido el fuego de viernes a domingo, que es el que más anticipa la semana.
- **Actualización semanal:** baja los datos nuevos, reentrena si cambiaron las fuentes y predice
  la semana; ver [Actualización semanal](#actualización-semanal).

Los resultados medidos están en [`reports/RESULTADOS.md`](reports/RESULTADOS.md). En 2024–2025, años
que el modelo no vio, señalar cada semana el 5 % del territorio con mayor riesgo capturó el 40,6 % de
las celdas con fuego (modo operativo) y el 32,8 % (modo estacional); la recurrencia histórica sola capturó
el 27 % y un modelo por provincia, el 8,5 %. Qué se probó para llegar ahí, y qué no sirvió, está en
[`docs/experimentos-2026-11-25.md`](docs/experimentos-2026-11-25.md).

## Estructura

Sigue la convención de [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/):
los datos van dentro del proyecto y cada cosa tiene su carpeta.

```
incendios_cochabamba/
├── data/
│   ├── raw/            datos originales, tal como llegaron; el proyecto nunca los modifica
│   ├── external/       lo que baja `python -m incendios descargar`
│   ├── interim/        cachés: se pueden borrar y se rehacen solos
│   └── processed/      muestra de entrenamiento y predicciones de validación y prueba
├── models/             modelos entrenados
├── reports/            RESULTADOS.md, tablas, figuras, mapas, visor y predicciones semanales
├── notebooks/          para explorar sin tocar el código
├── incendios/          el paquete: malla, rásteres, focos, clima, variables, modelos, métricas…
│   └── pasos/          un módulo por comando (auditar, preparar, modelar, evaluar, predecir…)
├── tests/              pruebas automáticas
├── docs/               diseño, planes y decisiones tomadas (decisiones.md)
├── credenciales.json   sus claves (no se comparte; está fuera de git)
└── pyproject.toml      dependencias e instalación
```

## Instalación (una vez)

Con Python 3.10 o posterior, desde la carpeta del proyecto:

```
python -m venv .venv
.venv\Scripts\activate             (Windows)
source .venv/bin/activate          (Linux o Mac)
python -m pip install -e ".[descargas]"
python -m incendios diagnostico
```

`pip install -e` instala las dependencias y deja el proyecto disponible desde cualquier carpeta;
`[descargas]` agrega los paquetes para bajar datos mejorados (sin eso: `pip install -e .`).

El diagnóstico dice qué claves, paquetes y datos hay, y qué falta.

- **PowerShell no deja activar el entorno** ("la ejecución de scripts está deshabilitada"): ejecute una
  vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, o no lo active y use su Python directo:
  `.venv\Scripts\python -m incendios diagnostico`.
- **PyCharm:** elija como intérprete `.venv\Scripts\python.exe` (Windows) o `.venv/bin/python`. Para
  ejecutar con el botón Run: *Run > Edit Configurations > + > Python*, cambie *Script* por *Module*,
  escriba `incendios` y, en *Script parameters*, el comando (por ejemplo `diagnostico`). Sin parámetros
  muestra la lista de comandos.

## Comandos

Todo se ejecuta igual en Windows, Linux o Mac: `python -m incendios <comando>`, con el entorno
activado. `python -m incendios` solo muestra la lista; `python -m incendios <comando> -h`, las opciones.
Cada comando es un archivo de `incendios/pasos/` que también se puede ejecutar directo, por ejemplo
`python incendios/pasos/predecir.py --fecha 2027-09-15` o, en PyCharm, clic derecho > Run (con el
proyecto instalado con `pip install -e`). `diagnostico` es `descargar.py` sin opciones.

| Fase de CRISP-DM | Comando | Qué hace | Tiempo aprox.* |
|---|---|---|---|
| — | `diagnostico` | Claves, paquetes y datos disponibles | segundos |
| — | `descargar --que mejorados` | Baja clima ERA5-Land, capas, caminos y focos nuevos | horas la 1.ª vez |
| 2. Comprensión de los datos | `auditar` | Mide la calidad de cada fuente; tablas y figuras | 1 min |
| 3. Preparación | `preparar` | Malla de 500 m, variables y muestra de entrenamiento | 2 min |
| 4. Modelado | `modelar --busqueda --ablacion` | Compara variantes y algoritmos, entrena los modelos finales | 50 min |
| 5. Evaluación | `evaluar` | Métricas sobre la malla completa, figuras y `RESULTADOS.md` | 17 min |
| 3 a 6, de una vez | `entrenar` | `preparar`, `modelar`, `evaluar` y `visor` seguidos | 50 min |
| 6. Despliegue | `predecir --fecha 2027-09-15` | Mapa de la semana de esa fecha | 1 min |
| 6. Despliegue | `visor` | Visor web `reports/visor_riesgo.html` | 2 min |
| Operación | `semana` | Actualización semanal (ver abajo) | minutos |
| Operación | `programar` | Deja la actualización semanal programada cada lunes | segundos |

\* En una PC de 2 núcleos y 8 GB de RAM, medido antes de que los modelos de prueba y los finales
promediaran tres semillas (25-nov-2026). Eso suma ocho GBM a `modelar` y hace más lenta la importancia
por permutación de `evaluar`: cuente con unos 20 a 30 minutos más en `entrenar`. Sin `--busqueda`,
`modelar` usa la variante ya elegida (`models/seleccion.json`) y tardaba unos 37 min; sin `--ablacion`,
unos 28 min.

`models/` ya trae los modelos entrenados: puede ir directo a `predecir`. `evaluar` y `visor` necesitan
las predicciones de prueba que deja `modelar` en `data/processed/` (unos 300 MB que no vienen en el
repositorio): se crean al entrenar. Tampoco viene la muestra de entrenamiento (~1 GB): la arma
`preparar` en unos minutos. Si las predicciones guardadas son de otro modelo o de otros datos, `modelar`
las vuelve a calcular solo, y `visor` lo avisa. Los modelos guardados dependen de la versión de scikit-learn con que se
entrenaron (`models/versiones.json`): si `predecir` avisa o falla, reentrene con `entrenar`.

### Predecir

```
python -m incendios predecir --fecha 2027-09-15                    # estacional
python -m incendios predecir --fecha 2025-10-15 --modo operativo   # necesita datos hasta la víspera
python -m incendios predecir --anio 2027                           # calendario de las 52 semanas
```

Cada predicción escribe en `reports/mapas/`:

- `..._probabilidad.tif` y `..._clase.tif`: GeoTIFF en WGS 84 para abrir en QGIS o ArcGIS.
- `..._top100.csv`: las 100 celdas de mayor riesgo con latitud, longitud, municipio y provincia.
- `....png`: el mapa. `....json`: resumen y avisos.

Clases de riesgo (probabilidad de al menos un foco en la celda durante la semana):

| Clase | Probabilidad semanal |
|---|---|
| Muy alto | 1 % o más |
| Alto | 0,3 % a 1 % |
| Moderado | 0,1 % a 0,3 % |
| Bajo | 0,03 % a 0,1 % |
| Muy bajo | menos de 0,03 % |

La probabilidad media de una celda cualquiera en una semana cualquiera es 0,05 %, y casi ninguna
celda pasa del 5 %: "Muy alto" significa unas 20 veces más riesgo que la media, no que vaya a arder.
`reports/RESULTADOS.md` dice cuántas celdas de cada clase ardieron en los años de prueba.

## Datos

**`data/raw/`** tiene los datos originales (el zip `diplo.zip`), ordenados así:

| Carpeta | Qué tiene | En el zip original |
|---|---|---|
| `data/raw/geo/` | límites del departamento, provincias y municipios | `data/geo/` |
| `data/raw/focos/` | focos VIIRS S-NPP 2012–2025 (`viirs-snpp_AAAA_Bolivia.csv`) | `descargas/` |
| `data/raw/terrain/` | teselas de relieve Copernicus DEM | `data/terrain/` |
| `data/raw/landcover/` | teselas de cobertura ESA WorldCover | `data/landcover/` |
| `data/raw/weather/` | clima NASA POWER por provincia | `data/weather/` |
| `data/raw/fires/` | focos del departamento (solo para la auditoría) | `data/fires/` |

**`data/external/`** recibe todo lo que se baja: focos nuevos (`focos/`), teselas que faltaban
(`terrain/`, `landcover/`), clima ERA5-Land y POWER en malla nativa (`weather/`) y capas de Earth
Engine y caminos (`capas/`). Los focos y las teselas se leen de `raw` y `external` juntos; si un
archivo está en las dos, cuenta el de `external`.

### Datos mejorados

Con las claves de cada servicio, el proyecto baja datos más finos y los usa como variables nuevas:

| Dato | Fuente | Qué agrega al modelo | Clave en `credenciales.json` |
|---|---|---|---|
| Clima de 11 km | ERA5-Land, Copernicus CDS | clima de 0,1° en vez de 0,5°: distingue valles, puna y trópico | `copernicus_cds.key` |
| Cicatrices de quema | MODIS MCD64A1, Earth Engine | qué ardió los años anteriores y hace cuántos años | `earth_engine.proyecto` |
| Deforestación | Hansen GFC v1.12, Earth Engine | pérdida de bosque reciente y acumulada | `earth_engine.proyecto` |
| Uso del suelo anual | MapBiomas Bolivia, Earth Engine | bosque, pasto, agricultura y conversión reciente, año por año | `earth_engine.proyecto` |
| Población | WorldPop, Earth Engine | personas en la celda y en su entorno | `earth_engine.proyecto` |
| Caminos | OpenStreetMap, Overpass | distancia a caminos y sendas, densidad de caminos | no necesita |
| Focos recientes | NASA FIRMS | S-NPP hasta su cierre y NOAA-20 después | `firms.map_key` |

1. Complete `credenciales.json` (si no está, copie `credenciales.ejemplo.json` con ese nombre). Para
   Earth Engine va el **ID del proyecto de Google Cloud** registrado en Earth Engine (por ejemplo
   `pruebas-498417`), no una clave `AIza…`. La MAP_KEY de FIRMS es gratuita:
   https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. `python -m incendios descargar --que mejorados --autenticar`. `--autenticar` abre el navegador una
   vez para iniciar sesión en Google (Earth Engine). El clima ERA5-Land de 2012 a hoy son unos 180
   pedidos mensuales que Copernicus atiende en cola: la primera vez tarda horas. Si se corta, vuelva a
   ejecutarlo y sigue donde quedó.
3. `python -m incendios semana` reentrena solo cuando las fuentes cambiaron: guarda la evaluación
   anterior en `reports/anteriores/AAAA-MM-DD/` y escribe `reports/comparacion.md` con el antes y el
   después.

Partes sueltas: `--que clima_era5`, `capas`, `caminos`, `focos`; sin cuenta, `--que teselas` (relieve
y cobertura que falten: con las teselas ya bajadas a `data/external/`, el relieve cubre todo el
departamento, antes solo el 68 %) y `--que clima` (NASA POWER en su malla nativa).

Cómo entran al modelo, sin usar información del futuro: para predecir un año se usan quemas,
deforestación y uso del suelo **del año anterior o antes**, la población estimada de ese año y los
caminos actuales. Si una capa todavía no llega al año anterior (Hansen y MapBiomas salen con meses de
retraso), se usa su último año y la predicción lo avisa. ERA5-Land se publica con unos 5 días de
retraso, así que el modo operativo usa el clima hasta el martes anterior, igual al entrenar que al
predecir.

Estado actual (25 de noviembre de 2026): las descargas **ya se ejecutaron** y los modelos de `models/`
están entrenados con ellas: clima ERA5-Land 2012–2026 (703 series de 0,1°), las cuatro capas de Earth
Engine hasta 2025 (población hasta 2030), caminos de OpenStreetMap, las teselas de relieve que faltaban
y los focos de NOAA-20 desde el 2 de noviembre de 2026. `models/huella.json` dice exactamente con qué
archivos se entrenó; si `predecir` avisa que los datos cambiaron, reentrene.

### Comparar corridas

`INCENDIOS_DATOS` cambia la carpeta de datos de entrada (la que tiene `raw/` y `external/`) e
`INCENDIOS_SALIDAS` la carpeta donde va todo lo producido. Para comparar con y sin las mejoras:

```
INCENDIOS_CLIMA=power INCENDIOS_CAPAS=0 INCENDIOS_SALIDAS=corrida_base python -m incendios entrenar
python -m incendios entrenar
python -m incendios comparar --antes corrida_base/reports --despues reports
```

En Windows (PowerShell), defina cada variable antes: `$env:INCENDIOS_CAPAS="0"`. `INCENDIOS_CLIMA`
acepta `auto`, `era5` o `power`.

## Actualización semanal

```
python -m incendios semana
```

1. Baja los focos nuevos, el clima ERA5-Land de los últimos meses (sin clave de Copernicus, NASA POWER) y,
   si salió un año nuevo, las capas de Earth Engine. Lo que falle se avisa y se sigue con lo que hay.
2. Si los datos ya no son los del entrenamiento (la primera vez que hay ERA5-Land o capas, por ejemplo),
   reentrena: `preparar`, `modelar` con ablación y `evaluar`. Puede tardar más de una hora; pasa solo
   cuando cambian las fuentes, no cada semana.
3. Predice la semana (lunes a domingo) en modo operativo, o estacional si los datos no llegan, y deja en
   `reports/semanal/AAAA-MM-DD/` el GeoTIFF, el PNG, el CSV con las 100 celdas de mayor riesgo y un resumen.
4. Regenera el visor, que agrega el **seguimiento**: lo que el modelo dijo cada semana del año en curso y,
   cuando la semana ya pasó, los focos que hubo.

Opciones: `--sin-descargar` (solo predecir), `--fecha AAAA-MM-DD` (otra semana), `--no-reentrenar`
(solo avisar), `--registro` (copia todo a `reports/semanal/registro.txt`). Cada corrida agrega una línea
a `reports/semanal/registro.jsonl`.

Para que corra solo cada lunes a las 07:40: `python -m incendios programar`. En Windows crea la tarea
en el Programador de tareas (la PC tiene que estar encendida a esa hora); en Linux o Mac muestra la
línea para `crontab -e`. Necesita haber instalado el proyecto con `pip install -e` (ver Instalación).

**Satélites.** NASA dejó de producir los focos de S-NPP el 2 de noviembre de 2026 a las 13:00 UTC. El
proyecto usa S-NPP hasta el 1 de noviembre y NOAA-20 desde el 2 (mismo instrumento, 375 m, órbita
parecida); la cadena está en `FUENTES_FOCOS`, en `incendios/config.py`. El modelo se entrenó con S-NPP,
y se comprobó que sigue funcionando con NOAA-20: alimentado y juzgado con focos NOAA-20 en 2024–2025
captura lo mismo (39,4 % al 5 %) y reentrenarlo con NOAA-20 desde 2018 gana poco (40,0 %); ver
`docs/experimentos-2026-11-25.md`. Cuando haya unos años de NOAA-20 propio, se puede reentrenar con
NOAA-20 de punta a punta (`descargar --que focos --noaa20-historico` baja la serie desde 2018).

**Claves.** `credenciales.json` está fuera de git y de los informes; no lo comparta ni lo suba a ningún
lado. Las variables de entorno `CDSAPI_KEY`, `EE_PROYECTO` y `FIRMS_MAP_KEY` lo reemplazan si prefiere no
guardar claves en archivos.

## Qué mide y qué no

- La variable objetivo son **focos de calor VIIRS**: incluyen chaqueos y quemas agrícolas, no solo
  incendios forestales. El 80 % está en el trópico (Chapare y Carrasco).
- Que una celda de 500 m "tenga fuego" en una semana depende mucho de qué satélite pasó mientras el
  fuego estaba activo: en 2024–2025, de las celdas-semana donde S-NPP vio fuego, NOAA-20 (el mismo
  instrumento, 50 minutos después) lo vio en la misma celda y semana solo en el 30 %. Eso pone un techo
  bajo al acierto celda por celda, y explica por qué las mejoras del modelo se miden en décimas.
- Se retiran las **fuentes de calor fijas** (hornos, chimeneas, antorchas de gas): celdas con
  detecciones en 10 o más semanas, casi siempre nocturnas. La lista está en
  `reports/tablas/auditoria_fuentes_fijas.csv`; los umbrales, en `incendios/config.py`.
- El modo estacional no sabe si el año será seco o húmedo: acierta dónde y en qué semanas, pero el
  nivel general puede quedar por encima o por debajo según el año.
- El modelo actual usa clima ERA5-Land, con celdas de unos 11 km: sirve para saber **cuándo**, no
  **dónde**, y sigue siendo mucho más grueso que la celda de 500 m. Sin ese archivo, el proyecto vuelve a
  NASA POWER (celdas de unos 55 km).
- La cobertura de suelo de WorldCover es de 2021 (el uso del suelo año por año viene de MapBiomas). El
  relieve cubre todo el departamento desde que se bajaron las teselas que faltaban.
- El fuego en la región depende sobre todo de la actividad humana: en la importancia por permutación el
  clima aporta poco frente al historial de fuego, la época del año y el fuego reciente.

## Pruebas

```
python -m pip install -e ".[pruebas]"
python -m pytest
```

Incluyen las pruebas contra la fuga de información: alteran focos y clima futuros y comprueban que
las variables de la semana predicha no cambian.
