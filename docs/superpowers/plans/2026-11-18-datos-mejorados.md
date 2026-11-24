# Datos mejorados y actualización semanal — plan

**Objetivo:** con las claves del usuario, bajar clima ERA5-Land (Copernicus CDS), área quemada MODIS,
pérdida de bosque Hansen, cobertura MapBiomas y población WorldPop (Earth Engine) y caminos (OSM
Overpass); que el modelo los use como variables sin fuga; y actualizar + predecir cada semana.

**Restricción:** el entorno de construcción no tiene internet. Todo se prueba con datos simulados;
la descarga real y el reentrenamiento corren en la PC del usuario.

## Reglas fijas
- Claves solo en `credenciales.json` (fuera de git); nunca en código, informes ni el documento del proyecto.
- Sin capas en `data/capas/`, el modelo y las matrices quedan idénticos a hoy (compatibilidad).
- Capas por año: el año objetivo Y usa quemas, pérdida y cobertura de Y-1 o antes; población de Y; caminos fijos.
- ERA5-Land: retraso de publicación de 5 días; el modo operativo usa clima hasta lunes-1-retraso, igual al entrenar y al predecir.
- Fuente POWER sin cambios de comportamiento (correcciones de 2020, ajuste a la malla MERRA, retraso 0).

## Tareas
- [x] 1. `incendios/credenciales.py`, `credenciales.ejemplo.json`, `credenciales.json`, `.gitignore`. Pruebas: `tests/test_credenciales.py`.
- [x] 2. `clima.py`: fuente (columna `fuente`), sin correcciones POWER ni ajuste MERRA para ERA5, `retraso` en `ClimaSemanal`; `flujo.archivo_clima` prefiere `clima_era5land_diario.csv.gz`; `INCENDIOS_CLIMA` fuerza la fuente.
- [x] 3. `incendios/era5.py`: solicitudes mensuales (5 horas UTC: 00,10,13,16,19; 7 variables), lectura NetCDF (zip, netCDF4 o scipy), diario por fecha local (UTC-4), puntos útiles, tabla en el esquema del proyecto.
- [x] 4. `incendios/capas_ee.py` (cliente EE inyectable, malla de salida = malla del proyecto, computePixels por trozos), `incendios/capas.py` (archivo `.npz` por capa).
- [x] 5. `incendios/caminos.py`: consulta Overpass por teselas, lectura de `out geom`, rasterizado, distancias y densidad.
- [x] 6. Variables: `capas.CapasExtra.bloque(anio)`, grupos nuevos, `Constructor(capas=...)`, `INCENDIOS_CAPAS=0` las apaga, huella incluye capas; guardas en muestra y predicción.
- [x] 7. Focos encadenados S-NPP → NOAA-20 (`config.FUENTES_FOCOS`).
- [x] 8. Scripts: `descargar_datos_mejorados.py` (`--que clima_era5|capas|caminos|...`, `--diagnostico`), `actualizar_semana.py`, `programar_semanal.bat`.
- [x] 9. README, `requirements-descargas.txt`, `docs/decisiones.md`, informe, suite completa, paquete.
