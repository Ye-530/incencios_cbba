# Decisiones tomadas durante el proyecto

El encargo dejó abiertos muchos puntos ("eso ya velo tú"), así que estas decisiones las tomé yo
mientras construía. Cada una dice qué se decidió, por qué y qué cuesta si resultó un error.

> Desde la versión 2 (18 de noviembre de 2026, decisión 24) el proyecto usa la estructura de
> Cookiecutter Data Science y un solo comando. Donde este texto dice `scripts/0X_...py` o "paso 0X",
> hoy es `python -m incendios auditar | preparar | modelar | evaluar | predecir | visor`; lo que
> estaba en `salidas/` está en `models/`, `reports/`, `data/interim/` y `data/processed/`.

## Preparación de los datos

1. **Se retiran las fuentes de calor fijas.** Regla: celdas con detecciones en 10 o más semanas
   distintas y 80 % o más de detecciones nocturnas. Salen 10 celdas (8 en el departamento: 3 en
   Capinota, 3 al sur de la ciudad de Cochabamba y 2 en Entre Ríos) y 350 detecciones.
   *Por qué:* las primeras celdas del ranking eran hornos o antorchas de gas que FIRMS entrega como
   fuego de vegetación. *Si fue un error:* una celda de vegetación real que cumpliera la regla se
   perdería como caso positivo; es muy improbable (del orden de 1 en 10.000). La lista se calculó con
   los 14 años, incluidos los de prueba: no es una variable del modelo, pero quitó 16 casos de la prueba.
2. **El clima tiene 10 series distintas, no 9.** Las 16 provincias caen en 10 celdas del reanálisis.
   En un mensaje anterior se dijo 9 por error.
3. **Muestra de 4.000 celdas sin fuego por semana, no 6.000.** *Por qué:* la matriz queda en 0,8 GB y
   corre en una PC de 8 GB; hay unas 40 celdas sin fuego por cada una con fuego. *Si fue un error:*
   se sube `NEG_POR_SEMANA` en `incendios/config.py` y se repiten los pasos 02 a 04.

## Modelado

4. **Restricciones de monotonía en el clima y en la actividad de todo el departamento, y clima
   compacto de 6 variables.** *Por qué:* sin ellas el modelo memorizaba combinaciones de clima de años
   concretos (pérdida en validación 0,0926 frente a 0,0877). *Si fue un error:* si una relación real
   no fuera monótona, se pierde esa parte de la señal. La variante libre queda medida en la búsqueda.
5. **Probabilidad sin recalibrar.** El plan pedía una recalibración isotónica. Ajustada con 2022–2023,
   años de mucho fuego, sobrestimó 2024–2025: 1,41 veces en modo operativo y 1,97 en estacional, frente
   a 1,11 y 1,39 sin recalibrar. *Si fue un error:* la decisión se tomó viendo la prueba, así que la
   cifra de nivel de la opción elegida es algo optimista. El orden de las celdas y la captura no cambian.
6. **Umbrales de las clases de riesgo: 0,03 %, 0,1 %, 0,3 % y 1 %.** Los iniciales llegaban a 5 % y
   la clase "Muy alto" quedaba vacía, porque casi ninguna celda supera el 5 % en una semana. Con los
   nuevos, cada clase arde unas tres veces más que la anterior. *Si fue un error:* son una convención
   de presentación fijada mirando la prueba; se cambian en `UMBRALES_CLASES` sin reentrenar.

## Despliegue y entregables

7. **Script de descarga incluido pero no probado.** `scripts/descargar_datos_mejorados.py` baja las
   teselas que faltan, el clima POWER en su malla nativa y los focos de FIRMS. *Por qué:* pediste datos
   de mejor calidad y el entorno donde se construyó no tenía internet. *Si falla:* solo están probadas
   las direcciones y el formato de las tablas; hay que revisar su salida la primera vez.
   *Actualización (25 de noviembre de 2026):* las descargas ya corrieron (decisión 27).
8. **Satélite configurable y huella de los datos.** `PATRON_FOCOS` permite pasar a NOAA-20 sin tocar
   el código, y cada modelo guarda qué datos y qué versión de scikit-learn usó, para avisar si se
   predice con otros. Sin costo.
9. **Informe como documento de Claude y no como archivo del repositorio.** Es lo que se lee y se
   presenta; el repositorio lleva el README y `salidas/RESULTADOS.md` con las cifras. Si se necesita
   dentro del repositorio, el documento se exporta a Markdown o Word.
10. **Sin enchufe para capas extra** (caminos, NDVI de 500 m, cicatrices de quema). Sin esos datos no se
    podía probar; queda como siguiente paso. *Actualización:* caminos, quemas, deforestación, uso del
    suelo y población ya entran (decisiones 12 a 23); el NDVI sigue afuera.
11. **Se construyó sin una aprobación previa del diseño**, porque delegaste esas decisiones. Si algo
    no sirve, se rehace sobre el mismo código.

## Datos mejorados y actualización semanal (noviembre de 2026)

12. **Cicatrices de quema de MODIS MCD64A1 (500 m) y no de MapBiomas Fuego (30 m).** *Por qué:* no se
    pudo confirmar la ruta de MapBiomas Fuego en Earth Engine; MODIS está en el catálogo oficial y llega
    hasta el año pasado. *Si fue un error:* MODIS no ve quemas chicas (de menos de unas 25 ha); cambiar
    de fuente es cambiar una entrada en `incendios/capas_ee.py`.
13. **Sin lluvia CHIRPS (5 km).** ERA5-Land ya trae la lluvia a 11 km. *Costo:* la lluvia sigue siendo
    más gruesa de lo posible.
14. **ERA5-Land con 5 horas por día (00, 10, 13, 16 y 19 UTC) y no las 24.** Divide por cinco el tamaño
    de cada pedido. La máxima del día sale de las 15 h locales y puede quedar 0,5 a 1 °C por debajo de
    la real; el viento a 10 m se guarda como su equivalente a 2 m, que es lo que espera el proyecto.
    *Costo:* un sesgo fijo, igual al entrenar que al predecir.
15. **Retraso de 5 días para el clima ERA5-Land.** El modo operativo usa el clima hasta el lunes menos
    6 días, al entrenar y al predecir, para poder predecir cada lunes con lo ya publicado. *Costo:* el
    modelo no ve los últimos días de clima antes de la semana. Con NASA POWER sigue en 0, como antes.
16. **Años de las capas: el año anterior al predicho para quemas, pérdida de bosque y uso del suelo;
    el mismo año para la población; caminos actuales.** *Costo:* dos fugas chicas que se aceptaron.
    Los caminos de OpenStreetMap son los de hoy, así que al entrenar con 2016 aparecen caminos abiertos
    después. Y MapBiomas suaviza cada año mirando los años vecinos, así que el uso del suelo de un año
    lleva algo del siguiente. La ablación del paso 03 mide cuánto aporta cada grupo.
17. **Si una capa no llega al año anterior, se usa su último año y se avisa.** Hansen y MapBiomas salen
    con meses de retraso. Las variables se calculan como si el año predicho fuera el siguiente al último
    disponible, y cada predicción lo dice. *Costo:* en esos meses, la información de esa capa es vieja.
18. **Las variables nuevas van al final de la matriz, en los dos modos, y la predicción elige las columnas
    por nombre.** Así los modelos entrenados sin capas siguen dando exactamente lo mismo: se verificó con
    los datos reales (diferencia máxima 0,0 en las semanas de prueba). Sin costo.
19. **La actualización semanal reentrena sola cuando cambian las fuentes** (archivo de clima, teselas o
    capas presentes), no cuando llegan focos o años nuevos. Antes guarda la evaluación anterior y después
    escribe la comparación. *Costo:* la primera semana con datos nuevos tarda más de una hora; con
    `--no-reentrenar` solo avisa.
20. **El clima ERA5-Land no se usa hasta que esté completo desde 2012 y sin días faltantes.** Mientras
    tanto el modelo sigue con NASA POWER. *Por qué:* un archivo con huecos pasaría al modelo como datos
    faltantes sin que nadie lo note.
21. **Focos: S-NPP hasta el 1 de noviembre de 2026 y NOAA-20 desde el 2.** NASA dejó de producir S-NPP
    el 2 de noviembre a las 13:00 UTC, así que ese día no tiene la pasada de la tarde. *Costo:* NOAA-20
    detecta parecido pero no igual; el fuego reciente y el seguimiento pueden moverse un poco de nivel.
    Conviene reentrenar con NOAA-20 de punta a punta cuando haya unos años.
22. **Claves en `credenciales.json`, fuera de git y de los informes; la clave `AIza…` no se guardó.**
    Earth Engine desde Python no se autoriza con una clave de API sino con el inicio de sesión de Google
    y el ID de un proyecto de Google Cloud. El zip entregado incluye `credenciales.json` con la clave de
    Copernicus porque se pidió integrarla. *Costo:* quien reciba el zip recibe la clave; como pasó por el
    chat, conviene regenerarla en el perfil de Copernicus y pegar la nueva en el archivo.
23. **Descargas probadas contra servidores simulados, no contra los reales.** El entorno de construcción
    no tenía internet. Están probadas las respuestas esperadas: meses que el CDS no acepta enteros, días
    aún no publicados, cortes y reanudación, focos agregados sin repetir. Los activos de Earth Engine y la
    leyenda de MapBiomas se verificaron en el catálogo, pero no se ejecutaron.

## Estructura del proyecto (versión 2)

24. **Estructura de Cookiecutter Data Science y un solo comando, `python -m incendios`.** Los datos van
    dentro del proyecto: `data/raw` (los originales, que el proyecto nunca modifica), `data/external`
    (lo que se baja), `data/interim` (cachés), `data/processed` (muestra y predicciones), y además
    `models/` y `reports/`. Los scripts pasaron a `incendios/pasos/`, con un `main(argv)` cada uno
    (cada archivo también se ejecuta directo, y `descargar` sin opciones muestra el diagnóstico), y
    los `.bat` se eliminaron: `programar` crea la tarea de Windows con `schtasks`, o muestra la línea
    de cron en Linux o Mac. *Por qué:* con los datos en la carpeta de arriba del proyecto, cada forma
    de descomprimir dejaba los datos en otro lugar; y los `.bat` solo sirven en Windows. *Costo:*
    cambian todos los comandos (el texto de arriba usa los nombres viejos); el zip con los datos pesa
    unos 410 MB; la tarea programada necesita `pip install -e .` para encontrar el proyecto desde
    cualquier carpeta.
25. **Los focos y las teselas se leen de `raw` y `external` juntos; si un archivo está en las dos,
    cuenta el de `external`.** Así lo bajado completa o reemplaza a lo original sin tocarlo. *Costo:*
    un archivo bajado con un error tapa al original del mismo nombre hasta que se lo borre.
26. **Se verificó que la estructura nueva no cambia nada del modelo:** con los datos reales, los
    modelos guardados dan exactamente las mismas predicciones de prueba (diferencia 0,0) y su huella
    coincide, así que la actualización semanal no reentrena por la mudanza.

## Detalles técnicos

- `ventana_en_malla` recorta cada tesela antes de agregarla a la malla, en vez de pegarla entera:
  ahorra memoria con teselas de 1,3 GB y avisa si la tesela no está alineada.
- `dias_observados` recibe el conteo diario de detecciones del país, así descuenta también los días
  fuera del registro.
- La prueba del FWI con lluvia compara con el mismo día sin lluvia en vez de usar un umbral sin fuente;
  los factores de largo del día salen de cffdrs (agosto 8,3; octubre 9,4).
- Las anomalías de clima van dentro del vector de clima antecedente; `ClimaSemanal` dice si hay clima
  para una semana (`disponible`).
- `Constructor.matriz` acepta el año de historial a usar, para pedir un año cuyo historial aún no
  cerró sin devolver un mapa vacío. Una prueba encontró una fuga en la anomalía departamental (leía
  semanas del final del registro) y se corrigió.
- `flujo.py` se escribió antes que su prueba; se retiró, se escribió la prueba, se vio fallar y se
  repuso.
- `clases_de_riesgo` vive en `evaluacion.py` porque la usan la evaluación y la predicción.
- Todos los modelos pasan por `ColumnasVacias`, porque scikit-learn 1.9.1 falla con columnas sin datos
  (relieve ausente en partes del departamento).

## Revisión del 25 de noviembre de 2026

27. **Las descargas ya corrieron y el modelo de `models/` está entrenado con todo.** ERA5-Land completo
    2012–2026 (703 series), quemas MODIS, pérdida de bosque Hansen y MapBiomas hasta 2025, WorldPop
    hasta 2030, caminos de OpenStreetMap, las 12 teselas de relieve y los focos de NOAA-20. El relieve
    pasó de cubrir el 68 % del departamento al 100 %. La auditoría (`auditar`) se volvió a correr con
    estos datos; antes sus tablas y figuras seguían describiendo el estado anterior.
28. **El modelo final es un solo GBM por modo, no un promedio de tres semillas.** Un resumen del proyecto
    decía que se promediaban tres entrenamientos con distinta semilla; el código entrena un
    `HistGradientBoostingClassifier` con `SEMILLA = 42` por régimen (`incendios/pasos/modelar.py`). Las
    cifras de `RESULTADOS.md` corresponden a ese modelo único. Promediar semillas es una mejora posible
    (triplica el tiempo de `modelar`), no algo ya hecho. *Actualización:* desde la decisión 34 los
    modelos de prueba y los finales sí promedian tres semillas.
29. **FIRMS: 5 días por consulta.** El código ya pedía tramos de 5 días; dos pruebas esperaban 10 y
    fallaban. Se dejó 5 (conservador; la documentación de la API ha dicho 10) y las pruebas siguen a la
    constante `MAX_DIAS_FIRMS`.
30. **El modo operativo avisa si los focos no llegan al domingo previo.** Antes, con el registro de
    focos cortado a mitad de la semana anterior, la predicción salía sin aviso aunque el "fuego reciente"
    estuviera incompleto (el aviso de días de satélite solo saltaba con menos de 4 días).
31. **La auditoría entiende ERA5-Land.** Dibujaba celdas MERRA-2 de 0,5° × 0,625° para cualquier clima y
    tenía títulos fijos ("cuatro de cada diez focos caen donde falta el relieve"); ahora usa el tamaño
    de celda de la fuente, títulos calculados y documenta el salto de lluvia de 2020 con el archivo
    POWER aunque el modelo use ERA5-Land.
32. **Los mapas de ejemplo de `reports/mapas/` se regeneraron.** El estacional de 2027 se había generado
    con un modelo que no coincidía con los datos (su resumen lo avisaba).

## Mejoras del modelo (25 de noviembre de 2026)

Las mediciones están en `docs/experimentos-2026-11-25.md`. Antes de decidir se midió el ruido: con solo
cambiar la semilla, la captura al 5 % en validación se mueve hasta 1,3 puntos, y el intervalo al 95 % de
una diferencia entre variantes (remuestreando semanas) es de unos ±3 puntos. Por eso una variante solo
se aceptó si ganaba en dos períodos de validación (2020–2021 y 2022–2023) y con dos semillas.

33. **Fuego de los últimos días (grupo `fuego_dias`, modo operativo).** Celdas con fuego de viernes a
    domingo antes del lunes (en la celda y en 3×3, 11×11 y 41×41 celdas), con fuego el domingo (11×11) y
    el total del departamento de esos tres días, con restricción de monotonía como las otras variables
    departamentales. Mejoró la captura al 1 % y al 10 % en las cuatro corridas y la del 5 % en tres de
    cuatro (en promedio, alrededor de un punto). Va al final de la matriz, así los modelos anteriores
    siguen leyendo sus columnas en el mismo lugar. *Costo:* el modo operativo necesita los focos hasta el
    domingo; si llegan solo hasta el sábado, la predicción lo avisa (decisión 30).
34. **Los modelos de prueba y los finales promedian tres semillas (`config.SEMILLAS_GBM`).** El promedio de
    dos semillas fue igual o mejor que la mejor semilla sola en las cuatro corridas. Con la tercera
    semilla (2026), en validación 2022–2023 el promedio de tres dio lo mismo que el de dos (captura al
    5 % 40,1 % en ambos; PR-AUC 0,0163 y 0,0165) y más que cualquier semilla sola (39,0 a 39,7 %): la
    tercera no suma acierto, suma estabilidad. El número de árboles es el elegido para un modelo solo. La fase de validación y la ablación siguen con una semilla, para que sus comparaciones sean
    parejas y para no triplicar su tiempo. *Costo:* `modelar` entrena cuatro GBM más por régimen. La fila
    "recalibrada (descartada)" de `RESULTADOS.md` ajusta la isotónica con predicciones de validación de un
    modelo solo y la aplica al promedio de tres de la prueba: sirve como referencia, no como comparación
    exacta.
35. **Se descartaron:** ubicación (latitud y longitud), historial extra (años desde el último fuego,
    años con fuego, máximo anual), intensidad del fuego reciente (detecciones, FRP, semanas desde el
    último fuego), suavizado espacial de las probabilidades, mezcla con el bosque aleatorio y usar la
    unión de los satélites S-NPP y NOAA-20. Las cuatro primeras empeoraron la validación; las dos últimas
    no ganaron lo suficiente. La variable de días de satélite de la semana anterior tenía una fuga (su
    valor dependía del lunes de la semana predicha) y no se usa.
36. **Se mantiene la cadena de satélites (S-NPP hasta el 1-nov-2026 y NOAA-20 después).** El modelo
    entrenado con S-NPP, alimentado y juzgado con focos NOAA-20 en 2024–2025, captura 39,4 % al 5 %;
    reentrenado con NOAA-20 desde abril de 2018, 40,0 %. Los dos satélites coinciden solo en el 30 % de
    las celdas-semana con fuego: buena parte del "dónde" de un foco es azar de la pasada.
37. **La huella del modelo incluye las variables y los satélites, y las predicciones guardadas quedan
    atadas a ella.** Una revisión independiente vio que `modelar` reutilizaba modelos y predicciones de
    otra configuración sin avisar si cambiaban las variables. Ahora `modelar` recalcula las predicciones
    cuya huella no coincide (las de versiones anteriores, sin huella, también), y `visor` avisa. Para los
    modelos de versiones anteriores, cuya huella no traía la lista de variables, se compara la lista que
    guarda el propio modelo (las que usa más las excluidas): si el código arma otras, `modelar` y
    `semana` reentrenan, y `predecir` avisa. La predicción también avisa si no hubo satélite de viernes a
    domingo, porque entonces el fuego de los últimos días sale en cero (pasó en 4 de las ~600 semanas).
38. **La muestra de entrenamiento ya no se versiona.** Pesaba 1 GB en Git LFS (la cuota gratuita de
    GitHub es de 1 GB al mes) y `preparar` la rehace en unos minutos. *Costo:* `evaluar` necesita correr
    `preparar` antes (o `entrenar`, que lo hace todo).

## Pendientes menores

- El modelo no usa todavía las coberturas de MapBiomas para recalcular la distancia a cultivos ni a
  zonas urbanas (esas siguen saliendo de WorldCover 2021).
- La búsqueda de variantes del modelo (`modelar --busqueda`) no se repitió con los datos mejorados;
  `models/seleccion.json` conserva la variante `clima_compacto` elegida con NASA POWER. Con el ruido
  medido (decisión 33), sus diferencias habría que confirmarlas en dos períodos.
- El mapa PNG de `predecir` no dibuja el contorno del departamento ni los municipios.
- La importancia por variable usa una sola repetición; la de grupos usa más.
- El agente de usuario de las descargas (`AGENTE`, en `incendios/pasos/descargar.py`) lleva un correo
  personal, como pide la etiqueta de Overpass; cámbielo si el proyecto cambia de responsable.
- `data/external/weather/clima_era5land_diario.csv.gz` pesa 88 MB y crece unos 6 MB por año: GitHub
  rechaza archivos de más de 100 MB, así que hacia 2028 habrá que llevarlo a Git LFS o armarlo desde
  los diarios por año (`era5land/diario/`), que ya están en el repositorio.
- Desde el 2-nov-2026 FIRMS ofrece NOAA-20 y NOAA-21. Usar los dos a la vez no se probó con NOAA-21; la
  unión S-NPP + NOAA-20 no mejoró el modelo (decisión 35).

## Revisión

La revisión final del código la hizo el mismo autor; no hubo un revisor independiente.
Lo mismo vale para la parte de datos mejorados y actualización semanal. Las variables nuevas y el
cambio del 25 de noviembre de 2026 sí pasaron por una revisión independiente: tres revisores (fugas,
errores numéricos, uso operativo) y dos verificadores por hallazgo; lo confirmado se corrigió
(decisiones 35 y 37).
