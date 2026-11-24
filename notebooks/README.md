# Notebooks

Para explorar y graficar sin tocar el código del proyecto. Abra Jupyter desde la carpeta del
proyecto (con el entorno activado) y empiece así:

```python
from incendios import flujo, prediccion

ctx = flujo.cargar_todo()                      # datos de data/raw y data/external
r = prediccion.predecir(ctx, "2027-09-15", "estacional")
mapa = ctx.a_malla(r["prob"])                   # probabilidad por celda de 500 m
```

Convención de nombres: `01_exploracion_focos.ipynb`, `02_mapas_por_zona.ipynb`… Lo que se
vuelva parte del flujo pasa al paquete `incendios/`, no se queda en un notebook.
