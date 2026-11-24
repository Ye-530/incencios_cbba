"""Un solo comando para todo el proyecto, en Windows, Linux o Mac:

    python -m incendios <comando> [opciones]

Datos
    diagnostico   qué claves, paquetes y datos hay, y qué falta
    descargar     baja los datos mejorados: --que mejorados | clima_era5 | capas | caminos | focos | teselas | clima

CRISP-DM
    auditar       2. comprensión de los datos: inventario, problemas y figuras
    preparar      3. preparación: variables y muestra de entrenamiento
    modelar       4. modelado: entrena y compara modelos (--ablacion, --busqueda, --rehacer)
    evaluar       5. evaluación: métricas, figuras y reports/RESULTADOS.md
    entrenar      preparar + modelar + evaluar + visor, de una vez
    predecir      6. despliegue: mapa de riesgo de una semana (--fecha) o de un año (--anio)
    visor         arma reports/visor_riesgo.html

Operación
    semana        actualización semanal: baja lo nuevo, reentrena si cambiaron los datos y predice
    comparar      compara dos evaluaciones (antes y después de cambiar los datos)
    programar     deja la actualización semanal programada cada lunes

Ayuda de cada comando:  python -m incendios <comando> -h
"""
import importlib
import sys

# comando -> (módulo en incendios/pasos, argumentos que se agregan al principio)
COMANDOS = {
    "diagnostico": ("descargar", ["--diagnostico"]),
    "descargar": ("descargar", []),
    "auditar": ("auditar", []),
    "preparar": ("preparar", []),
    "modelar": ("modelar", []),
    "evaluar": ("evaluar", []),
    "entrenar": ("entrenar", []),
    "predecir": ("predecir", []),
    "visor": ("visor", []),
    "semana": ("semana", []),
    "comparar": ("comparar", []),
    "programar": ("programar", []),
}


def _consola_utf8():
    # En Windows, al redirigir la salida a un archivo, la consola puede no aceptar acentos o símbolos.
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            try:
                flujo.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv=None):
    _consola_utf8()
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help", "ayuda"):
        print(__doc__)
        return 0
    nombre, resto = argv[0], argv[1:]
    if nombre not in COMANDOS:
        print(f"No existe el comando «{nombre}».\n")
        print(__doc__)
        return 2
    modulo, previos = COMANDOS[nombre]
    paso = importlib.import_module(f"incendios.pasos.{modulo}")
    anterior = sys.argv
    sys.argv = [f"python -m incendios {nombre}", *resto]       # así se ve el nombre en la ayuda de cada paso
    try:
        resultado = paso.main(previos + resto)
    finally:
        sys.argv = anterior
    return resultado if isinstance(resultado, int) else 0


if __name__ == "__main__":
    sys.exit(main())
