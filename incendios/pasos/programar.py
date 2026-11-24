"""Deja programada la actualización semanal (cada lunes a las 07:40, salvo que indique otra hora).

    python -m incendios programar                 Windows: crea la tarea. Linux o Mac: muestra la línea de cron.
    python -m incendios programar --hora 06:30
    python -m incendios programar --quitar        Windows: borra la tarea.

La tarea ejecuta `python -m incendios semana --registro` con el mismo Python con que se corre este
comando (el de su entorno .venv), y lo que pasa queda en reports/semanal/registro.txt. Para que la
tarea encuentre el proyecto desde cualquier carpeta, instálelo una vez en su entorno:
pip install -e .  (desde la carpeta del proyecto).
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from incendios.rutas import RAIZ

NOMBRE_TAREA = "Incendios Cochabamba semanal"
DIAS_CRON = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6}


def comando_semana(python=None):
    return [python or sys.executable, "-m", "incendios", "semana", "--registro"]


def orden_windows(hora, dia, python=None):
    """Orden de schtasks que crea (o reemplaza) la tarea semanal."""
    return ["schtasks", "/Create", "/F", "/SC", "WEEKLY", "/D", dia, "/ST", hora, "/TN", NOMBRE_TAREA,
            "/TR", subprocess.list2cmdline(comando_semana(python))]


def linea_cron(hora, dia, python=None):
    horas, minutos = (int(x) for x in hora.split(":"))
    return f"{minutos} {horas} * * {DIAS_CRON[dia]} {shlex.join(comando_semana(python))}"


def proyecto_instalado(python=None):
    """¿Ese Python encuentra este proyecto aunque se lo ejecute desde otra carpeta?"""
    try:
        r = subprocess.run([python or sys.executable, "-c", "import incendios; print(incendios.__file__)"],
                           cwd=Path.home(), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and Path(r.stdout.strip()).resolve().is_relative_to(RAIZ)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hora", default="07:40", help="hora local, HH:MM (por defecto 07:40)")
    ap.add_argument("--dia", default="MON", choices=list(DIAS_CRON), help="día de la semana (por defecto MON)")
    ap.add_argument("--quitar", action="store_true", help="borrar la tarea programada (Windows)")
    args = ap.parse_args(argv)
    if os.name == "nt" and args.quitar:
        r = subprocess.run(["schtasks", "/Delete", "/F", "/TN", NOMBRE_TAREA])
        return r.returncode
    if not proyecto_instalado():
        print("Antes de programar, instale el proyecto en su entorno (una sola vez), desde la carpeta del "
              "proyecto:\n    python -m pip install -e .\ny vuelva a ejecutar: python -m incendios programar")
        return 1
    if os.name == "nt":
        r = subprocess.run(orden_windows(args.hora, args.dia))
        if r.returncode == 0:
            print(f"Listo: la tarea «{NOMBRE_TAREA}» corre cada {args.dia} a las {args.hora}.\n"
                  "La PC tiene que estar encendida a esa hora. Lo que pase queda en reports\\semanal\\registro.txt")
        return r.returncode
    print("En Linux o Mac, agregue esta línea con `crontab -e`:\n")
    print(f"    {linea_cron(args.hora, args.dia)}\n")
    print("Lo que pase queda en reports/semanal/registro.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
