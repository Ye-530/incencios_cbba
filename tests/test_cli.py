"""El comando único: python -m incendios <comando>."""
import importlib
import json
import os
import subprocess
import sys

import pytest

from incendios import __main__ as cli
from incendios.rutas import RAIZ
from tests.test_actualizacion import resumen_eval


def test_sin_comando_muestra_la_ayuda_con_todos_los_comandos(capsys):
    assert cli.main([]) == 0
    texto = capsys.readouterr().out
    assert all(comando in texto for comando in cli.COMANDOS)


def test_un_comando_desconocido_lo_dice_y_muestra_la_ayuda(capsys):
    assert cli.main(["entrena"]) == 2
    assert "entrena" in capsys.readouterr().out


def test_diagnostico_es_descargar_con_diagnostico():
    assert cli.COMANDOS["diagnostico"] == ("descargar", ["--diagnostico"])


def test_cada_comando_apunta_a_un_paso_con_main():
    for modulo, _ in cli.COMANDOS.values():
        assert callable(importlib.import_module(f"incendios.pasos.{modulo}").main)


def test_el_comando_comparar_escribe_la_tabla(tmp_path):
    for nombre, pr_auc in (("antes", 0.10), ("despues", 0.12)):
        (tmp_path / nombre / "tablas").mkdir(parents=True)
        (tmp_path / nombre / "tablas" / "evaluacion_resumen.json").write_text(
            json.dumps(resumen_eval(pr_auc, 0.5)), encoding="utf-8")
    assert cli.main(["comparar", "--antes", str(tmp_path / "antes"), "--despues", str(tmp_path / "despues")]) == 0
    md = (tmp_path / "despues" / "comparacion.md").read_text(encoding="utf-8")
    assert "PR-AUC" in md and "+0.0200" in md


PASOS = sorted(p.stem for p in (RAIZ / "incendios" / "pasos").glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("paso", PASOS)
def test_cada_paso_se_puede_ejecutar_como_archivo(paso, tmp_path):
    """Como hace PyCharm con clic derecho > Run: el archivo directo, con el proyecto en PYTHONPATH."""
    entorno = {**os.environ, "PYTHONPATH": str(RAIZ)}
    r = subprocess.run([sys.executable, str(RAIZ / "incendios" / "pasos" / f"{paso}.py"), "-h"],
                       cwd=tmp_path, env=entorno, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert "usage:" in r.stdout


def test_el_visor_acepta_guardar_la_pagina_sin_cabecera():
    from pathlib import Path
    from incendios.pasos import visor
    assert visor.opciones(["--solo-contenido", "pagina.html"]).solo_contenido == Path("pagina.html")
    assert visor.opciones([]).solo_contenido is None
