"""Programar la actualización semanal sin archivos .bat."""
from incendios.pasos import programar, semana


def test_la_tarea_de_windows_corre_la_semana_con_el_python_del_entorno():
    orden = programar.orden_windows("07:40", "MON", r"C:\Program Files\Py\python.exe")
    assert orden[:2] == ["schtasks", "/Create"] and "/F" in orden
    assert orden[orden.index("/TR") + 1] == '"C:\\Program Files\\Py\\python.exe" -m incendios semana --registro'
    assert orden[orden.index("/TN") + 1] == programar.NOMBRE_TAREA
    assert orden[orden.index("/ST") + 1] == "07:40" and orden[orden.index("/D") + 1] == "MON"


def test_la_linea_de_cron_es_la_misma_tarea():
    assert programar.linea_cron("07:40", "MON", "/home/u/.venv/bin/python") == \
        "40 7 * * 1 /home/u/.venv/bin/python -m incendios semana --registro"


def test_el_registro_copia_la_salida_a_un_archivo(tmp_path, capsys):
    ruta = tmp_path / "semanal" / "registro.txt"
    with semana.registrar_en(ruta):
        print("hola registro")
    assert "hola registro" in capsys.readouterr().out
    assert "hola registro" in ruta.read_text(encoding="utf-8")
