"""Dónde está cada cosa: estructura de Cookiecutter Data Science dentro del proyecto."""
from incendios.rutas import RAIZ, Rutas


def test_por_defecto_todo_queda_dentro_del_proyecto():
    r = Rutas.del_proyecto(entorno={})
    assert r.raw == RAIZ / "data" / "raw" and r.external == RAIZ / "data" / "external"
    assert r.interim == RAIZ / "data" / "interim" and r.processed == RAIZ / "data" / "processed"
    assert r.modelos == RAIZ / "models" and r.informes == RAIZ / "reports"


def test_las_variables_de_entorno_separan_la_entrada_de_lo_que_se_produce(tmp_path):
    r = Rutas.del_proyecto(entorno={"INCENDIOS_DATOS": str(tmp_path / "d"), "INCENDIOS_SALIDAS": str(tmp_path / "s")})
    assert r.raw == tmp_path / "d" / "raw" and r.external == tmp_path / "d" / "external"
    assert r.interim == tmp_path / "s" / "data" / "interim"          # lo producido va a la otra carpeta
    assert r.modelos == tmp_path / "s" / "models" and r.informes == tmp_path / "s" / "reports"


def test_lo_que_se_baja_va_a_external_y_se_lee_junto_con_raw(tmp_path):
    r = Rutas.en(tmp_path)
    assert r.focos == (tmp_path / "data" / "raw" / "focos", tmp_path / "data" / "external" / "focos")
    assert r.focos_bajados == r.focos[1]
    assert r.relieve == (r.raw / "terrain", r.external / "terrain") and r.relieve_bajado == r.relieve[1]
    assert r.cobertura == (r.raw / "landcover", r.external / "landcover") and r.cobertura_bajada == r.cobertura[1]
    assert r.geo == r.raw / "geo"
    assert r.clima_power.parent == r.raw / "weather"
    assert r.clima_power_nativo.parent == r.clima_era5.parent == r.external / "weather"
    assert r.era5_meses == r.external / "weather" / "era5land"
    assert r.capas == r.external / "capas"


def test_lo_que_se_produce_va_a_interim_processed_models_y_reports(tmp_path):
    r = Rutas.en(tmp_path)
    assert r.estaticas == tmp_path / "data" / "interim" / "estaticas.npz"
    assert r.muestra == tmp_path / "data" / "processed" / "muestra"
    assert r.predicciones == tmp_path / "data" / "processed" / "predicciones"
    assert r.tablas == tmp_path / "reports" / "tablas" and r.figuras == tmp_path / "reports" / "figuras"
    assert r.mapas == tmp_path / "reports" / "mapas" and r.resultados == tmp_path / "reports" / "RESULTADOS.md"
    assert r.visor == tmp_path / "reports" / "visor_riesgo.html"
    assert r.semanal == tmp_path / "reports" / "semanal" and r.anteriores == tmp_path / "reports" / "anteriores"
