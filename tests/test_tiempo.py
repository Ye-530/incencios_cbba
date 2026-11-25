from datetime import date

import pandas as pd
import pytest

from incendios import tiempo


def test_la_epoca_es_un_lunes():
    assert tiempo.EPOCA == date(2012, 1, 2)
    assert tiempo.EPOCA.isoweekday() == 1


@pytest.mark.parametrize("fecha, esperado", [
    ("2012-01-02", 0), ("2012-01-08", 0), ("2012-01-09", 1), ("2012-01-01", -1),
])
def test_semana_de_fecha_cuenta_semanas_de_lunes_a_domingo(fecha, esperado):
    assert tiempo.semana_de_fecha(fecha) == esperado


def test_semana_de_fecha_acepta_series_de_pandas():
    s = pd.to_datetime(pd.Series(["2012-01-02", "2012-01-15", "2012-01-16"]))
    assert tiempo.semana_de_fecha(s).tolist() == [0, 1, 2]
    assert tiempo.semana_de_fecha(pd.DatetimeIndex(s)).tolist() == [0, 1, 2]


def test_inicio_de_semana_es_el_lunes():
    assert tiempo.inicio_semana(0) == date(2012, 1, 2)
    assert tiempo.inicio_semana(1) == date(2012, 1, 9)
    assert tiempo.inicio_semana(tiempo.semana_de_fecha("2024-09-18")).isoweekday() == 1


def test_info_usa_anio_y_semana_iso():
    info = tiempo.info_semana(tiempo.semana_de_fecha("2020-12-31"))
    assert (info["anio"], info["semana_anio"]) == (2020, 53)
    assert info["inicio"] == date(2020, 12, 28) and info["fin"] == date(2021, 1, 3)
    # el domingo 3 de enero de 2021 sigue siendo la semana 53 de 2020
    assert tiempo.semana_de_fecha("2021-01-03") == tiempo.semana_de_fecha("2020-12-31")


def test_dia_del_anio_es_el_del_jueves():
    info = tiempo.info_semana(tiempo.semana_de_fecha("2024-09-16"))  # lunes
    assert info["dia_anio"] == date(2024, 9, 19).timetuple().tm_yday


def test_anios_con_53_y_52_semanas():
    assert len(tiempo.semanas_de_anio(2020)) == 53
    assert len(tiempo.semanas_de_anio(2015)) == 53
    assert len(tiempo.semanas_de_anio(2021)) == 52
    semanas = tiempo.semanas_de_anio(2021)
    assert semanas == list(range(semanas[0], semanas[0] + 52))
    assert tiempo.info_semana(semanas[0])["semana_anio"] == 1


def test_ida_y_vuelta_entre_iso_e_indice():
    for idx in (0, 100, 468, 729):
        info = tiempo.info_semana(idx)
        assert tiempo.semana_de_iso(info["anio"], info["semana_anio"]) == idx


def test_semana_iso_inexistente_es_error():
    with pytest.raises(ValueError):
        tiempo.semana_de_iso(2021, 53)
