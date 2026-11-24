"""Variables de las capas extra (quemas, deforestación, uso del suelo, población, caminos)."""
import numpy as np
import pytest

from incendios import capas, variables
from incendios.malla import Malla
from tests.test_variables import NC, NF, mundo, sem

MALLA = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=NC, nfilas=NF)
CLASES = ["bosque", "natural", "pasto", "agricultura", "mosaico", "no_vegetado", "agua", "conversion3"]


def anual(nombre, anios, nombres_var, valor, estaticas=None):
    """Capa sintética: cada variable vale `valor(variable, año)` en toda la malla."""
    anuales = {v: np.stack([np.full((NF, NC), valor(v, a), np.float32) for a in anios]) for v in nombres_var}
    return capas.Capa(nombre, list(anios), anuales, estaticas or {}, {"activo": f"prueba/{nombre}"})


def todas(hasta=2018):
    return {
        "quemas": anual("quemas", range(2001, hasta + 1), ["quema"], lambda v, a: (a - 2000) / 100),
        "perdida_bosque": anual("perdida_bosque", range(2001, hasta + 1), ["perdida"], lambda v, a: (a - 2000) / 1000,
                                {"arboles_2000": np.full((NF, NC), 0.7, np.float32)}),
        "cobertura": anual("cobertura", range(2010, hasta + 1), CLASES,
                           lambda v, a: CLASES.index(v) / 10 + (a - 2000) / 1000),
        "poblacion": anual("poblacion", range(2015, 2031), ["poblacion"], lambda v, a: float(a - 2000)),
        "caminos": capas.Capa("caminos", [], {}, {n: np.full((NF, NC), k + 1.0, np.float32) for k, n in
                                                  enumerate(("dist_via_km", "dist_huella_km", "vias_km_km2"))},
                              {"fuente": "OpenStreetMap"}),
    }


def valores(extra, anio):
    return dict(zip(extra.nombres(), extra.mapas(anio)))


def test_grupos_en_orden_fijo_y_un_mapa_por_variable():
    extra = capas.CapasExtra(todas(), MALLA)
    assert list(extra.grupos()) == ["quemas_previas", "deforestacion", "uso_suelo", "poblacion", "caminos"]
    mapas = extra.mapas(2017)
    assert len(mapas) == len(extra.nombres()) == len(set(extra.nombres()))
    assert all(m.shape == (NF, NC) and m.dtype == np.float32 for m in mapas)


def test_solo_entran_los_grupos_de_las_capas_presentes():
    extra = capas.CapasExtra({"caminos": todas()["caminos"]}, MALLA)
    assert list(extra.grupos()) == ["caminos"]
    assert extra.nombres() == ["dist_via_km", "dist_huella_km", "vias_km_km2"]


def test_el_anio_objetivo_usa_las_capas_del_anio_anterior_y_la_poblacion_del_mismo():
    v = valores(capas.CapasExtra(todas(), MALLA), 2017)
    assert v["q_ult"][0, 0] == pytest.approx(0.16)                          # quemas de 2016
    assert v["q_3"][0, 0] == pytest.approx(0.14 + 0.15 + 0.16)              # 2014–2016
    assert v["per_ult"][0, 0] == pytest.approx(0.016)
    assert v["per_acum"][0, 0] == pytest.approx(sum(range(1, 17)) / 1000)   # 2001–2016
    assert v["arb_2000"][0, 0] == pytest.approx(0.7)
    assert v["mb_pasto"][0, 0] == pytest.approx(0.2 + 0.016)                # cobertura de 2016
    assert v["mb_v11_agro"][0, 0] == pytest.approx(0.2 + 0.3 + 0.4 + 3 * 0.016)
    assert v["pob_log"][0, 0] == pytest.approx(np.log1p(17.0))              # población de 2017
    assert v["dist_huella_km"][0, 0] == 2.0


def test_nada_del_anio_objetivo_ni_posterior_cambia_las_variables():
    base = capas.CapasExtra(todas(), MALLA).mapas(2017)
    alterada = todas()
    for nombre, desde in (("quemas", 2017), ("perdida_bosque", 2017), ("cobertura", 2017), ("poblacion", 2018)):
        capa = alterada[nombre]
        for variable, cubo in capa.anuales.items():
            for k, anio in enumerate(capa.anios):
                if anio >= desde:
                    cubo[k] += 5.0
    despues = capas.CapasExtra(alterada, MALLA).mapas(2017)
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(base, despues))
    # y la prueba sí detecta un cambio en el año anterior
    alterada["quemas"].anuales["quema"][alterada["quemas"].anios.index(2016)] += 5.0
    cambiadas = capas.CapasExtra(alterada, MALLA).mapas(2017)
    assert not all(np.array_equal(a, b, equal_nan=True) for a, b in zip(base, cambiadas))


def test_anios_desde_la_ultima_quema():
    def quema(v, a):
        return 0.0
    juego = todas()
    juego["quemas"] = anual("quemas", range(2001, 2019), ["quema"], quema)
    cubo = juego["quemas"].anuales["quema"]
    cubo[juego["quemas"].anios.index(2014), 2, 2] = 0.5
    cubo[juego["quemas"].anios.index(2016), 3, 3] = 0.6
    cubo[juego["quemas"].anios.index(2016), 4, 4] = 0.05                  # apenas rozada: no cuenta
    v = valores(capas.CapasExtra(juego, MALLA), 2017)
    assert v["q_desde"][2, 2] == 3 and v["q_desde"][3, 3] == 1
    assert v["q_desde"][4, 4] == capas.TOPE_ANIOS_SIN_QUEMA == v["q_desde"][8, 8]


def test_capas_desactualizadas_usan_su_ultimo_anio_y_avisan():
    extra = capas.CapasExtra(todas(hasta=2016), MALLA)
    assert extra.avisos(2017) == []
    v = valores(extra, 2019)
    assert v["q_ult"][0, 0] == pytest.approx(0.16)                          # el último que hay: 2016
    avisos = extra.avisos(2019)
    assert any("quemas" in a and "2016" in a and "2018" in a for a in avisos)
    assert not any("poblacion" in a for a in avisos)                        # hay población hasta 2030


def test_sin_anios_previos_el_grupo_queda_vacio_y_avisa():
    juego = todas()
    juego["cobertura"] = anual("cobertura", range(2016, 2019), CLASES, lambda v, a: 0.1)
    extra = capas.CapasExtra(juego, MALLA)
    v = valores(extra, 2016)
    assert all(np.isnan(v[n]).all() for n in extra.grupos()["uso_suelo"])
    assert any("cobertura" in a for a in extra.avisos(2016))


def test_desde_carpeta_lee_solo_las_capas_conocidas(tmp_path):
    assert capas.CapasExtra.desde_carpeta(tmp_path, MALLA) is None
    c = todas()["caminos"]
    capas.guardar(tmp_path / "caminos.npz", "caminos", MALLA, [], {}, c.estaticas, c.meta)
    q = todas()["quemas"]
    capas.guardar(tmp_path / "quemas.npz", "quemas", MALLA, q.anios, q.anuales, {}, q.meta)
    capas.guardar(tmp_path / "otra_cosa.npz", "otra_cosa", MALLA, [], {}, c.estaticas, {})
    extra = capas.CapasExtra.desde_carpeta(tmp_path, MALLA)
    assert list(extra.grupos()) == ["quemas_previas", "caminos"]
    assert extra.resumen()["quemas"]["anios"] == [2001, 2018]
    assert extra.resumen()["caminos"]["anios"] == "fija"


def test_el_constructor_agrega_las_capas_al_final_de_ambos_regimenes():
    extra = capas.CapasExtra(todas(), MALLA)
    sin, con = mundo([]), mundo([], capas=extra)
    semana = sem(2017, 30)
    for regimen in variables.REGIMENES:
        base = sin.nombres(regimen)
        assert con.nombres(regimen) == base + extra.nombres()
        m_sin, m_con = sin.matriz(semana, regimen), con.matriz(semana, regimen)
        assert m_con.shape == (NF * NC, len(con.nombres(regimen))) and m_con.dtype == np.float32
        assert np.array_equal(m_con[:, :len(base)], m_sin, equal_nan=True)
        assert np.allclose(m_con[:, con.nombres(regimen).index("q_ult")], 0.16)
        grupos = con.grupos(regimen)
        assert grupos["caminos"] == ["dist_via_km", "dist_huella_km", "vias_km_km2"]
        assert sorted(n for v in grupos.values() for n in v) == sorted(con.nombres(regimen))
