import numpy as np
import pytest

from incendios import caminos, capas
from incendios.malla import KM_POR_GRADO_LON_ECUADOR, Malla

MALLA = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=20, nfilas=10)
C = MALLA.celda


def via(id_, clase, puntos):
    return {"type": "way", "id": id_, "tags": {"highway": clase},
            "geometry": [{"lat": la, "lon": lo} for lo, la in puntos]}


def ancho_celda_km(lat):
    return C * KM_POR_GRADO_LON_ECUADOR * np.cos(np.deg2rad(lat))


def test_la_consulta_pide_caminos_y_huellas_en_el_recuadro():
    q = caminos.consulta((-17.5, -66.0, -17.0, -65.5))
    assert q.startswith("[out:json]") and q.rstrip().endswith("out geom;")
    assert "(-17.5000,-66.0000,-17.0000,-65.5000)" in q          # sur, oeste, norte, este
    for clase in ("primary", "residential", "track", "path"):
        assert clase in q


def test_las_teselas_cubren_toda_la_malla_sin_pasar_de_un_grado():
    real = Malla(oeste=-67.01, norte=-15.775, celda=0.005, ncols=556, nfilas=582)
    t = caminos.teselas(real)
    assert all(n - s <= 1.0 + 1e-9 and e - o <= 1.0 + 1e-9 for s, o, n, e in t)
    lon, lat = real.centros()
    for la in list(lat[::37]) + [lat[-1]]:
        for lo in list(lon[::37]) + [lon[-1]]:
            assert any(s <= la <= n and o <= lo <= e for s, o, n, e in t)


def test_la_respuesta_se_lee_sin_repetidos_y_por_tipo():
    calle = via(1, "primary", [(-65.99, -17.02), (-65.95, -17.02)])
    r1 = {"elements": [calle, via(2, "track", [(-65.99, -17.03), (-65.98, -17.04)]),
                       {"type": "node", "id": 9, "lat": -17.0, "lon": -66.0}]}
    r2 = {"elements": [calle, via(3, "path", [(-65.97, -17.01), (-65.96, -17.01)]),
                       via(4, "construction", [(-65.97, -17.02), (-65.96, -17.02)]),
                       via(5, "secondary", [(-65.97, -17.02)])]}           # un solo punto: no es línea
    lineas = caminos.leer([r1, r2])
    assert {l["id"]: l["tipo"] for l in lineas} == {1: "via", 2: "huella", 3: "huella"}
    assert np.allclose(lineas[0]["lon"], [-65.99, -65.95])


def test_la_longitud_se_reparte_entre_las_celdas_que_cruza():
    lat = MALLA.norte - 4.5 * C                                    # centro de la fila 4
    linea = {"id": 1, "tipo": "via", "lon": np.array([-66.0 + 2.5 * C, -66.0 + 6.5 * C]),
             "lat": np.array([lat, lat])}
    otra = dict(linea, id=2, tipo="huella")
    km = caminos.longitud_por_celda(MALLA, [linea, otra], "via")
    ancho = ancho_celda_km(lat)
    assert km.sum() == pytest.approx(4 * ancho, rel=1e-6)
    assert km[4, 3:6] == pytest.approx([ancho] * 3, rel=1e-6)
    assert km[4, 2] == pytest.approx(ancho / 2, rel=1e-6) and km[4, 6] == pytest.approx(ancho / 2, rel=1e-6)
    assert km.sum() == pytest.approx(km[4].sum())


def test_una_linea_que_sale_de_la_malla_solo_cuenta_lo_de_adentro():
    lat = MALLA.norte - 0.5 * C
    linea = {"id": 1, "tipo": "via", "lon": np.array([-66.0 - 3 * C, -66.0 + 2 * C]), "lat": np.array([lat, lat])}
    km = caminos.longitud_por_celda(MALLA, [linea], "via")
    assert km.sum() == pytest.approx(2 * ancho_celda_km(lat), rel=1e-6)


def test_distancias_y_densidad_de_caminos():
    km_via = np.zeros(MALLA.forma, np.float32)
    km_via[:, 5] = 0.5                                              # un camino norte-sur en la columna 5
    v = caminos.variables_caminos(MALLA, km_via, np.zeros(MALLA.forma, np.float32))
    assert set(v) == set(caminos.VARIABLES)
    lat_media = MALLA.norte - MALLA.nfilas * C / 2
    assert v["dist_via_km"][3, 5] == 0
    assert v["dist_via_km"][3, 8] == pytest.approx(3 * ancho_celda_km(lat_media), rel=1e-4)
    assert np.isnan(v["dist_huella_km"]).all()                     # sin huellas: sin dato, no cero
    assert v["vias_km_km2"][3, 5] > 0 and v["vias_km_km2"][3, 15] == 0


def test_la_densidad_de_una_malla_uniforme_es_km_por_area():
    uniforme = np.full(MALLA.forma, 0.3, np.float32)
    v = caminos.variables_caminos(MALLA, uniforme, uniforme)
    esperado = 0.6 / MALLA.area_km2()
    assert np.allclose(v["vias_km_km2"], esperado, rtol=2e-3)       # también en los bordes
    assert np.all(v["dist_via_km"] == 0) and np.all(v["dist_huella_km"] == 0)


def test_descargar_pide_cada_tesela_una_vez_y_arma_la_capa(tmp_path):
    pedidas = []

    def pedir(q):
        pedidas.append(q)
        return {"elements": [via(1, "primary", [(-65.99, -17.0225), (-65.95, -17.0225)])]}
    ruta = caminos.descargar(MALLA, tmp_path, pedir)
    assert len(pedidas) == len(caminos.teselas(MALLA)) >= 1
    c = capas.cargar(ruta, MALLA)
    assert c.estatica("dist_via_km")[4, 2] == 0 and c.estatica("dist_via_km")[0, 0] > 0
    assert c.meta["fuente"].startswith("OpenStreetMap") and c.meta["vias"] == 1
    caminos.descargar(MALLA, tmp_path, pedir)          # lo ya bajado no se vuelve a pedir
    assert len(pedidas) == len(caminos.teselas(MALLA))


def test_caminos_se_guardan_como_capa_fija(tmp_path):
    variables = {n: np.ones(MALLA.forma, np.float32) for n in caminos.VARIABLES}
    ruta = caminos.guardar(tmp_path, MALLA, variables, {"fuente": "OpenStreetMap"})
    c = capas.cargar(ruta, MALLA)
    assert ruta.name == "caminos.npz" and c.anios == []
    assert set(c.estaticas) == set(caminos.VARIABLES) and c.meta["fuente"] == "OpenStreetMap"
