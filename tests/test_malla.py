import numpy as np
import pytest

from incendios.malla import Malla, malla_cochabamba, rasterizar_capas, rasterizar_geometria


@pytest.fixture
def chica():
    # 10 x 10 celdas de 1 grado; esquina noroeste en (lon 0, lat 10)
    return Malla(oeste=0.0, norte=10.0, celda=1.0, ncols=10, nfilas=10)


def cuadrado(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def test_punto_junto_a_la_esquina_noroeste_cae_en_la_celda_0_0(chica):
    filas, cols = chica.fila_col(np.array([0.01]), np.array([9.99]))
    assert (filas[0], cols[0]) == (0, 0)


def test_fila_crece_hacia_el_sur_y_columna_hacia_el_este(chica):
    filas, cols = chica.fila_col(np.array([3.5]), np.array([7.5]))
    assert (filas[0], cols[0]) == (2, 3)


def test_puntos_fuera_o_en_el_borde_este_y_sur_dan_menos_uno(chica):
    lon = np.array([-0.5, 10.0, 5.0, 5.0, 12.0])
    lat = np.array([5.0, 5.0, 0.0, 10.5, -3.0])
    filas, cols = chica.fila_col(lon, lat)
    assert filas.tolist() == [-1] * 5
    assert cols.tolist() == [-1] * 5
    assert chica.id_celda(lon, lat).tolist() == [-1] * 5


def test_id_de_celda_es_fila_por_ncols_mas_columna(chica):
    assert chica.id_celda(np.array([3.5]), np.array([7.5]))[0] == 2 * 10 + 3


def test_centros_estan_a_media_celda_del_borde(chica):
    lon, lat = chica.centros()
    assert lon[0] == pytest.approx(0.5) and lon[-1] == pytest.approx(9.5)
    assert lat[0] == pytest.approx(9.5) and lat[-1] == pytest.approx(0.5)
    assert lon.shape == (10,) and lat.shape == (10,)


def test_malla_de_cochabamba_tiene_las_dimensiones_del_diseno():
    m = malla_cochabamba()
    assert (m.oeste, m.norte, m.celda, m.ncols, m.nfilas) == (-67.010, -15.775, 0.005, 556, 582)
    assert m.forma == (582, 556)


def test_area_de_celda_cerca_de_17_sur_es_0_294_km2():
    m = malla_cochabamba()
    area = m.area_km2()
    fila = m.fila_col(np.array([-66.0]), np.array([-17.0]))[0][0]
    assert area.shape == m.forma
    assert area[fila, 0] == pytest.approx(0.2943, abs=0.0005)


def test_rasterizar_poligono_marca_celdas_con_centro_dentro(chica):
    geom = {"type": "Polygon", "coordinates": [cuadrado(2, 2, 5, 5)]}
    m = rasterizar_geometria(chica, geom)
    assert m.dtype == bool and m.shape == (10, 10)
    assert m.sum() == 9
    # lat 2..5 -> filas 5,6,7 ; lon 2..5 -> cols 2,3,4
    assert m[5:8, 2:5].all()


def test_rasterizar_respeta_huecos(chica):
    geom = {"type": "Polygon", "coordinates": [cuadrado(1, 1, 8, 8), cuadrado(3, 3, 6, 6)]}
    m = rasterizar_geometria(chica, geom)
    assert m.sum() == 49 - 9
    assert not m[5, 4]  # centro (4.5, 4.5) cae en el hueco


def test_rasterizar_multipoligono_une_las_partes(chica):
    geom = {"type": "MultiPolygon", "coordinates": [[cuadrado(0, 0, 2, 2)], [cuadrado(7, 7, 10, 10)]]}
    m = rasterizar_geometria(chica, geom)
    assert m.sum() == 4 + 9


def test_rasterizar_capas_devuelve_id_por_celda_y_nombres(chica):
    gj = {"features": [
        {"properties": {"NAME_3": "Uno"}, "geometry": {"type": "Polygon", "coordinates": [cuadrado(0, 0, 5, 10)]}},
        {"properties": {"NAME_3": "Dos"}, "geometry": {"type": "Polygon", "coordinates": [cuadrado(5, 0, 10, 5)]}},
    ]}
    ids, nombres = rasterizar_capas(chica, gj, "NAME_3")
    assert nombres == ["Uno", "Dos"]
    assert ids.dtype == np.int16
    assert (ids[:, :5] == 0).all()
    assert (ids[5:, 5:] == 1).all()
    assert (ids[:5, 5:] == -1).all()
