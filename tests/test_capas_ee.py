import numpy as np
import pytest

from incendios import capas, capas_ee
from incendios.malla import Malla

MALLA = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=8, nfilas=6)


class ClienteFalso:
    """Devuelve, para cada banda 'variable_año', el año más una décima por variable."""

    def __init__(self, anios):
        self._anios = list(anios)
        self.pedidos = []

    def anios(self, nombre):
        return list(self._anios)

    def pixeles(self, nombre, bandas, malla):
        self.pedidos.append(list(bandas))
        salida = {}
        definicion = capas_ee.CAPAS[nombre]
        for b in bandas:
            if b in definicion["estaticas"]:
                valor = 0.5
            else:
                var, _, anio = b.rpartition("_")
                valor = int(anio) + 0.1 * definicion["anuales"].index(var)
            arr = np.full(malla.forma, valor, dtype=np.float32)
            arr[0, 0] = np.nan                                   # una celda sin dato
            salida[b] = arr
        return salida


def test_la_grilla_de_salida_es_la_malla_del_proyecto():
    g = capas_ee.grilla(MALLA)
    assert g["dimensions"] == {"width": 8, "height": 6}
    assert g["affineTransform"] == {"scaleX": 0.005, "shearX": 0, "translateX": -66.0,
                                    "shearY": 0, "scaleY": -0.005, "translateY": -17.0}
    assert g["crsCode"] == "EPSG:4326"
    assert capas_ee.transformacion(MALLA) == [0.005, 0, -66.0, 0, -0.005, -17.0]


def test_trozos_respetan_el_tamano_maximo():
    assert capas_ee.trozos(list("abcdefg"), 3) == [list("abc"), list("def"), list("g")]
    # en la malla real, ninguna solicitud pasa del límite de bytes
    real = Malla(oeste=-67.01, norte=-15.775, celda=0.005, ncols=556, nfilas=582)
    assert capas_ee.bandas_por_pedido(real, deseadas=50) * 556 * 582 * 4 <= capas_ee.LIMITE_BYTES


def test_convierte_un_arreglo_estructurado_y_marca_sin_dato():
    arr = np.zeros((2, 3), dtype=[("a", "<f4"), ("b", "<f4")])
    arr["a"] = 0.25
    arr["b"] = -1.0
    arr["b"][1, 2] = 7.0
    d = capas_ee.a_diccionario(arr, ["a", "b"])
    assert d["a"].dtype == np.float32 and np.all(d["a"] == 0.25)
    assert np.isnan(d["b"][0, 0]) and d["b"][1, 2] == 7.0


def test_convierte_un_arreglo_comun_de_dos_o_tres_dimensiones():
    tres = np.stack([np.full((2, 3), 1.0), np.full((2, 3), -1.0)], axis=-1)
    d = capas_ee.a_diccionario(tres, ["x", "y"])
    assert np.all(d["x"] == 1.0) and np.isnan(d["y"]).all()
    assert np.all(capas_ee.a_diccionario(np.full((2, 3), 3.0), ["z"])["z"] == 3.0)
    with pytest.raises(ValueError, match="bandas"):
        capas_ee.a_diccionario(tres, ["solo_una"])


def test_descarga_una_capa_por_anio_y_la_guarda(tmp_path):
    cliente = ClienteFalso(range(2001, 2004))
    ruta = capas_ee.descargar("perdida_bosque", MALLA, tmp_path, cliente)
    c = capas.cargar(ruta, MALLA)
    assert c.anios == [2001, 2002, 2003]
    assert c.anual("perdida", 2002)[3, 3] == pytest.approx(2002.0)
    assert np.isnan(c.anual("perdida", 2002)[0, 0])
    assert c.estatica("arboles_2000")[2, 2] == pytest.approx(0.5)
    assert c.meta["activo"] == capas_ee.CAPAS["perdida_bosque"]["activo"]
    assert all(len(p) <= capas_ee.CAPAS["perdida_bosque"]["bandas_por_pedido"] for p in cliente.pedidos)


def test_la_cobertura_trae_cada_grupo_y_la_conversion(tmp_path):
    ruta = capas_ee.descargar("cobertura", MALLA, tmp_path, ClienteFalso([2014]))
    c = capas.cargar(ruta, MALLA)
    assert set(capas_ee.CAPAS["cobertura"]["anuales"]) == {
        "bosque", "natural", "pasto", "agricultura", "mosaico", "no_vegetado", "agua", "conversion3"}
    for i, var in enumerate(capas_ee.CAPAS["cobertura"]["anuales"]):
        assert c.anual(var, 2014)[1, 1] == pytest.approx(2014 + 0.1 * i)


def test_una_segunda_descarga_solo_pide_los_anios_nuevos(tmp_path):
    capas_ee.descargar("quemas", MALLA, tmp_path, ClienteFalso(range(2001, 2004)))
    cliente = ClienteFalso(range(2001, 2006))
    ruta = capas_ee.descargar("quemas", MALLA, tmp_path, cliente)
    pedidas = {b for p in cliente.pedidos for b in p}
    assert pedidas == {"quema_2004", "quema_2005"}
    assert capas.cargar(ruta, MALLA).anios == [2001, 2002, 2003, 2004, 2005]


def test_solo_baja_desde_el_primer_anio_util(tmp_path):
    cliente = ClienteFalso(range(1985, 2025))
    ruta = capas_ee.descargar("cobertura", MALLA, tmp_path, cliente)
    assert capas.cargar(ruta, MALLA).anios[0] == capas_ee.CAPAS["cobertura"]["desde"]


def test_un_archivo_de_otra_malla_se_rechaza(tmp_path):
    ruta = capas_ee.descargar("poblacion", MALLA, tmp_path, ClienteFalso([2015]))
    otra = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=9, nfilas=6)
    with pytest.raises(ValueError, match="malla"):
        capas.cargar(ruta, otra)


def test_guardar_y_cargar_capas_sin_anios(tmp_path):
    ruta = tmp_path / "caminos.npz"
    capas.guardar(ruta, "caminos", MALLA, anios=[], anuales={},
                  estaticas={"dist_via_km": np.ones(MALLA.forma, np.float32)}, meta={"fuente": "OSM"})
    c = capas.cargar(ruta, MALLA)
    assert c.anios == [] and c.meta["fuente"] == "OSM"
    assert c.estatica("dist_via_km").shape == MALLA.forma
