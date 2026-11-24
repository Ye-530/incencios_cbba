import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from incendios import graficos
from incendios.malla import Malla


def test_la_rampa_de_riesgo_tiene_un_color_por_clase_con_fuego_posible():
    assert graficos.CLASES == ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"]
    assert len(graficos.RAMPA_RIESGO) == 4               # "Muy bajo" no se pinta
    assert graficos.RAMPA_RIESGO == ["#e99b78", "#d5683b", "#b33a0a", "#7d1d02"]


def test_imagen_de_clases_pinta_cada_clase_y_deja_transparente_lo_de_afuera():
    clases = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    mascara = np.array([[True, True, True], [True, True, False]])
    img = graficos.imagen_clases(clases, mascara)
    assert img.shape == (2, 3, 4) and img.dtype == np.uint8
    assert img[1, 2, 3] == 0                                  # fuera del departamento: transparente
    assert tuple(img[0, 0]) == graficos.rgba(graficos.FONDO_DEPTO)
    assert tuple(img[0, 1]) == graficos.rgba("#e99b78")
    assert tuple(img[1, 1]) == graficos.rgba("#7d1d02")


def test_contornos_dibuja_un_trazo_por_anillo():
    gj = {"features": [
        {"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]], [[.2, .2], [.4, .2], [.4, .4], [.2, .2]]]}},
        {"geometry": {"type": "MultiPolygon", "coordinates": [[[[2, 2], [3, 2], [3, 3], [2, 2]]], [[[4, 4], [5, 4], [5, 5], [4, 4]]]]}},
    ]}
    fig, ax = plt.subplots()
    graficos.contornos(ax, gj)
    assert len(ax.lines) == 4
    plt.close(fig)


def test_mapa_de_clases_se_guarda_como_png(tmp_path):
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=20, nfilas=10)
    clases = np.random.default_rng(0).integers(0, 5, size=(10, 20)).astype(np.uint8)
    graficos.estilo()
    fig, ax = plt.subplots(figsize=(4, 3))
    graficos.mapa_clases(ax, malla, clases, np.ones((10, 20), bool))
    graficos.leyenda_clases(ax)
    ruta = graficos.guardar(fig, tmp_path / "sub" / "mapa.png")
    assert ruta.exists() and ruta.stat().st_size > 2000
    assert ax.get_xlim() == (-66.0, -65.9) and ax.get_ylim() == (-17.05, -17.0)


def test_estilo_usa_tipografia_sin_serifa_y_rejilla_solida():
    graficos.estilo()
    assert plt.rcParams["font.family"] == ["sans-serif"]
    assert plt.rcParams["grid.linestyle"] in ("-", "solid")
    assert plt.rcParams["axes.spines.top"] is False


def test_titulos_pone_un_titulo_en_negrita_y_un_subtitulo_normal():
    graficos.estilo()
    fig, ax = plt.subplots()
    graficos.titulos(ax, "El hallazgo", "De qué son los datos")
    assert ax.get_title(loc="left") == "De qué son los datos"
    assert ax.title.get_fontweight() == "bold" or ax._left_title.get_fontweight() == "normal"
    textos = [t for t in ax.texts if t.get_text() == "El hallazgo"]
    assert len(textos) == 1 and textos[0].get_fontweight() == "bold"
    plt.close(fig)
