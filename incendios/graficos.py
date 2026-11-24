"""Estilo y piezas comunes de las figuras (matplotlib).

Paleta: la de referencia de la guía de visualización del proyecto, validada con
su script (separación para daltonismo y contraste). Las clases de riesgo son una
escala ordinal, así que usan UN solo tono de claro a oscuro.
"""
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# tinta y superficie
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
FONDO_DEPTO = "#f0efec"          # celdas del departamento sin riesgo apreciable

# series (identidad), en orden fijo
AZUL, NARANJA, AGUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIS_CONTEXTO = "#b5b3ab"
# magnitud (un tono, claro -> oscuro)
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CLASES = ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"]
RAMPA_RIESGO = ["#e99b78", "#d5683b", "#b33a0a", "#7d1d02"]   # Bajo ... Muy alto


def rgba(color, alfa=255):
    r, g, b = (int(round(255 * c)) for c in matplotlib.colors.to_rgb(color))
    return (r, g, b, alfa)


def estilo():
    """Aplica el estilo común: marcas finas, rejilla tenue y sólida, sin marcos."""
    matplotlib.rcParams.update({
        "font.family": ["sans-serif"],
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial", "Helvetica"],
        "font.size": 9.5,
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "text.color": TINTA, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.edgecolor": EJE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": REJILLA, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "xtick.color": TENUE, "ytick.color": TENUE, "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def guardar(fig, ruta):
    import matplotlib.pyplot as plt
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def _anillos(geojson):
    for entidad in geojson["features"]:
        geom = entidad["geometry"]
        poligonos = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poligono in poligonos:
            for anillo in poligono:
                yield np.asarray(anillo, dtype=float)


def contornos(ax, geojson, color=TINTA_2, ancho=0.5, **kw):
    """Dibuja el borde de cada polígono (sin relleno)."""
    for anillo in _anillos(geojson):
        ax.plot(anillo[:, 0], anillo[:, 1], color=color, linewidth=ancho, solid_capstyle="butt",
                zorder=kw.get("zorder", 3))


def extension(malla):
    return (malla.oeste, malla.oeste + malla.ncols * malla.celda,
            malla.norte - malla.nfilas * malla.celda, malla.norte)


def imagen_clases(clases, mascara):
    """Imagen RGBA de las clases de riesgo (0 a 4). Fuera del departamento queda transparente."""
    paleta = np.array([rgba(FONDO_DEPTO)] + [rgba(c) for c in RAMPA_RIESGO], dtype=np.uint8)
    img = paleta[np.clip(np.nan_to_num(clases, nan=0).astype(int), 0, 4)]
    img[~mascara] = (0, 0, 0, 0)
    return img


def preparar_mapa(ax, malla, recorte=None):
    oeste, este, sur, norte = recorte or extension(malla)
    ax.set_xlim(oeste, este)
    ax.set_ylim(sur, norte)
    ax.set_aspect(1.0 / np.cos(np.deg2rad((sur + norte) / 2.0)))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ("left", "bottom"):
        ax.spines[lado].set_visible(False)


def mapa_clases(ax, malla, clases, mascara, recorte=None):
    """Mapa de clases de riesgo sobre la malla; cada celda de 500 m es un píxel nítido."""
    ax.imshow(imagen_clases(clases, mascara), extent=extension(malla), interpolation="nearest",
              origin="upper", zorder=1)
    preparar_mapa(ax, malla, recorte)


def leyenda_clases(ax, **kw):
    piezas = [Patch(facecolor=FONDO_DEPTO, edgecolor=EJE, linewidth=0.5, label=CLASES[0])]
    piezas += [Patch(facecolor=c, label=n) for c, n in zip(RAMPA_RIESGO, CLASES[1:])]
    opciones = {"title": "Riesgo", "loc": "lower left", "fontsize": 8, "title_fontsize": 8.5,
                "handlelength": 1.2, "handleheight": 1.0, "borderaxespad": 0.2}
    opciones.update(kw)
    leyenda = ax.legend(handles=piezas, **opciones)
    leyenda._legend_box.align = "left"
    return leyenda


def marca_focos():
    """Entrada de leyenda para los focos observados."""
    return Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=AZUL,
                  markeredgecolor=SUPERFICIE, markeredgewidth=0.8, label="Foco observado (VIIRS)")


def titulos(ax, titulo, subtitulo=None):
    """Título en negrita (lo que la figura muestra) y, debajo, un subtítulo normal (qué datos son)."""
    if subtitulo:
        ax.set_title(subtitulo, loc="left", fontsize=9.5, fontweight="normal", color=TINTA_2, pad=8)
        desplazamiento = 24
    else:
        ax.set_title("")
        desplazamiento = 8
    return ax.annotate(titulo, xy=(0, 1), xycoords="axes fraction", xytext=(0, desplazamiento),
                       textcoords="offset points", fontsize=11.5, fontweight="bold", color=TINTA,
                       ha="left", va="bottom", annotation_clip=False)
