import numpy as np
import pytest

from incendios.malla import Malla
from incendios import rasters


def test_leer_geotiff_devuelve_arreglo_origen_y_pixel(tmp_path, tif):
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    tif(tmp_path / "a.tif", arr, oeste=10.0, norte=20.0, pix=0.5)
    leido, oeste, norte, pix = rasters.leer_geotiff(tmp_path / "a.tif")
    assert np.array_equal(leido, arr)
    assert (oeste, norte, pix) == (10.0, 20.0, 0.5)


def test_leer_geotiff_conserva_flotantes(tmp_path, tif):
    arr = np.linspace(0, 5000, 12, dtype=np.float32).reshape(3, 4)
    tif(tmp_path / "dem.tif", arr, oeste=-66.0, norte=-17.0, pix=0.25)
    leido, *_ = rasters.leer_geotiff(tmp_path / "dem.tif")
    assert leido.dtype == np.float32
    assert np.allclose(leido, arr)


def test_fracciones_por_bloque_cuenta_cada_clase():
    arr = np.array([
        [10, 10, 10, 30],
        [10, 10, 30, 30],
        [40, 40, 50, 50],
        [40, 10, 50, 50],
    ], dtype=np.uint8)
    f = rasters.fracciones_por_bloque(arr, factor=2, clases=[10, 30, 40, 50])
    assert f.shape == (2, 2, 4) and f.dtype == np.float32
    assert f[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert f[0, 1].tolist() == [0.25, 0.75, 0.0, 0.0]
    assert f[1, 0].tolist() == [0.25, 0.0, 0.75, 0.0]
    assert f[1, 1].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_terreno_de_un_plano_que_baja_hacia_el_norte():
    pix = 1 / 3600
    metros_por_fila = pix * 110574.0
    subida = metros_por_fila * np.tan(np.deg2rad(10.0))
    filas = np.arange(8, dtype=np.float32)[:, None]
    dem = (1000.0 + subida * filas) * np.ones((8, 8), dtype=np.float32)  # más alto hacia el sur
    t = rasters.terreno_por_bloque(dem, factor=4, pix=pix, lat_norte=-17.0)
    assert set(t) == {"elev_media", "elev_rango", "pendiente", "nortitud", "estitud"}
    assert t["pendiente"].shape == (2, 2)
    assert np.allclose(t["pendiente"], 10.0, atol=0.05)
    assert (t["nortitud"] > 0.15).all()      # mira al norte (ladera soleada en el hemisferio sur)
    assert np.allclose(t["estitud"], 0.0, atol=1e-3)
    assert t["elev_media"][0, 0] == pytest.approx(1000.0 + subida * 1.5, rel=1e-5)
    assert t["elev_rango"][0, 0] == pytest.approx(subida * 3, rel=1e-4)


def test_terreno_plano_no_tiene_pendiente_ni_orientacion():
    t = rasters.terreno_por_bloque(np.full((4, 4), 250.0, np.float32), factor=2, pix=1 / 3600, lat_norte=-17.0)
    assert np.allclose(t["pendiente"], 0.0)
    assert np.allclose(t["nortitud"], 0.0) and np.allclose(t["estitud"], 0.0)


def test_ventana_de_tesela_desalineada_es_error():
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=8, nfilas=6)
    with pytest.raises(ValueError, match="alineada"):
        rasters.ventana_en_malla(malla, oeste=-66.002, norte=-17.0, pix=0.00125, forma=(24, 16))


def test_ventana_de_tesela_que_no_toca_la_malla_es_none():
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=8, nfilas=6)
    assert rasters.ventana_en_malla(malla, oeste=-60.0, norte=-17.0, pix=0.00125, forma=(24, 16)) is None


def test_estaticas_con_teselas_parciales_dejan_nan_donde_no_hay_dato(tmp_path, tif):
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=8, nfilas=6)
    dir_lc, dir_dem = tmp_path / "landcover", tmp_path / "terrain"
    dir_lc.mkdir()
    dir_dem.mkdir()
    # cobertura: solo la mitad oeste (4 columnas de celdas), todo bosque salvo la primera celda (pasto)
    lc = np.full((6 * 4, 4 * 4), 10, dtype=np.uint8)
    lc[:4, :4] = 30
    tif(dir_lc / "lc.tif", lc, oeste=-66.0, norte=-17.0, pix=0.005 / 4)
    # relieve: solo la mitad este, a 3000 m
    dem = np.full((6 * 2, 4 * 2), 3000.0, dtype=np.float32)
    tif(dir_dem / "dem.tif", dem, oeste=-66.0 + 4 * 0.005, norte=-17.0, pix=0.005 / 2)

    est = rasters.variables_estaticas(malla, dir_lc, dir_dem)

    assert est["lc_arbol"].shape == (6, 8) and est["lc_arbol"].dtype == np.float32
    assert est["lc_pasto"][0, 0] == 1.0 and est["lc_arbol"][0, 0] == 0.0
    assert est["lc_arbol"][1, 1] == 1.0
    assert np.isnan(est["lc_arbol"][:, 4:]).all()
    assert np.allclose(est["elev_media"][:, 4:], 3000.0)
    assert np.isnan(est["elev_media"][:, :4]).all()
    assert np.isnan(est["pendiente"][:, :4]).all()


def test_estaticas_sin_ninguna_tesela_son_todo_nan(tmp_path):
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=8, nfilas=6)
    (tmp_path / "v").mkdir()
    est = rasters.variables_estaticas(malla, tmp_path / "v", tmp_path / "no_existe")
    assert all(np.isnan(v).all() for v in est.values())
    assert "lc_cultivo" in est and "elev_media" in est


def test_las_teselas_se_leen_de_varias_carpetas_y_la_ultima_manda(tmp_path, tif):
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=8, nfilas=6)
    originales, bajadas = tmp_path / "raw", tmp_path / "external"
    originales.mkdir()
    bajadas.mkdir()
    # mitad oeste en los datos originales, mitad este bajada después
    tif(originales / "oeste.tif", np.full((12, 8), 1000.0, np.float32), oeste=-66.0, norte=-17.0, pix=0.005 / 2)
    tif(bajadas / "este.tif", np.full((12, 8), 3000.0, np.float32), oeste=-66.0 + 4 * 0.005, norte=-17.0,
        pix=0.005 / 2)
    est = rasters.variables_estaticas(malla, [originales, bajadas], [originales, bajadas])
    assert np.allclose(est["elev_media"][:, :4], 1000.0) and np.allclose(est["elev_media"][:, 4:], 3000.0)
    # la misma tesela en las dos carpetas: cuenta la bajada
    tif(bajadas / "oeste.tif", np.full((12, 8), 2000.0, np.float32), oeste=-66.0, norte=-17.0, pix=0.005 / 2)
    est = rasters.variables_estaticas(malla, [originales, bajadas], [originales, bajadas])
    assert np.allclose(est["elev_media"][:, :4], 2000.0)
