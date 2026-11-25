import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from incendios import config, geotiff, modelos, prediccion, rasters
from incendios.malla import Malla


# ------------------------------------------------------------------ GeoTIFF
def test_el_geotiff_escrito_se_vuelve_a_leer_con_el_mismo_origen_y_valores(tmp_path):
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=7, nfilas=5)
    arr = np.linspace(0, 1, 35, dtype=np.float32).reshape(5, 7)
    arr[0, 0] = np.nan
    ruta = geotiff.escribir_geotiff(tmp_path / "prob.tif", arr, malla)
    leido, oeste, norte, pix = rasters.leer_geotiff(ruta)
    assert (oeste, norte, pix) == (-66.0, -17.0, 0.005)
    assert leido.dtype == np.float32 and leido.shape == (5, 7)
    assert leido[0, 0] == geotiff.SIN_DATO                      # NaN -> valor "sin dato"
    assert np.allclose(leido.ravel()[1:], arr.ravel()[1:])


def test_el_geotiff_declara_wgs84_y_el_valor_sin_dato(tmp_path):
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=4, nfilas=3)
    ruta = geotiff.escribir_geotiff(tmp_path / "clase.tif", np.zeros((3, 4), np.uint8), malla, sin_dato=255)
    with Image.open(ruta) as im:
        claves = tuple(im.tag_v2[34735])
        assert im.tag_v2[42113].strip("\x00") == "255"
        assert im.mode == "L"
    assert claves[:4] == (1, 1, 0, 4)
    assert (2048, 0, 1, 4326) in [claves[i:i + 4] for i in range(4, len(claves), 4)]     # EPSG:4326


def test_forma_distinta_a_la_malla_es_error(tmp_path):
    malla = Malla(oeste=-66.0, norte=-17.0, celda=0.005, ncols=4, nfilas=3)
    with pytest.raises(ValueError, match="forma"):
        geotiff.escribir_geotiff(tmp_path / "x.tif", np.zeros((3, 5), np.float32), malla)


# ------------------------------------------------------------------ predicción
@pytest.fixture(scope="module")
def modelo_estacional(ctx_sintetico, muestra_sintetica):
    return modelos.entrenar("gbm", "estacional", muestra_sintetica, ctx_sintetico, rapido=True)


@pytest.fixture(scope="module")
def modelo_operativo(ctx_sintetico, muestra_sintetica):
    return modelos.entrenar("gbm", "operativo", muestra_sintetica, ctx_sintetico, rapido=True)


def test_prediccion_estacional_devuelve_probabilidad_y_clase_por_celda(ctx_sintetico, modelo_estacional):
    r = prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo_estacional)
    assert r["prob"].shape == (ctx_sintetico.n,) and r["clase"].shape == (ctx_sintetico.n,)
    assert np.isfinite(r["prob"]).all() and r["prob"].min() >= 0 and r["prob"].max() <= 1
    assert set(np.unique(r["clase"])) <= {0, 1, 2, 3, 4}
    assert str(r["info"]["inicio"]) == "2018-09-10" and str(r["info"]["fin"]) == "2018-09-16"
    assert r["modo"] == "estacional" and r["avisos"] == []


def test_la_temporada_de_fuego_tiene_mas_riesgo_que_la_de_lluvias(ctx_sintetico, modelo_estacional):
    seca = prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo_estacional)["prob"]
    lluvias = prediccion.predecir(ctx_sintetico, "2018-02-14", "estacional", modelo=modelo_estacional)["prob"]
    assert seca.mean() > 5 * lluvias.mean()


def test_fecha_anterior_al_historial_es_un_error_claro(ctx_sintetico, modelo_estacional):
    with pytest.raises(ValueError, match="2015"):
        prediccion.predecir(ctx_sintetico, "2013-09-12", "estacional", modelo=modelo_estacional)


def test_estacional_para_un_anio_sin_historial_cerrado_usa_el_ultimo_y_avisa(ctx_sintetico, modelo_estacional):
    # los datos sintéticos terminan en 2019: 2020 se puede; 2022 no tiene el historial de 2021
    sin_aviso = prediccion.predecir(ctx_sintetico, "2020-09-16", "estacional", modelo=modelo_estacional)
    assert sin_aviso["avisos"] == []
    r = prediccion.predecir(ctx_sintetico, "2022-09-21", "estacional", modelo=modelo_estacional)   # misma semana ISO 38
    assert r["info"]["semana_anio"] == sin_aviso["info"]["semana_anio"] == 38
    assert len(r["avisos"]) == 1 and "2019" in r["avisos"][0] and "historial" in r["avisos"][0]
    assert r["prob"].max() > 0.001                               # no es un mapa vacío
    # misma semana del año y mismo historial: el mapa se parece mucho al de 2020
    assert np.corrcoef(r["prob"], sin_aviso["prob"])[0, 1] > 0.95
    assert r["prob"].mean() == pytest.approx(sin_aviso["prob"].mean(), rel=0.2)


def test_operativo_mas_alla_de_los_datos_es_un_error_que_dice_hasta_donde_hay(ctx_sintetico, modelo_operativo):
    with pytest.raises(ValueError, match="2019-12-31"):
        prediccion.predecir(ctx_sintetico, "2020-09-16", "operativo", modelo=modelo_operativo)


def test_operativo_dentro_de_los_datos_funciona(ctx_sintetico, modelo_operativo):
    r = prediccion.predecir(ctx_sintetico, "2019-09-11", "operativo", modelo=modelo_operativo)
    assert r["prob"].shape == (ctx_sintetico.n,) and r["avisos"] == []


def test_operativo_avisa_si_los_focos_no_llegan_a_la_vispera(ctx_sintetico, modelo_operativo):
    import dataclasses
    # mismos datos, pero el registro de focos dice que termina antes del domingo previo a la semana pedida
    ctx = dataclasses.replace(ctx_sintetico, notas={**ctx_sintetico.notas, "ultima_fecha_focos": "2019-09-05"})
    r = prediccion.predecir(ctx, "2019-09-11", "operativo", modelo=modelo_operativo)       # semana del 09-09
    assert len(r["avisos"]) == 1 and "2019-09-05" in r["avisos"][0] and "incompleta" in r["avisos"][0]
    al_dia = dataclasses.replace(ctx_sintetico, notas={**ctx_sintetico.notas, "ultima_fecha_focos": "2019-09-08"})
    assert prediccion.predecir(al_dia, "2019-09-11", "operativo", modelo=modelo_operativo)["avisos"] == []


def test_modo_desconocido_o_modelo_de_otro_modo_es_error(ctx_sintetico, modelo_estacional):
    with pytest.raises(ValueError, match="modo"):
        prediccion.predecir(ctx_sintetico, "2018-09-12", "mensual", modelo=modelo_estacional)
    with pytest.raises(ValueError, match="modo"):
        prediccion.predecir(ctx_sintetico, "2018-09-12", "operativo", modelo=modelo_estacional)


def test_tabla_de_celdas_de_mayor_riesgo(ctx_sintetico, modelo_estacional):
    r = prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo_estacional)
    t = prediccion.tabla_top(ctx_sintetico, r["prob"], n=25)
    assert list(t.columns) == ["puesto", "latitud", "longitud", "probabilidad", "clase", "municipio", "provincia"]
    assert len(t) == 25 and t["puesto"].tolist() == list(range(1, 26))
    assert (np.diff(t["probabilidad"]) <= 0).all()
    assert set(t["municipio"]) <= set(ctx_sintetico.municipios)
    m = ctx_sintetico.malla
    assert t["longitud"].between(m.oeste, m.oeste + m.ncols * m.celda).all()
    # el punto caliente sintético está en filas 25–34, columnas 5–14
    fila, col = m.fila_col(t["longitud"].to_numpy(), t["latitud"].to_numpy())
    assert ((fila >= 25) & (fila < 35) & (col >= 5) & (col < 15)).mean() > 0.8


def test_guardar_prediccion_escribe_geotiff_csv_png_y_resumen(ctx_sintetico, modelo_estacional, tmp_path):
    r = prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo_estacional)
    rutas = prediccion.guardar_prediccion(ctx_sintetico, r, tmp_path / "pred")
    assert set(rutas) == {"probabilidad_tif", "clase_tif", "top_csv", "mapa_png", "resumen_json"}
    assert all(p.exists() and p.stat().st_size > 0 for p in rutas.values())
    assert rutas["probabilidad_tif"].name == "riesgo_estacional_2018-09-10_probabilidad.tif"
    prob, oeste, norte, pix = rasters.leer_geotiff(rutas["probabilidad_tif"])
    m = ctx_sintetico.malla
    assert prob.shape == m.forma and (oeste, norte, pix) == (m.oeste, m.norte, m.celda)
    assert (prob[~ctx_sintetico.mascara] == geotiff.SIN_DATO).all()
    clase, *_ = rasters.leer_geotiff(rutas["clase_tif"])
    assert set(np.unique(clase)) <= {0, 1, 2, 3, 4, 255}
    assert (clase[~ctx_sintetico.mascara] == 255).all()
    assert len(pd.read_csv(rutas["top_csv"])) == 100
    resumen = json.loads(rutas["resumen_json"].read_text(encoding="utf-8"))
    assert resumen["semana_inicio"] == "2018-09-10" and resumen["modo"] == "estacional"
    assert resumen["umbrales"] == list(config.UMBRALES_CLASES)
    assert sum(resumen["celdas_por_clase"].values()) == ctx_sintetico.n


def test_cargar_modelo_que_no_existe_dice_que_paso_falta(tmp_path):
    with pytest.raises(FileNotFoundError, match="incendios entrenar"):
        prediccion.cargar_modelo(tmp_path, "estacional")


def test_avisa_si_el_modelo_se_entreno_con_otros_datos(ctx_sintetico, muestra_sintetica):
    modelo = modelos.entrenar("gbm", "estacional", muestra_sintetica, ctx_sintetico, rapido=True)
    assert modelo.info["huella"] == ctx_sintetico.notas["huella"]
    assert prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo)["avisos"] == []
    modelo.info["huella"] = dict(modelo.info["huella"], clima="otro_archivo.csv")
    avisos = prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo)["avisos"]
    assert len(avisos) == 1 and "incendios entrenar" in avisos[0]
    # que haya focos más recientes no invalida el modelo
    modelo.info["huella"] = dict(ctx_sintetico.notas["huella"], focos_hasta="2018-12-31")
    assert prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo)["avisos"] == []


def test_un_modelo_que_no_se_puede_abrir_explica_como_resolverlo(tmp_path):
    carpeta = tmp_path / "modelos"
    carpeta.mkdir()
    (carpeta / "final_estacional.joblib").write_bytes(b"esto no es un modelo")
    (carpeta / "versiones.json").write_text('{"scikit-learn": "9.9.9", "python": "3.99"}', encoding="utf-8")
    with pytest.raises(RuntimeError) as error:
        prediccion.cargar_modelo(carpeta, "estacional")
    mensaje = str(error.value)
    assert "9.9.9" in mensaje and "incendios entrenar" in mensaje


def test_guardar_versiones_registra_con_que_se_entreno(tmp_path):
    import sklearn
    ruta = prediccion.guardar_versiones(tmp_path)
    versiones = json.loads(ruta.read_text(encoding="utf-8"))
    assert versiones["scikit-learn"] == sklearn.__version__
    assert set(versiones) >= {"python", "numpy", "scikit-learn", "joblib"}


def test_avisa_si_la_version_de_scikit_learn_no_es_la_del_entrenamiento(ctx_sintetico, modelo_estacional, tmp_path):
    import joblib
    carpeta = tmp_path / "modelos"
    carpeta.mkdir()
    joblib.dump(modelo_estacional, carpeta / "final_estacional.joblib")
    (carpeta / "versiones.json").write_text('{"scikit-learn": "0.0.1"}', encoding="utf-8")
    modelo = prediccion.cargar_modelo(carpeta, "estacional")
    assert "0.0.1" in modelo.aviso_version
    r = prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo)
    assert any("scikit-learn" in a for a in r["avisos"])


def test_el_csv_de_celdas_se_abre_bien_en_excel(ctx_sintetico, modelo_estacional, tmp_path):
    r = prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo_estacional)
    rutas = prediccion.guardar_prediccion(ctx_sintetico, r, tmp_path / "pred")
    assert rutas["top_csv"].read_bytes().startswith(b"\xef\xbb\xbf")        # marca BOM: Excel respeta los acentos


# ------------------------------------------------------------------ capas extra
def test_un_modelo_con_capas_predice_con_ellas_y_avisa_si_estan_viejas(ctx_con_capas, muestra_con_capas):
    for modo in ("estacional", "operativo"):
        modelo = modelos.entrenar("gbm", modo, muestra_con_capas, ctx_con_capas, rapido=True)
        assert "q_ult" in modelo.variables and "dist_via_km" in modelo.variables
        r = prediccion.predecir(ctx_con_capas, "2018-09-12", modo, modelo=modelo)
        assert r["prob"].shape == (ctx_con_capas.n,) and np.isfinite(r["prob"]).all()
        assert r["avisos"] == []
    futuro = prediccion.predecir(ctx_con_capas, "2022-09-14", "estacional",
                                 modelo=modelos.entrenar("gbm", "estacional", muestra_con_capas, ctx_con_capas,
                                                         rapido=True))
    assert any("quemas" in a and "2019" in a and "2021" in a for a in futuro["avisos"])


def test_un_modelo_con_capas_sin_las_capas_da_un_error_claro(ctx_sintetico, ctx_con_capas, muestra_con_capas):
    modelo = modelos.entrenar("gbm", "estacional", muestra_con_capas, ctx_con_capas, rapido=True)
    with pytest.raises(ValueError, match="capas"):
        prediccion.predecir(ctx_sintetico, "2018-09-12", "estacional", modelo=modelo)


def test_un_modelo_sin_capas_predice_igual_aunque_ahora_haya_capas(ctx_sintetico, ctx_con_capas,
                                                                     modelo_estacional, modelo_operativo):
    for modo, modelo in (("estacional", modelo_estacional), ("operativo", modelo_operativo)):
        antes = prediccion.predecir(ctx_sintetico, "2018-09-12", modo, modelo=modelo)
        ahora = prediccion.predecir(ctx_con_capas, "2018-09-12", modo, modelo=modelo)
        assert np.array_equal(antes["prob"], ahora["prob"])
        assert len(ahora["avisos"]) == 1 and "otros archivos" in ahora["avisos"][0]


def test_los_modelos_y_sus_versiones_van_a_la_carpeta_models(tmp_path):
    from incendios.rutas import Rutas
    rutas = Rutas.en(tmp_path)
    ruta = prediccion.guardar_versiones(rutas.modelos)
    assert ruta == tmp_path / "models" / "versiones.json" and "scikit-learn" in ruta.read_text()
    with pytest.raises(FileNotFoundError, match="entrenar"):
        prediccion.cargar_modelo(rutas.modelos, "operativo")


def test_avisa_si_el_codigo_suma_variables_que_el_modelo_no_conoce(ctx_sintetico, muestra_sintetica,
                                                                    modelo_operativo):
    from incendios import variables
    # un modelo entrenado antes de que existiera fuego_dias: sin esas columnas y sin saber que existían
    viejo = modelos.entrenar("gbm", "operativo", muestra_sintetica, ctx_sintetico, rapido=True,
                             excluir=variables.FUEGO_DIAS)
    viejo.info["excluidas"] = []
    avisos = prediccion.predecir(ctx_sintetico, "2019-09-11", "operativo", modelo=viejo)["avisos"]
    assert len(avisos) == 1 and "rec_d" in avisos[0] and "incendios entrenar" in avisos[0]
    assert modelos.variables_cambiaron(viejo, ctx_sintetico)
    assert not modelos.variables_cambiaron(modelo_operativo, ctx_sintetico)


def test_avisa_si_no_hubo_satelite_el_fin_de_semana_anterior(ctx_sintetico, modelo_operativo):
    import dataclasses
    conteo = ctx_sintetico.conteo_diario.copy()
    conteo.loc[pd.Timestamp("2019-09-07"):pd.Timestamp("2019-09-08")] = 0      # sábado y domingo sin pasada
    ctx = dataclasses.replace(ctx_sintetico, conteo_diario=conteo)
    avisos = prediccion.predecir(ctx, "2019-09-11", "operativo", modelo=modelo_operativo)["avisos"]
    assert len(avisos) == 1 and "07/09" in avisos[0] and "08/09" in avisos[0]
    assert prediccion.predecir(ctx_sintetico, "2019-09-11", "operativo", modelo=modelo_operativo)["avisos"] == []
