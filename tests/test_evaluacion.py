import numpy as np
import pandas as pd
import pytest

from incendios import evaluacion


@pytest.fixture
def carpeta(tmp_path):
    """10 semanas × 400 celdas; el modelo 'bueno' ordena bien, 'malo' al azar."""
    rng = np.random.default_rng(0)
    n_sem, n = 10, 400
    riesgo = np.tile(np.linspace(0.0005, 0.25, n), (n_sem, 1))      # riesgo real por celda
    y = (rng.random((n_sem, n)) < riesgo).astype(np.uint8)
    np.save(tmp_path / "y.npy", y)
    np.save(tmp_path / "y_tol.npy", np.maximum(y, np.roll(y, 1, axis=1)))
    nuevo = np.zeros((n_sem, n), dtype=bool)
    nuevo[:, :200] = True
    np.save(tmp_path / "nuevo.npy", nuevo)
    np.save(tmp_path / "semanas.npy", np.arange(600, 600 + n_sem, dtype=np.int32))
    np.save(tmp_path / "pred_bueno.npy", riesgo.astype(np.float32))
    np.save(tmp_path / "pred_malo.npy", rng.random((n_sem, n)).astype(np.float32))
    return tmp_path


def test_cargar_predicciones_encuentra_los_modelos(carpeta):
    p = evaluacion.cargar_predicciones(carpeta)
    assert sorted(p["pred"]) == ["bueno", "malo"]
    assert p["y"].shape == (10, 400) and p["semanas"].tolist() == list(range(600, 610))


def test_cargar_predicciones_sin_archivos_dice_que_paso_falta(tmp_path):
    with pytest.raises(FileNotFoundError, match="incendios entrenar"):
        evaluacion.cargar_predicciones(tmp_path)


def test_tabla_de_modelos_ordena_del_mejor_al_peor(carpeta):
    t = evaluacion.tabla_modelos(evaluacion.cargar_predicciones(carpeta))
    assert t.index.tolist() == ["bueno", "malo"]
    # con riesgo lineal entre celdas, el 10 % más alto concentra 1 - 0,9² = 19 % del fuego
    assert t.loc["bueno", "captura_10"] == pytest.approx(0.19, abs=0.03)
    assert t.loc["malo", "captura_10"] == pytest.approx(0.10, abs=0.03)
    assert {"roc_auc", "pr_auc", "area_50", "area_80", "precision_1", "precision_1_tol", "captura_nuevos_5"} <= set(t.columns)
    assert t.loc["bueno", "precision_1_tol"] >= t.loc["bueno", "precision_1"]


def test_filtrar_por_celdas_y_semanas(carpeta):
    p = evaluacion.cargar_predicciones(carpeta)
    sub = evaluacion.filtrar(p, celdas=np.arange(400) >= 300, semanas=np.arange(10) < 4)
    assert sub["y"].shape == (4, 100)
    assert sub["pred"]["bueno"].shape == (4, 100)
    assert sub["semanas"].tolist() == [600, 601, 602, 603]


def test_tabla_por_grupo_calcula_cada_zona_por_separado(carpeta):
    p = evaluacion.cargar_predicciones(carpeta)
    zona = (np.arange(400) >= 200).astype(int)
    t = evaluacion.tabla_por_grupo(p, "bueno", zona, ["baja", "alta"])
    assert t.index.tolist() == ["baja", "alta"]
    assert t.loc["alta", "tasa_base"] > t.loc["baja", "tasa_base"]
    assert t.loc["alta", "n_pos"] + t.loc["baja", "n_pos"] == int(p["y"].sum())


def test_calibracion_compara_lo_predicho_con_lo_observado(carpeta):
    p = evaluacion.cargar_predicciones(carpeta)
    t = evaluacion.calibracion(p["pred"]["bueno"], p["y"], bordes=[0, 0.05, 0.15, 1.0])
    assert list(t.columns) == ["desde", "hasta", "n", "predicha", "observada"]
    assert len(t) == 3 and t["n"].sum() == 4000
    assert np.allclose(t["predicha"], t["observada"], atol=0.03)       # el modelo "bueno" es el riesgo real


def test_serie_semanal_suma_esperados_y_observados(carpeta):
    p = evaluacion.cargar_predicciones(carpeta)
    t = evaluacion.serie_semanal(p["pred"]["bueno"], p["y"], p["semanas"])
    assert list(t.columns) == ["semana", "inicio", "observadas", "esperadas"]
    assert len(t) == 10
    assert t["observadas"].sum() == int(p["y"].sum())
    assert t["esperadas"].iloc[0] == pytest.approx(float(p["pred"]["bueno"][0].sum()), rel=1e-5)
    assert str(t["inicio"].iloc[0]) == "2023-07-03"


def test_tabla_de_clases_reparte_celdas_y_focos(carpeta):
    p = evaluacion.cargar_predicciones(carpeta)
    t = evaluacion.tabla_clases(p["pred"]["bueno"], p["y"], umbrales=[0.01, 0.05, 0.10, 0.20])
    assert t["clase"].tolist() == ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"]
    assert t["pct_celdas"].sum() == pytest.approx(100.0)
    assert t["pct_focos"].sum() == pytest.approx(100.0)
    assert (np.diff(t["frecuencia_observada"]) > 0).all()             # más riesgo, más fuego
    assert t["umbral_inferior"].tolist() == [0.0, 0.01, 0.05, 0.10, 0.20]


def test_clases_de_riesgo_respetan_los_umbrales():
    p = np.array([0.0, 0.0099, 0.01, 0.07, 0.2, 0.9, np.nan])
    clases = evaluacion.clases_de_riesgo(p, [0.01, 0.05, 0.10, 0.20])
    assert clases.tolist() == [0, 0, 1, 2, 4, 4, 0]
    assert clases.dtype == np.uint8


def test_importancia_por_permutacion_encuentra_la_variable_que_importa():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5)).astype(np.float32)
    y = (X[:, 1] + 0.5 * X[:, 3] + rng.normal(0, 0.5, 4000) > 1.0).astype(np.uint8)
    modelo = LogisticRegression().fit(X, y)
    grupos = {"uno": [0], "dos": [1], "tres_cuatro": [2, 3], "cinco": [4]}
    t = evaluacion.importancia_permutacion(lambda M: modelo.predict_proba(M)[:, 1], X, y, grupos,
                                           repeticiones=3, semilla=1)
    assert list(t.columns) == ["grupo", "caida_pr_auc", "desvio"]
    assert t["grupo"].iloc[0] == "dos"                                  # ordenada de mayor a menor
    assert t.set_index("grupo").loc["dos", "caida_pr_auc"] > 0.2
    assert abs(t.set_index("grupo").loc["uno", "caida_pr_auc"]) < 0.02
    assert t.set_index("grupo").loc["tres_cuatro", "caida_pr_auc"] > 0.01
    assert np.array_equal(X, X)                                          # no modifica la matriz original


def test_importancia_no_altera_la_matriz_de_entrada():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3)).astype(np.float32)
    copia = X.copy()
    y = (X[:, 0] > 0).astype(np.uint8)
    evaluacion.importancia_permutacion(lambda M: M[:, 0], X, y, {"a": [0], "b": [1, 2]}, repeticiones=2, semilla=0)
    assert np.array_equal(X, copia)


def test_tabla_en_markdown_con_formato_por_columna():
    t = pd.DataFrame({"modelo": ["a", "b"], "captura": [0.3712, 0.05], "n": [1200, 7]})
    md = evaluacion.tabla_md(t, {"captura": "pct1", "n": "entero"}, encabezados={"captura": "Captura"})
    lineas = md.splitlines()
    assert lineas[0] == "| modelo | Captura | n |"
    assert lineas[1] == "|---|---:|---:|"
    assert lineas[2] == "| a | 37,1 % | 1.200 |"
    assert lineas[3] == "| b | 5,0 % | 7 |"


def test_formatos_numericos_en_espanol():
    assert evaluacion.formatear(0.12345, "pct1") == "12,3 %"
    assert evaluacion.formatear(0.0005, "pct2") == "0,05 %"
    assert evaluacion.formatear(1234567, "entero") == "1.234.567"
    assert evaluacion.formatear(0.93456, "dec3") == "0,935"
    assert evaluacion.formatear(float("nan"), "pct1") == "—"
    assert evaluacion.formatear("texto", "pct1") == "texto"


def test_resumen_calibracion_mide_nivel_y_perdida():
    y = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    r = evaluacion.resumen_calibracion(np.full((2, 4), 0.5), y)
    assert r["observadas"] == 2
    assert r["esperadas"] == pytest.approx(4.0)
    assert r["razon"] == pytest.approx(2.0)                 # se esperaba el doble de lo que hubo
    assert r["perdida_log"] == pytest.approx(np.log(2.0))


def test_resumen_calibracion_perdida_finita_con_probabilidades_extremas():
    r = evaluacion.resumen_calibracion(np.array([[0.0, 1.0]]), np.array([[1, 0]], dtype=np.uint8))
    assert np.isfinite(r["perdida_log"])


def test_resumen_calibracion_sin_fuego_no_divide_por_cero():
    r = evaluacion.resumen_calibracion(np.array([[0.1, 0.2]]), np.zeros((1, 2), dtype=np.uint8))
    assert r["observadas"] == 0 and np.isnan(r["razon"])
