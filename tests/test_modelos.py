import warnings

import numpy as np
import pytest

from incendios import modelos


def datos(n=3000, p=6, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.5, n) > 1.0).astype(np.uint8)
    return X, y


@pytest.mark.parametrize("nombre", modelos.NOMBRES)
def test_una_columna_sin_ningun_dato_no_rompe_el_entrenamiento(nombre):
    # pasa cuando falta toda una capa (p. ej. ninguna tesela de relieve)
    X, y = datos()
    X[:, 3] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # tampoco debe avisar nada
        modelo = modelos.crear(nombre, rapido=True).fit(X, y)
        p = modelo.predict_proba(X)[:, 1]
    assert np.isfinite(p).all()
    assert p[y == 1].mean() > p[y == 0].mean() + 0.2
    # si al predecir esa columna sí trae datos, se ignora igual
    X2 = X.copy()
    X2[:, 3] = 99.0
    assert np.allclose(modelo.predict_proba(X2)[:, 1], p)


@pytest.mark.parametrize("nombre", modelos.NOMBRES)
def test_los_faltantes_parciales_se_toleran(nombre):
    X, y = datos()
    X[::3, 1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = modelos.crear(nombre, rapido=True).fit(X, y).predict_proba(X)[:, 1]
    assert np.isfinite(p).all()


def test_modelo_desconocido_es_error():
    with pytest.raises(ValueError, match="desconocido"):
        modelos.crear("red_neuronal")


def _riesgo_en_u(n=20000, semilla=0):
    """El fuego observado es alto con poca lluvia y también (por azar del periodo) con mucha."""
    rng = np.random.default_rng(semilla)
    lluvia = rng.uniform(0, 100, n)
    otra = rng.normal(size=n)
    p = np.where(lluvia < 20, 0.30, np.where(lluvia > 80, 0.25, 0.03))
    y = (rng.random(n) < p).astype(np.uint8)
    return np.column_stack([lluvia, otra]).astype(np.float32), y


def _curva(modelo, valores):
    X = np.column_stack([valores, np.zeros(len(valores))]).astype(np.float32)
    return modelo.predict_proba(X)[:, 1]


def test_con_restriccion_mas_lluvia_nunca_sube_el_riesgo():
    X, y = _riesgo_en_u()
    modelo = modelos.crear("gbm", rapido=True, variables=["prec_30", "otra"]).fit(X, y)
    curva = _curva(modelo, np.linspace(0, 100, 51))
    assert (np.diff(curva) <= 1e-9).all()
    assert curva[0] > curva[-1] + 0.05


def test_sin_restriccion_el_modelo_copia_la_forma_de_u():
    X, y = _riesgo_en_u()
    modelo = modelos.crear("gbm", rapido=True, variables=["prec_30", "otra"], monotono=False).fit(X, y)
    curva = _curva(modelo, np.linspace(0, 100, 51))
    assert curva[-1] > curva[25] + 0.1                 # vuelve a subir con mucha lluvia


def test_las_restricciones_cubren_el_clima_antecedente_y_la_actividad_departamental():
    from incendios import clima
    for nombre in clima.VARS_ANTECEDENTE + ["rec_dep_1", "rec_dep_4", "rec_dep_anom", "rec_dep_d3"]:
        assert modelos.MONOTONIA[nombre] in (-1, 1), nombre
    assert modelos.MONOTONIA["prec_90"] == -1 and modelos.MONOTONIA["fwi_7"] == 1
    assert modelos.MONOTONIA["hrmin_7"] == -1 and modelos.MONOTONIA["dias_sin_lluvia"] == 1
    # nada estático, de historial ni de vecindario lleva restricción
    assert not any(n.startswith(("lc_", "his_", "est_", "rec_c", "rec_v", "n_", "elev")) for n in modelos.MONOTONIA)


def test_entrenar_aplica_las_restricciones_segun_el_nombre_de_cada_columna(ctx_sintetico, muestra_sintetica):
    operativo = modelos.entrenar("gbm", "operativo", muestra_sintetica, ctx_sintetico, rapido=True)
    nombres = ctx_sintetico.constructor.nombres("operativo")
    restricciones = list(operativo.estimador[-1].monotonic_cst)
    assert len(restricciones) == len(nombres)
    assert restricciones[nombres.index("prec_30")] == -1
    assert restricciones[nombres.index("fwi_7")] == 1
    assert restricciones[nombres.index("his_c3")] == 0 and restricciones[nombres.index("lc_arbol")] == 0
    estacional = modelos.entrenar("gbm", "estacional", muestra_sintetica, ctx_sintetico, rapido=True)
    assert not any(estacional.estimador[-1].monotonic_cst or [])     # sin variables restringidas: queda libre


def test_entrenar_puede_excluir_variables(ctx_sintetico, muestra_sintetica):
    from incendios import clima
    modelo = modelos.entrenar("gbm", "operativo", muestra_sintetica, ctx_sintetico, rapido=True,
                              excluir=clima.VARS_ANTECEDENTE)
    todas = ctx_sintetico.constructor.nombres("operativo")
    assert modelo.variables == [n for n in todas if n not in clima.VARS_ANTECEDENTE]
    assert len(modelo.columnas) == len(todas) - len(clima.VARS_ANTECEDENTE)
    X = muestra_sintetica["X"][:300].copy()
    antes = modelo.puntaje(X)
    X[:, [todas.index(n) for n in clima.VARS_ANTECEDENTE]] = 1e6       # ya no las mira
    assert np.array_equal(modelo.puntaje(X), antes)
    assert modelo.info["excluidas"] == list(clima.VARS_ANTECEDENTE)


def test_excluir_una_variable_que_no_existe_es_error(ctx_sintetico, muestra_sintetica):
    with pytest.raises(ValueError, match="no_existe"):
        modelos.entrenar("gbm", "operativo", muestra_sintetica, ctx_sintetico, rapido=True, excluir=["no_existe"])


def test_la_parada_temprana_devuelve_un_numero_de_arboles_y_su_historial(ctx_sintetico, muestra_sintetica):
    m = muestra_sintetica
    entrena, valida = m["anio"] <= 2016, m["anio"] == 2017
    n_iter, historial = modelos.iteraciones_con_parada("operativo", m, ctx_sintetico, entrena, valida,
                                                       verboso=False, max_leaf_nodes=7)
    assert n_iter % modelos.PASO_PARADA == 0 and modelos.PASO_PARADA <= n_iter <= modelos.ITER_MAX
    assert [h["n_iter"] for h in historial] == list(range(modelos.PASO_PARADA, historial[-1]["n_iter"] + 1, modelos.PASO_PARADA))
    # se queda con el último número de árboles que mejoró la pérdida en más de 1e-5: una mejora menor
    # que eso no justifica más árboles, así que la pérdida elegida está a menos de 1e-5 de la mínima
    perdidas = {h["n_iter"]: h["perdida"] for h in historial}
    assert perdidas[n_iter] <= min(perdidas.values()) + 1e-5
    assert all(perdidas[k] > perdidas[n_iter] + 1e-5 for k in perdidas if k < n_iter)
    assert set(historial[0]) == {"n_iter", "perdida", "pr_auc_muestra"}


def test_el_ensamble_promedia_las_probabilidades_de_sus_modelos():
    X, y = datos()
    a = modelos.crear("gbm", semilla=1, rapido=True).fit(X, y)
    b = modelos.crear("gbm", semilla=2, rapido=True).fit(X, y)
    ens = modelos.Ensamble([a, b])
    assert np.allclose(ens.predict_proba(X), (a.predict_proba(X) + b.predict_proba(X)) / 2)
    assert list(ens.classes_) == [0, 1]


def test_entrenar_con_varias_semillas_promedia_un_modelo_por_semilla(ctx_sintetico, muestra_sintetica):
    uno = modelos.entrenar("gbm", "operativo", muestra_sintetica, ctx_sintetico, rapido=True, semilla=7)
    varios = modelos.entrenar("gbm", "operativo", muestra_sintetica, ctx_sintetico, rapido=True, semillas=[7, 8, 9])
    assert isinstance(varios.estimador, modelos.Ensamble) and len(varios.estimador.estimadores) == 3
    assert varios.info["semillas"] == [7, 8, 9] and uno.info["semillas"] == [7]
    X = muestra_sintetica["X"][:500]
    sueltos = [modelos.entrenar("gbm", "operativo", muestra_sintetica, ctx_sintetico, rapido=True, semilla=s)
               for s in (7, 8, 9)]
    from incendios.muestreo import corregir_prior
    crudo = np.mean([m.estimador.predict_proba(X[:, m.columnas])[:, 1] for m in sueltos], axis=0)
    assert np.allclose(varios.puntaje(X), corregir_prior(crudo, muestra_sintetica["tasa_neg"]), atol=1e-6)
    assert np.allclose(uno.puntaje(X), sueltos[0].puntaje(X))


def test_el_ensamble_es_un_clasificador_para_scikit_learn():
    from sklearn.base import is_classifier
    X, y = datos()
    assert is_classifier(modelos.Ensamble([modelos.crear("gbm", rapido=True).fit(X, y)]))
