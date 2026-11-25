import numpy as np
import pytest

from incendios.rutas import Rutas
from incendios import flujo, metricas, modelos, tiempo
from tests import mundo


@pytest.fixture(scope="module")
def ctx(ctx_sintetico):
    return ctx_sintetico


@pytest.fixture(scope="module")
def muestra(muestra_sintetica):
    return muestra_sintetica


def test_carga_el_departamento_y_sus_divisiones(ctx):
    assert ctx.n == 38 * 40                       # las dos filas del norte quedan fuera
    assert ctx.lon.shape == ctx.lat.shape == (ctx.n,)
    assert ctx.provincias == ["Cercado", "Chapare"]
    assert set(np.unique(ctx.provincia_id)) == {0, 1}
    assert ctx.municipios == ["Cochabamba", "VillaTunari", "Totora"]
    assert ctx.provincia_de_municipio == ["Cercado", "Chapare", "Chapare"]


def test_asigna_zonas_por_municipio(ctx):
    assert ctx.zonas == ["Trópico", "Región Metropolitana", "Valles, Cono Sur y Andina"]
    zona_de = {ctx.municipios[m]: ctx.zonas[z] for m, z in set(zip(ctx.municipio_id.tolist(), ctx.zona_id.tolist()))}
    assert zona_de == {"Cochabamba": "Región Metropolitana", "VillaTunari": "Trópico",
                       "Totora": "Valles, Cono Sur y Andina"}


def test_las_notas_resumen_los_datos(ctx):
    assert ctx.notas["series_clima"] == 2
    assert ctx.notas["cobertura_landcover"] == 1.0
    assert ctx.notas["cobertura_dem"] == pytest.approx(0.5)
    assert ctx.notas["ultima_fecha_focos"] == "2019-12-31"


def test_semanas_utiles_excluyen_el_apagon_y_la_semana_siguiente(ctx):
    utiles = set(ctx.semanas_utiles(2016, 2016))
    apagon = {tiempo.semana_de_fecha(d) for d in ["2016-08-03", "2016-08-10"]}
    # 1–7 ago: 0 días de satélite. 8–14 ago: 4 días (vale por sí misma, pero su semana anterior no).
    assert not (apagon & utiles)
    assert tiempo.semana_de_fecha("2016-08-17") in utiles          # 15–21 ago: ella y la anterior tienen >= 4 días
    assert len(utiles) == 52 - 2


def test_semanas_utiles_empiezan_cuando_hay_tres_anios_de_historial(ctx):
    assert ctx.semanas_utiles(2012, 2014) == []
    assert len(ctx.semanas_utiles(2015, 2015)) == 53


def test_la_muestra_tiene_todos_los_positivos_y_los_negativos_pedidos(ctx, muestra):
    semanas = ctx.semanas_utiles(2015, 2017)
    positivos = sum(int(ctx.constructor.etiquetas(s).sum()) for s in semanas)
    assert positivos > 500
    assert int(muestra["y"].sum()) == positivos
    assert int((muestra["y"] == 0).sum()) == 300 * len(semanas)
    assert muestra["X"].shape == (len(muestra["y"]), len(ctx.constructor.nombres("operativo")))
    assert muestra["X"].dtype == np.float32
    assert 0.15 < muestra["tasa_neg"] < 0.25                        # 300 de ~1.520 celdas
    assert set(np.unique(muestra["anio"])) == {2015, 2016, 2017}
    assert muestra["nuevo"].dtype == bool and len(muestra["fila"]) == len(muestra["y"])


def test_columnas_del_regimen_estacional_son_las_primeras(ctx):
    estacional = flujo.columnas_de(ctx, "estacional")
    assert estacional.tolist() == list(range(len(ctx.constructor.nombres("estacional"))))
    assert len(flujo.columnas_de(ctx, "operativo")) == len(ctx.constructor.nombres("operativo"))


def test_matriz_provincia_da_lo_mismo_a_todas_las_celdas_de_una_provincia(ctx):
    s = tiempo.semana_de_fecha("2017-09-13")
    X = flujo.matriz_provincia(ctx, s)
    assert X.shape[0] == ctx.n
    for p in (0, 1):
        de_la_provincia = X[ctx.provincia_id == p]
        assert (de_la_provincia == de_la_provincia[0]).all()
    assert not np.array_equal(X[ctx.provincia_id == 0][0], X[ctx.provincia_id == 1][0])
    filas = np.array([0, 5, ctx.n - 1])
    assert np.array_equal(flujo.matriz_provincia(ctx, s, filas), X[filas])


def test_las_estaticas_se_guardan_en_cache_y_se_recalculan_si_cambian_las_teselas(ctx, tif):
    cache = ctx.rutas.estaticas
    assert cache.exists()
    antes = cache.stat().st_mtime_ns
    est = flujo.cargar_estaticas(ctx.malla, ctx.rutas, verboso=False)
    assert cache.stat().st_mtime_ns == antes                        # no se recalculó
    assert np.isnan(est["elev_media"][:, 20:]).all()
    # llega la tesela que faltaba (mitad este)
    dem = np.full((40 * 2, 20 * 2), 300.0, dtype=np.float32)
    nueva = ctx.rutas.relieve_bajado / "dem_este.tif"            # bajada después: va a external
    nueva.parent.mkdir(parents=True, exist_ok=True)
    tif(nueva, dem, ctx.malla.oeste + 20 * ctx.malla.celda, ctx.malla.norte, ctx.malla.celda / 2)
    try:
        est = flujo.cargar_estaticas(ctx.malla, ctx.rutas, verboso=False)
        assert np.allclose(est["elev_media"][:, 20:], 300.0)
    finally:
        nueva.unlink()
        flujo.cargar_estaticas(ctx.malla, ctx.rutas, verboso=False)


def test_carpeta_de_datos_equivocada_da_un_error_que_dice_que_hacer(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.raw"):
        flujo.cargar_todo(Rutas.en(tmp_path), malla=mundo.MALLA, verboso=False)


@pytest.mark.parametrize("nombre", modelos.NOMBRES)
def test_cada_tipo_de_modelo_entrena_y_devuelve_probabilidades(ctx, muestra, nombre):
    modelo = modelos.entrenar(nombre, "operativo", muestra, ctx, rapido=True)
    p = modelo.puntaje(muestra["X"][:500])
    assert p.shape == (500,) and np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all()
    assert modelo.regimen == "operativo" and modelo.nombre == nombre


def test_el_modelo_estacional_solo_usa_variables_estacionales(ctx, muestra):
    modelo = modelos.entrenar("gbm", "estacional", muestra, ctx, rapido=True)
    assert modelo.variables == ctx.constructor.nombres("estacional")
    X = muestra["X"][:200].copy()
    antes = modelo.puntaje(X)
    X[:, len(modelo.variables):] = -12345.0          # destrozar lo operativo no debe cambiar nada
    assert np.array_equal(modelo.puntaje(X), antes)


def test_flujo_completo_supera_al_azar_y_respeta_la_tasa_real(ctx, muestra, tmp_path):
    gbm = modelos.entrenar("gbm", "operativo", muestra, ctx, rapido=True)
    provincia = modelos.entrenar_provincia(muestra, ctx, rapido=True)
    semanas = ctx.semanas_utiles(2018, 2018)
    rng = np.random.default_rng(0)
    carpeta = flujo.puntuar_malla(ctx, semanas, {
        "gbm": lambda c, s, X: gbm.puntaje(X),
        "provincia": lambda c, s, X: provincia.puntaje_semana(c, s),
        "azar": lambda c, s, X: rng.random(c.n),
    }, tmp_path / "pred", verboso=False)

    y = np.load(carpeta / "y.npy")
    assert y.shape == (len(semanas), ctx.n) and y.dtype == np.uint8
    assert np.array_equal(np.load(carpeta / "semanas.npy"), np.array(semanas))
    y_tol, nuevo = np.load(carpeta / "y_tol.npy"), np.load(carpeta / "nuevo.npy")
    assert (y_tol >= y).all() and y_tol.sum() > y.sum()
    assert nuevo.dtype == bool and nuevo.shape == y.shape

    grupos = np.repeat(np.arange(len(semanas)), ctx.n)
    r = {n: metricas.resumen(y.ravel(), np.load(carpeta / f"pred_{n}.npy").ravel(), grupos)
         for n in ("gbm", "provincia", "azar")}
    assert r["azar"]["captura_10"] == pytest.approx(0.10, abs=0.03)
    assert r["gbm"]["captura_10"] > 0.5                        # el punto caliente es el 6,5 % del área
    assert r["gbm"]["captura_10"] > r["provincia"]["captura_10"] + 0.2
    assert r["gbm"]["roc_auc"] > 0.9
    # la probabilidad corregida al prior queda en el orden de la tasa real
    assert np.load(carpeta / "pred_gbm.npy").mean() == pytest.approx(y.mean(), rel=0.5)
    # el modelo por provincia da un solo valor por provincia y semana
    pp = np.load(carpeta / "pred_provincia.npy")
    assert all(len(np.unique(pp[k][ctx.provincia_id == p])) == 1 for k in (0, 30) for p in (0, 1))


def test_la_muestra_se_guarda_y_se_vuelve_a_cargar_igual(ctx, muestra, tmp_path):
    flujo.guardar_muestra(muestra, ctx, tmp_path / "preparado")
    leida, info = flujo.cargar_muestra(tmp_path / "preparado")
    for clave in ("X", "y", "semana", "fila", "nuevo", "anio"):
        assert np.array_equal(leida[clave], muestra[clave], equal_nan=clave == "X"), clave
    assert leida["tasa_neg"] == muestra["tasa_neg"]
    assert info["variables"]["operativo"] == ctx.constructor.nombres("operativo")
    assert info["variables"]["estacional"] == ctx.constructor.nombres("estacional")
    assert set(info["grupos"]["operativo"]) >= {"cobertura", "terreno", "historial", "fuego_reciente"}
    assert info["filas"] == len(muestra["y"]) and info["positivos"] == int(muestra["y"].sum())
    assert info["notas"]["series_clima"] == 2


def test_cargar_muestra_inexistente_dice_que_paso_falta(tmp_path):
    with pytest.raises(FileNotFoundError, match="incendios preparar"):
        flujo.cargar_muestra(tmp_path / "nada")


def test_se_prefiere_el_clima_en_malla_nativa_si_existe(tmp_path):
    rutas = Rutas.en(tmp_path)
    with pytest.raises(FileNotFoundError, match="clima"):
        flujo.archivo_clima(rutas)
    original, nativa = rutas.clima_power, rutas.clima_power_nativo
    for ruta in (original, nativa):
        ruta.parent.mkdir(parents=True, exist_ok=True)
    original.write_text("x")
    assert flujo.archivo_clima(rutas) == original
    nativa.write_text("x")
    assert flujo.archivo_clima(rutas) == nativa


def test_la_huella_de_datos_dice_con_que_archivos_se_construyo(ctx):
    huella = ctx.notas["huella"]
    assert huella["clima"] == "clima_suelo_diario_cochabamba_2012_2025.csv"
    assert huella["teselas"] == ["dem.tif", "lc.tif"]
    assert huella["focos_hasta"] == "2019-12-31"


def test_si_el_clima_no_trae_nombres_de_provincia_se_usa_la_serie_mas_cercana(ctx):
    import copy
    por_nombre = flujo.serie_de_provincia(ctx)
    otro = copy.copy(ctx)
    otro.clima = copy.copy(ctx.clima)
    otro.clima.tabla = ctx.clima.tabla.assign(provincias=["celda_a", "celda_b"])
    assert flujo.serie_de_provincia(otro).tolist() == por_nombre.tolist()


def test_las_fuentes_de_calor_fijas_se_retiran_de_los_focos(ctx):
    fila, col = mundo.CHIMENEA
    celda = fila * mundo.MALLA.ncols + col
    assert ctx.fuentes["celda"].tolist() == [celda]
    assert ctx.fuentes["municipio"].tolist() == ["VillaTunari"]
    assert ctx.fuentes["latitud"].iloc[0] == pytest.approx(mundo.MALLA.norte - (fila + 0.5) * mundo.MALLA.celda)
    assert ctx.notas["fuentes_persistentes"] == 1
    assert celda not in set(ctx.focos["celda"])
    # ni la variable objetivo ni el historial la ven arder
    k = int(np.flatnonzero(ctx.idx == celda)[0])
    assert sum(int(ctx.constructor.etiquetas(s)[k]) for s in ctx.semanas_utiles(2015, 2017)) == 0


def test_se_prefiere_el_clima_era5_y_se_puede_forzar_la_fuente(tmp_path, monkeypatch):
    rutas = Rutas.en(tmp_path)
    original, era5 = rutas.clima_power, rutas.clima_era5
    for ruta in (original, era5):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text("x")
    monkeypatch.delenv("INCENDIOS_CLIMA", raising=False)
    assert flujo.archivo_clima(rutas) == era5
    monkeypatch.setenv("INCENDIOS_CLIMA", "power")
    assert flujo.archivo_clima(rutas) == original
    monkeypatch.setenv("INCENDIOS_CLIMA", "era5")
    assert flujo.archivo_clima(rutas) == era5
    era5.unlink()
    with pytest.raises(FileNotFoundError, match="ERA5"):
        flujo.archivo_clima(rutas)


# ---------------------------------------------------------------------- capas extra
def test_las_capas_extra_entran_como_grupos_nuevos_y_quedan_en_la_huella(ctx, ctx_con_capas):
    from incendios import capas
    c = ctx_con_capas.constructor
    for regimen in ("estacional", "operativo"):
        assert set(capas.GRUPOS_CAPAS) <= set(c.grupos(regimen))
        base = ctx.constructor.nombres(regimen)
        assert c.nombres(regimen)[:len(base)] == base
    assert ctx_con_capas.notas["huella"]["capas"] == ["caminos", "cobertura", "perdida_bosque", "poblacion", "quemas"]
    assert ctx_con_capas.notas["capas"]["quemas"]["anios"] == [2001, 2019]
    assert ctx.constructor.capas is None and ctx.notas["huella"]["capas"] == []


def test_las_capas_se_pueden_apagar_con_una_variable_de_entorno(ctx_con_capas, tmp_path, monkeypatch):
    monkeypatch.setenv("INCENDIOS_CAPAS", "0")
    apagado = flujo.cargar_todo(Rutas(datos=ctx_con_capas.rutas.datos, salida=tmp_path), malla=mundo.MALLA,
                                verboso=False)
    assert apagado.constructor.capas is None and apagado.notas["huella"]["capas"] == []


def test_una_muestra_armada_con_otras_variables_se_rechaza(ctx, ctx_con_capas, muestra, tmp_path):
    flujo.guardar_muestra(muestra, ctx, tmp_path)
    flujo.cargar_muestra(tmp_path, ctx)                              # con las mismas variables: bien
    with pytest.raises(ValueError, match="incendios preparar"):
        flujo.cargar_muestra(tmp_path, ctx_con_capas)


def test_la_huella_anotada_dice_si_hay_que_reentrenar(ctx, tmp_path):
    huella = ctx.notas["huella"]
    assert flujo.datos_cambiaron(tmp_path, huella)                  # nunca se anotó
    flujo.anotar_huella(tmp_path, huella)
    assert not flujo.datos_cambiaron(tmp_path, huella)
    assert not flujo.datos_cambiaron(tmp_path, dict(huella, focos_hasta="2030-01-01"))   # focos nuevos: no
    assert flujo.datos_cambiaron(tmp_path, dict(huella, capas=["quemas"]))
    assert flujo.datos_cambiaron(tmp_path, dict(huella, clima="clima_era5land_diario.csv.gz"))
    assert flujo.datos_cambiaron(tmp_path, dict(huella, variables=huella["variables"] + ["otra"]))    # código nuevo
    assert flujo.datos_cambiaron(tmp_path, dict(huella, satelites=["NOAA-20 2018-04-01–"]))


def test_la_huella_guarda_las_variables_y_los_satelites(ctx):
    huella = ctx.notas["huella"]
    assert huella["variables"] == ctx.constructor.nombres("operativo")
    assert [s.split()[0] for s in huella["satelites"]] == ["S-NPP", "NOAA-20"]


def test_una_huella_anterior_sin_variables_ni_satelites_no_obliga_a_reentrenar(ctx):
    nueva = ctx.notas["huella"]
    vieja = {k: v for k, v in nueva.items() if k not in ("variables", "satelites")}
    assert not flujo.huellas_distintas(vieja, nueva) and flujo.claves_distintas(nueva, vieja) == []
    assert flujo.claves_distintas(vieja, dict(nueva, clima="otro.csv")) == ["clima"]


def test_las_notas_dicen_de_que_satelite_son_los_focos(ctx):
    assert list(ctx.notas["focos_por_satelite"]) == ["S-NPP"]
    assert ctx.notas["focos_por_satelite"]["S-NPP"][1] <= "2019-12-31"
