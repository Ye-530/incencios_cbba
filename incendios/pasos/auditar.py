"""CRISP-DM fase 2 — Comprensión de los datos.

Mide la calidad de cada fuente y deja tablas y figuras en reports/.

    python -m incendios auditar
"""
import argparse
import json


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

from incendios import clima as mod_clima
from incendios import config, flujo, focos as mod_focos, graficos as g
from incendios.tiempo import inicio_semana

ERA5_PASO = 0.1          # grados: tamaño de celda de ERA5-Land (incendios/era5.py)


def tabla_focos(ctx, tablas):
    f = ctx.focos[ctx.mascara.ravel()[ctx.focos["celda"].to_numpy()]].copy()
    f["anio"], f["mes"] = f["fecha"].dt.year, f["fecha"].dt.month
    por_mes = f.pivot_table(index="anio", columns="mes", values="frp", aggfunc="size", fill_value=0)
    por_mes.to_csv(tablas / "auditoria_focos_anio_mes.csv")
    en_depto = dict(zip(ctx.idx, ctx.provincia_id))
    f["provincia"] = [ctx.provincias[en_depto[c]] for c in f["celda"]]
    area = pd.Series(ctx.malla.area_km2().ravel()[ctx.idx]).groupby(ctx.provincia_id).sum()
    area.index = [ctx.provincias[i] for i in area.index]
    por_prov = f.groupby("provincia").size().rename("detecciones").to_frame()
    por_prov["porcentaje"] = (100 * por_prov["detecciones"] / por_prov["detecciones"].sum()).round(1)
    por_prov["area_km2"] = area.round(0)
    anios = f["anio"].nunique()
    por_prov["por_100km2_anio"] = (por_prov["detecciones"] / por_prov["area_km2"] * 100 / anios).round(1)
    por_prov.sort_values("detecciones", ascending=False).to_csv(tablas / "auditoria_focos_provincia.csv")
    return f, por_mes, por_prov


def tabla_apagones(ctx, tablas):
    dias = mod_focos.dias_sin_satelite(ctx.conteo_diario)
    serie = pd.Series(dias)
    racha = (serie.diff().dt.days != 1).cumsum()
    rachas = serie.groupby(racha).agg(desde="min", hasta="max", dias="size").reset_index(drop=True)
    rachas.to_csv(tablas / "auditoria_dias_sin_satelite.csv", index=False)
    pocas = [(str(inicio_semana(s)), int(d)) for s, d in enumerate(ctx.dias_obs)
             if d < config.MIN_DIAS_OBSERVADOS and 3 <= s < ctx.n_semanas - 1]
    return dias, rachas, pocas


def auditoria_clima(ctx, tablas):
    crudo = mod_clima.cargar_clima(flujo.archivo_clima(ctx.rutas))
    puntos = crudo.groupby("provincia")[["lat", "lon"]].first()
    ctx.clima.tabla.to_csv(tablas / "auditoria_clima_series.csv", index=False)
    lluvia = crudo[crudo["precipitacion"] > 100][["provincia", "fecha", "precipitacion", "temp_max"]]
    lluvia.sort_values("precipitacion", ascending=False).to_csv(tablas / "auditoria_lluvia_artefacto.csv", index=False)
    anual = crudo[crudo["precipitacion"] <= config.LLUVIA_MAXIMA_CREIBLE].groupby(
        ["provincia", "anio"])["precipitacion"].sum().groupby("provincia").mean().round(0)
    anual.rename("lluvia_anual_mm").to_csv(tablas / "auditoria_lluvia_anual.csv")
    return crudo, puntos, lluvia, anual


def cobertura_capas(ctx, focos_depto, tablas):
    est = flujo.cargar_estaticas(ctx.malla, ctx.rutas, verboso=False)
    area = ctx.malla.area_km2().ravel()[ctx.idx]
    celdas_foco = focos_depto["celda"].to_numpy()
    filas = []
    for capa, variable in (("Relieve (Copernicus DEM 30 m)", "elev_media"),
                           ("Cobertura de suelo (ESA WorldCover 10 m)", "lc_arbol")):
        valido = np.isfinite(est[variable].ravel())
        filas.append({"capa": capa,
                      "area_cubierta_pct": round(100 * area[valido[ctx.idx]].sum() / area.sum(), 1),
                      "focos_cubiertos_pct": round(100 * valido[celdas_foco].mean(), 1)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(tablas / "auditoria_cobertura.csv", index=False)
    return est, tabla


# --------------------------------------------------------------------------- figuras
def figura_anio_mes(por_mes, ruta):
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    datos = por_mes.reindex(columns=range(1, 13), fill_value=0)
    limites = [0, 50, 200, 500, 1000, 2000, 4000, 8000]
    cmap = ListedColormap(g.RAMPA_AZUL)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    malla = ax.pcolormesh(np.arange(13), np.arange(len(datos) + 1), datos.to_numpy(),
                          cmap=cmap, norm=BoundaryNorm(limites, cmap.N), edgecolors=g.SUPERFICIE, linewidth=1.5)
    ax.set_xticks(np.arange(12) + 0.5, meses)
    ax.set_yticks(np.arange(len(datos)) + 0.5, datos.index)
    ax.invert_yaxis()
    ax.grid(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_visible(False)
    for i, total in enumerate(datos.sum(axis=1)):
        ax.text(12.15, i + 0.5, f"{total:,.0f}".replace(",", "."), va="center", fontsize=8.5, color=g.TINTA_2)
    ax.text(12.15, len(datos) + 0.5, "Total\ndel año", fontsize=8.5, color=g.TENUE, va="center")
    barra = fig.colorbar(malla, ax=ax, pad=0.09, shrink=0.8, ticks=limites)
    barra.outline.set_visible(False)
    barra.set_label("Detecciones por mes")
    g.titulos(ax, "El fuego se concentra de agosto a noviembre",
              "Detecciones VIIRS de vegetación en Cochabamba, por año y mes")
    return g.guardar(fig, ruta)


def figura_cobertura(ctx, focos_depto, est, limite, tabla, ruta):
    m = ctx.malla
    periodo = f"{focos_depto['fecha'].min().year}–{focos_depto['fecha'].max().year}"
    cuenta = np.bincount(focos_depto["celda"].to_numpy(), minlength=m.nfilas * m.ncols).reshape(m.forma)
    bloque = 5
    nf, nc = m.nfilas // bloque, m.ncols // bloque
    densidad = cuenta[:nf * bloque, :nc * bloque].reshape(nf, bloque, nc, bloque).sum(axis=(1, 3)).astype(float)
    dentro = ctx.mascara[:nf * bloque, :nc * bloque].reshape(nf, bloque, nc, bloque).any(axis=(1, 3))
    densidad[~dentro] = np.nan
    limites = [0, 1, 5, 20, 50, 100, 200, 2000]
    cmap = ListedColormap([g.FONDO_DEPTO] + g.RAMPA_AZUL[1:])
    oeste, este, sur, norte = g.extension(m)
    fig, ax = plt.subplots(figsize=(8.0, 7.4))
    img = ax.imshow(densidad, extent=(oeste, oeste + nc * bloque * m.celda, norte - nf * bloque * m.celda, norte),
                    cmap=cmap, norm=BoundaryNorm(limites, cmap.N), interpolation="nearest", zorder=1)
    g.preparar_mapa(ax, m)
    g.contornos(ax, limite, color=g.TINTA_2, ancho=0.9)
    lon, lat = m.centros()
    con_dem = np.isfinite(est["elev_media"])
    sin_dem = (~con_dem & ctx.mascara).sum() / ctx.mascara.sum()
    if sin_dem >= 0.01:                      # el borde del relieve solo tiene sentido si falta en una zona apreciable
        ax.contour(lon, lat, con_dem.astype(float), levels=[0.5], colors=[g.NARANJA], linewidths=1.6, zorder=4)
    con_lc = np.isfinite(est["lc_arbol"])
    if (~con_lc & ctx.mascara).any():
        lat_limite = lat[np.flatnonzero(con_lc.any(axis=1)).max()] - m.celda / 2
        ax.plot([oeste, este], [lat_limite, lat_limite], color=g.TINTA, linewidth=1.2, zorder=4)
        ax.text(oeste + 0.04, lat_limite - 0.30, "Al sur de la línea negra\nno hay cobertura de suelo",
                ha="left", va="top", fontsize=8.5, color=g.TINTA)
    filas, cols = np.nonzero(con_dem)
    ax.text(lon[cols.min()] + 0.04, lat[filas.min()] - 0.06,
            f"Recuadro con relieve:\n{tabla.iloc[0].area_cubierta_pct:.0f} % del área,\n"
            f"{tabla.iloc[0].focos_cubiertos_pct:.0f} % de los focos",
            fontsize=8.5, color=g.TINTA, va="top", zorder=5,
            bbox={"facecolor": g.SUPERFICIE, "edgecolor": "none", "alpha": 0.85, "pad": 2})
    barra = fig.colorbar(img, ax=ax, shrink=0.55, pad=0.02, ticks=limites[:-1])
    barra.outline.set_visible(False)
    barra.set_label(f"Detecciones {periodo} por bloque de 2,5 km")
    sin_relieve = 100.0 - float(tabla.iloc[0].focos_cubiertos_pct)
    if sin_relieve >= 1.0:
        titulo = f"El {sin_relieve:.0f} % de los focos cae donde falta el relieve"
    else:
        titulo = "Las teselas de relieve cubren todo el departamento"
    g.titulos(ax, titulo, f"Densidad de focos {periodo} y zona cubierta por las teselas disponibles")
    return g.guardar(fig, ruta)


def figura_clima(ctx, puntos, limite, provincias_gj, ruta):
    """Dónde está el clima: celdas MERRA-2 de ~55 × 65 km con POWER, o la malla de 0,1° de ERA5-Land."""
    power = ctx.clima.fuente == mod_clima.FUENTE_POWER
    dlon, dlat = (mod_clima.MERRA_DLON, mod_clima.MERRA_DLAT) if power else (ERA5_PASO, ERA5_PASO)
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    g.preparar_mapa(ax, ctx.malla, recorte=(-67.35, -64.0, -18.9, -15.6))
    g.contornos(ax, provincias_gj, color=g.EJE, ancho=0.6)
    g.contornos(ax, limite, color=g.TINTA_2, ancho=1.0)
    for fila in ctx.clima.tabla.itertuples():
        x0, y0 = fila.lon_c - dlon / 2, fila.lat_c - dlat / 2
        ax.add_patch(Rectangle((x0, y0), dlon, dlat, facecolor=g.NARANJA, alpha=0.10, edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((x0, y0), dlon, dlat, facecolor="none", edgecolor=g.NARANJA,
                               linewidth=1.2 if power else 0.4, zorder=4))
        n = fila.provincias.count(",") + 1
        if power and n > 1:
            ax.text(fila.lon_c, y0 + 0.03, f"{n} provincias,\nla misma serie", ha="center", va="bottom",
                    fontsize=8, color=g.TINTA, zorder=6)
    if power:
        ax.scatter(puntos["lon"], puntos["lat"], s=34, color=g.AZUL, edgecolor=g.SUPERFICIE, linewidth=1.2, zorder=5)
        titulo = f"{len(puntos)} puntos de clima, pero solo {len(ctx.clima.tabla)} series distintas"
        subtitulo = "Cada recuadro es una celda de ~55 × 65 km del reanálisis que usa NASA POWER"
        punto = "Punto consultado (centro de provincia)"
    else:
        titulo = f"{len(ctx.clima.tabla)} series de clima ERA5-Land, una por celda de 0,1° (~11 km)"
        subtitulo = "El modelo interpola a cada celda de 500 m las cuatro series más cercanas"
        punto = None
    g.titulos(ax, titulo, subtitulo)
    piezas = [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=g.NARANJA, linewidth=1.2, label="Celda de clima")]
    if punto:
        piezas.insert(0, plt.Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=g.AZUL,
                                    markeredgecolor=g.SUPERFICIE, label=punto))
    ax.legend(handles=piezas, loc="lower left", fontsize=8.5)
    return g.guardar(fig, ruta)


def figura_suelo(crudo, ruta):
    """El salto de lluvia de junio de 2020 en NASA POWER y su efecto en la humedad del suelo (solo POWER)."""
    if mod_clima.fuente_de(crudo) != mod_clima.FUENTE_POWER:
        return None
    provincia = "Cercado" if "Cercado" in set(crudo["provincia"]) else crudo["provincia"].iloc[0]
    d = crudo[crudo["provincia"] == provincia].set_index("fecha")["humedad_suelo_raiz"]
    d = d[d > -900]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for anio, serie in d.groupby(d.index.year):
        if anio == 2020:
            continue
        ax.plot(serie.index.dayofyear, serie.to_numpy(), color=g.GRIS_CONTEXTO, linewidth=0.9, alpha=0.8)
    if 2020 in set(d.index.year):
        s = d[d.index.year == 2020]
        ax.plot(s.index.dayofyear, s.to_numpy(), color=g.NARANJA, linewidth=2.0)
        salto = pd.Timestamp("2020-06-04").dayofyear
        ax.annotate("4 de junio de 2020:\nla lluvia registrada salta a 704 mm en un día",
                    xy=(salto + 1, float(s.loc["2020-06-05"])), xytext=(salto + 28, float(s.max()) + 0.015),
                    fontsize=8.5, color=g.TINTA, va="center",
                    arrowprops={"arrowstyle": "-", "color": g.TINTA_2, "linewidth": 0.8})
        ax.text(s.index.dayofyear[-1] + 3, float(s.iloc[-1]), "2020", color=g.TINTA, fontsize=9, va="center")
    ax.set_xticks([1, 60, 121, 182, 244, 305], ["ene", "mar", "may", "jul", "sep", "nov"])
    ax.set_xlim(1, 380)
    ax.set_ylabel("Humedad del suelo en la zona de raíces (0–1)")
    g.titulos(ax, "Un error de lluvia contamina la humedad de suelo de toda la temporada 2020",
              f"NASA POWER, serie de {provincia}; en gris, los demás años de 2012 a 2025")
    return g.guardar(fig, ruta)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    g.estilo()
    ctx = flujo.cargar_todo()
    tablas, figuras = ctx.rutas.tablas, ctx.rutas.figuras
    tablas.mkdir(parents=True, exist_ok=True)
    geo = ctx.rutas.geo
    limite = json.loads((geo / "cochabamba_limite.geojson").read_text(encoding="utf-8"))
    provincias_gj = json.loads((geo / "cochabamba_provincias.geojson").read_text(encoding="utf-8"))

    focos_depto, por_mes, por_prov = tabla_focos(ctx, tablas)
    ctx.fuentes.round({"nocturnas": 3, "latitud": 4, "longitud": 4}).to_csv(
        tablas / "auditoria_fuentes_fijas.csv", index=False, encoding="utf-8-sig")
    dias, rachas, pocas = tabla_apagones(ctx, tablas)
    crudo, puntos, lluvia, anual = auditoria_clima(ctx, tablas)
    # el problema de lluvia de 2020 es de NASA POWER: se documenta con ese archivo aunque el modelo use ERA5-Land
    power = mod_clima.cargar_clima(ctx.rutas.clima_power) if ctx.rutas.clima_power.exists() else crudo
    est, cobertura = cobertura_capas(ctx, focos_depto, tablas)

    todas = pd.read_csv(ctx.rutas.fuegos_originales / "nasa_viirs_cochabamba_historico_2012_2025.csv") \
        if (ctx.rutas.fuegos_originales / "nasa_viirs_cochabamba_historico_2012_2025.csv").exists() else None
    cuenta = np.bincount(focos_depto["celda"].to_numpy(), minlength=ctx.malla.nfilas * ctx.malla.ncols)[ctx.idx]
    orden = np.sort(cuenta)[::-1].cumsum() / cuenta.sum()
    semana_celda = focos_depto.assign(semana=(focos_depto["fecha"] - pd.Timestamp("2012-01-02")).dt.days // 7) \
        .drop_duplicates(["semana", "celda"])
    ndvi_ruta = ctx.rutas.raw / "weather" / "ndvi_cochabamba_historico.csv"
    ndvi = pd.read_csv(ndvi_ruta) if ndvi_ruta.exists() else None
    resumen = {
        "celdas_departamento": ctx.n,
        "area_km2": round(float(ctx.malla.area_km2().ravel()[ctx.idx].sum())),
        "detecciones_archivo_usuario": None if todas is None else int(len(todas)),
        "detecciones_vegetacion_depto": int(len(focos_depto)),      # ya sin las fuentes de calor fijas
        "fuentes_de_calor_fijas": {
            "regla": f"{config.FUENTE_MIN_SEMANAS} o más semanas con detección y al menos "
                     f"{config.FUENTE_MIN_NOCTURNAS:.0%} de detecciones nocturnas",
            "celdas": int(len(ctx.fuentes)),
            "celdas_en_departamento": int(ctx.fuentes["en_departamento"].sum()),
            "detecciones_retiradas": int(ctx.fuentes["detecciones"].sum()),
            "celdas_semana_retiradas_en_departamento": int(
                ctx.fuentes.loc[ctx.fuentes["en_departamento"], "semanas"].sum()),
        },
        "periodo_focos": [str(focos_depto["fecha"].min().date()), str(focos_depto["fecha"].max().date())],
        "pct_focos_tropico": round(float(por_prov.loc[por_prov.index.isin(["Chapare", "Carrasco"]), "porcentaje"].sum()), 1),
        "celdas_con_fuego_alguna_vez_pct": round(100 * float((cuenta > 0).mean()), 1),
        "celda_semanas_positivas": int(len(semana_celda)),
        "tasa_celda_semana_pct": round(100 * len(semana_celda) / (ctx.n * int((ctx.dias_obs > 0).sum())), 4),
        "concentracion": {f"top_{p}pct_celdas": round(100 * float(orden[int(ctx.n * p / 100) - 1]), 1) for p in (1, 5, 10)},
        "dias_sin_satelite": int(len(dias)),
        "rachas_sin_satelite": [[str(r.desde.date()), str(r.hasta.date()), int(r.dias)]
                                for r in rachas.itertuples() if r.dias >= 2 or r.desde.month in (7, 8, 9, 10, 11)],
        "semanas_con_pocos_dias": pocas,
        "fuente_clima": ctx.clima.fuente,
        "puntos_clima": int(len(puntos)),
        "series_clima_distintas": int(len(ctx.clima.tabla)),
        "lluvia_maxima_registrada_mm": float(crudo["precipitacion"].max()),
        "valores_lluvia_mayor_100mm": int(len(lluvia)),
        "fechas_lluvia_mayor_100mm": sorted({str(f.date()) for f in lluvia["fecha"]}),
        "lluvia_anual_power_mm": {p: float(v) for p, v in anual.items()},
        "cobertura": cobertura.to_dict(orient="records"),
        "ndvi": None if ndvi is None else {"desde": str(ndvi["date"].min()), "hasta": str(ndvi["date"].max()),
                                           "unidad": "promedio por provincia", "decadas": int(ndvi["date"].nunique())},
    }
    (tablas / "auditoria_resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")

    figura_anio_mes(por_mes, figuras / "aud_focos_anio_mes.png")
    figura_cobertura(ctx, focos_depto, est, limite, cobertura, figuras / "aud_cobertura_capas.png")
    figura_clima(ctx, puntos, limite, provincias_gj, figuras / "aud_clima_celdas.png")
    figura_suelo(power, figuras / "aud_suelo_2020.png")
    print(json.dumps(resumen, indent=1, ensure_ascii=False))
    print(f"Tablas en {tablas} y figuras en {figuras}")


if __name__ == "__main__":
    main()
