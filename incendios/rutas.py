"""Dónde está cada cosa. Estructura de Cookiecutter Data Science, todo dentro del proyecto:

    data/raw/        datos tal como llegaron (el zip original); el proyecto nunca los modifica
    data/external/   lo que baja `python -m incendios descargar`
    data/interim/    cachés: se pueden borrar y se rehacen solos
    data/processed/  muestra de entrenamiento y predicciones de validación y prueba
    models/          modelos entrenados
    reports/         resultados, tablas, figuras, mapas, visor y predicciones semanales

Para comparar corridas: INCENDIOS_DATOS cambia la carpeta de datos de entrada (la que tiene raw/ y
external/) e INCENDIOS_SALIDAS la carpeta donde va todo lo producido, con su propio data/interim,
data/processed, models/ y reports/.
"""
import os
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO_CLIMA_POWER = "clima_suelo_diario_cochabamba_2012_2025.csv"
ARCHIVO_CLIMA_NATIVO = "clima_suelo_diario_cochabamba_malla_nativa.csv"
ARCHIVO_CLIMA_ERA5 = "clima_era5land_diario.csv.gz"


@dataclass(frozen=True)
class Rutas:
    datos: Path        # tiene raw/ y external/
    salida: Path       # tiene data/interim, data/processed, models/ y reports/

    @classmethod
    def del_proyecto(cls, entorno=None):
        entorno = os.environ if entorno is None else entorno
        return cls(datos=Path(entorno.get("INCENDIOS_DATOS") or RAIZ / "data"),
                   salida=Path(entorno.get("INCENDIOS_SALIDAS") or RAIZ))

    @classmethod
    def en(cls, carpeta):
        """La misma estructura dentro de otra carpeta (pruebas, corridas aparte)."""
        carpeta = Path(carpeta)
        return cls(datos=carpeta / "data", salida=carpeta)

    # ------------------------------------------------------------------ entrada
    @property
    def raw(self):
        return self.datos / "raw"

    @property
    def external(self):
        return self.datos / "external"

    @property
    def geo(self):
        return self.raw / "geo"

    @property
    def focos(self):
        """Carpetas con CSV de focos: los originales y los que se bajan, se leen juntos."""
        return (self.raw / "focos", self.focos_bajados)

    @property
    def focos_bajados(self):
        return self.external / "focos"

    @property
    def relieve(self):
        return (self.raw / "terrain", self.relieve_bajado)

    @property
    def relieve_bajado(self):
        return self.external / "terrain"

    @property
    def cobertura(self):
        return (self.raw / "landcover", self.cobertura_bajada)

    @property
    def cobertura_bajada(self):
        return self.external / "landcover"

    @property
    def fuegos_originales(self):
        return self.raw / "fires"

    @property
    def clima_power(self):
        return self.raw / "weather" / ARCHIVO_CLIMA_POWER

    @property
    def clima_power_nativo(self):
        return self.external / "weather" / ARCHIVO_CLIMA_NATIVO

    @property
    def clima_era5(self):
        return self.external / "weather" / ARCHIVO_CLIMA_ERA5

    @property
    def era5_meses(self):
        return self.external / "weather" / "era5land"

    @property
    def capas(self):
        return self.external / "capas"

    # ------------------------------------------------------------------ lo que se produce
    @property
    def interim(self):
        return self.salida / "data" / "interim"

    @property
    def processed(self):
        return self.salida / "data" / "processed"

    @property
    def modelos(self):
        return self.salida / "models"

    @property
    def informes(self):
        return self.salida / "reports"

    @property
    def estaticas(self):
        return self.interim / "estaticas.npz"

    @property
    def muestra(self):
        return self.processed / "muestra"

    @property
    def predicciones(self):
        return self.processed / "predicciones"

    @property
    def tablas(self):
        return self.informes / "tablas"

    @property
    def figuras(self):
        return self.informes / "figuras"

    @property
    def mapas(self):
        return self.informes / "mapas"

    @property
    def resultados(self):
        return self.informes / "RESULTADOS.md"

    @property
    def visor(self):
        return self.informes / "visor_riesgo.html"

    @property
    def semanal(self):
        return self.informes / "semanal"

    @property
    def anteriores(self):
        return self.informes / "anteriores"
