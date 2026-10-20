"""Mesa de operaciones: con las predicciones de cada rueda, rebalancea y valoriza las carteras.

Separada del entrenamiento a proposito: la simulacion la alimenta con las
predicciones recien calculadas, y ``rearmar`` con las guardadas en
``predicciones.jsonl`` / ``backtests.jsonl``. Asi, cambiar una regla de
cartera (comision, frecuencia de rebalanceo, una estrategia) no obliga a
reentrenar los modelos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.errors import ApiMlError
from src.simulacion.cartera import Cartera
from src.simulacion.estrategias import (
    PERFILES,
    GarchCache,
    consenso_por_ticker,
    equal_weight,
    mejor_modelo_por_ticker,
    normalizar,
    pesos_black_litterman,
    pesos_consenso,
    pesos_cuantil,
    pesos_histeresis,
    pesos_inclinacion,
    pesos_mantener,
    pesos_senal,
    predicciones_del_mejor,
)
from src.simulacion.snapshot import Snapshot

# Modelos con backtest en vivo en el comparador (``_LIVE_BACKTEST_MODELS`` en
# src.registry): su accuracy pondera el consenso.
MODELOS_CON_BACKTEST = ("lstm", "xgboost")
MODELOS = ("lstm", "xgboost", "transformer", "arima", "macro")
# Candidatos de la eleccion del mejor modelo por ticker: "prod" replica la
# eleccion de produccion (solo los que tienen backtest en vivo); "todos" suma
# el resto de los locales.
SELECCIONES = {"prod": MODELOS_CON_BACKTEST, "todos": MODELOS}
# Modelos de panel (``predict_panel``) con predicciones generadas aparte
# (``predicciones_panel``). Solo arman sus propias carteras: no entran en el
# consenso ni en la eleccion del mejor modelo (no tienen backtest en vivo), asi
# que sumarlos no cambia ninguna cartera de ``MODELOS``.
MODELOS_PANEL = ("ensamble", "panel_transformer", "reversion")
# Reglas de cartera que rotan menos que ``senal_`` (solo para modelos de panel):
# prefijo -> descripcion.
REGLAS_PANEL = {
    "histeresis": "compra el 15% superior y mantiene mientras siga en el 30% superior",
    "cuantil30": "partes iguales en el 30% superior",
    "inclinado": "todo el panel, 1,5 partes a las 'alza' y 0,5 a las 'baja'",
}
# Solo en la mesa simple (``completa=False``, simulacion de un solo modelo).
REGLAS_SIMPLES = {
    "mantiene": "compra las 'alza', mantiene las 'neutral' que ya tenia y vende solo las 'baja'",
}
CUANTIL_ENTRAR, CUANTIL_MANTENER = 0.15, 0.30
INCLINACION = 0.5
BENCHMARK_BUY_HOLD = "benchmark_equal_weight_buy_hold"
BENCHMARK_REBALANCEO = "benchmark_equal_weight_rebalanceo"
BENCHMARK_MERVAL = "benchmark_merval"

Predicciones = dict[str, dict[str, dict]]  # modelo -> ticker -> prediccion o backtest


def nombres_estrategias(modelos_panel: tuple[str, ...] = (), completa: bool = True) -> list[str]:
    """Carteras de la mesa; las de ``modelos_panel`` van al final, despues de los benchmarks.

    Sin ``completa`` quedan solo los benchmarks y, de cada modelo de panel, la
    senal y las ``REGLAS_PANEL`` (sin Black-Litterman, consenso ni mejor modelo).
    """
    nombres = []
    if completa:
        nombres += [f"senal_{m}" for m in MODELOS]
        nombres += [f"bl_{m}_{p.value}" for m in MODELOS for p in PERFILES]
        nombres += [f"consenso_{p.value}" for p in PERFILES]
        for seleccion in SELECCIONES:
            nombres.append(f"senal_mejor_{seleccion}")
            nombres += [f"bl_mejor_{seleccion}_{p.value}" for p in PERFILES]
    nombres += [BENCHMARK_BUY_HOLD, BENCHMARK_REBALANCEO]
    for m in modelos_panel:
        nombres.append(f"senal_{m}")
        if completa:
            nombres += [f"bl_{m}_{p.value}" for p in PERFILES]
        else:
            nombres += [f"{regla}_{m}" for regla in REGLAS_SIMPLES]
        nombres += [f"{regla}_{m}" for regla in REGLAS_PANEL]
    return nombres


@dataclass
class ResultadoRueda:
    rebalanceo: bool
    objetivos: dict[str, dict[str, float]] = field(default_factory=dict)
    operaciones: list[dict] = field(default_factory=list)
    errores: list[dict] = field(default_factory=list)
    elecciones: list[dict] = field(default_factory=list)
    valores: dict = field(default_factory=dict)


class Mesa:
    def __init__(
        self,
        snapshot: Snapshot,
        capital: float,
        comision: float,
        rebalanceo: int,
        modelos_panel: tuple[str, ...] = (),
        completa: bool = True,
    ):
        """Sin ``completa`` (simulacion de un solo modelo, ``un_modelo``) los
        ``modelos_panel`` pueden tener cualquier nombre y solo se arman sus
        carteras simples y los benchmarks (ver ``nombres_estrategias``)."""
        desconocidos = set(modelos_panel) - set(MODELOS_PANEL)
        if completa and desconocidos:
            raise ValueError(f"modelos de panel desconocidos: {sorted(desconocidos)}")
        self.snapshot = snapshot
        self.capital = capital
        self.rebalanceo = rebalanceo
        self.modelos_panel = tuple(modelos_panel)
        self.completa = completa
        self.carteras = {
            n: Cartera(n, efectivo=capital, comision=comision)
            for n in nombres_estrategias(self.modelos_panel, completa)
        }
        self.merval_base: float | None = None
        self.ruedas = 0

    def toca_rebalancear(self) -> bool:
        return self.ruedas % self.rebalanceo == 0

    def precios(self, dia: pd.Timestamp) -> dict[str, float]:
        """Cierre de ``dia`` (o el ultimo conocido) de cada ticker del panel lider."""
        return {
            t: p
            for t in PANEL_LIDER_TICKERS
            if t in self.snapshot.prices and (p := self.snapshot.close_on(t, dia)) is not None
        }

    def objetivos(
        self,
        dia: pd.Timestamp,
        predicciones: Predicciones,
        backtests: Predicciones,
        universo: list[str],
    ) -> tuple[dict[str, dict[str, float]], list[dict], list[dict]]:
        """Pesos objetivo de cada cartera para el rebalanceo de ``dia``, errores
        de Black-Litterman y el modelo elegido en cada ticker por cada seleccion.
        """
        objetivos: dict[str, dict[str, float]] = {}
        errores: list[dict] = []
        garch = GarchCache()

        def fetch_hist(ticker: str, days: int) -> pd.DataFrame:
            return self.snapshot.history(ticker, dia, days)

        def senal_y_bl(sufijo: str, preds: dict[str, dict]) -> None:
            objetivos[f"senal_{sufijo}"] = pesos_senal(preds, universo)
            for perfil in PERFILES:
                nombre = f"bl_{sufijo}_{perfil.value}"
                try:
                    pesos = pesos_black_litterman(preds, perfil, fetch_hist, garch, universo)
                    objetivos[nombre] = normalizar(pesos)
                except (ApiMlError, ValueError, ArithmeticError) as exc:
                    errores.append({"fecha": dia.date(), "cartera": nombre, "error": str(exc)})

        if self.completa:
            for modelo in MODELOS:
                senal_y_bl(modelo, predicciones.get(modelo, {}))
        for modelo in self.modelos_panel:
            preds = predicciones.get(modelo, {})
            if self.completa:
                senal_y_bl(modelo, preds)
            else:
                objetivos[f"senal_{modelo}"] = pesos_senal(preds, universo)
                tenencias = set(self.carteras[f"mantiene_{modelo}"].tenencias)
                objetivos[f"mantiene_{modelo}"] = pesos_mantener(preds, universo, tenencias)
            tenencias = set(self.carteras[f"histeresis_{modelo}"].tenencias)
            objetivos[f"histeresis_{modelo}"] = pesos_histeresis(
                preds, universo, tenencias, CUANTIL_ENTRAR, CUANTIL_MANTENER
            )
            objetivos[f"cuantil30_{modelo}"] = pesos_cuantil(preds, universo, CUANTIL_MANTENER)
            objetivos[f"inclinado_{modelo}"] = pesos_inclinacion(preds, universo, INCLINACION)

        elecciones: list[dict] = []
        if self.completa:
            for seleccion, candidatos in SELECCIONES.items():
                elegidos = mejor_modelo_por_ticker(backtests, candidatos, universo)
                senal_y_bl(f"mejor_{seleccion}", predicciones_del_mejor(predicciones, elegidos))
                elecciones += [
                    {"fecha": dia.date(), "seleccion": seleccion, "symbol": t, **e}
                    for t, e in elegidos.items()
                ]

            locales = {m: predicciones.get(m, {}) for m in MODELOS}
            for perfil in PERFILES:
                consenso = consenso_por_ticker(locales, universo, perfil)
                objetivos[f"consenso_{perfil.value}"] = pesos_consenso(consenso)

        objetivos[BENCHMARK_REBALANCEO] = equal_weight(universo)
        if self.ruedas == 0:
            objetivos[BENCHMARK_BUY_HOLD] = equal_weight(universo)
        return objetivos, errores, elecciones

    def operar(
        self, dia: pd.Timestamp, predicciones: Predicciones, backtests: Predicciones
    ) -> ResultadoRueda:
        """Rebalancea al cierre de ``dia`` si corresponde y valoriza todas las carteras."""
        precios = self.precios(dia)
        resultado = ResultadoRueda(rebalanceo=self.toca_rebalancear())
        if resultado.rebalanceo:
            objetivos, errores, elecciones = self.objetivos(
                dia, predicciones, backtests, sorted(precios)
            )
            resultado.objetivos, resultado.errores, resultado.elecciones = (
                objetivos,
                errores,
                elecciones,
            )
            for nombre, pesos in objetivos.items():
                for op in self.carteras[nombre].rebalancear(pesos, precios):
                    resultado.operaciones.append({"fecha": dia.date(), "cartera": nombre, **op})

        merval = self.snapshot.merval_on(dia)
        if self.merval_base is None:
            self.merval_base = merval
        valores = {n: c.valor(precios) for n, c in self.carteras.items()}
        if merval is not None and self.merval_base:
            valores[BENCHMARK_MERVAL] = self.capital * merval / self.merval_base
        resultado.valores = {"fecha": dia.date(), "ccl": self.snapshot.ccl_on(dia), **valores}
        self.ruedas += 1
        return resultado

    def to_dict(self) -> dict:
        return {
            "ruedas_procesadas": self.ruedas,
            "merval_base": self.merval_base,
            "carteras": {n: c.to_dict() for n, c in self.carteras.items()},
        }

    def cargar(self, estado: dict) -> None:
        self.ruedas = estado["ruedas_procesadas"]
        self.merval_base = estado["merval_base"]
        self.carteras = {n: Cartera.from_dict(c) for n, c in estado["carteras"].items()}
