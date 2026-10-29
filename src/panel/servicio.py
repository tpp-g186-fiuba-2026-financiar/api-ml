"""El Transformer V2 servido por la API como modelo ``transformer``.

El V2 predice todas las acciones del panel lider juntas (``predict_panel``) y
en dolares CCL, asi que no entra en ``TrendService`` (una accion por vez con
el historico que baja la API). Este servicio:

- baja una vez por dia los historicos del panel y el CCL del data-colector
  (la misma fuente con la que se entrena, ver ``train_transformer_v2``) y
  predice el panel entero para cada fecha que le piden, con cache: el
  comparador, Black-Litterman y el paper trading piden ticker por ticker;
- al retorno relativo del V2 le suma la tendencia del panel
  (``tendencia_panel``) para que ``predicted_close`` quede en pesos, como el
  del resto de los modelos;
- saca la senal del orden relativo del dia (``senal_por_cuantil``): "alza" es
  el 15% del panel con mejor prediccion relativa, no "va a subir";
- para un ticker fuera del panel, o si el artefacto del V2 no cargo, responde
  el modelo de respaldo (el Transformer original, ``transformer-v1``), con
  ``model_version`` = ``respaldo:<version>``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import fetch_history_collector_or_yahoo
from src.data_quality import is_stale
from src.errors import ApiMlError, DataUnavailableError, NotEnoughDataError
from src.panel.gbm import RUEDAS_PREDICCION
from src.trend_common import derive_trend_output, senal_por_cuantil

# "alza" / "baja" para el 15% superior / inferior del panel (3 y 3 de 20), la
# configuracion de las simulaciones (``un_modelo --cuantil 0.15``).
CUANTIL_SENAL = 0.15
# Ruedas con las que se estima la tendencia del panel (un anio).
RUEDAS_TENDENCIA = 250


def tendencia_panel(
    historias: dict[str, pd.DataFrame], horizonte: int, ruedas: int = RUEDAS_TENDENCIA
) -> float:
    """Retorno log esperado del panel a ``horizonte`` ruedas, en pesos: el
    promedio entre acciones del retorno log diario medio de las ultimas
    ``ruedas``, por ``horizonte``.

    No es una prediccion del modelo (el V2 solo ordena las acciones entre si):
    es sobre todo el arrastre de la inflacion y del dolar, para que el precio
    predicho quede en la escala de los demas modelos.
    """
    diarios = []
    for df in historias.values():
        close = df["close"].to_numpy(dtype=np.float64)
        if len(close) > ruedas:
            diarios.append(np.log(close[-1] / close[-1 - ruedas]) / ruedas)
    return float(np.mean(diarios)) * horizonte if diarios else 0.0


class PanelTrendService:
    """Misma interfaz que ``TrendService`` (``load``, ``is_loaded``, ``version``,
    ``predict``, ``predict_on``, ``backtest_on``) para un modelo de panel."""

    def __init__(
        self,
        name: str,
        model_path: str,
        loader: Callable[[Path], object],
        respaldo,
        history_days: int,
        fetch_hist: Callable[[str, int], pd.DataFrame] = fetch_history_collector_or_yahoo,
        proveedor_factory: Callable[[], Callable] | None = None,
    ):
        self.name = name
        self.model_path = Path(model_path)
        self._loader = loader
        self.respaldo = respaldo
        self.history_days = history_days
        self._fetch_hist = fetch_hist
        self._proveedor_factory = proveedor_factory
        self._model = None
        # Black-Litterman pide las predicciones en paralelo: un solo hilo baja
        # los datos y predice el panel, el resto espera y usa la cache.
        self._lock = threading.Lock()
        self._dia_datos: str | None = None
        self._historias: dict[str, pd.DataFrame] = {}
        self._predicciones: dict[pd.Timestamp, dict[str, dict]] = {}

    def load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            self._model = self._loader(self.model_path)
        except ApiMlError as exc:
            print(f"  [warn] no se pudo cargar el modelo '{self.name}': {exc}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self.respaldo.is_loaded

    @property
    def version(self) -> str:
        if self._model is not None:
            return self._model.version
        return f"respaldo:{self.respaldo.version}"

    def _en_panel(self, symbol: str) -> bool:
        return self._model is not None and symbol in self._model.universo

    # ------------------------------------------------------------------ datos
    def _actualizar_datos(self) -> None:
        """Historicos del panel y CCL, una vez por dia."""
        hoy = date.today().isoformat()
        if self._dia_datos == hoy:
            return
        ruedas = self.history_days + RUEDAS_PREDICCION
        historias = {}
        for symbol in self._model.universo:
            try:
                historias[symbol] = self._fetch_hist(symbol, ruedas)
            except ApiMlError as exc:
                print(f"  [warn] {self.name}: sin historico de {symbol}: {exc}")
        if not historias:
            raise DataUnavailableError(f"{self.name}: no se pudo bajar ningun historico del panel")
        if self._proveedor_factory is not None:
            self._model.proveedor = self._proveedor_factory()
        self._historias = historias
        self._predicciones = {}
        self._dia_datos = hoy

    def _predecir(self, cortes: dict[str, pd.DataFrame]) -> dict[str, dict]:
        relativas = self._model.predict_panel(cortes)
        if not relativas:
            raise NotEnoughDataError(f"{self.name}: el panel no tiene datos suficientes ese dia")
        relativas = senal_por_cuantil(relativas, CUANTIL_SENAL)
        horizonte = self._model.config.horizonte
        tendencia = tendencia_panel(cortes, horizonte)
        salida = {}
        for symbol, r in relativas.items():
            relativo = float(np.log(r["predicted_close"] / r["last_close"]))
            resultado = derive_trend_output(cortes[symbol], relativo + tendencia, horizonte)
            resultado["signal"] = r["signal"]
            salida[symbol] = resultado
        return salida

    def _panel(self, as_of: pd.Timestamp | None) -> dict[str, dict]:
        """Predicciones del panel con los datos hasta ``as_of`` (None: hasta hoy)."""
        with self._lock:
            self._actualizar_datos()
            cortes = {
                s: df if as_of is None else df[df.index <= as_of]
                for s, df in self._historias.items()
            }
            cortes = {s: df for s, df in cortes.items() if not df.empty}
            if not cortes:
                raise NotEnoughDataError(f"{self.name}: sin datos del panel hasta {as_of}")
            dia = max(df.index[-1] for df in cortes.values())
            if dia not in self._predicciones:
                self._predicciones[dia] = self._predecir(cortes)
            return self._predicciones[dia]

    # ------------------------------------------------------------- prediccion
    def _etiquetar(self, resultado: dict, symbol: str, version: str) -> dict:
        resultado = dict(resultado)
        resultado["symbol"] = symbol
        resultado["model"] = self.name
        resultado["model_version"] = version
        return resultado

    def _de_respaldo(self, resultado: dict) -> dict:
        resultado["model"] = self.name
        resultado["model_version"] = f"respaldo:{resultado['model_version']}"
        return resultado

    def _panel_o_nada(self, as_of: pd.Timestamp | None) -> dict[str, dict]:
        """Como ``_panel``, pero sin datos del panel (data-colector caido, sin
        CCL) devuelve {} y el pedido lo responde el respaldo."""
        try:
            return self._panel(as_of)
        except DataUnavailableError as exc:
            print(f"  [warn] {self.name}: sin datos del panel, responde el respaldo: {exc}")
            return {}

    def predict(self, symbol: str) -> dict:
        symbol = symbol.strip().upper()
        if not self._en_panel(symbol):
            return self._de_respaldo(self.respaldo.predict(symbol))
        resultado = self._panel_o_nada(None).get(symbol)
        if resultado is None:
            return self._de_respaldo(self.respaldo.predict(symbol))
        if is_stale(self._historias[symbol]):
            raise NotEnoughDataError(
                f"{symbol}: la serie casi no tiene movimiento de precio (datos estancados)"
            )
        return self._etiquetar(resultado, symbol, self._model.version)

    def predict_on(self, df: pd.DataFrame, symbol: str) -> dict:
        """Con el historico que bajo la API solo se usa su ultima fecha: el V2
        predice con los datos del panel hasta ese dia (asi el backtest
        walk-forward no mira el futuro)."""
        symbol = symbol.strip().upper()
        if not self._en_panel(symbol):
            return self._de_respaldo(self.respaldo.predict_on(df, symbol))
        if is_stale(df):
            raise NotEnoughDataError(
                f"{symbol}: la serie casi no tiene movimiento de precio (datos estancados)"
            )
        resultado = self._panel_o_nada(df.index[-1]).get(symbol)
        if resultado is None:
            return self._de_respaldo(self.respaldo.predict_on(df, symbol))
        return self._etiquetar(resultado, symbol, self._model.version)

    def backtest_on(self, df: pd.DataFrame, horizon: int) -> dict | None:
        """Sin backtest en vivo (el comparador solo lo pide para lstm / xgboost)."""
        return None
