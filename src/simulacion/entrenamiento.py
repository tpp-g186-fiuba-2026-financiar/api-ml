"""Reentrenamiento simulado: replica lo que hacen los crons de produccion.

- LSTM / XGBoost / Transformer (``retrain.yml``, lunes a viernes despues del
  cierre): se entrena un candidato con las ultimas ``history_days`` ruedas de
  todos los tickers y se promueve solo si supera al vigente
  (``src.retrain_guard.promote_if_better``, la misma funcion de produccion,
  sidecar ``.score.json`` incluido).
- Macro (``retrain-macro.yml``, el 1 de cada mes): se entrena con el universo
  ``UNIVERSE`` y se guarda solo si el AUC fuera de muestra llega al minimo.
- ARIMA no se entrena: se ajusta por ticker en cada prediccion.

Los artefactos van a un directorio propio de la corrida, nunca a ``models/``.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data import MACRO_COLUMN, attach_macro_feature
from src.data_quality import drop_stale
from src.errors import ApiMlError
from src.lstm import TrendModel
from src.macro_trend import UNIVERSE, MacroConfig, MacroTrendModel
from src.retrain_guard import force_promote, promote_if_better
from src.simulacion.snapshot import Snapshot
from src.train_macro import DEFAULT_MIN_AUC
from src.transformer import TransformerTrendModel
from src.xgb_trend import XGBTrendModel

# Minimo de ruedas por ticker para entrar al entrenamiento (``fetch_histories``).
MIN_RUEDAS_ENTRENAMIENTO = 60


def histories_at(snapshot: Snapshot, hasta: pd.Timestamp, days: int) -> dict[str, pd.DataFrame]:
    """Lo mismo que ``src.lstm.fetch_histories`` devolveria al cierre de ``hasta``."""
    rate = snapshot.rate_until(hasta)
    if rate is not None:
        rate = rate.iloc[-days:]
    raw: dict[str, pd.DataFrame] = {}
    for symbol in snapshot.tickers:
        df = snapshot.history(symbol, hasta, days)
        if len(df) >= MIN_RUEDAS_ENTRENAMIENTO:
            raw[symbol] = df
    return {
        symbol: attach_macro_feature(df, MACRO_COLUMN, rate)
        for symbol, df in _quiet_drop_stale(raw).items()
    }


def _quiet_drop_stale(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    with contextlib.redirect_stdout(io.StringIO()):
        return drop_stale(raw)


@dataclass
class ModeloDiario:
    """Un modelo de tendencia global con el gate de promocion de produccion."""

    name: str
    factory: Callable[[], object]
    loader: Callable[[Path], object]
    model_path: Path
    horizon: int = 5
    vigente: object | None = None

    def bootstrap(self, histories: dict[str, pd.DataFrame]) -> dict:
        """Modelo vigente al arrancar la simulacion (entrenado y guardado sin gate)."""
        candidate = self.factory()
        candidate.fit(histories)
        score = force_promote(candidate, str(self.model_path), histories, self.horizon)
        self.vigente = candidate
        return {"model": self.name, "promoted": True, "bootstrap": True, **score}

    def resume(self) -> None:
        """Recarga el vigente de disco al retomar una corrida cortada."""
        if self.model_path.exists():
            self.vigente = self.loader(self.model_path)

    def retrain(self, histories: dict[str, pd.DataFrame]) -> dict:
        candidate = self.factory()
        candidate.fit(histories)
        promoted, score = promote_if_better(
            candidate, str(self.model_path), self.loader, histories, self.horizon
        )
        if promoted:
            self.vigente = candidate
        return {"model": self.name, "promoted": promoted, **score}


def build_modelos_diarios(artifacts_dir: Path) -> list[ModeloDiario]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return [
        ModeloDiario("lstm", TrendModel, TrendModel.load, artifacts_dir / "lstm.pt"),
        ModeloDiario("xgboost", XGBTrendModel, XGBTrendModel.load, artifacts_dir / "xgb.pkl"),
        ModeloDiario(
            "transformer",
            TransformerTrendModel,
            TransformerTrendModel.load,
            artifacts_dir / "transformer.pt",
        ),
    ]


@dataclass
class ModeloMacro:
    """Modelo ``macro``: reentreno mensual con umbral minimo de AUC."""

    snapshot: Snapshot
    model_path: Path
    min_auc: float = DEFAULT_MIN_AUC
    vigente: MacroTrendModel | None = None
    _hasta: pd.Timestamp | None = field(default=None, repr=False)

    def _macro_provider(self) -> pd.DataFrame:
        # El vigente predice con la macro "de hoy": la del dia simulado.
        return self.snapshot.macro_frame(self._hasta)

    def set_today(self, hasta: pd.Timestamp) -> None:
        self._hasta = hasta
        if self.vigente is not None:
            self.vigente.macro_provider = self._macro_provider

    def _histories(self, hasta: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return {
            s: self.snapshot.history(s, hasta)
            for s in UNIVERSE
            if s in self.snapshot.prices and not self.snapshot.history(s, hasta).empty
        }

    def retrain(self, hasta: pd.Timestamp, force: bool = False) -> dict:
        if not self.snapshot.macro_raw:
            return {"model": "macro", "promoted": False, "error": "snapshot sin series macro"}
        model = MacroTrendModel(config=MacroConfig())
        try:
            metrics = model.fit(self._histories(hasta), self.snapshot.macro_frame(hasta))
        except ApiMlError as exc:
            return {"model": "macro", "promoted": False, "error": str(exc)}
        promoted = force or metrics["oos_auc"] >= self.min_auc
        if promoted:
            model.save(self.model_path)
            self.vigente = model
            self.set_today(hasta)
        return {
            "model": "macro",
            "promoted": promoted,
            "bootstrap": force,
            "oos_auc": metrics["oos_auc"],
            "min_auc": self.min_auc,
        }

    def resume(self) -> None:
        if self.model_path.exists():
            self.vigente = MacroTrendModel.load(self.model_path)


def macro_retrain_due(dia: pd.Timestamp, siguiente: pd.Timestamp | None) -> bool:
    """El cron corre el 1 de cada mes: le toca a la ultima rueda antes de ese dia.

    Si el 1 cae en una rueda, se entrena al cierre de ese mismo dia; si cae en
    fin de semana o feriado, con los datos de la rueda anterior.
    """
    if siguiente is None:
        return False
    primero = (dia + pd.offsets.MonthBegin(1)).normalize()
    if dia.day == 1:
        return True
    return primero < siguiente


# Frecuencias de reentreno de los modelos diarios. "diario" replica el cron de
# produccion (lunes a viernes); "semanal" reentrena solo en la ultima rueda de
# cada semana, para corridas mas cortas.
FRECUENCIAS_REENTRENO = ("diario", "semanal")


def reentreno_due(frecuencia: str, dia: pd.Timestamp, siguiente: pd.Timestamp | None) -> bool:
    """Si al cierre de ``dia`` se reentrenan LSTM / XGBoost / Transformer."""
    if frecuencia == "diario":
        return True
    if frecuencia == "semanal":
        return siguiente is None or siguiente.to_period("W") != dia.to_period("W")
    raise ValueError(f"frecuencia de reentreno desconocida: {frecuencia}")
