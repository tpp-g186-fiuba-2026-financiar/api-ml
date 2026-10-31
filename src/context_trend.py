"""Modelo "contexto": alza/baja a 20 ruedas usando el contexto del mercado local.

Los modelos que miran solo la serie de la accion miden ~azar (AUC ~0.51 en
validacion walk-forward). Lo unico que mejoro de forma consistente fue sumar
contexto del mercado: hacia donde viene la canasta
de lideres, cuan abierta o cerrada esta la suba (amplitud) y como viene la
accion frente a esa canasta (fuerza relativa y beta). Ensamble de arboles con
peso decreciente en el tiempo y senal solo en los extremos, con umbrales
calibrados fuera de muestra.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.data import fetch_history
from src.data_quality import drop_stale, is_stale
from src.errors import ApiMlError, NotEnoughDataError, StaleArtifactError
from src.macro_trend import TECH_FEATURES, UNIVERSE, technical_features
from src.trend_common import (
    calibrated_probability,
    derive_trend_output,
    fit_probability_calibration,
    weak_signal_sides,
)

HORIZON = 20
MIN_ROWS = 260
# Canasta que hace de proxy del Merval: fija (no depende de la fecha), asi el
# contexto es identico en entrenamiento y en prediccion.
LEADERS = ("GGAL", "YPFD", "BMA", "PAMP", "TXAR", "ALUA", "TECO2", "CEPU")
HALF_LIFE_YEARS = 3.0  # los datos recientes pesan mas al entrenar
ENSEMBLE_SIZE = 6

CONTEXT_FEATURES = [
    "m_ret5", "m_ret20", "m_ret60", "m_vol20", "m_dd252", "m_breadth50",
    "m_rel5", "m_rel20", "m_rel60", "m_beta60",
]  # fmt: skip
FEATURES = TECH_FEATURES + CONTEXT_FEATURES


def market_frame(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Serie del mercado (canasta de lideres) a partir de sus historicos."""
    closes = pd.DataFrame({s: d["close"] for s, d in histories.items() if s in LEADERS})
    if closes.shape[1] < 3:
        raise NotEnoughDataError("faltan historicos de los lideres para armar el contexto")
    returns = np.log(closes).diff()
    mkt = returns.mean(axis=1).fillna(0.0)
    index = mkt.cumsum()
    frame = pd.DataFrame(index=closes.index)
    frame["m_lr"] = mkt
    for k in (5, 20, 60):
        frame[f"m_ret{k}"] = index.diff(k)
    frame["m_vol20"] = mkt.rolling(20).std()
    frame["m_dd252"] = index - index.rolling(252, min_periods=100).max()
    frame["m_breadth50"] = (closes > closes.rolling(50).mean()).where(closes.notna()).mean(axis=1)
    return frame


def context_features(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Variables de contexto de una accion, alineadas a su calendario."""
    m = market.reindex(df.index, method="ffill")
    log_c = np.log(df["close"])
    r = log_c.diff()
    f = pd.DataFrame(index=df.index)
    for k in (5, 20, 60):
        f[f"m_ret{k}"] = m[f"m_ret{k}"]
    f["m_vol20"], f["m_dd252"], f["m_breadth50"] = m["m_vol20"], m["m_dd252"], m["m_breadth50"]
    for k in (5, 20, 60):
        f[f"m_rel{k}"] = log_c.diff(k) - m[f"m_ret{k}"]
    f["m_beta60"] = r.rolling(60).cov(m["m_lr"]) / m["m_lr"].rolling(60).var()
    return f.replace([np.inf, -np.inf], np.nan)


def build_features(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Variables de una accion, en el orden de ``FEATURES``."""
    return technical_features(df).join(context_features(df, market))[FEATURES]


_market_cache: dict[str, pd.DataFrame] = {}


def todays_market_frame(days: int = 1500) -> pd.DataFrame:
    """Contexto de mercado con los historicos actuales de los lideres (cacheado por dia)."""
    key = dt.date.today().isoformat()
    if key in _market_cache:
        return _market_cache[key]
    histories = {}
    for symbol in LEADERS:
        try:
            histories[symbol] = fetch_history(symbol, days)
        except ApiMlError as exc:
            print(f"  [warn] contexto: sin historico de {symbol}: {exc}")
    frame = market_frame(histories)
    _market_cache.clear()
    _market_cache[key] = frame
    return frame


@dataclass
class ContextConfig:
    horizon: int = HORIZON
    signal_quantile: float = 0.15  # "alza"/"baja" solo en el 15% mas extremo de cada lado
    oos_fraction: float = 0.25  # ultimo tramo de fechas usado para medir y calibrar
    oos_folds: int = 4
    seed: int = 0


def _xgb():
    """``src.xgb_trend`` importa este modulo: se importa recien al usarlo."""
    from src import xgb_trend

    return xgb_trend


def _features() -> list[str]:
    """Las variables del modelo: las mismas de XGBoost (con sus rangos)."""
    return _xgb().ALL_FEATURES


def _fit_ensemble(x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int) -> list:
    models = []
    for k in range(ENSEMBLE_SIZE):
        model = HistGradientBoostingClassifier(
            max_depth=3 + (k % 2), learning_rate=0.05, max_iter=150, min_samples_leaf=200,
            l2_regularization=1.0, max_features=0.5, random_state=seed + k,
        )  # fmt: skip
        models.append(model.fit(x, y, sample_weight=weights))
    return models


def _score(models: list, x: np.ndarray) -> np.ndarray:
    """P(baja): promedio del ensamble."""
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


def _decay_weights(index: pd.Index, cutoff: pd.Timestamp) -> np.ndarray:
    age_years = (cutoff - pd.DatetimeIndex(index)).days.to_numpy() / 365.0
    return 0.5 ** (age_years / HALF_LIFE_YEARS)


@dataclass
class ContextTrendModel:
    config: ContextConfig = field(default_factory=ContextConfig)
    tickers: list[str] = field(default_factory=list)
    trained_at: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    baja_threshold: float | None = None  # P(baja) >= umbral -> "baja"
    alza_threshold: float | None = None  # P(baja) <= umbral -> "alza"
    alza_move: float = 0.0  # retorno log tipico cuando la senal de alza acierta
    baja_move: float = 0.0  # retorno log tipico cuando la senal de baja acierta (negativo)
    # Universo de acciones del dia (ver ``xgb_trend.todays_universe``): de ahi
    # salen el contexto de mercado y el puesto de la accion frente a sus pares.
    universe_provider: Callable[[], object] | None = None
    # P(sube) del modelo -> frecuencia real fuera de muestra (ver fit_probability_calibration).
    calibration: dict | None = None
    _models: list | None = None

    @property
    def version(self) -> str:
        return "untrained" if self.trained_at is None else f"contexto-{self.trained_at}"

    def _panel(self, histories: dict[str, pd.DataFrame], market: pd.DataFrame) -> pd.DataFrame:
        # Mismas variables que XGBoost (tecnicas + contexto + distancia a la
        # tendencia + puesto frente a las otras acciones ese dia): con ellas el
        # AUC por mes paso de 0,573 a 0,589 en el walk-forward 2016-2026
        # (diferencia pareada +0,016, IC95 [+0,000; +0,031]). Cambia el algoritmo.
        xgb_trend = _xgb()
        panel = xgb_trend.XGBTrendModel(xgb_trend.XGBConfig(horizon=self.config.horizon))._panel(
            histories, market
        )
        panel["down"] = (panel["fwd"] < 0).astype(int)
        return panel

    def _oos_scores(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Predicciones fuera de muestra del ultimo tramo, con reentrenos sucesivos."""
        cfg = self.config
        days = np.sort(panel.index.unique())
        first = int(len(days) * (1 - cfg.oos_fraction))
        edges = np.linspace(first, len(days), cfg.oos_folds + 1).astype(int)
        index, x, y = panel.index.values, panel[_features()].to_numpy(), panel["down"].to_numpy()
        parts = []
        for k in range(cfg.oos_folds):
            start = days[edges[k]]
            end = days[edges[k + 1]] if edges[k + 1] < len(days) else None
            purge = days[max(edges[k] - cfg.horizon, 0)]
            tr = np.where(index <= purge)[0]
            te = np.where((index >= start) & ((index < end) if end is not None else True))[0]
            if len(tr) < 3000 or len(te) == 0:
                continue
            weights = _decay_weights(panel.index[tr], pd.Timestamp(purge))
            models = _fit_ensemble(x[tr], y[tr], weights, cfg.seed)
            parts.append(
                pd.DataFrame(
                    {"p": _score(models, x[te]), "down": y[te], "fwd": panel["fwd"].to_numpy()[te]},
                    index=panel.index[te],
                )
            )
        if not parts:
            raise NotEnoughDataError("no hay historia suficiente para medir fuera de muestra")
        return pd.concat(parts)

    def _apply_signal_edge(self) -> None:
        """Apaga el lado (alza/baja) cuya ventaja fuera de muestra es chica."""
        alza_weak, baja_weak = weak_signal_sides(self.metrics)
        if alza_weak:
            self.alza_threshold = float("-inf")
            self.metrics["alza_disabled"] = 1.0
        if baja_weak:
            self.baja_threshold = float("inf")
            self.metrics["baja_disabled"] = 1.0

    def fit_panel(self, panel: pd.DataFrame) -> dict[str, float]:
        """Calibra umbrales fuera de muestra y entrena el modelo final."""
        cfg = self.config
        oos = self._oos_scores(panel)
        p, down = oos["p"].to_numpy(), oos["down"].to_numpy()
        q = cfg.signal_quantile
        self.baja_threshold = float(np.quantile(p, 1 - q))
        self.alza_threshold = float(np.quantile(p, q))
        fwd_log = np.log1p(oos["fwd"].to_numpy())
        is_baja, is_alza = p >= self.baja_threshold, p <= self.alza_threshold
        downs, ups = fwd_log[is_baja & (fwd_log < 0)], fwd_log[is_alza & (fwd_log > 0)]
        self.baja_move = float(np.median(downs)) if len(downs) else 0.0
        self.alza_move = float(np.median(ups)) if len(ups) else 0.0
        self.calibration = fit_probability_calibration(1 - p, 1 - down)
        baja_hits = float(down[is_baja].mean()) if is_baja.any() else float("nan")
        alza_hits = float(1 - down[is_alza].mean()) if is_alza.any() else float("nan")
        # Un lado solo se emite si fuera de muestra le gano a su tasa base: una
        # senal que acierta menos que no opinar no tiene que llegar al usuario.
        baja_off = not baja_hits > float(down.mean())
        alza_off = not alza_hits > float(1 - down.mean())
        if baja_off:
            self.baja_threshold = float("inf")
        if alza_off:
            self.alza_threshold = float("-inf")
        self.metrics = {
            "oos_samples": float(len(oos)),
            "oos_auc": round(float(roc_auc_score(down, p)), 4),
            "oos_down_rate": round(float(down.mean()), 4),
            "oos_baja_samples": float(is_baja.sum()),
            "oos_baja_hit_rate": round(baja_hits, 4),
            "oos_alza_samples": float(is_alza.sum()),
            "oos_alza_hit_rate": round(alza_hits, 4),
            "baja_disabled": float(baja_off),
            "alza_disabled": float(alza_off),
        }
        self._apply_signal_edge()
        latest = pd.Timestamp(panel.index.max())
        weights = _decay_weights(panel.index, latest)
        self._models = _fit_ensemble(
            panel[_features()].to_numpy(), panel["down"].to_numpy(), weights, cfg.seed
        )
        self.trained_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        return self.metrics

    def fit(self, histories: dict[str, pd.DataFrame]) -> dict[str, float]:
        market = market_frame(histories)
        histories = {s: d for s, d in drop_stale(histories).items() if len(d) >= MIN_ROWS}
        if not histories:
            raise NotEnoughDataError("no hay suficientes datos para entrenar")
        panel = self._panel(histories, market)
        if len(panel) < 3000:
            raise NotEnoughDataError(f"muy pocas filas para entrenar ({len(panel)})")
        self.tickers = sorted(histories)
        return self.fit_panel(panel)

    def probability_down(self, df: pd.DataFrame) -> float:
        if self._models is None:
            raise NotEnoughDataError("el modelo no esta entrenado")
        if len(df) < MIN_ROWS:
            raise NotEnoughDataError(f"se necesitan al menos {MIN_ROWS} ruedas, hay {len(df)}")
        if is_stale(df):
            raise NotEnoughDataError(
                "la serie casi no tiene movimiento de precio (>30% de ruedas sin cambio)"
            )
        xgb_trend = _xgb()
        universe = (self.universe_provider or xgb_trend.todays_universe)()
        # Solo se usan filas hasta la ultima fecha de ``df`` (el contexto es "as of"):
        # asi el backtest con cutoff no mira el futuro.
        own = xgb_trend.build_features(df, universe.market.loc[: df.index[-1]]).iloc[[-1]]
        row = own.join(xgb_trend.rank_against_peers(own, universe))[_features()].to_numpy()
        return float(_score(self._models, row)[0])

    def day_cuts(self, asof: pd.Timestamp) -> tuple[float, float] | None:
        """Cortes de P(baja) del ranking de ``asof`` (misma regla que XGBoost:
        ``xgb_trend.SIGNAL_SHARE`` de cada punta), cacheados."""
        xgb_trend = _xgb()
        universe = (self.universe_provider or xgb_trend.todays_universe)()
        key = (id(universe), pd.Timestamp(asof))
        cache = self.__dict__.setdefault("_cuts_cache", {})
        if key not in cache:
            rows = xgb_trend.peer_rows(universe, pd.Timestamp(asof))
            scores = _score(self._models, rows[_features()].to_numpy()) if len(rows) else []
            if len(cache) > 64:
                cache.clear()
            cache[key] = xgb_trend.ranking_cuts(scores)
        return cache[key]

    def predict_df(self, df: pd.DataFrame) -> dict:
        p_down = self.probability_down(df)
        cuts = self.day_cuts(df.index[-1])
        # Con ranking del dia: baja si esta en el 20% con mas P(baja), alza en
        # el 20% con menos. Sin ranking, los umbrales fijos de entrenamiento.
        baja_cut, alza_cut = cuts if cuts else (self.baja_threshold, self.alza_threshold)
        if p_down >= float(baja_cut):
            signal, log_return = "baja", self.baja_move
        elif p_down <= float(alza_cut):
            signal, log_return = "alza", self.alza_move
        else:
            signal, log_return = "neutral", 0.0  # sin vista: precio proyectado = ultimo cierre
        out = derive_trend_output(df, log_return, self.config.horizon, neutral_band=0.0)
        out["signal"] = signal
        out["probability_down"] = round(p_down, 4)
        out["probability_up"] = round(1 - p_down, 4)
        out["probability_up_calibrated"] = calibrated_probability(self.calibration, 1 - p_down)
        down_rate = self.metrics.get("oos_down_rate")
        out["base_up_rate"] = (self.calibration or {}).get(
            "base_up", None if down_rate is None else round(1 - float(down_rate), 4)
        )
        return out

    def save(self, path: str | Path) -> None:
        if self._models is None:
            raise NotEnoughDataError("no hay un modelo entrenado para guardar")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": self._models,
                "config": self.config.__dict__,
                "tickers": self.tickers,
                "trained_at": self.trained_at,
                "metrics": self.metrics,
                "baja_threshold": self.baja_threshold,
                "alza_threshold": self.alza_threshold,
                "alza_move": self.alza_move,
                "baja_move": self.baja_move,
                "calibration": self.calibration,
                "feature_names": _features(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> ContextTrendModel:
        blob = joblib.load(Path(path))
        if blob.get("feature_names") != _features():
            raise StaleArtifactError(
                "el artefacto 'contexto' se entreno con otras variables: hay que reentrenarlo"
            )
        try:
            model = cls(
                config=ContextConfig(**blob["config"]),
                tickers=blob.get("tickers", []),
                trained_at=blob.get("trained_at"),
                metrics=blob.get("metrics", {}),
                baja_threshold=blob["baja_threshold"],
                alza_threshold=blob["alza_threshold"],
                alza_move=blob["alza_move"],
                baja_move=blob["baja_move"],
            )
            model._models = blob["models"]
            model.calibration = blob.get("calibration")
            model._apply_signal_edge()
        except (TypeError, KeyError) as exc:
            raise StaleArtifactError(
                f"el artefacto 'contexto' tiene un formato desactualizado ({exc!r}): "
                "hay que reentrenarlo"
            ) from exc
        return model


__all__ = ["ContextTrendModel", "ContextConfig", "UNIVERSE", "HORIZON", "FEATURES"]
