"""Modelo de tendencia basado en XGBoost (v2): alza/baja a 20 ruedas.

La v1 era un regresor sobre la ventana plana de 30 ruedas x 8 variables
(acierto direccional ~47%, peor que tirar una moneda). Los barridos walk-forward
(ver ``notebooks/evaluacion_xgb_v2.py``) mostraron que la direccion de una
accion aislada no se predice, pero si hay senal *relativa y selectiva*:

- **Clasificador** P(sube a 20 ruedas) con arboles muy chicos y regularizados
  (profundidad 2, ``min_child_weight`` alto), en vez de regresion MSE.
- **Variables**: tecnicas + contexto de mercado (las de ``context_trend``) y,
  ademas, el **rango de cada variable frente al resto del universo ese mismo
  dia** (percentil cross-seccional): la senal documentada es cross-seccional.
- **Peso decreciente en el tiempo** (vida media 3 anios) y promedio de varias
  semillas, combinado mitad y mitad con una **regresion logistica** sobre las
  mismas variables (ver ``_score``).
- **Abstencion calibrada**: solo opina en los extremos (``signal_quantile`` de
  cada lado), con umbrales fijados fuera de muestra. Un lado que fuera de
  muestra no le gana a su tasa base se apaga solo.

Fuera de los extremos la salida es ``neutral``: el modelo no pretende acertar la
direccion de todas las acciones todos los dias, y su accuracy global NO supera
a "siempre sube". Lo que mide es cuanto mejor que la tasa base acierta cuando
opina.

``xgboost`` se importa de forma perezosa (solo al entrenar) para que el
servicio pueda arrancar aunque la dependencia no este instalada.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.context_trend import FEATURES as CONTEXT_FEATURES
from src.context_trend import HORIZON, MIN_ROWS, market_frame
from src.context_trend import build_features as context_features
from src.data import fetch_history
from src.data_quality import drop_stale, is_stale
from src.errors import ApiMlError, NotEnoughDataError, StaleArtifactError
from src.fair_value import TREND_FEATURES, trend_features
from src.feature_labels import feature_help, feature_label, feature_unit
from src.macro_trend import UNIVERSE
from src.trend_common import derive_trend_output, weak_signal_sides

# Tecnicas + contexto de mercado (las del modelo contexto) + distancia a la
# recta de tendencia del ultimo semestre y año (src.fair_value).
FEATURES = CONTEXT_FEATURES + TREND_FEATURES
RANK_FEATURES = [f"r_{name}" for name in FEATURES]
ALL_FEATURES = FEATURES + RANK_FEATURES
HALF_LIFE_YEARS = 3.0  # los datos recientes pesan mas al entrenar
MIN_PEERS = 5  # companias minimas para armar el rango cross-seccional
PEER_MAX_AGE_DAYS = 10  # un par sin datos recientes no entra en el rango
XGB_PARAMS = {
    "max_depth": 2,
    "n_estimators": 250,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 200,
    "reg_lambda": 10.0,
}


@dataclass
class XGBConfig:
    horizon: int = HORIZON
    signal_quantile: float = 0.10  # "alza"/"baja" solo en el 10% mas extremo de cada lado
    oos_fraction: float = 0.25  # ultimo tramo de fechas usado para medir y calibrar
    oos_folds: int = 4
    n_seeds: int = 3
    seed: int = 0


# --------------------------------------------------------------------------- #
# Universo (contexto de mercado + variables de los pares), cacheado por dia
# --------------------------------------------------------------------------- #
@dataclass
class Universe:
    """Variables de cada accion del universo, para rankear una accion contra sus pares."""

    market: pd.DataFrame
    features: dict[str, pd.DataFrame]


def build_features(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Variables de una accion, en el orden de ``FEATURES``."""
    return context_features(df, market).join(trend_features(df["close"]))[FEATURES]


def build_universe(histories: dict[str, pd.DataFrame]) -> Universe:
    market = market_frame(histories)
    usable = {s: d for s, d in drop_stale(histories).items() if len(d) >= MIN_ROWS}
    return Universe(market, {s: build_features(d, market) for s, d in usable.items()})


_universe_cache: dict[str, Universe] = {}


def todays_universe(days: int = 1500) -> Universe:
    """Universo con los historicos actuales (cacheado por dia)."""
    key = dt.date.today().isoformat()
    if key in _universe_cache:
        return _universe_cache[key]

    def load(symbol: str) -> tuple[str, pd.DataFrame | None]:
        try:
            return symbol, fetch_history(symbol, days)
        except ApiMlError as exc:
            print(f"  [warn] xgboost: sin historico de {symbol}: {exc}")
            return symbol, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = dict(pool.map(load, UNIVERSE))
    universe = build_universe({s: d for s, d in loaded.items() if d is not None})
    _universe_cache.clear()
    _universe_cache[key] = universe
    return universe


def rank_against_peers(own: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    """Percentil de cada variable de ``own`` (1 fila) frente a los pares ese dia.

    Usa solo filas hasta la fecha de ``own`` (el universo es "as of"), asi el
    backtest con cutoff no mira el futuro.
    """
    asof = own.index[-1]
    peers = []
    for frame in universe.features.values():
        part = frame.loc[:asof]
        if len(part) and (asof - part.index[-1]).days <= PEER_MAX_AGE_DAYS:
            peers.append(part.iloc[-1])
    if len(peers) < MIN_PEERS:
        raise NotEnoughDataError(f"faltan acciones para rankear (hay {len(peers)})")
    both = pd.concat([pd.DataFrame(peers)[FEATURES], own[FEATURES]], ignore_index=True)
    ranked = both.rank(pct=True).iloc[[-1]]
    ranked.columns = RANK_FEATURES
    ranked.index = own.index
    return ranked


# --------------------------------------------------------------------------- #
def _fit_ensemble(x: np.ndarray, y: np.ndarray, weights: np.ndarray, cfg: XGBConfig) -> list:
    import xgboost as xgb  # import perezoso

    models = []
    for k in range(cfg.n_seeds):
        model = xgb.XGBClassifier(
            **XGB_PARAMS, random_state=cfg.seed + k, n_jobs=4, tree_method="hist",
            eval_metric="logloss",
        )  # fmt: skip
        models.append(model.fit(x, y, sample_weight=weights))
    models.append(_fit_linear(x, y, weights))
    return models


def _fit_linear(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    """Regresion logistica muy regularizada sobre las mismas variables."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=500)
    )
    return model.fit(_finite(x), y, logisticregression__sample_weight=weights)


def _finite(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, np.nan)


def _finite_or_none(value: float | None) -> float | None:
    return round(float(value), 4) if value is not None and np.isfinite(value) else None


def _is_tree(model) -> bool:
    return hasattr(model, "get_booster")


def _score(models: list, x: np.ndarray) -> np.ndarray:
    """P(sube): mitad el promedio de las semillas de XGBoost, mitad la logistica.

    En la simulacion de paper trading 2016-2026
    (``notebooks/simulacion_paper_trading.py``) la combinacion separo las
    acciones fuertes de las debiles bastante mejor que XGBoost solo, en los dos
    tramos 2016-2020 y 2021-2026. Artefactos sin logistica: solo XGBoost.
    """
    trees = [m.predict_proba(x)[:, 1] for m in models if _is_tree(m)]
    linear = [m.predict_proba(_finite(x))[:, 1] for m in models if not _is_tree(m)]
    p_trees = np.mean(trees, axis=0)
    return (p_trees + np.mean(linear, axis=0)) / 2 if linear else p_trees


def _decay_weights(index: pd.Index, cutoff: pd.Timestamp) -> np.ndarray:
    age_years = (cutoff - pd.DatetimeIndex(index)).days.to_numpy() / 365.0
    return 0.5 ** (age_years / HALF_LIFE_YEARS)


@dataclass
class XGBTrendModel:
    config: XGBConfig = field(default_factory=XGBConfig)
    tickers: list[str] = field(default_factory=list)
    trained_at: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    alza_threshold: float | None = None  # P(sube) >= umbral -> "alza"
    baja_threshold: float | None = None  # P(sube) <= umbral -> "baja"
    alza_move: float = 0.0  # retorno log tipico cuando la senal de alza acierta
    baja_move: float = 0.0  # retorno log tipico cuando la senal de baja acierta (negativo)
    universe_provider: Callable[[], Universe] | None = None
    _models: list | None = None

    @property
    def version(self) -> str:
        return "untrained" if self.trained_at is None else f"xgb-{self.trained_at}"

    # ------------------------------------------------------------------ #
    def _panel(self, histories: dict[str, pd.DataFrame], market: pd.DataFrame) -> pd.DataFrame:
        h = self.config.horizon
        frames = []
        for symbol, df in histories.items():
            f = build_features(df, market)
            f["fwd"] = df["close"].shift(-h) / df["close"] - 1.0
            f["ticker"] = symbol
            frames.append(f)
        panel = pd.concat(frames)
        panel = panel[panel["fwd"].notna() & (panel["fwd"] != 0)].sort_index(kind="stable")
        # El indice (fecha) se repite entre acciones: asignar por posicion, no con join.
        panel[RANK_FEATURES] = panel.groupby(level=0)[FEATURES].rank(pct=True).to_numpy()
        panel["up"] = (panel["fwd"] > 0).astype(int)
        return panel

    def _oos_scores(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Predicciones fuera de muestra del ultimo tramo, con reentrenos sucesivos."""
        cfg = self.config
        days = np.sort(panel.index.unique())
        first = int(len(days) * (1 - cfg.oos_fraction))
        edges = np.linspace(first, len(days), cfg.oos_folds + 1).astype(int)
        index, x, y = panel.index.values, panel[ALL_FEATURES].to_numpy(), panel["up"].to_numpy()
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
            models = _fit_ensemble(x[tr], y[tr], weights, cfg)
            parts.append(
                pd.DataFrame(
                    {"p": _score(models, x[te]), "up": y[te], "fwd": panel["fwd"].to_numpy()[te]},
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
            self.alza_threshold = float("inf")
            self.metrics["alza_disabled"] = 1.0
        if baja_weak:
            self.baja_threshold = float("-inf")
            self.metrics["baja_disabled"] = 1.0

    def fit_panel(self, panel: pd.DataFrame) -> dict[str, float]:
        """Calibra umbrales fuera de muestra y entrena el modelo final."""
        cfg = self.config
        oos = self._oos_scores(panel)
        p, up = oos["p"].to_numpy(), oos["up"].to_numpy()
        q = cfg.signal_quantile
        self.alza_threshold = float(np.quantile(p, 1 - q))
        self.baja_threshold = float(np.quantile(p, q))
        fwd_log = np.log1p(oos["fwd"].to_numpy())
        is_alza, is_baja = p >= self.alza_threshold, p <= self.baja_threshold
        ups, downs = fwd_log[is_alza & (fwd_log > 0)], fwd_log[is_baja & (fwd_log < 0)]
        self.alza_move = float(np.median(ups)) if len(ups) else 0.0
        self.baja_move = float(np.median(downs)) if len(downs) else 0.0
        alza_hits = float(up[is_alza].mean()) if is_alza.any() else float("nan")
        baja_hits = float(1 - up[is_baja].mean()) if is_baja.any() else float("nan")
        base_up = float(up.mean())
        # Un lado solo se emite si fuera de muestra le gano a su tasa base: una
        # senal que acierta menos que no opinar no tiene que llegar al usuario.
        alza_off = not alza_hits > base_up
        baja_off = not baja_hits > 1 - base_up
        if alza_off:
            self.alza_threshold = float("inf")
        if baja_off:
            self.baja_threshold = float("-inf")
        self.metrics = {
            "oos_samples": float(len(oos)),
            "oos_auc": round(float(roc_auc_score(up, p)), 4),
            "oos_up_rate": round(base_up, 4),
            "oos_alza_samples": float(is_alza.sum()),
            "oos_alza_hit_rate": round(alza_hits, 4),
            "oos_baja_samples": float(is_baja.sum()),
            "oos_baja_hit_rate": round(baja_hits, 4),
            "alza_disabled": float(alza_off),
            "baja_disabled": float(baja_off),
        }
        self._apply_signal_edge()
        latest = pd.Timestamp(panel.index.max())
        weights = _decay_weights(panel.index, latest)
        self._models = _fit_ensemble(
            panel[ALL_FEATURES].to_numpy(), panel["up"].to_numpy(), weights, cfg
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

    # ------------------------------------------------------------------ #
    def _latest_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """Variables de la ultima rueda de ``df`` (1 fila, columnas ``ALL_FEATURES``)."""
        if self._models is None:
            raise NotEnoughDataError("el modelo no esta entrenado")
        if len(df) < MIN_ROWS:
            raise NotEnoughDataError(f"se necesitan al menos {MIN_ROWS} ruedas, hay {len(df)}")
        if is_stale(df):
            raise NotEnoughDataError(
                "la serie casi no tiene movimiento de precio (>30% de ruedas sin cambio)"
            )
        universe = (self.universe_provider or todays_universe)()
        # Contexto "as of": solo filas hasta la ultima fecha de ``df`` (sin mirar el futuro).
        own = build_features(df, universe.market.loc[: df.index[-1]]).iloc[[-1]]
        return own.join(rank_against_peers(own, universe))[ALL_FEATURES]

    def probability_up(self, df: pd.DataFrame) -> float:
        row = self._latest_row(df).to_numpy()
        return float(_score(self._models, row)[0])

    def explain(self, df: pd.DataFrame) -> dict:
        """Cuanto empujo cada variable la P(sube) de la ultima rueda (SHAP de TreeSHAP).

        Las contribuciones estan en log-odds y se promedian entre las semillas
        de XGBoost: ``base_value + sum(contribution)`` es su margen medio. Es una
        aproximacion de ``probability_up``: el ensamble promedia probabilidades
        (no margenes) y la mitad logistica no entra en el desglose.
        """
        import xgboost as xgb  # import perezoso

        frame = self._latest_row(df)
        # Los modelos se entrenaron con arrays sin nombres: DMatrix sin
        # feature_names y se emparejan por posicion con ALL_FEATURES.
        matrix = xgb.DMatrix(frame.to_numpy())
        contribs = np.mean(
            [
                m.get_booster().predict(matrix, pred_contribs=True)[0]
                for m in self._models
                if _is_tree(m)
            ],
            axis=0,
        )
        values = frame.iloc[0].to_numpy(dtype=float)
        rows = [
            {
                "feature": name,
                "label": feature_label(name),
                "help": feature_help(name),
                "unit": feature_unit(name),
                "value": None if np.isnan(value) else round(float(value), 6),
                "contribution": round(float(contribution), 6),
            }
            for name, value, contribution in zip(ALL_FEATURES, values, contribs[:-1], strict=True)
        ]
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        p_up = float(_score(self._models, frame.to_numpy())[0])
        return {
            "as_of": frame.index[-1].strftime("%Y-%m-%d"),
            "probability_up": round(p_up, 4),
            "base_value": round(float(contribs[-1]), 6),
            "contributions": rows,
            "signal": self.signal_for(p_up),
            # None: ese lado esta apagado (no le gana a lo normal fuera de muestra).
            "alza_threshold": _finite_or_none(self.alza_threshold),
            "baja_threshold": _finite_or_none(self.baja_threshold),
            "base_up_rate": self.metrics.get("oos_up_rate"),
        }

    def signal_for(self, p_up: float) -> str:
        """La misma regla que ``predict_df``: solo opina en los extremos calibrados."""
        if p_up >= float(self.alza_threshold):
            return "alza"
        if p_up <= float(self.baja_threshold):
            return "baja"
        return "neutral"

    def predict_df(self, df: pd.DataFrame) -> dict:
        p_up = self.probability_up(df)
        signal = self.signal_for(p_up)
        if signal == "alza":
            log_return = self.alza_move
        elif signal == "baja":
            log_return = self.baja_move
        else:
            log_return = 0.0  # sin vista: precio proyectado = ultimo cierre
        out = derive_trend_output(df, log_return, self.config.horizon, neutral_band=0.0)
        out["signal"] = signal
        out["probability_up"] = round(p_up, 4)
        out["probability_down"] = round(1 - p_up, 4)
        return out

    # ------------------------------------------------------------------ #
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
                "alza_threshold": self.alza_threshold,
                "baja_threshold": self.baja_threshold,
                "alza_move": self.alza_move,
                "baja_move": self.baja_move,
                "feature_names": ALL_FEATURES,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> XGBTrendModel:
        blob = joblib.load(Path(path))
        if blob.get("feature_names") != ALL_FEATURES:
            raise StaleArtifactError(
                "el artefacto 'xgboost' se entreno con otras variables (v1): hay que reentrenarlo"
            )
        try:
            model = cls(
                config=XGBConfig(**blob["config"]),
                tickers=blob.get("tickers", []),
                trained_at=blob.get("trained_at"),
                metrics=blob.get("metrics", {}),
                alza_threshold=blob["alza_threshold"],
                baja_threshold=blob["baja_threshold"],
                alza_move=blob["alza_move"],
                baja_move=blob["baja_move"],
            )
            model._models = blob["models"]
            model._apply_signal_edge()
        except (TypeError, KeyError) as exc:
            raise StaleArtifactError(
                f"el artefacto 'xgboost' tiene un formato desactualizado ({exc!r}): "
                "hay que reentrenarlo"
            ) from exc
        return model


__all__ = ["XGBTrendModel", "XGBConfig", "Universe", "ALL_FEATURES", "UNIVERSE", "HORIZON"]
