"""Modelo de tendencia basado en XGBoost.

Comparte todo lo posible con el LSTM para que la comparacion sea justa:

- **Mismos features base** (``build_features`` de ``src.lstm``): retornos log,
  cambio de volumen y rango porcentual.
- **Mismo target**: retorno log acumulado de los proximos ``horizon`` dias.
- **Misma ventana**: se toman los ultimos ``window`` dias. Como XGBoost es un
  modelo tabular (no secuencial), la ventana se "aplana" a un unico vector de
  ``window * n_features`` columnas.
- **Misma salida**: se deriva con ``derive_trend_output``.

Asi, entrenando sobre los mismos tickers y el mismo periodo, se pueden comparar
las metricas de validacion (MAE del retorno y acierto direccional) contra el LSTM.

``xgboost`` se importa de forma perezosa (solo al entrenar/cargar) para que el
servicio pueda arrancar aunque la dependencia todavia no este instalada.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.errors import NotEnoughDataError
from src.lstm import FEATURE_NAMES, build_features, make_windows
from src.trend_common import derive_trend_output


@dataclass
class XGBConfig:
    window: int = 30
    horizon: int = 5  # dias hacia adelante (igual que el LSTM, para comparar)
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    val_fraction: float = 0.15
    neutral_band: float = 0.01
    seed: int = 42


@dataclass
class XGBTrendModel:
    config: XGBConfig = field(default_factory=XGBConfig)
    tickers: list[str] = field(default_factory=list)
    trained_at: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    _booster: object | None = None  # xgboost.XGBRegressor

    @property
    def version(self) -> str:
        if self.trained_at is None:
            return "untrained"
        return f"xgb-{self.trained_at}"

    # ------------------------------------------------------------------ #
    def _flatten_windows(
        self, feats: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aplana cada ventana (window, n_features) a un vector plano."""
        X, y = make_windows(feats, target, self.config.window)
        if len(X) == 0:
            return X, y
        return X.reshape(len(X), -1), y

    def fit(self, histories: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Entrena el modelo pooleando ventanas aplanadas de todos los tickers."""
        import xgboost as xgb  # import perezoso

        cfg = self.config
        np.random.seed(cfg.seed)

        train_X, train_y, val_X, val_y, used = [], [], [], [], []
        for symbol, df in histories.items():
            feats, target = build_features(df, cfg.horizon)
            X, y = self._flatten_windows(feats, target)
            if len(X) < 30:
                continue
            split = int(len(X) * (1 - cfg.val_fraction))
            train_X.append(X[:split])
            train_y.append(y[:split])
            val_X.append(X[split:])
            val_y.append(y[split:])
            used.append(symbol)

        if not train_X:
            raise NotEnoughDataError("no hay suficientes datos para entrenar")

        Xtr = np.concatenate(train_X)
        ytr = np.concatenate(train_y)
        Xva = np.concatenate(val_X) if val_X else np.empty((0, Xtr.shape[1]))
        yva = np.concatenate(val_y) if val_y else np.empty((0,))
        self.tickers = used

        self._booster = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=cfg.seed,
            objective="reg:squarederror",
        )
        eval_set = [(Xva, yva)] if len(Xva) else None
        self._booster.fit(Xtr, ytr, eval_set=eval_set, verbose=False)

        self.trained_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        self.metrics = self._evaluate(Xva, yva)
        return self.metrics

    # ------------------------------------------------------------------ #
    def predict_df(self, df: pd.DataFrame) -> dict:
        if self._booster is None:
            raise NotEnoughDataError("el modelo no esta entrenado")
        cfg = self.config
        feats, _ = build_features(df, cfg.horizon)
        if len(feats) < cfg.window:
            raise NotEnoughDataError(
                f"se necesitan al menos {cfg.window + 1} ruedas, hay {len(feats)}"
            )
        window = feats[-cfg.window :].reshape(1, -1)
        log_return = float(self._booster.predict(window)[0])
        return derive_trend_output(df, log_return, cfg.horizon, cfg.neutral_band)

    # ------------------------------------------------------------------ #
    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        if len(X) == 0 or self._booster is None:
            return {}
        pred = self._booster.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        directional = float(np.mean(np.sign(pred) == np.sign(y)))
        return {
            "val_samples": float(len(y)),
            "val_mae_logret": round(mae, 6),
            "val_directional_accuracy": round(directional, 4),
        }

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        if self._booster is None:
            raise NotEnoughDataError("no hay un modelo entrenado para guardar")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "booster": self._booster,
                "config": self.config.__dict__,
                "tickers": self.tickers,
                "trained_at": self.trained_at,
                "metrics": self.metrics,
                "feature_names": FEATURE_NAMES,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> XGBTrendModel:
        blob = joblib.load(Path(path))
        model = cls(
            config=XGBConfig(**blob["config"]),
            tickers=blob.get("tickers", []),
            trained_at=blob.get("trained_at"),
            metrics=blob.get("metrics", {}),
        )
        model._booster = blob["booster"]
        return model
