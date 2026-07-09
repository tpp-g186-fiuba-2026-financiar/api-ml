"""Modelo LSTM de tendencia para acciones del Merval.

La idea es sencilla y explicable:

1. A partir del OHLCV se derivan features estacionarias (retornos logaritmicos),
   de modo que un unico modelo pueda generalizar a varios tickers sin importar
   la escala de precios de cada uno.
2. Una red LSTM mira una ventana de los ultimos ``window`` dias y predice el
   retorno logaritmico del proximo dia.
3. De ese retorno se deriva una senal de tendencia (alza / baja / neutral) y se
   complementa con el RSI clasico (sobrecompra / sobreventa) que define el
   anteproyecto.

El artefacto entrenado (pesos + parametros de normalizacion + metadata) se
serializa con ``torch.save`` en un unico archivo ``.pt``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.errors import NotEnoughDataError

FEATURE_NAMES = ["log_return", "log_volume_change", "range_pct"]


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, horizon: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Convierte un OHLCV en (X_features, y_target).

    ``X`` tiene una fila por dia con las columnas de ``FEATURE_NAMES``.
    ``y`` es el retorno logaritmico *acumulado* de los proximos ``horizon`` dias
    (la "tendencia" a futuro), mas estable y predecible que el retorno de un
    unico dia. Las filas sin target futuro completo quedan en ``NaN``.
    """
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    log_close = np.log(close)
    log_return = np.diff(log_close)  # alineado a los dias 1..T-1
    safe_vol = np.where(volume <= 0, np.nan, volume)
    log_volume_change = np.diff(np.log(safe_vol))
    range_pct = ((high - low) / np.where(close == 0, np.nan, close))[1:]

    feats = np.column_stack([log_return, log_volume_change, range_pct])

    # Para la feature j (dia d = j + 1), el target es el retorno acumulado
    # log(close[d + horizon] / close[d]).
    n_feats = len(feats)
    target = np.full(n_feats, np.nan)
    for j in range(n_feats):
        d = j + 1
        if d + horizon < len(close):
            target[j] = log_close[d + horizon] - log_close[d]

    # Limpiamos NaN/inf (volumen 0, etc.) rellenando con 0 en las features.
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, target


def make_windows(
    feats: np.ndarray, target: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Arma secuencias deslizantes de largo ``window``.

    Para cada posicion ``i`` la entrada es ``feats[i:i+window]`` y la salida es
    ``target[i+window-1]`` (el retorno del dia siguiente al final de la ventana).
    """
    xs, ys = [], []
    for i in range(len(feats) - window):
        y = target[i + window - 1]
        if np.isnan(y):
            continue
        xs.append(feats[i : i + window])
        ys.append(y)
    if not xs:
        return np.empty((0, window, feats.shape[1])), np.empty((0,))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def rsi(close: np.ndarray, period: int = 14) -> float | None:
    """RSI de Wilder sobre el ultimo valor de la serie."""
    if len(close) < period + 1:
        return None
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


# --------------------------------------------------------------------------- #
# Red neuronal
# --------------------------------------------------------------------------- #
class _LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


@dataclass
class TrainConfig:
    window: int = 30
    horizon: int = 5  # dias hacia adelante para definir la tendencia a predecir
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # regularizacion L2 contra el overfitting
    patience: int = 8  # early stopping: epocas sin mejora en validacion
    val_fraction: float = 0.15
    neutral_band: float = 0.01  # |retorno esperado| por debajo => "neutral"
    seed: int = 42


# --------------------------------------------------------------------------- #
# Wrapper de alto nivel: entrenamiento, inferencia, persistencia
# --------------------------------------------------------------------------- #
@dataclass
class TrendModel:
    config: TrainConfig = field(default_factory=TrainConfig)
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    target_mean: float = 0.0
    target_std: float = 1.0
    tickers: list[str] = field(default_factory=list)
    trained_at: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    _net: _LSTMRegressor | None = None

    @property
    def version(self) -> str:
        if self.trained_at is None:
            return "untrained"
        return f"lstm-{self.trained_at}"

    # ------------------------------------------------------------------ #
    def fit(self, histories: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Entrena el modelo pooleando ventanas de todos los tickers."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        cfg = self.config

        train_X, train_y, val_X, val_y = [], [], [], []
        used: list[str] = []
        for symbol, df in histories.items():
            feats, target = build_features(df, cfg.horizon)
            X, y = make_windows(feats, target, cfg.window)
            if len(X) < 30:  # muy pocas ventanas, se descarta el ticker
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
        Xva = np.concatenate(val_X) if val_X else np.empty((0, cfg.window, len(FEATURE_NAMES)))
        yva = np.concatenate(val_y) if val_y else np.empty((0,))
        self.tickers = used

        # Normalizacion calculada solo con el set de entrenamiento.
        flat = Xtr.reshape(-1, Xtr.shape[-1])
        self.feature_mean = flat.mean(axis=0)
        self.feature_std = flat.std(axis=0) + 1e-8
        self.target_mean = float(ytr.mean())
        self.target_std = float(ytr.std() + 1e-8)

        Xtr_t = self._to_tensor(self._scale_features(Xtr))
        ytr_t = torch.tensor((ytr - self.target_mean) / self.target_std, dtype=torch.float32)

        self._net = _LSTMRegressor(len(FEATURE_NAMES), cfg.hidden_size, cfg.num_layers, cfg.dropout)
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        loss_fn = nn.MSELoss()

        has_val = len(Xva) > 0
        Xva_t = self._to_tensor(self._scale_features(Xva)) if has_val else None
        yva_scaled = (yva - self.target_mean) / self.target_std if has_val else None

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        n = len(Xtr_t)
        for epoch in range(cfg.epochs):
            self._net.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                optimizer.zero_grad()
                pred = self._net(Xtr_t[idx])
                loss = loss_fn(pred, ytr_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)

            # Early stopping sobre la perdida de validacion.
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_pred = self._net(Xva_t)
                    val_loss = float(
                        loss_fn(val_pred, torch.tensor(yva_scaled, dtype=torch.float32)).item()
                    )
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                val_loss = float("nan")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  epoch {epoch + 1:>3}/{cfg.epochs}  "
                    f"train_mse={epoch_loss / n:.5f}  val_mse={val_loss:.5f}"
                )

            if has_val and epochs_no_improve >= cfg.patience:
                print(f"  early stopping en epoca {epoch + 1} (mejor val_mse={best_val:.5f})")
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self.trained_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        self.metrics = self._evaluate(Xva, yva)
        return self.metrics

    # ------------------------------------------------------------------ #
    def predict_df(self, df: pd.DataFrame) -> dict:
        """Predice la tendencia del proximo dia a partir de un OHLCV."""
        if self._net is None:
            raise NotEnoughDataError("el modelo no esta entrenado")
        cfg = self.config
        feats, _ = build_features(df, cfg.horizon)
        if len(feats) < cfg.window:
            raise NotEnoughDataError(
                f"se necesitan al menos {cfg.window + 1} ruedas, hay {len(feats)}"
            )
        window = feats[-cfg.window :][None, :, :]
        x = self._to_tensor(self._scale_features(window))
        self._net.eval()
        with torch.no_grad():
            scaled = float(self._net(x).item())
        log_return = scaled * self.target_std + self.target_mean

        last_close = float(df["close"].iloc[-1])
        expected_return = float(np.expm1(log_return))
        predicted_close = float(last_close * np.exp(log_return))

        if expected_return > cfg.neutral_band:
            signal = "alza"
        elif expected_return < -cfg.neutral_band:
            signal = "baja"
        else:
            signal = "neutral"

        rsi_value = rsi(df["close"].to_numpy(dtype=np.float64))
        if rsi_value is None:
            condition = "indeterminado"
        elif rsi_value >= 70:
            condition = "sobrecompra"
        elif rsi_value <= 30:
            condition = "sobreventa"
        else:
            condition = "neutral"

        return {
            "signal": signal,
            "horizon_days": cfg.horizon,
            "predicted_close": round(predicted_close, 4),
            "last_close": round(last_close, 4),
            "rsi": round(rsi_value, 2) if rsi_value is not None else None,
            "condition": condition,
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
        }

    # ------------------------------------------------------------------ #
    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        if len(X) == 0 or self._net is None:
            return {}
        x = self._to_tensor(self._scale_features(X))
        self._net.eval()
        with torch.no_grad():
            scaled = self._net(x).numpy()
        pred = scaled * self.target_std + self.target_mean
        mae = float(np.mean(np.abs(pred - y)))
        # Acierto direccional: que el signo del retorno coincida.
        directional = float(np.mean(np.sign(pred) == np.sign(y)))
        return {
            "val_samples": float(len(y)),
            "val_mae_logret": round(mae, 6),
            "val_directional_accuracy": round(directional, 4),
        }

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean) / self.feature_std

    @staticmethod
    def _to_tensor(X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        if self._net is None:
            raise NotEnoughDataError("no hay un modelo entrenado para guardar")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "config": self.config.__dict__,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "tickers": self.tickers,
                "trained_at": self.trained_at,
                "metrics": self.metrics,
                "feature_names": FEATURE_NAMES,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TrendModel:
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = TrainConfig(**blob["config"])
        model = cls(
            config=cfg,
            feature_mean=np.asarray(blob["feature_mean"]),
            feature_std=np.asarray(blob["feature_std"]),
            target_mean=blob["target_mean"],
            target_std=blob["target_std"],
            tickers=blob.get("tickers", []),
            trained_at=blob.get("trained_at"),
            metrics=blob.get("metrics", {}),
        )
        net = _LSTMRegressor(len(FEATURE_NAMES), cfg.hidden_size, cfg.num_layers, cfg.dropout)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        model._net = net
        return model


def fetch_histories(symbols: Iterable[str], days: int, fetch_fn) -> dict[str, pd.DataFrame]:
    """Descarga el historico de cada simbolo, tolerando fallas individuales."""
    histories: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = fetch_fn(symbol, days)
        except Exception as exc:  # noqa: BLE001 - se reporta y se continua
            print(f"  [skip] {symbol}: {exc}")
            continue
        if len(df) < 60:
            print(f"  [skip] {symbol}: pocas ruedas ({len(df)})")
            continue
        histories[symbol] = df
        print(f"  [ok]   {symbol}: {len(df)} ruedas")
    return histories
