from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.errors import NotEnoughDataError

# Reutilizamos el feature engineering del modelo LSTM tal cual: la parte de
# "convertir OHLCV en features estacionarias" no cambia por usar otra
# arquitectura.
from src.lstm import FEATURE_NAMES, build_features, make_windows

# Mismo post-procesamiento (senal / RSI / condicion / confianza) que usan el
# resto de los modelos del registro (ver XGBTrendModel), para que las salidas
# sean directamente comparables entre modelos.
from src.trend_common import derive_trend_output

__all__ = ["TransformerTrainConfig", "TransformerTrendModel"]


# --------------------------------------------------------------------------- #
# Red neuronal
# --------------------------------------------------------------------------- #
class _LearnedPositionalEncoding(nn.Module):
    """Embedding posicional aprendido para secuencias de largo fijo."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding[:, : x.size(1), :]


class _TransformerRegressor(nn.Module):
    """Encoder de Transformer con token [CLS] para regresion de secuencias."""

    def __init__(
        self,
        input_size: int,
        window: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # +1 por el token [CLS] que se antepone a la secuencia.
        self.pos_encoding = _LearnedPositionalEncoding(window + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        h = self.input_proj(x) * math.sqrt(self.input_proj.out_features)
        cls = self.cls_token.expand(batch, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        cls_out = self.norm(h[:, 0, :])
        return self.head(cls_out).squeeze(-1)


@dataclass
class TransformerTrainConfig:
    window: int = 30
    horizon: int = 5  # dias hacia adelante para definir la tendencia a predecir
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.1
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # regularizacion L2 contra el overfitting
    patience: int = 8  # early stopping: epocas sin mejora en validacion
    val_fraction: float = 0.15
    neutral_band: float = 0.01  # |retorno esperado| por debajo => "neutral"
    warmup_epochs: int = 3  # calentamiento de learning rate, tipico en Transformers
    seed: int = 42

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) debe ser divisible por nhead ({self.nhead})"
            )


# --------------------------------------------------------------------------- #
# Wrapper de alto nivel: entrenamiento, inferencia, persistencia
# --------------------------------------------------------------------------- #
@dataclass
class TransformerTrendModel:
    config: TransformerTrainConfig = field(default_factory=TransformerTrainConfig)
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    target_mean: float = 0.0
    target_std: float = 1.0
    tickers: list[str] = field(default_factory=list)
    trained_at: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    _net: _TransformerRegressor | None = None

    @property
    def version(self) -> str:
        if self.trained_at is None:
            return "untrained"
        return f"transformer-{self.trained_at}"

    # ------------------------------------------------------------------ #
    def fit(self, histories: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Entrena el modelo pooleando ventanas de todos los tickers.

        Misma logica de armado de datos que ``TrendModel.fit`` (LSTM): se
        recibe un dict ``{ticker: dataframe_ohlcv}``, se generan ventanas por
        ticker y se poolean en un unico set de entrenamiento/validacion.
        """
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

        self._net = _TransformerRegressor(
            input_size=len(FEATURE_NAMES),
            window=cfg.window,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        # Warmup lineal seguido de decaimiento coseno: estandar para entrenar
        # Transformers de forma estable, incluso en modelos chicos como este.
        def lr_lambda(epoch: int) -> float:
            if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
                return (epoch + 1) / cfg.warmup_epochs
            progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
                # Los Transformers se benefician de clipear el gradiente.
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            scheduler.step()

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
                    f"train_mse={epoch_loss / n:.5f}  val_mse={val_loss:.5f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
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
        """Predice la tendencia del proximo dia a partir de un OHLCV.

        Misma firma que el resto de los modelos del registro: solo calcula el
        retorno logaritmico predicho y delega el post-procesamiento (senal,
        RSI, condicion, confianza) en ``derive_trend_output``, igual que
        ``XGBTrendModel.predict_df``.
        """
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

        return derive_trend_output(df, log_return, cfg.horizon, cfg.neutral_band)

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
                "architecture": "transformer",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TransformerTrendModel:
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = TransformerTrainConfig(**blob["config"])
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
        net = _TransformerRegressor(
            input_size=len(FEATURE_NAMES),
            window=cfg.window,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
        net.load_state_dict(blob["state_dict"])
        net.eval()
        model._net = net
        return model