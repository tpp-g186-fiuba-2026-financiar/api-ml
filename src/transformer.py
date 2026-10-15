from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.errors import NotEnoughDataError

# Reutilizamos el feature engineering del modelo LSTM tal cual: la parte de
# "convertir OHLCV en features estacionarias" no cambia por usar otra
# arquitectura.
from src.lstm import FEATURE_NAMES, build_features, check_feature_compatibility, make_windows

# Mismo post-procesamiento (senal / RSI / condicion / confianza) que usan el
# resto de los modelos del registro (ver XGBTrendModel), para que las salidas
# sean directamente comparables entre modelos.
from src.trend_common import derive_trend_output, ic_por_fecha

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
    # --- Variantes en evaluacion (los defaults reproducen el modelo original) ---
    # "absoluto": retorno log a `horizon` ruedas. "relativo": exceso sobre el
    # promedio de los tickers de entrenamiento ese mismo dia; saca la deriva en
    # pesos (inflacion / devaluacion) que sesga el modelo absoluto hacia "alza".
    # Con "relativo" la senal se lee como "mejor / peor que el panel".
    target: str = "absoluto"
    # "todos": todos los tickers recibidos. "panel": solo el panel lider.
    universo: str = "todos"
    loss: str = "mse"  # "mse" o "huber" (menos sensible a saltos extremos)
    # Recorte del target escalado a +-clip_target desvios (None = sin recorte).
    clip_target: float | None = None
    # Metrica de early stopping: "mse" o "ic" (correlacion de rangos por dia
    # entre prediccion y retorno real en validacion; premia ordenar bien los
    # tickers en vez de acercarse a la media).
    seleccion: str = "mse"
    # Ventanas descartadas entre train y validacion: con target a `horizon`
    # ruedas, las ultimas ventanas de train comparten futuro con las primeras
    # de validacion. Con gap=horizon no hay solapamiento.
    gap: int = 0
    n_semillas: int = 1  # ensamble: promedio de redes entrenadas con semillas distintas
    # Si se define, al terminar `fit` se reemplaza `neutral_band` por este
    # cuantil de |retorno predicho| en validacion (0.5 = la mitad de los casos
    # da "neutral"). Con target relativo las predicciones son mucho mas chicas
    # que el 1% fijo y casi todo quedaria "neutral".
    banda_cuantil: float | None = None
    # Features relativas al panel lider (ver ``_features_panel``): retorno a 5
    # y 20 ruedas contra la mediana del panel, ranking del dia, volatilidad
    # relativa y retorno del mercado. Necesitan ver los demas tickers del dia,
    # asi que se predice con ``predict_panel`` (``predict_df`` no alcanza).
    features_panel: bool = False
    # Target y retornos en unidades de volatilidad (20 ruedas) del ticker: un
    # -5% de una accion tranquila pesa mas que el de una volatil, y los saltos
    # de las acciones volatiles dejan de dominar la loss.
    escala_vol: bool = False
    # Subconjunto de features que ve la red, por nombre (``FEATURE_NAMES`` y,
    # con ``features_panel``, ``_FEATURES_PANEL``). None = todas. Con una
    # senal tan debil, cada input de mas es ruido que la red puede ajustar.
    columnas: list[str] | None = None

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) debe ser divisible por nhead ({self.nhead})"
            )
        opciones = {
            "target": ("absoluto", "relativo"),
            "universo": ("todos", "panel"),
            "loss": ("mse", "huber"),
            "seleccion": ("mse", "ic"),
        }
        for nombre, validos in opciones.items():
            if getattr(self, nombre) not in validos:
                raise ValueError(f"{nombre} debe ser uno de {validos}")
        if self.n_semillas < 1:
            raise ValueError("n_semillas debe ser al menos 1")
        if self.banda_cuantil is not None and not 0 < self.banda_cuantil < 1:
            raise ValueError("banda_cuantil debe estar entre 0 y 1")
        if self.columnas is not None:
            disponibles = nombres_features(self, todas=True)
            invalidas = [c for c in self.columnas if c not in disponibles]
            if not self.columnas or invalidas:
                raise ValueError(f"columnas invalidas {invalidas}: elegir entre {disponibles}")
        if self.banda_cuantil is not None and self.escala_vol:
            # La banda se calibraria en unidades de volatilidad, no de retorno.
            raise ValueError("banda_cuantil no se puede combinar con escala_vol")


# --------------------------------------------------------------------------- #
# Armado de datos
# --------------------------------------------------------------------------- #
_Ventanas = tuple[np.ndarray, np.ndarray, np.ndarray]
_FEATURES_PANEL = ["ret5_rel", "ret20_rel", "rank5", "vol20_rel", "mkt_ret5", "mkt_ret20"]
_RUEDAS_VOL = 20
_MINIMO_PANEL = 5  # tickers del panel con dato para que la mediana represente al mercado


def nombres_features(cfg: TransformerTrainConfig, todas: bool = False) -> list[str]:
    """Features que ve la red (``todas``: antes de aplicar ``columnas``)."""
    nombres = FEATURE_NAMES + (_FEATURES_PANEL if cfg.features_panel else [])
    return nombres if todas or cfg.columnas is None else list(cfg.columnas)


def n_features(cfg: TransformerTrainConfig) -> int:
    return len(nombres_features(cfg))


def _log_close(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    return np.log(close.where(close > 0))


def contexto_panel(histories: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame] | None:
    """Retornos y volatilidad del panel lider por fecha (``None`` si hay menos
    de ``_MINIMO_PANEL`` tickers). Cada valor usa solo cierres hasta esa fecha."""
    logs = {s: _log_close(df) for s, df in histories.items() if s in PANEL_LIDER_TICKERS}
    if len(logs) < _MINIMO_PANEL:
        return None
    ret5 = pd.DataFrame({s: lc.diff(5) for s, lc in logs.items()})
    ret20 = pd.DataFrame({s: lc.diff(20) for s, lc in logs.items()})
    vol = pd.DataFrame({s: lc.diff().rolling(_RUEDAS_VOL).std() for s, lc in logs.items()})
    ok = ret5.notna().sum(axis=1) >= _MINIMO_PANEL
    return {
        "ret5": ret5.where(ok, axis=0),
        "med_ret5": ret5.median(axis=1).where(ok),
        "med_ret20": ret20.median(axis=1).where(ok),
        "med_vol": vol.median(axis=1).where(ok),
    }


def _features_panel(log_close: pd.Series, vol: pd.Series, contexto: dict) -> np.ndarray:
    """Columnas ``_FEATURES_PANEL`` de un ticker, una fila por rueda."""
    idx = log_close.index
    r5, r20 = log_close.diff(5), log_close.diff(20)
    panel5 = contexto["ret5"].reindex(idx)
    con_dato = panel5.notna().sum(axis=1).replace(0, np.nan)
    # Posicion del ticker entre los retornos a 5 ruedas del panel: -0,5 el peor, +0,5 el mejor.
    rank = (panel5.lt(r5, axis=0).sum(axis=1) / con_dato - 0.5).where(r5.notna())
    m5 = contexto["med_ret5"].reindex(idx)
    m20 = contexto["med_ret20"].reindex(idx)
    vol_rel = np.log(vol / contexto["med_vol"].reindex(idx))
    return np.column_stack([r5 - m5, r20 - m20, rank, vol_rel, m5, m20]).astype(np.float64)


def _features(
    df: pd.DataFrame, cfg: TransformerTrainConfig, contexto: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, target y escala de cada fila (alineadas como ``build_features``:
    la fila j corresponde a la rueda ``df.index[j + 1]``). La escala es la
    volatilidad del ticker si ``escala_vol``, si no 1."""
    feats, target = build_features(df, cfg.horizon)
    log_close = _log_close(df)
    vol = log_close.diff().rolling(_RUEDAS_VOL).std()
    if cfg.features_panel:
        if contexto is None:
            raise NotEnoughDataError("features_panel necesita el panel del dia (predict_panel)")
        extra = _features_panel(log_close, vol, contexto)[1:]
        feats = np.column_stack([feats, extra])
    escala = np.ones(len(feats))
    if cfg.escala_vol:
        escala = vol.to_numpy(dtype=np.float64)[1:]
        escala = np.where(escala > 0, escala, np.nan)
        target = target / escala
        # Columnas que son retornos: log_return, momentum_10 y los relativos al panel.
        columnas = [FEATURE_NAMES.index("log_return"), FEATURE_NAMES.index("momentum_10")]
        if cfg.features_panel:
            columnas += [len(FEATURE_NAMES), len(FEATURE_NAMES) + 1]
        feats[:, columnas] = feats[:, columnas] / escala[:, None]
    if cfg.columnas is not None:
        todas = nombres_features(cfg, todas=True)
        feats = feats[:, [todas.index(c) for c in cfg.columnas]]
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, target, escala


def _ventanas_con_fecha(
    df: pd.DataFrame, cfg: TransformerTrainConfig, contexto: dict | None = None
) -> _Ventanas:
    """Como ``make_windows``, pero ademas devuelve la fecha de cada ventana.

    La fecha es la ultima rueda que ve la ventana (desde donde se predice), y
    sirve para comparar tickers en el mismo dia (target relativo e IC).
    """
    feats, target, _ = _features(df, cfg, contexto)
    X, y = make_windows(feats.astype(np.float32), target, cfg.window)
    # make_windows usa target[i + window - 1], que corresponde a la rueda
    # df.index[i + window] (las features arrancan en la segunda rueda).
    validas = [
        i for i in range(len(feats) - cfg.window) if not np.isnan(target[i + cfg.window - 1])
    ]
    fechas = df.index[[i + cfg.window for i in validas]].to_numpy()
    return X, y, fechas


def _a_exceso(por_ticker: dict[str, _Ventanas]) -> dict[str, _Ventanas]:
    """Resta a cada target el promedio de los tickers con ventana en esa fecha.

    Las fechas con menos de 3 tickers se descartan: el promedio no representa
    al mercado.
    """
    todos = pd.DataFrame(
        {
            "fecha": np.concatenate([f for _, _, f in por_ticker.values()]),
            "y": np.concatenate([y for _, y, _ in por_ticker.values()]),
        }
    )
    grupos = todos.groupby("fecha")["y"]
    media, cantidad = grupos.mean(), grupos.size()
    salida = {}
    for symbol, (X, y, fechas) in por_ticker.items():
        n = cantidad.reindex(fechas).to_numpy()
        ok = n >= 3
        if ok.sum() < 30:
            continue
        exceso = y[ok] - media.reindex(fechas[ok]).to_numpy(dtype=np.float32)
        salida[symbol] = (X[ok], exceso.astype(np.float32), fechas[ok])
    return salida


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
    _nets: list[_TransformerRegressor] = field(default_factory=list)
    # Dispositivo de entrenamiento ("cpu" o "cuda"). No se guarda en el
    # artefacto: las redes vuelven a CPU al terminar de entrenar.
    dispositivo: str = "cpu"

    @property
    def version(self) -> str:
        if self.trained_at is None:
            return "untrained"
        return f"transformer-{self.trained_at}"

    @property
    def _net(self) -> _TransformerRegressor | None:
        """Primera red del ensamble (la unica si n_semillas == 1)."""
        return self._nets[0] if self._nets else None

    # ------------------------------------------------------------------ #
    def fit(self, histories: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Entrena el modelo pooleando ventanas de todos los tickers.

        Misma logica de armado de datos que ``TrendModel.fit`` (LSTM): se
        recibe un dict ``{ticker: dataframe_ohlcv}``, se generan ventanas por
        ticker y se poolean en un unico set de entrenamiento/validacion.
        """
        np.random.seed(self.config.seed)
        cfg = self.config

        contexto = contexto_panel(histories) if cfg.features_panel else None
        if cfg.features_panel and contexto is None:
            raise NotEnoughDataError("features_panel necesita al menos 5 tickers del panel")
        if cfg.universo == "panel":
            histories = {s: df for s, df in histories.items() if s in PANEL_LIDER_TICKERS}

        por_ticker: dict[str, _Ventanas] = {}
        for symbol, df in histories.items():
            X, y, fechas = _ventanas_con_fecha(df, cfg, contexto)
            if len(X) < 30:  # muy pocas ventanas, se descarta el ticker
                continue
            por_ticker[symbol] = (X, y, fechas)
        if cfg.target == "relativo":
            por_ticker = _a_exceso(por_ticker)

        if not por_ticker:
            raise NotEnoughDataError("no hay suficientes datos para entrenar")

        train_X, train_y, val_X, val_y, val_f = [], [], [], [], []
        for X, y, fechas in por_ticker.values():
            split = int(len(X) * (1 - cfg.val_fraction))
            fin_train = max(split - cfg.gap, 0) if split < len(X) else split
            train_X.append(X[:fin_train])
            train_y.append(y[:fin_train])
            val_X.append(X[split:])
            val_y.append(y[split:])
            val_f.append(fechas[split:])

        Xtr = np.concatenate(train_X)
        ytr = np.concatenate(train_y)
        Xva = np.concatenate(val_X)
        yva = np.concatenate(val_y)
        fva = np.concatenate(val_f)
        self.tickers = list(por_ticker)

        # Normalizacion calculada solo con el set de entrenamiento.
        flat = Xtr.reshape(-1, Xtr.shape[-1])
        self.feature_mean = flat.mean(axis=0)
        self.feature_std = flat.std(axis=0) + 1e-8
        self.target_mean = float(ytr.mean())
        self.target_std = float(ytr.std() + 1e-8)

        ytr_scaled = (ytr - self.target_mean) / self.target_std
        if cfg.clip_target is not None:
            ytr_scaled = np.clip(ytr_scaled, -cfg.clip_target, cfg.clip_target)
        disp = torch.device(self.dispositivo)
        Xtr_t = self._to_tensor(self._scale_features(Xtr)).to(disp)
        ytr_t = torch.tensor(ytr_scaled, dtype=torch.float32, device=disp)

        has_val = len(Xva) > 0
        Xva_t = self._to_tensor(self._scale_features(Xva)) if has_val else None
        yva_scaled = (yva - self.target_mean) / self.target_std if has_val else None

        self._nets = []
        for k in range(cfg.n_semillas):
            if cfg.n_semillas > 1:
                print(f"  semilla {k + 1}/{cfg.n_semillas}")
            self._nets.append(
                self._entrenar_red(cfg.seed + k, Xtr_t, ytr_t, Xva_t, yva_scaled, fva)
            )

        self.trained_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        self.metrics = self._evaluate(Xva, yva, fva)
        if cfg.banda_cuantil is not None and has_val:
            pred = self._forward(Xva_t) * self.target_std + self.target_mean
            cfg.neutral_band = float(np.quantile(np.abs(np.expm1(pred)), cfg.banda_cuantil))
            self.metrics["neutral_band"] = round(cfg.neutral_band, 6)
        return self.metrics

    def _entrenar_red(
        self,
        seed: int,
        Xtr_t: torch.Tensor,
        ytr_t: torch.Tensor,
        Xva_t: torch.Tensor | None,
        yva_scaled: np.ndarray | None,
        fva: np.ndarray,
    ) -> _TransformerRegressor:
        cfg = self.config
        torch.manual_seed(seed)
        net = _TransformerRegressor(
            input_size=n_features(cfg),
            window=cfg.window,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        ).to(Xtr_t.device)
        if Xva_t is not None:
            Xva_t = Xva_t.to(Xtr_t.device)
            yva_t = torch.tensor(yva_scaled, dtype=torch.float32, device=Xtr_t.device)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        # Warmup lineal seguido de decaimiento coseno: estandar para entrenar
        # Transformers de forma estable, incluso en modelos chicos como este.
        def lr_lambda(epoch: int) -> float:
            if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
                return (epoch + 1) / cfg.warmup_epochs
            progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        loss_fn: nn.Module = nn.HuberLoss(delta=1.0) if cfg.loss == "huber" else nn.MSELoss()
        mse = nn.MSELoss()

        has_val = Xva_t is not None
        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        n = len(Xtr_t)
        for epoch in range(cfg.epochs):
            net.train()
            perm = torch.randperm(n).to(Xtr_t.device)
            epoch_loss = 0.0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                optimizer.zero_grad()
                pred = net(Xtr_t[idx])
                loss = loss_fn(pred, ytr_t[idx])
                loss.backward()
                # Los Transformers se benefician de clipear el gradiente.
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            scheduler.step()

            # Early stopping sobre la metrica de validacion (menor es mejor:
            # con seleccion "ic" se usa -IC).
            if has_val:
                net.eval()
                with torch.no_grad():
                    val_pred = net(Xva_t)
                val_loss = float(mse(val_pred, yva_t).item())
                if cfg.seleccion == "ic":
                    val_score = -ic_por_fecha(val_pred.cpu().numpy(), yva_scaled, fva)
                else:
                    val_score = val_loss
                if val_score < best_val - 1e-5:
                    best_val = val_score
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                val_loss = float("nan")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  epoch {epoch + 1:>3}/{cfg.epochs}  "
                    f"train_loss={epoch_loss / n:.5f}  val_mse={val_loss:.5f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if has_val and epochs_no_improve >= cfg.patience:
                print(f"  early stopping en epoca {epoch + 1} (mejor val={best_val:.5f})")
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net.cpu()

    # ------------------------------------------------------------------ #
    def predict_df(self, df: pd.DataFrame) -> dict:
        """Predice la tendencia del proximo dia a partir de un OHLCV.

        Misma firma que el resto de los modelos del registro: solo calcula el
        retorno logaritmico predicho y delega el post-procesamiento (senal,
        RSI, condicion, confianza) en ``derive_trend_output``, igual que
        ``XGBTrendModel.predict_df``. Con ``target="relativo"`` el retorno
        predicho es el exceso sobre el promedio del mercado.
        """
        if not self._nets:
            raise NotEnoughDataError("el modelo no esta entrenado")
        ventana, escala = self._ventana_prediccion(df, None)
        log_return = float(self._log_returns(ventana[None], np.array([escala]))[0])
        return derive_trend_output(df, log_return, self.config.horizon, self.config.neutral_band)

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """Predice varios tickers del mismo dia en un solo batch. Es la forma de
        predecir con ``features_panel`` (el contexto sale de ``dfs``); los
        tickers sin datos suficientes se omiten."""
        if not self._nets:
            raise NotEnoughDataError("el modelo no esta entrenado")
        contexto = contexto_panel(dfs) if self.config.features_panel else None
        ventanas, escalas, symbols = [], [], []
        for symbol, df in dfs.items():
            try:
                ventana, escala = self._ventana_prediccion(df, contexto)
            except NotEnoughDataError:
                continue
            ventanas.append(ventana)
            escalas.append(escala)
            symbols.append(symbol)
        if not symbols:
            return {}
        log_returns = self._log_returns(np.stack(ventanas), np.array(escalas))
        cfg = self.config
        return {
            s: derive_trend_output(dfs[s], float(r), cfg.horizon, cfg.neutral_band)
            for s, r in zip(symbols, log_returns, strict=True)
        }

    def _ventana_prediccion(
        self, df: pd.DataFrame, contexto: dict | None
    ) -> tuple[np.ndarray, float]:
        """Ultima ventana de features de ``df`` y su escala."""
        cfg = self.config
        feats, _, escala = _features(df, cfg, contexto)
        if len(feats) < cfg.window:
            raise NotEnoughDataError(
                f"se necesitan al menos {cfg.window + 1} ruedas, hay {len(feats)}"
            )
        if not np.isfinite(escala[-1]):
            raise NotEnoughDataError("sin volatilidad reciente para escalar el target")
        return feats[-cfg.window :], float(escala[-1])

    def _log_returns(self, ventanas: np.ndarray, escalas: np.ndarray) -> np.ndarray:
        """Retorno log predicho para un batch de ventanas."""
        scaled = self._forward(self._to_tensor(self._scale_features(ventanas)))
        return (scaled * self.target_std + self.target_mean) * escalas

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        """Prediccion escalada: promedio de las redes del ensamble."""
        with torch.no_grad():
            return np.mean([net(x).numpy() for net in self._nets], axis=0)

    # ------------------------------------------------------------------ #
    def _evaluate(
        self, X: np.ndarray, y: np.ndarray, fechas: np.ndarray | None = None
    ) -> dict[str, float]:
        if len(X) == 0 or not self._nets:
            return {}
        scaled = self._forward(self._to_tensor(self._scale_features(X)))
        pred = scaled * self.target_std + self.target_mean
        mae = float(np.mean(np.abs(pred - y)))
        # Acierto direccional: que el signo del retorno coincida.
        directional = float(np.mean(np.sign(pred) == np.sign(y)))
        metrics = {
            "val_samples": float(len(y)),
            "val_mae_logret": round(mae, 6),
            "val_directional_accuracy": round(directional, 4),
        }
        if fechas is not None:
            metrics["val_ic"] = round(ic_por_fecha(pred, y, fechas), 4)
        return metrics

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean) / self.feature_std

    @staticmethod
    def _to_tensor(X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        if not self._nets:
            raise NotEnoughDataError("no hay un modelo entrenado para guardar")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                # "state_dict" (primera red) se mantiene para artefactos de una
                # sola red; "state_dicts" tiene el ensamble completo.
                "state_dict": self._nets[0].state_dict(),
                "state_dicts": [net.state_dict() for net in self._nets],
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
        check_feature_compatibility(blob.get("feature_names"))
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
        for state in blob.get("state_dicts") or [blob["state_dict"]]:
            net = _TransformerRegressor(
                input_size=n_features(cfg),
                window=cfg.window,
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                num_layers=cfg.num_layers,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
            )
            net.load_state_dict(state)
            net.eval()
            model._nets.append(net)
        return model
