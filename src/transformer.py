from __future__ import annotations

import datetime as dt
import functools
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.data import COMMODITY_YAHOO_SYMBOLS
from src.errors import DataUnavailableError, NotEnoughDataError, StaleArtifactError

# Reutilizamos el feature engineering del modelo LSTM tal cual: la parte de
# "convertir OHLCV en features estacionarias" no cambia por usar otra
# arquitectura.
from src.lstm import FEATURE_NAMES, build_features, check_feature_compatibility, make_windows
from src.macro_data import SERIES, build_macro_frame, fetch_collector_series
from src.panel.datos import en_dolares
from src.simulacion.limpieza import (
    ajustar_salto,
    detectar_saltos,
    retorno_mercado,
    sin_precios_no_positivos,
)

# Mismo post-procesamiento (senal / RSI / condicion / confianza) que usan el
# resto de los modelos del registro (ver XGBTrendModel), para que las salidas
# sean directamente comparables entre modelos.
from src.trend_common import acierto_balanceado, derive_trend_output, ic_por_fecha

__all__ = ["TransformerTrainConfig", "TransformerTrendModel"]


# --------------------------------------------------------------------------- #
# Red neuronal
# --------------------------------------------------------------------------- #
class _SinusoidalPositionalEncoding(nn.Module):
    """Encoding posicional fijo de "Attention is all you need".

    ``PE(p, 2i) = sin(p / 10000^(2i/d))`` y ``PE(p, 2i+1) = cos(p / 10000^(2i/d))``:
    cada par de dimensiones es una onda de otra frecuencia, asi que la posicion
    de cada dia de la ventana queda codificada sin entrenar nada, y la distancia
    entre dos dias es una rotacion fija (la atencion puede usar "hace k dias").
    La posicion 0 es el token [CLS]; las siguientes, los dias de la ventana del
    mas viejo al mas reciente. No es un parametro: no se guarda en el artefacto.
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        posicion = torch.arange(max_len, dtype=torch.float32)[:, None]
        frecuencia = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(posicion * frecuencia)
        pe[0, :, 1::2] = torch.cos(posicion * frecuencia)[:, : d_model // 2]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


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
        self.pos_encoding = _SinusoidalPositionalEncoding(window + 1, d_model)

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
    # Red y entrenamiento: los de la busqueda con Optuna (``src.simulacion.
    # optuna_transformer``, corrida ``transformer``, trial 17): learning rate mas
    # bajo y red mas chica. En la confirmacion (5 semillas x 2 cortes) le gano a
    # la config anterior (``CONFIG_ORIGINAL``) por +0,021 de acierto balanceado
    # y +0,053 de IC, en 7 y 8 de 10 pares; IC positivo en los dos cortes.
    window: int = 20
    horizon: int = 5  # dias hacia adelante para definir la tendencia a predecir
    d_model: int = 16
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 32
    dropout: float = 0.3593
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1.783e-4
    weight_decay: float = 2.083e-3  # regularizacion L2 contra el overfitting
    patience: int = 3  # early stopping: la mejor epoca sale entre la 1 y la 4
    val_fraction: float = 0.15
    neutral_band: float = 0.01  # |retorno esperado| por debajo => "neutral"
    warmup_epochs: int = 3  # calentamiento de learning rate, tipico en Transformers
    seed: int = 42
    # "clasificacion": la red da el logit de "sube" (target > 0) y se entrena
    # con BCE balanceada (pos_weight = bajas / subas de train), asi no aprende
    # "siempre alza" por la deriva en pesos. "regresion": el retorno con MSE /
    # Huber (el modelo original). Ver ``_salida`` para el retorno esperado.
    objetivo: str = "clasificacion"
    # "usd": precios pasados a dolares con el CCL del dia antes de calcular
    # features y target (en pesos, "sube" acierta solo por la inflacion y la
    # devaluacion). ``predicted_close`` se informa en pesos, a CCL constante.
    moneda: str = "usd"
    # Saltos sin ajustar (splits, canjes) y precios <= 0 corregidos como en la
    # simulacion (``src.simulacion.limpieza``), al entrenar y al predecir.
    limpiar: bool = True
    # Feature extra: cambio log a 5 ruedas del riesgo pais (con su retraso de
    # publicacion, ``src.macro_data.SERIES``), igual para todas las acciones.
    riesgo_pais: bool = True
    # Walk-forward: la ultima ``val_fraction`` de las fechas se parte en
    # ``folds`` bloques consecutivos; cada bloque se valida con una red
    # entrenada solo con lo anterior (menos ``gap``). Las metricas ``val_*``
    # salen de esas predicciones fuera de muestra y la cantidad de epocas, de
    # la mediana de las mejores de cada bloque.
    folds: int = 3
    # Al final se entrena con todas las fechas (incluida la validacion, la
    # parte mas reciente) por esa cantidad de epocas. Sin esto, el modelo final
    # es el del ultimo bloque y nunca ve la ultima ``val_fraction / folds``.
    reentrenar_completo: bool = True
    # --- Variantes en evaluacion (``CONFIG_ORIGINAL`` reproduce el modelo original) ---
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
    # Fechas descartadas entre train y validacion: con target a `horizon`
    # ruedas, las ultimas ventanas de train comparten futuro con las primeras
    # de validacion. None = `horizon` (sin solapamiento).
    gap: int | None = None
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
    escala_vol: bool = True
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
            "objetivo": ("clasificacion", "regresion"),
            "moneda": ("usd", "ars"),
        }
        for nombre, validos in opciones.items():
            if getattr(self, nombre) not in validos:
                raise ValueError(f"{nombre} debe ser uno de {validos}")
        if self.objetivo == "clasificacion" and (self.loss != "mse" or self.clip_target):
            raise ValueError("loss y clip_target solo aplican con objetivo='regresion'")
        if self.n_semillas < 1:
            raise ValueError("n_semillas debe ser al menos 1")
        if self.gap is not None and self.gap < 0:
            raise ValueError("gap no puede ser negativo")
        if self.folds < 1:
            raise ValueError("folds debe ser al menos 1")
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

    @property
    def necesita_proveedor(self) -> bool:
        """Si hace falta el CCL o el riesgo pais (``Proveedor``) para entrenar y predecir."""
        return self.moneda == "usd" or self.riesgo_pais


# Opciones que reproducen el Transformer original (hiperparametros de antes de
# Optuna, pesos, regresion, sin limpieza, un solo holdout). Las usan las
# variantes de la simulacion, para compararse con corridas viejas, y ``load``
# para los artefactos anteriores.
CONFIG_ORIGINAL: dict = {
    "window": 30,
    "d_model": 32,
    "dim_feedforward": 64,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "patience": 8,
    "objetivo": "regresion",
    "moneda": "ars",
    "limpiar": False,
    "riesgo_pais": False,
    "escala_vol": False,
    "folds": 1,
    "reentrenar_completo": False,
}

# ``proveedor(hasta) -> (ccl, macro)``: CCL y tabla macro (con la columna
# ``riesgo_pais``) con datos hasta ``hasta``, la misma firma que los modelos de
# panel (``src.panel``).
Proveedor = Callable[[pd.Timestamp], tuple[pd.Series | None, pd.DataFrame | None]]
_SERIES_PROVEEDOR = ("ccl", "riesgo_pais")
_proveedor_del_dia: tuple[dt.date, Proveedor] | None = None
_proveedor_lock = threading.Lock()


def proveedor_collector() -> Proveedor:
    """CCL y riesgo pais del data-colector, bajados una vez por dia (la API
    corre dias seguidos con el mismo proceso)."""
    global _proveedor_del_dia
    with _proveedor_lock:
        hoy = dt.date.today()
        if _proveedor_del_dia is None or _proveedor_del_dia[0] != hoy:
            crudas = {n: fetch_collector_series(*SERIES[n][:2]) for n in _SERIES_PROVEEDOR}

            @functools.lru_cache(maxsize=1024)
            def proveedor(hasta: pd.Timestamp):
                cortas = {n: s[s.index <= hasta] for n, s in crudas.items()}
                macro = build_macro_frame({"riesgo_pais": cortas["riesgo_pais"]}, end=hasta)
                return cortas["ccl"], macro

            _proveedor_del_dia = (hoy, proveedor)
        return _proveedor_del_dia[1]


def cotiza_en_dolares(symbol: str) -> bool:
    """Commodities (futuros de Yahoo) y especies en dolares de BYMA (sufijo
    ``D``: ALUAD, A3D, ...). YPFD es la excepcion: cotiza en pesos."""
    symbol = symbol.strip().upper()
    if symbol in COMMODITY_YAHOO_SYMBOLS:
        return True
    return symbol.endswith("D") and symbol not in PANEL_LIDER_TICKERS


# --------------------------------------------------------------------------- #
# Armado de datos
# --------------------------------------------------------------------------- #
_Ventanas = tuple[np.ndarray, np.ndarray, np.ndarray]
_FEATURES_PANEL = ["ret5_rel", "ret20_rel", "rank5", "vol20_rel", "mkt_ret5", "mkt_ret20"]
_RUEDAS_VOL = 20
_MINIMO_PANEL = 5  # tickers del panel con dato para que la mediana represente al mercado
_FEATURE_RIESGO_PAIS = "riesgo_pais_chg5"
# Clave del encoding posicional aprendido de las redes anteriores al sinusoidal.
_POSICIONAL_APRENDIDO = "pos_encoding.pos_embedding"
_COLUMNA_RIESGO_PAIS = "riesgo_pais"  # columna que ``_preparar`` agrega al OHLCV


def nombres_features(cfg: TransformerTrainConfig, todas: bool = False) -> list[str]:
    """Features que ve la red (``todas``: antes de aplicar ``columnas``)."""
    nombres = (
        FEATURE_NAMES
        + (_FEATURES_PANEL if cfg.features_panel else [])
        + ([_FEATURE_RIESGO_PAIS] if cfg.riesgo_pais else [])
    )
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
    if cfg.riesgo_pais:
        # Sin la columna (o antes del primer dato) queda NaN y se limpia a 0.
        rp = df.get(_COLUMNA_RIESGO_PAIS, pd.Series(np.nan, index=df.index)).astype(float)
        rp_chg5 = np.log(rp.where(rp > 0)).diff(5).to_numpy(dtype=np.float64)[1:]
        feats = np.column_stack([feats, rp_chg5])
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
    # Solo el dia: Yahoo manda las acciones a las 14:00 y los futuros a las
    # 04:00, y el corte por fecha y el IC agrupan tickers del mismo dia.
    fechas = df.index.normalize()[[i + cfg.window for i in validas]].to_numpy()
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


def _bloques_walk_forward(
    por_ticker: dict[str, _Ventanas], val_fraction: float, folds: int, gap: int
) -> list[tuple[np.datetime64, np.datetime64, np.datetime64]]:
    """Bloques ``(fin_train, inicio, fin)`` del walk-forward, comunes a todos los tickers.

    La ultima ``val_fraction`` de las fechas (de todos los tickers juntos) se
    parte en ``folds`` bloques consecutivos ``[inicio, fin)``; cada uno se
    entrena con las fechas anteriores a ``fin_train``, con ``gap`` fechas
    descartadas en el medio. Cortar por fecha y no por ticker evita que las
    fechas de validacion de un ticker sean de train para otro: las acciones se
    mueven juntas y el movimiento del mercado de esos dias se filtraria.
    """
    fechas = np.unique(np.concatenate([f for _, _, f in por_ticker.values()]))
    n = len(fechas)
    n_val = int(round(n * val_fraction))
    if n_val == 0:
        return []
    folds = min(folds, n_val)
    inicios = [n - n_val + k * n_val // folds for k in range(folds)]
    if inicios[0] - gap <= 0:
        raise NotEnoughDataError(f"muy pocas fechas ({n}) para validar {n_val} con gap {gap}")
    finales = [fechas[i] for i in inicios[1:]] + [fechas[-1] + np.timedelta64(1, "D")]
    return [(fechas[i - gap], fechas[i], f) for i, f in zip(inicios, finales, strict=True)]


def _juntar(por_ticker: dict[str, _Ventanas], desde, hasta) -> _Ventanas:
    """Ventanas de todos los tickers con fecha en ``[desde, hasta)`` (None = sin limite)."""
    partes = []
    for X, y, fechas in por_ticker.values():
        m = np.ones(len(fechas), dtype=bool)
        if desde is not None:
            m &= fechas >= desde
        if hasta is not None:
            m &= fechas < hasta
        partes.append((X[m], y[m], fechas[m]))
    return tuple(np.concatenate(cols) for cols in zip(*partes, strict=True))


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
    # Retorno diario del mercado (mediana del panel lider) visto al entrenar:
    # al predecir una sola accion, separa un salto propio de un derrumbe
    # general (``src.simulacion.limpieza``). Se guarda en el artefacto.
    mercado: pd.Series | None = None
    # CCL y riesgo pais (``Proveedor``). No se guarda: None = data-colector.
    proveedor: Proveedor | None = field(default=None, repr=False)

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
        return self.fit_ventanas(self.ventanas(histories))

    def ventanas(self, histories: dict[str, pd.DataFrame]) -> dict[str, _Ventanas]:
        """Ventanas ``(X, y, fechas)`` de cada ticker que entra al entrenamiento:
        limpias, en la moneda del modelo y con sus features. Dependen solo de la
        parte de datos de la config (``window``, ``horizon``, ``moneda``, ...):
        una busqueda de hiperparametros de la red las arma una vez."""
        cfg = self.config
        if cfg.universo == "panel":
            histories = {s: df for s, df in histories.items() if s in PANEL_LIDER_TICKERS}
        histories = {s: df for s, df in histories.items() if self.acepta(s)}
        histories = self._preparar_todos(histories)
        contexto = contexto_panel(histories) if cfg.features_panel else None
        if cfg.features_panel and contexto is None:
            raise NotEnoughDataError("features_panel necesita al menos 5 tickers del panel")

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
        return por_ticker

    def fit_ventanas(
        self,
        por_ticker: dict[str, _Ventanas],
        al_terminar_bloque: Callable[[int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        """Walk-forward y entrenamiento final sobre ventanas ya armadas (``ventanas``).

        ``al_terminar_bloque(k, metricas)`` se llama despues de cada bloque con
        las metricas fuera de muestra acumuladas hasta ahi (una busqueda puede
        cortar ahi un candidato malo).
        """
        cfg = self.config
        gap = cfg.horizon if cfg.gap is None else cfg.gap
        bloques = _bloques_walk_forward(por_ticker, cfg.val_fraction, cfg.folds, gap)
        self.tickers = list(por_ticker)

        # Walk-forward: una red (o un ensamble) por bloque, que predice su bloque
        # sin haberlo visto. De ahi salen las metricas y las epocas.
        preds, reales, fechas_oos, epocas = [], [], [], []
        for k, (fin_train, inicio, fin) in enumerate(bloques):
            if len(bloques) > 1:
                print(f"  bloque {k + 1}/{len(bloques)} (valida desde {str(inicio)[:10]})")
            Xtr, ytr, _ = _juntar(por_ticker, None, fin_train)
            Xva, yva, fva = _juntar(por_ticker, inicio, fin)
            nets, mejores = self._entrenar_ensamble(Xtr, ytr, Xva, yva, fva)
            epocas += mejores
            self._nets = nets
            preds.append(self._salida(self._to_tensor(self._scale_features(Xva))))
            reales.append(yva)
            fechas_oos.append(fva)
            if al_terminar_bloque is not None:
                al_terminar_bloque(
                    k,
                    self._metricas(
                        np.concatenate(preds), np.concatenate(reales), np.concatenate(fechas_oos)
                    ),
                )

        n_epocas = max(1, int(np.median(epocas))) if epocas else cfg.epochs
        if not bloques or cfg.reentrenar_completo:
            X, y, _ = _juntar(por_ticker, None, None)
            self._nets, _ = self._entrenar_ensamble(X, y, epocas=n_epocas)

        self.trained_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        self.metrics = {}
        if preds:
            pred = np.concatenate(preds)
            self.metrics = self._metricas(pred, np.concatenate(reales), np.concatenate(fechas_oos))
            self.metrics["folds"] = len(bloques)
            self.metrics["epocas"] = n_epocas
            if cfg.banda_cuantil is not None:
                cfg.neutral_band = float(np.quantile(np.abs(np.expm1(pred)), cfg.banda_cuantil))
                self.metrics["neutral_band"] = round(cfg.neutral_band, 6)
        return self.metrics

    def _entrenar_ensamble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Xva: np.ndarray | None = None,
        yva: np.ndarray | None = None,
        fva: np.ndarray | None = None,
        epocas: int | None = None,
    ) -> tuple[list[_TransformerRegressor], list[int]]:
        """Ajusta las escalas con ``X``/``y`` y entrena ``n_semillas`` redes (con
        early stopping si hay validacion, si no ``epocas``). Devuelve las redes y
        la mejor epoca de cada una."""
        cfg = self.config
        # Normalizacion calculada solo con el set de entrenamiento.
        flat = X.reshape(-1, X.shape[-1])
        self.feature_mean = flat.mean(axis=0)
        self.feature_std = flat.std(axis=0) + 1e-8
        has_val = Xva is not None and len(Xva) > 0
        if cfg.objetivo == "clasificacion":
            # target_std guarda el movimiento tipico (|retorno| medio) para pasar
            # la probabilidad a un retorno esperado (ver ``_salida``).
            self.target_mean = 0.0
            self.target_std = float(np.mean(np.abs(y)) + 1e-8)
            y_scaled = (y > 0).astype(np.float32)
            yva_scaled = (yva > 0).astype(np.float32) if has_val else None
        else:
            self.target_mean = float(y.mean())
            self.target_std = float(y.std() + 1e-8)
            y_scaled = (y - self.target_mean) / self.target_std
            if cfg.clip_target is not None:
                y_scaled = np.clip(y_scaled, -cfg.clip_target, cfg.clip_target)
            yva_scaled = (yva - self.target_mean) / self.target_std if has_val else None
        disp = torch.device(self.dispositivo)
        X_t = self._to_tensor(self._scale_features(X)).to(disp)
        y_t = torch.tensor(y_scaled, dtype=torch.float32, device=disp)
        Xva_t = self._to_tensor(self._scale_features(Xva)) if has_val else None

        nets, mejores = [], []
        for k in range(cfg.n_semillas):
            if cfg.n_semillas > 1:
                print(f"  semilla {k + 1}/{cfg.n_semillas}")
            net, mejor = self._entrenar_red(
                cfg.seed + k, X_t, y_t, Xva_t, yva_scaled, fva, epocas=epocas
            )
            nets.append(net)
            mejores.append(mejor)
        return nets, mejores

    def _entrenar_red(
        self,
        seed: int,
        Xtr_t: torch.Tensor,
        ytr_t: torch.Tensor,
        Xva_t: torch.Tensor | None,
        yva_scaled: np.ndarray | None,
        fva: np.ndarray | None,
        epocas: int | None = None,
    ) -> tuple[_TransformerRegressor, int]:
        """Entrena una red y devuelve la mejor epoca (1-based). Con validacion,
        early stopping; sin validacion, ``epocas`` (default ``cfg.epochs``)."""
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
        loss_fn: nn.Module
        if cfg.objetivo == "clasificacion":
            subas = float(ytr_t.sum().item())
            bajas = float(len(ytr_t)) - subas
            peso = torch.tensor(bajas / max(subas, 1.0), device=Xtr_t.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=peso)
        elif cfg.loss == "huber":
            loss_fn = nn.HuberLoss(delta=1.0)
        else:
            loss_fn = nn.MSELoss()
        # En validacion, la perdida del objetivo sin recortes (MSE en regresion).
        val_fn: nn.Module = loss_fn if cfg.objetivo == "clasificacion" else nn.MSELoss()

        has_val = Xva_t is not None
        best_val = float("inf")
        best_state = None
        best_epoch = epocas or cfg.epochs
        epochs_no_improve = 0

        n = len(Xtr_t)
        for epoch in range(epocas or cfg.epochs):
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
                val_loss = float(val_fn(val_pred, yva_t).item())
                if cfg.seleccion == "ic":
                    val_score = -ic_por_fecha(val_pred.cpu().numpy(), yva_scaled, fva)
                else:
                    val_score = val_loss
                if val_score < best_val - 1e-5:
                    best_val = val_score
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                val_loss = float("nan")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  epoch {epoch + 1:>3}/{cfg.epochs}  "
                    f"train_loss={epoch_loss / n:.5f}  val_loss={val_loss:.5f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if has_val and epochs_no_improve >= cfg.patience:
                print(f"  early stopping en epoca {epoch + 1} (mejor val={best_val:.5f})")
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net.cpu(), best_epoch

    # ------------------------------------------------------------------ #
    def _preparar_todos(self, histories: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Historicos para entrenar: limpios (y guarda el mercado con el que se
        detectaron los saltos), en la moneda del modelo y con el riesgo pais."""
        cfg = self.config
        if cfg.limpiar:
            positivos = {s: sin_precios_no_positivos(df) for s, df in histories.items()}
            self.mercado = retorno_mercado(positivos)
            histories = {s: self._limpiar(df) for s, df in positivos.items()}
        if not cfg.necesita_proveedor or not histories:
            return histories
        hasta = max(df.index[-1] for df in histories.values() if not df.empty)
        ccl, macro = self._proveedor()(hasta)
        return {s: self._en_moneda(df, ccl, macro) for s, df in histories.items()}

    def _preparar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lo mismo que ``_preparar_todos`` para un solo historico (al predecir)."""
        cfg = self.config
        if cfg.limpiar:
            df = self._limpiar(sin_precios_no_positivos(df))
        if not cfg.necesita_proveedor or df.empty:
            return df
        ccl, macro = self._proveedor()(df.index[-1])
        return self._en_moneda(df, ccl, macro)

    def _limpiar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta los saltos de ``df`` que el mercado no acompano (sin mercado
        conocido para esa fecha, se compara contra 0)."""
        mercado = self.mercado if self.mercado is not None else pd.Series(dtype=float)
        for fecha, factor in detectar_saltos(df, mercado):
            df = ajustar_salto(df, fecha, factor)
        return df

    def _en_moneda(
        self, df: pd.DataFrame, ccl: pd.Series | None, macro: pd.DataFrame | None
    ) -> pd.DataFrame:
        cfg = self.config
        if cfg.moneda == "usd":
            if ccl is None or ccl.dropna().empty:
                raise DataUnavailableError("sin CCL para pasar los precios a dolares")
            # Antes del primer CCL conocido no hay precio en dolares.
            df = en_dolares(df, ccl).dropna(subset=["close"])
        if cfg.riesgo_pais:
            df = df.copy()
            rp = None if macro is None else macro.get(_COLUMNA_RIESGO_PAIS)
            df[_COLUMNA_RIESGO_PAIS] = (
                np.nan if rp is None else rp.reindex(df.index.normalize()).to_numpy()
            )
        return df

    def acepta(self, symbol: str) -> bool:
        """Si el modelo puede predecir ``symbol``: en dolares, no los que ya
        cotizan en dolares (dividirlos por el CCL no tiene sentido)."""
        return self.config.moneda != "usd" or not cotiza_en_dolares(symbol)

    def _proveedor(self) -> Proveedor:
        return self.proveedor if self.proveedor is not None else proveedor_collector()

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
        ventana, escala = self._ventana_prediccion(self._preparar(df), None)
        log_return = float(self._log_returns(ventana[None], np.array([escala]))[0])
        return derive_trend_output(df, log_return, self.config.horizon, self.config.neutral_band)

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """Predice varios tickers del mismo dia en un solo batch. Es la forma de
        predecir con ``features_panel`` (el contexto sale de ``dfs``); los
        tickers sin datos suficientes se omiten."""
        if not self._nets:
            raise NotEnoughDataError("el modelo no esta entrenado")
        preparados = {s: self._preparar(df) for s, df in dfs.items()}
        contexto = contexto_panel(preparados) if self.config.features_panel else None
        ventanas, escalas, symbols = [], [], []
        for symbol, df in preparados.items():
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
        return self._salida(self._to_tensor(self._scale_features(ventanas))) * escalas

    def _salida(self, x: torch.Tensor) -> np.ndarray:
        """Retorno esperado en unidades del target.

        En regresion es la prediccion desescalada. En clasificacion, con ``p``
        la probabilidad (balanceada) de que suba y ``m`` el movimiento tipico de
        train, es ``p * m - (1 - p) * m = (2p - 1) * m``: el signo es el de
        ``p - 0.5`` y el tamano crece con la conviccion. Con ``p`` cerca de 0.5
        el retorno queda dentro de ``neutral_band`` y la senal es "neutral".
        """
        scaled = self._forward(x)
        if self.config.objetivo == "clasificacion":
            return np.tanh(scaled / 2) * self.target_std  # tanh(z / 2) = 2 * sigmoid(z) - 1
        return scaled * self.target_std + self.target_mean

    def _forward(self, x: torch.Tensor) -> np.ndarray:
        """Salida cruda (logit o prediccion escalada): promedio de las redes del ensamble."""
        with torch.no_grad():
            return np.mean([net(x).numpy() for net in self._nets], axis=0)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _metricas(
        pred: np.ndarray, y: np.ndarray, fechas: np.ndarray | None = None
    ) -> dict[str, float]:
        """Metricas de ``pred`` contra ``y`` (en unidades del target: con
        ``escala_vol``, retornos sobre volatilidad)."""
        if len(y) == 0:
            return {}
        mae = float(np.mean(np.abs(pred - y)))
        # Acierto direccional: que el signo del retorno coincida.
        directional = float(np.mean(np.sign(pred) == np.sign(y)))
        metrics = {
            "val_samples": float(len(y)),
            "val_mae_logret": round(mae, 6),
            "val_directional_accuracy": round(directional, 4),
            # Sin inflarse en periodos alcistas: decir siempre lo mismo da 0.5.
            "val_acierto_balanceado": round(acierto_balanceado(pred, y), 4),
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
                "mercado": self.mercado,
                "feature_names": FEATURE_NAMES,
                "architecture": "transformer",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, proveedor: Proveedor | None = None) -> TransformerTrendModel:
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        check_feature_compatibility(blob.get("feature_names"))
        estados = blob.get("state_dicts") or [blob["state_dict"]]
        if any(_POSICIONAL_APRENDIDO in estado for estado in estados):
            raise StaleArtifactError(
                "el artefacto usa el encoding posicional aprendido; hay que reentrenarlo"
            )
        # Las opciones que no estan en el artefacto son de antes de existir:
        # las del modelo original.
        cfg = TransformerTrainConfig(**(CONFIG_ORIGINAL | blob["config"]))
        model = cls(
            config=cfg,
            feature_mean=np.asarray(blob["feature_mean"]),
            feature_std=np.asarray(blob["feature_std"]),
            target_mean=blob["target_mean"],
            target_std=blob["target_std"],
            tickers=blob.get("tickers", []),
            trained_at=blob.get("trained_at"),
            metrics=blob.get("metrics", {}),
            mercado=blob.get("mercado"),
            proveedor=proveedor,
        )
        for state in estados:
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
