"""Modelo "macro": predice alza/baja a 20 ruedas con variables macro de Argentina."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from src.data_quality import drop_stale, is_stale
from src.errors import NotEnoughDataError, StaleArtifactError
from src.indicators import macd_histogram, rsi_series
from src.macro_data import todays_macro_frame
from src.trend_common import derive_trend_output

HORIZON = 20
MIN_ROWS = 260
UNIVERSE = (
    "CRES", "ALUA", "TECO2", "BBAR", "METR", "SUPV", "BYMA", "PAMP", "COME", "CEPU",
    "EDN", "TRAN", "GGAL", "TXAR", "VALO", "TGSU2", "TGNO4", "YPFD", "BMA", "LOMA",
)  # fmt: skip
ELECTIONS = pd.to_datetime(
    [
        "2017-10-22", "2019-08-11", "2019-10-27", "2021-09-12", "2021-11-14",
        "2023-08-13", "2023-10-22", "2023-11-19", "2025-10-26", "2027-10-24",
    ]
)  # fmt: skip

TECH_FEATURES = [
    "t_ret1", "t_ret3", "t_ret5", "t_ret10", "t_ret20", "t_ret60", "t_ret120", "t_ret250",
    "t_down_streak", "t_up_streak", "t_dd252", "t_dd60", "t_dist_min60", "t_new_low252",
    "t_sma20", "t_sma50", "t_sma200", "t_death_cross", "t_gap", "t_gap_down",
    "t_vol5", "t_vol20", "t_vol60", "t_vol_ratio5_20", "t_vol_ratio20_60", "t_downdev",
    "t_skew20", "t_kurt20", "t_gk20", "t_gk_ratio", "t_volz", "t_volz_x_ret", "t_vol_chg5",
    "t_obv_slope", "t_amihud", "t_rsi14", "t_rsi5", "t_macd", "t_stoch", "t_boll",
]  # fmt: skip
MACRO_FEATURES = [
    "a_ccl1", "a_ccl5", "a_ccl20", "a_ccl60", "a_brecha", "a_brecha_chg20", "a_brecha_z",
    "a_mep_ccl", "a_blue_gap", "a_of20", "a_rp", "a_rp_chg5", "a_rp_chg20", "a_rp_z",
    "a_res20", "a_res60", "a_badlar", "a_badlar_chg20", "a_bm20", "a_real_rate",
]  # fmt: skip
EVENT_FEATURES = ["e_days_to_el", "e_days_since_el", "e_el_window", "e_month_end", "e_aguinaldo"]
FEATURES = TECH_FEATURES + MACRO_FEATURES + EVENT_FEATURES


def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Indicadores tecnicos de la accion."""
    c, o, h, lo = df["close"], df["open"], df["high"], df["low"]
    volume = df["volume"].replace(0, np.nan)
    log_c = np.log(c)
    r = log_c.diff()
    f = pd.DataFrame(index=df.index)
    for k in (1, 3, 5, 10, 20, 60, 120, 250):
        f[f"t_ret{k}"] = log_c.diff(k)
    down, up = (r < 0).astype(int), (r > 0).astype(int)
    group = (down.diff() != 0).cumsum()
    f["t_down_streak"] = down.groupby(group).cumsum() * down
    group = (up.diff() != 0).cumsum()
    f["t_up_streak"] = up.groupby(group).cumsum() * up
    f["t_dd252"] = c / c.rolling(252, min_periods=60).max() - 1
    f["t_dd60"] = c / c.rolling(60).max() - 1
    f["t_dist_min60"] = c / c.rolling(60).min() - 1
    f["t_new_low252"] = (c <= c.rolling(252, min_periods=60).min() * 1.0001).astype(float)
    s20, s50 = c.rolling(20).mean(), c.rolling(50).mean()
    s200 = c.rolling(200, min_periods=120).mean()
    f["t_sma20"], f["t_sma50"], f["t_sma200"] = c / s20 - 1, c / s50 - 1, c / s200 - 1
    f["t_death_cross"] = (s50 < s200).astype(float)
    f["t_gap"] = o / c.shift() - 1
    f["t_gap_down"] = (f["t_gap"] < -0.02).astype(float)
    for k in (5, 20, 60):
        f[f"t_vol{k}"] = r.rolling(k).std()
    f["t_vol_ratio5_20"] = f["t_vol5"] / f["t_vol20"]
    f["t_vol_ratio20_60"] = f["t_vol20"] / f["t_vol60"]
    f["t_downdev"] = r.where(r < 0).rolling(20, min_periods=3).std() / f["t_vol20"]
    f["t_skew20"], f["t_kurt20"] = r.rolling(20).skew(), r.rolling(20).kurt()
    hl = np.log(h / lo.replace(0, np.nan)) ** 2
    co = np.log(c / o.replace(0, np.nan)) ** 2
    gk = (0.5 * hl - (2 * np.log(2) - 1) * co).clip(lower=0)
    f["t_gk20"] = np.sqrt(gk.rolling(20).mean())
    f["t_gk_ratio"] = np.sqrt(gk.rolling(5).mean()) / f["t_gk20"].replace(0, np.nan)
    log_volume = np.log1p(df["volume"])
    f["t_volz"] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()
    f["t_volz_x_ret"] = f["t_volz"].fillna(0) * np.sign(r)
    f["t_vol_chg5"] = log_volume.diff(5)
    obv = (np.sign(r).fillna(0) * df["volume"]).cumsum()
    f["t_obv_slope"] = (obv - obv.shift(20)) / (df["volume"].rolling(20).sum() + 1)
    amihud = (r.abs() / (c * volume)).rolling(20).mean()
    f["t_amihud"] = np.log1p(amihud / (amihud.rolling(250, min_periods=60).median() + 1e-12))
    close = np.asarray(c, dtype=float)
    f["t_rsi14"], f["t_rsi5"] = rsi_series(close), rsi_series(close, 5)
    f["t_macd"] = macd_histogram(close) / close
    lo14, hi14 = lo.rolling(14).min(), h.rolling(14).max()
    f["t_stoch"] = (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    f["t_boll"] = (c - s20) / c.rolling(20).std()
    return f.replace([np.inf, -np.inf], np.nan)


def macro_features(macro: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Variables macro alineadas a ``index`` con el ultimo dato conocido."""
    ccl, oficial = macro["ccl"], macro["oficial"]
    rp = macro["riesgo_pais"]
    a = pd.DataFrame(index=macro.index)
    for k in (1, 5, 20, 60):
        a[f"a_ccl{k}"] = np.log(ccl).diff(k)
    a["a_brecha"] = ccl / oficial - 1
    a["a_brecha_chg20"] = a["a_brecha"].diff(20)
    roll = a["a_brecha"].rolling(250, min_periods=60)
    a["a_brecha_z"] = (a["a_brecha"] - roll.mean()) / roll.std()
    a["a_mep_ccl"] = _col(macro, "mep") / ccl - 1
    a["a_blue_gap"] = _col(macro, "blue") / oficial - 1
    a["a_of20"] = np.log(_col(macro, "mayorista")).diff(20)
    a["a_rp"] = rp
    a["a_rp_chg5"], a["a_rp_chg20"] = np.log(rp).diff(5), np.log(rp).diff(20)
    a["a_rp_z"] = (rp - rp.rolling(250, min_periods=60).mean()) / rp.rolling(
        250, min_periods=60
    ).std()
    reservas = _col(macro, "reservas")
    a["a_res20"], a["a_res60"] = np.log(reservas).diff(20), np.log(reservas).diff(60)
    a["a_badlar"] = macro["badlar"]
    a["a_badlar_chg20"] = macro["badlar"].diff(20)
    a["a_bm20"] = np.log(_col(macro, "base_monetaria")).diff(20)
    a["a_real_rate"] = macro["badlar"] - (np.log(ccl).diff(20) * (365 / 20) * 100)
    a = a.replace([np.inf, -np.inf], np.nan)
    return a.reindex(index, method="ffill")[MACRO_FEATURES]


def _col(macro: pd.DataFrame, name: str) -> pd.Series:
    """Columna opcional: si falta queda en NaN."""
    return macro[name] if name in macro.columns else pd.Series(np.nan, index=macro.index)


def event_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cercania a elecciones, fin de mes y aguinaldo."""
    days = index.values.astype("datetime64[D]")
    elections = ELECTIONS.values.astype("datetime64[D]")
    to_next = [
        (elections[elections >= d].min() - d).astype(int) if (elections >= d).any() else 999
        for d in days
    ]
    since_last = [
        (d - elections[elections <= d].max()).astype(int) if (elections <= d).any() else 999
        for d in days
    ]
    e = pd.DataFrame(index=index)
    e["e_days_to_el"] = np.minimum(to_next, 90)
    e["e_days_since_el"] = np.minimum(since_last, 90)
    e["e_el_window"] = ((e["e_days_to_el"] <= 10) | (e["e_days_since_el"] <= 10)).astype(float)
    e["e_month_end"] = (index.is_month_end | (index + pd.Timedelta(days=3)).is_month_start).astype(
        float
    )
    e["e_aguinaldo"] = index.month.isin([6, 12]).astype(float)
    return e


def build_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Variables de una accion, en el orden de ``FEATURES``."""
    f = technical_features(df)
    f = f.join(macro_features(macro, df.index)).join(event_features(df.index))
    return f[FEATURES]


@dataclass
class MacroConfig:
    horizon: int = HORIZON
    signal_quantile: float = 0.12  # "alza"/"baja" solo en el 12% mas extremo de cada lado
    oos_fraction: float = (
        0.25  # ultimo tramo de fechas usado para medir y calibrar fuera de muestra
    )
    oos_folds: int = 4  # en cuantos reentrenos sucesivos se parte ese tramo
    seed: int = 0


def _new_models(seed: int):
    return (
        HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=500,
            l2_regularization=5.0, random_state=seed,
        ),
        make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=200, min_samples_leaf=200, max_features=0.3, n_jobs=1,
                random_state=seed,
            ),
        ),
    )  # fmt: skip


def _fit_models(x: np.ndarray, y: np.ndarray, seed: int) -> tuple:
    models = _new_models(seed)
    for m in models:
        m.fit(x, y)
    return models


def _score(models: tuple, x: np.ndarray) -> np.ndarray:
    """P(baja): promedio de los dos modelos del ensamble."""
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


@dataclass
class MacroTrendModel:
    config: MacroConfig = field(default_factory=MacroConfig)
    tickers: list[str] = field(default_factory=list)
    trained_at: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    baja_threshold: float | None = None  # P(baja) >= umbral -> "baja"
    alza_threshold: float | None = None  # P(baja) <= umbral -> "alza"
    alza_move: float = 0.0  # retorno log tipico cuando la senal de alza acierta
    baja_move: float = 0.0  # retorno log tipico cuando la senal de baja acierta (negativo)
    macro_provider: Callable[[], pd.DataFrame] | None = None
    _models: tuple | None = None

    @property
    def version(self) -> str:
        return "untrained" if self.trained_at is None else f"macro-{self.trained_at}"

    def _panel(self, histories: dict[str, pd.DataFrame], macro: pd.DataFrame) -> pd.DataFrame:
        h = self.config.horizon
        frames = []
        for symbol, df in histories.items():
            f = build_features(df, macro)
            f["fwd"] = df["close"].shift(-h) / df["close"] - 1.0
            f["ticker"] = symbol
            frames.append(f)
        panel = pd.concat(frames)
        panel = panel[panel["fwd"].notna() & (panel["fwd"] != 0)]
        panel["down"] = (panel["fwd"] < 0).astype(int)
        return panel.sort_index(kind="stable")

    def _oos_scores(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Predicciones fuera de muestra del ultimo tramo, con reentrenos sucesivos."""
        cfg = self.config
        days = np.sort(panel.index.unique())
        first = int(len(days) * (1 - cfg.oos_fraction))
        edges = np.linspace(first, len(days), cfg.oos_folds + 1).astype(int)
        index, x, y = panel.index.values, panel[FEATURES].to_numpy(), panel["down"].to_numpy()
        parts = []
        for k in range(cfg.oos_folds):
            start = days[edges[k]]
            end = days[edges[k + 1]] if edges[k + 1] < len(days) else None
            purge = days[max(edges[k] - cfg.horizon, 0)]
            tr = np.where(index <= purge)[0]
            te = np.where((index >= start) & ((index < end) if end is not None else True))[0]
            if len(tr) < 3000 or len(te) == 0:
                continue
            models = _fit_models(x[tr], y[tr], cfg.seed)
            parts.append(
                pd.DataFrame(
                    {"p": _score(models, x[te]), "down": y[te], "fwd": panel["fwd"].to_numpy()[te]},
                    index=panel.index[te],
                )
            )
        if not parts:
            raise NotEnoughDataError("no hay historia suficiente para medir fuera de muestra")
        return pd.concat(parts)

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
        baja_hits = float(down[is_baja].mean()) if is_baja.any() else float("nan")
        alza_hits = float(1 - down[is_alza].mean()) if is_alza.any() else float("nan")
        self.metrics = {
            "oos_samples": float(len(oos)),
            "oos_auc": round(float(roc_auc_score(down, p)), 4),
            "oos_down_rate": round(float(down.mean()), 4),
            "oos_baja_samples": float(is_baja.sum()),
            "oos_baja_hit_rate": round(baja_hits, 4),
            "oos_alza_samples": float(is_alza.sum()),
            "oos_alza_hit_rate": round(alza_hits, 4),
        }
        self._models = _fit_models(panel[FEATURES].to_numpy(), panel["down"].to_numpy(), cfg.seed)
        self.trained_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        return self.metrics

    def fit(self, histories: dict[str, pd.DataFrame], macro: pd.DataFrame) -> dict[str, float]:
        histories = {s: d for s, d in drop_stale(histories).items() if len(d) >= MIN_ROWS}
        if not histories:
            raise NotEnoughDataError("no hay suficientes datos para entrenar")
        panel = self._panel(histories, macro)
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
        macro = (self.macro_provider or todays_macro_frame)()
        row = build_features(df, macro).iloc[[-1]].to_numpy()
        return float(_score(self._models, row)[0])

    def predict_df(self, df: pd.DataFrame) -> dict:
        p_down = self.probability_down(df)
        if p_down >= float(self.baja_threshold):
            signal, log_return = "baja", self.baja_move
        elif p_down <= float(self.alza_threshold):
            signal, log_return = "alza", self.alza_move
        else:
            signal, log_return = "neutral", 0.0  # sin vista: precio proyectado = ultimo cierre
        out = derive_trend_output(df, log_return, self.config.horizon, neutral_band=0.0)
        out["signal"] = signal
        out["probability_down"] = round(p_down, 4)
        out["probability_up"] = round(1 - p_down, 4)
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
                "feature_names": FEATURES,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> MacroTrendModel:
        blob = joblib.load(Path(path))
        if blob.get("feature_names") != FEATURES:
            raise StaleArtifactError(
                "el artefacto 'macro' se entreno con otras variables: hay que reentrenarlo"
            )
        try:
            model = cls(
                config=MacroConfig(**blob["config"]),
                tickers=blob.get("tickers", []),
                trained_at=blob.get("trained_at"),
                metrics=blob.get("metrics", {}),
                baja_threshold=blob["baja_threshold"],
                alza_threshold=blob["alza_threshold"],
                alza_move=blob["alza_move"],
                baja_move=blob["baja_move"],
            )
            model._models = blob["models"]
        except (TypeError, KeyError) as exc:
            raise StaleArtifactError(
                f"el artefacto 'macro' tiene un formato desactualizado ({exc!r}): "
                "hay que reentrenarlo"
            ) from exc
        return model
