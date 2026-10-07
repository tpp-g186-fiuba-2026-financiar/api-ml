"""Experimento de modelos de prediccion alza/baja con features de las investigaciones.

Pregunta: ¿algun modelo + conjunto de features distingue dias de suba de dias de
baja mejor que el azar, y mejor que lo que ya tenemos? Se mide con AUC (que
no depende de que el mercado suba mas de lo que baja: 0.50 = azar, 1.00 = perfecto)
y con la tasa de aciertos de las predicciones mas confiables.

Walk-forward honesto: se reentrena cada ~63 ruedas con datos hasta
(inicio del tramo - horizonte), sin mirar el futuro. Universo: 20 acciones locales
(A3 excluida por datos estancados). Prueba desde 2020.

Conjuntos de features (cada uno suma al anterior):
  F0 actuales        7 features que usan hoy los modelos
  F1 + EDA           dist_max60, rsi_vol_interaction, Bollinger (EDA issue #135)
  F2 + series        lags de retornos, medias/desvios moviles, momentum, EMA 12/26,
                     volatilidad Garman-Klass (papers 2 y 3 de la investigacion)
  F3 + mercado       retorno/volatilidad del mercado (promedio de las 20 acciones,
                     proxy del Merval), fuerza relativa y tasa EEUU (TNX)
Modelos: regresion logistica (paper 1), gradient boosting (paper 3, tipo LightGBM)
y SVM rbf (Cao y Tay).

Uso (desde api-ml/):  .venv/bin/python notebooks/evaluacion_modelos.py cache.pkl [out.csv]
"""

from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.indicators import ema, macd_histogram, rsi_series, sma  # noqa: E402

warnings.filterwarnings("ignore")

UNIVERSE = ["CRES", "ALUA", "TECO2", "BBAR", "METR", "SUPV", "BYMA", "PAMP", "COME", "CEPU",
            "EDN", "TRAN", "GGAL", "TXAR", "VALO", "TGSU2", "TGNO4", "YPFD", "BMA", "LOMA"]
HORIZONS = [5, 10, 20]
START, FIRST_TEST, REFIT_EVERY = "2017-01-01", "2020-01-01", 63

F0 = ["log_return", "log_volume_change", "range_pct", "rsi_norm", "sma20_ratio", "macd_norm", "momentum_10"]
F1 = F0 + ["dist_max60", "rsi_vol_interaction", "boll_z"]
F2 = F1 + [f"lag{k}" for k in range(1, 6)] + ["rmean5", "rmean20", "rstd5", "rstd20", "mom5", "mom20",
                                               "ema_ratio", "gk_vol20", "gk_ratio", "oversold"]
F3 = F2 + ["mkt_ret5", "mkt_ret20", "mkt_ret60", "mkt_vol20", "rel20", "mkt_dist_max60", "tnx_chg5"]
SETS = {"F0 actuales": F0, "F1 +EDA": F1, "F2 +series": F2, "F3 +mercado": F3}


def stock_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].to_numpy(float)
    c = df["close"]
    lc = np.log(c)
    r = lc.diff()
    f = pd.DataFrame(index=df.index)
    f["log_return"] = r
    v = df["volume"].replace(0, np.nan)
    f["log_volume_change"] = np.log(v).diff()
    f["range_pct"] = (df["high"] - df["low"]) / c
    f["rsi14"] = rsi_series(close)
    f["rsi_norm"] = (f["rsi14"] - 50) / 50
    f["sma20_ratio"] = close / sma(close, 20) - 1
    f["macd_norm"] = macd_histogram(close) / close
    f["momentum_10"] = lc.diff(10)
    f["dist_max60"] = c / c.rolling(60).max() - 1
    lv = np.log1p(df["volume"])
    f["rsi_vol_interaction"] = f["rsi_norm"] * ((lv - lv.rolling(60).mean()) / lv.rolling(60).std()).fillna(0)
    f["boll_z"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    for k in range(1, 6):
        f[f"lag{k}"] = r.shift(k)
    f["rmean5"], f["rmean20"] = r.rolling(5).mean(), r.rolling(20).mean()
    f["rstd5"], f["rstd20"] = r.rolling(5).std(), r.rolling(20).std()
    f["mom5"], f["mom20"] = lc.diff(5), lc.diff(20)
    f["ema_ratio"] = ema(close, 12) / ema(close, 26) - 1
    # Garman-Klass (paper 2): volatilidad diaria con apertura/maximo/minimo/cierre
    hl = np.log(df["high"] / df["low"].replace(0, np.nan)) ** 2
    co = np.log(c / df["open"].replace(0, np.nan)) ** 2
    gk = (0.5 * hl - (2 * np.log(2) - 1) * co).clip(lower=0)
    f["gk_vol20"] = np.sqrt(gk.rolling(20).mean())
    f["gk_ratio"] = np.sqrt(gk.rolling(5).mean()) / f["gk_vol20"].replace(0, np.nan)
    f["oversold"] = (f["rsi14"] <= 30).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)


def build_panel(hist: dict, tnx: pd.Series | None, h: int) -> pd.DataFrame:
    frames, closes = [], {}
    for t in UNIVERSE:
        df = hist[t][hist[t].index >= START]
        f = stock_features(df)
        f["fwd"] = df["close"].shift(-h) / df["close"] - 1
        f["ticker"] = t
        frames.append(f)
        closes[t] = df["close"]
    cl = pd.DataFrame(closes).sort_index()
    rets = np.log(cl).diff()
    mkt_r = rets.mean(axis=1)
    mkt_lvl = mkt_r.fillna(0).cumsum()
    mkt = pd.DataFrame(index=cl.index)
    for k in (5, 20, 60):
        mkt[f"mkt_ret{k}"] = mkt_lvl.diff(k)
    mkt["mkt_vol20"] = mkt_r.rolling(20).std()
    mkt["mkt_dist_max60"] = mkt_lvl - mkt_lvl.rolling(60).max()
    mkt["fwd_mkt"] = cl.pct_change(h).shift(-h).mean(axis=1)
    if tnx is not None:
        mkt["tnx_chg5"] = tnx.reindex(cl.index, method="ffill").diff(5)
    else:
        mkt["tnx_chg5"] = 0.0
    p = pd.concat(frames)
    p = p.join(mkt, how="left")
    p["rel20"] = p["mom20"] - p["mkt_ret20"]
    p = p.dropna(subset=["fwd", "rsi14"])
    p = p[p["fwd"] != 0]
    p["up"] = (p["fwd"] > 0).astype(int)
    p["beat"] = (p["fwd"] > p["fwd_mkt"]).astype(int)
    p = p.fillna({c: 0.0 for c in F3})
    return p.sort_index(kind="stable")


def make_model(kind: str):
    if kind == "logit":
        return LogisticRegression(C=0.1, max_iter=400)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                              min_samples_leaf=300, l2_regularization=2.0, random_state=0)
    return SVC(kernel="rbf", C=1.0, gamma="scale")


def walk_forward_scores(p: pd.DataFrame, feats: list[str], kind: str, target: str, h: int) -> pd.Series:
    dates = p.index.to_numpy()
    days = np.unique(dates)
    starts = days[days >= np.datetime64(FIRST_TEST)][::REFIT_EVERY]
    score = pd.Series(np.nan, index=np.arange(len(p)))
    X_all, y_all = p[feats].to_numpy(float), p[target].to_numpy()
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else None
        cut = days[max(np.searchsorted(days, s) - h, 0)]
        tr = np.where(dates <= cut)[0]
        te = np.where((dates >= s) & ((dates < e) if e is not None else True))[0]
        if len(tr) < 1000 or len(te) == 0:
            continue
        if kind == "svm":
            tr = tr[-5000:]
        sc = StandardScaler().fit(X_all[tr])
        xtr = np.clip(sc.transform(X_all[tr]), -5, 5)
        xte = np.clip(sc.transform(X_all[te]), -5, 5)
        m = make_model(kind).fit(xtr, y_all[tr])
        s_te = m.decision_function(xte) if kind == "svm" else m.predict_proba(xte)[:, 1]
        score.iloc[te] = s_te
    return score


def auc_ci(y: np.ndarray, s: np.ndarray, months: np.ndarray, reps: int = 200, seed: int = 0):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    point = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    um = np.unique(months)
    idx_by = {m: np.where(months == m)[0] for m in um}
    vals = []
    for _ in range(reps):
        pick = rng.choice(um, size=len(um), replace=True)
        ix = np.concatenate([idx_by[m] for m in pick])
        if len(np.unique(y[ix])) > 1:
            vals.append(roc_auc_score(y[ix], s[ix]))
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main() -> None:
    hist = pickle.load(open(sys.argv[1], "rb"))
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("evaluacion_modelos.csv")
    try:
        from src.data import fetch_interest_rate

        tnx = fetch_interest_rate("us", "TNX", 4000)
    except Exception as exc:  # noqa: BLE001
        print("sin TNX:", exc)
        tnx = None
    rows = []
    grid = [("logit", n) for n in SETS] + [("hgb", n) for n in SETS] + [("svm", "F1 +EDA"), ("svm", "F3 +mercado")]
    for h in HORIZONS:
        p = build_panel(hist, tnx, h)
        months = p.index.to_period("M").astype(str).to_numpy()
        test = (p.index >= FIRST_TEST)
        for target in ("up", "beat"):
            for kind, set_name in grid:
                if target == "beat" and kind == "svm":
                    continue
                t0 = time.time()
                sc = walk_forward_scores(p, SETS[set_name], kind, target, h).to_numpy()
                m = test & ~np.isnan(sc)
                y, s = p[target].to_numpy()[m], sc[m]
                a, lo, hi = auc_ci(y, s, months[m])
                row = {"horizonte": h, "objetivo": "sube" if target == "up" else "le gana al mercado",
                       "modelo": kind, "features": set_name, "n": int(m.sum()), "auc": a, "auc_lo": lo, "auc_hi": hi}
                # sobreventa: ¿ordena bien dentro de los dias de RSI<=30?
                ov = m & (p["rsi14"].to_numpy() <= 30)
                if ov.sum() > 100 and len(np.unique(p[target].to_numpy()[ov])) > 1:
                    row["auc_sobreventa"] = roc_auc_score(p[target].to_numpy()[ov], sc[ov])
                    row["n_sobreventa"] = int(ov.sum())
                # tasa de acierto de las predicciones mas confiables
                if len(s) > 200:
                    q10, q90 = np.quantile(s, [0.1, 0.9])
                    row["base_%"] = y.mean() * 100
                    row["sube_en_top10_%"] = y[s >= q90].mean() * 100
                    row["sube_en_bottom10_%"] = y[s <= q10].mean() * 100
                rows.append(row)
                print(f"h={h} {target} {kind:5s} {set_name:12s} AUC={a:.3f} [{lo:.3f},{hi:.3f}] ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
