"""Evaluacion de la vara a superar: azar, "siempre sube", regla de RSI y
regresion logistica, con Correct Up / Correct Down (Cao y Tay) separados.

Por que existe: la accuracy direccional sola engana. Como las acciones
argentinas suben mas dias de los que bajan (y el peso se devalua), un
"modelo" que siempre dice "sube" ya acierta ~55% a 5 dias. Un modelo solo
aporta algo si supera esa vara, no el 50%.

Metricas (por metodo y horizonte, sobre casos NO solapados por ticker):
- acierto: de las veces que dio senal (sube/baja), cuantas acerto.
- cobertura: % de dias en que dio senal (el resto es "neutral"/abstencion).
- acierto en alzas (Correct Up): de los dias que realmente subio, en cuantos
  dijo "sube". acierto en bajas (Correct Down): idem con bajas.
- Se calcula sobre todos los dias y sobre los dias de sobreventa (RSI14 <= 30),
  que es donde el EDA/experimento encontro senal.

Walk-forward honesto: la logistica se reentrena cada ~63 ruedas usando solo
datos hasta (inicio del tramo - horizonte), sin mirar el futuro.

Uso (desde api-ml/):
    .venv/bin/python notebooks/evaluacion_baselines.py [cache.pkl]
Si se pasa un .pkl con {ticker: DataFrame} se usa; si no, baja del data-colector.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.indicators import macd_histogram, rsi_series, sma  # noqa: E402

UNIVERSE = [
    "CRES", "ALUA", "TECO2", "BBAR", "METR", "SUPV", "BYMA", "PAMP", "COME", "CEPU",
    "EDN", "TRAN", "GGAL", "TXAR", "VALO", "TGSU2", "TGNO4", "YPFD", "BMA", "LOMA",
]
# A3 queda afuera: su precio no cambia en ~91% de los dias (datos estancados), lo que
# dispara RSI=100 ("sobrecompra") sin movimiento real y ensucia cualquier medicion.
HORIZONS = [5, 10, 20]
START = "2017-01-01"
FIRST_TEST = "2020-01-01"  # al menos ~3 anios de entrenamiento
REFIT_EVERY = 63
BASE_FEATURES = ["log_return", "log_volume_change", "range_pct", "rsi_norm",
                 "sma20_ratio", "macd_norm", "momentum_10"]
EXTRA_FEATURES = ["dist_max60", "rsi_vol_interaction", "boll_z"]


def load_histories(cache: str | None) -> dict[str, pd.DataFrame]:
    if cache and Path(cache).exists():
        return pickle.load(open(cache, "rb"))
    from src.data import fetch_history_collector

    return {t: fetch_history_collector(t, days=4000) for t in UNIVERSE}


def features_for(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    f = pd.DataFrame(index=df.index)
    lc = np.log(close)
    f["log_return"] = pd.Series(lc, index=df.index).diff()
    f["log_volume_change"] = np.log(pd.Series(np.where(vol <= 0, np.nan, vol), index=df.index)).diff()
    f["range_pct"] = (df["high"] - df["low"]) / df["close"]
    f["rsi14"] = rsi_series(close)
    f["rsi_norm"] = (f["rsi14"] - 50.0) / 50.0
    f["sma20_ratio"] = close / sma(close, 20) - 1.0
    f["macd_norm"] = macd_histogram(close) / close
    f["momentum_10"] = pd.Series(lc, index=df.index).diff(10)
    f["dist_max60"] = df["close"] / df["close"].rolling(60).max() - 1.0
    vz = (np.log1p(df["volume"]) - np.log1p(df["volume"]).rolling(60).mean()) / np.log1p(df["volume"]).rolling(60).std()
    f["rsi_vol_interaction"] = f["rsi_norm"] * vz.fillna(0.0)
    f["boll_z"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
    return f.replace([np.inf, -np.inf], np.nan)


def build_panel(hist: dict[str, pd.DataFrame], h: int) -> pd.DataFrame:
    rows = []
    for t in UNIVERSE:
        if t not in hist:
            continue
        df = hist[t][hist[t].index >= START]
        f = features_for(df)
        f["fwd"] = df["close"].shift(-h) / df["close"] - 1.0
        f["ticker"] = t
        f["pos"] = np.arange(len(f))
        rows.append(f)
    p = pd.concat(rows)
    p = p.dropna(subset=["fwd", "rsi14"])
    p = p[p["fwd"] != 0]  # precio sin cambio: no hay direccion que acertar
    p["up"] = (p["fwd"] > 0).astype(int)
    p[BASE_FEATURES + EXTRA_FEATURES] = p[BASE_FEATURES + EXTRA_FEATURES].fillna(0.0)
    return p


def logistic_predictions(p: pd.DataFrame, feats: list[str], h: int, band: float = 0.0) -> pd.Series:
    """Prediccion walk-forward: +1 sube, -1 baja, 0 sin senal."""
    pred = pd.Series(np.nan, index=range(len(p)))
    p = p.reset_index().rename(columns={"index": "date"})
    dates = p["date"]
    test_start = pd.Timestamp(FIRST_TEST)
    all_days = np.sort(dates.unique())
    fold_starts = [d for i, d in enumerate(all_days) if d >= test_start and (i % REFIT_EVERY == 0 or d == all_days[all_days >= test_start][0])]
    fold_starts = list(all_days[all_days >= test_start][::REFIT_EVERY])
    for k, s in enumerate(fold_starts):
        e = fold_starts[k + 1] if k + 1 < len(fold_starts) else None
        cutoff_idx = np.searchsorted(all_days, s) - h  # purga: el target mira h dias adelante
        cutoff = all_days[max(cutoff_idx, 0)]
        tr = p[p["date"] <= cutoff]
        te = p[(p["date"] >= s) & ((p["date"] < e) if e is not None else True)]
        if len(tr) < 500 or te.empty:
            continue
        sc = StandardScaler().fit(tr[feats])
        xtr = np.clip(sc.transform(tr[feats]), -5, 5)
        xte = np.clip(sc.transform(te[feats]), -5, 5)
        clf = LogisticRegression(C=0.5, max_iter=500).fit(xtr, tr["up"])
        prob = clf.predict_proba(xte)[:, 1]
        sig = np.where(prob > 0.5 + band, 1, np.where(prob < 0.5 - band, -1, 0))
        pred.loc[te.index] = sig
    return pred


def metrics(sig: pd.Series, up: pd.Series) -> dict:
    sig = sig.dropna()
    up = up.loc[sig.index]
    n = len(sig)
    if n == 0:
        return {}
    called = sig != 0
    hit = ((sig == 1) & (up == 1)) | ((sig == -1) & (up == 0))
    return {
        "n": n,
        "cobertura": called.mean() * 100,
        "acierto": hit[called].mean() * 100 if called.any() else np.nan,
        "acierto_alzas": ((sig == 1) & (up == 1)).sum() / max((up == 1).sum(), 1) * 100,
        "acierto_bajas": ((sig == -1) & (up == 0)).sum() / max((up == 0).sum(), 1) * 100,
        "alzas_reales": up.mean() * 100,
    }


def dedupe_events(ev: pd.DataFrame, h: int) -> list:
    """Indices de eventos separados >= h ruedas dentro de cada ticker (sin solapar)."""
    keep = []
    for _, g in ev.sort_values("pos").groupby("ticker"):
        last = -10**9
        for idx, pos in zip(g.index, g["pos"]):
            if pos - last >= h:
                keep.append(idx)
                last = pos
    return keep


def nonoverlap_mask(p: pd.DataFrame, h: int) -> pd.Series:
    return (p["pos"] % h == 0)


def main() -> None:
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    hist = load_histories(cache)
    out = []
    for h in HORIZONS:
        p = build_panel(hist, h)
        p = p[p.index >= FIRST_TEST]
        full = build_panel(hist, h)
        methods = {
            "Azar (50%)": None,
            "Siempre sube": pd.Series(1, index=range(len(full))),
            "Regla RSI completa (<=30 sube, >=70 baja)": pd.Series(
                np.where(full["rsi14"] <= 30, 1, np.where(full["rsi14"] >= 70, -1, 0)), index=range(len(full))
            ),
            "Regla RSI solo sobreventa (<=30 sube)": pd.Series(
                np.where(full["rsi14"] <= 30, 1, 0), index=range(len(full))
            ),
            "Regla RSI solo sobrecompra (>=70 baja)": pd.Series(
                np.where(full["rsi14"] >= 70, -1, 0), index=range(len(full))
            ),
            "Logistica (7 features actuales)": logistic_predictions(full, BASE_FEATURES, h),
            "Logistica (+3 features nuevas)": logistic_predictions(full, BASE_FEATURES + EXTRA_FEATURES, h),
        }
        full = full.reset_index().rename(columns={"index": "date"})
        full["date"] = pd.to_datetime(full["date"])
        in_test = (full["date"] >= FIRST_TEST) & nonoverlap_mask(full, h)
        sv_events = dedupe_events(full[(full["date"] >= FIRST_TEST) & (full["rsi14"] <= 30)], h)
        for name, sig in methods.items():
            for slice_name, mask in [("todos los dias", in_test), ("dias de sobreventa", full.index.isin(sv_events))]:
                if sig is None:
                    n = int(mask.sum()); up = full.loc[mask, "up"]
                    m = {"n": n, "cobertura": 100.0, "acierto": 50.0, "acierto_alzas": 50.0,
                         "acierto_bajas": 50.0, "alzas_reales": up.mean() * 100}
                else:
                    s = pd.Series(sig.to_numpy(), index=full.index)[mask]
                    m = metrics(s, full.loc[mask, "up"])
                if m:
                    out.append({"horizonte": h, "metodo": name, "muestra": slice_name, **m})
    df = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    for h in HORIZONS:
        for sl in ["todos los dias", "dias de sobreventa"]:
            d = df[(df.horizonte == h) & (df.muestra == sl)].drop(columns=["horizonte", "muestra"])
            print(f"\n### Horizonte {h} dias — {sl}")
            print(d.round(1).to_string(index=False))
    ledger_path = Path(__file__).resolve().parents[1] / "models" / "paper_trading_ledger.json"
    if ledger_path.exists():
        import json

        res = pd.DataFrame(json.load(open(ledger_path))["resolved"])
        res["up"] = (res["realized_log_return"] > 0).astype(int)
        res["sig"] = res["signal"].map({"alza": 1, "baja": -1}).fillna(0).astype(int)
        print("\n### Modelos desplegados en paper trading (fuera de muestra real, horizonte 5)")
        rows = []
        for model, g in res.groupby("model"):
            m = metrics(g["sig"], g["up"])
            rows.append({"modelo": model, **m})
        print(pd.DataFrame(rows).round(1).to_string(index=False))
    df.to_csv(Path(__file__).with_name("evaluacion_baselines.csv"), index=False)


if __name__ == "__main__":
    main()
