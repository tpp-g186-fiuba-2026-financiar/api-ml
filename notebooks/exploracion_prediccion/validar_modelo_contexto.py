"""Validacion walk-forward del modelo "contexto" con el codigo de produccion.

Por cada anio de prueba se arma un modelo nuevo SOLO con datos anteriores
(purga del horizonte), con la misma calibracion de umbrales fuera de muestra que
usa el entrenamiento real, y se mide cuanto aciertan sus senales en ese anio
frente a la tasa base del mismo anio. Se corre tambien sin las variables de
contexto para aislar cuanto aporta cada parte.

    python notebooks/exploracion_prediccion/validar_modelo_contexto.py [--from 2016]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from src import context_trend as ct  # noqa: E402
from src.data import fetch_history  # noqa: E402
from src.macro_trend import UNIVERSE  # noqa: E402


def one_year(year, panel, features):
    ct.FEATURES = features
    cutoff = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=int(ct.HORIZON * 1.6) + 2)
    train = panel[panel.index < cutoff]
    test = panel[(panel.index.year == year)]
    if len(train) < 6000 or test.empty:
        return None
    model = ct.ContextTrendModel()
    model.fit_panel(train)
    p = ct._score(model._models, test[features].to_numpy())
    out = test[["ticker", "fwd", "down"]].copy()
    out["p"] = p
    out["baja_thr"], out["alza_thr"] = model.baja_threshold, model.alza_threshold
    return out


def summarize(name, o):
    o = o.assign(
        m=o.index.to_period("M"), ybase_down=o.groupby(o.index.year).down.transform("mean")
    )
    auc = roc_auc_score(o.down, o.p)
    rng = np.random.default_rng(0)
    months = list(o.m.unique())
    by_m = {k: g for k, g in o.groupby("m")}

    def boot(sel_fn, base_fn):
        diffs = []
        for _ in range(300):
            s = pd.concat([by_m[k] for k in rng.choice(months, len(months))])
            sel = sel_fn(s)
            diffs.append((sel[0] - base_fn(s, sel[1])) if len(sel[1]) else np.nan)
        return np.nanpercentile(diffs, [2.5, 97.5])

    lines = [f"{name}: AUC {auc:.3f}"]
    for label, mask_fn, hit_fn, base_fn in (
        ("ALZA", lambda d: d.p <= d.alza_thr, lambda d: 1 - d.down, lambda d: 1 - d.ybase_down),
        ("BAJA", lambda d: d.p >= d.baja_thr, lambda d: d.down, lambda d: d.ybase_down),
    ):
        sel = o[mask_fn(o)]
        hit, base = hit_fn(sel).mean(), base_fn(sel).mean()

        def sel_fn(s, mask_fn=mask_fn, hit_fn=hit_fn):
            chosen = s[mask_fn(s)]
            return hit_fn(chosen).mean(), chosen

        def b_fn(s, chosen, base_fn=base_fn):
            return base_fn(chosen).mean()

        lo, hi = boot(sel_fn, b_fn)
        lines.append(
            f"   {label}: {len(sel)} senales ({len(sel) / len(o):.0%}), acierta {hit:.1%} vs base "
            f"{base:.1%} -> ventaja {100 * (hit - base):+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        )
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", type=int, default=2016)
    parser.add_argument("--days", type=int, default=4000)
    args = parser.parse_args()

    print("Descargando historicos...")
    histories = {s: fetch_history(s, args.days) for s in UNIVERSE}
    market = ct.market_frame(histories)
    histories = {s: d for s, d in histories.items() if len(d) >= ct.MIN_ROWS}
    panel = ct.ContextTrendModel()._panel(histories, market)
    years = list(range(args.start, int(panel.index.max().year) + 1))

    for name, feats in (
        ("SIN contexto (solo tecnicas)", ct.TECH_FEATURES),
        ("CON contexto", ct.FEATURES),
    ):
        outs = Parallel(n_jobs=-1)(delayed(one_year)(y, panel, feats) for y in years)
        o = pd.concat([x for x in outs if x is not None])
        summarize(name, o)
        recent = o[o.index.year >= max(args.start, years[-1] - 2)]
        summarize(name + f" [{years[-1] - 2}+]", recent)


if __name__ == "__main__":
    main()
