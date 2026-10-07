"""Valida el modelo "macro" TAL COMO ESTA EN PRODUCCION (src.macro_trend) con walk-forward.

Reentrena cada 126 ruedas desde 2020 con datos hasta (inicio del tramo - horizonte), usando
``MacroTrendModel.fit_panel`` (lo mismo que hace produccion), y mide fuera de muestra:
AUC con intervalo (bloques por mes), AUC por anio, y cuanto acierta cada senal ("alza" / "baja").

Uso (desde api-ml/, con el data-colector sirviendo las series macro):
    DATA_SOURCE=collector MACRO_COLLECTOR_URL=http://localhost:3055 \
        .venv/bin/python notebooks/investigacion_macro/validar_modelo_macro.py [horizonte]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import fetch_history  # noqa: E402
from src.macro_data import fetch_argentina_macro  # noqa: E402
from src.macro_trend import FEATURES, MIN_ROWS, UNIVERSE, MacroConfig, MacroTrendModel, _score  # noqa: E402

warnings.filterwarnings("ignore")
REFIT = 126


def block_ci(values_fn, months: np.ndarray, reps: int = 200) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    um = np.unique(months)
    idx = {m: np.where(months == m)[0] for m in um}
    vals = [
        values_fn(np.concatenate([idx[m] for m in rng.choice(um, len(um), replace=True)]))
        for _ in range(reps)
    ]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    histories = {s: fetch_history(s, 4000) for s in UNIVERSE}
    macro = fetch_argentina_macro()
    model = MacroTrendModel(config=MacroConfig(horizon=horizon))
    panel = model._panel({s: d for s, d in histories.items() if len(d) >= MIN_ROWS}, macro)
    panel = panel[panel.index >= "2017-01-02"]
    dates = panel.index.values
    X, y = panel[FEATURES].to_numpy(), panel["down"].to_numpy()
    days = np.unique(dates)
    starts = days[days >= np.datetime64("2020-01-01")][::REFIT]
    rows = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else None
        cut = days[max(np.searchsorted(days, s) - horizon, 0)]
        tr_all = np.where(dates <= cut)[0]
        te = np.where((dates >= s) & ((dates < e) if e is not None else True))[0]
        if len(tr_all) < 3000 or len(te) == 0:
            continue
        sub = panel.iloc[tr_all]
        m = MacroTrendModel(config=MacroConfig(horizon=horizon))
        m.fit_panel(
            sub
        )  # exactamente lo que hace produccion: calibra fuera de muestra y entrena con todo
        pt = _score(m._models, X[te])
        rows.append(
            pd.DataFrame(
                {
                    "date": dates[te],
                    "p": pt,
                    "down": y[te],
                    "sig": np.where(
                        pt >= m.baja_threshold, -1, np.where(pt <= m.alza_threshold, 1, 0)
                    ),
                }
            )
        )
        print(f"  tramo {k + 1}/{len(starts)}", flush=True)
    r = pd.concat(rows)
    dts = pd.DatetimeIndex(r["date"])
    months = dts.to_period("M").astype(str).to_numpy()
    yy, pp = r["down"].to_numpy(), r["p"].to_numpy()
    lo, hi = block_ci(lambda ix: roc_auc_score(yy[ix], pp[ix]), months)
    print(f"\nHorizonte {horizon} ruedas, {len(r)} predicciones fuera de muestra (2020-2026)")
    print(f"AUC = {roc_auc_score(yy, pp):.3f}  IC95% [{lo:.3f}, {hi:.3f}]  (0.50 = azar)")
    per = {
        int(a): roc_auc_score(yy[dts.year == a], pp[dts.year == a])
        for a in range(2020, 2027)
        if len(np.unique(yy[dts.year == a])) > 1
    }
    print(
        "AUC por anio:",
        {a: round(v, 3) for a, v in per.items()},
        f"-> {sum(v > .5 for v in per.values())}/{len(per)} anios sobre 0.50",
    )
    sig = r["sig"].to_numpy()
    for label, code, hit, base in (
        ("baja", -1, yy == 1, (yy == 1).mean()),
        ("alza", 1, yy == 0, (yy == 0).mean()),
    ):
        m = sig == code
        l2, h2 = block_ci(
            lambda ix, m=m, hit=hit: hit[ix][m[ix]].mean() if m[ix].any() else np.nan, months
        )
        print(
            f"  senal {label}: {m.mean() * 100:4.1f}% de los casos (n={m.sum()}), acierto {hit[m].mean() * 100:5.1f}% "
            f"IC95% [{l2 * 100:.1f}, {h2 * 100:.1f}]  vs base {base * 100:.1f}%"
        )
    print(f"  neutral: {(sig == 0).mean() * 100:4.1f}% de los casos")


if __name__ == "__main__":
    main()
