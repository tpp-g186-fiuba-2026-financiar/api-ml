"""Control de series de precios sin movimiento (ej. A3)."""

from __future__ import annotations

import pandas as pd

STALE_MAX_UNCHANGED = 0.30


def is_stale(df: pd.DataFrame) -> bool:
    """True si el precio no cambia en mas del 30% de las ruedas."""
    if len(df) < 2:
        return True
    unchanged = (df["close"].diff().dropna() == 0).mean()
    return bool(unchanged > STALE_MAX_UNCHANGED)


def drop_stale(histories: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    kept = {}
    for symbol, df in histories.items():
        if is_stale(df):
            print(f"  [skip] {symbol}: serie estancada (>30% de ruedas sin cambio de precio)")
            continue
        kept[symbol] = df
    return kept
