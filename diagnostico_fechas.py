"""Corre esto en tu entorno real (donde fetch_available_tickers/fetch_history
si funcionan) para ver, por ticker, cuantos records tiene y su rango de
fechas -- antes de asumir cual es el problema.
"""
from src.config import settings
from src.data import fetch_available_tickers, fetch_history

tickers = fetch_available_tickers()
print(f"Cantidad de tickers de fetch_available_tickers(): {len(tickers)}")
print(tickers)
print()

historiales = {}
for ticker in tickers:
    try:
        df = fetch_history(ticker, settings.history_days)
        historiales[ticker] = df
        print(
            f"{ticker:10s}  {len(df):4d} filas   "
            f"{df.index.min()}  ->  {df.index.max()}"
        )
    except Exception as exc:
        print(f"{ticker:10s}  ERROR: {exc}")

print()
print("Interseccion de FECHAS COMPLETAS (con hora):")
fechas_comunes = None
for ticker, df in historiales.items():
    fechas = set(df.index)
    fechas_comunes = fechas if fechas_comunes is None else (fechas_comunes & fechas)
print(f"  {len(fechas_comunes)} fechas en comun")

print()
print("Interseccion normalizando a SOLO FECHA (sin hora):")
fechas_comunes_norm = None
for ticker, df in historiales.items():
    fechas = set(df.index.normalize())
    fechas_comunes_norm = fechas if fechas_comunes_norm is None else (fechas_comunes_norm & fechas)
print(f"  {len(fechas_comunes_norm)} fechas en comun")