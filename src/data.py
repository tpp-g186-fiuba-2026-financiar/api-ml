"""Obtencion de series historicas OHLCV para un ticker.

Dos fuentes posibles, seleccionables por configuracion (`settings.data_source`):

- ``yahoo``    : se consulta directamente la API publica de Yahoo Finance.
- ``collector``: se consulta el servicio interno ``data-colector``.

Ambas devuelven un ``pandas.DataFrame`` con la misma forma: indice temporal
ascendente y columnas ``open, high, low, close, volume``.
"""

from __future__ import annotations

import certifi
import pandas as pd
import requests

from src.config import settings
from src.errors import DataUnavailableError

_YF_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_REQUEST_TIMEOUT = 20


def to_yahoo_symbol(symbol: str) -> str:
    """Mapea un ticker "limpio" (GGAL) al simbolo de Yahoo (GGAL.BA).

    Si el simbolo ya trae un sufijo (contiene ``.`` o ``=``, p.ej. ``GC=F``,
    ``AAPL``) se respeta tal cual.
    """
    symbol = symbol.strip().upper()
    if "." in symbol or "=" in symbol:
        return symbol
    if not settings.market_suffix:
        return symbol
    return f"{symbol}{settings.market_suffix}"


def fetch_history_yahoo(symbol: str, days: int | None = None) -> pd.DataFrame:
    """Descarga el historico diario desde la API de Yahoo Finance."""
    days = days or settings.history_days
    yf_symbol = to_yahoo_symbol(symbol)
    # ``range`` admite valores tipo "2y"; pedimos con margen y luego recortamos.
    range_param = "max" if days > 1825 else f"{max(days // 252 + 1, 1)}y"
    params = {"range": range_param, "interval": "1d", "includeAdjustedClose": "true"}
    headers = {"User-Agent": "Mozilla/5.0 (financiar-api-ml)"}

    try:
        resp = requests.get(
            _YF_CHART_URL.format(symbol=yf_symbol),
            params=params,
            headers=headers,
            verify=certifi.where(),
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:  # ValueError = JSON invalido
        raise DataUnavailableError(
            f"no se pudo obtener data de Yahoo para {yf_symbol}: {exc}"
        ) from exc

    result = (payload.get("chart") or {}).get("result")
    if not result:
        raise DataUnavailableError(f"Yahoo no devolvio datos para {yf_symbol}")

    node = result[0]
    timestamps = node.get("timestamp")
    quote = ((node.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps or not quote:
        raise DataUnavailableError(f"respuesta de Yahoo sin candles para {yf_symbol}")

    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s"),
    )
    return _clean(frame, days)


def fetch_history_collector(symbol: str, days: int | None = None) -> pd.DataFrame:
    """Descarga el historico desde el servicio ``data-colector``."""
    days = days or settings.history_days
    base = settings.data_collector_url
    if not base:
        raise DataUnavailableError("data_collector_url no esta configurada")

    url = f"{base.rstrip('/')}/historical-data/{symbol.strip().upper()}"
    try:
        resp = requests.post(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataUnavailableError(
            f"no se pudo obtener data del collector para {symbol}: {exc}"
        ) from exc

    rows = payload.get("data") or []
    if not rows:
        raise DataUnavailableError(f"el collector no devolvio candles para {symbol}")

    frame = pd.DataFrame(
        {
            "open": [float(r["open_amount"]) for r in rows],
            "high": [float(r["high_amount"]) for r in rows],
            "low": [float(r["low_amount"]) for r in rows],
            "close": [float(r["close_amount"]) for r in rows],
            "volume": [float(r["volume"]) for r in rows],
        },
        index=pd.to_datetime([int(r["ts"]) for r in rows], unit="ms"),
    )
    return _clean(frame, days)


def fetch_history(symbol: str, days: int | None = None) -> pd.DataFrame:
    """Despacha al backend configurado en ``settings.data_source``."""
    if settings.data_source == "collector":
        return fetch_history_collector(symbol, days)
    return fetch_history_yahoo(symbol, days)


def fetch_available_tickers() -> list[str]:
    """Todos los tickers cacheados en ``data-colector`` (BYMA + CEDEARs + commodities + ETFs).

    Se usa para entrenar los modelos de tendencia con todo lo disponible en
    vez de una lista fija hardcodeada. ``data-colector`` no distingue el
    mercado en la respuesta de este endpoint, asi que no se filtra nada aca.
    """
    base = settings.data_collector_url
    if not base:
        raise DataUnavailableError("data_collector_url no esta configurada")

    url = f"{base.rstrip('/')}/available-tickers"
    try:
        resp = requests.post(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataUnavailableError(f"no se pudo obtener el listado de tickers: {exc}") from exc

    tickers = ((payload.get("message") or {}).get("tickers")) or []
    if not tickers:
        raise DataUnavailableError("data-colector no devolvio ningun ticker disponible")
    return list(tickers)


def _clean(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    frame = frame[_OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if days and len(frame) > days:
        frame = frame.iloc[-days:]
    return frame
