"""Obtencion de series historicas OHLCV para un ticker.

Dos fuentes posibles, seleccionables por configuracion (`settings.data_source`):

- ``yahoo``    : se consulta directamente la API publica de Yahoo Finance.
- ``collector``: se consulta el servicio interno ``data-colector``.

Ambas devuelven un ``pandas.DataFrame`` con la misma forma: indice temporal
ascendente y columnas ``open, high, low, close, volume``.

Tambien vive aca ``fetch_macro_series``/``attach_macro_feature``: la serie
macro (tasa de interes) exogena que se suma como feature en los modelos de
tendencia (ver ``FEATURE_NAMES`` en ``src.lstm``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import certifi
import numpy as np
import pandas as pd
import requests

from src.config import settings
from src.errors import ApiMlError, DataUnavailableError

_YF_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_REQUEST_TIMEOUT = 20
_AVAILABLE_TICKERS_ATTEMPTS = 3
# (conexion, lectura) contra el data-colector. Si responde, contesta en 1-3 s;
# esperar mas solo alarga la caida.
COLLECTOR_TIMEOUT = (5, 15)
# Despues de un timeout / 5xx del data-colector no se le vuelve a pegar por
# este tiempo: cada pedido falla en el acto (y el historico sale de Yahoo) en
# vez de esperar el timeout por cada ticker. XGBoost, Contexto y el
# Transformer de panel bajan ~20 acciones cada uno; con el colector colgado
# eso eran minutos por pedido.
COLLECTOR_RETRY_AFTER = 5 * 60
# Las velas son diarias: un historico bajado sirve por este tiempo. Sin cache,
# cada modelo (y cada pedido) volvia a bajar las mismas series.
HISTORY_TTL = 30 * 60

# Commodities que `data-colector` expone como tickers (`available-tickers`),
# pero que no son acciones de BYMA -- no llevan sufijo `.BA`, van directo a su
# futuro de Yahoo. Mismo mapeo que usa el scrapper de commodities del repo
# `data-colector` (`commodities_persist_historical_price.rs::COMMODITIES`).
COMMODITY_YAHOO_SYMBOLS = {
    "GOLD": "GC=F",
    "OIL": "CL=F",
}


def to_yahoo_symbol(symbol: str) -> str:
    """Mapea un ticker "limpio" (GGAL) al simbolo de Yahoo (GGAL.BA).

    Si el simbolo ya trae un sufijo (contiene ``.`` o ``=``, p.ej. ``GC=F``,
    ``AAPL``) se respeta tal cual. Los commodities (``GOLD``, ``OIL``) se
    resuelven antes que nada a su futuro de Yahoo -- si no, caian en la regla
    de abajo y quedaban con el sufijo `.BA` agregado por error (Yahoo
    respondia 404 para ``GOLD.BA``/``OIL.BA``, que no existen).
    """
    symbol = symbol.strip().upper()
    if symbol in COMMODITY_YAHOO_SYMBOLS:
        return COMMODITY_YAHOO_SYMBOLS[symbol]
    if "." in symbol or "=" in symbol:
        return symbol
    if not settings.market_suffix:
        return symbol
    return f"{symbol}{settings.market_suffix}"


def fetch_history_yahoo(symbol: str, days: int | None = None) -> pd.DataFrame:
    """Descarga el historico diario desde la API de Yahoo Finance."""
    days = days or settings.history_days
    yf_symbol = to_yahoo_symbol(symbol)
    # Fechas explicitas en vez de ``range``: con ``range=max`` (o rangos muy largos)
    # Yahoo ignora ``interval=1d`` y devuelve velas MENSUALES (~320 filas en 25
    # anios), lo que dejaba entrenando con 4000 "ruedas" a modelos que en
    # realidad recibian meses. Se pide con margen de calendario y luego se recorta.
    now = int(time.time())
    period1 = now - int(days * 1.6 * 86400) - 30 * 86400
    params = {
        "period1": period1,
        "period2": now + 86400,
        "interval": "1d",
        "includeAdjustedClose": "true",
    }
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
    return _clean(_adjust_for_corporate_actions(frame, node), days)


def _adjust_for_corporate_actions(frame: pd.DataFrame, node: dict) -> pd.DataFrame:
    """Lleva OHLC a precios ajustados por dividendos y splits.

    Sin ajustar, un split o un dividendo aparecen como una caida de un dia que
    nunca existio, y contaminan retornos, indicadores y el target. Yahoo manda
    ``adjclose``: se aplica el mismo factor (ajustado / cierre) a apertura,
    maximo, minimo y cierre. Si no viene, se deja la serie como esta.
    """
    adj = ((node.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    if not adj or len(adj) != len(frame):
        return frame
    adjusted = pd.Series(adj, index=frame.index, dtype="float64")
    factor = adjusted / frame["close"].astype("float64")
    factor = factor.where(np.isfinite(factor) & (factor > 0))
    if factor.isna().all():
        return frame
    factor = factor.ffill().bfill()
    out = frame.copy()
    for column in ("open", "high", "low", "close"):
        out[column] = frame[column].astype("float64") * factor
    return out


_collector_down: dict[str, float] = {}
_history_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
_macro_cache: dict[int, tuple[float, pd.Series | None]] = {}
_cache_lock = threading.Lock()


def reset_caches() -> None:
    """Olvida historicos cacheados y caidas del colector (tests)."""
    with _cache_lock:
        _collector_down.clear()
        _history_cache.clear()
        _macro_cache.clear()


def collector_post(url: str, timeout: float | tuple[float, float] = COLLECTOR_TIMEOUT) -> dict:
    """POST al data-colector que corta en el acto si hace poco no respondio.

    Solo un timeout, un error de conexion o un 5xx marcan al colector como
    caido; un 404 o un ticker sin datos no (el colector anda, falta el dato).
    """
    host = url.split("/", 3)[2] if "://" in url else url
    down_since = _collector_down.get(host)
    if down_since is not None and time.monotonic() - down_since < COLLECTOR_RETRY_AFTER:
        raise DataUnavailableError(
            f"data-colector sin respuesta hace poco, no se reintenta ({url})"
        )
    try:
        resp = requests.post(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.ConnectionError, requests.Timeout) as exc:
        _collector_down[host] = time.monotonic()
        raise DataUnavailableError(f"data-colector no responde ({url}): {exc}") from exc
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code >= 500:
            _collector_down[host] = time.monotonic()
        raise DataUnavailableError(f"data-colector respondio con error ({url}): {exc}") from exc
    except (requests.RequestException, ValueError) as exc:  # ValueError = JSON invalido
        raise DataUnavailableError(f"respuesta invalida del data-colector ({url}): {exc}") from exc
    _collector_down.pop(host, None)
    return payload


def _cached_history(key: tuple, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    hit = _history_cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < HISTORY_TTL:
        return hit[1].copy()
    frame = loader()
    with _cache_lock:
        _history_cache[key] = (time.monotonic(), frame)
    return frame.copy()


def fetch_history_collector(symbol: str, days: int | None = None) -> pd.DataFrame:
    """Descarga el historico desde el servicio ``data-colector``."""
    days = days or settings.history_days
    base = settings.data_collector_url
    if not base:
        raise DataUnavailableError("data_collector_url no esta configurada")

    url = f"{base.rstrip('/')}/historical-data/{symbol.strip().upper()}"
    try:
        payload = collector_post(url)
    except DataUnavailableError as exc:
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


def fetch_history_collector_or_yahoo(symbol: str, days: int | None = None) -> pd.DataFrame:
    """Historico del data-colector; si no responde, el de Yahoo (cacheado)."""
    days = days or settings.history_days
    symbol = symbol.strip().upper()

    def load() -> pd.DataFrame:
        try:
            return fetch_history_collector(symbol, days)
        except DataUnavailableError as exc:
            print(f"  [warn] {exc}; se usa Yahoo")
            return fetch_history_yahoo(symbol, days)

    return _cached_history(("collector", symbol, days), load)


def fetch_history(symbol: str, days: int | None = None) -> pd.DataFrame:
    """Despacha al backend configurado en ``settings.data_source``."""
    if settings.data_source == "collector":
        return fetch_history_collector_or_yahoo(symbol, days)
    days = days or settings.history_days
    symbol = symbol.strip().upper()
    return _cached_history(("yahoo", symbol, days), lambda: fetch_history_yahoo(symbol, days))


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
    last_error: Exception | None = None
    for attempt in range(1, _AVAILABLE_TICKERS_ATTEMPTS + 1):
        try:
            resp = requests.post(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < _AVAILABLE_TICKERS_ATTEMPTS:
                print(
                    f"  [warn] listado de tickers: intento {attempt}/"
                    f"{_AVAILABLE_TICKERS_ATTEMPTS} fallo; reintentando"
                )
                time.sleep(attempt * 2)
    else:
        raise DataUnavailableError(
            f"no se pudo obtener el listado de tickers despues de "
            f"{_AVAILABLE_TICKERS_ATTEMPTS} intentos: {last_error}"
        ) from last_error

    tickers = ((payload.get("message") or {}).get("tickers")) or []
    if not tickers:
        raise DataUnavailableError("data-colector no devolvio ningun ticker disponible")
    return list(tickers)


MACRO_COLUMN = "macro_rate"


def fetch_interest_rate(source: str, series: str, days: int | None = None) -> pd.Series:
    """Descarga una serie de tasas de interes desde ``data-colector``.

    Es una serie macro global (``source``/``series``, ej. ``"us"``/``"TNX"``),
    no esta atada a un ticker. ``data-colector`` responde siempre HTTP 200 y
    codifica el error real en el body (``{"status": 500, "message": {...}}``)
    en vez de usar el status code -- por eso se valida ``payload["status"]``
    ademas de ``raise_for_status()``.
    """
    base = settings.data_collector_url
    if not base:
        raise DataUnavailableError("data_collector_url no esta configurada")
    days = days or settings.history_days

    url = f"{base.rstrip('/')}/interest-rate/{source}/{series}"
    try:
        payload = collector_post(url)
    except DataUnavailableError as exc:
        raise DataUnavailableError(f"no se pudo obtener la tasa {source}/{series}: {exc}") from exc

    if payload.get("status") != 200:
        reason = (payload.get("message") or {}).get("error", "error desconocido")
        raise DataUnavailableError(f"data-colector no pudo obtener {source}/{series}: {reason}")

    rows = payload.get("data") or []
    if not rows:
        raise DataUnavailableError(f"data-colector no devolvio datos para {source}/{series}")

    values = pd.Series(
        [float(r["value"]) for r in rows],
        index=pd.to_datetime([int(r["ts"]) for r in rows], unit="ms"),
    )
    values = values[~values.index.duplicated(keep="last")].sort_index()
    if days and len(values) > days:
        values = values.iloc[-days:]
    return values


def fetch_macro_series(days: int | None = None) -> pd.Series | None:
    """Serie macro por defecto (``settings.macro_rate_source``/``macro_rate_series``)
    para enriquecer los features de los modelos de tendencia (ver
    ``attach_macro_feature`` en este modulo y ``FEATURE_NAMES`` en ``src.lstm``).

    Nunca propaga la excepcion: si la fuente externa falla (ya paso con el
    BCRA), se loguea y se sigue sin esta feature en vez de romper el
    entrenamiento/prediccion de todos los tickers por una serie opcional.
    """
    days = days or settings.history_days
    hit = _macro_cache.get(days)
    if hit is not None and time.monotonic() - hit[0] < HISTORY_TTL:
        return hit[1]
    try:
        values = fetch_interest_rate(settings.macro_rate_source, settings.macro_rate_series, days)
    except ApiMlError as exc:
        print(
            f"  [warn] no se pudo obtener la serie macro "
            f"({settings.macro_rate_source}/{settings.macro_rate_series}): {exc}"
        )
        values = None
    # Tambien se cachea la falla: la serie es opcional y no vale esperar al
    # colector en cada prediccion.
    with _cache_lock:
        _macro_cache[days] = (time.monotonic(), values)
    return values


def attach_macro_feature(df: pd.DataFrame, column: str, macro: pd.Series | None) -> pd.DataFrame:
    """Alinea una serie macro al indice de fechas de ``df`` (copia, no muta).

    Usa el ultimo valor conocido a esa fecha o antes (``ffill``) -- nunca uno
    futuro, para no filtrar informacion "del futuro" en el backtest
    walk-forward (``src/backtest.py``). Si ``macro`` es ``None`` (fetch
    fallido) o una fecha de ``df`` es anterior al primer dato macro
    disponible, esa fecha queda en ``NaN``; ``build_features`` lo trata igual
    que el warmup del resto de los indicadores (se limpia a 0).
    """
    df = df.copy()
    if macro is None or macro.empty:
        df[column] = float("nan")
        return df
    df[column] = macro.sort_index().reindex(df.index, method="ffill")
    return df


def _clean(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    frame = frame[_OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if days and len(frame) > days:
        frame = frame.iloc[-days:]
    return frame
