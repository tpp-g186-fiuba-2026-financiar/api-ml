"""Series macro de Argentina servidas por el data-colector, como tabla diaria."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from src.config import settings
from src.errors import DataUnavailableError

_REQUEST_TIMEOUT = 60

# nombre -> (ruta en el data-colector, serie, retraso en dias calendario)
SERIES: dict[str, tuple[str, str, int]] = {
    "ccl": ("macro/argdatos", "CCL", 2),
    "mep": ("macro/argdatos", "MEP", 2),
    "oficial": ("macro/argdatos", "OFICIAL", 2),
    "mayorista": ("macro/argdatos", "MAYORISTA", 2),
    "blue": ("macro/argdatos", "BLUE", 2),
    "riesgo_pais": ("macro/argdatos", "RIESGO_PAIS", 2),
    "reservas": ("interest-rate/ar", "RESERVAS", 5),
    "badlar": ("interest-rate/ar", "BADLAR", 5),
    "base_monetaria": ("interest-rate/ar", "BASE_MONETARIA", 5),
}
REQUIRED = ("ccl", "oficial", "riesgo_pais", "badlar")


def _base_url() -> str:
    base = settings.macro_collector_url or settings.data_collector_url
    if not base:
        raise DataUnavailableError("data_collector_url no esta configurada")
    return base.rstrip("/")


def fetch_collector_series(route: str, series: str) -> pd.Series:
    """Serie diaria del data-colector."""
    url = f"{_base_url()}/{route}/{series}"
    try:
        resp = requests.post(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataUnavailableError(f"no se pudo obtener {route}/{series}: {exc}") from exc

    # el data-colector responde 200 y manda el error en el body
    if payload.get("status") != 200:
        reason = (payload.get("message") or {}).get("error", "error desconocido")
        raise DataUnavailableError(f"data-colector no pudo obtener {route}/{series}: {reason}")
    rows = payload.get("data") or []
    if not rows:
        raise DataUnavailableError(f"data-colector no devolvio datos para {route}/{series}")

    values = pd.Series(
        [float(r["value"]) for r in rows],
        index=pd.to_datetime([int(r["ts"]) for r in rows], unit="ms").normalize(),
    )
    return values[~values.index.duplicated(keep="last")].sort_index()


def build_macro_frame(raw: dict[str, pd.Series], end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Tabla diaria con relleno hacia adelante y el retraso de cada serie."""
    if not raw:
        raise DataUnavailableError("no hay series macro disponibles")
    last = max(s.index[-1] for s in raw.values())
    end = max(end, last) if end is not None else last
    start = min(s.index[0] for s in raw.values())
    index = pd.date_range(start, end, freq="D")
    frame = pd.DataFrame(index=index)
    for name, series in raw.items():
        lag = SERIES[name][2]
        daily = series.reindex(index.union(series.index)).ffill().reindex(index)
        frame[name] = daily.shift(lag)
    return frame


def fetch_argentina_macro() -> pd.DataFrame:
    """Descarga las series; falla si falta alguna de ``REQUIRED``."""
    raw: dict[str, pd.Series] = {}
    errors: list[str] = []
    for name, (route, series, _lag) in SERIES.items():
        try:
            raw[name] = fetch_collector_series(route, series)
        except DataUnavailableError as exc:
            errors.append(str(exc))
    missing = [n for n in REQUIRED if n not in raw]
    if missing:
        raise DataUnavailableError(
            f"faltan series macro imprescindibles {missing}: {'; '.join(errors)}"
        )
    for err in errors:
        print(f"  [warn] {err}")
    return build_macro_frame(raw, end=pd.Timestamp(dt.date.today()))


_CACHE: dict[str, pd.DataFrame] = {}


def todays_macro_frame() -> pd.DataFrame:
    """Macro de hoy, una descarga por dia y por proceso."""
    key = dt.date.today().isoformat()
    if key not in _CACHE:
        frame = fetch_argentina_macro()
        _CACHE.clear()
        _CACHE[key] = frame
    return _CACHE[key]
