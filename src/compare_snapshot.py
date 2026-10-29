"""Snapshot diario de ``TrendRegistry.compare()``.

El paper trading diario (``.github/workflows/paper-trading.yml``) ya corre
todos los modelos sobre cada accion despues del cierre. Guarda ademas la
comparacion completa (prediccion, RSI, backtest de cada modelo) en
``models/compare_snapshot.json``, y el comparador la sirve desde ahi en vez
de correr todos los modelos en vivo al abrir la pantalla.

Las velas son diarias: lo calculado despues del cierre sigue valiendo hasta
la corrida del dia habil siguiente. Si el job no corrio, el snapshot queda
vencido y el comparador vuelve a calcular en vivo.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

DEFAULT_SNAPSHOT_PATH = "models/compare_snapshot.json"
RUN_HOUR_UTC = 21
# El cron de GitHub puede arrancar tarde: hasta este margen despues de la hora
# programada, el snapshot de la corrida anterior sigue valiendo.
RUN_GRACE = dt.timedelta(hours=3)


def last_scheduled_run(now: dt.datetime) -> dt.datetime:
    """Ultima corrida programada (lunes a viernes, 21:00 UTC) en o antes de ``now``."""
    candidate = now.replace(hour=RUN_HOUR_UTC, minute=0, second=0, microsecond=0)
    if candidate > now:
        candidate -= dt.timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= dt.timedelta(days=1)
    return candidate


def is_fresh(generated_at: dt.datetime, now: dt.datetime | None = None) -> bool:
    """True si despues de ``generated_at`` no paso ninguna corrida programada."""
    now = now or dt.datetime.now(dt.UTC)
    return generated_at >= last_scheduled_run(now - RUN_GRACE)


REMOTE_TTL = dt.timedelta(minutes=10)
_remote_cache: dict[str, tuple[dt.datetime, dict | None]] = {}


def fetch_remote_json(url: str, timeout: float = 10.0) -> dict | None:
    """JSON de ``url`` guardado en memoria ``REMOTE_TTL`` (tambien si fallo, para
    no reintentar en cada pedido). None si no se pudo bajar o no es JSON."""
    import requests

    now = dt.datetime.now(dt.UTC)
    cached = _remote_cache.get(url)
    if cached and now - cached[0] < REMOTE_TTL:
        return cached[1]
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        data = None
    _remote_cache[url] = (now, data if isinstance(data, dict) else None)
    return _remote_cache[url][1]


def parse_snapshot(raw: dict | None) -> dict | None:
    """``{"generated_at": datetime, "symbols": {...}}`` o None si no es valido."""
    try:
        generated_at = dt.datetime.strptime(raw["generated_at"], "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, KeyError):
        return None
    symbols = raw.get("symbols")
    if not isinstance(symbols, dict):
        return None
    return {"generated_at": generated_at.replace(tzinfo=dt.UTC), "symbols": symbols}


def load_snapshot(path: str | Path) -> dict | None:
    """``{"generated_at": datetime, "symbols": {...}}`` o None si no hay o esta roto."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return parse_snapshot(raw)


def save_snapshot(
    path: str | Path, symbols: dict[str, dict], now: dt.datetime | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = now or dt.datetime.now(dt.UTC)
    payload = {"generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "symbols": symbols}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=_json_default))


def _json_default(value):
    """numpy y fechas que puedan quedar en la respuesta de algun modelo."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"no serializable: {type(value).__name__}")
