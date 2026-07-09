"""Cliente HTTP generico para invocar modelos deployados en Modal.

Las URLs de cada modelo se configuran en ``src.config.settings`` (una por
modelo, ninguna por default). Modal reentrena al vuelo en cada request
(ver `models/modelos/*.py`), asi que el timeout es mas generoso que el de
``src.data.fetch_history``.
"""

from __future__ import annotations

import requests

from src.errors import DataUnavailableError

_TIMEOUT = 90  # el LSTM en Modal reentrena al vuelo, puede tardar varios segundos


def call_modal(base_url: str, ticker: str, **extra_params: str | int) -> dict:
    params = {"ticker": ticker, **extra_params}
    try:
        resp = requests.get(base_url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataUnavailableError(f"no se pudo conectar con Modal ({base_url}): {exc}") from exc
