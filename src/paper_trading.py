"""Paper trading: prediccion diaria "hacia adelante" con los modelos ya
entrenados/cargados, guardando un historial persistente de aciertos/errores
por modelo a lo largo del tiempo.

A diferencia de ``src.backtest`` (que recorre el historico completo con un
cutoff movil en una sola corrida), este script esta pensado para correr una
vez por dia (cron / job programado). En cada corrida:

1. Resuelve las predicciones pendientes de corridas anteriores cuyo horizonte
   ya se cumplio, comparandolas contra el cierre real que ya se conoce.
2. Registra una prediccion nueva de "hoy" por cada ticker/modelo cargado --
   a diferencia del backtest, aca si entran los modelos remotos (``*-modal``):
   no hay cutoff historico que respetar, la prediccion de "hoy" es
   exactamente lo que devuelven en vivo.
3. Recalcula el resumen acumulado (directional_accuracy, mae, etc.) por
   modelo con todo lo resuelto hasta el momento.

Todo se persiste en un JSON (default ``models/paper_trading_ledger.json``)
que sirve de estado entre corridas.

Uso:
    python -m src.paper_trading
    python -m src.paper_trading --tickers GGAL YPFD
    python -m src.paper_trading --models lstm xgboost lstm-modal
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import summarize
from src.config import settings
from src.data import fetch_available_tickers, fetch_history
from src.errors import ApiMlError, DataUnavailableError
from src.registry import TrendRegistry, build_registry

DEFAULT_LEDGER_PATH = "models/paper_trading_ledger.json"


def _empty_ledger() -> dict:
    return {"pending": [], "resolved": []}


def load_ledger(path: Path) -> dict:
    if not path.exists():
        return _empty_ledger()
    data = json.loads(path.read_text())
    data.setdefault("pending", [])
    data.setdefault("resolved", [])
    return data


def save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pending": ledger["pending"],
        "resolved": ledger["resolved"],
        "summary": ledger.get("summary", {}),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _pending_key(entry: dict) -> tuple:
    return (entry["symbol"], entry["model"], entry["as_of"])


def select_model_names(registry: TrendRegistry, requested: list[str] | None) -> list[str]:
    """A diferencia del backtest, aca no se excluyen los modelos remotos."""
    if not requested:
        return [n for n in registry.names() if registry.resolve(n).is_loaded]

    selected = []
    for name in requested:
        service = registry.resolve(name)
        if not service.is_loaded:
            print(f"  [skip] '{name}': no esta entrenado")
            continue
        selected.append(name)
    return selected


def resolve_pending(pending: list[dict], days: int) -> tuple[list[dict], list[dict]]:
    """Intenta resolver cada prediccion pendiente contra el cierre real.

    Vuelve a descargar el historico de cada simbolo (una vez por simbolo) y
    ubica el ``as_of`` guardado dentro de esa serie con ``method="pad"``
    (la ultima rueda conocida a esa fecha o antes -- cubre el caso de
    ``arima-modal``, cuyo ``as_of`` es una aproximacion a "hoy", no
    necesariamente una rueda real). Si ya pasaron al menos ``horizon_days``
    ruedas desde ahi, el horizonte se cumplio y se puede comparar contra lo
    que realmente paso.
    """
    still_pending: list[dict] = []
    newly_resolved: list[dict] = []
    histories: dict[str, pd.DataFrame | None] = {}

    for entry in pending:
        symbol = entry["symbol"]
        if symbol not in histories:
            try:
                histories[symbol] = fetch_history(symbol, days)
            except ApiMlError as exc:
                print(f"  [skip] {symbol}: {exc}")
                histories[symbol] = None
        df = histories[symbol]
        if df is None:
            still_pending.append(entry)
            continue

        idx = df.index.get_indexer([pd.Timestamp(entry["as_of"])], method="pad")[0]
        if idx == -1:
            # as_of quedo fuera de la ventana de historia descargada (muy
            # vieja): no se puede resolver nunca, se descarta en vez de
            # quedar pendiente para siempre.
            print(f"  [drop] {symbol}/{entry['model']} as_of={entry['as_of']}: fuera de rango")
            continue

        future_idx = idx + int(entry["horizon_days"])
        if future_idx >= len(df):
            still_pending.append(entry)
            continue

        realized_close = float(df["close"].iloc[future_idx])
        newly_resolved.append(
            {
                **entry,
                "realized_close": realized_close,
                "realized_log_return": float(np.log(realized_close / entry["last_close"])),
                "resolved_at": df.index[future_idx].strftime("%Y-%m-%d"),
            }
        )

    return still_pending, newly_resolved


def record_new_predictions(
    registry: TrendRegistry,
    model_names: list[str],
    tickers: list[str],
    existing_keys: set[tuple],
) -> list[dict]:
    """Corre cada modelo cargado sobre cada ticker y arma las entradas nuevas.

    ``existing_keys`` evita duplicar una prediccion ya registrada para el
    mismo (simbolo, modelo, as_of) -- pasa si el job se corre mas de una vez
    el mismo dia, o antes de que cierre una nueva rueda.
    """
    new_entries: list[dict] = []
    for name in model_names:
        service = registry.resolve(name)
        for symbol in tickers:
            try:
                pred = service.predict(symbol)
            except ApiMlError as exc:
                print(f"  [skip] {name}/{symbol}: {exc}")
                continue

            entry = {
                "symbol": pred["symbol"],
                "model": name,
                "model_version": pred["model_version"],
                "as_of": pred["as_of"],
                "signal": pred["signal"],
                "horizon_days": int(pred["horizon_days"]),
                "last_close": float(pred["last_close"]),
                "predicted_close": float(pred["predicted_close"]),
                "predicted_log_return": float(
                    np.log(float(pred["predicted_close"]) / float(pred["last_close"]))
                ),
            }
            key = _pending_key(entry)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_entries.append(entry)
    return new_entries


def build_summary(resolved: list[dict]) -> dict[str, dict]:
    """Agrega, por modelo, las metricas de ``src.backtest.summarize`` sobre
    todo lo resuelto hasta el momento -- el "historial de aciertos/errores"."""
    summary: dict[str, dict] = {}
    for name in sorted({r["model"] for r in resolved}):
        model_records = [r for r in resolved if r["model"] == name]
        per_ticker = {
            symbol: summarize([r for r in model_records if r["symbol"] == symbol])
            for symbol in sorted({r["symbol"] for r in model_records})
        }
        summary[name] = {"overall": summarize(model_records), "per_ticker": per_ticker}
    return summary


def known_tickers(ledger: dict) -> list[str]:
    """Fallback si el catalogo del collector esta temporalmente caido."""
    entries = [*ledger.get("pending", []), *ledger.get("resolved", [])]
    return sorted({entry["symbol"] for entry in entries if entry.get("symbol")})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper trading: predicciones diarias hacia adelante, historial persistente.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="*", help="Tickers a evaluar (default: todos los de data-colector)."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Modelos a evaluar (default: todos los cargados, incluidos *-modal).",
    )
    parser.add_argument(
        "--days", type=int, default=settings.history_days, help="Ruedas de historia a usar."
    )
    parser.add_argument(
        "--ledger", default=DEFAULT_LEDGER_PATH, help="Donde persistir el historial (JSON)."
    )
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    ledger = load_ledger(ledger_path)

    print(f"Pendientes de corridas anteriores: {len(ledger['pending'])}")
    still_pending, newly_resolved = resolve_pending(ledger["pending"], args.days)
    print(f"  [ok] {len(newly_resolved)} resueltas, {len(still_pending)} siguen pendientes")
    ledger["pending"] = still_pending
    ledger["resolved"].extend(newly_resolved)

    if args.tickers:
        tickers = args.tickers
    else:
        try:
            tickers = fetch_available_tickers()
        except DataUnavailableError as exc:
            tickers = known_tickers(ledger)
            if not tickers:
                ledger["summary"] = build_summary(ledger["resolved"])
                save_ledger(ledger_path, ledger)
                raise
            print(
                f"  [warn] {exc}\n"
                f"  [fallback] se usaran {len(tickers)} tickers ya conocidos por el ledger"
            )
    registry = build_registry(args.days)
    registry.load_all()
    model_names = select_model_names(registry, args.models)
    if not model_names:
        raise SystemExit("Ningun modelo cargado para registrar predicciones.")

    print(f"\nRegistrando predicciones de hoy: {len(tickers)} tickers x {len(model_names)} modelos")
    existing_keys = {_pending_key(e) for e in ledger["pending"]}
    existing_keys |= {_pending_key(e) for e in ledger["resolved"]}
    new_entries = record_new_predictions(registry, model_names, tickers, existing_keys)
    print(f"  [ok] {len(new_entries)} predicciones nuevas")
    ledger["pending"].extend(new_entries)

    ledger["summary"] = build_summary(ledger["resolved"])
    save_ledger(ledger_path, ledger)

    print(f"\nLedger actualizado: {ledger_path}")
    print(f"  pendientes: {len(ledger['pending'])}  |  resueltas: {len(ledger['resolved'])}")
    for name, data in ledger["summary"].items():
        print(f"  [{name}] {data['overall']}")


if __name__ == "__main__":
    main()
