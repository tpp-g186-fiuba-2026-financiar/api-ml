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
from src.compare_snapshot import DEFAULT_SNAPSHOT_PATH, save_snapshot
from src.config import settings
from src.consensus import compute_consensus
from src.data import fetch_available_tickers, fetch_history
from src.errors import ApiMlError, DataUnavailableError
from src.registry import TrendRegistry, build_registry
from src.schemas import PerfilRiesgo

DEFAULT_LEDGER_PATH = "models/paper_trading_ledger.json"

CONSENSUS_PREFIX = "consensus-"
CONSENSUS_VERSION = "consensus-v1"
# Clasificacion del consenso -> signal de trend, asi ``summarize`` lo mide
# igual que a cualquier modelo (signal_hit_rate = "cuando dio senal, acerto?").
_CONSENSUS_SIGNAL = {"sobrecompra": "alza", "sobreventa": "baja", "neutral": "neutral"}


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


def compare_all(registry: TrendRegistry, tickers: list[str]) -> dict[str, dict]:
    """``compare()`` de cada ticker calculado de cero (lo que guarda el snapshot)."""
    compared: dict[str, dict] = {}
    for symbol in tickers:
        try:
            compared[symbol.strip().upper()] = registry.compare(symbol, use_snapshot=False)
        except ApiMlError as exc:
            print(f"  [skip] {symbol}: {exc}")
    return compared


def _compared_prediction(compared: dict[str, dict] | None, name: str, symbol: str) -> dict | None:
    entry = (compared or {}).get(symbol.strip().upper())
    pred = (entry or {}).get("predictions", {}).get(name)
    if not pred or "reason" in pred or "signal" not in pred:
        return None
    return pred


def record_new_predictions(
    registry: TrendRegistry,
    model_names: list[str],
    tickers: list[str],
    existing_keys: set[tuple],
    compared: dict[str, dict] | None = None,
) -> list[dict]:
    """Corre cada modelo cargado sobre cada ticker y arma las entradas nuevas.

    ``existing_keys`` evita duplicar una prediccion ya registrada para el
    mismo (simbolo, modelo, as_of) -- pasa si el job se corre mas de una vez
    el mismo dia, o antes de que cierre una nueva rueda. Si ``compared`` ya
    trae la prediccion de ese modelo para ese ticker, se usa esa en vez de
    volver a correr el modelo.
    """
    new_entries: list[dict] = []
    for name in model_names:
        service = registry.resolve(name)
        for symbol in tickers:
            pred = _compared_prediction(compared, name, symbol)
            if pred is None:
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


def _model_quality_lookup(resolved: list[dict]) -> dict[tuple[str, str], dict]:
    """Accuracy direccional y cantidad de casos por (modelo, ticker), calculada
    solo con lo ya resuelto: es la calidad que el consenso hubiera conocido al
    momento de predecir (sin mirar hacia adelante). Los consensos no entran
    como insumo de si mismos."""
    base = [r for r in resolved if not r["model"].startswith(CONSENSUS_PREFIX)]
    lookup: dict[tuple[str, str], dict] = {}
    for model, data in build_summary(base).items():
        for symbol, stats in data["per_ticker"].items():
            lookup[(model, symbol)] = {
                "directional_accuracy": stats.get("directional_accuracy"),
                "observations": stats.get("n_predictions"),
            }
    return lookup


def record_consensus_entries(
    pending: list[dict],
    new_entries: list[dict],
    resolved: list[dict],
    existing_keys: set[tuple],
) -> list[dict]:
    """Por cada ticker, combina las predicciones de hoy de todos los modelos
    con ``compute_consensus`` (una vez por perfil de riesgo) y las registra
    como un modelo mas (``consensus-<perfil>``), para medir cuantas veces
    acerto lo que se le muestra al usuario."""
    pool = [e for e in [*pending, *new_entries] if not e["model"].startswith(CONSENSUS_PREFIX)]
    latest_as_of: dict[str, str] = {}
    for e in pool:
        latest_as_of[e["symbol"]] = max(latest_as_of.get(e["symbol"], ""), e["as_of"])

    quality = _model_quality_lookup(resolved)
    by_symbol: dict[str, list[dict]] = {}
    for e in pool:
        if e["as_of"] == latest_as_of[e["symbol"]]:
            by_symbol.setdefault(e["symbol"], []).append(e)

    entries: list[dict] = []
    for symbol, group in sorted(by_symbol.items()):
        predictions = {
            e["model"]: {
                "signal": e["signal"],
                "last_close": e["last_close"],
                "predicted_close": e["predicted_close"],
                "backtest": quality.get((e["model"], symbol)),
            }
            for e in group
        }
        as_of = latest_as_of[symbol]
        last_close = group[0]["last_close"]
        for profile in PerfilRiesgo:
            result = compute_consensus(predictions, profile)
            if result["classification"] == "sin_datos":
                continue
            entry = {
                "symbol": symbol,
                "model": f"{CONSENSUS_PREFIX}{profile.value}",
                "model_version": CONSENSUS_VERSION,
                "as_of": as_of,
                "signal": _CONSENSUS_SIGNAL[result["classification"]],
                "horizon_days": int(group[0]["horizon_days"]),
                "last_close": last_close,
                "predicted_close": float(last_close * np.exp(result["composite_score"])),
                "predicted_log_return": float(result["composite_score"]),
            }
            key = _pending_key(entry)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            entries.append(entry)
    return entries


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
        # Separado por version y horizonte: un XGBoost de 5 ruedas y otro de 20
        # son modelos distintos aunque compartan el nombre.
        by_version = {
            f"{version} · {horizon}r": summarize(
                [
                    r
                    for r in model_records
                    if r.get("model_version") == version and r.get("horizon_days") == horizon
                ]
            )
            for version, horizon in sorted(
                {(r.get("model_version"), r.get("horizon_days")) for r in model_records},
                key=lambda key: (str(key[0]), key[1] or 0),
            )
        }
        summary[name] = {
            "overall": summarize(model_records),
            "per_ticker": per_ticker,
            "by_version": by_version,
        }
    return summary


def _signal_hit(entry: dict) -> bool | None:
    """Mismo criterio que ``signal_hit_rate`` de ``src.backtest.summarize``:
    acierta si el signo del retorno realizado coincide con alza/baja. None si
    todavia no se resolvio o si la senal fue neutral (no opino)."""
    if "realized_close" not in entry or entry.get("signal") not in ("alza", "baja"):
        return None
    realized = float(np.sign(np.log(entry["realized_close"] / entry["last_close"])))
    return realized == (1.0 if entry["signal"] == "alza" else -1.0)


def signal_history(ledger: dict, symbols: list[str] | None = None, limit: int = 200) -> dict:
    """Historial de senales del ledger (pendientes + resueltas), la mas nueva
    primero, filtrado por ``symbols`` si se pasa. ``summary`` cuenta solo las
    filas devueltas que ya tienen resultado (``hit`` no nulo)."""
    wanted = {s.strip().upper() for s in symbols} if symbols else None
    rows = []
    for entry in [*ledger.get("pending", []), *ledger.get("resolved", [])]:
        if wanted is not None and entry.get("symbol", "").upper() not in wanted:
            continue
        rows.append(
            {
                "symbol": entry["symbol"],
                "model": entry["model"],
                "model_version": entry.get("model_version"),
                "as_of": entry["as_of"],
                "signal": entry["signal"],
                "horizon_days": int(entry["horizon_days"]),
                "last_close": entry["last_close"],
                "predicted_close": entry["predicted_close"],
                "realized_close": entry.get("realized_close"),
                "hit": _signal_hit(entry),
                "resolved_at": entry.get("resolved_at"),
            }
        )
    rows.sort(key=lambda r: (r["as_of"], r["symbol"], r["model"]), reverse=True)
    rows = rows[:limit]
    evaluated = [r["hit"] for r in rows if r["hit"] is not None]
    hits = sum(evaluated)
    return {
        "rows": rows,
        "summary": {
            "evaluated": len(evaluated),
            "hits": hits,
            "hit_rate": round(hits / len(evaluated), 4) if evaluated else None,
            # Fecha de la ultima senal registrada y del ultimo guardado del
            # ledger: si quedaron viejas, el job diario esta frenado.
            "last_signal": max((r["as_of"] for r in rows), default=None),
            "updated_at": ledger.get("updated_at"),
        },
    }


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
    parser.add_argument(
        "--snapshot",
        default=DEFAULT_SNAPSHOT_PATH,
        help="Donde guardar la comparacion del dia que sirve el comparador (JSON).",
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

    print(f"\nComparando modelos en {len(tickers)} tickers...")
    compared = compare_all(registry, tickers)
    print(f"  [ok] {len(compared)} tickers con precios")
    if not compared:
        ledger["summary"] = build_summary(ledger["resolved"])
        save_ledger(ledger_path, ledger)
        raise SystemExit(
            "No se pudo bajar el historico de ningun ticker (fuente: "
            f"{settings.data_source}). Revisar Yahoo / data-colector: no se registran "
            "predicciones ni se actualiza el snapshot."
        )
    save_snapshot(args.snapshot, compared)
    print(f"  [ok] snapshot del comparador: {args.snapshot}")

    print(f"\nRegistrando predicciones de hoy: {len(tickers)} tickers x {len(model_names)} modelos")
    existing_keys = {_pending_key(e) for e in ledger["pending"]}
    existing_keys |= {_pending_key(e) for e in ledger["resolved"]}
    new_entries = record_new_predictions(
        registry, model_names, list(compared), existing_keys, compared
    )
    print(f"  [ok] {len(new_entries)} predicciones nuevas")
    consensus_entries = record_consensus_entries(
        ledger["pending"], new_entries, ledger["resolved"], existing_keys
    )
    print(f"  [ok] {len(consensus_entries)} lecturas de consenso nuevas")
    ledger["pending"].extend(new_entries)
    ledger["pending"].extend(consensus_entries)

    ledger["summary"] = build_summary(ledger["resolved"])
    save_ledger(ledger_path, ledger)

    print(f"\nLedger actualizado: {ledger_path}")
    print(f"  pendientes: {len(ledger['pending'])}  |  resueltas: {len(ledger['resolved'])}")
    for name, data in ledger["summary"].items():
        print(f"  [{name}] {data['overall']}")


if __name__ == "__main__":
    main()


def track_record(ledger: dict, min_casos: int = 20) -> dict:
    """Que hubiera pasado siguiendo las senales de cada modelo, con todo lo ya
    resuelto del ledger (todas las acciones, no solo las de un usuario).

    Por modelo y version: cuantas veces dijo alza y baja y cuanto acerto,
    contra la tasa base de esas mismas fechas; y el retorno promedio de
    "comprar cuando dice alza" contra "comprar todo lo que el modelo miro"
    (la vara: sin elegir nada). La curva acumula esos retornos mes a mes.
    Retornos log en pesos, al horizonte de cada senal.
    """
    resueltas = [r for r in ledger.get("resolved", []) if "realized_log_return" in r]
    grupos: dict[tuple[str, str], list[dict]] = {}
    for r in resueltas:
        # Las copias de Modal se reentrenan en cada pedido y cada uno trae
        # otra version: se miran como una sola.
        version = "modal" if r["model"].endswith("-modal") else r.get("model_version") or "?"
        grupos.setdefault((r["model"], version), []).append(r)

    filas = []
    for (modelo, version), rs in sorted(grupos.items()):
        if len(rs) < min_casos:
            continue
        ret = np.array([r["realized_log_return"] for r in rs], dtype=float)
        sube = ret > 0
        alza = np.array([r["signal"] == "alza" for r in rs])
        baja = np.array([r["signal"] == "baja" for r in rs])
        meses: dict[str, list[tuple[float, bool]]] = {}
        for r, x, a in zip(rs, ret, alza, strict=True):
            meses.setdefault(r["as_of"][:7], []).append((float(x), bool(a)))
        curva, acum_sigue, acum_todas = [], 0.0, 0.0
        for mes in sorted(meses):
            casos = meses[mes]
            todas = float(np.mean([x for x, _ in casos]))
            elegidas = [x for x, a in casos if a]
            # sin senales de alza ese mes, "seguirlas" es quedarse afuera (0)
            sigue = float(np.mean(elegidas)) if elegidas else 0.0
            acum_sigue += sigue
            acum_todas += todas
            curva.append(
                {
                    "mes": mes,
                    "sigue": round(acum_sigue, 5),
                    "todas": round(acum_todas, 5),
                    "senales": len(elegidas),
                }
            )
        filas.append(
            {
                "model": modelo,
                "model_version": version,
                "horizon_days": int(rs[0].get("horizon_days") or 0),
                "casos": len(rs),
                "desde": min(r["as_of"] for r in rs),
                "hasta": max(r["as_of"] for r in rs),
                "tasa_suba": float(sube.mean()),
                "alza": {
                    "n": int(alza.sum()),
                    "acierto": float(sube[alza].mean()) if alza.any() else None,
                    "retorno_medio": float(ret[alza].mean()) if alza.any() else None,
                },
                "baja": {
                    "n": int(baja.sum()),
                    "acierto": float((~sube[baja]).mean()) if baja.any() else None,
                },
                "retorno_medio_todas": float(ret.mean()),
                "curva": curva,
            }
        )
    filas.sort(key=lambda f: (f["model"], f["hasta"]), reverse=False)
    return {"updated_at": ledger.get("updated_at"), "modelos": filas}
