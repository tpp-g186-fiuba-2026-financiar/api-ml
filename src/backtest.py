"""Backtesting walk-forward de los modelos de tendencia ya entrenados/cargados.

A diferencia de las metricas de validacion que guarda cada modelo en su
artefacto (un unico split calculado al momento de entrenar), este script
recorre el historico completo con un cutoff movil: en cada punto le muestra
al modelo solo los datos hasta ese dia y compara la senal que dio contra lo
que efectivamente paso ``horizon`` dias despues. Sirve para:

- Comparar LSTM / XGBoost / Transformer / ARIMA en igualdad de condiciones
  sobre el mismo historico.
- Dejar una foto (JSON) de cada corrida, para poder decir "esta version del
  modelo mejoro respecto a la anterior" con numeros, no a ojo.

Los modelos remotos (``*-modal``, corren en Modal via HTTP) quedan afuera:
su ``predict_on`` no respeta el DataFrame recortado, siempre pega contra el
historico en vivo (ver ``RemoteTrendService`` en ``src.registry``), asi que
"backtestearlos" repetiria la misma prediccion actual en cada cutoff.

Ojo (in-sample parcial): los modelos pooled (LSTM/XGBoost/Transformer) se
entrenaron sobre todo el historico disponible en su momento, asi que parte
de este backtest puede caer dentro del periodo de entrenamiento. Sigue
siendo util para monitorear la calidad de la senal *tal como esta deployada
hoy*, y sobre todo para comparar corridas entre si (antes/despues de un
cambio de features, de hiperparametros, de reentrenamiento, etc.).

Uso:
    python -m src.backtest                       # todos los tickers, modelos locales
    python -m src.backtest --tickers GGAL YPFD
    python -m src.backtest --models lstm xgboost
    python -m src.backtest --step 20 --min-history 300
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.data import (
    MACRO_COLUMN,
    attach_macro_feature,
    fetch_available_tickers,
    fetch_history,
    fetch_macro_series,
)
from src.errors import ApiMlError
from src.registry import (
    OnDemandTrendService,
    RemoteTrendService,
    TrendRegistry,
    TrendService,
    build_registry,
)

DEFAULT_STEP = 10
DEFAULT_MIN_HISTORY = 200


def walk_forward(
    service: TrendService | OnDemandTrendService,
    df: pd.DataFrame,
    symbol: str,
    *,
    step: int,
    min_history: int,
) -> list[dict]:
    """Predice con un cutoff movil y compara contra lo que paso despues.

    En cada ``cutoff`` se le pasa al modelo solo ``df.iloc[:cutoff + 1]`` (como
    si ese fuera el ultimo dato disponible) y se compara ``predicted_close``
    contra el cierre real ``horizon`` ruedas mas adelante, que en ese momento
    el modelo no vio.
    """
    records: list[dict] = []
    n = len(df)
    for cutoff in range(min_history, n, step):
        window_df = df.iloc[: cutoff + 1]
        try:
            pred = service.predict_on(window_df, symbol)
        except ApiMlError:
            continue

        horizon = int(pred["horizon_days"])
        future_idx = cutoff + horizon
        if future_idx >= n:
            break  # cutoffs mas altos necesitan todavia mas futuro

        last_close = float(pred["last_close"])
        predicted_close = float(pred["predicted_close"])
        future_close = float(df["close"].iloc[future_idx])
        records.append(
            {
                "symbol": symbol,
                "as_of": pred["as_of"],
                "signal": pred["signal"],
                "predicted_log_return": float(np.log(predicted_close / last_close)),
                "realized_log_return": float(np.log(future_close / last_close)),
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    """Agrega metricas de calidad de senal a partir de los registros de ``walk_forward``."""
    if not records:
        return {"n_predictions": 0}

    pred_ret = np.array([r["predicted_log_return"] for r in records])
    real_ret = np.array([r["realized_log_return"] for r in records])
    def _position(signal: str) -> float:
        if signal == "alza":
            return 1.0
        if signal == "baja":
            return -1.0
        return 0.0

    position = np.array([_position(r["signal"]) for r in records])
    non_neutral = position != 0
    # PnL si se hubiese seguido la senal al pie de la letra (long en alza,
    # short en baja, afuera en neutral), sin costos de transaccion.
    strategy_ret = position * real_ret

    summary = {
        "n_predictions": len(records),
        "mae_logret": round(float(np.mean(np.abs(pred_ret - real_ret))), 6),
        "rmse_logret": round(float(np.sqrt(np.mean((pred_ret - real_ret) ** 2))), 6),
        "directional_accuracy": round(float(np.mean(np.sign(pred_ret) == np.sign(real_ret))), 4),
        "neutral_rate": round(float(np.mean(~non_neutral)), 4),
        "avg_strategy_logret": round(float(np.mean(strategy_ret)), 6),
        "avg_buy_hold_logret": round(float(np.mean(real_ret)), 6),
    }
    summary["signal_hit_rate"] = (
        round(float(np.mean(np.sign(real_ret[non_neutral]) == position[non_neutral])), 4)
        if non_neutral.any()
        else None
    )
    return summary


def select_model_names(registry: TrendRegistry, requested: list[str] | None) -> list[str]:
    """Filtra los modelos remotos (Modal): no se pueden backtestear con cutoff historico."""
    all_names = registry.names()
    remote_names = {n for n in all_names if isinstance(registry.resolve(n), RemoteTrendService)}

    if not requested:
        return [n for n in all_names if n not in remote_names]

    selected = []
    for name in requested:
        if name in remote_names:
            print(
                f"  [skip] '{name}': es un modelo remoto (Modal), su predict_on no respeta "
                "el cutoff historico -- no se puede backtestear con este script."
            )
            continue
        selected.append(name)
    return selected


def _fetch_histories(tickers: list[str], days: int, min_rows: int) -> dict[str, pd.DataFrame]:
    """Descarga el historico de cada ticker, con la feature macro ya alineada.

    Sin esto, los modelos que usan ``macro_rate_chg5`` (ver ``FEATURE_NAMES``
    en ``src.lstm``) predecirian con esa feature siempre en 0/NaN durante el
    backtest -- distinto a como se entrenaron -- y los numeros del backtest
    quedarian peores de lo que el modelo realmente da en produccion.
    """
    macro = fetch_macro_series(days)
    histories: dict[str, pd.DataFrame] = {}
    for symbol in tickers:
        try:
            df = fetch_history(symbol, days)
        except ApiMlError as exc:
            print(f"  [skip] {symbol}: {exc}")
            continue
        if len(df) < min_rows:
            print(f"  [skip] {symbol}: pocas ruedas ({len(df)})")
            continue
        histories[symbol] = attach_macro_feature(df, MACRO_COLUMN, macro)
        print(f"  [ok]   {symbol}: {len(df)} ruedas")
    return histories


def run_backtest(
    registry: TrendRegistry,
    histories: dict[str, pd.DataFrame],
    model_names: list[str],
    *,
    step: int,
    min_history: int,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for name in model_names:
        service = registry.resolve(name)
        if not service.is_loaded:
            print(f"  [skip] modelo '{name}': no esta entrenado")
            continue

        per_ticker: dict[str, dict] = {}
        all_records: list[dict] = []
        for symbol, df in histories.items():
            records = walk_forward(service, df, symbol, step=step, min_history=min_history)
            if records:
                per_ticker[symbol] = summarize(records)
                all_records.extend(records)
            print(f"  [{name}] {symbol}: {len(records)} predicciones")

        results[name] = {
            "version": service.version,
            "overall": summarize(all_records),
            "per_ticker": per_ticker,
        }
    return results


def _print_comparison(results: dict[str, dict]) -> None:
    rows = [
        {"model": name, "version": data["version"], **data["overall"]}
        for name, data in results.items()
    ]
    if not rows:
        print("\nSin resultados (ningun modelo tenia suficientes predicciones).")
        return
    table = pd.DataFrame(rows).set_index("model")
    print("\nComparacion (pooled sobre todos los tickers):\n")
    print(table.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtesting walk-forward de los modelos de tendencia ya cargados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="*", help="Tickers a evaluar (default: todos los de data-colector)."
    )
    parser.add_argument(
        "--models", nargs="*", help="Modelos a evaluar (default: todos los locales, sin *-modal)."
    )
    parser.add_argument(
        "--days", type=int, default=settings.history_days, help="Ruedas de historia a usar."
    )
    parser.add_argument(
        "--step", type=int, default=DEFAULT_STEP, help="Cada cuantas ruedas mover el cutoff."
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=DEFAULT_MIN_HISTORY,
        help="Ruedas minimas de historia antes del primer cutoff.",
    )
    parser.add_argument(
        "--out", default="models/backtest_results.json", help="Donde guardar el detalle en JSON."
    )
    args = parser.parse_args()

    tickers = args.tickers or fetch_available_tickers()
    print(f"Descargando historico de {len(tickers)} tickers ({args.days} ruedas)...")
    histories = _fetch_histories(tickers, args.days, args.min_history + 10)
    if not histories:
        raise SystemExit("No se pudo descargar data de ningun ticker. Abortando.")

    registry = build_registry(args.days)
    registry.load_all()
    model_names = select_model_names(registry, args.models)
    if not model_names:
        raise SystemExit("Ningun modelo valido para backtestear.")

    print(f"\nEvaluando modelos: {', '.join(model_names)}")
    print(f"Tickers: {len(histories)}  |  step={args.step}  |  min_history={args.min_history}\n")

    results = run_backtest(
        registry, histories, model_names, step=args.step, min_history=args.min_history
    )
    _print_comparison(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {
            "days": args.days,
            "step": args.step,
            "min_history": args.min_history,
            "tickers": sorted(histories),
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nDetalle guardado en: {out_path}")


if __name__ == "__main__":
    main()
