"""Entrenamiento del modelo XGBoost de tendencia.

Uso:

    python -m src.train_xgb                       # todos los tickers disponibles en data-colector
    python -m src.train_xgb --tickers GGAL YPFD   # subconjunto
    python -m src.train_xgb --n-estimators 500 --max-depth 5

Descarga el historico de cada ticker (fuente segun ``settings.data_source``),
entrena un unico modelo XGBoost y guarda el artefacto en
``settings.xgb_model_path``. Usa exactamente los mismos features, target y
tickers que el LSTM, para que las metricas sean comparables.
"""

from __future__ import annotations

import argparse

from src.config import settings
from src.data import fetch_available_tickers, fetch_history
from src.lstm import fetch_histories
from src.retrain_guard import force_promote, promote_if_better
from src.xgb_trend import XGBConfig, XGBTrendModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo XGBoost de tendencia.")
    parser.add_argument(
        "--tickers", nargs="*", help="Tickers a usar (default: todos los de data-colector)."
    )
    parser.add_argument("--days", type=int, default=settings.history_days, help="Ruedas a usar.")
    parser.add_argument("--window", type=int, default=XGBConfig.window)
    parser.add_argument("--horizon", type=int, default=XGBConfig.horizon)
    parser.add_argument("--n-estimators", type=int, default=XGBConfig.n_estimators)
    parser.add_argument("--max-depth", type=int, default=XGBConfig.max_depth)
    parser.add_argument("--learning-rate", type=float, default=XGBConfig.learning_rate)
    parser.add_argument("--out", default=settings.xgb_model_path, help="Ruta del artefacto .pkl")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Guardar aunque el candidato no supere al vigente (salta el gate de promocion).",
    )
    args = parser.parse_args()

    tickers = args.tickers or fetch_available_tickers()
    print(f"Fuente de datos: {settings.data_source}")
    print(f"Descargando historico de {len(tickers)} tickers ({args.days} ruedas)...")
    histories = fetch_histories(tickers, args.days, fetch_history)
    if not histories:
        raise SystemExit("No se pudo descargar data de ningun ticker. Abortando.")

    config = XGBConfig(
        window=args.window,
        horizon=args.horizon,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    model = XGBTrendModel(config=config)

    print(f"\nEntrenando XGBoost con {len(histories)} tickers...")
    metrics = model.fit(histories)
    print("\nMetricas de validacion:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\nBacktesteando el candidato (walk-forward, pooled sobre todos los tickers)...")
    if args.force:
        score = force_promote(model, args.out, histories, config.horizon)
        print(f"  candidato: {score}  (--force: se guardo sin comparar)")
    else:
        promoted, score = promote_if_better(
            model, args.out, XGBTrendModel.load, histories, config.horizon
        )
        print(f"  candidato: {score}")
        if not promoted:
            print("\nEl candidato no supera al vigente -- no se promueve, no se toca el artefacto.")
            return

    print(f"\nModelo guardado en: {args.out}")
    print(f"Version: {model.version}")


if __name__ == "__main__":
    main()
