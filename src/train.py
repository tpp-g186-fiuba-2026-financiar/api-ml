"""Entrenamiento del modelo LSTM de tendencia.

Uso:

    python -m src.train                       # todos los tickers disponibles en data-colector
    python -m src.train --tickers GGAL YPFD   # subconjunto
    python -m src.train --epochs 60 --window 40

Descarga el historico de cada ticker (fuente segun ``settings.data_source``),
entrena un unico modelo LSTM y guarda el artefacto en ``settings.lstm_model_path``.
"""

from __future__ import annotations

import argparse

from src.config import settings
from src.data import fetch_available_tickers, fetch_history
from src.lstm import TrainConfig, TrendModel, fetch_histories


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo LSTM de tendencia.")
    parser.add_argument(
        "--tickers", nargs="*", help="Tickers a usar (default: todos los de data-colector)."
    )
    parser.add_argument("--days", type=int, default=settings.history_days, help="Ruedas a usar.")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--window", type=int, default=TrainConfig.window)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--out", default=settings.lstm_model_path, help="Ruta del artefacto .pt")
    args = parser.parse_args()

    tickers = args.tickers or fetch_available_tickers()
    print(f"Fuente de datos: {settings.data_source}")
    print(f"Descargando historico de {len(tickers)} tickers ({args.days} ruedas)...")
    histories = fetch_histories(tickers, args.days, fetch_history)
    if not histories:
        raise SystemExit("No se pudo descargar data de ningun ticker. Abortando.")

    config = TrainConfig(
        window=args.window,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    model = TrendModel(config=config)

    print(f"\nEntrenando LSTM con {len(histories)} tickers...")
    metrics = model.fit(histories)

    model.save(args.out)
    print("\nMetricas de validacion:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nModelo guardado en: {args.out}")
    print(f"Version: {model.version}")


if __name__ == "__main__":
    main()
