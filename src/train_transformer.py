from __future__ import annotations

import argparse

from src.config import settings
from src.data import fetch_history
from src.lstm import default_merval_tickers, fetch_histories
from src.transformer import TransformerTrainConfig, TransformerTrendModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo Transformer de tendencia.")
    parser.add_argument("--tickers", nargs="*", help="Tickers a usar (default: panel Merval).")
    parser.add_argument("--days", type=int, default=settings.history_days, help="Ruedas a usar.")
    parser.add_argument("--window", type=int, default=TransformerTrainConfig.window)
    parser.add_argument("--horizon", type=int, default=TransformerTrainConfig.horizon)
    parser.add_argument("--epochs", type=int, default=TransformerTrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TransformerTrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TransformerTrainConfig.learning_rate)
    parser.add_argument("--d-model", type=int, default=TransformerTrainConfig.d_model)
    parser.add_argument("--nhead", type=int, default=TransformerTrainConfig.nhead)
    parser.add_argument("--num-layers", type=int, default=TransformerTrainConfig.num_layers)
    parser.add_argument(
        "--dim-feedforward", type=int, default=TransformerTrainConfig.dim_feedforward
    )
    parser.add_argument("--dropout", type=float, default=TransformerTrainConfig.dropout)
    parser.add_argument(
        "--warmup-epochs", type=int, default=TransformerTrainConfig.warmup_epochs
    )
    parser.add_argument(
        "--out", default=settings.transformer_model_path, help="Ruta del artefacto .pt"
    )
    args = parser.parse_args()

    tickers = args.tickers or default_merval_tickers()
    print(f"Fuente de datos: {settings.data_source}")
    print(f"Descargando historico de {len(tickers)} tickers ({args.days} ruedas)...")
    histories = fetch_histories(tickers, args.days, fetch_history)
    if not histories:
        raise SystemExit("No se pudo descargar data de ningun ticker. Abortando.")

    config = TransformerTrainConfig(
        window=args.window,
        horizon=args.horizon,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
    )
    model = TransformerTrendModel(config=config)

    print(f"\nEntrenando Transformer con {len(histories)} tickers...")
    metrics = model.fit(histories)

    model.save(args.out)
    print("\nMetricas de validacion:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nModelo guardado en: {args.out}")
    print(f"Version: {model.version}")


if __name__ == "__main__":
    main()
