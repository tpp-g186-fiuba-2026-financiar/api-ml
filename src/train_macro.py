"""Entrena el modelo macro: ``python -m src.train_macro``."""

from __future__ import annotations

import argparse

from src.config import settings
from src.data import fetch_history
from src.errors import ApiMlError
from src.macro_data import fetch_argentina_macro
from src.macro_trend import HORIZON, UNIVERSE, MacroConfig, MacroTrendModel

DEFAULT_MIN_AUC = 0.52


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de tendencia 'macro'.")
    parser.add_argument("--tickers", nargs="*", help="Tickers a usar (default: universo medido).")
    parser.add_argument("--days", type=int, default=4000)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--out", default=settings.macro_model_path, help="Ruta del artefacto .pkl")
    parser.add_argument("--min-auc", type=float, default=DEFAULT_MIN_AUC)
    args = parser.parse_args()

    tickers = args.tickers or list(UNIVERSE)
    print(f"Descargando historico de {len(tickers)} tickers ({args.days} ruedas)...")
    histories = {}
    for symbol in tickers:
        try:
            histories[symbol] = fetch_history(symbol, args.days)
            print(f"  [ok]   {symbol}: {len(histories[symbol])} ruedas")
        except ApiMlError as exc:
            print(f"  [skip] {symbol}: {exc}")
    if not histories:
        raise SystemExit("No se pudo descargar data de ningun ticker. Abortando.")

    print("\nDescargando series macro del data-colector...")
    macro = fetch_argentina_macro()
    print(f"  {len(macro.columns)} series, {macro.index[0].date()} -> {macro.index[-1].date()}")

    model = MacroTrendModel(config=MacroConfig(horizon=args.horizon))
    print(f"\nEntrenando con {len(histories)} tickers (horizonte {args.horizon} ruedas)...")
    metrics = model.fit(histories, macro)
    print("\nMedicion fuera de muestra (ultimo 25% de las fechas, reentrenos sucesivos):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"  umbral de 'baja' (P baja >=): {model.baja_threshold:.3f}")
    print(f"  umbral de 'alza' (P baja <=): {model.alza_threshold:.3f}")

    if metrics["oos_auc"] < args.min_auc:
        print(
            f"\nAUC fuera de muestra {metrics['oos_auc']} < {args.min_auc}: "
            "no se guarda, queda el artefacto vigente."
        )
        return
    model.save(args.out)
    print(f"\nModelo guardado en: {args.out}")
    print(f"Version: {model.version}")


if __name__ == "__main__":
    main()
