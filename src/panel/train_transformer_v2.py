"""Entrena el Transformer V2 (``src.panel.transformer_v2``) con datos del data-colector.

A diferencia de ``train_transformer`` (el original), siempre usa el
data-colector, sin importar ``DATA_SOURCE``: los precios (OHLC ajustados) y
el CCL para pasarlos a dolares salen de ahi. Entrena con las acciones del
panel lider y guarda el artefacto en ``--out`` (por default
``settings.transformer_v2_model_path``), que la API sirve como "transformer".

Por default pasa por el gate por IC fuera de muestra (``src.panel.gate``):
entrena dos veces (evaluacion y final) y solo guarda si el candidato ordena
mejor que el azar. ``--force`` salta el gate (entrena una vez y guarda). Lo
corre el cron mensual ``retrain-transformer-v2.yml``.

Uso:
    make train-transformer-v2
    make train-transformer-v2 ARGS="--force"
    make train-transformer-v2 ARGS="--collector-url http://localhost:3000 --dispositivo cuda"
    make train-transformer-v2 ARGS="--collector-url http://localhost:3000 --dispositivo cuda \
        --macro --perdida bce --days 1500 --cobertura-senal 0.2"
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.config import settings
from src.data import fetch_history_collector
from src.errors import DataUnavailableError
from src.lstm import fetch_histories
from src.macro_data import REQUIRED, SERIES, build_macro_frame, fetch_collector_series
from src.panel.gate import entrenar_con_gate
from src.panel.transformer_v2 import HISTORIA_V2, TransformerV2Config, TransformerV2Model


def proveedor_collector(macro: bool = False):
    """``proveedor(hasta) -> (ccl, macro)`` con las series del data-colector (se
    piden una sola vez). El CCL va sin retraso (para pasar precios a dolares);
    la tabla macro, solo si ``macro``, con los retrasos de publicacion de cada
    serie (``build_macro_frame``, como en produccion y en la simulacion)."""
    nombres = list(SERIES) if macro else ["ccl"]
    crudas: dict[str, pd.Series] = {}
    for nombre in nombres:
        ruta, serie, _retraso = SERIES[nombre]
        try:
            crudas[nombre] = fetch_collector_series(ruta, serie)
        except DataUnavailableError as exc:
            if nombre in REQUIRED:
                raise
            print(f"  [warn] {exc}")

    def proveedor(hasta: pd.Timestamp):
        cortas = {n: s[s.index <= hasta] for n, s in crudas.items()}
        tabla = build_macro_frame(cortas, end=hasta) if macro else None
        return cortas["ccl"], tabla

    return proveedor


def main() -> None:
    defaults = TransformerV2Config()
    parser = argparse.ArgumentParser(description="Entrena el Transformer V2 de tendencia.")
    parser.add_argument(
        "--collector-url",
        default=settings.data_collector_url,
        help="URL del data-colector (desde el host: http://localhost:3000).",
    )
    parser.add_argument("--days", type=int, default=HISTORIA_V2, help="Ruedas de historia a usar.")
    parser.add_argument("--dispositivo", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-semillas", type=int, default=defaults.n_semillas)
    parser.add_argument("--horizonte", type=int, default=defaults.horizonte)
    parser.add_argument("--perdida", choices=defaults.perdidas_validas(), default=defaults.perdida)
    parser.add_argument("--macro", action="store_true", help="Sumar el token macro del dia.")
    parser.add_argument(
        "--cobertura-senal",
        type=float,
        default=defaults.cobertura_senal,
        help="Fraccion de predicciones (las mas fuertes) con senal; sin valor: todas.",
    )
    parser.add_argument(
        "--peso-reversion",
        type=float,
        default=defaults.peso_reversion,
        help="Peso de la regla de reversion en el ensamble (0 = solo la red).",
    )
    parser.add_argument(
        "--centrar-senal", action="store_true", help="Restarle al score su mediana."
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--nhead", type=int, default=defaults.nhead)
    parser.add_argument("--dim-feedforward", type=int, default=defaults.dim_feedforward)
    parser.add_argument("--capas-cruzadas", type=int, default=defaults.capas_cruzadas)
    parser.add_argument("--ventana", type=int, default=defaults.ventana)
    parser.add_argument("--epocas", type=int, default=defaults.epocas)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--out", default=settings.transformer_v2_model_path, help="Ruta del artefacto .pt"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Guardar sin pasar por el gate por IC (entrena una sola vez).",
    )
    args = parser.parse_args()
    settings.data_collector_url = args.collector_url

    tickers = list(PANEL_LIDER_TICKERS)
    print(f"Data-colector: {args.collector_url}")
    print(f"Descargando historico de {len(tickers)} tickers ({args.days} ruedas)...")
    histories = fetch_histories(tickers, args.days, fetch_history_collector)
    if not histories:
        raise SystemExit("No se pudo descargar data de ningun ticker. Abortando.")
    print("Descargando el CCL" + (" y las series macro..." if args.macro else "..."))
    proveedor = proveedor_collector(args.macro)

    config = TransformerV2Config(
        n_semillas=args.n_semillas,
        ventana=args.ventana,
        epocas=args.epocas,
        horizonte=args.horizonte,
        perdida=args.perdida,
        macro=args.macro,
        cobertura_senal=args.cobertura_senal,
        peso_reversion=args.peso_reversion,
        centrar_senal=args.centrar_senal,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        capas_cruzadas=args.capas_cruzadas,
        seed=args.seed,
    )

    def fabrica() -> TransformerV2Model:
        return TransformerV2Model(config, proveedor=proveedor, dispositivo=args.dispositivo)

    print(f"\nEntrenando Transformer V2 con {len(histories)} tickers en {args.dispositivo}...")
    print(f"Config: {asdict(config)}")
    if not args.force:
        print("Gate por IC: entrena sin las ultimas ruedas, mide el IC ahi y, si pasa, reentrena.")
        promovido, score = entrenar_con_gate(
            fabrica, TransformerV2Model.load, histories, args.out, config.horizonte
        )
        print(f"  candidato: {score}")
        if not promovido:
            print("\nEl candidato no ordena mejor que el azar -- no se toca el artefacto.")
            return
        print(f"\nModelo guardado en: {args.out}")
        return

    modelo = fabrica()
    metrics = modelo.fit(histories)
    print("\nMetricas de validacion (antes del reentreno con todos los datos):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    modelo.save(args.out)
    print(f"\nModelo guardado en: {args.out}")
    print(f"Version: {modelo.version} ({modelo.trained_at})")


if __name__ == "__main__":
    main()
