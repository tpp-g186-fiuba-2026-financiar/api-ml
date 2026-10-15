"""Rehace las carteras de una corrida con otras reglas, sin reentrenar ni predecir.

Lee las predicciones y backtests guardados por el simulador y vuelve a pasar
cada rueda por la ``Mesa`` con otra comision, capital o frecuencia de
rebalanceo. Con los mismos parametros reproduce exactamente ``valores.jsonl``
de la corrida original.

El backtest en vivo solo se guarda en los dias de rebalanceo de la corrida
original. Si ``--rebalanceo`` no es multiplo del original, en los dias sin
backtest se usa el ultimo guardado (de hasta ``rebalanceo original - 1`` ruedas
antes). Solo lo usan el consenso y la eleccion del mejor modelo; las carteras
de senal y Black-Litterman no dependen del backtest.

Si ``predicciones.jsonl`` trae predicciones de modelos de panel
(``mesa.MODELOS_PANEL``, generadas con ``predicciones_panel``), se suman sus
carteras; las del resto quedan iguales.

Uso:
    python -m src.simulacion.rearmar --nombre junio --salida comision-1pct --comision 0.01
    python -m src.simulacion.rearmar --nombre junio --salida rebalanceo-10 --rebalanceo 10
    python -m src.simulacion.rearmar --nombre junio --salida rebalanceo-1 --rebalanceo 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.simulacion.mesa import MODELOS_CON_BACKTEST, MODELOS_PANEL, Mesa, Predicciones
from src.simulacion.simulador import RESULTADOS_DIR, guardar_rueda, silenciar_warnings
from src.simulacion.snapshot import Snapshot


def _leer_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cargar_por_dia(
    corrida: Path, arrastrar_backtest: bool = False
) -> tuple[dict[str, Predicciones], dict[str, Predicciones]]:
    """Predicciones y backtests guardados, como los devolvio ``Simulacion.predecir``.

    Igual que en la corrida, el backtest de ``MODELOS_CON_BACKTEST`` viaja
    dentro de su prediccion (pondera el consenso). Con ``arrastrar_backtest``,
    los dias sin backtest guardado usan el del ultimo dia que lo tuvo.
    """
    backtests: dict[str, Predicciones] = defaultdict(lambda: defaultdict(dict))
    for fila in _leer_jsonl(corrida / "backtests.jsonl"):
        fecha, modelo, symbol = fila.pop("fecha"), fila.pop("model"), fila.pop("symbol")
        backtests[fecha][modelo][symbol] = fila
    predicciones: dict[str, Predicciones] = defaultdict(lambda: defaultdict(dict))
    for fila in _leer_jsonl(corrida / "predicciones.jsonl"):
        predicciones[fila.pop("fecha")][fila["model"]][fila["symbol"]] = fila

    ultimo: Predicciones = {}
    for fecha in sorted(predicciones):
        if fecha in backtests:
            ultimo = backtests[fecha]
        elif arrastrar_backtest and ultimo:
            backtests[fecha] = ultimo
        for modelo in MODELOS_CON_BACKTEST:
            for symbol, fila in predicciones[fecha].get(modelo, {}).items():
                bt = backtests.get(fecha, {}).get(modelo, {}).get(symbol)
                if bt is not None:
                    fila["backtest"] = bt
    return predicciones, backtests


def rearmar(
    corrida: Path,
    salida: Path,
    snapshot: Snapshot,
    *,
    capital: float | None = None,
    comision: float | None = None,
    rebalanceo: int | None = None,
) -> dict:
    """Rehace las carteras de ``corrida`` en ``salida``; devuelve la config usada."""
    original = json.loads((corrida / "config.json").read_text())
    config = {
        "corrida": original["nombre"],
        "capital": original["capital"] if capital is None else capital,
        "comision": original["comision"] if comision is None else comision,
        "rebalanceo": original["rebalanceo"] if rebalanceo is None else rebalanceo,
    }
    if config["rebalanceo"] < 1:
        raise SystemExit("--rebalanceo tiene que ser al menos 1")
    config["backtest_arrastrado"] = config["rebalanceo"] % original["rebalanceo"] != 0

    predicciones, backtests = cargar_por_dia(corrida, config["backtest_arrastrado"])
    fechas = sorted(predicciones)
    if not fechas:
        raise SystemExit(f"{corrida} no tiene predicciones guardadas")
    presentes = {m for por_modelo in predicciones.values() for m in por_modelo}
    modelos_panel = tuple(m for m in MODELOS_PANEL if m in presentes)
    if modelos_panel:
        config["modelos_panel"] = list(modelos_panel)

    if salida.exists():
        shutil.rmtree(salida)
    salida.mkdir(parents=True)
    (salida / "config.json").write_text(json.dumps(config, indent=2))

    mesa = Mesa(
        snapshot, config["capital"], config["comision"], config["rebalanceo"], modelos_panel
    )
    for fecha in fechas:
        guardar_rueda(
            salida, mesa.operar(pd.Timestamp(fecha), predicciones[fecha], backtests[fecha])
        )
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rehace las carteras de una corrida sin reentrenar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--nombre", required=True, help="Corrida a rearmar.")
    parser.add_argument("--salida", required=True, help="Subcarpeta en <corrida>/rearmados/.")
    parser.add_argument("--capital", type=float)
    parser.add_argument("--comision", type=float)
    parser.add_argument("--rebalanceo", type=int)
    args = parser.parse_args(argv)
    silenciar_warnings()

    corrida = Path(RESULTADOS_DIR) / args.nombre
    original = json.loads((corrida / "config.json").read_text())
    snapshot = Snapshot.load(original["snapshot_dir"])
    salida = corrida / "rearmados" / args.salida
    rearmar(
        corrida,
        salida,
        snapshot,
        capital=args.capital,
        comision=args.comision,
        rebalanceo=args.rebalanceo,
    )
    print(f"Carteras rearmadas en {salida}")


if __name__ == "__main__":
    main()
