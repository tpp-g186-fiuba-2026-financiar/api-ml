"""Predicciones diarias de los modelos de panel para sumarlas a una corrida existente.

El simulador predice ticker por ticker (``predict_df``) y guarda / promueve
modelos con ``save`` / ``load`` y el gate; los modelos de panel
(``predict_panel``: todos los tickers del dia juntos) no tienen nada de eso.
Para ver sus carteras alcanza con sus predicciones: este modulo las genera
rueda por rueda en el formato de ``predicciones.jsonl`` y las agrega a una
**copia** de una corrida (la original no se toca); ``rearmar`` arma despues
todas las carteras, las de los modelos de panel incluidas (``mesa.MODELOS_PANEL``).

Como en las corridas existentes (``--reentreno semanal``): el primer modelo se
entrena con datos hasta la rueda anterior al periodo y se reentrena al cierre
de la ultima rueda de cada semana (sin gate: cada reentreno reemplaza al
anterior). Entrena y predice con los precios limpios (``limpieza``: la historia
incluye el split sin ajustar de ECOG del 2025-08-19); ``last_close`` es el
cierre crudo del dia, como en el simulador, y ``predicted_close`` se escala
igual. La senal es por cuantil del dia (``senal_por_cuantil``).

Uso (desde la raiz de api_ml; el Transformer de panel conviene en el host con GPU):
    python -m src.simulacion.predicciones_panel --corrida ene-may-2026-semanal \
        --copia ene-may-2026-panel --modelos ensamble reversion
    python -m src.simulacion.predicciones_panel --corrida ene-may-2026-semanal \
        --copia ene-may-2026-panel --modelos panel_transformer --dispositivo cuda
    python -m src.simulacion.rearmar --nombre ene-may-2026-panel --salida rebalanceo-5
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.data import MACRO_COLUMN, attach_macro_feature
from src.panel.ensamble import EnsambleConfig, EnsamblePanelModel
from src.panel.gbm import PanelGBMConfig
from src.panel.transformer import PanelTransformerConfig, PanelTransformerModel
from src.simulacion.busqueda import proveedor_snapshot
from src.simulacion.entrenamiento import histories_at, reentreno_due
from src.simulacion.experimento import ReversionPanelModel
from src.simulacion.limpieza import limpiar
from src.simulacion.mesa import MODELOS_PANEL
from src.simulacion.simulador import RESULTADOS_DIR, append_jsonl, silenciar_warnings
from src.simulacion.snapshot import Snapshot
from src.trend_common import senal_por_cuantil

CUANTIL_SENAL = 0.15
DIR_PANEL = "panel"  # <copia>/panel/<modelo>.jsonl
PREDICCIONES_BASE = "predicciones_base.jsonl"  # las de la corrida original
# Lo que ``rearmar`` y ``reporte`` leen de la corrida, ademas de las predicciones.
ARCHIVOS_COPIA = ("backtests.jsonl", "promociones.jsonl")

Fabrica = Callable[[], object]


def fabricas(snapshot: Snapshot, dispositivo: str = "cpu") -> dict[str, Fabrica]:
    """Modelo de panel (nombre de ``MODELOS_PANEL``) -> constructor sin entrenar.

    Configuraciones elegidas en la busqueda (``informe-mejora-modelos``):
    ensamble Ridge + reversion con senal por cuantil 0,15 (fase 5), la mejor
    variante del Transformer de panel (fase 4) y la regla de reversion a 5
    ruedas como control sin ML.
    """
    proveedor = proveedor_snapshot(snapshot)
    return {
        "ensamble": lambda: EnsamblePanelModel(
            EnsambleConfig(0.5, PanelGBMConfig(tipo="ridge"), CUANTIL_SENAL), proveedor=proveedor
        ),
        "panel_transformer": lambda: PanelTransformerModel(
            PanelTransformerConfig(
                ventana=1, dropout=0.3, weight_decay=0.3, learning_rate=1e-4, n_semillas=5
            ),
            proveedor=proveedor,
            dispositivo=dispositivo,
        ),
        "reversion": ReversionPanelModel,
    }


def _entrenar(
    fabrica: Fabrica, snapshot: Snapshot, hasta: pd.Timestamp, days: int
) -> tuple[object, dict]:
    """Modelo entrenado con datos hasta ``hasta`` y lo que devolvio su ``fit``."""
    modelo = fabrica()
    with contextlib.redirect_stdout(io.StringIO()):
        metricas = modelo.fit(histories_at(snapshot, hasta, days))
    return modelo, metricas if isinstance(metricas, dict) else {}


def predecir_dia(
    modelo,
    crudo: Snapshot,
    limpio: Snapshot,
    dia: pd.Timestamp,
    days: int,
    cuantil: float | None = CUANTIL_SENAL,
    tickers: list[str] = PANEL_LIDER_TICKERS,
) -> dict[str, dict]:
    """Prediccion de ``dia`` para los ``tickers`` que cotizaron ese dia, con
    ``last_close`` / ``predicted_close`` en la escala del snapshot crudo."""
    dfs = {}
    for symbol in tickers:
        if symbol not in limpio.prices or symbol not in crudo.prices:
            continue
        df = limpio.history(symbol, dia, days)
        if df.empty or df.index[-1] != dia:
            continue
        dfs[symbol] = attach_macro_feature(df, MACRO_COLUMN, limpio.rate_until(dia))
    if not dfs:
        return {}
    with contextlib.redirect_stdout(io.StringIO()):
        resultados = modelo.predict_panel(dfs)
    if cuantil is not None and resultados:
        resultados = senal_por_cuantil(resultados, cuantil)
    salida = {}
    for symbol, r in resultados.items():
        crudo_df = crudo.prices[symbol]
        if dia not in crudo_df.index:
            continue
        cierre = float(crudo_df.at[dia, "close"])
        escala = cierre / r["last_close"] if r["last_close"] else np.nan
        if not np.isfinite(escala) or escala <= 0:
            continue
        salida[symbol] = {
            **r,
            "predicted_close": round(r["predicted_close"] * escala, 4),
            "last_close": round(cierre, 4),
        }
    return salida


def generar(
    nombre: str,
    fabrica: Fabrica,
    crudo: Snapshot,
    limpio: Snapshot,
    desde: str,
    hasta: str,
    *,
    reentreno: str = "semanal",
    days: int = 750,
    cuantil: float | None = CUANTIL_SENAL,
    destino: Path | None = None,
    entrenamientos: Path | None = None,
) -> list[dict]:
    """Predicciones de ``nombre`` para cada rueda de ``[desde, hasta]``, una
    fila por (dia, ticker) con el formato de ``predicciones.jsonl``. Si se pasa
    ``destino``, las va agregando ahi (se pisa al empezar); si se pasa
    ``entrenamientos``, una fila por entrenamiento con la fecha de corte de los
    datos, los segundos y lo que devolvio el ``fit`` (tambien se pisa)."""
    ruedas = crudo.trading_days()
    periodo = ruedas[(ruedas >= pd.Timestamp(desde)) & (ruedas <= pd.Timestamp(hasta))]
    anteriores = ruedas[ruedas < pd.Timestamp(desde)]
    if periodo.empty or anteriores.empty:
        raise ValueError(f"sin ruedas para predecir en {desde} - {hasta}")
    for archivo in (destino, entrenamientos):
        if archivo is not None:
            archivo.parent.mkdir(parents=True, exist_ok=True)
            archivo.write_text("")

    def entrenar(hasta: pd.Timestamp) -> object:
        inicio = time.monotonic()
        modelo, metricas = _entrenar(fabrica, limpio, hasta, days)
        segundos = time.monotonic() - inicio
        print(f"[{nombre}] entrenado con datos hasta {hasta.date()} ({segundos:.0f} s)",
              flush=True)  # fmt: skip
        if entrenamientos is not None:
            fila = {"fecha": str(hasta.date()), "segundos": round(segundos, 1), **metricas}
            append_jsonl(entrenamientos, [fila])
        return modelo

    entrenado = anteriores[-1]
    modelo = entrenar(entrenado)
    filas: list[dict] = []
    for i, dia in enumerate(periodo):
        version = f"{getattr(modelo, 'version', nombre)}-{entrenado.date()}"
        del_dia = [
            {"fecha": str(dia.date()), **r, "symbol": s, "model": nombre, "model_version": version}
            for s, r in sorted(predecir_dia(modelo, crudo, limpio, dia, days, cuantil).items())
        ]
        filas += del_dia
        if destino is not None:
            append_jsonl(destino, del_dia)
        siguiente = periodo[i + 1] if i + 1 < len(periodo) else None
        if siguiente is not None and reentreno_due(reentreno, dia, siguiente):
            modelo, entrenado = entrenar(dia), dia
    return filas


def preparar_copia(original: Path, copia: Path) -> dict:
    """Crea ``copia`` con lo que ``rearmar`` necesita de ``original`` (si no
    existe) y devuelve su config. La original no se modifica."""
    if not (copia / "config.json").exists():
        copia.mkdir(parents=True, exist_ok=True)
        config = json.loads((original / "config.json").read_text())
        config.update(nombre=copia.name, origen=original.name)
        (copia / "config.json").write_text(json.dumps(config, indent=2))
        shutil.copyfile(original / "predicciones.jsonl", copia / PREDICCIONES_BASE)
        for archivo in ARCHIVOS_COPIA:
            if (original / archivo).exists():
                shutil.copyfile(original / archivo, copia / archivo)
    return json.loads((copia / "config.json").read_text())


def unir(copia: Path) -> int:
    """Reescribe ``predicciones.jsonl`` de la copia: las de la corrida original
    mas las de cada modelo de panel generado. Devuelve las lineas de panel."""
    panel = sorted((copia / DIR_PANEL).glob("*.jsonl"))
    lineas = 0
    with (copia / "predicciones.jsonl").open("w") as salida:
        salida.write((copia / PREDICCIONES_BASE).read_text())
        for archivo in panel:
            texto = archivo.read_text()
            lineas += texto.count("\n")
            salida.write(texto)
    return lineas


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Predicciones de los modelos de panel sobre una copia de una corrida.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--corrida", required=True, help="Corrida original (no se modifica).")
    parser.add_argument("--copia", required=True, help="Nombre de la copia en resultados/.")
    parser.add_argument("--modelos", nargs="*", default=[], choices=MODELOS_PANEL)
    parser.add_argument("--dispositivo", default="cpu")
    args = parser.parse_args(argv)
    silenciar_warnings()

    original, copia = Path(RESULTADOS_DIR) / args.corrida, Path(RESULTADOS_DIR) / args.copia
    if original.resolve() == copia.resolve():
        raise SystemExit("la copia tiene que tener otro nombre que la corrida original")
    config = preparar_copia(original, copia)
    if args.modelos:
        crudo = Snapshot.load(config["snapshot_dir"])
        limpio = limpiar(crudo)[0]
        disponibles = fabricas(limpio, args.dispositivo)
        for nombre in args.modelos:
            destino = copia / DIR_PANEL / f"{nombre}.jsonl"
            filas = generar(
                nombre,
                disponibles[nombre],
                crudo,
                limpio,
                config["desde"],
                config["hasta"],
                reentreno=config.get("reentreno", "semanal"),
                days=config.get("history_days", 750),
                destino=destino,
            )
            print(f"[{nombre}] {len(filas)} predicciones en {destino}", flush=True)
    print(f"{unir(copia)} predicciones de panel en {copia / 'predicciones.jsonl'}")


if __name__ == "__main__":
    main()
