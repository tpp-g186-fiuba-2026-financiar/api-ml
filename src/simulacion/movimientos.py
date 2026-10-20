"""Registro de movimientos de una corrida: cada compra / venta con lo que decian los modelos.

Junta en ``movimientos.csv`` (en la carpeta de las carteras: la corrida o un
rearmado) lo que la simulacion guarda por separado:

- la operacion (``operaciones.jsonl``): fecha, cartera, ticker, compra o
  venta, cantidad, precio, monto y comision;
- de que estrategia es la cartera (senal, Black-Litterman, consenso, mejor
  modelo, benchmark), que modelo la maneja y con que perfil;
- el peso objetivo del ticker en esa cartera ese dia (``pesos.jsonl``);
- la prediccion del modelo de la cartera para ese ticker ese dia: senal,
  retorno esperado (log) y horizonte. En las de "mejor modelo", el modelo que
  se eligio para ese ticker (``elecciones.jsonl``);
- la senal de **todos** los modelos para ese ticker ese dia (``senal_<modelo>``);
- lo que paso despues: retorno log real (en pesos) a 5 y a 20 ruedas desde el
  cierre de la operacion (vacio si el snapshot no llega).

Uso:
    python -m src.simulacion.movimientos --nombre contexto-2026
    python -m src.simulacion.movimientos --nombre contexto-2026 --rearmado sin-comision
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulacion.estrategias import PERFILES
from src.simulacion.mesa import MODELOS, MODELOS_PANEL, REGLAS_PANEL
from src.simulacion.snapshot import Snapshot

RESULTADOS_DIR = "simulacion/resultados"
ARCHIVO = "movimientos.csv"
HORIZONTES_REALES = (5, 20)
_PERFILES = {p.value for p in PERFILES}


def _leer_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True, dtype=False, convert_dates=False)


def describir_cartera(nombre: str) -> dict:
    """Estrategia, modelo y perfil de una cartera a partir de su nombre."""
    partes = nombre.split("_")
    perfil = partes[-1] if partes[-1] in _PERFILES else None
    sin_perfil = partes[:-1] if perfil else partes
    estrategia = sin_perfil[0]
    if estrategia == "benchmark":
        return {"estrategia": "benchmark", "modelo": None, "perfil": None}
    if estrategia == "consenso":
        return {"estrategia": "consenso", "modelo": None, "perfil": perfil}
    modelo = "_".join(sin_perfil[1:]) or None
    if estrategia in ("senal", "bl") or estrategia in REGLAS_PANEL:
        return {"estrategia": estrategia, "modelo": modelo, "perfil": perfil}
    return {"estrategia": nombre, "modelo": None, "perfil": None}


def _pesos(carteras: Path) -> pd.DataFrame:
    filas = [
        {"fecha": r["fecha"], "cartera": r["cartera"], "ticker": t, "peso_objetivo": w}
        for r in _leer_jsonl(carteras / "pesos.jsonl").to_dict("records")
        for t, w in (r["pesos"] or {}).items()
    ]
    return pd.DataFrame(filas, columns=["fecha", "cartera", "ticker", "peso_objetivo"])


def _predicciones(corrida: Path) -> pd.DataFrame:
    """Una fila por (fecha, modelo, ticker) con senal, retorno esperado y horizonte."""
    preds = _leer_jsonl(corrida / "predicciones.jsonl")
    if preds.empty:
        return pd.DataFrame(columns=["fecha", "modelo", "ticker", "senal", "retorno_esperado"])
    return pd.DataFrame(
        {
            "fecha": preds["fecha"].astype(str),
            "modelo": preds["model"],
            "ticker": preds["symbol"],
            "senal": preds["signal"],
            "retorno_esperado": np.log(
                preds["predicted_close"].astype(float) / preds["last_close"].astype(float)
            ),
            "horizonte": preds["horizon_days"],
        }
    )


def _retornos_reales(snapshot: Snapshot, fechas_tickers: pd.DataFrame) -> pd.DataFrame:
    """Retorno log real a ``HORIZONTES_REALES`` ruedas desde el cierre de cada fecha."""
    filas = []
    for ticker, grupo in fechas_tickers.groupby("ticker"):
        precios = snapshot.prices.get(ticker)
        for fecha in grupo["fecha"].unique():
            fila = {"fecha": fecha, "ticker": ticker}
            if precios is not None:
                close = precios["close"]
                pos = close.index.searchsorted(pd.Timestamp(fecha))
                for h in HORIZONTES_REALES:
                    ok = pos < len(close) and pos + h < len(close)
                    real = np.log(close.iloc[pos + h] / close.iloc[pos]) if ok else np.nan
                    fila[f"retorno_real_{h}"] = float(real)
            filas.append(fila)
    return pd.DataFrame(filas)


def movimientos(corrida: Path, snapshot: Snapshot, rearmado: str | None = None) -> pd.DataFrame:
    """Operaciones de las carteras de ``corrida`` (o de su rearmado) con el contexto
    de cada una (ver docstring del modulo)."""
    carteras = corrida / "rearmados" / rearmado if rearmado else corrida
    ops = _leer_jsonl(carteras / "operaciones.jsonl")
    if ops.empty:
        return pd.DataFrame()
    ops["fecha"] = ops["fecha"].astype(str)
    descripcion = pd.DataFrame([describir_cartera(c) for c in ops["cartera"]], index=ops.index)
    df = pd.concat([ops, descripcion], axis=1)

    pesos = _pesos(carteras)
    df = df.merge(pesos, on=["fecha", "cartera", "ticker"], how="left")
    df["peso_objetivo"] = df["peso_objetivo"].fillna(0.0)

    # Modelo que decide cada fila: el de la cartera, o el elegido para ese ticker
    # en las de "mejor modelo".
    df["modelo_usado"] = df["modelo"]
    elecciones = _leer_jsonl(carteras / "elecciones.jsonl")
    if not elecciones.empty:
        elegido = elecciones.assign(fecha=elecciones["fecha"].astype(str))[
            ["fecha", "seleccion", "symbol", "model"]
        ].rename(columns={"symbol": "ticker", "model": "modelo_elegido"})
        elegido["modelo"] = "mejor_" + elegido.pop("seleccion")
        df = df.merge(elegido, on=["fecha", "modelo", "ticker"], how="left")
        df["modelo_usado"] = df["modelo_elegido"].fillna(df["modelo_usado"])
        df = df.drop(columns="modelo_elegido")

    preds = _predicciones(corrida)
    propia = preds.rename(columns={"modelo": "modelo_usado", "senal": "senal_modelo"})
    df = df.merge(propia, on=["fecha", "modelo_usado", "ticker"], how="left")
    todas = preds.pivot_table(
        index=["fecha", "ticker"], columns="modelo", values="senal", aggfunc="first"
    )
    orden = [m for m in (*MODELOS, *MODELOS_PANEL) if m in todas.columns]
    todas = todas[orden].add_prefix("senal_").reset_index()
    df = df.merge(todas, on=["fecha", "ticker"], how="left")

    reales = _retornos_reales(snapshot, df[["fecha", "ticker"]].drop_duplicates())
    df = df.merge(reales, on=["fecha", "ticker"], how="left")
    columnas = [
        "fecha", "cartera", "estrategia", "modelo", "perfil", "ticker", "lado", "cantidad",
        "precio", "monto", "comision", "peso_objetivo", "modelo_usado", "senal_modelo",
        "retorno_esperado", "horizonte",
    ]  # fmt: skip
    resto = [c for c in df.columns if c not in columnas]
    return df[columnas + resto].sort_values(["fecha", "cartera", "lado", "ticker"])


def escribir(corrida: Path, snapshot: Snapshot, rearmado: str | None = None) -> Path | None:
    """Escribe ``movimientos.csv`` en la carpeta de las carteras; None si no hubo operaciones."""
    df = movimientos(corrida, snapshot, rearmado)
    if df.empty:
        return None
    destino = (corrida / "rearmados" / rearmado if rearmado else corrida) / ARCHIVO
    df.to_csv(destino, index=False)
    return destino


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Registro de movimientos de una corrida.")
    parser.add_argument("--nombre", required=True, help="Corrida.")
    parser.add_argument("--rearmado", help="Usar las carteras de <corrida>/rearmados/<nombre>.")
    args = parser.parse_args(argv)
    corrida = Path(RESULTADOS_DIR) / args.nombre
    config = json.loads((corrida / "config.json").read_text())
    destino = escribir(corrida, Snapshot.load(config["snapshot_dir"]), args.rearmado)
    print(f"Movimientos en {destino}" if destino else "La corrida no tiene operaciones")


if __name__ == "__main__":
    main()
