"""Acierto de la direccion (sube / baja) en dolares de una corrida de ``busqueda``.

``busqueda`` mide el retorno real en pesos. Para la direccion eso engaña: con
inflacion, en pesos sube la mayoria de los casos (2023: 58% de las subas a 5
ruedas del panel, 49% en dolares CCL) y un modelo que dice siempre "alza"
parece acertar. Aca, para cada prediccion guardada de la corrida, el retorno
real se recalcula en dolares CCL y se mide:

- ``bal_usd``: acierto balanceado del signo de la prediccion contra el retorno
  real en dolares (promedio del acierto en las subas y en las bajas: decir
  siempre "alza" da 0,5 haya las subas que haya). Es la metrica principal.
- ``bal_ars``: lo mismo contra el retorno en pesos (para ver cuanto infla la
  inflacion) y ``acierto_ars``: acierto comun en pesos (lo que se veia antes).
- ``pct_alza``: fraccion de predicciones positivas.
- ``prec_alza`` / ``prec_baja``: de las predicciones "sube" / "baja", cuantas
  subieron / bajaron en dolares, menos la tasa base de subas / bajas
  (> 0 = la prediccion aporta).
- ``bal_top``: ``bal_usd`` de la **senal selectiva**: solo el ``COBERTURA``
  (20%) de los casos con la prediccion mas fuerte (``|pred|``) del periodo; el
  resto seria "neutral". El umbral sale del propio periodo (en produccion,
  ``cobertura_senal`` lo calibra con los datos de entrenamiento): mide si las
  predicciones fuertes aciertan mas, no el umbral.

Cada variante se mide contra el retorno a su ``horizonte`` (de ``configs.json``
de la corrida; default 5 ruedas). Con varias semillas se promedian las
predicciones (ensamble) antes de tomar el signo. Se suman dos referencias
calculadas sobre las mismas filas que las variantes a 5 ruedas: "siempre alza"
y la reversion a 5 ruedas en dolares (predice lo contrario del retorno en
dolares de las ultimas 5 ruedas).

Uso:
    python -m src.simulacion.direccion --nombre v2-direccion-2 \
        --snapshot simulacion/snapshot-collector --limpiar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.panel_datos import ccl_en
from src.simulacion.busqueda import DIR_EXPERIMENTOS, PERIODOS, cargar_snapshot
from src.simulacion.experimento import HORIZON
from src.simulacion.snapshot import Snapshot
from src.trend_common import acierto_balanceado

COBERTURA = 0.2
REFERENCIA_ALZA = "siempre_alza"
REFERENCIA_REVERSION = "reversion_usd"


def retornos_usd(snapshot: Snapshot, horizonte: int = HORIZON) -> pd.DataFrame:
    """Retorno log en dolares CCL a ``horizonte`` ruedas (``futuro``) y de las
    ultimas ``horizonte`` ruedas (``pasado``), por (fecha, symbol)."""
    ultima = max(df.index[-1] for df in snapshot.prices.values())
    ccl = snapshot.ccl_hasta(ultima)
    partes = []
    for symbol, df in snapshot.prices.items():
        usd = np.log(df["close"] / ccl_en(ccl, df.index))
        partes.append(
            pd.DataFrame(
                {
                    "fecha": df.index.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "futuro": (usd.shift(-horizonte) - usd).to_numpy(),
                    "pasado": (usd - usd.shift(horizonte)).to_numpy(),
                }
            )
        )
    return pd.concat(partes, ignore_index=True)


def metricas(pred: pd.Series, real_usd: pd.Series, real_ars: pd.Series) -> dict:
    sube = real_usd > 0
    alza = pred > 0
    return {
        "bal_usd": acierto_balanceado(pred.to_numpy(), real_usd.to_numpy()),
        "bal_ars": acierto_balanceado(pred.to_numpy(), real_ars.to_numpy()),
        "acierto_ars": float((alza == (real_ars > 0)).mean()),
        "pct_alza": float(alza.mean()),
        "prec_alza": float(sube[alza].mean() - sube.mean()) if alza.any() else np.nan,
        "prec_baja": float((~sube)[~alza].mean() - (~sube).mean()) if (~alza).any() else np.nan,
        "bal_top": _acierto_top(pred, real_usd),
    }


def _acierto_top(pred: pd.Series, real_usd: pd.Series, cobertura: float = COBERTURA) -> float:
    """``bal_usd`` sobre los casos con ``|pred|`` en el cuantil superior ``cobertura``."""
    fuerza = pred.abs()
    top = fuerza >= fuerza.quantile(1 - cobertura)
    return acierto_balanceado(pred[top].to_numpy(), real_usd[top].to_numpy())


def horizontes(salida: Path) -> dict[str, int]:
    """Horizonte de cada variante de la corrida (``configs.json``; default 5)."""
    archivo = salida / "configs.json"
    configs = json.loads(archivo.read_text()) if archivo.exists() else {}
    return {v: (c or {}).get("horizonte", HORIZON) for v, c in configs.items()}


def por_periodo(salida: Path, retornos: dict[int, pd.DataFrame], periodos: str) -> pd.DataFrame:
    """Una fila por (variante, periodo) con las metricas del ensamble de semillas,
    mas las referencias sobre las filas de las variantes a 5 ruedas.
    ``retornos``: ``retornos_usd`` por horizonte."""
    h_variante = horizontes(salida)
    archivos = sorted((salida / "predicciones").glob("*.csv"))
    filas = []
    for desde, _ in PERIODOS[periodos]:
        del_periodo = [a for a in archivos if a.stem.endswith(f"__{desde}")]
        variantes = sorted({a.stem.split("__")[0] for a in del_periodo})
        base = None
        for variante in variantes:
            df = pd.concat(
                pd.read_csv(a) for a in del_periodo if a.stem.startswith(f"{variante}__")
            )
            df = df.groupby(["fecha", "symbol"], as_index=False).agg(
                pred=("pred", "mean"), real=("real", "first")
            )
            h = h_variante.get(variante, HORIZON)
            df = df.merge(retornos[h], on=["fecha", "symbol"]).dropna(subset=["futuro"])
            filas.append({"variante": variante, "desde": desde, **_fila(df, df["pred"])})
            if h == HORIZON and (base is None or len(df) > len(base)):
                base = df
        if base is not None:
            filas.append(
                {"variante": REFERENCIA_ALZA, "desde": desde, **_fila(base, base["pred"] * 0 + 1)}
            )
            rev = base.dropna(subset=["pasado"])
            filas.append(
                {"variante": REFERENCIA_REVERSION, "desde": desde, **_fila(rev, -rev["pasado"])}
            )
    return pd.DataFrame(filas)


def _fila(df: pd.DataFrame, pred: pd.Series) -> dict:
    return {"n": len(df), **metricas(pred, df["futuro"], df["real"])}


def resumen(tabla: pd.DataFrame) -> pd.DataFrame:
    """Promedio entre periodos, peor periodo, periodos con ``bal_usd`` > 0,5 y el
    t de ``bal_usd - 0,5`` entre periodos."""
    filas = []
    for variante, g in tabla.groupby("variante"):
        exceso = g["bal_usd"] - 0.5
        error = exceso.std() / np.sqrt(len(exceso)) if len(exceso) > 1 else np.nan
        filas.append(
            {
                "variante": variante,
                "periodos": len(g),
                "bal_usd": g["bal_usd"].mean(),
                "min_periodo": g["bal_usd"].min(),
                "periodos_>0,5": int((g["bal_usd"] > 0.5).sum()),
                "t": exceso.mean() / error if error and error > 0 else np.nan,
                "bal_ars": g["bal_ars"].mean(),
                "acierto_ars": g["acierto_ars"].mean(),
                "pct_alza": g["pct_alza"].mean(),
                "prec_alza": g["prec_alza"].mean(),
                "prec_baja": g["prec_baja"].mean(),
                "bal_top": g["bal_top"].mean(),
                "top_periodos_>0,5": int((g["bal_top"] > 0.5).sum()),
            }
        )
    return pd.DataFrame(filas).set_index("variante").sort_values("bal_usd", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acierto de la direccion en dolares.")
    parser.add_argument("--nombre", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--limpiar", action="store_true", help="Precios sin saltos (limpieza).")
    parser.add_argument("--periodos", choices=sorted(PERIODOS), default="eleccion")
    args = parser.parse_args()
    salida = DIR_EXPERIMENTOS / args.nombre
    snapshot = cargar_snapshot(args.snapshot, args.limpiar)
    retornos = {h: retornos_usd(snapshot, h) for h in {HORIZON, *horizontes(salida).values()}}
    tabla = por_periodo(salida, retornos, args.periodos)
    tabla.to_csv(salida / f"direccion_{args.periodos}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(resumen(tabla).round(4))


if __name__ == "__main__":
    main()
