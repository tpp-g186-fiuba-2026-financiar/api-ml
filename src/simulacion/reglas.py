"""Ronda 3: reglas sin entrenamiento a 5 y 20 ruedas y senal selectiva neta de costos.

Dos preguntas, sin entrenar redes (ver ``notas/analisis-transformer-v2.md``):

1. **Hay senal a 20 ruedas?** Reglas de la literatura calculadas con los
   precios en dolares CCL (solo hacia atras): reversion de corto plazo,
   momentum (con y sin el salto de un mes de Jegadeesh-Titman), momentum del
   mercado y "valuacion tecnica" (distancia a la media de 200 ruedas y al
   maximo / minimo de 52 semanas). Se miden con ``significancia`` (IC,
   ``bal_relativo``, ``bal_usd``... con intervalos por bloques de fechas) a
   cada horizonte, contra la reversion del mismo horizonte. Opcionalmente se
   suman las predicciones de una corrida de ``busqueda`` (``--corrida``).
2. **Rinde una senal selectiva solo comprando y pagando costos?** Para cada
   regla o modelo: el umbral de ``|pred|`` sale de las predicciones de las
   ``VENTANA`` ruedas anteriores (no usa resultados, asi que no hay fuga); se
   compra cada accion con ``pred > 0`` por encima del umbral y se mide el
   retorno simple a ``horizonte`` ruedas en dolares, menos ``COSTO`` (ida y
   vuelta), y su exceso sobre el promedio del panel del mismo dia (que no paga
   costos: la comparacion es conservadora). Intervalos por bloques de fechas.

Uso:
    python -m src.simulacion.reglas --nombre r3-reglas --snapshot simulacion/snapshot-collector \\
        --limpiar --corrida r3-modelos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.panel.datos import ccl_en
from src.simulacion.busqueda import DIR_EXPERIMENTOS, PERIODOS, cargar_snapshot
from src.simulacion.direccion import horizontes
from src.simulacion.experimento import HORIZON
from src.simulacion.significancia import NIVEL, legible, pesos_bootstrap, significancia
from src.simulacion.snapshot import DEFAULT_DIR, Snapshot

HORIZONTES = (5, 20)
PERIODOS_DEFAULT = ("eleccion", "previos")
COBERTURAS = (0.05, 0.10, 0.20)
VENTANA = 250  # ruedas de predicciones para calibrar el umbral
MINIMO = 60  # ruedas minimas de predicciones previas para tener umbral
COSTO = 0.01  # ida y vuelta (comision de 0,5% por operacion)
N_BOOT = 2000


def precios_usd(snapshot: Snapshot, tickers: list[str] = PANEL_LIDER_TICKERS) -> pd.DataFrame:
    """Log del cierre en dolares CCL, ``[fecha, ticker]`` (NaN donde no cotizo)."""
    ultima = max(df.index[-1] for df in snapshot.prices.values())
    ccl = snapshot.ccl_hasta(ultima)
    columnas = {
        s: np.log(snapshot.prices[s]["close"] / ccl_en(ccl, snapshot.prices[s].index))
        for s in tickers
        if s in snapshot.prices
    }
    return pd.DataFrame(columnas).sort_index()


def reglas(p: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prediccion de cada regla por (fecha, ticker): mas alto = mejor. Todas usan
    solo precios hasta la fecha."""

    def ret(k: int) -> pd.DataFrame:
        return p - p.shift(k)

    def del_mercado(serie: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(np.repeat(serie.to_numpy()[:, None], p.shape[1], 1),
                            index=p.index, columns=p.columns).where(p.notna())  # fmt: skip

    media_200 = p.rolling(200, min_periods=150).mean()
    maximo = p.rolling(252, min_periods=200).max()
    minimo = p.rolling(252, min_periods=200).min()
    return {
        "rev_5": -ret(5),
        "rev_20": -ret(20),
        "mom_60": ret(60),
        "mom_120": ret(120),
        "mom_250": ret(250),
        "mom_6_1": p.shift(20) - p.shift(125),
        "mom_12_1": p.shift(20) - p.shift(250),
        "mercado_mom_60": del_mercado(ret(60).mean(axis=1)),
        "mercado_rev_20": del_mercado(-ret(20).mean(axis=1)),
        # Valuacion tecnica: lejos abajo de su media larga = "barata" = compra.
        "val_media_200": -(p - media_200),
        # Signo de George-Hwang: cerca del maximo (valor alto, <= 0) sigue subiendo.
        "cerca_max_252": p - maximo,
        "lejos_min_252": p - minimo,
    }


def _en_periodos(index: pd.DatetimeIndex, periodos: list[tuple[str, str]]) -> pd.Series:
    """Periodo (``desde``) de cada fecha, NaN fuera de los periodos."""
    desde = pd.Series(np.nan, index=index, dtype=object)
    for inicio, fin in periodos:
        desde[(index >= pd.Timestamp(inicio)) & (index <= pd.Timestamp(fin))] = inicio
    return desde


def filas(
    pred: pd.DataFrame, p: pd.DataFrame, horizonte: int, periodos: list[tuple[str, str]]
) -> pd.DataFrame:
    """Formato de ``significancia``: ``desde``, ``fecha``, ``symbol``, ``pred``,
    ``futuro`` (retorno log en USD a ``horizonte``) y ``pasado``, en las fechas de
    los periodos con prediccion y retorno conocidos."""
    pred = pred.reindex(index=p.index, columns=p.columns)
    futuro, pasado = p.shift(-horizonte) - p, p - p.shift(horizonte)
    desde = _en_periodos(p.index, periodos)
    dentro = desde.notna().to_numpy()
    partes = []
    for symbol in p.columns:
        partes.append(
            pd.DataFrame(
                {
                    "desde": desde.to_numpy()[dentro],
                    "fecha": p.index[dentro].strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "pred": pred[symbol].to_numpy()[dentro],
                    "futuro": futuro[symbol].to_numpy()[dentro],
                    "pasado": pasado[symbol].to_numpy()[dentro],
                }
            )
        )
    df = pd.concat(partes, ignore_index=True).dropna(subset=["pred", "futuro"])
    return df.sort_values(["fecha", "symbol"]).reset_index(drop=True)


def umbrales(pred: pd.DataFrame, cobertura: float, ventana: int = VENTANA,
             minimo: int = MINIMO) -> pd.Series:  # fmt: skip
    """Umbral de ``|pred|`` de cada fecha: el cuantil ``1 - cobertura`` de las
    predicciones de las ``ventana`` fechas anteriores con prediccion (no incluye
    la fecha). NaN si hay menos de ``minimo`` fechas previas."""
    fuerza = pred.abs()
    con_pred = fuerza.notna().any(axis=1).to_numpy()
    valores = fuerza.to_numpy()
    fechas = np.flatnonzero(con_pred)
    salida = pd.Series(np.nan, index=pred.index)
    for j, t in enumerate(fechas):
        if j < minimo:
            continue
        previas = valores[fechas[max(0, j - ventana) : j]]
        salida.iloc[t] = np.nanquantile(previas, 1 - cobertura)
    return salida


def apuestas(pred: pd.DataFrame, p: pd.DataFrame, horizonte: int,
             periodos: list[tuple[str, str]], cobertura: float) -> pd.DataFrame:  # fmt: skip
    """Una fila por compra de la senal selectiva: ``fecha``, ``symbol``, retorno
    simple ``ret`` a ``horizonte`` ruedas y el del panel ese dia (``panel``)."""
    pred = pred.reindex(index=p.index, columns=p.columns)
    ret = np.expm1(p.shift(-horizonte) - p)
    panel = ret.mean(axis=1)
    umbral = umbrales(pred, cobertura)
    dentro = _en_periodos(p.index, periodos).notna()
    compra = (pred > 0) & pred.abs().ge(umbral, axis=0) & ret.notna()
    compra = compra & dentro.to_numpy()[:, None]
    t, n = np.nonzero(compra.to_numpy())
    return pd.DataFrame(
        {
            "fecha": p.index[t],
            "symbol": p.columns[n],
            "ret": ret.to_numpy()[t, n],
            "panel": panel.to_numpy()[t],
        }
    )


def resumen_apuestas(df: pd.DataFrame, costo: float = COSTO, bloque: int = 20,
                     nivel: float = NIVEL, seed: int = 0) -> dict:  # fmt: skip
    """Retorno medio por apuesta (bruto, neto de ``costo``) y exceso neto sobre el
    panel, con intervalos por bootstrap de bloques de fechas."""
    if df.empty:
        return {"apuestas": 0}
    df = df.assign(neto=df["ret"] - costo, exceso=df["ret"] - df["panel"] - costo,
                   acierto=(df["ret"] > 0).astype(float))  # fmt: skip
    por_fecha = df.groupby("fecha").agg(
        n=("ret", "size"), neto=("neto", "sum"), exceso=("exceso", "sum")
    )
    pesos = pesos_bootstrap(len(por_fecha), N_BOOT, bloque, seed)
    cola = (1 - nivel) / 2
    n = pesos @ por_fecha["n"].to_numpy()
    fila = {
        "apuestas": len(df),
        "fechas": len(por_fecha),
        "acierto": float(df["acierto"].mean()),
        "bruto": float(df["ret"].mean()),
    }
    for col in ("neto", "exceso"):
        fila[col] = float(df[col].mean())
        fila[f"{col}_lo"], fila[f"{col}_hi"] = np.quantile(
            (pesos @ por_fecha[col].to_numpy()) / n, [cola, 1 - cola]
        )
    return fila


# --------------------------------------------------------------------------- #
def predicciones_corrida(
    salida: Path, index: pd.DatetimeIndex
) -> dict[str, tuple[int, pd.DataFrame]]:
    """Predicciones de cada variante de una corrida de ``busqueda`` (promedio de
    semillas), en formato ancho, con su horizonte."""
    h = horizontes(salida)
    por_variante: dict[str, list[pd.DataFrame]] = {}
    for archivo in sorted((salida / "predicciones").glob("*.csv")):
        por_variante.setdefault(archivo.stem.split("__")[0], []).append(pd.read_csv(archivo))
    salida_ = {}
    for variante, dfs in por_variante.items():
        df = pd.concat(dfs).groupby(["fecha", "symbol"])["pred"].mean().unstack()
        df.index = pd.to_datetime(df.index)
        salida_[variante] = (h.get(variante, HORIZON), df.reindex(index))
    return salida_


def _texto(df: pd.DataFrame) -> str:
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        return df.to_string()


def analizar(snapshot: Snapshot, periodos: list[str], horizontes_: tuple[int, ...],
             corrida: Path | None = None) -> tuple[dict, pd.DataFrame]:  # fmt: skip
    """``{(periodos, h): tabla de significancia}`` y la tabla de la senal selectiva."""
    p = precios_usd(snapshot)
    candidatos = {(n, h): pred for h in horizontes_ for n, pred in reglas(p).items()}
    if corrida is not None:
        for variante, (h, pred) in predicciones_corrida(corrida, p.index).items():
            if h in horizontes_:
                candidatos[(variante, h)] = pred
    tablas, selectiva = {}, []
    for nombre_periodos in periodos:
        lista = PERIODOS[nombre_periodos]
        for h in horizontes_:
            variantes = {
                n: filas(pred, p, h, lista) for (n, hh), pred in candidatos.items() if hh == h
            }
            variantes = {n: df for n, df in variantes.items() if len(df)}
            tablas[(nombre_periodos, h)] = significancia(
                variantes, f"rev_{h}" if h in (5, 20) else None, N_BOOT, max(10, h)
            )
        for (n, h), pred in candidatos.items():
            for cobertura in COBERTURAS:
                fila = resumen_apuestas(apuestas(pred, p, h, lista, cobertura), bloque=max(10, h))
                selectiva.append({"periodos": nombre_periodos, "variante": n, "h": h,
                                  "cobertura": cobertura, **fila})  # fmt: skip
    return tablas, pd.DataFrame(selectiva)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Reglas a 5 y 20 ruedas y senal selectiva neta de costos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--nombre", required=True, help="Carpeta de salida en experimentos/.")
    parser.add_argument("--snapshot", default=DEFAULT_DIR)
    parser.add_argument("--limpiar", action="store_true", help="Precios sin saltos (limpieza).")
    parser.add_argument("--periodos", nargs="+", choices=sorted(PERIODOS),
                        default=list(PERIODOS_DEFAULT))  # fmt: skip
    parser.add_argument("--horizontes", nargs="+", type=int, default=list(HORIZONTES))
    parser.add_argument("--corrida", help="Corrida de busqueda cuyas predicciones se suman.")
    args = parser.parse_args(argv)

    salida = DIR_EXPERIMENTOS / args.nombre
    salida.mkdir(parents=True, exist_ok=True)
    snapshot = cargar_snapshot(args.snapshot, args.limpiar)
    corrida = DIR_EXPERIMENTOS / args.corrida if args.corrida else None
    tablas, selectiva = analizar(snapshot, args.periodos, tuple(args.horizontes), corrida)
    (salida / "opciones.json").write_text(json.dumps(vars(args), indent=2))
    for (nombre_periodos, h), tabla in tablas.items():
        tabla.to_csv(salida / f"ic_{nombre_periodos}_h{h}.csv")
        print(f"\n== {nombre_periodos}, horizonte {h} (intervalos al {NIVEL:.0%}) ==")
        print(_texto(legible(tabla, f"rev_{h}")))
    selectiva.to_csv(salida / "selectiva.csv", index=False)
    print("\n== Senal selectiva: solo compras, neto de costos ==")
    print(_texto(selectiva.round(4)))


if __name__ == "__main__":
    main()
