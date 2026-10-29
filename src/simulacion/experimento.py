"""Experimento rapido de calidad de prediccion: compara variantes de modelos sin operar carteras.

La simulacion completa (``simulador``) tarda ~40 minutos por periodo porque
reentrena cada semana y opera todas las carteras. Para elegir entre variantes
de un modelo alcanza con algo mas barato: entrenar una vez al inicio de cada
periodo (con datos hasta ese dia), predecir todas las ruedas del periodo para
el panel lider sin reentrenar y medir que tan bien ordena los tickers.

Metricas por variante y periodo (retornos log a ``horizon`` ruedas):

- ``ic``: promedio entre ruedas de la correlacion de rangos (Spearman) entre
  prediccion y retorno real de los tickers del panel. Es la metrica principal:
  las carteras compran los tickers mejor rankeados, no el nivel del mercado.
- ``ic_t``: estadistico t del IC, contando ``ruedas / horizon`` observaciones
  independientes (las ventanas de ruedas consecutivas se solapan).
- ``hit_rel``: acierto del signo de prediccion y retorno, ambos relativos al
  promedio del panel ese dia.
- ``ls``: retorno medio del tercio con mejor prediccion menos el del peor tercio (%).
- ``pct_alza`` / ``pct_baja``, ``precision_alza`` (que fraccion de las "alza"
  subio) y ``precision_baja`` (que fraccion de las "baja" bajo) contra
  ``base_up`` (fraccion de casos que subieron): miden el sesgo de la senal
  absoluta. Una senal solo aporta si su precision supera a su tasa base
  (``base_up`` para las alza, ``1 - base_up`` para las baja).
- ``ret_alza`` / ``ret_baja`` / ``ret_medio``: retorno log medio real (%) de
  los casos con senal "alza", "baja" y de todos (la magnitud que la precision
  no muestra: comprar las "alza" rinde ``ret_alza - ret_medio`` sobre el panel).

Uso (los resultados quedan en ``simulacion/experimentos/<nombre>/``):
    python -m src.simulacion.experimento --nombre transformer --variantes base relativo
    python -m src.simulacion.experimento --nombre transformer --resumen
    python -m src.simulacion.experimento --nombre limpio --variantes base --limpiar
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.config import settings
from src.data import MACRO_COLUMN, attach_macro_feature
from src.errors import ApiMlError
from src.lstm import TrendModel
from src.simulacion.entrenamiento import histories_at
from src.simulacion.limpieza import limpiar
from src.simulacion.snapshot import DEFAULT_DIR, Snapshot
from src.transformer import CONFIG_ORIGINAL, TransformerTrainConfig, TransformerTrendModel
from src.trend_common import centrar_predicciones, derive_trend_output
from src.xgb_trend import XGBTrendModel

DIR_EXPERIMENTOS = Path("simulacion/experimentos")
HORIZON = 5

# Periodos de evaluacion (desde, hasta): alcistas, bajistas y laterales.
PERIODOS = [
    ("2024-07-01", "2024-12-30"),
    ("2025-01-02", "2025-05-30"),
    ("2025-06-02", "2025-10-31"),
    ("2026-01-02", "2026-05-29"),
    ("2026-06-01", "2026-10-02"),
]


class ReversionModel:
    """Baseline sin entrenamiento: predice que el retorno de las ultimas
    ``horizon`` ruedas se revierte. Cualquier modelo deberia superarlo."""

    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        return {}

    def predict_df(self, df: pd.DataFrame) -> dict:
        close = df["close"].to_numpy(dtype=np.float64)
        if len(close) <= HORIZON:
            raise ApiMlError("historia insuficiente")
        pasado = float(np.log(close[-1] / close[-1 - HORIZON]))
        return derive_trend_output(df, -pasado, HORIZON)


class ReversionPanelModel(ReversionModel):
    """La misma regla de reversion, pero como modelo de panel (``predict_panel``,
    todos los tickers del dia juntos). Sirve para verificar que el camino de los
    modelos de panel mide igual que el de los modelos de a un ticker."""

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        resultados = {}
        for symbol, df in dfs.items():
            try:
                resultados[symbol] = self.predict_df(df)
            except ApiMlError:
                continue
        return resultados


def _transformer(**cambios) -> Callable[[], object]:
    # Variantes del Transformer original, comparables con corridas viejas.
    config = CONFIG_ORIGINAL | cambios
    return lambda: TransformerTrendModel(config=TransformerTrainConfig(**config))


# Variantes disponibles: nombre -> fabrica de un modelo sin entrenar.
_ROBUSTO = dict(
    target="relativo",
    universo="panel",
    loss="huber",
    clip_target=3.0,
    gap=HORIZON,
    banda_cuantil=0.5,
)
VARIANTES: dict[str, Callable[[], object]] = {
    "reversion": ReversionModel,
    "lstm": TrendModel,
    "xgboost": XGBTrendModel,
    "base": _transformer(),
    "relativo": _transformer(target="relativo"),
    "relativo_panel": _transformer(target="relativo", universo="panel"),
    "robusto": _transformer(**_ROBUSTO),
    "robusto_ic": _transformer(**_ROBUSTO, seleccion="ic"),
    "robusto_chico": _transformer(
        **_ROBUSTO, d_model=16, nhead=2, num_layers=1, dim_feedforward=32
    ),
    "robusto_ens": _transformer(**_ROBUSTO, n_semillas=5),
    "robusto_chico_ens": _transformer(
        **_ROBUSTO, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, n_semillas=5
    ),
}


def _con_macro(snapshot: Snapshot, df: pd.DataFrame, hasta: pd.Timestamp) -> pd.DataFrame:
    rate = snapshot.rate_until(hasta)
    return attach_macro_feature(df, MACRO_COLUMN, rate)


def _predecir_dia(modelo, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Predicciones de un dia: en un solo batch si el modelo tiene
    ``predict_panel`` (necesario con features relativas al panel)."""
    if hasattr(modelo, "predict_panel"):
        return modelo.predict_panel(dfs)
    resultados = {}
    for symbol, df in dfs.items():
        try:
            resultados[symbol] = modelo.predict_df(df)
        except ApiMlError:
            continue
    return resultados


def predecir_periodo(
    snapshot: Snapshot,
    fabrica: Callable[[], object],
    desde: str,
    hasta: str,
    tickers: list[str],
    days: int = settings.history_days,
    days_fit: int | None = None,
    reentrenar_cada: int | None = None,
) -> pd.DataFrame:
    """Entrena con datos hasta la rueda anterior a ``desde`` y predice cada
    rueda de ``[desde, hasta]`` para ``tickers``.

    ``days`` son las ruedas de historia de cada prediccion y, salvo que se pase
    ``days_fit``, tambien las del entrenamiento. Sin ``reentrenar_cada`` se
    entrena una sola vez; con ``reentrenar_cada = k``, un modelo nuevo cada
    ``k`` ruedas del periodo, con datos hasta la rueda anterior (como haria
    produccion con un reentreno periodico). Los targets de las ultimas ruedas
    de cada entrenamiento no se conocen todavia y quedan afuera solos.

    Devuelve una fila por (fecha, ticker) con el retorno log predicho y el real
    a ``HORIZON`` ruedas (solo los casos con retorno real conocido).
    """
    if reentrenar_cada is not None and reentrenar_cada < 1:
        raise ValueError("reentrenar_cada tiene que ser >= 1")
    ruedas = snapshot.trading_days()
    del_periodo = ruedas[(ruedas >= pd.Timestamp(desde)) & (ruedas <= pd.Timestamp(hasta))]

    def entrenar(dia: pd.Timestamp):
        corte = ruedas[ruedas < dia][-1]
        nuevo = fabrica()
        with contextlib.redirect_stdout(io.StringIO()):
            nuevo.fit(histories_at(snapshot, corte, days_fit or days))
        return nuevo

    modelo = entrenar(pd.Timestamp(desde))
    config = getattr(modelo, "config", None)
    relativo = getattr(config, "target", None) == "relativo"
    filas = []
    for i, dia in enumerate(del_periodo):
        if reentrenar_cada and i > 0 and i % reentrenar_cada == 0:
            modelo = entrenar(dia)
        dfs, reales = {}, {}
        for symbol in tickers:
            if symbol not in snapshot.prices:
                continue
            completo = snapshot.prices[symbol]
            pos = completo.index.searchsorted(dia)
            if pos >= len(completo) or completo.index[pos] != dia or pos + HORIZON >= len(completo):
                continue
            dfs[symbol] = _con_macro(snapshot, snapshot.history(symbol, dia, days), dia)
            close = completo["close"].to_numpy()
            reales[symbol] = float(np.log(close[pos + HORIZON] / close[pos]))
        del_dia = _predecir_dia(modelo, dfs)
        if relativo:
            # Como en la simulacion: la senal relativa se centra con el promedio del dia.
            del_dia = centrar_predicciones(del_dia, config.neutral_band)
        for symbol, result in del_dia.items():
            filas.append(
                {
                    "fecha": dia.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "pred": float(np.log(result["predicted_close"] / result["last_close"])),
                    "signal": result["signal"],
                    "real": reales[symbol],
                }
            )
    return pd.DataFrame(filas)


def metricas(df: pd.DataFrame) -> dict:
    """Metricas de calidad de un periodo (ver docstring del modulo)."""
    ics, ls = {}, []
    for fecha, g in df.groupby("fecha"):
        if len(g) < 5 or g["pred"].nunique() < 2:
            continue
        ics[fecha] = g["pred"].rank().corr(g["real"].rank())
        rango = g["pred"].rank(pct=True)
        ls.append(g["real"][rango > 2 / 3].mean() - g["real"][rango <= 1 / 3].mean())
    serie = pd.Series(ics).sort_index()
    desvio = serie.std()
    # Las ventanas de ruedas consecutivas se solapan: se cuentan len / HORIZON
    # observaciones independientes.
    ic_t = (
        float(serie.mean() / desvio * np.sqrt(len(serie) / HORIZON))
        if len(serie) > 1 and desvio > 0
        else 0.0
    )
    rel_pred = df["pred"] - df.groupby("fecha")["pred"].transform("mean")
    rel_real = df["real"] - df.groupby("fecha")["real"].transform("mean")
    alza, baja = df["signal"] == "alza", df["signal"] == "baja"
    return {
        "n": int(len(df)),
        "ic": round(float(serie.mean()), 4) if len(serie) else 0.0,
        "ic_t": round(ic_t, 2),
        "hit_rel": round(float((np.sign(rel_pred) == np.sign(rel_real)).mean()), 4),
        "ls": round(float(np.mean(ls)) * 100, 3) if ls else 0.0,
        "pct_alza": round(float(alza.mean()), 4),
        "pct_baja": round(float(baja.mean()), 4),
        "precision_alza": round(float((df["real"][alza] > 0).mean()), 4) if alza.any() else None,
        "precision_baja": round(float((df["real"][baja] < 0).mean()), 4) if baja.any() else None,
        "base_up": round(float((df["real"] > 0).mean()), 4),
        # Retorno log medio real (%) de los casos con cada senal y del total.
        "ret_alza": round(float(df["real"][alza].mean()) * 100, 3) if alza.any() else None,
        "ret_baja": round(float(df["real"][baja].mean()) * 100, 3) if baja.any() else None,
        "ret_medio": round(float(df["real"].mean()) * 100, 3) if len(df) else None,
    }


def correr(
    snapshot: Snapshot,
    nombre: str,
    variantes: list[str],
    periodos: list[tuple[str, str]] = PERIODOS,
    tickers: list[str] = PANEL_LIDER_TICKERS,
    base: Path = DIR_EXPERIMENTOS,
) -> list[dict]:
    """Corre cada variante en cada periodo y agrega una linea por par a
    ``<base>/<nombre>/resultados.jsonl`` (las predicciones van a
    ``predicciones/<variante>_<desde>.csv``)."""
    salida = base / nombre
    (salida / "predicciones").mkdir(parents=True, exist_ok=True)
    filas = []
    for variante in variantes:
        for desde, hasta in periodos:
            print(f"[{variante}] {desde} -> {hasta}", flush=True)
            df = predecir_periodo(snapshot, VARIANTES[variante], desde, hasta, tickers)
            df.to_csv(salida / "predicciones" / f"{variante}_{desde}.csv", index=False)
            fila = {"variante": variante, "desde": desde, "hasta": hasta, **metricas(df)}
            print(f"  {fila}", flush=True)
            with (salida / "resultados.jsonl").open("a") as f:
                f.write(json.dumps(fila) + "\n")
            filas.append(fila)
    return filas


def resumen(nombre: str, base: Path = DIR_EXPERIMENTOS) -> str:
    """Tabla de texto: IC por periodo y promedios por variante (la ultima
    corrida de cada par variante/periodo pisa a las anteriores)."""
    filas = [json.loads(linea) for linea in (base / nombre / "resultados.jsonl").open()]
    df = pd.DataFrame(filas).drop_duplicates(["variante", "desde"], keep="last")
    ic = df.pivot(index="variante", columns="desde", values="ic")
    ic.columns = [f"IC {c[:7]}" for c in ic.columns]
    medias = df.groupby("variante")[["ic", "hit_rel", "ls", "pct_alza", "pct_baja"]].mean()
    tabla = ic.join(medias.rename(columns=lambda c: f"{c} medio"))
    tabla["periodos IC>0"] = df.assign(pos=df["ic"] > 0).groupby("variante")["pos"].sum()
    return tabla.sort_values("ic medio", ascending=False).round(3).to_string()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara variantes de modelos por IC.")
    parser.add_argument("--nombre", required=True)
    parser.add_argument("--variantes", nargs="*", default=[], choices=sorted(VARIANTES))
    parser.add_argument("--snapshot", default=DEFAULT_DIR)
    parser.add_argument("--resumen", action="store_true", help="Solo imprimir el resumen.")
    parser.add_argument("--limpiar", action="store_true", help="Precios sin saltos (limpieza).")
    args = parser.parse_args()
    if args.variantes:
        snapshot = Snapshot.load(args.snapshot)
        if args.limpiar:
            snapshot = limpiar(snapshot)[0]
        correr(snapshot, args.nombre, args.variantes, PERIODOS, base=DIR_EXPERIMENTOS)
    print(resumen(args.nombre, base=DIR_EXPERIMENTOS))


if __name__ == "__main__":
    main()
