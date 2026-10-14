"""Datos de panel para los modelos que miran todas las acciones del dia juntas.

Arma, a partir de los historicos por ticker (los mismos ``histories`` que
reciben los demas modelos), tablas alineadas ``[fecha x ticker]`` de precios y
volumen, con una mascara de las celdas validas:

- **Universo**: ``"panel"`` (panel lider) o ``"amplio"`` (panel lider + otras
  acciones argentinas con historia, hoy solo TECO2). Quedan afuera los CEDEARs,
  las especies en dolares (sufijo ``D``) y los futuros (OIL, GOLD): se mueven
  por razones ajenas al mercado local.
- **Moneda**: ``"usd"`` divide open/high/low/close por el CCL de ese dia
  (saca la deriva de la inflacion y la devaluacion; ver ``ccl_en``); ``"ars"``
  los deja en pesos. El volumen queda en nominales. Se usa el CCL y no el
  oficial porque es el tipo de cambio al que cotizan las acciones (arbitraje
  con los ADRs): con cepo, deflactar por el oficial mete la brecha en los
  retornos (correlacion 0,46 con sus cambios) y los saltos de devaluacion.
- **Liquidez**: una celda es valida si el ticker tiene dato ese dia y en las
  ultimas ``ventana_liquidez`` ruedas no estuvo quieto (precio sin cambio o
  volumen 0) mas de ``umbral_quieto`` de las veces. Mirar solo hacia atras
  garantiza que la mascara de una fecha no cambia con datos posteriores.
- **Sin relleno**: un ticker sin rueda ese dia queda en NaN y fuera de la
  mascara; no se repite el precio anterior (eso inventaria retornos 0).

La limpieza de saltos sin ajustar (splits, canjes) se hace antes, sobre el
snapshot (``src.simulacion.limpieza``, opcion ``--limpiar`` de las busquedas).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS

PRECIOS = ["open", "high", "low", "close"]
CAMPOS = [*PRECIOS, "volume"]

# Sector de cada accion: agrupa las que se mueven por los mismos motivos.
SECTORES: dict[str, str] = {
    "GGAL": "financiero", "BMA": "financiero", "BBAR": "financiero",
    "SUPV": "financiero", "VALO": "financiero", "BYMA": "financiero",
    "YPFD": "petroleo", "PAMP": "petroleo",
    "CEPU": "utilities", "EDN": "utilities", "TRAN": "utilities", "TGSU2": "utilities",
    "TGNO4": "utilities", "METR": "utilities", "ECOG": "utilities",
    "ALUA": "materiales", "TXAR": "materiales", "LOMA": "materiales",
    "CRES": "agro_holding", "COME": "agro_holding",
    "TECO2": "telecom",
}  # fmt: skip

UNIVERSOS: dict[str, list[str]] = {
    "panel": list(PANEL_LIDER_TICKERS),
    "amplio": [*PANEL_LIDER_TICKERS, "TECO2"],
}

VENTANA_LIQUIDEZ = 60
UMBRAL_QUIETO = 0.3


def universo(nombre: str) -> list[str]:
    """Tickers del universo ``nombre`` (``"panel"`` o ``"amplio"``)."""
    if nombre not in UNIVERSOS:
        raise ValueError(f"universo {nombre!r} desconocido ({sorted(UNIVERSOS)})")
    return list(UNIVERSOS[nombre])


def ccl_en(ccl: pd.Series, fechas: pd.DatetimeIndex) -> pd.Series:
    """Ultimo CCL conocido en cada fecha (NaN antes del primer dato).

    Se usa el CCL **del mismo dia**, sin el retraso de ``macro_frame`` (en la
    simulacion, ``Snapshot.ccl_hasta(hasta)``): el CCL de un dia se conoce al
    cierre de ese dia, y el retraso de produccion cubre la cache del
    data-colector, no la fecha del dato. Dividir por un CCL de dos dias antes
    agrega ruido (vol. diaria del panel en USD 4,17% vs 3,99%). Aca solo se
    alinea hacia atras: nunca se usa un CCL posterior a la fecha.
    """
    serie = ccl.dropna().sort_index()
    serie = serie[~serie.index.duplicated(keep="last")]
    return serie.reindex(serie.index.union(fechas)).ffill().reindex(fechas)


def en_dolares(df: pd.DataFrame, ccl: pd.Series) -> pd.DataFrame:
    """Precios divididos por el CCL de cada rueda; sin CCL conocido, NaN."""
    tipo_cambio = ccl_en(ccl, df.index)
    out = df.copy()
    for col in PRECIOS:
        if col in out:
            out[col] = out[col] / tipo_cambio
    return out


def ruedas_activas(
    df: pd.DataFrame, ventana: int = VENTANA_LIQUIDEZ, umbral: float = UMBRAL_QUIETO
) -> pd.Series:
    """True en las ruedas donde, mirando las ultimas ``ventana`` (inclusive),
    el ticker estuvo quieto (cierre sin cambio o volumen 0) a lo sumo
    ``umbral`` de las veces. Las primeras ruedas cuentan con lo que hay."""
    close = df["close"]
    quieta = close.diff().eq(0)
    if "volume" in df:
        quieta |= df["volume"].fillna(0).le(0)
    fraccion = quieta.astype(float).rolling(ventana, min_periods=1).mean()
    return fraccion.le(umbral)


@dataclass
class Panel:
    """Tablas ``[fecha x ticker]`` alineadas; ``mascara`` marca las celdas validas."""

    campos: dict[str, pd.DataFrame]
    mascara: pd.DataFrame
    moneda: str

    @property
    def fechas(self) -> pd.DatetimeIndex:
        return self.mascara.index

    @property
    def tickers(self) -> list[str]:
        return list(self.mascara.columns)

    def __getitem__(self, campo: str) -> pd.DataFrame:
        return self.campos[campo]

    def sectores(self) -> list[str]:
        return [SECTORES.get(t, "otro") for t in self.tickers]

    def cobertura(self) -> pd.DataFrame:
        """Por ticker: primera y ultima fecha valida y fraccion de fechas validas."""
        m = self.mascara
        return pd.DataFrame(
            {
                "desde": m.idxmax().where(m.any()),
                "hasta": m[::-1].idxmax().where(m.any()),
                "validas": m.mean().round(3),
            }
        )


def armar_panel(
    histories: dict[str, pd.DataFrame],
    tickers: list[str],
    ccl: pd.Series | None = None,
    moneda: str = "usd",
    ventana_liquidez: int = VENTANA_LIQUIDEZ,
    umbral_quieto: float = UMBRAL_QUIETO,
) -> Panel:
    """Panel de ``tickers`` (los que esten en ``histories``) en ``moneda``.

    Las fechas son la union de las ruedas de esos tickers. Una celda es valida
    si hay dato, el ticker esta activo (ver ``ruedas_activas``), el precio es
    positivo y, en dolares, hay CCL conocido.
    """
    if moneda not in ("usd", "ars"):
        raise ValueError(f"moneda {moneda!r} desconocida (usd o ars)")
    if moneda == "usd" and ccl is None:
        raise ValueError("para armar el panel en dolares hace falta el CCL")
    presentes = [t for t in tickers if t in histories and not histories[t].empty]
    if not presentes:
        raise ValueError("ninguno de los tickers pedidos tiene historia")

    fechas = pd.DatetimeIndex(sorted(set().union(*(histories[t].index for t in presentes))))
    campos = {c: pd.DataFrame(np.nan, index=fechas, columns=presentes) for c in CAMPOS}
    mascara = pd.DataFrame(False, index=fechas, columns=presentes)
    for t in presentes:
        df = histories[t].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        activa = ruedas_activas(df, ventana_liquidez, umbral_quieto)
        if moneda == "usd":
            df = en_dolares(df, ccl)
        for c in CAMPOS:
            if c in df:
                campos[c].loc[df.index, t] = df[c].astype(float)
        valida = activa & df["close"].gt(0) & df["close"].notna()
        mascara.loc[df.index, t] = valida.to_numpy()
    return Panel(campos=campos, mascara=mascara, moneda=moneda)
