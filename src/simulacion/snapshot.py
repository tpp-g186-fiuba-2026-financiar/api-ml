"""Snapshot local de los datos que usa la simulacion.

La simulacion reentrena y predice cientos de veces: si cada paso le pegara a
Yahoo o al data-colector de produccion, seria lenta, no reproducible (Yahoo
corrige series viejas) y cargaria servicios ajenos. Por eso se descarga todo
una sola vez a disco y la simulacion lee solamente de ahi.

Mismas fuentes que el reentrenamiento de produccion (``retrain.yml``):

- historicos OHLCV: ``fetch_history`` (Yahoo por default, ``DATA_SOURCE``);
- listado de tickers y serie TNX: data-colector (``DATA_COLLECTOR_URL``);
- series macro de Argentina (modelo ``macro``): data-colector.

Uso:
    python -m src.simulacion.snapshot
    python -m src.simulacion.snapshot --dir simulacion/snapshot --days 1825
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data import (
    fetch_available_tickers,
    fetch_history,
    fetch_history_yahoo,
    fetch_interest_rate,
)
from src.errors import ApiMlError
from src.macro_data import SERIES, build_macro_frame, fetch_collector_series

DEFAULT_DIR = "simulacion/snapshot"
# Ruedas a descargar por ticker: alcanza para las 750 que usa produccion
# antes de la fecha de inicio. No subir de 1825 si los precios vienen de Yahoo:
# con mas, ``fetch_history_yahoo`` pide ``range=max`` y Yahoo responde con velas
# mensuales en vez de diarias. Desde el collector (``DATA_SOURCE=collector``) se
# puede pedir mas (tiene ~10 anios diarios); el Merval, que siempre sale de
# Yahoo, se baja igual con ``DEFAULT_DAYS`` como maximo.
DEFAULT_DAYS = 1825

_PRECIOS = "precios"
_TNX = "tnx.csv"
_MERVAL = "merval.csv"
# Indice de referencia (benchmark), en pesos.
MERVAL_YAHOO = "^MERV"
_MACRO = "macro"
_META = "meta.json"


def _normalize_index(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Indice por fecha (sin hora): Yahoo marca cada rueda con la hora de apertura."""
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def fetch_merval(days: int) -> pd.DataFrame:
    """Indice Merval de Yahoo (sin el sufijo ``.BA`` que agrega ``to_yahoo_symbol``)."""
    original = settings.market_suffix
    settings.market_suffix = ""
    try:
        return fetch_history_yahoo(MERVAL_YAHOO, days)
    finally:
        settings.market_suffix = original


def download(directory: str | Path, days: int = DEFAULT_DAYS) -> dict:
    """Descarga todo al directorio y devuelve la metadata guardada."""
    root = Path(directory)
    (root / _PRECIOS).mkdir(parents=True, exist_ok=True)
    (root / _MACRO).mkdir(parents=True, exist_ok=True)

    tickers = fetch_available_tickers()
    print(f"{len(tickers)} tickers en data-colector")
    descargados: list[str] = []
    fallidos: dict[str, str] = {}
    for symbol in tickers:
        try:
            df = fetch_history(symbol, days)
        except ApiMlError as exc:
            print(f"  [skip] {symbol}: {exc}")
            fallidos[symbol] = str(exc)
            continue
        _normalize_index(df).to_csv(root / _PRECIOS / f"{symbol}.csv", index_label="date")
        descargados.append(symbol)
        rango = f"{df.index[0].date()} -> {df.index[-1].date()}"
        print(f"  [ok]   {symbol}: {len(df)} ruedas ({rango})")

    tnx_ok = True
    try:
        tnx = fetch_interest_rate(settings.macro_rate_source, settings.macro_rate_series, days)
        _normalize_index(tnx).rename("value").to_csv(root / _TNX, index_label="date")
        print(f"  [ok]   tasa {settings.macro_rate_source}/{settings.macro_rate_series}")
    except ApiMlError as exc:
        tnx_ok = False
        print(f"  [warn] tasa {settings.macro_rate_source}/{settings.macro_rate_series}: {exc}")

    merval_ok = True
    try:
        merval = fetch_merval(min(days, DEFAULT_DAYS))
        _normalize_index(merval).to_csv(root / _MERVAL, index_label="date")
        print(f"  [ok]   merval: {len(merval)} ruedas")
    except ApiMlError as exc:
        merval_ok = False
        print(f"  [warn] merval: {exc}")

    macro_ok: list[str] = []
    for name, (route, series, _lag) in SERIES.items():
        try:
            values = fetch_collector_series(route, series)
        except ApiMlError as exc:
            print(f"  [warn] macro {name}: {exc}")
            continue
        values.rename("value").to_csv(root / _MACRO / f"{name}.csv", index_label="date")
        macro_ok.append(name)
        print(f"  [ok]   macro {name}: {len(values)}")

    meta = {
        "downloaded_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": settings.data_source,
        "data_collector_url": settings.data_collector_url,
        "days": days,
        "available_tickers": tickers,
        "tickers": descargados,
        "failed": fallidos,
        "macro_rate": tnx_ok,
        "merval": merval_ok,
        "macro_series": macro_ok,
    }
    (root / _META).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def _read_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return frame["value"].astype(float)


@dataclass
class Snapshot:
    """Datos descargados por ``download``, con cortes "como si fuera tal fecha".

    Todo acceso pasa por una fecha de corte (``hasta``): nunca devuelve una
    rueda posterior, que es la garantia de que la simulacion no mira el futuro.
    """

    prices: dict[str, pd.DataFrame]
    tickers: list[str]
    rate: pd.Series | None
    macro_raw: dict[str, pd.Series]
    meta: dict
    merval: pd.Series | None = None

    @classmethod
    def load(cls, directory: str | Path) -> Snapshot:
        root = Path(directory)
        meta_path = root / _META
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no hay snapshot en {root}: correr `python -m src.simulacion.snapshot`"
            )
        meta = json.loads(meta_path.read_text())
        prices = {}
        for symbol in meta["tickers"]:
            path = root / _PRECIOS / f"{symbol}.csv"
            prices[symbol] = pd.read_csv(path, index_col="date", parse_dates=["date"]).astype(float)
        rate = _read_series(root / _TNX) if (root / _TNX).exists() else None
        macro_raw = {
            name: _read_series(root / _MACRO / f"{name}.csv") for name in meta["macro_series"]
        }
        merval = None
        if (root / _MERVAL).exists():
            merval = pd.read_csv(root / _MERVAL, index_col="date", parse_dates=["date"])["close"]
        return cls(
            prices=prices,
            tickers=list(meta["tickers"]),
            rate=rate,
            macro_raw=macro_raw,
            meta=meta,
            merval=merval.astype(float) if merval is not None else None,
        )

    def trading_days(self, reference: str = "GGAL") -> pd.DatetimeIndex:
        """Calendario de ruedas (el de un ticker liquido del panel lider)."""
        return self.prices[reference].index

    def history(self, symbol: str, hasta: pd.Timestamp, days: int | None = None) -> pd.DataFrame:
        """Ultimas ``days`` ruedas de ``symbol`` hasta ``hasta`` inclusive."""
        df = self.prices[symbol]
        df = df.loc[df.index <= hasta]
        if days and len(df) > days:
            df = df.iloc[-days:]
        return df

    def rate_until(self, hasta: pd.Timestamp) -> pd.Series | None:
        if self.rate is None:
            return None
        return self.rate.loc[self.rate.index <= hasta]

    def macro_frame(self, hasta: pd.Timestamp) -> pd.DataFrame:
        """Tabla macro como la veria ``todays_macro_frame`` ese dia (con retrasos)."""
        raw = {name: s.loc[s.index <= hasta] for name, s in self.macro_raw.items()}
        raw = {name: s for name, s in raw.items() if not s.empty}
        return build_macro_frame(raw, end=hasta)

    @staticmethod
    def _last_until(series: pd.Series | None, fecha: pd.Timestamp) -> float | None:
        if series is None:
            return None
        values = series.loc[series.index <= fecha].dropna()
        return float(values.iloc[-1]) if not values.empty else None

    def merval_on(self, fecha: pd.Timestamp) -> float | None:
        return self._last_until(self.merval, fecha)

    def ccl_on(self, fecha: pd.Timestamp) -> float | None:
        """Dolar CCL publicado a ``fecha`` (para expresar las carteras en dolares)."""
        return self._last_until(self.macro_raw.get("ccl"), fecha)

    def ccl_hasta(self, hasta: pd.Timestamp) -> pd.Series | None:
        """Serie diaria del CCL hasta ``hasta`` inclusive, **sin** el retraso de
        ``macro_frame``: cada dato con su propia fecha. Es para pasar precios a
        dolares (``src.panel.datos``): el CCL de un dia se conoce al cierre de ese
        dia; el retraso de produccion cubre la cache del data-colector, no la
        fecha del dato."""
        serie = self.macro_raw.get("ccl")
        if serie is None:
            return None
        return serie.loc[serie.index <= hasta]

    def close_on(self, symbol: str, fecha: pd.Timestamp) -> float | None:
        """Ultimo cierre conocido de ``symbol`` a ``fecha`` (None si no cotizaba)."""
        df = self.prices.get(symbol)
        if df is None:
            return None
        close = df["close"].loc[df.index <= fecha]
        return float(close.iloc[-1]) if not close.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el snapshot de datos de la simulacion.")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="Directorio destino.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Ruedas por ticker.")
    args = parser.parse_args()
    meta = download(args.dir, args.days)
    print(f"\nSnapshot guardado en {args.dir}: {len(meta['tickers'])} tickers")


if __name__ == "__main__":
    main()
