"""Simulacion dia por dia: reentrenar, predecir y operar carteras como en produccion.

Para cada rueda ``d`` desde ``--desde`` hasta ``--hasta``, con datos hasta el
cierre de ``d`` (nunca posteriores):

1. Reentrena LSTM / XGBoost / Transformer con el gate de promocion de
   produccion (todos los dias, solo en la ultima rueda de la semana o una vez
   por mes, segun ``--reentreno``) y los modelos macro y contexto el 1 de cada
   mes. Ver ``entrenamiento``.
   Cada version promovida queda guardada en ``artefactos/versiones/``.
2. Predice con los modelos vigentes para todos los tickers del snapshot
   (como el paper trading diario) y lo guarda en ``predicciones.jsonl``. En
   los dias de rebalanceo calcula ademas el backtest en vivo de cada modelo
   para los tickers del panel lider (``backtests.jsonl``).
3. Cada ``--rebalanceo`` ruedas, rebalancea al cierre de ``d`` cada cartera:
   una por senal de cada modelo, una de Black-Litterman por modelo y perfil,
   una de consenso por perfil, las de "mejor modelo por ticker" (senal y
   Black-Litterman siguiendo en cada ticker al modelo con mejor accuracy
   ajustada del backtest en vivo) y los benchmarks. Ver ``mesa``.
4. Valoriza todas las carteras al cierre y lo agrega a ``valores.jsonl``.

Al terminar escribe ``movimientos.csv`` (cada compra / venta con lo que decian
los modelos, ver ``movimientos``) y rehace las carteras sin comision en
``rearmados/sin-comision/`` (con su propio ``movimientos.csv``), para ver
cuanto se comen las comisiones; el reporte compara las dos.

La corrida se puede cortar y retomar (o extender con un ``--hasta`` posterior):
el estado se guarda al final de cada dia en ``estado.json`` y los modelos
vigentes en ``artefactos/``. Las carteras se pueden rehacer sin reentrenar con
``python -m src.simulacion.rearmar``.

Uso:
    python -m src.simulacion.simulador --nombre junio --desde 2026-06-01 --reentreno semanal
    python -m src.simulacion.simulador --nombre prueba --desde 2026-09-01 --hasta 2026-09-15
    python -m src.simulacion.simulador --nombre contexto --desde 2026-01-02 --reentreno mensual \
        --snapshot simulacion/snapshot-collector
    python -m src.simulacion.simulador --nombre tr-relativo --desde 2026-06-01 --reentreno semanal \
        --transformer '{"target": "relativo", "universo": "panel", "n_semillas": 5}'
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.arima_trend import ArimaTrendModel
from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.config import settings
from src.data import MACRO_COLUMN, attach_macro_feature
from src.data_quality import is_stale
from src.errors import ApiMlError
from src.simulacion import movimientos
from src.simulacion.cartera import CAPITAL_INICIAL_DEFAULT, COMISION_DEFAULT
from src.simulacion.entrenamiento import (
    FRECUENCIAS_REENTRENO,
    ModeloContexto,
    ModeloMacro,
    build_modelos_diarios,
    histories_at,
    macro_retrain_due,
    reentreno_due,
)
from src.simulacion.estrategias import backtest_sin_neutrales
from src.simulacion.mesa import MODELOS_CON_BACKTEST, Mesa, Predicciones, ResultadoRueda
from src.simulacion.snapshot import DEFAULT_DIR as SNAPSHOT_DIR
from src.simulacion.snapshot import Snapshot
from src.trend_common import backtest_predict_df, centrar_predicciones

RESULTADOS_DIR = "simulacion/resultados"
REBALANCEO_DEFAULT = 5
# Rearmado que se hace al terminar cada corrida (mismas predicciones, sin comision).
SIN_COMISION = "sin-comision"
# Modelos cuyo backtest se mide solo sobre predicciones con vista (ver
# ``backtest_sin_neutrales``).
MODELOS_SIN_NEUTRALES = ("macro", "contexto")


@dataclass
class Config:
    nombre: str
    desde: str
    hasta: str
    rebalanceo: int = REBALANCEO_DEFAULT
    capital: float = CAPITAL_INICIAL_DEFAULT
    comision: float = COMISION_DEFAULT
    reentreno: str = "diario"
    history_days: int = settings.history_days
    snapshot_dir: str = SNAPSHOT_DIR
    # Campos de TransformerTrainConfig a cambiar (vacio = el de produccion).
    transformer: dict = field(default_factory=dict)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _quiet(fn, *args, **kwargs):
    """Los ``fit`` imprimen cada epoca; en la simulacion solo interesa el resumen."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def guardar_rueda(out: Path, resultado: ResultadoRueda) -> None:
    """Agrega lo operado y valorizado en una rueda a los ``.jsonl`` de ``out``."""
    fecha = resultado.valores["fecha"]
    append_jsonl(out / "elecciones.jsonl", resultado.elecciones)
    append_jsonl(
        out / "pesos.jsonl",
        [{"fecha": fecha, "cartera": n, "pesos": p} for n, p in resultado.objetivos.items()],
    )
    append_jsonl(out / "valores.jsonl", [resultado.valores])
    append_jsonl(out / "operaciones.jsonl", resultado.operaciones)
    append_jsonl(out / "errores.jsonl", resultado.errores)


class Simulacion:
    def __init__(self, config: Config, snapshot: Snapshot):
        self.config = config
        self.snapshot = snapshot
        self.out = Path(RESULTADOS_DIR) / config.nombre
        self.artefactos = self.out / "artefactos"
        self.modelos = build_modelos_diarios(self.artefactos, config.transformer)
        self.macro = ModeloMacro(snapshot, self.artefactos / "macro.pkl")
        self.contexto = ModeloContexto(snapshot, self.artefactos / "contexto.pkl")
        self.arima = ArimaTrendModel()
        self.mesa = Mesa(snapshot, config.capital, config.comision, config.rebalanceo)
        self.ultimo_dia: str | None = None

    # ------------------------------------------------------------------ #
    # Estado
    # ------------------------------------------------------------------ #
    @property
    def _estado_path(self) -> Path:
        return self.out / "estado.json"

    def guardar_estado(self) -> None:
        estado = {"ultimo_dia": self.ultimo_dia, **self.mesa.to_dict()}
        tmp = self._estado_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(estado, indent=2, default=str))
        tmp.replace(self._estado_path)

    def cargar_estado(self) -> bool:
        if not self._estado_path.exists():
            return False
        estado = json.loads(self._estado_path.read_text())
        self.ultimo_dia = estado["ultimo_dia"]
        self.mesa.cargar(estado)
        for modelo in self.modelos:
            modelo.resume()
        self.macro.resume()
        self.contexto.resume()
        return True

    # ------------------------------------------------------------------ #
    # Entrenamiento
    # ------------------------------------------------------------------ #
    def _model_paths(self) -> dict[str, Path]:
        mensuales = {"macro": self.macro.model_path, "contexto": self.contexto.model_path}
        return {m.name: m.model_path for m in self.modelos} | mensuales

    def archivar(self, filas: list[dict]) -> None:
        """Copia cada version promovida a ``artefactos/versiones/<modelo>/<fecha>``.

        Con todas las versiones guardadas se puede rehacer cualquier prediccion
        o backtest de la corrida sin volver a entrenar.
        """
        paths = self._model_paths()
        for fila in filas:
            if not fila.get("promoted"):
                continue
            path = paths[fila["model"]]
            destino = self.artefactos / "versiones" / fila["model"]
            destino.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destino / f"{fila['fecha']}{path.suffix}")

    def bootstrap(self, previo: pd.Timestamp) -> None:
        """Modelos vigentes al arrancar: entrenados con datos hasta la rueda previa."""
        histories = histories_at(self.snapshot, previo, self.config.history_days)
        filas = []
        for modelo in self.modelos:
            filas.append({"fecha": previo.date(), **_quiet(modelo.bootstrap, histories)})
        for mensual in (self.macro, self.contexto):
            filas.append({"fecha": previo.date(), **_quiet(mensual.retrain, previo, force=True)})
        self.archivar(filas)
        append_jsonl(self.out / "promociones.jsonl", filas)
        for fila in filas:
            print(f"  [bootstrap] {fila['model']}: {fila.get('directional_accuracy', fila)}")

    def reentrenar(self, dia: pd.Timestamp) -> list[dict]:
        histories = histories_at(self.snapshot, dia, self.config.history_days)
        return [{"fecha": dia.date(), **_quiet(m.retrain, histories)} for m in self.modelos]

    # ------------------------------------------------------------------ #
    # Prediccion
    # ------------------------------------------------------------------ #
    def _history_with_macro(self, symbol: str, dia: pd.Timestamp) -> pd.DataFrame:
        days = self.config.history_days
        rate = self.snapshot.rate_until(dia)
        if rate is not None:
            rate = rate.iloc[-days:]
        return attach_macro_feature(self.snapshot.history(symbol, dia, days), MACRO_COLUMN, rate)

    def _backtest(self, name: str, model, df: pd.DataFrame, horizon: int) -> dict | None:
        try:
            if name in MODELOS_SIN_NEUTRALES:
                return backtest_sin_neutrales(model, df, horizon)
            bt = backtest_predict_df(model, df, horizon)
        except (ApiMlError, ValueError, ArithmeticError):
            return None
        return {k: v for k, v in bt.items() if k != "series"} if bt else None

    def predecir(
        self, dia: pd.Timestamp, con_backtest: set[str]
    ) -> tuple[Predicciones, Predicciones]:
        """Predicciones del dia (modelo -> ticker -> respuesta de ``predict_df``)
        y backtests en vivo (modelo -> ticker -> metricas).

        Solo se predice un ticker si tuvo rueda ese dia (si no, la prediccion
        repetiria la del ultimo dia que cotizo). ``con_backtest`` son los
        tickers para los que se calcula ademas el backtest en vivo de todos los
        modelos (solo hace falta en los dias de rebalanceo). Como en
        produccion, solo el de ``MODELOS_CON_BACKTEST`` viaja dentro de la
        prediccion (y pondera el consenso); el resto se usa solo para elegir
        el mejor modelo por ticker. Las predicciones de modelos con target
        relativo se centran con el promedio del panel lider del dia
        (``centrar_predicciones``).
        """
        self.macro.set_today(dia)
        self.contexto.set_today(dia)
        vigentes = {m.name: m.vigente for m in self.modelos}
        vigentes["arima"] = self.arima
        vigentes["macro"] = self.macro.vigente
        vigentes["contexto"] = self.contexto.vigente
        predicciones: Predicciones = {name: {} for name in vigentes}
        backtests: Predicciones = {name: {} for name in vigentes}
        for symbol in self.snapshot.tickers:
            df = self._history_with_macro(symbol, dia)
            if df.empty or df.index[-1] != dia or is_stale(df):
                continue
            for name, model in vigentes.items():
                if model is None:
                    continue
                try:
                    result = _quiet(model.predict_df, df)
                except (ApiMlError, ValueError, ArithmeticError):
                    continue
                result.update(symbol=symbol, model=name, model_version=model.version)
                if symbol in con_backtest:
                    bt = self._backtest(name, model, df, int(result["horizon_days"]))
                    if bt is not None:
                        backtests[name][symbol] = bt
                        if name in MODELOS_CON_BACKTEST:
                            result["backtest"] = bt
                predicciones[name][symbol] = result
        for name, model in vigentes.items():
            config = getattr(model, "config", None)
            if getattr(config, "target", None) == "relativo":
                predicciones[name] = centrar_predicciones(
                    predicciones[name], config.neutral_band, PANEL_LIDER_TICKERS
                )
        return predicciones, backtests

    # ------------------------------------------------------------------ #
    def procesar_dia(self, dia: pd.Timestamp, siguiente: pd.Timestamp | None) -> None:
        inicio = time.time()
        promociones = []
        reentrena = reentreno_due(self.config.reentreno, dia, siguiente)
        if reentrena:
            promociones = self.reentrenar(dia)

        rebalancea = self.mesa.toca_rebalancear()
        universo = set(self.mesa.precios(dia)) if rebalancea else set()
        predicciones, backtests = self.predecir(dia, universo)

        filas = [
            {"fecha": dia.date(), **{k: v for k, v in p.items() if k != "backtest"}}
            for preds in predicciones.values()
            for p in preds.values()
        ]
        append_jsonl(self.out / "predicciones.jsonl", filas)
        append_jsonl(
            self.out / "backtests.jsonl",
            [
                {"fecha": dia.date(), "model": m, "symbol": t, **bt}
                for m, por_ticker in backtests.items()
                for t, bt in por_ticker.items()
            ],
        )

        guardar_rueda(self.out, self.mesa.operar(dia, predicciones, backtests))

        if siguiente is not None and macro_retrain_due(dia, siguiente):
            for mensual in (self.macro, self.contexto):
                promociones.append({"fecha": dia.date(), **_quiet(mensual.retrain, dia)})
        self.archivar(promociones)
        append_jsonl(self.out / "promociones.jsonl", promociones)

        self.ultimo_dia = dia.date().isoformat()
        self.guardar_estado()

        promovidos = [p["model"] for p in promociones if p.get("promoted")]
        print(
            f"{dia.date()}  {time.time() - inicio:5.1f}s  predicciones={len(filas)}  "
            f"{'reentreno' if reentrena else '-'}  promovidos={promovidos or '-'}  "
            f"{'rebalanceo' if rebalancea else ''}"
        )

    def correr(self) -> None:
        cfg = self.config
        calendario = self.snapshot.trading_days()
        desde, hasta = pd.Timestamp(cfg.desde), pd.Timestamp(cfg.hasta)
        dias = [d for d in calendario if desde <= d <= hasta]
        if not dias:
            raise SystemExit("no hay ruedas en el rango pedido")

        self.out.mkdir(parents=True, exist_ok=True)
        config_path = self.out / "config.json"
        if self.cargar_estado():
            guardada = json.loads(config_path.read_text())
            ignorar = {"hasta", "snapshot"}
            actual = asdict(cfg)
            # Corridas anteriores a la opcion "transformer" no la guardaban.
            guardada.setdefault("transformer", {})
            if any(guardada.get(k) != v for k, v in actual.items() if k not in ignorar):
                raise SystemExit(
                    f"{self.out} ya tiene una corrida con otra configuracion: usar otro --nombre"
                )
            print(f"Retomando {cfg.nombre} despues de {self.ultimo_dia}")
        else:
            snapshot = self.snapshot.meta["downloaded_at"]
            config_path.write_text(json.dumps({**asdict(cfg), "snapshot": snapshot}, indent=2))
            previas = calendario[calendario < dias[0]]
            print(f"Bootstrap de modelos con datos hasta {previas[-1].date()}...")
            self.bootstrap(previas[-1])

        for dia in dias:
            if self.ultimo_dia is not None and dia.date().isoformat() <= self.ultimo_dia:
                continue
            siguientes = calendario[calendario > dia]
            siguiente = siguientes[0] if len(siguientes) else None
            self.procesar_dia(dia, siguiente)
        self.cerrar()
        print(f"\nResultados en {self.out}")

    def cerrar(self) -> None:
        """Movimientos de la corrida y, si se cobro comision, las carteras sin comision."""
        movimientos.escribir(self.out, self.snapshot)
        if self.config.comision > 0:
            # Import diferido: ``rearmar`` importa de este modulo.
            from src.simulacion.rearmar import rearmar

            rearmar(self.out, self.out / "rearmados" / SIN_COMISION, self.snapshot, comision=0.0)
            movimientos.escribir(self.out, self.snapshot, SIN_COMISION)


def ultima_rueda_completa(snapshot: Snapshot) -> str:
    """Ultima rueda anterior al dia de la descarga (la del dia puede estar incompleta)."""
    descarga = pd.Timestamp(snapshot.meta["downloaded_at"][:10])
    dias = snapshot.trading_days()
    return dias[dias < descarga][-1].date().isoformat()


def silenciar_warnings() -> None:
    # ARIMA y GARCH avisan de cada ajuste que no converge del todo: ruido en una
    # corrida de cientos de ajustes por dia.
    warnings.filterwarnings("ignore", module="statsmodels")
    warnings.filterwarnings("ignore", module="arch")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Simulacion historica con reentrenamiento.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--nombre", required=True, help="Nombre de la corrida (carpeta).")
    parser.add_argument("--desde", required=True, help="Primera rueda simulada (YYYY-MM-DD).")
    parser.add_argument("--hasta", help="Ultima rueda (default: la ultima completa del snapshot).")
    parser.add_argument("--rebalanceo", type=int, default=REBALANCEO_DEFAULT)
    parser.add_argument("--capital", type=float, default=CAPITAL_INICIAL_DEFAULT)
    parser.add_argument("--comision", type=float, default=COMISION_DEFAULT)
    parser.add_argument("--reentreno", choices=FRECUENCIAS_REENTRENO, default="diario")
    parser.add_argument("--snapshot", default=SNAPSHOT_DIR)
    parser.add_argument(
        "--transformer",
        type=json.loads,
        default={},
        help="Variante del Transformer: JSON con campos de TransformerTrainConfig.",
    )
    args = parser.parse_args(argv)
    silenciar_warnings()

    snapshot = Snapshot.load(args.snapshot)
    config = Config(
        nombre=args.nombre,
        desde=args.desde,
        hasta=args.hasta or ultima_rueda_completa(snapshot),
        rebalanceo=args.rebalanceo,
        capital=args.capital,
        comision=args.comision,
        reentreno=args.reentreno,
        snapshot_dir=args.snapshot,
        transformer=args.transformer,
    )
    Simulacion(config, snapshot).correr()


if __name__ == "__main__":
    main()
