"""Busqueda de hiperparametros del Transformer V2 con Optuna, sin fuga hacia lo que se reporta.

Diseno (ver ``notas/analisis-transformer-v2.md``):

- **Modelo**: el V2 relativo (target sin la media del dia, early stopping por
  la perdida de validacion), con toda la historia disponible hasta
  ``HISTORIA`` ruedas.
- **Folds de busqueda** (``FOLDS``): walk-forward 2019-2023 (los 5 periodos
  previos y los 3 primeros de eleccion). En cada fold se entrena con datos
  hasta la rueda anterior y se predice el periodo sin reentrenar.
- **Objetivo**: IC del modelo **menos** IC de la regla de reversion del mismo
  horizonte, sobre las mismas filas, promediado en los folds. Asi el
  horizonte (5, 10 o 20 ruedas) entra en la busqueda sin premiar al problema
  que da IC mas alto: se premia lo que el modelo agrega sobre la regla.
- **Optuna**: TPE multivariado, pruning por mediana despues de cada fold,
  estudio en SQLite (se corta y se retoma), tope de tiempo (``--horas``).
- **Cierre** (``cerrar``): el top ``TOP_K`` se reevalua con otras semillas
  (la mejor de una busqueda sale inflada), se calcula la **PBO** (probabilidad
  de sobreajuste por CSCV sobre la matriz trials x folds), la configuracion
  elegida se evalua **una vez** en 2024-2025 (``EVALUACION``) contra el
  ensamble y la reversion, y se escribe un informe PDF con los graficos.

Uso (en el host con GPU):
    python -m src.simulacion.optuna_v2 buscar --nombre optuna-v2 --horas 5 --dispositivo cuda
    python -m src.simulacion.optuna_v2 cerrar --nombre optuna-v2 --dispositivo cuda
    python -m src.simulacion.optuna_v2 noche --nombre optuna-v2 --horas 5 --dispositivo cuda
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.errors import ApiMlError
from src.panel.ensamble import EnsambleConfig, EnsamblePanelModel
from src.panel.gbm import PanelGBMConfig
from src.panel.transformer_v2 import HISTORIA_V2, TransformerV2Config, TransformerV2Model
from src.simulacion.busqueda import (
    DIR_EXPERIMENTOS,
    PERIODOS_ELECCION,
    PERIODOS_PREVIOS,
    cargar_snapshot,
    proveedor_snapshot,
)
from src.simulacion.direccion import REFERENCIA_REVERSION, referencias, retornos_usd
from src.simulacion.experimento import predecir_periodo
from src.simulacion.significancia import significancia
from src.simulacion.snapshot import Snapshot

SNAPSHOT = "simulacion/snapshot-collector"
FOLDS = PERIODOS_PREVIOS + PERIODOS_ELECCION[:3]  # 2019-07 a 2023-10
EVALUACION = PERIODOS_ELECCION[3:]  # 2024-01 a 2025-10
HORIZONTES = (5, 10, 20)
HISTORIA = HISTORIA_V2
BASE = {"target": "sin_media", "seleccion": "perdida", "n_semillas": 1}
TOP_K = 5
SEMILLAS_REEVALUACION = (1, 2)
SEMILLAS_EVALUACION = 3
NOMBRE_ESTUDIO = "v2"


# --------------------------------------------------------------------------- #
# Espacio y evaluacion de una configuracion
# --------------------------------------------------------------------------- #
def espacio(trial: optuna.Trial) -> dict:
    """Config del V2 propuesta por ``trial`` (``BASE`` + hiperparametros)."""
    return BASE | {
        "horizonte": trial.suggest_categorical("horizonte", list(HORIZONTES)),
        "ventana": trial.suggest_categorical("ventana", [1, 5, 10, 20]),
        "d_model": trial.suggest_categorical("d_model", [16, 32, 64]),
        "nhead": trial.suggest_categorical("nhead", [2, 4]),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [32, 64, 128]),
        "capas_temporales": trial.suggest_int("capas_temporales", 1, 2),
        "capas_cruzadas": trial.suggest_int("capas_cruzadas", 0, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1.0, log=True),
        "perdida": trial.suggest_categorical("perdida", ["huber", "bce"]),
        "macro": trial.suggest_categorical("macro", [False, True]),
        "vida_media_anios": trial.suggest_categorical("vida_media_anios", [1.0, 2.0, 4.0]),
    }


def _ic_por_fecha(df: pd.DataFrame, columna: str) -> pd.Series:
    def uno(g: pd.DataFrame) -> float:
        if len(g) < 3 or g[columna].nunique() < 2 or g["futuro"].nunique() < 2:
            return np.nan
        return float(g[columna].rank().corr(g["futuro"].rank()))

    return df.groupby("fecha")[[columna, "futuro"]].apply(uno)


def con_retornos(pred: pd.DataFrame, retornos: pd.DataFrame) -> pd.DataFrame:
    """Predicciones (``fecha``, ``symbol``, ``pred``) con el retorno en USD real
    (``futuro``) y pasado (``pasado``) al horizonte de ``retornos``."""
    pred = pred[["fecha", "symbol", "pred"]].assign(fecha=lambda d: d["fecha"].astype(str))
    df = pred.merge(retornos, on=["fecha", "symbol"])
    return df.dropna(subset=["futuro", "pasado"]).reset_index(drop=True)


def ic_contra_reversion(df: pd.DataFrame) -> dict:
    """IC del modelo y de la reversion (``-pasado``) sobre las mismas filas."""
    df = df.assign(rev=-df["pasado"])
    ic = _ic_por_fecha(df, "pred").mean()
    ic_rev = _ic_por_fecha(df, "rev").mean()
    return {
        "ic": float(ic),
        "ic_rev": float(ic_rev),
        "dif": float(ic - ic_rev),
        "fechas": int(df["fecha"].nunique()),
    }


class Evaluador:
    """Entrena y predice el V2 en un periodo y mide el IC contra la reversion."""

    def __init__(
        self,
        snapshot: Snapshot,
        dispositivo: str = "cpu",
        historia: int = HISTORIA,
        extra: dict | None = None,
    ):
        """``extra``: campos que pisan los de cada config (``--prueba``: pocas epocas)."""
        self.snapshot = snapshot
        self.dispositivo = dispositivo
        self.historia = historia
        self.extra = extra or {}
        self.proveedor = proveedor_snapshot(snapshot)
        self._retornos: dict[int, pd.DataFrame] = {}

    def retornos(self, horizonte: int) -> pd.DataFrame:
        if horizonte not in self._retornos:
            self._retornos[horizonte] = retornos_usd(self.snapshot, horizonte)
        return self._retornos[horizonte]

    def predecir(self, fabrica, desde: str, hasta: str, horizonte: int) -> pd.DataFrame:
        with contextlib.redirect_stdout(io.StringIO()):
            pred = predecir_periodo(
                self.snapshot, fabrica, desde, hasta, PANEL_LIDER_TICKERS, days_fit=self.historia
            )
        if pred.empty:
            raise ApiMlError(f"sin predicciones en {desde} - {hasta}")
        return con_retornos(pred, self.retornos(horizonte))

    def v2(self, config: dict, seed: int = 0):
        def fabrica():
            return TransformerV2Model(
                TransformerV2Config(**(config | self.extra | {"seed": seed})),
                proveedor=self.proveedor,
                dispositivo=self.dispositivo,
            )

        return fabrica

    def fold(self, config: dict, desde: str, hasta: str, seed: int = 0) -> dict:
        df = self.predecir(self.v2(config, seed), desde, hasta, config["horizonte"])
        return ic_contra_reversion(df)


# --------------------------------------------------------------------------- #
# Busqueda
# --------------------------------------------------------------------------- #
def objetivo(evaluador: Evaluador, folds: list[tuple[str, str]] = FOLDS):
    def funcion(trial: optuna.Trial) -> float:
        config = espacio(trial)
        difs, ics = [], []
        for i, (desde, hasta) in enumerate(folds):
            r = evaluador.fold(config, desde, hasta)
            difs.append(r["dif"])
            ics.append(r["ic"])
            trial.set_user_attr("difs", difs)
            trial.set_user_attr("ics", ics)
            trial.report(float(np.mean(difs)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(difs))

    return funcion


def estudio(salida: Path) -> optuna.Study:
    salida.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=NOMBRE_ESTUDIO,
        storage=f"sqlite:///{salida / 'estudio.db'}",
        direction="maximize",
        # Sin semilla fija: al retomar el estudio no se repiten las mismas propuestas.
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
        load_if_exists=True,
    )


def _avisar(s: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Una linea por trial terminado, para seguir la busqueda."""
    valor = f"{trial.value:+.4f}" if trial.value is not None else "-"
    duracion = trial.duration.total_seconds() if trial.duration else 0.0
    mejor = f"{s.best_value:+.4f}" if completos(s) else "-"
    print(
        f"{time.strftime('%H:%M:%S')} trial {trial.number} {trial.state.name.lower()} "
        f"valor {valor} folds {len(trial.user_attrs.get('difs', []))} "
        f"h{trial.params.get('horizonte')} ({duracion:.0f} s)  mejor {mejor}",
        flush=True,
    )


def _liberar_gpu(*_) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch siempre esta en api_ml
        pass


def buscar(
    salida: Path,
    evaluador: Evaluador,
    horas: float | None,
    trials: int | None,
    folds: list[tuple[str, str]] = FOLDS,
) -> optuna.Study:
    s = estudio(salida)
    s.optimize(
        objetivo(evaluador, folds),
        n_trials=trials,
        timeout=horas * 3600 if horas else None,
        catch=(ApiMlError, RuntimeError, ValueError),
        callbacks=[_avisar, _liberar_gpu],
        gc_after_trial=True,
    )
    return s


# --------------------------------------------------------------------------- #
# Cierre: reevaluacion, PBO, evaluacion e informe
# --------------------------------------------------------------------------- #
def completos(s: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    estado = optuna.trial.TrialState.COMPLETE
    return [t for t in s.trials if t.state == estado and t.value is not None]


def config_de(trial: optuna.trial.FrozenTrial) -> dict:
    return espacio(optuna.trial.FixedTrial(trial.params))


def pbo(matriz: np.ndarray) -> float | None:
    """Probabilidad de sobreajuste del backtest (CSCV, Bailey et al.) de una
    matriz ``[configuracion, fold]``: en cada particion de los folds en dos
    mitades se elige la mejor configuracion en una mitad y se mira su rango en
    la otra; PBO = fraccion de particiones en que queda por debajo de la mediana."""
    n, s = matriz.shape
    if n < 2 or s < 2 or s % 2:
        return None
    logits = []
    for adentro in itertools.combinations(range(s), s // 2):
        afuera = [j for j in range(s) if j not in adentro]
        mejor = int(np.argmax(matriz[:, list(adentro)].mean(axis=1)))
        fuera = matriz[:, afuera].mean(axis=1)
        rango = (fuera < fuera[mejor]).sum() + 1  # 1 = peor, n = mejor
        w = rango / (n + 1)
        logits.append(np.log(w / (1 - w)))
    return float(np.mean(np.array(logits) <= 0))


def reevaluar(
    s: optuna.Study,
    evaluador: Evaluador,
    folds: list[tuple[str, str]] = FOLDS,
    k: int = TOP_K,
    semillas: tuple[int, ...] = SEMILLAS_REEVALUACION,
) -> pd.DataFrame:
    """El top ``k`` de la busqueda con otras semillas: dif media por semilla."""
    filas = []
    for trial in sorted(completos(s), key=lambda t: t.value, reverse=True)[:k]:
        config = config_de(trial)
        por_semilla = [trial.value]
        for seed in semillas:
            por_semilla.append(
                float(np.mean([evaluador.fold(config, d, h, seed)["dif"] for d, h in folds]))
            )
        filas.append(
            {
                "trial": trial.number,
                "horizonte": config["horizonte"],
                "busqueda": trial.value,
                "media": float(np.mean(por_semilla)),
                "desvio": float(np.std(por_semilla, ddof=1)),
                "semillas": len(por_semilla),
                "config": json.dumps(config),
            }
        )
    return pd.DataFrame(filas).sort_values("media", ascending=False).reset_index(drop=True)


def evaluar(
    config: dict, evaluador: Evaluador, periodos: list[tuple[str, str]] = EVALUACION
) -> pd.DataFrame:
    """La configuracion elegida (ensamble de ``SEMILLAS_EVALUACION`` redes) y el
    ensamble Ridge + reversion del mismo horizonte, en ``periodos``, con
    intervalos (``significancia``) contra la reversion."""
    h = config["horizonte"]
    v2 = config | {"n_semillas": SEMILLAS_EVALUACION}
    ridge = PanelGBMConfig(tipo="ridge", horizonte=h)

    def ensamble():
        return EnsamblePanelModel(EnsambleConfig(0.5, ridge, None), proveedor=evaluador.proveedor)

    partes: dict[str, list[pd.DataFrame]] = {"v2_optuna": [], "ensamble": []}
    for desde, hasta in periodos:
        for nombre, fabrica in (("v2_optuna", evaluador.v2(v2)), ("ensamble", ensamble)):
            partes[nombre].append(evaluador.predecir(fabrica, desde, hasta, h).assign(desde=desde))
    variantes = {n: pd.concat(d, ignore_index=True) for n, d in partes.items()}
    variantes |= referencias(variantes["v2_optuna"])
    return significancia(variantes, REFERENCIA_REVERSION, bloque=max(10, h))


def cerrar(
    salida: Path,
    evaluador: Evaluador,
    folds: list[tuple[str, str]] = FOLDS,
    periodos: list[tuple[str, str]] = EVALUACION,
    k: int = TOP_K,
    semillas: tuple[int, ...] = SEMILLAS_REEVALUACION,
) -> dict:
    """Reevaluacion del top, PBO, evaluacion 2024-2025 e informe. Devuelve el
    resumen (tambien en ``resumen.json``) con la config elegida."""
    s = estudio(salida)
    hechos = completos(s)
    if not hechos:
        raise SystemExit(f"{salida} no tiene trials completos")
    matriz = np.array([t.user_attrs["difs"] for t in hechos])
    top = reevaluar(s, evaluador, folds, k, semillas)
    top.to_csv(salida / "top.csv", index=False)
    elegida = json.loads(top.iloc[0]["config"])
    evaluacion = evaluar(elegida, evaluador, periodos)
    evaluacion.to_csv(salida / "evaluacion.csv")
    resumen = {
        "trials": len(s.trials),
        "completos": len(hechos),
        "podados": sum(t.state == optuna.trial.TrialState.PRUNED for t in s.trials),
        "fallidos": sum(t.state == optuna.trial.TrialState.FAIL for t in s.trials),
        "pbo": pbo(matriz),
        "elegida": elegida,
        "mejor_busqueda": {"trial": s.best_trial.number, "valor": s.best_value},
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2))
    # Config para ``un_modelo``: la elegida, con las 5 semillas de las otras corridas.
    para_simular = {k: v for k, v in elegida.items() if k != "n_semillas"} | {"historia": HISTORIA}
    (salida / "mejor.json").write_text(json.dumps(para_simular))
    from src.simulacion.reporte_optuna import generar

    generar(salida)
    return resumen


# --------------------------------------------------------------------------- #
def _paso(nombre: str, funcion, registro: Path) -> object | None:
    """Corre un paso de la cadena sin cortarla si falla (lo deja en el registro)."""
    inicio = time.monotonic()
    print(f"\n=== {nombre} ===", flush=True)
    try:
        resultado = funcion()
        estado = "ok"
    except Exception:  # noqa: BLE001 - la cadena sigue con el proximo paso
        resultado = None
        estado = "fallo"
        traceback.print_exc()
    linea = f"{time.strftime('%H:%M:%S')} {nombre}: {estado} ({time.monotonic() - inicio:.0f} s)"
    with registro.open("a") as fh:
        fh.write(linea + "\n")
    print(linea, flush=True)
    return resultado


# Version en miniatura de toda la cadena (``--prueba``): verifica que no se rompa nada
# antes de dejarla corriendo de noche.
PRUEBA = {
    "folds": [("2023-01-02", "2023-01-20"), ("2023-06-01", "2023-06-20")],
    "evaluacion": [("2024-01-02", "2024-01-19")],
    "simulaciones": (("2025", "2025-01-02", "2025-01-31"),),
    "extra": {"epocas": 2, "paciencia": 1, "n_semillas": 1},
    "historia": 600,
}
SIMULACIONES = (("2025", "2025-01-02", "2025-12-30"), ("2026", "2026-01-02", None))


def noche(
    salida: Path,
    evaluador: Evaluador,
    horas: float | None,
    snapshot_dir: str,
    dispositivo: str,
    trials: int | None = None,
    prueba: bool = False,
) -> None:
    """Busqueda, cierre y simulaciones de 2025 y 2026 con la config elegida."""
    from src.simulacion import un_modelo

    registro = salida / "noche.txt"
    salida.mkdir(parents=True, exist_ok=True)
    folds = PRUEBA["folds"] if prueba else FOLDS
    periodos = PRUEBA["evaluacion"] if prueba else EVALUACION
    reeval = {"k": 2, "semillas": (1,)} if prueba else {}
    _paso("busqueda", lambda: buscar(salida, evaluador, horas, trials, folds), registro)
    resumen = _paso(
        "cierre", lambda: cerrar(salida, evaluador, folds, periodos, **reeval), registro
    )
    if not resumen:
        return
    config = json.loads((salida / "mejor.json").read_text())
    if prueba:
        config |= PRUEBA["extra"] | {"historia": PRUEBA["historia"]}
    simulaciones = PRUEBA["simulaciones"] if prueba else SIMULACIONES
    for nombre, desde, hasta in simulaciones:

        def simular(desde=desde, hasta=hasta, nombre=nombre):
            snap = Snapshot.load(snapshot_dir)
            cfg = un_modelo.ConfigUnModelo(
                nombre=f"{salida.name}-{nombre}",
                desde=desde,
                hasta=hasta or un_modelo.ultima_rueda_completa(snap),
                config=config,
                cuantil=0.15,
                snapshot_dir=snapshot_dir,
            )
            return un_modelo.correr(cfg, dispositivo)

        _paso(f"simulacion {nombre}", simular, registro)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Busqueda de hiperparametros del Transformer V2 con Optuna.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("accion", choices=["buscar", "cerrar", "noche"])
    parser.add_argument("--nombre", required=True, help="Carpeta en simulacion/experimentos/.")
    parser.add_argument("--snapshot", default=SNAPSHOT)
    parser.add_argument("--dispositivo", default="cpu")
    parser.add_argument("--horas", type=float, help="Tope de tiempo de la busqueda.")
    parser.add_argument("--trials", type=int, help="Tope de trials de la busqueda.")
    parser.add_argument("--prueba", action="store_true", help="Cadena en miniatura (noche).")
    args = parser.parse_args(argv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    salida = DIR_EXPERIMENTOS / args.nombre
    snapshot = cargar_snapshot(args.snapshot, limpiar=True)
    if args.prueba:
        evaluador = Evaluador(snapshot, args.dispositivo, PRUEBA["historia"], PRUEBA["extra"])
    else:
        evaluador = Evaluador(snapshot, args.dispositivo)
    if args.accion == "buscar":
        s = buscar(salida, evaluador, args.horas, args.trials)
        print(f"{len(s.trials)} trials; mejor {s.best_value:.4f} (trial {s.best_trial.number})")
    elif args.accion == "cerrar":
        print(json.dumps(cerrar(salida, evaluador), indent=2))
    else:
        if not (args.horas or args.trials):
            parser.error("noche necesita --horas o --trials")
        noche(
            salida, evaluador, args.horas, args.snapshot, args.dispositivo, args.trials, args.prueba
        )


if __name__ == "__main__":
    main()
