"""Busqueda de hiperparametros del Transformer (``src/transformer.py``) con Optuna.

Dos etapas, sobre un snapshot (nunca contra el collector):

1. **Busqueda** (``--trials``): TPE sobre la red y la ventana, con una semilla.
   La metrica es el acierto balanceado del walk-forward del propio
   entrenamiento (lo mismo que mira el gate de produccion). Cada candidato
   reporta despues de cada bloque del walk-forward y ``MedianPruner`` corta los
   que van por debajo de la mediana.
2. **Confirmacion** (``--top``): los mejores de la busqueda y la config actual
   de produccion se reevaluan con ``--semillas`` semillas y en cada fecha de
   ``--cortes`` (por default, el final del snapshot y fin de 2025: otra
   ventana de validacion, para no quedarse con el que tuvo suerte en una). La
   diferencia entre semillas es grande (IC 0,02-0,07 con la misma config),
   asi que solo cuenta lo que se sostiene aca.

Para que sea rapida:
- las ventanas (limpieza, dolares, riesgo pais, features) se arman una vez por
  ``(corte, ventana)`` antes de repartir el trabajo (``TransformerTrendModel.ventanas``);
- varios procesos en paralelo con pocos hilos de torch cada uno (la red es
  chica y un proceso no aprovecha los 16 nucleos);
- ``patience=3`` (la mejor epoca sale entre la 1 y la 4) y sin el reentreno
  final con todas las fechas (no cambia las metricas fuera de muestra).

Resultados en ``simulacion/optuna/<nombre>/``: ``estudio.db`` (retomable: los
trials ya hechos no se repiten), ``confirmacion.csv`` y ``resumen.md``.

Uso:
    python -m src.simulacion.optuna_transformer --nombre transformer-1
    python -m src.simulacion.optuna_transformer --nombre prueba --trials 4 --top 2 --semillas 2
    python -m src.simulacion.optuna_transformer --nombre transformer-1 --solo-confirmar
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
from dataclasses import fields
from pathlib import Path

import optuna
import pandas as pd

from src.config import settings
from src.simulacion.entrenamiento import histories_at, proveedor_snapshot
from src.simulacion.snapshot import Snapshot
from src.transformer import TransformerTrainConfig, TransformerTrendModel

DIR_OPTUNA = Path("simulacion/optuna")
SNAPSHOT_DEFAULT = "simulacion/snapshot-collector"
ESTUDIO = "transformer"
VENTANAS = (10, 20, 30)
# Lo que cambia respecto de produccion solo para buscar mas rapido.
BASE_BUSQUEDA = {"patience": 3, "reentrenar_completo": False}
SEMILLA = TransformerTrainConfig.seed
CORTES_DEFAULT = ("fin", "2025-12-30")
PROCESOS_DEFAULT = 5
HILOS_DEFAULT = 3

# (corte, ventana) -> ventanas por ticker. Se llena en el proceso principal antes
# de crear los trabajadores, que lo heredan (fork) sin volver a armarlo.
_DATOS: dict[tuple[str, int], dict] = {}


def sugerir(trial: optuna.Trial) -> dict:
    """Espacio de busqueda: la regularizacion (lo unico que importo en la del V2),
    el tamano de la red, el lote y la ventana."""
    d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "d_model": d_model,
        "dim_feedforward": 2 * d_model,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "window": trial.suggest_categorical("window", list(VENTANAS)),
    }


def config_de_params(params: dict) -> dict:
    """Config del Transformer a partir de los parametros de un trial."""
    return {**params, "dim_feedforward": 2 * params["d_model"]}


def config_actual() -> dict:
    """La config de produccion (defaults), con su paciencia."""
    return {"patience": TransformerTrainConfig.patience}


def preparar(snapshot: Snapshot, corte: pd.Timestamp, ventana: int) -> dict:
    """Ventanas de todos los tickers con datos hasta ``corte``, como al entrenar en produccion."""
    modelo = TransformerTrendModel(
        TransformerTrainConfig(window=ventana), proveedor=proveedor_snapshot(snapshot)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return modelo.ventanas(histories_at(snapshot, corte, settings.history_days))


def evaluar(config: dict, por_ticker: dict, semilla: int, reporte=None) -> dict[str, float]:
    """Metricas fuera de muestra del walk-forward de una config."""
    cfg = TransformerTrainConfig(**(BASE_BUSQUEDA | config | {"seed": semilla}))
    with contextlib.redirect_stdout(io.StringIO()):
        return TransformerTrendModel(cfg).fit_ventanas(por_ticker, reporte)


def objetivo(trial: optuna.Trial, corte: str) -> float:
    config = sugerir(trial)

    def reporte(k: int, metricas: dict) -> None:
        trial.report(metricas["val_acierto_balanceado"], k)
        if trial.should_prune():
            raise optuna.TrialPruned()

    metricas = evaluar(config, _DATOS[(corte, config["window"])], SEMILLA, reporte)
    trial.set_user_attr("ic", metricas.get("val_ic"))
    trial.set_user_attr("epocas", metricas.get("epocas"))
    return float(metricas["val_acierto_balanceado"])


def _storage(salida: Path) -> optuna.storages.RDBStorage:
    # Varios procesos escriben en el mismo SQLite: esperar el lock en vez de fallar.
    return optuna.storages.RDBStorage(
        f"sqlite:///{salida / 'estudio.db'}", engine_kwargs={"connect_args": {"timeout": 120}}
    )


def _trabajador(salida: Path, corte: str, n_trials: int, hilos: int, semilla: int) -> None:
    import torch

    torch.set_num_threads(hilos)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.load_study(
        study_name=ESTUDIO,
        storage=_storage(salida),
        sampler=optuna.samplers.TPESampler(seed=semilla, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    estudio.optimize(lambda t: objetivo(t, corte), n_trials=n_trials)


def _terminados(estudio: optuna.Study) -> int:
    estados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(t.state in estados for t in estudio.trials)


def buscar(salida: Path, corte: str, trials: int, procesos: int, hilos: int) -> optuna.Study:
    """Reparte los trials que faltan entre ``procesos`` trabajadores."""
    estudio = optuna.create_study(
        study_name=ESTUDIO, storage=_storage(salida), direction="maximize", load_if_exists=True
    )
    faltan = trials - _terminados(estudio)
    if faltan > 0:
        partes = [faltan // procesos + (i < faltan % procesos) for i in range(procesos)]
        ctx = mp.get_context("fork")
        trabajadores = [
            ctx.Process(target=_trabajador, args=(salida, corte, n, hilos, SEMILLA + i))
            for i, n in enumerate(partes)
            if n
        ]
        for p in trabajadores:
            p.start()
        for p in trabajadores:
            p.join()
    return optuna.load_study(study_name=ESTUDIO, storage=_storage(salida))


def candidatos(estudio: optuna.Study, top: int) -> dict[str, dict]:
    """La config actual y los ``top`` mejores trials completos."""
    completos = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    mejores = sorted(completos, key=lambda t: t.value, reverse=True)[:top]
    return {"actual": config_actual()} | {
        f"trial_{t.number}": config_de_params(t.params) for t in mejores
    }


def _evaluar_tarea(tarea: tuple[str, dict, str, int]) -> dict:
    nombre, config, corte, semilla = tarea
    ventana = config.get("window", TransformerTrainConfig.window)
    metricas = evaluar(config, _DATOS[(corte, ventana)], semilla)
    return {
        "candidato": nombre,
        "corte": corte,
        "semilla": semilla,
        "acierto_balanceado": metricas.get("val_acierto_balanceado"),
        "ic": metricas.get("val_ic"),
        "accuracy": metricas.get("val_directional_accuracy"),
        "epocas": metricas.get("epocas"),
    }


def _iniciar_hilos(hilos: int) -> None:
    import torch

    torch.set_num_threads(hilos)


def confirmar(
    configs: dict[str, dict], cortes: list[str], semillas: int, procesos: int, hilos: int
) -> pd.DataFrame:
    """Cada candidato con ``semillas`` semillas en cada corte, en paralelo."""
    tareas = [
        (nombre, config, corte, SEMILLA + s)
        for nombre, config in configs.items()
        for corte in cortes
        for s in range(semillas)
    ]
    ctx = mp.get_context("fork")
    with ctx.Pool(procesos, initializer=_iniciar_hilos, initargs=(hilos,)) as pool:
        return pd.DataFrame(pool.map(_evaluar_tarea, tareas, chunksize=1))


def resumen(filas: pd.DataFrame) -> pd.DataFrame:
    """Por candidato: medias y desvios entre semillas (todos los cortes juntos),
    la media de cada corte y la diferencia pareada (misma semilla y corte)
    contra la config actual."""
    agrupado = filas.groupby("candidato")
    tabla = pd.DataFrame(
        {
            "acierto_balanceado": agrupado["acierto_balanceado"].mean(),
            "desvio": agrupado["acierto_balanceado"].std(),
            "ic": agrupado["ic"].mean(),
            "epocas": agrupado["epocas"].median(),
        }
    )
    por_corte = filas.pivot_table(index="candidato", columns="corte", values="acierto_balanceado")
    tabla = tabla.join(por_corte.add_prefix("bal_"))
    if "actual" in tabla.index:
        base = filas[filas["candidato"] == "actual"].set_index(["corte", "semilla"])
        dif = filas.join(base[["acierto_balanceado"]], on=["corte", "semilla"], rsuffix="_actual")
        dif = dif["acierto_balanceado"] - dif["acierto_balanceado_actual"]
        tabla["vs_actual"] = dif.groupby(filas["candidato"]).mean()
        tabla["gana_en"] = (dif > 0).groupby(filas["candidato"]).mean()
    return tabla.sort_values("acierto_balanceado", ascending=False).round(4)


def escribir_resumen(
    salida: Path, tabla: pd.DataFrame, configs: dict[str, dict], estudio: optuna.Study
) -> str:
    estados = pd.Series([t.state.name for t in estudio.trials]).value_counts().to_dict()
    mejor = tabla.index[0]
    campos = {f.name for f in fields(TransformerTrainConfig)}
    config = {k: v for k, v in configs[mejor].items() if k in campos}
    texto = "\n".join(
        [
            "# Busqueda de hiperparametros del Transformer",
            "",
            f"Trials: {estados}. Metrica: acierto balanceado del walk-forward.",
            "",
            tabla.to_string(),
            "",
            f"Mejor: `{mejor}`",
            "",
            "```json",
            json.dumps(config, indent=2),
            "```",
            "",
            "`vs_actual`: diferencia media pareada (misma semilla y corte) contra la config",
            "actual; `gana_en`: fraccion de pares en que le gana. Con diferencias de menos",
            "de ~0,01 o `gana_en` cerca de 0,5, quedarse con la actual.",
        ]
    )
    (salida / "resumen.md").write_text(texto + "\n")
    return texto


def _cortes(snapshot: Snapshot, cortes: list[str]) -> dict[str, pd.Timestamp]:
    fin = snapshot.trading_days()[-1]
    return {c: (fin if c == "fin" else pd.Timestamp(c)) for c in cortes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda de hiperparametros del Transformer.")
    parser.add_argument("--nombre", required=True)
    parser.add_argument("--snapshot", default=SNAPSHOT_DEFAULT)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--top", type=int, default=8, help="Trials que pasan a la confirmacion.")
    parser.add_argument("--semillas", type=int, default=5, help="Semillas de la confirmacion.")
    parser.add_argument(
        "--cortes",
        nargs="+",
        default=list(CORTES_DEFAULT),
        help="Fechas de datos de la confirmacion ('fin' = ultima rueda del snapshot); "
        "la busqueda usa la primera.",
    )
    parser.add_argument("--procesos", type=int, default=PROCESOS_DEFAULT)
    parser.add_argument(
        "--hilos", type=int, default=HILOS_DEFAULT, help="Hilos de torch por proceso."
    )
    parser.add_argument("--solo-confirmar", action="store_true", help="Sin trials nuevos.")
    args = parser.parse_args()

    salida = DIR_OPTUNA / args.nombre
    salida.mkdir(parents=True, exist_ok=True)
    snapshot = Snapshot.load(args.snapshot)
    cortes = _cortes(snapshot, args.cortes)
    print(f"Armando ventanas ({len(cortes)} cortes x {len(VENTANAS)} ventanas)...")
    for nombre, fecha in cortes.items():
        for ventana in VENTANAS:
            _DATOS[(nombre, ventana)] = preparar(snapshot, fecha, ventana)

    corte_busqueda = args.cortes[0]
    if args.solo_confirmar:
        estudio = optuna.load_study(study_name=ESTUDIO, storage=_storage(salida))
    else:
        print(f"Busqueda: {args.trials} trials en {args.procesos} procesos...")
        estudio = buscar(salida, corte_busqueda, args.trials, args.procesos, args.hilos)

    configs = candidatos(estudio, args.top)
    n = f"{len(configs)} candidatos x {args.semillas} semillas x {len(cortes)} cortes"
    print(f"Confirmacion: {n}...")
    filas = confirmar(configs, list(cortes), args.semillas, args.procesos, args.hilos)
    filas.to_csv(salida / "confirmacion.csv", index=False)
    print(escribir_resumen(salida, resumen(filas), configs, estudio))


if __name__ == "__main__":
    main()
