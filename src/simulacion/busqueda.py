"""Busqueda de configuraciones de modelos con varias semillas y periodos separados.

Complementa a ``experimento`` (misma forma de medir: entrenar al inicio de cada
periodo, predecir sin reentrenar, IC sobre el panel lider) con lo que hace
falta para elegir sin engañarse:

- **Periodos separados**: se elige solo con ``PERIODOS_ELECCION`` (2022-2025);
  ``PERIODOS_CONFIRMACION`` (2026) se miran una vez, con la configuracion ya
  elegida.
- **Varias semillas por configuracion**: el desvio del IC entre semillas es el
  piso de ruido; una diferencia mas chica que eso no cuenta.
- **Ensamble gratis**: se guardan las predicciones de cada semilla, y el
  resumen calcula tambien el IC de su promedio (equivale a ``n_semillas``).
- **Paralelo y retomable**: cada (variante, semilla, periodo) es un trabajo
  independiente; los que ya estan en ``resultados.jsonl`` no se repiten.
- **Opciones de corrida** (``--limpiar``: precios sin saltos, ver ``limpieza``;
  ``--dispositivo cuda``: entrenar en GPU): quedan en ``opciones.json`` y no
  se pueden cambiar al retomar, para no mezclar resultados.
- **Cualquier modelo**: la clave ``"modelo"`` de una config elige el modelo de
  ``MODELOS`` (default ``"transformer"``); el resto de la config, con la
  semilla, va a su constructor (salvo ``"historia"``: ruedas de entrenamiento,
  default las de produccion, y ``"reentrenar_cada"``: ruedas entre reentrenos
  dentro del periodo, default uno solo al inicio). Los modelos de panel (con ``predict_panel``)
  reciben todos los tickers del dia juntos.
- **Referencias**: ``--referencias`` suma al resumen variantes de corridas ya
  hechas (``REFERENCIAS`` por default: Transformer actual y baselines de
  ``limpio``), sin volver a correrlas. Solo si se corrieron con la misma limpieza.

Planes:
- ``ablacion``: formulaciones del problema (target, universo, loss, recorte,
  gap, seleccion), sacando un componente por vez de la version completa.
- ``hiper``: muestra aleatoria de hiperparametros de la red sobre una
  formulacion base (``--base``).
- ``v2``: Transformer V2 contra el original (``PLAN_V2``; ``make buscar-transformer-v2``).
- ``rapido``: ajustes e hiperparametros del V2, una semilla (``PLAN_V2_RAPIDO``;
  ``make probar-transformer-v2``).
- ``r2``: ronda 2 del plan de mejora, filtro con modelos baratos: cabeza
  relativa, cabeza de mercado, meta-labeling y dos cabezas (``PLAN_R2``;
  ``make filtro-r2``).
- ``r2-transformer``: el V2 como cabeza relativa contra el ensamble
  (``PLAN_R2_TRANSFORMER``; ``make transformer-r2``, GPU).
- ``config``: configuraciones puntuales (``--configs``), p. ej. para confirmar.

Uso:
    python -m src.simulacion.busqueda --nombre ablacion --plan ablacion --procesos 14
    python -m src.simulacion.busqueda --nombre hiper --plan hiper --base '{"target": "relativo"}'
    python -m src.simulacion.busqueda --nombre limpio --plan ablacion --limpiar \
        --dispositivo cuda --procesos 3
    python -m src.simulacion.busqueda --nombre ablacion --resumen
    python -m src.simulacion.busqueda --nombre panel --plan config --limpiar --semillas 1 \
        --configs '{"rev_panel": {"modelo": "reversion_panel"}}' --referencias
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.panel.dos_cabezas import DosCabezasModel
from src.panel.ensamble import EnsambleConfig, EnsamblePanelModel
from src.panel.gbm import PanelGBMConfig, PanelGBMModel
from src.panel.mercado import MercadoConfig, MercadoModel
from src.panel.meta import MetaReversionConfig, MetaReversionModel
from src.panel.transformer import PanelTransformerConfig, PanelTransformerModel
from src.panel.transformer_v2 import CONFIG_ANTERIOR, TransformerV2Config, TransformerV2Model
from src.simulacion import por_ticker
from src.simulacion.experimento import (
    DIR_EXPERIMENTOS,
    VARIANTES,
    ReversionPanelModel,
    metricas,
    predecir_periodo,
)
from src.simulacion.limpieza import limpiar as limpiar_snapshot
from src.simulacion.snapshot import DEFAULT_DIR, Snapshot
from src.transformer import TransformerTrainConfig, TransformerTrendModel

PERIODOS_ELECCION = [
    ("2022-07-01", "2022-11-30"),
    ("2023-01-02", "2023-05-31"),
    ("2023-06-01", "2023-10-31"),
    ("2024-01-02", "2024-05-31"),
    ("2024-07-01", "2024-12-30"),
    ("2025-01-02", "2025-05-30"),
    ("2025-06-02", "2025-10-31"),
]
PERIODOS_CONFIRMACION = [
    ("2026-01-02", "2026-05-29"),
    ("2026-06-01", "2026-10-02"),
]
# Anteriores a los de eleccion, nunca usados para elegir nada: segunda muestra
# independiente para las reglas y modelos baratos de la ronda 3.
PERIODOS_PREVIOS = [
    ("2019-07-01", "2019-11-29"),
    ("2020-01-02", "2020-05-29"),
    ("2020-07-01", "2020-11-30"),
    ("2021-01-04", "2021-05-31"),
    ("2021-06-01", "2021-10-29"),
]
PERIODOS = {
    "eleccion": PERIODOS_ELECCION,
    "previos": PERIODOS_PREVIOS,
    "confirmacion": PERIODOS_CONFIRMACION,
    "todos": PERIODOS_ELECCION + PERIODOS_CONFIRMACION,
}
SEMILLA_BASE = 42

_COMPLETO = dict(
    target="relativo", universo="panel", loss="huber", clip_target=3.0, gap=5, banda_cuantil=0.5
)


def _sin(clave: str, **extra) -> dict:
    return {k: v for k, v in (_COMPLETO | extra).items() if k != clave}


PLAN_ABLACION: dict[str, dict] = {
    "absoluto": {},
    "relativo_todos": {"target": "relativo"},
    "relativo_panel": {"target": "relativo", "universo": "panel"},
    "completo": dict(_COMPLETO),
    "completo_sin_huber": _sin("loss"),
    "completo_sin_clip": _sin("clip_target"),
    "completo_sin_gap": _sin("gap"),
    "completo_sin_panel": _sin("universo"),
    "completo_ic": _COMPLETO | {"seleccion": "ic"},
}
# Transformer V2 (``src.panel.transformer_v2``, objetivo: direccion en dolares)
# contra el original: los defaults y una variante por cada eleccion a validar.
# Una red por trabajo (``n_semillas=1``): el ensamble lo arman las semillas de
# la busqueda. La direccion en dolares (y la de la senal selectiva) se mide con
# ``src.simulacion.direccion``.
# Con los defaults de entonces (``CONFIG_ANTERIOR``): los de ahora son los de Optuna.
_V2 = {"modelo": "transformer_v2", "n_semillas": 1, **CONFIG_ANTERIOR}
PLAN_V2: dict[str, dict] = {
    "original": {"modelo": "transformer"},
    "v2": dict(_V2),
    # Red
    "v2_w5": _V2 | {"ventana": 5},
    "v2_w20": _V2 | {"ventana": 20},
    "v2_sin_cruzada": _V2 | {"capas_cruzadas": 0},
    # Entrenamiento
    "v2_sin_reentreno": _V2 | {"reentrenar_completo": False},
    "v2_combinada": _V2 | {"perdida": "combinada"},
    "v2_bce": _V2 | {"perdida": "bce"},
    # Datos y pregunta
    "v2_macro": _V2 | {"macro": True},
    "v2_historia": _V2 | {"historia": 1500},
    "v2_h20": _V2 | {"horizonte": 20},
    # Las tres que deberian sumar juntas
    "v2_macro_bce_historia": _V2 | {"macro": True, "perdida": "bce", "historia": 1500},
}
# Prueba rapida del V2 (una semilla, ~15 min en GPU): primero los dos ajustes de
# la senal (centrar, ensamble con la reversion) y despues hiperparametros, todos
# con la senal centrada. ``make probar-transformer-v2``.
_V2C = _V2 | {"centrar_senal": True}
PLAN_V2_RAPIDO: dict[str, dict] = {
    "v2_base": dict(_V2),
    "v2_centrado": dict(_V2C),
    "v2_rev25": _V2C | {"peso_reversion": 0.25},
    "v2_rev50": _V2C | {"peso_reversion": 0.5},
    "v2_lr1e3": _V2C | {"learning_rate": 1e-3},
    "v2_reg_baja": _V2C | {"dropout": 0.1, "weight_decay": 0.01},
    "v2_d64": _V2C | {"d_model": 64, "nhead": 8, "dim_feedforward": 128},
    "v2_w5": _V2C | {"ventana": 5},
    "v2_cruz2": _V2C | {"capas_cruzadas": 2},
    "v2_combinada": _V2C | {"perdida": "combinada"},
}
# Ronda 2 del plan de mejora (``notas/analisis-transformer-v2.md``): filtro con
# modelos baratos (CPU, minutos) antes de entrenar el Transformer. Cada grupo se
# mide con su metrica en ``src.simulacion.significancia``: la cabeza relativa
# con IC y ``bal_relativo``, la de mercado con ``bal_mercado``, el meta-labeling
# con ``bal_top`` y las dos cabezas juntas con ``bal_usd``.
_ENS = {"modelo": "ensamble_panel"}
_MERC = {"modelo": "mercado", "historia": 2400}
_META = {"modelo": "meta_reversion"}
PLAN_R2: dict[str, dict] = {
    # Cabeza relativa: reentreno, historia y zona neutral
    "ens": dict(_ENS),
    "ens_re20": _ENS | {"reentrenar_cada": 20},
    "ens_h1500": _ENS | {"historia": 1500},
    "ens_h2400": _ENS | {"historia": 2400},
    "ens_zn025": _ENS | {"zona_neutral": 0.25},
    "ens_zn05": _ENS | {"zona_neutral": 0.5},
    "gbm_h1500": {"modelo": "panel_gbm", "historia": 1500},
    # Cabeza de mercado (2400 ruedas: toda la historia del snapshot del collector)
    "merc_regla": {"modelo": "mercado", "tipo": "regla"},
    "merc_ridge_750": {"modelo": "mercado"},
    "merc_ridge": dict(_MERC),
    "merc_ridge_macro": _MERC | {"macro": True},
    "merc_ridge_zn05": _MERC | {"zona_neutral": 0.5},
    "merc_logit": _MERC | {"tipo": "logit"},
    "merc_logit_macro": _MERC | {"tipo": "logit", "macro": True},
    "merc_ridge_re20": _MERC | {"reentrenar_cada": 20},
    # Meta-labeling sobre la reversion
    "meta_logit": dict(_META),
    "meta_gbm": _META | {"tipo": "gbm"},
    "meta_logit_macro": _META | {"macro": True},
    "meta_logit_h1500": _META | {"historia": 1500},
    "meta_logit_zn05": _META | {"zona_neutral": 0.5},
    # Dos cabezas (mercado + ensamble relativo)
    "dos_regla": {"modelo": "dos_cabezas", "mercado": {"tipo": "regla"}},
    "dos_ridge_macro": {
        "modelo": "dos_cabezas",
        "mercado": {"macro": True},
        "historia": 2400,
    },
}
# Ronda 2, Transformer como cabeza relativa (GPU; lo corre el usuario): el V2 con
# target sin la media del dia, early stopping por la perdida de validacion y la
# historia larga, contra el ensamble con la misma historia (CPU, en la misma
# corrida para que la diferencia sea pareada). ``make transformer-r2``.
_V2_REL = _V2 | {
    "target": "sin_media",
    "seleccion": "perdida",
    "historia": 1500,
}
PLAN_R2_TRANSFORMER: dict[str, dict] = {
    "ens_h1500": _ENS | {"historia": 1500},
    "t_rel": dict(_V2_REL),
    "t_rel_zn05": _V2_REL | {"zona_neutral": 0.5},
    "t_rel_ic": _V2_REL | {"seleccion": "ic"},
}
# Ronda 3: modelos lineales a 20 ruedas (CPU) contra el ensamble a 5. Las reglas
# sin entrenamiento y la senal selectiva estan en ``src.simulacion.reglas``.
PLAN_R3: dict[str, dict] = {
    "ens": dict(_ENS),
    "ens_h20": _ENS | {"horizonte": 20},
    "ens_h20_h1500": _ENS | {"horizonte": 20, "historia": 1500},
    "ridge_h20": {"modelo": "panel_gbm", "tipo": "ridge", "horizonte": 20},
    "ridge_h20_h1500": {"modelo": "panel_gbm", "tipo": "ridge", "horizonte": 20, "historia": 1500},
}
# Baselines sin semillas (deterministicos o con semilla propia).
BASELINES = ["reversion", "xgboost", "lstm"]
# Corridas ya hechas contra las que se compara (precios limpios, periodos de
# eleccion), como "<corrida>:<variante>": el Transformer actual (absoluto,
# 5 semillas) y los baselines.
REFERENCIAS = ["limpio:absoluto", "limpio:reversion", "limpio:lstm", "limpio:xgboost"]


def proveedor_snapshot(snapshot: Snapshot):
    """CCL del dia y tabla macro (con retrasos) conocidos a cada fecha del snapshot."""

    def proveedor(hasta: pd.Timestamp):
        macro = snapshot.macro_frame(hasta) if snapshot.macro_raw else None
        return snapshot.ccl_hasta(hasta), macro

    return proveedor


def _transformer(config: dict, dispositivo: str, snapshot: Snapshot | None) -> object:
    return TransformerTrendModel(config=TransformerTrainConfig(**config), dispositivo=dispositivo)


def _panel_gbm(config: dict, dispositivo: str, snapshot: Snapshot | None) -> object:
    proveedor = proveedor_snapshot(snapshot) if snapshot is not None else None
    return PanelGBMModel(config=PanelGBMConfig(**config), proveedor=proveedor)


def _ensamble_panel(config: dict, dispositivo: str, snapshot: Snapshot | None) -> object:
    """``peso_reversion`` y ``cuantil_senal`` son del ensamble; el resto va al Ridge."""
    config = dict(config)
    peso = config.pop("peso_reversion", 0.5)
    cuantil = config.pop("cuantil_senal", None)
    ridge = PanelGBMConfig(**({"tipo": "ridge"} | config))
    proveedor = proveedor_snapshot(snapshot) if snapshot is not None else None
    return EnsamblePanelModel(EnsambleConfig(peso, ridge, cuantil), proveedor=proveedor)


def _panel_transformer(config: dict, dispositivo: str, snapshot: Snapshot | None) -> object:
    proveedor = proveedor_snapshot(snapshot) if snapshot is not None else None
    return PanelTransformerModel(
        PanelTransformerConfig(**config), proveedor=proveedor, dispositivo=dispositivo
    )


def _transformer_v2(config: dict, dispositivo: str, snapshot: Snapshot | None) -> object:
    proveedor = proveedor_snapshot(snapshot) if snapshot is not None else None
    return TransformerV2Model(
        TransformerV2Config(**config), proveedor=proveedor, dispositivo=dispositivo
    )


def _mercado(config: dict, dispositivo: str, snapshot: Snapshot | None) -> object:
    proveedor = proveedor_snapshot(snapshot) if snapshot is not None else None
    return MercadoModel(MercadoConfig(**config), proveedor=proveedor)


def _meta_reversion(config: dict, dispositivo: str, snapshot: Snapshot | None) -> object:
    proveedor = proveedor_snapshot(snapshot) if snapshot is not None else None
    return MetaReversionModel(MetaReversionConfig(**config), proveedor=proveedor)


def _dos_cabezas(config: dict, dispositivo: str, snapshot: Snapshot | None) -> object:
    """``mercado``: config de ``MercadoConfig``; ``relativo``: config de otro
    modelo de ``MODELOS`` (con su clave ``"modelo"``, default ``ensamble_panel``);
    ``peso_mercado``. La semilla va a las dos cabezas."""
    seed = config.get("seed", 0)
    relativo = dict(config.get("relativo", {}))
    constructor = MODELOS[relativo.pop("modelo", "ensamble_panel")]
    return DosCabezasModel(
        mercado=_mercado({**config.get("mercado", {}), "seed": seed}, dispositivo, snapshot),
        relativo=constructor({**relativo, "seed": seed}, dispositivo, snapshot),
        peso_mercado=config.get("peso_mercado", 1.0),
    )


# Modelos entrenables: nombre -> constructor(config, dispositivo, snapshot). La
# config incluye ``seed``; un modelo deterministico puede ignorarla. El
# snapshot es para los modelos que necesitan datos ademas de los historicos
# (CCL, macro), siempre cortados a la fecha de cada prediccion.
MODELOS: dict[str, Callable[[dict, str, Snapshot | None], object]] = {
    "transformer": _transformer,
    "panel_gbm": _panel_gbm,
    "ensamble_panel": _ensamble_panel,
    "panel_transformer": _panel_transformer,
    "transformer_v2": _transformer_v2,
    "reversion_panel": lambda config, dispositivo, snapshot: ReversionPanelModel(),
    "mercado": _mercado,
    "meta_reversion": _meta_reversion,
    "dos_cabezas": _dos_cabezas,
    # Los de produccion que predicen ticker por ticker (ver ``por_ticker``).
    "lstm": por_ticker.lstm,
    "xgboost": por_ticker.xgboost,
    "arima": por_ticker.arima,
    "macro": por_ticker.macro,
    "contexto": por_ticker.contexto,
}

ESPACIO_HIPER: dict[str, list] = {
    "window": [20, 30, 45, 60],
    "d_model": [16, 32, 64],
    "num_layers": [1, 2, 3],
    "ff_mult": [2, 4],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "learning_rate": [3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-4, 1e-3, 1e-2],
    "batch_size": [64, 128, 256],
    "patience": [5, 8, 12],
}


def muestrear_hiper(n: int, base: dict, seed: int = 0) -> dict[str, dict]:
    """``n`` configuraciones al azar de ``ESPACIO_HIPER`` sobre ``base``, mas
    la base sola (``hiper_base``). Reproducible con ``seed``."""
    rng = random.Random(seed)
    planes = {"hiper_base": dict(base)}
    while len(planes) < n + 1:
        elegido = {k: rng.choice(v) for k, v in ESPACIO_HIPER.items()}
        d_model = elegido["d_model"]
        config = dict(base)
        config.update(
            window=elegido["window"],
            d_model=d_model,
            nhead=max(2, d_model // 8),
            num_layers=elegido["num_layers"],
            dim_feedforward=d_model * elegido.pop("ff_mult"),
            dropout=elegido["dropout"],
            learning_rate=elegido["learning_rate"],
            weight_decay=elegido["weight_decay"],
            batch_size=elegido["batch_size"],
            patience=elegido["patience"],
        )
        nombre = "h_" + "_".join(
            str(config[k])
            for k in (
                "window",
                "d_model",
                "num_layers",
                "dropout",
                "learning_rate",
                "weight_decay",
                "batch_size",
                "patience",
            )
        )
        planes[nombre] = config
    return planes


def trabajos(
    configs: dict[str, dict],
    semillas: int,
    periodos: list[tuple[str, str]],
    baselines: list[str] | None = None,
) -> list[dict]:
    """Un trabajo por (variante, semilla, periodo); los baselines con una sola semilla."""
    lista = []
    for nombre, config in configs.items():
        modelo = config.get("modelo", "transformer")
        if modelo not in MODELOS:
            raise ValueError(f"{nombre}: modelo {modelo!r} desconocido ({sorted(MODELOS)})")
        for s in range(semillas):
            for desde, hasta in periodos:
                lista.append(
                    {
                        "variante": nombre,
                        "config": config,
                        "semilla": s,
                        "desde": desde,
                        "hasta": hasta,
                    }
                )
    for nombre in baselines or []:
        for desde, hasta in periodos:
            lista.append(
                {"variante": nombre, "config": None, "semilla": 0, "desde": desde, "hasta": hasta}
            )
    return lista


def _clave(t: dict) -> tuple:
    return (t["variante"], t["semilla"], t["desde"])


_SNAPSHOT: Snapshot | None = None
_DISPOSITIVO = "cpu"


def cargar_snapshot(snapshot_dir: str, limpiar: bool = False) -> Snapshot:
    snapshot = Snapshot.load(snapshot_dir)
    return limpiar_snapshot(snapshot)[0] if limpiar else snapshot


def _init_worker(snapshot_dir: str, limpiar: bool = False, dispositivo: str = "cpu") -> None:
    global _SNAPSHOT, _DISPOSITIVO
    torch.set_num_threads(1)
    _SNAPSHOT = cargar_snapshot(snapshot_dir, limpiar)
    _DISPOSITIVO = dispositivo


def _fabrica(trabajo: dict, snapshot: Snapshot | None = None):
    if trabajo["config"] is None:
        return VARIANTES[trabajo["variante"]]
    config = dict(trabajo["config"])
    constructor = MODELOS[config.pop("modelo", "transformer")]
    # Las usa ``ejecutar``, no el modelo.
    config.pop("historia", None)
    config.pop("reentrenar_cada", None)
    config["seed"] = SEMILLA_BASE + trabajo["semilla"]
    return lambda: constructor(config, _DISPOSITIVO, snapshot or _SNAPSHOT)


def ejecutar(trabajo: dict, snapshot: Snapshot | None = None, salida: Path | None = None) -> dict:
    """Corre un trabajo: guarda sus predicciones (si hay ``salida``) y devuelve sus metricas."""
    snapshot = snapshot or _SNAPSHOT
    with contextlib.redirect_stdout(io.StringIO()):
        df = predecir_periodo(
            snapshot,
            _fabrica(trabajo, snapshot),
            trabajo["desde"],
            trabajo["hasta"],
            PANEL_LIDER_TICKERS,
            # ``historia``: ruedas de entrenamiento (default: las de produccion).
            days_fit=(trabajo["config"] or {}).get("historia"),
            # ``reentrenar_cada``: ruedas entre reentrenos (default: uno por periodo).
            reentrenar_cada=(trabajo["config"] or {}).get("reentrenar_cada"),
        )
    if salida is not None:
        nombre = f"{trabajo['variante']}__{trabajo['semilla']}__{trabajo['desde']}.csv"
        df[["fecha", "symbol", "pred", "real"]].to_csv(
            salida / "predicciones" / nombre, index=False
        )
    return {**{k: trabajo[k] for k in ("variante", "semilla", "desde", "hasta")}, **metricas(df)}


def _ejecutar_en_worker(args: tuple[dict, str]) -> dict:
    trabajo, salida = args
    return ejecutar(trabajo, salida=Path(salida))


def correr(
    snapshot_dir: str,
    nombre: str,
    lista: list[dict],
    procesos: int,
    base: Path = DIR_EXPERIMENTOS,
    limpiar: bool = False,
    dispositivo: str = "cpu",
) -> int:
    """Corre en paralelo los trabajos que todavia no estan en ``resultados.jsonl``."""
    salida = base / nombre
    (salida / "predicciones").mkdir(parents=True, exist_ok=True)
    opciones = {"limpiar": limpiar, "dispositivo": dispositivo}
    archivo_opciones = salida / "opciones.json"
    if archivo_opciones.exists():
        previas = json.loads(archivo_opciones.read_text())
        if previas != opciones:
            raise SystemExit(f"{nombre} se corrio con {previas}, no con {opciones}")
    archivo_opciones.write_text(json.dumps(opciones))
    resultados = salida / "resultados.jsonl"
    hechos = set()
    if resultados.exists():
        hechos = {_clave(json.loads(linea)) for linea in resultados.open()}
    pendientes = [t for t in lista if _clave(t) not in hechos]
    configs = {t["variante"]: t["config"] for t in lista}
    (salida / "configs.json").write_text(json.dumps(configs, indent=2))
    print(f"{len(pendientes)} trabajos pendientes de {len(lista)}", flush=True)
    if not pendientes:
        return 0
    with mp.get_context("fork").Pool(
        procesos, initializer=_init_worker, initargs=(snapshot_dir, limpiar, dispositivo)
    ) as pool:
        args = [(t, str(salida)) for t in pendientes]
        for i, fila in enumerate(pool.imap_unordered(_ejecutar_en_worker, args), 1):
            with resultados.open("a") as f:
                f.write(json.dumps(fila) + "\n")
            etiqueta = f"{fila['variante']} s{fila['semilla']} {fila['desde']}"
            print(f"[{i}/{len(pendientes)}] {etiqueta} ic={fila['ic']}", flush=True)
    return len(pendientes)


def _ic_ensamble(salida: Path, variante: str, desde: str) -> float | None:
    archivos = sorted((salida / "predicciones").glob(f"{variante}__*__{desde}.csv"))
    if len(archivos) < 2:
        return None
    df = pd.concat(pd.read_csv(a) for a in archivos)
    promedio = df.groupby(["fecha", "symbol"], as_index=False).agg(
        pred=("pred", "mean"), real=("real", "first")
    )
    promedio["signal"] = "neutral"
    return metricas(promedio)["ic"]


def _filas(salida: Path, periodos: str, solo: str | None = None) -> list[dict]:
    """Una fila de resumen por variante de la corrida ``salida`` (o solo ``solo``)."""
    df = pd.DataFrame([json.loads(linea) for linea in (salida / "resultados.jsonl").open()])
    desdes = {d for d, _ in PERIODOS[periodos]}
    df = df[df["desde"].isin(desdes)].drop_duplicates(["variante", "semilla", "desde"], keep="last")
    if solo is not None:
        df = df[df["variante"] == solo]
        if df.empty:
            raise SystemExit(f"{salida.name} no tiene resultados de {solo} en esos periodos")
    filas = []
    for variante, g in df.groupby("variante"):
        por_periodo = g.groupby("desde")["ic"].agg(["mean", "std", "size"])
        ens = {d: _ic_ensamble(salida, variante, d) for d in por_periodo.index}
        tiene_ens = all(v is not None for v in ens.values())
        serie = pd.Series(ens) if tiene_ens else por_periodo["mean"]
        error = serie.std() / np.sqrt(len(serie)) if len(serie) > 1 else np.nan
        filas.append(
            {
                "variante": variante,
                "semillas": int(por_periodo["size"].min()),
                "periodos": len(por_periodo),
                "ic_medio": por_periodo["mean"].mean(),
                "ruido": por_periodo["std"].mean(),
                "ic_ens": serie.mean() if tiene_ens else np.nan,
                "t_periodos": serie.mean() / error if error and error > 0 else np.nan,
                "min_periodo": serie.min(),
                "periodos_pos": int((serie > 0).sum()),
                "ls_medio": g["ls"].mean(),
                "ventaja_alza": _ventaja(g, "precision_alza", g["base_up"]),
                "ventaja_baja": _ventaja(g, "precision_baja", 1 - g["base_up"]),
                "pct_senal": float((g["pct_alza"] + g["pct_baja"]).mean())
                if "pct_baja" in g
                else np.nan,
                "exceso_alza": _ventaja(g, "ret_alza", g.get("ret_medio")),
                "exceso_baja": _ventaja(g, "ret_baja", g.get("ret_medio")),
            }
        )
    return filas


def _ventaja(g: pd.DataFrame, columna: str, tasa_base: pd.Series | None) -> float:
    """Media de una metrica de una senal menos su base (precision - tasa base, o
    retorno de la senal - retorno medio). NaN si nunca se dio o la corrida es vieja."""
    if columna not in g or tasa_base is None:
        return np.nan
    return float((g[columna].astype(float) - tasa_base).mean())


def _limpieza(salida: Path) -> bool | None:
    archivo = salida / "opciones.json"
    return json.loads(archivo.read_text()).get("limpiar") if archivo.exists() else None


def resumen(
    nombre: str,
    base: Path = DIR_EXPERIMENTOS,
    periodos: str = "eleccion",
    referencias: list[str] | None = None,
) -> pd.DataFrame:
    """Una fila por variante, ordenada por IC del ensamble (o medio, si no hay):

    - ``ic_medio``: promedio entre periodos del IC medio entre semillas.
    - ``ruido``: desvio del IC entre semillas, promediado entre periodos.
    - ``ic_ens``: IC del promedio de las semillas (ensamble), promediado entre periodos.
    - ``t_periodos``: IC del ensamble (o medio) / su error estandar entre periodos.
    - ``min_periodo`` y ``periodos_pos``: peor periodo y cuantos dan IC > 0.
    - ``ventaja_alza`` / ``ventaja_baja``: precision de cada senal menos su tasa
      base (> 0 = la senal aporta; NaN = nunca se dio o la corrida es vieja).
    - ``pct_senal``: fraccion de casos con senal "alza" o "baja".
    - ``exceso_alza`` / ``exceso_baja``: retorno log real (%) de los casos con
      esa senal menos el de todos (alza > 0 y baja < 0 = la senal aporta).

    ``referencias`` ("<corrida>:<variante>") suma filas de otras corridas,
    que tienen que haberse hecho con la misma limpieza de precios.
    """
    salida = base / nombre
    filas = _filas(salida, periodos)
    for ref in referencias or []:
        corrida, _, variante = ref.partition(":")
        if _limpieza(base / corrida) != _limpieza(salida):
            raise SystemExit(f"{ref} no se corrio con la misma limpieza que {nombre}")
        filas += [f | {"variante": ref} for f in _filas(base / corrida, periodos, variante)]
    tabla = pd.DataFrame(filas).set_index("variante")
    orden = tabla["ic_ens"].fillna(tabla["ic_medio"])
    return tabla.loc[orden.sort_values(ascending=False).index].round(4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda de configuraciones de modelos.")
    parser.add_argument("--nombre", required=True)
    parser.add_argument(
        "--plan",
        choices=["ablacion", "hiper", "v2", "rapido", "r2", "r2-transformer", "r3", "config"],
    )
    parser.add_argument("--base", type=json.loads, default={}, help="Formulacion base (hiper).")
    parser.add_argument("--configs", type=json.loads, default={}, help="{nombre: config} (config).")
    parser.add_argument("--n", type=int, default=30, help="Configuraciones del plan hiper.")
    parser.add_argument("--semillas", type=int, default=5)
    parser.add_argument("--periodos", choices=sorted(PERIODOS), default="eleccion")
    parser.add_argument("--baselines", action="store_true", help="Sumar reversion/xgboost/lstm.")
    parser.add_argument(
        "--referencias",
        nargs="*",
        help="Sumar al resumen '<corrida>:<variante>' ya corridas (sin valores: REFERENCIAS).",
    )
    parser.add_argument("--procesos", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    parser.add_argument("--snapshot", default=DEFAULT_DIR)
    parser.add_argument("--limpiar", action="store_true", help="Precios sin saltos (limpieza).")
    parser.add_argument("--dispositivo", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resumen", action="store_true", help="Solo imprimir el resumen.")
    args = parser.parse_args()

    if args.plan and not args.resumen:
        if args.plan == "ablacion":
            configs = PLAN_ABLACION
        elif args.plan == "v2":
            configs = PLAN_V2
        elif args.plan == "rapido":
            configs = PLAN_V2_RAPIDO
        elif args.plan == "r2":
            configs = PLAN_R2
        elif args.plan == "r2-transformer":
            configs = PLAN_R2_TRANSFORMER
        elif args.plan == "r3":
            configs = PLAN_R3
        elif args.plan == "hiper":
            configs = muestrear_hiper(args.n, args.base)
        else:
            configs = args.configs
        lista = trabajos(
            configs,
            args.semillas,
            PERIODOS[args.periodos],
            BASELINES if args.baselines else None,
        )
        correr(
            args.snapshot,
            args.nombre,
            lista,
            args.procesos,
            base=DIR_EXPERIMENTOS,
            limpiar=args.limpiar,
            dispositivo=args.dispositivo,
        )
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        referencias = REFERENCIAS if args.referencias == [] else args.referencias
        print(
            resumen(
                args.nombre, base=DIR_EXPERIMENTOS, periodos=args.periodos, referencias=referencias
            )
        )


if __name__ == "__main__":
    main()
