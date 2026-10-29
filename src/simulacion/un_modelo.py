"""Simulacion rapida de un solo modelo: entrenar, medir la direccion y operar carteras.

El simulador completo (``simulador``) entrena LSTM, XGBoost, Transformer y
macro, ajusta ARIMA, calcula backtests en vivo y arma Black-Litterman, consenso
y "mejor modelo" para cada uno: tarda horas. Para probar una variante de un
modelo (p. ej. el Transformer V2) alcanza con esto:

1. **Entrenar** el modelo con datos hasta la rueda anterior a ``--desde`` y
   reentrenarlo segun ``--reentreno`` (``mensual`` por default; ``nunca`` deja
   el primero). Las metricas que devuelve cada ``fit`` (las de validacion del
   V2) quedan en ``entrenamientos.jsonl``. Entrena con los precios limpios
   (``limpieza``), como la busqueda.
2. **Predecir** cada rueda los tickers del panel lider que cotizaron
   (``predicciones.jsonl``, formato del simulador).
3. **Medir la direccion en dolares** (``significancia``): ``bal_usd``,
   ``bal_top``, ``ic``, ``bal_mercado`` y ``bal_relativo`` con intervalos por
   bootstrap de bloques de fechas, junto con las referencias sin entrenamiento
   (``direccion.referencias``) y la diferencia pareada contra la reversion
   (``direccion.csv``).
4. **Carteras simples** (``Mesa`` sin ``completa``): la senal del modelo, las
   reglas que rotan menos (``mesa.REGLAS_PANEL``) y los benchmarks, con la
   comision y el rebalanceo pedidos (``carteras.csv``). Sin Black-Litterman,
   consenso ni backtests en vivo.

Todo queda en ``simulacion/resultados/<nombre>/`` con un ``resumen.md``.
``--solo-analisis`` rehace los pasos 3 y 4 con las predicciones guardadas (sin
entrenar) y ``rearmar`` rehace las carteras con otras reglas.

El modelo es cualquiera de ``busqueda.MODELOS``: los de panel y los de
produccion que predicen ticker por ticker (``lstm``, ``xgboost``, ``arima``,
``macro``, ``contexto``, ver ``por_ticker``). ``--config`` cambia campos de su
config (como las variantes de la busqueda: ``"modelo"`` e ``"historia"``
tambien se aceptan dentro del JSON).

Uso:
    python -m src.simulacion.un_modelo --nombre ens-2025 --desde 2025-01-02 --hasta 2025-12-30 \\
        --snapshot simulacion/snapshot-collector
    python -m src.simulacion.un_modelo --nombre pt-2025 --modelo panel_transformer \\
        --desde 2025-01-02 --config '{"n_semillas": 3}' --reentreno nunca --dispositivo cuda
    python -m src.simulacion.un_modelo --nombre lstm-2026 --modelo lstm --desde 2026-01-02
    python -m src.simulacion.un_modelo --nombre ens-2025 --solo-analisis --comision 0.01
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.simulacion.busqueda import MODELOS
from src.simulacion.cartera import CAPITAL_INICIAL_DEFAULT, COMISION_DEFAULT
from src.simulacion.direccion import REFERENCIA_REVERSION, referencias, retornos_usd
from src.simulacion.entrenamiento import FRECUENCIAS_REENTRENO
from src.simulacion.limpieza import limpiar
from src.simulacion.mesa import Mesa
from src.simulacion.predicciones_panel import generar
from src.simulacion.reporte import SIN_PDF, tabla_carteras
from src.simulacion.reporte_un_modelo import generar as generar_reporte
from src.simulacion.significancia import NIVEL, legible, significancia
from src.simulacion.simulador import (
    REBALANCEO_DEFAULT,
    RESULTADOS_DIR,
    guardar_rueda,
    silenciar_warnings,
    ultima_rueda_completa,
)
from src.simulacion.snapshot import DEFAULT_DIR as SNAPSHOT_DIR
from src.simulacion.snapshot import Snapshot

MODELO_DEFAULT = "ensamble_panel"
REENTRENO_DEFAULT = "mensual"
ARCHIVOS_CARTERAS = ("valores", "pesos", "operaciones", "errores", "elecciones")


@dataclass
class ConfigUnModelo:
    nombre: str
    desde: str
    hasta: str
    modelo: str = MODELO_DEFAULT
    # Campos de la config del modelo a cambiar (vacio = la default del modelo).
    config: dict = field(default_factory=dict)
    # Ruedas de historia para entrenar; None = las de produccion.
    historia: int | None = None
    reentreno: str = REENTRENO_DEFAULT
    # Senal por cuantil del dia (``senal_por_cuantil``); None = la del modelo.
    cuantil: float | None = None
    rebalanceo: int = REBALANCEO_DEFAULT
    capital: float = CAPITAL_INICIAL_DEFAULT
    comision: float = COMISION_DEFAULT
    snapshot_dir: str = SNAPSHOT_DIR
    limpiar: bool = True

    def __post_init__(self) -> None:
        config = dict(self.config)
        self.modelo = config.pop("modelo", self.modelo)
        historia = config.pop("historia", self.historia)
        if historia is None:
            historia = settings.history_days
        self.historia = int(historia)
        self.config = config
        if self.modelo not in MODELOS:
            raise ValueError(f"modelo {self.modelo!r} desconocido ({sorted(MODELOS)})")
        if self.reentreno not in FRECUENCIAS_REENTRENO:
            raise ValueError(f"reentreno {self.reentreno!r} desconocido")


# --------------------------------------------------------------------------- #
# Entrenamiento y prediccion
# --------------------------------------------------------------------------- #
def entrenar_y_predecir(
    cfg: ConfigUnModelo, crudo: Snapshot, limpio: Snapshot, out: Path, dispositivo: str = "cpu"
) -> list[dict]:
    """Predicciones de cada rueda del periodo (``predicciones.jsonl``) y una fila
    por entrenamiento (``entrenamientos.jsonl``)."""
    constructor = MODELOS[cfg.modelo]
    return generar(
        cfg.modelo,
        lambda: constructor(dict(cfg.config), dispositivo, limpio),
        crudo,
        limpio,
        cfg.desde,
        cfg.hasta,
        reentreno=cfg.reentreno,
        days=cfg.historia,
        cuantil=cfg.cuantil,
        destino=out / "predicciones.jsonl",
        entrenamientos=out / "entrenamientos.jsonl",
    )


# --------------------------------------------------------------------------- #
# Analisis
# --------------------------------------------------------------------------- #
def tabla_direccion(
    predicciones: pd.DataFrame, snapshot: Snapshot, nombre: str, contra: str = REFERENCIA_REVERSION
) -> pd.DataFrame | None:
    """``significancia`` del modelo y de las referencias sobre las mismas filas
    (las predicciones cuyo horizonte ya se cumplio); None si no hay ninguna."""
    if predicciones.empty:
        return None
    horizonte = int(predicciones["horizon_days"].iloc[0])
    df = predicciones[["fecha", "symbol"]].assign(
        pred=np.log(predicciones["predicted_close"] / predicciones["last_close"])
    )
    df = df.merge(retornos_usd(snapshot, horizonte), on=["fecha", "symbol"])
    df = df.dropna(subset=["futuro"])
    if df.empty:
        return None
    # Un solo periodo: el umbral de ``bal_top`` es el de toda la corrida.
    df.insert(0, "desde", df["fecha"].min())
    return significancia({nombre: df} | referencias(df), contra)


COBERTURAS_COMPRA = (1.0, 0.5, 0.2, 0.1, 0.05)


def metricas_compra(predicciones: pd.DataFrame, snapshot: Snapshot) -> dict[str, pd.DataFrame]:
    """Lo que le importa a quien compra, en dolares a ``horizon_days`` ruedas.

    - ``por_senal``: por senal del modelo (alza / neutral / baja), cuantas veces,
      que fraccion subio (la **certeza**: para "alza", la precision), el retorno
      simple medio (la **fuerza**) y su exceso sobre el promedio del panel del
      mismo dia. La fila ``todas`` es la tasa base.
    - ``confianza``: entre las predicciones positivas, el ``cobertura`` mas fuerte
      (umbral del propio periodo: es descriptivo), con las mismas columnas. Si
      el modelo sabe cuando esta seguro, el acierto y el exceso suben al bajar la
      cobertura.
    """
    vacio = {"por_senal": pd.DataFrame(), "confianza": pd.DataFrame()}
    if predicciones.empty:
        return vacio
    horizonte = int(predicciones["horizon_days"].iloc[0])
    df = predicciones[["fecha", "symbol", "signal"]].assign(
        pred=np.log(predicciones["predicted_close"] / predicciones["last_close"])
    )
    df = df.merge(retornos_usd(snapshot, horizonte)[["fecha", "symbol", "futuro"]],
                  on=["fecha", "symbol"]).dropna(subset=["futuro"])  # fmt: skip
    if df.empty:
        return vacio
    df["ret"] = np.expm1(df["futuro"])
    df["exceso"] = df["ret"] - df.groupby("fecha")["ret"].transform("mean")

    def fila(nombre: str, g: pd.DataFrame) -> dict:
        return {
            "grupo": nombre,
            "casos": len(g),
            "subio": float((g["ret"] > 0).mean()),
            "ret_medio": float(g["ret"].mean()),
            "exceso_medio": float(g["exceso"].mean()),
        }

    por_senal = [fila("todas", df)]
    por_senal += [fila(s, g) for s, g in df.groupby("signal") if len(g)]
    compras = df[df["pred"] > 0]
    confianza = []
    for cobertura in COBERTURAS_COMPRA:
        if compras.empty:
            break
        top = compras[compras["pred"] >= compras["pred"].quantile(1 - cobertura)]
        confianza.append({"cobertura": cobertura, **fila(f"{cobertura:.0%}", top)})
    return {"por_senal": pd.DataFrame(por_senal), "confianza": pd.DataFrame(confianza)}


def operar(
    cfg: ConfigUnModelo, predicciones: pd.DataFrame, snapshot: Snapshot, out: Path
) -> pd.DataFrame:
    """Rueda por rueda, las carteras simples del modelo y los benchmarks
    (``valores.jsonl`` y compania); devuelve sus metricas."""
    for nombre in ARCHIVOS_CARTERAS:
        (out / f"{nombre}.jsonl").unlink(missing_ok=True)
    por_dia: dict[str, dict[str, dict]] = defaultdict(dict)
    for fila in predicciones.to_dict("records"):
        fila = dict(fila)
        por_dia[fila.pop("fecha")][fila["symbol"]] = fila
    mesa = Mesa(snapshot, cfg.capital, cfg.comision, cfg.rebalanceo, (cfg.modelo,), completa=False)
    ruedas = snapshot.trading_days()
    valores = []
    for dia in ruedas[(ruedas >= pd.Timestamp(cfg.desde)) & (ruedas <= pd.Timestamp(cfg.hasta))]:
        preds = {cfg.modelo: por_dia.get(str(dia.date()), {})}
        resultado = mesa.operar(dia, preds, {})
        guardar_rueda(out, resultado)
        valores.append(resultado.valores)
    if not valores:
        return pd.DataFrame()
    return tabla_carteras(pd.DataFrame(valores))


def _texto(df: pd.DataFrame) -> str:
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        return df.to_string()


def resumen_md(
    cfg: ConfigUnModelo,
    entrenamientos: pd.DataFrame,
    direccion: pd.DataFrame | None,
    carteras: pd.DataFrame,
) -> str:
    partes = [
        f"# {cfg.nombre}",
        "",
        f"Modelo `{cfg.modelo}` del {cfg.desde} al {cfg.hasta}, reentreno {cfg.reentreno}, "
        f"{cfg.historia} ruedas de historia.",
        "",
        f"Config: `{json.dumps(cfg.config)}`",
        "",
    ]
    if not entrenamientos.empty:
        partes += ["## Entrenamientos", "", "```", _texto(entrenamientos), "```", ""]
    partes += ["## Direccion en dolares", ""]
    if direccion is None:
        partes += ["Ninguna prediccion con el horizonte cumplido.", ""]
    else:
        partes += [
            f"Intervalos al {NIVEL:.0%} (bootstrap de bloques de fechas); "
            f"`vs_{REFERENCIA_REVERSION}`: diferencia pareada contra la reversion.",
            "",
            "```",
            _texto(legible(direccion, REFERENCIA_REVERSION)),
            "```",
            "",
        ]
    partes += [
        "## Carteras",
        "",
        f"Comision {cfg.comision:.2%} por operacion, rebalanceo cada {cfg.rebalanceo} ruedas.",
        "",
        "```",
        _texto(carteras),
        "```",
        "",
    ]
    return "\n".join(partes)


def analizar(cfg: ConfigUnModelo, crudo: Snapshot, limpio: Snapshot, out: Path) -> str:
    """Direccion y carteras a partir de lo guardado en ``out``; escribe
    ``direccion.csv``, ``carteras.csv`` y ``resumen.md`` y devuelve el resumen."""
    archivo = out / "predicciones.jsonl"
    if not archivo.exists() or archivo.stat().st_size == 0:
        raise SystemExit(f"{out} no tiene predicciones guardadas")
    predicciones = pd.read_json(archivo, lines=True, dtype=False, convert_dates=False)
    entrenamientos = out / "entrenamientos.jsonl"
    entrenamientos = (
        pd.read_json(entrenamientos, lines=True, dtype=False, convert_dates=False)
        if entrenamientos.exists() and entrenamientos.stat().st_size
        else pd.DataFrame()
    )
    direccion = tabla_direccion(predicciones, limpio, cfg.modelo)
    if direccion is not None:
        direccion.to_csv(out / "direccion.csv")
    for nombre, tabla in metricas_compra(predicciones, limpio).items():
        tabla.to_csv(out / f"compra_{nombre}.csv", index=False)
    carteras = operar(cfg, predicciones, crudo, out)
    carteras.to_csv(out / "carteras.csv", index=False)
    texto = resumen_md(cfg, entrenamientos, direccion, carteras)
    (out / "resumen.md").write_text(texto)
    reporte = generar_reporte(out).resolve()
    rutas = f"\nReporte LaTeX: {reporte.with_suffix('.tex')}\n"
    if reporte.suffix == ".pdf":
        return texto + rutas + f"Reporte PDF:   {reporte}\n"
    return texto + rutas + SIN_PDF + "\n"


# --------------------------------------------------------------------------- #
def cargar_snapshots(cfg: ConfigUnModelo) -> tuple[Snapshot, Snapshot]:
    """Snapshot crudo (carteras, ``last_close``) y el de entrenamiento (limpio)."""
    crudo = Snapshot.load(cfg.snapshot_dir)
    return crudo, limpiar(crudo)[0] if cfg.limpiar else crudo


def correr(cfg: ConfigUnModelo, dispositivo: str = "cpu") -> str:
    out = Path(RESULTADOS_DIR) / cfg.nombre
    if (out / "config.json").exists():
        raise SystemExit(f"{out} ya existe: usar otro --nombre o --solo-analisis")
    crudo, limpio = cargar_snapshots(cfg)
    # Una clave mal escrita en --config falla aca, antes de crear la carpeta.
    MODELOS[cfg.modelo](dict(cfg.config), dispositivo, limpio)
    out.mkdir(parents=True, exist_ok=True)
    guardada = {**asdict(cfg), "solo_modelo": cfg.modelo, "dispositivo": dispositivo,
                "snapshot": crudo.meta.get("downloaded_at")}  # fmt: skip
    (out / "config.json").write_text(json.dumps(guardada, indent=2))
    entrenar_y_predecir(cfg, crudo, limpio, out, dispositivo)
    return analizar(cfg, crudo, limpio, out)


def reanalizar(nombre: str, cambios: dict) -> str:
    """``analizar`` de una corrida ya hecha, con otras reglas de cartera."""
    out = Path(RESULTADOS_DIR) / nombre
    if not (out / "config.json").exists():
        raise SystemExit(f"{out} no existe")
    guardada = json.loads((out / "config.json").read_text())
    campos = {k: v for k, v in guardada.items() if k in ConfigUnModelo.__dataclass_fields__}
    cfg = ConfigUnModelo(**(campos | cambios))
    crudo, limpio = cargar_snapshots(cfg)
    return analizar(cfg, crudo, limpio, out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Simulacion rapida de un solo modelo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--nombre", required=True, help="Nombre de la corrida (carpeta).")
    parser.add_argument("--desde", help="Primera rueda simulada (YYYY-MM-DD).")
    parser.add_argument("--hasta", help="Ultima rueda (default: la ultima completa del snapshot).")
    parser.add_argument("--modelo", choices=sorted(MODELOS), default=MODELO_DEFAULT)
    parser.add_argument(
        "--config", type=json.loads, default={}, help="JSON con campos de la config del modelo."
    )
    parser.add_argument(
        "--historia", type=int, help="Ruedas de historia (default: 2400 el V2, 750 el resto)."
    )
    parser.add_argument("--reentreno", choices=FRECUENCIAS_REENTRENO, default=REENTRENO_DEFAULT)
    parser.add_argument("--cuantil", type=float, help="Senal por cuantil del dia (p. ej. 0.15).")
    parser.add_argument("--rebalanceo", type=int, default=REBALANCEO_DEFAULT)
    parser.add_argument("--capital", type=float, default=CAPITAL_INICIAL_DEFAULT)
    parser.add_argument("--comision", type=float, default=COMISION_DEFAULT)
    parser.add_argument("--snapshot", default=SNAPSHOT_DIR)
    parser.add_argument("--sin-limpiar", action="store_true", help="Entrenar con precios crudos.")
    parser.add_argument("--dispositivo", default="cpu")
    parser.add_argument(
        "--solo-analisis",
        action="store_true",
        help="Rehacer direccion y carteras con las predicciones guardadas (sin entrenar); "
        "acepta --rebalanceo, --capital y --comision.",
    )
    args = parser.parse_args(argv)
    silenciar_warnings()

    if args.solo_analisis:
        explicitos = {
            k: getattr(args, k)
            for k in ("rebalanceo", "capital", "comision")
            if getattr(args, k) != parser.get_default(k)
        }
        print(reanalizar(args.nombre, explicitos))
        return
    if not args.desde:
        parser.error("--desde es obligatorio (salvo con --solo-analisis)")
    hasta = args.hasta or ultima_rueda_completa(Snapshot.load(args.snapshot))
    cfg = ConfigUnModelo(
        nombre=args.nombre,
        desde=args.desde,
        hasta=hasta,
        modelo=args.modelo,
        config=args.config,
        historia=args.historia,
        reentreno=args.reentreno,
        cuantil=args.cuantil,
        rebalanceo=args.rebalanceo,
        capital=args.capital,
        comision=args.comision,
        snapshot_dir=args.snapshot,
        limpiar=not args.sin_limpiar,
    )
    print(correr(cfg, args.dispositivo))


if __name__ == "__main__":
    main()
