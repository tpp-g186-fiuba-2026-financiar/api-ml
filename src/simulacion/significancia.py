"""Intervalos de confianza de las metricas de direccion de una corrida de ``busqueda``.

``direccion`` da el acierto de cada variante, pero no si una diferencia es real
o ruido. Las 20 acciones del panel en dolares se mueven juntas (el CCL las
arrastra) y los retornos a 5 ruedas de dias seguidos se superponen: las filas
no son independientes y un error estandar comun subestima el ruido. Aca:

1. **Bootstrap por bloques de fechas**: se remuestrean bloques de ``bloque``
   ruedas seguidas (con todas sus acciones), que conservan la correlacion
   entre acciones del mismo dia y entre dias cercanos. Todas las variantes usan
   las mismas fechas remuestreadas, asi la **diferencia pareada** contra una
   referencia tiene su propio intervalo (y ``p_mejor``: fraccion de
   remuestreos en que la variante le gana).
2. **Descomposicion de la direccion**: la direccion en dolares de una accion es
   la del mercado mas la de la accion contra el resto. ``bal_mercado`` es el
   acierto balanceado del signo del retorno medio del panel del dia (la
   prediccion del mercado de un modelo es el promedio de sus predicciones del
   dia); ``bal_relativo``, el del signo de la accion contra el promedio del dia
   (en la prediccion y en el real).

Metricas (estimacion sobre todas las filas juntas, no promedio de periodos):
``bal_usd``, ``bal_top`` (el 20% de predicciones mas fuertes de cada periodo,
como en ``direccion``), ``ic`` (Spearman medio por fecha), ``bal_mercado`` y
``bal_relativo``. Las referencias (``direccion.referencias``) entran como una
variante mas.

Uso:
    python -m src.simulacion.significancia --nombre v2-vs-original \\
        --snapshot simulacion/snapshot-collector --limpiar
    python -m src.simulacion.significancia --nombre fase5-senal --limpiar --contra ens
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.simulacion.busqueda import DIR_EXPERIMENTOS, PERIODOS, cargar_snapshot
from src.simulacion.direccion import (
    COBERTURA,
    REFERENCIA_REVERSION,
    horizontes,
    predicciones,
    referencias,
    retornos_usd,
)
from src.simulacion.experimento import HORIZON
from src.simulacion.snapshot import DEFAULT_DIR

BLOQUE = 10
N_BOOT = 2000
NIVEL = 0.90
METRICAS = ("bal_usd", "bal_top", "ic", "bal_mercado", "bal_relativo")
# Metricas con diferencia pareada contra la referencia.
PAREADAS = ("bal_usd", "bal_top", "ic", "bal_relativo")


def _aciertos(pred: pd.Series, real: pd.Series, prefijo: str) -> pd.DataFrame:
    """Columnas de conteo para un acierto balanceado: subas acertadas, subas,
    bajas acertadas y bajas (``real <= 0`` es baja, como ``acierto_balanceado``)."""
    sube = real > 0
    return pd.DataFrame(
        {
            f"{prefijo}_sube_ok": (sube & (pred > 0)).astype(float),
            f"{prefijo}_sube_n": sube.astype(float),
            f"{prefijo}_baja_ok": (~sube & (pred <= 0)).astype(float),
            f"{prefijo}_baja_n": (~sube).astype(float),
        }
    )


def _ic_dia(g: pd.DataFrame) -> float:
    if len(g) < 3 or g["pred"].nunique() < 2 or g["futuro"].nunique() < 2:
        return np.nan
    return float(g["pred"].rank().corr(g["futuro"].rank()))


def conteos_por_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Por fecha, todo lo que hace falta para las ``METRICAS`` como sumas: asi un
    remuestreo de fechas es un vector de pesos por fecha.

    ``df``: filas de una variante con ``desde``, ``fecha``, ``pred`` y ``futuro``.
    """
    df = df.reset_index(drop=True)
    fuerza = df["pred"].abs()
    umbral = fuerza.groupby(df["desde"]).transform(lambda f: f.quantile(1 - COBERTURA))
    top = fuerza >= umbral
    pred_rel = df["pred"] - df.groupby("fecha")["pred"].transform("mean")
    real_rel = df["futuro"] - df.groupby("fecha")["futuro"].transform("mean")
    filas = pd.concat(
        [
            df[["fecha"]],
            _aciertos(df["pred"], df["futuro"], "usd"),
            _aciertos(df["pred"], df["futuro"], "top").mul(top, axis=0),
            _aciertos(pred_rel, real_rel, "rel"),
        ],
        axis=1,
    )
    por_fecha = filas.groupby("fecha").sum()
    dia = df.groupby("fecha").agg(pred=("pred", "mean"), futuro=("futuro", "mean"))
    mercado = _aciertos(dia["pred"], dia["futuro"], "merc")
    mercado.index = dia.index
    ic = df.groupby("fecha")[["pred", "futuro"]].apply(_ic_dia)
    por_fecha = por_fecha.join(mercado)
    por_fecha["ic_suma"] = ic.fillna(0.0)
    por_fecha["ic_n"] = ic.notna().astype(float)
    return por_fecha


def _balanceado(c: Callable[[str], np.ndarray], prefijo: str) -> np.ndarray:
    """Acierto balanceado desde las sumas (vectorizado sobre remuestreos); si falta
    un lado cuenta solo el otro y sin casos da 0,5 (como ``acierto_balanceado``)."""
    sube_n, baja_n = c(f"{prefijo}_sube_n"), c(f"{prefijo}_baja_n")
    with np.errstate(invalid="ignore", divide="ignore"):
        sube = np.where(sube_n > 0, c(f"{prefijo}_sube_ok") / sube_n, np.nan)
        baja = np.where(baja_n > 0, c(f"{prefijo}_baja_ok") / baja_n, np.nan)
    partes = np.stack([sube, baja])
    validas = np.isfinite(partes).sum(0)
    return np.where(validas > 0, np.nansum(partes, 0) / np.maximum(validas, 1), 0.5)


def metricas_desde_sumas(sumas: pd.DataFrame | dict) -> dict[str, np.ndarray]:
    """``METRICAS`` desde columnas sumadas (escalares o un vector por remuestreo)."""

    def c(nombre: str) -> np.ndarray:
        return np.asarray(sumas[nombre], dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        ic = np.where(c("ic_n") > 0, c("ic_suma") / c("ic_n"), 0.0)
    return {
        "bal_usd": _balanceado(c, "usd"),
        "bal_top": _balanceado(c, "top"),
        "ic": ic,
        "bal_mercado": _balanceado(c, "merc"),
        "bal_relativo": _balanceado(c, "rel"),
    }


def pesos_bootstrap(
    n_fechas: int, n_boot: int = N_BOOT, bloque: int = BLOQUE, seed: int = 0
) -> np.ndarray:
    """``[n_boot, n_fechas]``: cuantas veces entra cada fecha en cada remuestreo
    (bootstrap de bloques moviles: bloques de ``bloque`` fechas seguidas con
    inicio al azar, hasta juntar ``n_fechas``)."""
    rng = np.random.default_rng(seed)
    bloque = max(1, min(bloque, n_fechas))
    n_bloques = -(-n_fechas // bloque)
    inicios = rng.integers(0, n_fechas - bloque + 1, size=(n_boot, n_bloques))
    indices = (inicios[..., None] + np.arange(bloque)).reshape(n_boot, -1)[:, :n_fechas]
    pesos = np.zeros((n_boot, n_fechas))
    np.add.at(pesos, (np.arange(n_boot)[:, None], indices), 1.0)
    return pesos


def significancia(
    variantes: dict[str, pd.DataFrame],
    contra: str | None = REFERENCIA_REVERSION,
    n_boot: int = N_BOOT,
    bloque: int = BLOQUE,
    nivel: float = NIVEL,
    seed: int = 0,
) -> pd.DataFrame:
    """Una fila por variante: cada metrica con su intervalo (``_lo`` / ``_hi``) y,
    si ``contra`` es una de las variantes, la diferencia pareada en las
    ``PAREADAS`` con su intervalo y ``p_mejor``.

    Todas las variantes usan los mismos remuestreos de fechas (la union de sus
    fechas, en orden; una fecha que falta en una variante no le suma nada)."""
    conteos = {v: conteos_por_fecha(df) for v, df in variantes.items() if len(df)}
    fechas = sorted(set().union(*(c.index for c in conteos.values())))
    pesos = pesos_bootstrap(len(fechas), n_boot, bloque, seed)
    cola = (1 - nivel) / 2
    puntos, remuestreos = {}, {}
    for variante, c in conteos.items():
        c = c.reindex(fechas, fill_value=0.0)
        puntos[variante] = metricas_desde_sumas(c.sum())
        remuestreos[variante] = metricas_desde_sumas(
            {col: pesos @ c[col].to_numpy() for col in c.columns}
        )
    filas = []
    for variante in conteos:
        fila: dict = {"variante": variante, "n": len(variantes[variante])}
        for m in METRICAS:
            fila[m] = float(puntos[variante][m])
            fila[f"{m}_lo"], fila[f"{m}_hi"] = np.quantile(
                remuestreos[variante][m], [cola, 1 - cola]
            )
        if contra in remuestreos:
            for m in PAREADAS:
                dif = remuestreos[variante][m] - remuestreos[contra][m]
                fila[f"dif_{m}"] = float(puntos[variante][m] - puntos[contra][m])
                fila[f"dif_{m}_lo"], fila[f"dif_{m}_hi"] = np.quantile(dif, [cola, 1 - cola])
                fila[f"p_mejor_{m}"] = float(np.mean(dif > 0))
        filas.append(fila)
    return pd.DataFrame(filas).set_index("variante").sort_values("bal_usd", ascending=False)


def _intervalo(tabla: pd.DataFrame, m: str) -> pd.Series:
    return tabla.apply(lambda f: f"{f[m]:.3f} [{f[f'{m}_lo']:.3f}, {f[f'{m}_hi']:.3f}]", axis=1)


def legible(tabla: pd.DataFrame, contra: str | None = None) -> pd.DataFrame:
    """Tabla para imprimir: ``valor [lo, hi]`` por metrica y la diferencia contra
    la referencia."""
    salida = pd.DataFrame({m: _intervalo(tabla, m) for m in METRICAS})
    if contra and "dif_bal_usd" in tabla:
        for m in ("bal_usd", "bal_top", "ic"):
            salida[f"vs_{contra}_{m}"] = _intervalo(tabla, f"dif_{m}")
            salida[f"p_mejor_{m}"] = tabla[f"p_mejor_{m}"].round(3)
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Intervalos de las metricas de direccion.")
    parser.add_argument("--nombre", required=True)
    parser.add_argument("--snapshot", default=DEFAULT_DIR)
    parser.add_argument("--limpiar", action="store_true", help="Precios sin saltos (limpieza).")
    parser.add_argument("--periodos", choices=sorted(PERIODOS), default="eleccion")
    parser.add_argument(
        "--contra", default=REFERENCIA_REVERSION, help="Variante para la diferencia pareada."
    )
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--bloque", type=int, default=BLOQUE, help="Ruedas por bloque.")
    args = parser.parse_args()
    salida = DIR_EXPERIMENTOS / args.nombre
    snapshot = cargar_snapshot(args.snapshot, args.limpiar)
    retornos = {h: retornos_usd(snapshot, h) for h in {HORIZON, *horizontes(salida).values()}}
    variantes, base = predicciones(salida, retornos, args.periodos)
    if len(base):
        variantes |= referencias(base)
    tabla = significancia(variantes, args.contra, args.n_boot, args.bloque)
    tabla.to_csv(salida / f"significancia_{args.periodos}.csv")
    print(f"Intervalos al {NIVEL:.0%}, bootstrap de bloques de {args.bloque} ruedas")
    with pd.option_context("display.width", 300, "display.max_columns", 20):
        print(legible(tabla, args.contra))


if __name__ == "__main__":
    main()
