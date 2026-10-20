"""Informe PDF de una busqueda de ``optuna_v2``: como fue la busqueda y si el ganador es real.

Lee el estudio (``estudio.db``) y lo que dejo ``cerrar`` (``top.csv``,
``evaluacion.csv``, ``resumen.json``) y escribe en ``<salida>/reporte/`` los
graficos, ``reporte.tex`` y, si hay ``pdflatex``, ``reporte.pdf``.

Uso:
    python -m src.simulacion.reporte_optuna --nombre optuna-v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.simulacion.busqueda import DIR_EXPERIMENTOS
from src.simulacion.reporte import (
    GRIS_BENCHMARK,
    PREAMBULO_TEX,
    TINTA,
    TINTA_SECUNDARIA,
    _estilo,
    _num,
    _tabla_tex,
    _tex,
)
from src.simulacion.reporte_un_modelo import compilar
from src.simulacion.significancia import legible

COLOR_HORIZONTE = {5: "#2a78d6", 10: "#eb6834", 20: "#1baf7a"}
PARAMETROS_CONTINUOS = ("learning_rate", "weight_decay", "dropout")


def cargar(salida: Path) -> dict:
    s = optuna.load_study(study_name="v2", storage=f"sqlite:///{salida / 'estudio.db'}")
    datos = {"estudio": s, "resumen": json.loads((salida / "resumen.json").read_text())}
    for nombre in ("top", "evaluacion"):
        archivo = salida / f"{nombre}.csv"
        kwargs = {"index_col": 0} if nombre == "evaluacion" else {}
        datos[nombre] = pd.read_csv(archivo, **kwargs) if archivo.exists() else pd.DataFrame()
    return datos


def _trials(s: optuna.Study) -> pd.DataFrame:
    filas = []
    for t in s.trials:
        difs = t.user_attrs.get("difs", [])
        filas.append(
            {
                "numero": t.number,
                "estado": t.state.name,
                "valor": t.value,
                "parcial": float(np.mean(difs)) if difs else np.nan,
                "folds": len(difs),
                **t.params,
            }
        )
    return pd.DataFrame(filas)


def graficos(datos: dict, destino: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    destino.mkdir(parents=True, exist_ok=True)
    generados: list[str] = []
    trials = _trials(datos["estudio"])
    completos = trials[trials["estado"] == "COMPLETE"]

    def figura(titulo: str, subtitulo: str, alto: float = 4.4, izquierda: float = 0.09):
        fig, ax = plt.subplots(figsize=(9, alto), dpi=150)
        fig.subplots_adjust(top=1 - 0.7 / alto, left=izquierda, right=0.97)
        fig.text(izquierda, 0.95, titulo, fontsize=12, color=TINTA, weight="bold")
        fig.text(izquierda, 0.95 - 0.2 / alto, subtitulo, fontsize=8.5, color=TINTA_SECUNDARIA)
        _estilo(ax)
        return fig, ax

    def guardar(fig, nombre: str) -> None:
        fig.savefig(destino / nombre, facecolor="white")
        plt.close(fig)
        generados.append(nombre)

    if not completos.empty:
        fig, ax = figura(
            "Historia de la busqueda",
            "Cada punto es un trial completo (IC del modelo menos IC de la reversion, "
            "promedio de los folds). Linea: mejor hasta ese momento.",
        )
        podados = trials[trials["estado"] == "PRUNED"]
        ax.scatter(
            podados["numero"],
            podados["parcial"],
            marker="x",
            s=14,
            color=GRIS_BENCHMARK,
            label="podado (valor parcial)",
        )
        for h, g in completos.groupby("horizonte"):
            ax.scatter(
                g["numero"],
                g["valor"],
                s=22,
                color=COLOR_HORIZONTE.get(h, TINTA),
                label=f"horizonte {h}",
            )
        mejor = completos.set_index("numero")["valor"].cummax()
        ax.step(mejor.index, mejor.to_numpy(), where="post", color=TINTA, linewidth=1.2)
        ax.axhline(0, color=TINTA_SECUNDARIA, linewidth=0.8, linestyle="--")
        ax.set_xlabel("trial", fontsize=8, color=TINTA_SECUNDARIA)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        guardar(fig, "historia.png")

        fig, ax = figura(
            "Resultado por horizonte",
            "Trials completos por horizonte. 0 = lo mismo que la regla de reversion.",
        )
        grupos = [(h, g["valor"].to_numpy()) for h, g in completos.groupby("horizonte")]
        posiciones = range(len(grupos))
        ax.boxplot(
            [v for _, v in grupos],
            positions=list(posiciones),
            widths=0.5,
            showfliers=False,
            medianprops={"color": TINTA},
        )
        for i, (h, v) in enumerate(grupos):
            x = np.random.default_rng(i).normal(i, 0.06, len(v))
            ax.scatter(x, v, s=16, color=COLOR_HORIZONTE.get(h, TINTA), zorder=3)
        ax.set_xticks(list(posiciones), [f"{h} ruedas" for h, _ in grupos])
        ax.axhline(0, color=TINTA_SECUNDARIA, linewidth=0.8, linestyle="--")
        guardar(fig, "horizontes.png")

        continuos = [p for p in PARAMETROS_CONTINUOS if p in completos]
        if continuos:
            fig, ejes = plt.subplots(1, len(continuos), figsize=(9, 3.4), dpi=150, sharey=True)
            ejes = np.atleast_1d(ejes)
            fig.subplots_adjust(top=0.78, left=0.09, right=0.97, wspace=0.08)
            fig.text(
                0.09,
                0.95,
                "Resultado segun cada hiperparametro",
                fontsize=12,
                color=TINTA,
                weight="bold",
            )
            for ax, p in zip(ejes, continuos, strict=True):
                _estilo(ax)
                colores = [COLOR_HORIZONTE.get(h, TINTA) for h in completos["horizonte"]]
                ax.scatter(completos[p], completos["valor"], s=14, color=colores)
                if p != "dropout":
                    ax.set_xscale("log")
                ax.set_title(p, fontsize=9, color=TINTA, loc="left")
                ax.axhline(0, color=TINTA_SECUNDARIA, linewidth=0.8, linestyle="--")
            guardar(fig, "parametros.png")

    try:
        importancias = optuna.importance.get_param_importances(datos["estudio"])
    except (RuntimeError, ValueError):
        importancias = {}
    if importancias:
        nombres = list(importancias)[::-1]
        fig, ax = figura(
            "Importancia de cada hiperparametro",
            "Fraccion de la variacion del objetivo que explica cada uno (fANOVA).",
            1.2 + 0.32 * len(nombres),
            0.25,
        )
        barras = ax.barh(nombres, [importancias[n] for n in nombres], color="#2a78d6")
        ax.bar_label(
            barras,
            labels=[f"{importancias[n]:.0%}" for n in nombres],
            fontsize=8,
            padding=3,
            color=TINTA,
        )
        ax.grid(axis="y", visible=False)
        guardar(fig, "importancia.png")

    top = datos["top"]
    if not top.empty:
        fig, ax = figura(
            "Top de la busqueda, reevaluado con otras semillas",
            "Gris: valor en la busqueda. Azul: media de todas las semillas, con su desvio.",
            1.2 + 0.45 * len(top),
            0.16,
        )
        y = np.arange(len(top))[::-1]
        ax.errorbar(
            top["media"],
            y,
            xerr=top["desvio"],
            fmt="o",
            color="#2a78d6",
            ecolor="#9ec5f5",
            elinewidth=2,
            capsize=0,
        )
        ax.scatter(top["busqueda"], y, color=GRIS_BENCHMARK, s=20, zorder=3)
        ax.set_yticks(
            y, [f"trial {n} (h{h})" for n, h in zip(top["trial"], top["horizonte"], strict=True)]
        )
        ax.axvline(0, color=TINTA_SECUNDARIA, linewidth=0.8, linestyle="--")
        ax.grid(axis="y", visible=False)
        guardar(fig, "top.png")

    evaluacion = datos["evaluacion"]
    if not evaluacion.empty:
        tabla = evaluacion.loc[
            [v for v in ("v2_optuna", "ensamble", "reversion_usd") if v in evaluacion.index]
        ].iloc[::-1]
        fig, ax = figura(
            "Evaluacion 2024-2025 (nunca vista por la busqueda)",
            "IC con intervalo al 90%.",
            1.4 + 0.45 * len(tabla),
            0.16,
        )
        y = np.arange(len(tabla))
        error = np.clip([tabla["ic"] - tabla["ic_lo"], tabla["ic_hi"] - tabla["ic"]], 0, None)
        ax.errorbar(
            tabla["ic"],
            y,
            xerr=error,
            fmt="o",
            color="#2a78d6",
            ecolor="#c0c7cf",
            elinewidth=2,
            capsize=0,
        )
        ax.set_yticks(y, tabla.index)
        ax.axvline(0, color=TINTA_SECUNDARIA, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        guardar(fig, "evaluacion.png")
    return generados


def _lectura(datos: dict) -> list[str]:
    r = datos["resumen"]
    frases = [
        f"{r['trials']} trials: {r['completos']} completos, {r['podados']} podados por el pruner "
        f"y {r['fallidos']} fallidos."
    ]
    top = datos["top"]
    if not top.empty:
        primero = top.iloc[0]
        frases.append(
            f"Mejor configuracion despues de reevaluar: trial {int(primero['trial'])}, horizonte "
            f"{int(primero['horizonte'])}, IC sobre la reversion {primero['media']:+.3f} "
            f"(desvio entre semillas {primero['desvio']:.3f}; en la busqueda daba "
            f"{primero['busqueda']:+.3f})."
        )
    if r.get("pbo") is not None:
        lectura = (
            "alta: el ganador es probablemente sobreajuste"
            if r["pbo"] > 0.5
            else "baja"
            if r["pbo"] < 0.2
            else "moderada"
        )
        frases.append(
            f"PBO (probabilidad de sobreajuste de la busqueda): {r['pbo']:.0%}, {lectura}."
        )
    evaluacion = datos["evaluacion"]
    if not evaluacion.empty and "v2_optuna" in evaluacion.index:
        f = evaluacion.loc["v2_optuna"]
        if f["dif_ic_lo"] > 0:
            veredicto = "le gana a la regla de reversion"
        elif f["dif_ic_hi"] < 0:
            veredicto = "pierde contra la regla de reversion"
        else:
            veredicto = "empata con la regla de reversion"
        frases.append(
            f"En 2024-2025 (nunca visto por la busqueda) el V2 elegido da IC {f['ic']:.3f} "
            f"[{f['ic_lo']:.3f}, {f['ic_hi']:.3f}] y {veredicto} (diferencia {f['dif_ic']:+.3f} "
            f"[{f['dif_ic_lo']:+.3f}, {f['dif_ic_hi']:+.3f}])."
        )
        if "ensamble" in evaluacion.index:
            frases.append(
                f"El ensamble Ridge + reversion del mismo horizonte da IC "
                f"{evaluacion.loc['ensamble', 'ic']:.3f}."
            )
    return frases


def latex(datos: dict, imagenes: list[str]) -> str:
    def figura(nombre: str) -> str:
        if nombre not in imagenes:
            return ""
        imagen = rf"\includegraphics[width=\linewidth]{{graficos/{nombre}}}"
        return f"\\begin{{figure}}[H]\n\\centering\n{imagen}\n\\end{{figure}}"

    def lista(items: list[str]) -> str:
        cuerpo = "\n".join(rf"  \item {_tex(t)}" for t in items)
        return f"\\begin{{itemize}}\n{cuerpo}\n\\end{{itemize}}" if items else ""

    from src.simulacion.optuna_v2 import EVALUACION, FOLDS, HISTORIA

    elegida = datos["resumen"]["elegida"]
    partes = [
        PREAMBULO_TEX,
        r"{\LARGE\bfseries Busqueda de hiperparametros del Transformer V2\par}",
        _tex(
            f"Optuna (TPE multivariado, pruning por mediana) sobre el V2 relativo con hasta "
            f"{HISTORIA} ruedas de historia. Objetivo: IC del modelo menos IC de la regla de "
            f"reversion del mismo horizonte, promedio de {len(FOLDS)} folds walk-forward del "
            f"{FOLDS[0][0]} al {FOLDS[-1][1]}. Evaluacion final, una sola vez, del "
            f"{EVALUACION[0][0]} al {EVALUACION[-1][1]}."
        ),
        r"\section*{Lo principal}",
        lista(_lectura(datos)),
        r"\section*{La busqueda}",
        figura("historia.png"),
        figura("horizontes.png"),
        figura("importancia.png"),
        figura("parametros.png"),
        r"\section*{El ganador: reevaluacion y evaluacion}",
        figura("top.png"),
    ]
    top = datos["top"]
    if not top.empty:
        filas = [
            [
                str(int(f["trial"])),
                str(int(f["horizonte"])),
                _num(f["busqueda"], 3),
                _num(f["media"], 3),
                _num(f["desvio"], 3),
            ]
            for f in top.to_dict("records")
        ]
        partes.append(
            _tabla_tex(["Trial", "Horizonte", "Busqueda", "Media semillas", "Desvio"], filas)
        )
    partes.append(_tex("Configuracion elegida:"))
    partes.append(
        _tabla_tex(["Hiperparametro", "Valor"], [[k, str(v)] for k, v in elegida.items()])
    )
    partes.append(figura("evaluacion.png"))
    evaluacion = datos["evaluacion"]
    if not evaluacion.empty:
        tabla = legible(evaluacion, "reversion_usd")
        filas = [[v, *[str(x) for x in f.to_numpy()[:5]]] for v, f in tabla.iterrows()]
        partes.append(_tabla_tex(["Variante", *tabla.columns[:5]], filas))
    partes.append(r"\end{document}")
    return "\n\n".join(p for p in partes if p) + "\n"


def generar(salida: Path) -> Path:
    datos = cargar(salida)
    destino = salida / "reporte"
    imagenes = graficos(datos, destino / "graficos")
    (destino / "reporte.tex").write_text(latex(datos, imagenes))
    return compilar(destino) or destino / "reporte.tex"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Informe PDF de una busqueda de optuna_v2.")
    parser.add_argument("--nombre", required=True)
    args = parser.parse_args(argv)
    print(f"Reporte en {generar(DIR_EXPERIMENTOS / args.nombre)}")


if __name__ == "__main__":
    main()
