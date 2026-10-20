"""Reporte en PDF de una corrida de ``un_modelo``: balance de las carteras, direccion y compras.

Lee lo que ``un_modelo`` dejo en ``simulacion/resultados/<nombre>/`` (no vuelve
a correr nada) y escribe en ``<nombre>/reporte/``:

- ``graficos/``: balance de cada cartera del modelo contra los benchmarks (en
  pesos y en dolares), drawdown en dolares, comisiones pagadas, IC con su
  intervalo contra las referencias y la certeza / fuerza de las compras.
- ``reporte.tex`` y, si hay ``pdflatex``, ``reporte.pdf``.

Uso:
    python -m src.simulacion.reporte_un_modelo --nombre v2-rel-2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulacion.mesa import (
    BENCHMARK_BUY_HOLD,
    BENCHMARK_MERVAL,
    BENCHMARK_REBALANCEO,
    REGLAS_PANEL,
    REGLAS_SIMPLES,
)
from src.simulacion.reporte import (
    COLOR_REGLA,
    ESTILO_BENCHMARK,
    ETIQUETAS_BENCHMARK,
    GRIS_BENCHMARK,
    PREAMBULO_TEX,
    TINTA,
    TINTA_SECUNDARIA,
    _estilo,
    _etiquetas_directas,
    _num,
    _pct,
    _tabla_tex,
    _tex,
    compilar,
)
from src.simulacion.simulador import RESULTADOS_DIR

REGLAS = ("senal", *REGLAS_SIMPLES, *REGLAS_PANEL)
COLORES = {**COLOR_REGLA, "mantiene": "#8250df"}
ETIQUETAS_REGLA = {
    "senal": "Senal (compra las 'alza')",
    "mantiene": "Senal, mantiene neutrales",
    "histeresis": "Histeresis",
    "cuantil30": "Cuantil 30%",
    "inclinado": "Inclinada",
}
DESCRIPCION_REGLA = {
    "senal": "partes iguales entre las 'alza' (vende las 'neutral' y las 'baja')",
    **REGLAS_SIMPLES,
    **REGLAS_PANEL,
}
BENCHMARKS = (BENCHMARK_REBALANCEO, BENCHMARK_BUY_HOLD, BENCHMARK_MERVAL)
# Referencias de ``direccion`` que no son reglas posibles (cotas) o no ordenan.
FUERA_DEL_GRAFICO = ("oraculo_mercado", "oraculo_relativo", "siempre_alza", "mercado_sigue",
                     "mercado_revierte")  # fmt: skip
REFERENCIA = "reversion_usd"


def _leer_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True, dtype=False, convert_dates=False)


def _leer_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() and path.stat().st_size else pd.DataFrame()


def cargar(out: Path) -> dict:
    datos = {"config": json.loads((out / "config.json").read_text())}
    for nombre in ("valores", "operaciones", "entrenamientos"):
        datos[nombre] = _leer_jsonl(out / f"{nombre}.jsonl")
    datos["carteras"] = _leer_csv(out / "carteras.csv")
    datos["direccion"] = _leer_csv(out / "direccion.csv", index_col=0)
    datos["por_senal"] = _leer_csv(out / "compra_por_senal.csv")
    datos["confianza"] = _leer_csv(out / "compra_confianza.csv")
    return datos


def _series(modelo: str) -> list[tuple[str, str, str, object]]:
    """``(columna, etiqueta, color, estilo)`` de las carteras del modelo y los benchmarks."""
    series = [(f"{r}_{modelo}", ETIQUETAS_REGLA[r], COLORES[r], "-") for r in REGLAS]
    return series + [
        (b, ETIQUETAS_BENCHMARK[b], GRIS_BENCHMARK, ESTILO_BENCHMARK[b]) for b in BENCHMARKS
    ]


# --------------------------------------------------------------------------- #
# Graficos
# --------------------------------------------------------------------------- #
def graficos(datos: dict, destino: Path) -> list[str]:
    """Genera los PNG en ``destino`` y devuelve sus nombres, en orden."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    destino.mkdir(parents=True, exist_ok=True)
    generados: list[str] = []
    modelo = datos["config"]["solo_modelo"]

    def figura(titulo: str, subtitulo: str, derecha: float = 0.76, izquierda: float = 0.09,
               alto: float = 4.6):  # fmt: skip
        fig, ax = plt.subplots(figsize=(9, alto), dpi=150)
        fig.subplots_adjust(right=derecha, top=1 - 0.69 / alto, left=izquierda)
        fig.text(0.09, 0.95, titulo, fontsize=12, color=TINTA, weight="bold")
        fig.text(0.09, 0.905, subtitulo, fontsize=8.5, color=TINTA_SECUNDARIA)
        _estilo(ax)
        return fig, ax

    def guardar(fig, nombre: str) -> None:
        fig.savefig(destino / nombre, facecolor="white")
        plt.close(fig)
        generados.append(nombre)

    def curvas(ax, tabla: pd.DataFrame, formato) -> None:
        finales = []
        for columna, etiqueta, color, estilo in _series(modelo):
            if columna not in tabla:
                continue
            ancho = 1.2 if color == GRIS_BENCHMARK else 2.0
            ax.plot(tabla.index, tabla[columna], color=color, linestyle=estilo, linewidth=ancho)
            ultimo = float(tabla[columna].iloc[-1])
            finales.append((ultimo, f"{etiqueta}  {formato(ultimo)}", color, tabla.index[-1]))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%y"))
        _etiquetas_directas(ax, finales)

    valores = datos["valores"]
    if not valores.empty:
        valores = valores.assign(fecha=pd.to_datetime(valores["fecha"])).set_index("fecha")
        ccl = valores.pop("ccl").astype(float).ffill().bfill()
        valores = valores.astype(float)
        capital = datos["config"]["capital"]
        comision = datos["config"]["comision"]

        fig, ax = figura(
            "Balance de la cuenta en pesos",
            f"Capital inicial ${capital:,.0f}".replace(",", ".")
            + f", comision {comision:.2%} por operacion. Gris: benchmarks.",
        )
        curvas(ax, valores / 1e6, lambda v: f"${v:.2f} M")
        ax.axhline(capital / 1e6, color="#d0d7de", linewidth=0.8, zorder=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("$%.2f M"))
        guardar(fig, "balance_pesos.png")

        usd = valores.div(ccl, axis=0)
        fig, ax = figura("Balance de la cuenta en dolares CCL",
                         "El mismo balance dividido por el CCL de cada dia.")  # fmt: skip
        curvas(ax, usd, lambda v: f"US${v:,.0f}".replace(",", "."))
        ax.axhline(float(usd.iloc[0].mean()), color="#d0d7de", linewidth=0.8, zorder=0)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"US${v:,.0f}".replace(",", "."))
        )
        guardar(fig, "balance_usd.png")

        caida = usd / usd.cummax() - 1
        fig, ax = figura("Drawdown en dolares", "Cuanto llego a estar por debajo de su maximo.")
        curvas(ax, caida, lambda v: f"{v:.0%}")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        guardar(fig, "drawdown_usd.png")

    operaciones = datos["operaciones"]
    if not operaciones.empty:
        capital = datos["config"]["capital"]
        pagado = operaciones.groupby("cartera")["comision"].sum() / capital
        filas = [(ETIQUETAS_REGLA[r], pagado.get(f"{r}_{modelo}", 0.0), COLORES[r]) for r in REGLAS]
        filas += [(ETIQUETAS_BENCHMARK[b], pagado.get(b, 0.0), GRIS_BENCHMARK)
                  for b in (BENCHMARK_REBALANCEO, BENCHMARK_BUY_HOLD)]  # fmt: skip
        fig, ax = figura(
            "Comisiones pagadas",
            "Suma de comisiones sobre el capital inicial.",
            0.95,
            0.3,
            1.2 + 0.45 * len(filas),
        )
        etiquetas, montos, colores = zip(*filas, strict=True)
        barras = ax.barh(etiquetas, montos, color=colores)
        ax.bar_label(barras, labels=[f"{m:.1%}" for m in montos], fontsize=8, padding=3,
                     color=TINTA)  # fmt: skip
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#eaeef2", linewidth=0.8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        guardar(fig, "comisiones.png")

    direccion = datos["direccion"]
    if not direccion.empty:
        tabla = direccion.drop(index=[v for v in FUERA_DEL_GRAFICO if v in direccion.index])
        tabla = tabla.sort_values("ic")
        fig, ax = figura("Orden de las acciones (IC)",
                         "Spearman por dia entre lo predicho y lo que paso, con intervalo al 90%. "
                         "0 = azar.", 0.95, 0.22, 1.4 + 0.45 * len(tabla))  # fmt: skip
        y = np.arange(len(tabla))
        colores = [COLOR_REGLA["senal"] if v == modelo else GRIS_BENCHMARK for v in tabla.index]
        error = np.clip([tabla["ic"] - tabla["ic_lo"], tabla["ic_hi"] - tabla["ic"]], 0, None)
        ax.errorbar(tabla["ic"], y, xerr=error, fmt="none", ecolor="#c0c7cf", elinewidth=2,
                    capsize=0)  # fmt: skip
        ax.scatter(tabla["ic"], y, color=colores, s=40, zorder=3)
        ax.axvline(0, color=TINTA_SECUNDARIA, linewidth=0.8)
        ax.set_yticks(y, tabla.index)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#eaeef2", linewidth=0.8)
        guardar(fig, "ic.png")

    confianza = datos["confianza"]
    por_senal = datos["por_senal"]
    if not confianza.empty and not por_senal.empty:
        base = por_senal.set_index("grupo").loc["todas"]
        fig, (izq, der) = plt.subplots(1, 2, figsize=(9, 4.0), dpi=150)
        fig.subplots_adjust(top=0.8, wspace=0.3, left=0.09, right=0.97)
        fig.text(0.09, 0.95, "Si solo compra cuando esta mas seguro", fontsize=12, color=TINTA,
                 weight="bold")  # fmt: skip
        fig.text(0.09, 0.9, "Entre las predicciones de suba, el x% mas fuerte. Linea: todas las "
                 "acciones (tasa base).", fontsize=8.5, color=TINTA_SECUNDARIA)  # fmt: skip
        x = [f"{c:.0%}" for c in confianza["cobertura"]]
        for ax, columna, referencia, titulo in (
            (izq, "subio", base["subio"], "Certeza: cuantas subieron"),
            (der, "ret_medio", base["ret_medio"], "Fuerza: retorno medio"),
        ):
            _estilo(ax)
            barras = ax.bar(x, confianza[columna], color=COLOR_REGLA["senal"])
            ax.axhline(referencia, color=TINTA_SECUNDARIA, linewidth=1, linestyle="--")
            formato = "{:.0%}" if columna == "subio" else "{:+.1%}"
            ax.bar_label(barras, labels=[formato.format(v) for v in confianza[columna]],
                         fontsize=8, padding=2, color=TINTA)  # fmt: skip
            ax.set_title(titulo, fontsize=9.5, color=TINTA, loc="left")
            ax.set_xlabel("compras mas fuertes (cobertura)", fontsize=8, color=TINTA_SECUNDARIA)
            decimales = 0 if columna == "subio" else 1
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(1.0, decimals=decimales)
            )
        guardar(fig, "confianza.png")
    return generados


# --------------------------------------------------------------------------- #
# Documento
# --------------------------------------------------------------------------- #
def _lectura(datos: dict) -> list[str]:
    """Frases con lo principal, armadas a partir de los numeros."""
    modelo = datos["config"]["solo_modelo"]
    frases = []
    carteras = datos["carteras"]
    if not carteras.empty and "retorno_total_usd" in carteras:
        c = carteras.set_index("cartera")["retorno_total_usd"]
        propias = c[[f"{r}_{modelo}" for r in REGLAS if f"{r}_{modelo}" in c.index]]
        if len(propias) and BENCHMARK_REBALANCEO in c.index:
            mejor = propias.idxmax()
            regla = mejor.removesuffix(f"_{modelo}")
            ew = c[BENCHMARK_REBALANCEO]
            veredicto = "le gana" if propias.max() > ew else "no le gana"
            frases.append(
                f"La mejor cartera del modelo es '{ETIQUETAS_REGLA[regla]}' con "
                f"{propias.max():+.1%} en dolares; {veredicto} a tener el panel entero en partes "
                f"iguales ({ew:+.1%})."
            )
            ganan = [
                ETIQUETAS_REGLA[k.removesuffix(f"_{modelo}")] for k, v in propias.items() if v > ew
            ]
            frases.append(
                f"Carteras del modelo por encima del equal weight: {len(ganan)} de {len(propias)}"
                + (f" ({', '.join(ganan)})." if ganan else ".")
            )
    direccion = datos["direccion"]
    if not direccion.empty and modelo in direccion.index:
        f = direccion.loc[modelo]
        if "dif_ic" in f and not np.isnan(f["dif_ic"]):
            if f["dif_ic_lo"] > 0:
                veredicto = "ordena mejor que la regla de reversion"
            elif f["dif_ic_hi"] < 0:
                veredicto = "ordena peor que la regla de reversion"
            else:
                veredicto = "empata con la regla de reversion"
            frases.append(
                f"Orden de las acciones: IC {f['ic']:.3f} [{f['ic_lo']:.3f}, {f['ic_hi']:.3f}]; "
                f"{veredicto} (diferencia {f['dif_ic']:+.3f} [{f['dif_ic_lo']:+.3f}, "
                f"{f['dif_ic_hi']:+.3f}])."
            )
        frases.append(
            f"Direccion (sube o baja en dolares): acierto balanceado {f['bal_usd']:.3f}; 0,5 es "
            "una moneda."
        )
    por_senal = datos["por_senal"]
    if not por_senal.empty and "alza" in set(por_senal["grupo"]):
        g = por_senal.set_index("grupo")
        alza, base = g.loc["alza"], g.loc["todas"]
        frases.append(
            f"Cuando dijo 'alza' ({int(alza['casos'])} casos) subio el {alza['subio']:.0%} de las "
            f"veces (tasa base {base['subio']:.0%}), con un retorno medio de "
            f"{alza['ret_medio']:+.2%} ({alza['exceso_medio']:+.2%} sobre el promedio del panel)."
        )
    return frases


def latex(datos: dict, imagenes: list[str]) -> str:
    cfg = datos["config"]
    modelo = cfg["solo_modelo"]

    def figura(nombre: str) -> str:
        if nombre not in imagenes:
            return ""
        imagen = rf"\includegraphics[width=\linewidth]{{graficos/{nombre}}}"
        return f"\\begin{{figure}}[H]\n\\centering\n{imagen}\n\\end{{figure}}"

    def lista(items: list[str]) -> str:
        if not items:
            return ""
        return (
            "\\begin{itemize}\n"
            + "\n".join(rf"  \item {_tex(t)}" for t in items)
            + "\n\\end{itemize}"
        )

    partes = [
        PREAMBULO_TEX,
        rf"{{\LARGE\bfseries Simulacion de {_tex(modelo)}: {_tex(cfg['nombre'])}\par}}",
        _tex(
            f"Del {cfg['desde']} al {cfg['hasta']}. Modelo `{modelo}`, "
            f"reentreno {cfg['reentreno']}, "
            f"{cfg['historia']} ruedas de historia, config `{json.dumps(cfg['config'])}`"
            + (f", senal por cuantil {cfg['cuantil']}" if cfg.get("cuantil") else "")
            + f". Capital inicial ${cfg['capital']:,.0f}, comision {cfg['comision']:.2%} por "
            f"operacion, rebalanceo cada {cfg['rebalanceo']} ruedas."
        ),
        r"\section*{Lo principal}",
        lista(_lectura(datos)),
        r"\section*{Balance de las carteras}",
        lista(
            [f"{ETIQUETAS_REGLA[r]}: {DESCRIPCION_REGLA[r]}." for r in REGLAS]
            + [
                "Benchmarks: el panel entero en partes iguales (con rebalanceo y comprado una vez) "
                "y el Merval."
            ]
        ),  # fmt: skip
        figura("balance_pesos.png"),
        figura("balance_usd.png"),
        figura("drawdown_usd.png"),
        figura("comisiones.png"),
    ]
    carteras = datos["carteras"]
    if not carteras.empty:
        filas = [
            [f["cartera"], _pct(f.get("retorno_total")), _pct(f.get("retorno_total_usd")),
             _pct(f.get("max_drawdown_usd")), _pct(f.get("volatilidad_anualizada")),
             _num(f.get("sharpe"))]
            for f in carteras.to_dict("records")
        ]  # fmt: skip
        partes.append(_tabla_tex(["Cartera", "Pesos", "Dolares", "Drawdown USD", "Vol. anual",
                                  "Sharpe"], filas))  # fmt: skip
    partes.append(r"\section*{Certeza y fuerza de las compras}")
    partes.append(_tex("Retornos en dolares a 5 ruedas (o al horizonte del modelo)."))
    por_senal = datos["por_senal"]
    if not por_senal.empty:
        filas = [[f["grupo"], str(int(f["casos"])), _pct(f["subio"]).lstrip("+"),
                  _pct(f["ret_medio"]), _pct(f["exceso_medio"])]
                 for f in por_senal.to_dict("records")]  # fmt: skip
        partes.append(_tabla_tex(["Senal", "Casos", "Subio", "Retorno medio",
                                  "Exceso sobre el panel"], filas))  # fmt: skip
    partes.append(figura("confianza.png"))
    partes.append(r"\section*{Orden y direccion}")
    partes.append(figura("ic.png"))
    direccion = datos["direccion"]
    if not direccion.empty:
        filas = []
        for v, f in direccion.iterrows():
            con_dif = "dif_ic" in f and not np.isnan(f["dif_ic"]) and v != REFERENCIA
            dif = (f"{f['dif_ic']:+.3f} [{f['dif_ic_lo']:+.3f}, {f['dif_ic_hi']:+.3f}]"
                   if con_dif else "-")  # fmt: skip
            filas.append([v, f"{f['ic']:.3f} [{f['ic_lo']:.3f}, {f['ic_hi']:.3f}]",
                          _num(f["bal_usd"], 3), _num(f["bal_top"], 3), _num(f["bal_mercado"], 3),
                          _num(f["bal_relativo"], 3), dif])  # fmt: skip
        partes.append(_tabla_tex(["Variante", "IC", "bal_usd", "bal_top", "bal_mercado",
                                  "bal_relativo", "IC vs reversion"], filas))  # fmt: skip
    entrenamientos = datos["entrenamientos"]
    if not entrenamientos.empty:
        partes.append(r"\section*{Entrenamientos}")
        val = [c for c in entrenamientos.columns if c.startswith("val_")]
        filas = [[str(f["fecha"]), _num(f.get("segundos"), 0)] + [_num(f[c], 4) for c in val]
                 for f in entrenamientos.to_dict("records")]  # fmt: skip
        partes.append(_tabla_tex(["Datos hasta", "Segundos", *val], filas))
    partes.append(r"\end{document}")
    return "\n\n".join(p for p in partes if p) + "\n"


def generar(out: Path) -> Path:
    """Escribe el reporte de la corrida ``out``; devuelve el PDF (o el ``.tex`` si
    no se pudo compilar)."""
    datos = cargar(out)
    destino = out / "reporte"
    imagenes = graficos(datos, destino / "graficos")
    (destino / "reporte.tex").write_text(latex(datos, imagenes))
    return compilar(destino) or destino / "reporte.tex"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reporte en PDF de una corrida de un_modelo.")
    parser.add_argument("--nombre", required=True, help="Corrida de un_modelo.")
    args = parser.parse_args(argv)
    print(f"Reporte en {generar(Path(RESULTADOS_DIR) / args.nombre)}")


if __name__ == "__main__":
    main()
