"""Reporte de una corrida: ``resumen.json``, ``reporte.md``, ``reporte.tex`` y graficos PNG.

Todo sale de los ``.jsonl`` que guarda el simulador (o ``rearmar``): se puede
regenerar las veces que haga falta sin volver a correr nada.

Uso:
    python -m src.simulacion.reporte --nombre desde-2026-06-semanal
    python -m src.simulacion.reporte --nombre desde-2026-06-semanal --rearmado comision-1pct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS
from src.simulacion.estrategias import PERFILES
from src.simulacion.mesa import (
    BENCHMARK_BUY_HOLD,
    BENCHMARK_MERVAL,
    BENCHMARK_REBALANCEO,
    MODELOS,
    MODELOS_PANEL,
    REGLAS_PANEL,
    SELECCIONES,
)
from src.simulacion.metricas import (
    calidad_modelos,
    metricas_cartera,
    promociones_por_modelo,
    resolver_predicciones,
    rotacion,
)
from src.simulacion.simulador import RESULTADOS_DIR, SIN_COMISION
from src.simulacion.snapshot import Snapshot

BENCHMARKS = (BENCHMARK_REBALANCEO, BENCHMARK_BUY_HOLD, BENCHMARK_MERVAL)
ETIQUETAS_BENCHMARK = {
    BENCHMARK_REBALANCEO: "Equal weight (rebalanceo)",
    BENCHMARK_BUY_HOLD: "Equal weight (buy & hold)",
    BENCHMARK_MERVAL: "Merval",
}
ETIQUETAS_PERFIL = {
    p.value: n for p, n in zip(PERFILES, ("Conservador", "Moderado", "Agresivo"), strict=True)
}

# Colores fijos por entidad (validados para daltonismo; con etiquetas directas).
COLOR_MODELO = {
    "lstm": "#2a78d6",
    "xgboost": "#eb6834",
    "transformer": "#1baf7a",
    "arima": "#eda100",
    "macro": "#e87ba4",
    "contexto": "#5b6b7f",
    "ensamble": "#8250df",
    "panel_transformer": "#0e7c86",
    "reversion": "#8a5a2b",
}
# Reglas de cartera de los modelos de panel (``senal_`` y ``REGLAS_PANEL``).
COLOR_REGLA = {
    "senal": "#2a78d6",
    "histeresis": "#eb6834",
    "cuantil30": "#1baf7a",
    "inclinado": "#e87ba4",
}
COLOR_PERFIL = {"conservative": "#4a3aa7", "moderate": "#008300", "aggressive": "#e34948"}
TINTA = "#1f2328"
TINTA_SECUNDARIA = "#59636e"
GRIS_BENCHMARK = "#8c959f"
ESTILO_MEJOR = {"mejor_prod": "-", "mejor_todos": "-."}
ESTILO_BENCHMARK = {
    BENCHMARK_REBALANCEO: "--",
    BENCHMARK_BUY_HOLD: ":",
    BENCHMARK_MERVAL: (0, (6, 2, 1, 2)),
}

LIMITACIONES = [
    "Se opera al precio de cierre del mismo dia en que se predice (sin slippage ni "
    "spread); la comision es una sola tasa sobre el monto operado.",
    "El efectivo no rinde interes: en pesos, con inflacion, castiga a las carteras que "
    "quedan parcialmente liquidas (las de senal y consenso).",
    "El universo es el de tickers que hoy existen en data-colector (sesgo de supervivencia).",
    "El modelo macro se entrena con el historico del snapshot (Yahoo, 1825 ruedas), no "
    "con el del data-colector que usa el cron de produccion.",
    "El consenso y la eleccion del mejor modelo usan solo los modelos locales de api_ml "
    "(sin los de Modal).",
    "En Black-Litterman con la seleccion 'todos', si en un ticker gana macro su vista es "
    "a 20 ruedas y se mezcla con vistas a 5 ruedas del resto.",
    "Las metricas son de un solo periodo de mercado: no alcanzan para afirmar que una "
    "estrategia es mejor que otra en general.",
]


def _leer_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True, dtype=False, convert_dates=False)


def _fechas(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "fecha" in df:
        df = df.assign(fecha=pd.to_datetime(df["fecha"]))
    return df


def cargar(corrida: Path, rearmado: str | None = None) -> dict[str, pd.DataFrame | dict]:
    """Archivos de la corrida; los de carteras salen del rearmado si se pide uno."""
    carteras = corrida / "rearmados" / rearmado if rearmado else corrida
    datos: dict[str, pd.DataFrame | dict] = {
        "config": json.loads((corrida / "config.json").read_text()),
        "config_carteras": json.loads((carteras / "config.json").read_text()),
    }
    for nombre in ("predicciones", "backtests", "promociones"):
        datos[nombre] = _fechas(_leer_jsonl(corrida / f"{nombre}.jsonl"))
    for nombre in ("valores", "operaciones", "elecciones", "errores"):
        datos[nombre] = _fechas(_leer_jsonl(carteras / f"{nombre}.jsonl"))
    return datos


def tabla_carteras(valores: pd.DataFrame) -> pd.DataFrame:
    """Metricas de cada cartera en pesos y en dolares CCL."""
    serie = valores.set_index("fecha")
    ccl = serie["ccl"].astype(float).ffill().bfill() if "ccl" in serie else None
    filas = []
    for cartera in serie.columns.drop("ccl", errors="ignore"):
        valor = serie[cartera].astype(float)
        fila = {"cartera": cartera, **metricas_cartera(valor)}
        if ccl is not None and ccl.notna().all():
            usd = metricas_cartera(valor / ccl)
            fila["retorno_total_usd"] = usd.get("retorno_total")
            fila["max_drawdown_usd"] = usd.get("max_drawdown")
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("retorno_total", ascending=False).reset_index(drop=True)


def calidad(datos: dict, snapshot: Snapshot) -> dict:
    """Calidad de las predicciones ya resueltas, en todo el universo y en el panel lider."""
    predicciones = datos["predicciones"]
    if predicciones.empty:
        return {}
    hasta = str(predicciones["fecha"].max().date())
    resueltas = resolver_predicciones(predicciones, snapshot, hasta)
    return {
        "todos_los_tickers": calidad_modelos(resueltas),
        "panel_lider": calidad_modelos(resueltas, universo=PANEL_LIDER_TICKERS),
    }


def resumen_elecciones(elecciones: pd.DataFrame) -> dict:
    """Que parte de las elecciones (ticker x rebalanceo) gano cada modelo."""
    if elecciones.empty:
        return {}
    resultado = {}
    for seleccion, grupo in elecciones.groupby("seleccion"):
        conteo = grupo["model"].value_counts()
        resultado[seleccion] = {m: round(float(n / conteo.sum()), 3) for m, n in conteo.items()}
    return resultado


def resumen(datos: dict, snapshot: Snapshot) -> dict:
    capital = float(datos["config_carteras"]["capital"])
    valores = datos["valores"]
    return {
        "config": datos["config"],
        "config_carteras": datos["config_carteras"],
        "periodo": {
            "desde": str(valores["fecha"].min().date()),
            "hasta": str(valores["fecha"].max().date()),
            "ruedas": len(valores),
        },
        "carteras": tabla_carteras(valores).to_dict("records"),
        "rotacion": rotacion(datos["operaciones"], capital),
        "calidad_modelos": calidad(datos, snapshot),
        "promociones": promociones_por_modelo(datos["promociones"]),
        "elecciones": resumen_elecciones(datos["elecciones"]),
        "errores": int(len(datos["errores"])),
    }


# ---------------------------------------------------------------------- #
# Graficos
# ---------------------------------------------------------------------- #
def _estilo(ax) -> None:
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d0d7de")
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=8)
    ax.grid(axis="y", color="#eaeef2", linewidth=0.8)
    ax.set_axisbelow(True)


def _etiquetas_directas(ax, finales: list[tuple[float, str, str, object]]) -> None:
    """Etiqueta cada serie en el margen derecho, a la altura de su ultimo valor.

    ``finales`` son ``(ultimo valor, texto, color, x del ultimo punto)``. Si dos
    etiquetas se pisan se separan, y una guia fina une cada una con el final
    de su linea; el punto de color al lado del texto identifica la serie.
    """
    if not finales:
        return
    finales = sorted(finales, key=lambda f: f[0])
    bajo, alto = ax.get_ylim()
    separacion = (alto - bajo) * 0.05
    posiciones: list[float] = []
    for valor, *_ in finales:
        y = valor if not posiciones else max(valor, posiciones[-1] + separacion)
        posiciones.append(y)
    exceso = posiciones[-1] - alto
    if exceso > 0:
        posiciones = [p - exceso for p in posiciones]
    margen = ax.get_yaxis_transform()  # x en fraccion del eje, y en datos
    for (valor, texto, color, x_final), y in zip(finales, posiciones, strict=True):
        ax.annotate(
            "",
            xy=(x_final, valor),
            xycoords="data",
            xytext=(1.01, y),
            textcoords=margen,
            arrowprops={"arrowstyle": "-", "color": "#c0c7cf", "linewidth": 0.6},
            annotation_clip=False,
        )
        ax.plot([1.02], [y], marker="o", markersize=4.5, color=color, transform=margen,
                clip_on=False)  # fmt: skip
        ax.text(1.035, y, texto, transform=margen, fontsize=7.5, color=TINTA, va="center")


def _curvas(ax, base100: pd.DataFrame, series: list[tuple[str, str, str, object]]) -> None:
    import matplotlib.dates

    finales = []
    for columna, etiqueta, color, estilo in series:
        if columna not in base100:
            continue
        ancho = 1.2 if color == GRIS_BENCHMARK else 2.0
        ax.plot(base100.index, base100[columna], color=color, linestyle=estilo, linewidth=ancho)
        finales.append((float(base100[columna].iloc[-1]), etiqueta, color, base100.index[-1]))
    ax.axhline(100, color="#d0d7de", linewidth=0.8, zorder=0)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))
    _etiquetas_directas(ax, finales)


def _series_benchmark() -> list[tuple[str, str, str, object]]:
    return [(b, ETIQUETAS_BENCHMARK[b], GRIS_BENCHMARK, ESTILO_BENCHMARK[b]) for b in BENCHMARKS]


def _series_por_modelo(prefijo: str, sufijo: str = "") -> list[tuple[str, str, str, object]]:
    series = [(f"{prefijo}{m}{sufijo}", m, COLOR_MODELO[m], "-") for m in MODELOS + MODELOS_PANEL]
    series += [
        (f"{prefijo}mejor_{s}{sufijo}", f"mejor ({s})", TINTA, ESTILO_MEJOR[f"mejor_{s}"])
        for s in SELECCIONES
    ]
    return series + _series_benchmark()


def _figura(titulo: str, subtitulo: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.subplots_adjust(right=0.78, top=0.86)
    fig.text(0.07, 0.95, titulo, fontsize=12, color=TINTA, weight="bold")
    fig.text(0.07, 0.905, subtitulo, fontsize=8.5, color=TINTA_SECUNDARIA)
    _estilo(ax)
    return fig, ax


def graficos(datos: dict, resumen_: dict, destino: Path) -> list[str]:
    """Genera los PNG y devuelve sus nombres."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    destino.mkdir(parents=True, exist_ok=True)
    generados: list[str] = []

    def guardar(fig, nombre: str) -> None:
        fig.savefig(destino / nombre, facecolor="white")
        plt.close(fig)
        generados.append(nombre)

    valores = datos["valores"].set_index("fecha").drop(columns="ccl", errors="ignore")
    base100 = valores.astype(float) / valores.astype(float).iloc[0] * 100
    subtitulo = "Valor de la cartera, base 100 al inicio (pesos). Gris: benchmarks."

    fig, ax = _figura("Senal de cada modelo: compra en partes iguales lo que da 'alza'", subtitulo)
    _curvas(ax, base100, _series_por_modelo("senal_"))
    guardar(fig, "carteras_senal.png")

    for perfil in PERFILES:
        etiqueta = ETIQUETAS_PERFIL[perfil.value]
        fig, ax = _figura(f"Black-Litterman, perfil {etiqueta.lower()}", subtitulo)
        _curvas(ax, base100, _series_por_modelo("bl_", f"_{perfil.value}"))
        guardar(fig, f"carteras_bl_{perfil.value}.png")

    for modelo in MODELOS_PANEL:
        if f"senal_{modelo}" not in base100:
            continue
        fig, ax = _figura(f"Reglas de cartera con las senales de {modelo}", subtitulo)
        series = [(f"{r}_{modelo}", r, COLOR_REGLA[r], "-") for r in ("senal", *REGLAS_PANEL)]
        _curvas(ax, base100, series + _series_benchmark())
        guardar(fig, f"carteras_reglas_{modelo}.png")

    fig, ax = _figura("Consenso de modelos por perfil", subtitulo)
    series = [
        (f"consenso_{p.value}", ETIQUETAS_PERFIL[p.value], COLOR_PERFIL[p.value], "-")
        for p in PERFILES
    ]
    _curvas(ax, base100, series + _series_benchmark())
    guardar(fig, "carteras_consenso.png")

    accuracy = _accuracy_mensual(resumen_)
    if not accuracy.empty:
        fig, ax = _figura(
            "Accuracy direccional mensual por modelo",
            "Predicciones resueltas, todos los tickers. Macro: solo predicciones con vista.",
        )
        finales = []
        for modelo in accuracy.columns:
            serie = accuracy[modelo].dropna()
            ax.plot(serie.index, serie.values, color=COLOR_MODELO[modelo], linewidth=2,
                    marker="o", markersize=5)  # fmt: skip
            if not serie.empty:
                finales.append(
                    (float(serie.iloc[-1]), modelo, COLOR_MODELO[modelo], serie.index[-1])
                )
        ax.axhline(0.5, color=GRIS_BENCHMARK, linestyle="--", linewidth=1)
        ax.text(accuracy.index[0], 0.5, " azar (50%)", fontsize=7.5, color=TINTA_SECUNDARIA,
                va="bottom")  # fmt: skip
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
        _etiquetas_directas(ax, finales)
        guardar(fig, "accuracy_mensual.png")

    promociones = datos["promociones"]
    if not promociones.empty and "directional_accuracy" in promociones:
        con_score = promociones[promociones["directional_accuracy"].notna()]
        modelos = [m for m in MODELOS if m in set(con_score["model"])]
        if modelos:
            fig, axes = plt.subplots(len(modelos), 1, figsize=(9, 2.2 * len(modelos)), dpi=150,
                                     sharex=True, squeeze=False)  # fmt: skip
            fig.suptitle("Gate de promocion: candidato vs vigente", fontsize=12, color=TINTA,
                         weight="bold", x=0.07, ha="left")  # fmt: skip
            for ax, modelo in zip(axes[:, 0], modelos, strict=True):
                _grafico_gate(ax, promociones[promociones["model"] == modelo], modelo)
            fig.tight_layout()
            guardar(fig, "promociones.png")

    elecciones = resumen_.get("elecciones") or {}
    if elecciones:
        fig, ax = _figura(
            "Mejor modelo por ticker: quien gano cada eleccion",
            "Proporcion de elecciones (ticker x rebalanceo) que gano cada modelo.",
        )
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#eaeef2", linewidth=0.8)
        nombres = list(elecciones)
        for i, seleccion in enumerate(nombres):
            izquierda = 0.0
            for modelo in MODELOS:
                ancho = elecciones[seleccion].get(modelo, 0.0)
                if ancho <= 0:
                    continue
                ax.barh(i, ancho, left=izquierda, color=COLOR_MODELO[modelo], height=0.55,
                        edgecolor="white", linewidth=2)  # fmt: skip
                if ancho >= 0.07:
                    ax.text(izquierda + ancho / 2, i, f"{modelo}\n{ancho:.0%}", ha="center",
                            va="center", fontsize=7, color=TINTA)  # fmt: skip
                izquierda += ancho
        ax.set_yticks(range(len(nombres)), [f"seleccion '{n}'" for n in nombres])
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlim(0, 1)
        guardar(fig, "elecciones.png")
    return generados


def _grafico_gate(ax, filas: pd.DataFrame, modelo: str) -> None:
    import matplotlib.dates
    import matplotlib.ticker

    _estilo(ax)
    filas = filas.sort_values("fecha")
    color = COLOR_MODELO[modelo]
    if "incumbent_directional_accuracy" in filas:
        # La vara que tenia que superar cada candidato (score cacheado del vigente).
        vigente = filas["incumbent_directional_accuracy"].astype(float)
        ax.scatter(filas["fecha"], vigente, marker="_", s=220, color=GRIS_BENCHMARK,
                   linewidth=2, zorder=2)  # fmt: skip
    promovido = filas["promoted"].astype(bool)
    candidato = filas["directional_accuracy"].astype(float)
    ax.scatter(filas.loc[promovido, "fecha"], candidato[promovido], color=color, s=28, zorder=3,
               edgecolor="white", linewidth=1)  # fmt: skip
    ax.scatter(filas.loc[~promovido, "fecha"], candidato[~promovido], facecolor="white",
               edgecolor=color, s=28, zorder=3, linewidth=1.5)  # fmt: skip
    ax.set_title(
        f"{modelo}: lleno = promovido, vacio = rechazado; raya gris = score a superar",
        fontsize=8.5,
        color=TINTA_SECUNDARIA,
        loc="left",
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))


def _accuracy_mensual(resumen_: dict) -> pd.DataFrame:
    """Modelo -> mes -> accuracy (macro y contexto: hit rate de sus senales con vista)."""
    por_modelo = (resumen_.get("calidad_modelos") or {}).get("todos_los_tickers") or {}
    columnas = {}
    for modelo, datos in por_modelo.items():
        clave = "signal_hit_rate" if modelo in ("macro", "contexto") else "directional_accuracy"
        columnas[modelo] = {
            pd.Period(mes).to_timestamp(): m.get(clave) for mes, m in datos["por_mes"].items()
        }
    tabla = pd.DataFrame(columnas).sort_index()
    return tabla.astype(float) if not tabla.empty else tabla


# ---------------------------------------------------------------------- #
# Documento: se arma una vez como bloques y se escribe en Markdown y LaTeX
# ---------------------------------------------------------------------- #
# Bloques: ("titulo", nivel, texto), ("parrafo", texto), ("lista", [textos]),
# ("tabla", encabezado, filas), ("imagenes", [archivos]). Los textos marcan
# codigo entre backticks, como en Markdown.
Bloque = tuple


def _pct(valor) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "-"
    return f"{valor:+.1%}" if valor else "0.0%"


def _num(valor, decimales: int = 2) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "-"
    return f"{valor:.{decimales}f}"


def bloques(resumen_: dict, imagenes: list[str]) -> list[Bloque]:
    cfg, cfg_carteras, periodo = (
        resumen_["config"],
        resumen_["config_carteras"],
        resumen_["periodo"],
    )

    def graficos_de(*prefijos: str) -> list[Bloque]:
        elegidas = [img for img in imagenes if img.startswith(prefijos)]
        return [("imagenes", elegidas)] if elegidas else []

    doc: list[Bloque] = [
        ("titulo", 1, f"Simulacion `{cfg['nombre']}`"),
        (
            "lista",
            [
                f"Periodo: {periodo['desde']} a {periodo['hasta']} ({periodo['ruedas']} ruedas).",
                f"Reentreno de LSTM / XGBoost / Transformer: {cfg.get('reentreno', 'diario')}; "
                "macro y contexto el 1 de cada mes.",
                f"Capital inicial ${cfg_carteras['capital']:,.0f}, comision "
                f"{cfg_carteras['comision']:.2%} por operacion, rebalanceo cada "
                f"{cfg_carteras['rebalanceo']} ruedas.",
                f"Errores de Black-Litterman (cartera sin cambios ese dia): {resumen_['errores']}.",
            ],
        ),
        ("titulo", 2, "Carteras"),
        ("parrafo", "Ordenadas por retorno total en pesos. Sharpe sin tasa libre de riesgo."),
    ]
    rotacion_ = resumen_["rotacion"]
    filas = []
    for c in resumen_["carteras"]:
        rot = rotacion_.get(c["cartera"], {})
        filas.append(
            [
                f"`{c['cartera']}`",
                _pct(c.get("retorno_total")),
                _pct(c.get("retorno_total_usd")),
                _pct(c.get("volatilidad_anualizada")).lstrip("+"),
                _num(c.get("sharpe")),
                _pct(c.get("max_drawdown")),
                _pct(rot.get("comisiones_sobre_capital")).lstrip("+"),
            ]
        )
    encabezado = ["Cartera", "Retorno ARS", "Retorno USD", "Vol. anual", "Sharpe", "Max DD",
                  "Comisiones"]  # fmt: skip
    doc.append(("tabla", encabezado, filas))
    doc += _bloques_sin_comision(resumen_)
    doc += graficos_de("carteras_")

    calidad_ = resumen_["calidad_modelos"]
    for universo, titulo in (("todos_los_tickers", "todos los tickers"),
                             ("panel_lider", "panel lider")):  # fmt: skip
        por_modelo = calidad_.get(universo) or {}
        if not por_modelo:
            continue
        filas = []
        for modelo in MODELOS + MODELOS_PANEL:
            if modelo not in por_modelo:
                continue
            m = por_modelo[modelo]["overall"]
            filas.append(
                [
                    modelo,
                    str(m.get("n_predictions", 0)),
                    _pct(m.get("directional_accuracy")).lstrip("+"),
                    _pct(m.get("signal_hit_rate")).lstrip("+"),
                    _pct(m.get("neutral_rate")).lstrip("+"),
                    _num(m.get("mae_logret"), 4),
                ]
            )
        encabezado = ["Modelo", "Predicciones", "Accuracy", "Hit rate (con vista)", "Neutrales",
                      "MAE log"]  # fmt: skip
        doc += [
            ("titulo", 2, f"Calidad de las predicciones ({titulo})"),
            ("tabla", encabezado, filas),
        ]
    doc.append(
        (
            "parrafo",
            "La accuracy cuenta todas las predicciones; el hit rate, solo las que dieron alza o "
            "baja. Para macro, que casi siempre da neutral, la medida que importa es el hit rate.",
        )
    )
    doc += graficos_de("accuracy_")

    if resumen_["promociones"]:
        filas = [
            [m, str(p["reentrenamientos"]), str(p["promociones"])]
            for m, p in resumen_["promociones"].items()
        ]
        doc += [
            ("titulo", 2, "Reentrenamientos"),
            ("tabla", ["Modelo", "Reentrenamientos", "Promovidos"], filas),
        ]
        doc += graficos_de("promociones")

    if resumen_["elecciones"]:
        filas = [
            [f"'{s}'", *[_pct(e.get(m, 0.0)).lstrip("+") for m in MODELOS]]
            for s, e in resumen_["elecciones"].items()
        ]
        doc += [
            ("titulo", 2, "Mejor modelo por ticker"),
            ("tabla", ["Seleccion", *MODELOS], filas),
        ]
        doc += graficos_de("elecciones")

    limitaciones = list(LIMITACIONES)
    periodo_reentreno = {"semanal": "una semana", "mensual": "un mes"}.get(cfg.get("reentreno"))
    if periodo_reentreno:
        limitaciones.insert(
            0,
            f"Reentreno {cfg['reentreno']}: produccion reentrena LSTM / XGBoost / Transformer "
            "todos los dias habiles, asi que los modelos vigentes pueden tener hasta "
            f"{periodo_reentreno} mas de antiguedad que en produccion.",
        )
    if cfg_carteras.get("backtest_arrastrado"):
        limitaciones.insert(
            0,
            "Rearmado con un rebalanceo que no es multiplo del de la corrida: en los dias sin "
            "backtest guardado, el consenso y la eleccion del mejor modelo usan el ultimo "
            "backtest disponible.",
        )
    doc += [("titulo", 2, "Limitaciones"), ("lista", limitaciones)]
    return doc


def _bloques_sin_comision(resumen_: dict) -> list[Bloque]:
    """Tabla con el retorno de cada cartera con la comision de la corrida y sin comision."""
    sin = {c["cartera"]: c for c in resumen_.get("sin_comision") or []}
    if not sin:
        return []
    filas = []
    for c in resumen_["carteras"]:
        s = sin.get(c["cartera"])
        if s is None:
            continue
        diferencia = (s.get("retorno_total") or 0.0) - (c.get("retorno_total") or 0.0)
        filas.append(
            [
                f"`{c['cartera']}`",
                _pct(c.get("retorno_total")),
                _pct(s.get("retorno_total")),
                _pct(diferencia),
                _pct(c.get("retorno_total_usd")),
                _pct(s.get("retorno_total_usd")),
            ]
        )
    comision = resumen_["config_carteras"]["comision"]
    return [
        ("titulo", 3, "Con y sin comision"),
        (
            "parrafo",
            f"Las mismas operaciones con comision {comision:.2%} y sin comision "
            f"(`rearmados/{SIN_COMISION}`). La diferencia es lo que se comen las comisiones: "
            "las carteras que mas rotan son las que mas pierden.",
        ),
        ("tabla", ["Cartera", "ARS con", "ARS sin", "Diferencia", "USD con", "USD sin"], filas),
    ]


def _tabla_md(encabezado: list[str], filas: list[list[str]]) -> str:
    alineacion = ["---"] + ["---:"] * (len(encabezado) - 1)
    lineas = ["| " + " | ".join(encabezado) + " |", "| " + " | ".join(alineacion) + " |"]
    lineas += ["| " + " | ".join(fila) + " |" for fila in filas]
    return "\n".join(lineas)


def markdown(resumen_: dict, imagenes: list[str]) -> str:
    partes: list[str] = []
    for bloque in bloques(resumen_, imagenes):
        tipo = bloque[0]
        if tipo == "titulo":
            partes.append("#" * bloque[1] + " " + bloque[2])
        elif tipo == "parrafo":
            partes.append(bloque[1])
        elif tipo == "lista":
            partes.append("\n".join(f"- {texto}" for texto in bloque[1]))
        elif tipo == "tabla":
            partes.append(_tabla_md(bloque[1], bloque[2]))
        elif tipo == "imagenes":
            partes.append("\n".join(f"![{img}](graficos/{img})" for img in bloque[1]))
    return "\n\n".join(partes) + "\n"


_ESCAPES_TEX = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex(texto: str) -> str:
    """Escapa ``texto`` para LaTeX; lo que va entre backticks queda en monoespaciado."""
    partes = texto.split("`")
    salida = []
    for i, parte in enumerate(partes):
        escapado = "".join(_ESCAPES_TEX.get(c, c) for c in parte)
        salida.append(rf"\texttt{{{escapado}}}" if i % 2 else escapado)
    return "".join(salida)


PREAMBULO_TEX = r"""\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish]{babel}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{float}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}
\setlength{\tabcolsep}{4pt}
\begin{document}
"""


def _tabla_tex(encabezado: list[str], filas: list[list[str]]) -> str:
    columnas = "l" + "r" * (len(encabezado) - 1)
    # La tabla de carteras (7 columnas y nombres largos) no entra en el ancho con letra normal.
    letra = r"\footnotesize" if len(encabezado) >= 7 else r"\small"
    lineas = [
        rf"\begingroup{letra}",
        rf"\begin{{longtable}}{{{columnas}}}",
        r"\toprule",
        " & ".join(_tex(h) for h in encabezado) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    lineas += [" & ".join(_tex(celda) for celda in fila) + r" \\" for fila in filas]
    lineas += [r"\bottomrule", r"\end{longtable}", r"\endgroup"]
    return "\n".join(lineas)


def latex(resumen_: dict, imagenes: list[str]) -> str:
    """El mismo reporte que ``markdown``, como documento LaTeX compilable."""
    partes = [PREAMBULO_TEX]
    for bloque in bloques(resumen_, imagenes):
        tipo = bloque[0]
        if tipo == "titulo":
            if bloque[1] == 1:
                partes.append(rf"{{\LARGE\bfseries {_tex(bloque[2])}\par}}")
            elif bloque[1] >= 3:
                partes.append(rf"\subsection*{{{_tex(bloque[2])}}}")
            else:
                partes.append(rf"\section*{{{_tex(bloque[2])}}}")
        elif tipo == "parrafo":
            partes.append(_tex(bloque[1]))
        elif tipo == "lista":
            items = "\n".join(rf"  \item {_tex(texto)}" for texto in bloque[1])
            partes.append(f"\\begin{{itemize}}\n{items}\n\\end{{itemize}}")
        elif tipo == "tabla":
            partes.append(_tabla_tex(bloque[1], bloque[2]))
        elif tipo == "imagenes":
            partes += [
                "\\begin{figure}[H]\n\\centering\n"
                rf"\includegraphics[width=\linewidth]{{graficos/{img}}}"
                "\n\\end{figure}"
                for img in bloque[1]
            ]
    partes.append(r"\end{document}")
    return "\n\n".join(partes) + "\n"


def generar(corrida: Path, snapshot: Snapshot, rearmado: str | None = None) -> Path:
    """Escribe ``resumen.json``, ``reporte.md``, ``reporte.tex`` y ``graficos/`` en la carpeta
    del reporte y la devuelve.
    """
    datos = cargar(corrida, rearmado)
    destino = (corrida / "rearmados" / rearmado if rearmado else corrida) / "reporte"
    destino.mkdir(parents=True, exist_ok=True)
    resumen_ = resumen(datos, snapshot)
    sin_comision = corrida / "rearmados" / SIN_COMISION / "valores.jsonl"
    if rearmado is None and sin_comision.exists():
        valores = _fechas(_leer_jsonl(sin_comision))
        resumen_["sin_comision"] = tabla_carteras(valores).to_dict("records")
    (destino / "resumen.json").write_text(
        json.dumps(resumen_, indent=2, ensure_ascii=False, default=str)
    )
    imagenes = graficos(datos, resumen_, destino / "graficos")
    (destino / "reporte.md").write_text(markdown(resumen_, imagenes))
    (destino / "reporte.tex").write_text(latex(resumen_, imagenes))
    return destino


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reporte de una corrida de la simulacion.")
    parser.add_argument("--nombre", required=True, help="Corrida.")
    parser.add_argument("--rearmado", help="Usar las carteras de <corrida>/rearmados/<nombre>.")
    args = parser.parse_args(argv)
    corrida = Path(RESULTADOS_DIR) / args.nombre
    config = json.loads((corrida / "config.json").read_text())
    destino = generar(corrida, Snapshot.load(config["snapshot_dir"]), args.rearmado)
    print(f"Reporte en {destino}")


if __name__ == "__main__":
    main()
