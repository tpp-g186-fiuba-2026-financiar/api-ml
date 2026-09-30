"""Reporte de calidad: backtest vs paper trading, semana a semana y por version.

El backtest (``src.backtest``) y el paper trading (``src.paper_trading``) miden
lo mismo -- si la senal de cada modelo acerto la direccion -- pero en
condiciones distintas: el backtest corre sobre historia que el modelo puede
haber visto al entrenar (in-sample parcial), el paper trading solo sobre
predicciones hechas *antes* de conocer el resultado. La diferencia entre ambos
es la medida honesta de cuanto se sobreestima un modelo con el backtest.

Este modulo no consulta ninguna API: lee ``paper_trading_ledger.json`` y
``backtest_results.json`` y produce:

- ``models/quality_report.md``: tablas legibles (brecha backtest vs paper,
  evolucion semanal desde la primera prediccion resuelta, y comparacion entre
  versiones de cada modelo: "esta version mejoro respecto de la anterior?").
- ``models/quality_history.json``: registro acumulativo, una entrada por dia de
  corrida, para poder ver la tendencia de la brecha en el tiempo.

Uso:
    python -m src.quality_report
    python -m src.quality_report --since 2026-08-01
    python -m src.quality_report --strict     # exit 1 si hay alertas
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

from src.backtest import summarize

DEFAULT_LEDGER = "models/paper_trading_ledger.json"
DEFAULT_BACKTEST = "models/backtest_results.json"
DEFAULT_REPORT = "models/quality_report.md"
DEFAULT_HISTORY = "models/quality_history.json"

# Con menos predicciones resueltas que esto no se saca ninguna conclusion.
DEFAULT_MIN_N = 30
# Brecha (backtest - paper) de accuracy direccional a partir de la cual se alerta.
DEFAULT_GAP_THRESHOLD = 0.10


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza (95%) de una proporcion. Evita sobreleer muestras chicas."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def with_interval(records: list[dict]) -> dict:
    """``summarize`` + intervalo de confianza de la accuracy direccional."""
    stats = summarize(records)
    n = stats["n_predictions"]
    if n:
        hits = round(stats["directional_accuracy"] * n)
        low, high = wilson_interval(hits, n)
        stats["accuracy_low"], stats["accuracy_high"] = round(low, 4), round(high, 4)
    return stats


def iso_week(as_of: str) -> str:
    year, week, _ = dt.date.fromisoformat(as_of).isocalendar()
    return f"{year}-W{week:02d}"


def filter_since(resolved: list[dict], since: str | None) -> list[dict]:
    return [r for r in resolved if since is None or r["as_of"] >= since]


def model_names(resolved: list[dict]) -> list[str]:
    return sorted({r["model"] for r in resolved})


def weekly_evolution(resolved: list[dict], model: str) -> list[dict]:
    """Una fila por semana ISO (segun ``as_of``) mas la accuracy acumulada hasta ella."""
    records = sorted((r for r in resolved if r["model"] == model), key=lambda r: r["as_of"])
    weeks = sorted({iso_week(r["as_of"]) for r in records})
    rows = []
    for week in weeks:
        in_week = [r for r in records if iso_week(r["as_of"]) == week]
        upto = [r for r in records if iso_week(r["as_of"]) <= week]
        rows.append(
            {
                "week": week,
                "week_stats": summarize(in_week),
                "cumulative_accuracy": summarize(upto)["directional_accuracy"],
            }
        )
    return rows


def version_comparison(resolved: list[dict], model: str, min_n: int) -> list[dict]:
    """Metricas por ``model_version`` en orden cronologico, con veredicto vs la anterior.

    El veredicto compara los intervalos de confianza de la accuracy direccional:
    solo dice "mejoro"/"empeoro" cuando no se solapan. Si alguna de las dos
    versiones tiene menos de ``min_n`` predicciones, es "pocos datos".
    """
    records = [r for r in resolved if r["model"] == model]
    first_seen: dict[str, str] = {}
    for r in sorted(records, key=lambda r: r["as_of"]):
        first_seen.setdefault(r["model_version"], r["as_of"])

    rows: list[dict] = []
    previous: dict | None = None
    for version, since in sorted(first_seen.items(), key=lambda item: item[1]):
        stats = with_interval([r for r in records if r["model_version"] == version])
        if stats["n_predictions"] < min_n:
            continue  # versiones con muestra insuficiente no se muestran ni se comparan
        row = {"version": version, "first_as_of": since, "stats": stats, "verdict": "primera"}
        if previous is not None:
            prev = previous["stats"]
            if stats["accuracy_low"] > prev["accuracy_high"]:
                row["verdict"] = "mejoro"
            elif stats["accuracy_high"] < prev["accuracy_low"]:
                row["verdict"] = "empeoro"
            else:
                row["verdict"] = "sin diferencia significativa"
        rows.append(row)
        previous = row
    return rows


def _paper_models_for(backtest_name: str, available: list[str]) -> list[str]:
    """Modelos del ledger que corresponden a uno del backtest (local y su variante Modal)."""
    return [m for m in (backtest_name, f"{backtest_name}-modal") if m in available]


def gap_table(backtest: dict, resolved: list[dict], min_n: int, gap_threshold: float) -> list[dict]:
    """Brecha backtest vs paper por modelo, mas el ranking en cada uno."""
    tickers = set(backtest.get("params", {}).get("tickers", []))
    available = model_names(resolved)
    rows = []
    for name, result in backtest["results"].items():
        for paper_name in _paper_models_for(name, available):
            mine = [r for r in resolved if r["model"] == paper_name]
            paper_all = with_interval(mine)
            paper_same = with_interval([r for r in mine if r["symbol"] in tickers])
            bt = result["overall"]
            gap = round(bt["directional_accuracy"] - paper_all["directional_accuracy"], 4)
            enough = paper_all["n_predictions"] >= min_n
            rows.append(
                {
                    "backtest_model": name,
                    "paper_model": paper_name,
                    "backtest": bt,
                    "paper_all": paper_all,
                    "paper_same_tickers": paper_same,
                    "gap_accuracy": gap,
                    "enough_data": enough,
                    "gap_alert": enough and gap > gap_threshold,
                    "paper_vs_chance": _vs_chance(paper_all, enough),
                }
            )
    _add_ranks(rows)
    return rows


def _vs_chance(stats: dict, enough: bool) -> str:
    """Compara la accuracy de paper contra el azar (50%) usando su intervalo de confianza."""
    if not enough:
        return "pocos datos"
    if stats["accuracy_low"] > 0.5:
        return "mejor que el azar"
    if stats["accuracy_high"] < 0.5:
        return "peor que el azar"
    return "indistinguible del azar"


def _add_ranks(rows: list[dict]) -> None:
    """Ranking por accuracy direccional en backtest y en paper (1 = mejor).

    Solo entre los modelos locales (sin ``-modal``): son los que existen en ambas
    mediciones sobre el mismo modelo, asi que el orden es comparable.
    """
    local = [r for r in rows if r["paper_model"] == r["backtest_model"]]
    by_bt = sorted(local, key=lambda r: -r["backtest"]["directional_accuracy"])
    by_paper = sorted(local, key=lambda r: -r["paper_all"]["directional_accuracy"])
    for row in rows:
        row["rank_backtest"] = by_bt.index(row) + 1 if row in local else None
        row["rank_paper"] = by_paper.index(row) + 1 if row in local else None


def build_snapshot(
    ledger: dict,
    backtest: dict,
    today: dt.date,
    since: str | None,
    min_n: int,
    gap_threshold: float,
) -> dict:
    resolved = filter_since(ledger["resolved"], since)
    dates = sorted(r["as_of"] for r in resolved)
    return {
        "date": today.isoformat(),
        "period": {"from": dates[0] if dates else None, "to": dates[-1] if dates else None},
        "backtest_run_at": backtest.get("run_at"),
        "backtest_tickers": backtest.get("params", {}).get("tickers", []),
        "n_resolved": len(resolved),
        "n_pending": len(ledger.get("pending", [])),
        "paper_by_model": {
            m: with_interval([r for r in resolved if r["model"] == m])
            for m in model_names(resolved)
        },
        "gaps": gap_table(backtest, resolved, min_n, gap_threshold),
    }


def update_history(path: Path, snapshot: dict) -> list[dict]:
    """Agrega el snapshot al registro; si ya hay uno del mismo dia lo reemplaza."""
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    history = [h for h in history if h["date"] != snapshot["date"]]
    history.append(
        {
            "date": snapshot["date"],
            "period": snapshot["period"],
            "n_resolved": snapshot["n_resolved"],
            "paper_accuracy": {
                m: s.get("directional_accuracy") for m, s in snapshot["paper_by_model"].items()
            },
            "gap_accuracy": {g["paper_model"]: g["gap_accuracy"] for g in snapshot["gaps"]},
        }
    )
    return sorted(history, key=lambda h: h["date"])


def alerts(snapshot: dict, gap_threshold: float) -> list[str]:
    out = []
    for g in snapshot["gaps"]:
        if g["gap_alert"]:
            out.append(
                f"{g['paper_model']}: el backtest sobreestima la accuracy en "
                f"{g['gap_accuracy'] * 100:.1f} pp (umbral {gap_threshold * 100:.0f} pp)"
            )
    return out


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_markdown(snapshot: dict, resolved: list[dict], min_n: int, gap_threshold: float) -> str:
    lines = [
        "# Calidad de los modelos: backtest vs paper trading",
        "",
        f"Generado el {snapshot['date']}. Periodo de paper trading analizado: "
        f"{snapshot['period']['from']} a {snapshot['period']['to']} "
        f"({snapshot['n_resolved']} predicciones ya comprobadas, "
        f"{snapshot['n_pending']} pendientes). Backtest del {snapshot['backtest_run_at']}.",
        "",
        "Lectura rapida: la **accuracy direccional** es el % de veces que el modelo acerto si el "
        "precio subia o bajaba (50% = azar). El backtest puede estar inflado porque los modelos "
        "ya vieron parte de esa historia al entrenar; el paper trading solo cuenta predicciones "
        "hechas antes de conocer el resultado, asi que es la medida real. Con menos de "
        f"{min_n} predicciones no se saca ninguna conclusion.",
        "",
        "## 1. Brecha backtest vs paper trading",
        "",
        "| Modelo (paper) | N paper | Acc. backtest | Acc. paper | Brecha | IC 95% paper | "
        "Paper vs azar | Ranking backtest → paper |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for g in snapshot["gaps"]:
        p = g["paper_all"]
        ci = f"{_pct(p.get('accuracy_low'))} – {_pct(p.get('accuracy_high'))}"
        rank = (
            f"{g['rank_backtest']} → {g['rank_paper']}" if g["rank_backtest"] is not None else "—"
        )
        flag = " ⚠️" if g["gap_alert"] else ""
        lines.append(
            f"| {g['paper_model']} | {p['n_predictions']} | "
            f"{_pct(g['backtest']['directional_accuracy'])} | {_pct(p['directional_accuracy'])} | "
            f"{g['gap_accuracy'] * 100:+.1f} pp{flag} | {ci} | {g['paper_vs_chance']} | {rank} |"
        )

    found = alerts(snapshot, gap_threshold)
    lines += ["", "**Alertas:**", ""]
    lines += [f"- {a}" for a in found] or ["- Ninguna."]

    lines += [
        "",
        "Mismos tickers que el backtest "
        f"({', '.join(snapshot['backtest_tickers']) or 'n/d'}), "
        "para comparar manzanas con manzanas:",
        "",
        "| Modelo (paper) | N | Acc. backtest | Acc. paper | Hit rate señales paper | "
        "Retorno estrategia paper |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in snapshot["gaps"]:
        p = g["paper_same_tickers"]
        lines.append(
            f"| {g['paper_model']} | {p['n_predictions']} | "
            f"{_pct(g['backtest']['directional_accuracy'])} | "
            f"{_pct(p.get('directional_accuracy'))} | {_pct(p.get('signal_hit_rate'))} | "
            f"{_pct(p.get('avg_strategy_logret'))} |"
        )

    lines += ["", "## 2. Evolucion semanal en paper trading", ""]
    for model in model_names(resolved):
        lines += [
            f"### {model}",
            "",
            "| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in weekly_evolution(resolved, model):
            s = row["week_stats"]
            small = " (pocos datos)" if s["n_predictions"] < min_n else ""
            lines.append(
                f"| {row['week']} | {s['n_predictions']}{small} | "
                f"{_pct(s['directional_accuracy'])} | {_pct(s['signal_hit_rate'])} | "
                f"{_pct(s['neutral_rate'])} | {_pct(row['cumulative_accuracy'])} |"
            )
        lines.append("")

    lines += [
        "## 3. Que se probo y si mejoro (por version del modelo)",
        "",
        f"Solo se listan versiones con al menos {min_n} predicciones comprobadas. "
        '"Mejoro"/"empeoro" significa que los intervalos de confianza no se solapan; '
        "si se solapan, la diferencia puede ser ruido.",
        "",
    ]
    for model in model_names(resolved):
        rows = version_comparison(resolved, model, min_n)
        lines.append(f"### {model}")
        lines.append("")
        if not rows:
            lines += ["Ninguna version alcanza la muestra minima todavia.", ""]
            continue
        lines += [
            "| Version | Desde | N | Acc. | IC 95% | vs version anterior |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
        for row in rows:
            s = row["stats"]
            lines.append(
                f"| {row['version']} | {row['first_as_of']} | {s['n_predictions']} | "
                f"{_pct(s['directional_accuracy'])} | "
                f"{_pct(s['accuracy_low'])} – {_pct(s['accuracy_high'])} | {row['verdict']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--backtest", default=DEFAULT_BACKTEST)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--history", default=DEFAULT_HISTORY)
    parser.add_argument("--since", help="solo predicciones con as_of >= YYYY-MM-DD")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD)
    parser.add_argument("--strict", action="store_true", help="exit 1 si hay alertas de brecha")
    args = parser.parse_args()

    ledger = load_json(Path(args.ledger))
    backtest = load_json(Path(args.backtest))
    snapshot = build_snapshot(
        ledger, backtest, dt.date.today(), args.since, args.min_n, args.gap_threshold
    )
    resolved = filter_since(ledger["resolved"], args.since)

    Path(args.report).write_text(
        render_markdown(snapshot, resolved, args.min_n, args.gap_threshold), encoding="utf-8"
    )
    history = update_history(Path(args.history), snapshot)
    Path(args.history).write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    found = alerts(snapshot, args.gap_threshold)
    print(f"Reporte: {args.report} | Registro: {args.history}")
    for line in found:
        print(f"ALERTA: {line}")
    if args.strict and found:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
