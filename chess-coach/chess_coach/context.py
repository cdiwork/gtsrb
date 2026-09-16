"""Joining what happened at the board to what was happening to you.

A PGN records the moves and nothing else. It cannot say how much sleep you
had, what time it was, or how many games deep into a session you were -- and
those are exactly the variables that would explain a run of bad games.

So this module takes a small CSV you fill in *before* each game and
correlates it against what the analysis measured afterwards. The
before-the-game part matters: every other number in this package is measured
from the game itself and is therefore contaminated by how the game went.
These fields are not.

The output is deliberately noisy with caveats. Correlating a handful of
columns against a handful of metrics over a dozen games is the easiest way
in the world to find something that is not there, so the report says how
many comparisons it ran and how many of them chance alone would decorate.
"""
from __future__ import annotations

import csv
import math
import statistics
from typing import Dict, List, Optional, Tuple

# What we correlate the context against. Each is (label, extractor).
METRICS = ("score", "accuracy", "blunders", "trick_hit_rate", "mean_loss")

TEMPLATE_COLUMNS = [
    "game_id", "hours_slept", "time_of_day", "games_already_today", "sharpness_1_5",
]


def load_context(path: str) -> Dict[str, Dict[str, str]]:
    """Read the context CSV, keyed by game id."""
    rows: Dict[str, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("game_id") or "").strip()
            if key:
                rows[key] = {k: (v or "").strip() for k, v in row.items()
                             if k and k != "game_id"}
    return rows


def write_template(games: List[Dict], path: str) -> int:
    """Emit a CSV pre-filled with game ids and blank columns to fill in."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TEMPLATE_COLUMNS)
        for game in games:
            writer.writerow([game.get("id", "")] + [""] * (len(TEMPLATE_COLUMNS) - 1))
    return len(games)


def _as_number(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs)
    return cov / (sx * sy)


def _critical_r(n: int) -> Optional[float]:
    """Roughly the |r| needed for p<0.05 at this sample size, two-tailed.

    A crude normal approximation on Fisher's z -- good enough to stop
    somebody treating r=0.4 over eight games as a discovery.
    """
    if n < 4:
        return None
    return math.tanh(1.96 / math.sqrt(n - 3))


def game_metrics(games: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Per-game outcome numbers, keyed by game id."""
    from .tricks import find_tricks

    out: Dict[str, Dict[str, float]] = {}
    for game in games:
        key = game.get("id") or ""
        if not key:
            continue
        counts = find_tricks([game])["counts"]
        out[key] = {
            "score": {"win": 1.0, "draw": 0.5}.get(game.get("result"), 0.0),
            "accuracy": game.get("accuracy", 0.0),
            "blunders": game.get("counts", {}).get("blunder", 0),
            "trick_hit_rate": (counts["found_share"] * 100)
            if counts["found_share"] is not None else None,
            "mean_loss": game.get("mean_loss", 0.0),
        }
    return out


def correlate(games: List[Dict], context: Dict[str, Dict[str, str]]) -> Dict:
    """Every context column against every outcome metric, with the caveats."""
    metrics = game_metrics(games)
    matched = [k for k in metrics if k in context]
    columns: List[str] = []
    for key in matched:
        for column in context[key]:
            if column not in columns:
                columns.append(column)

    numeric: List[Dict] = []
    grouped: List[Dict] = []
    for column in columns:
        pairs_numeric: Dict[str, List[Tuple[float, float]]] = {m: [] for m in METRICS}
        buckets: Dict[str, Dict[str, List[float]]] = {}
        for key in matched:
            raw = context[key].get(column, "")
            if not raw:
                continue
            value = _as_number(raw)
            for metric in METRICS:
                got = metrics[key].get(metric)
                if got is None:
                    continue
                if value is not None:
                    pairs_numeric[metric].append((value, got))
                else:
                    buckets.setdefault(raw, {}).setdefault(metric, []).append(got)
        for metric in METRICS:
            pairs = pairs_numeric[metric]
            if len(pairs) >= 4:
                r = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
                if r is not None:
                    crit = _critical_r(len(pairs))
                    numeric.append({
                        "column": column, "metric": metric, "n": len(pairs),
                        "r": round(r, 2),
                        "needed_for_significance": round(crit, 2) if crit else None,
                        "notable": bool(crit and abs(r) >= crit),
                    })
        for value, per_metric in buckets.items():
            row = {"column": column, "value": value,
                   "n": len(per_metric.get("score", []))}
            for metric, got in per_metric.items():
                if got:
                    row[metric] = round(statistics.mean(got), 1)
            if row["n"] >= 2:
                grouped.append(row)

    comparisons = len(numeric)
    return {
        "games_with_context": len(matched),
        "games_total": len(metrics),
        "columns": columns,
        "correlations": sorted(numeric, key=lambda d: -abs(d["r"])),
        "groups": grouped,
        "comparisons": comparisons,
        "expected_false_positives": round(0.05 * comparisons, 1),
        "caveats": _caveats(len(matched), comparisons),
    }


def _caveats(n: int, comparisons: int) -> List[str]:
    out = []
    if n < 10:
        out.append(
            f"Only {n} games carry context. Nothing here is evidence yet; "
            "20 is roughly where a correlation starts meaning something."
        )
    if comparisons >= 10:
        out.append(
            f"{comparisons} correlations were run, so chance alone would be "
            f"expected to decorate about {0.05 * comparisons:.1f} of them. "
            "Treat the largest one as a hypothesis to test on new games, not "
            "as a result."
        )
    out.append(
        "These fields are the only numbers in this report recorded *before* "
        "the game, so unlike everything else they cannot have been coloured "
        "by how it went. That is what makes them worth logging."
    )
    return out


def render_context(report: Dict) -> str:
    """The context section, as text."""
    if not report or not report.get("games_with_context"):
        return ""
    out = [
        "Context",
        "-------",
        f"{report['games_with_context']} of {report['games_total']} games "
        f"have context logged.",
        "",
    ]
    if report["correlations"]:
        out.append("  column                metric           n     r   need   ")
        for row in report["correlations"]:
            flag = "  <- notable" if row["notable"] else ""
            need = row["needed_for_significance"]
            out.append(
                f"  {row['column']:<20s}  {row['metric']:<14s} {row['n']:3d} "
                f"{row['r']:+5.2f}  {need if need else '-':>5}{flag}"
            )
        out.append("")
    for row in report["groups"]:
        bits = ", ".join(f"{k} {v}" for k, v in row.items()
                         if k not in ("column", "value", "n"))
        out.append(f"  {row['column']} = {row['value']} (n={row['n']}): {bits}")
    if report["groups"]:
        out.append("")
    for caveat in report["caveats"]:
        out.append(f"  ! {caveat}")
    return "\n".join(out)
