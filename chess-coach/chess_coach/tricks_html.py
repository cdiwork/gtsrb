"""The fun report, with boards.

Deliberately not styled like the weakness report. That one is a diagnosis
and reads like one; this is a highlight reel, and the thing it most needs to
do is make you want to look at the positions.
"""
from __future__ import annotations

from typing import Dict, List
from urllib.parse import quote

from .diagram import board_svg
from .report import _CSS, _e

_EXTRA = """
.trick { background: var(--surface); border: 1px solid var(--rule);
  border-radius: 10px; padding: 16px 18px; margin-bottom: 14px;
  display: grid; grid-template-columns: 260px 1fr; gap: 20px; align-items: start; }
.trick svg { width: 100%; height: auto; border-radius: 4px; }
.trick .grade { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; }
.g-brilliant { color: var(--series-2); }
.g-sparkling { color: var(--series-1); }
.g-neat { color: var(--ink-2); }
.g-audacious { color: var(--critical); }
.trick h3 { margin: 4px 0 8px; font-size: 1.15rem; }
.trick .why { color: var(--ink-2); margin-bottom: 8px; }
.trick .line { font-family: ui-monospace, Menlo, monospace; font-size: 0.84rem;
  background: var(--track); padding: 6px 9px; border-radius: 5px;
  display: inline-block; margin-bottom: 6px; }
.score { display: flex; gap: 22px; flex-wrap: wrap; margin-bottom: 26px; }
.score div { display: flex; flex-direction: column; }
.score .v { font-size: 1.9rem; font-weight: 600; letter-spacing: -0.02em; }
.score .k { color: var(--muted); font-size: 0.74rem; text-transform: uppercase;
  letter-spacing: 0.07em; }
@media (max-width: 620px) { .trick { grid-template-columns: 1fr; } }
"""


def _trick_card(trick: Dict) -> str:
    orientation = trick.get("side") != "black"
    try:
        board = board_svg(
            trick["fen"],
            better=trick["san"] if trick["kind"] != "played" else None,
            played=trick["san"] if trick["kind"] == "played" else None,
            orientation=orientation,
            size=260,
        )
    except Exception:
        board = ""
    link = "https://lichess.org/analysis/" + quote(trick["fen"], safe="")
    bits = [
        f'<div class="trick"><div>{board}</div><div>',
        f'<div class="grade g-{_e(trick["grade"])}">{_e(trick["grade"])} · '
        f'{_e(trick["beauty"])}</div>',
        f'<h3>{_e(trick["san"])} — move {_e(trick["move_number"])}</h3>',
    ]
    if trick.get("instead_of"):
        bits.append(f'<p>You played <code>{_e(trick["instead_of"])}</code>.</p>')
    if trick.get("reasons"):
        bits.append(f'<p class="why">{_e(", ".join(trick["reasons"]))}</p>')
    if trick.get("line"):
        bits.append(f'<div class="line">{_e(trick["line"])}</div><br>')
    opening = trick.get("opening") or ""
    suffix = f" · {_e(opening)}" if opening and opening != "unknown" else ""
    bits.append(
        f'<p><a href="{_e(link)}" target="_blank" rel="noopener">open the board</a>'
        f'{suffix}</p></div></div>'
    )
    return "".join(bits)


def render_tricks_html(report: Dict, limit: int = 8) -> str:
    counts = report["counts"]
    hero = report.get("hero", "you")
    parts = [f"<h1>Tricks on the board</h1>"]
    parts.append(
        f"<p>{report['games']} games. Ranked by how good the move would have "
        f"felt, not by what it was worth.</p>"
    )
    parts.append('<div class="score">')
    for value, key in (
        (counts["found"], "tricks you played"),
        (counts["missed"], "tricks you walked past"),
        (f"{counts['found_share']:.0%}" if counts["found_share"] is not None else "-",
         "hit rate"),
        (counts["per_game"], "chances per game"),
    ):
        parts.append(f'<div><span class="v">{_e(value)}</span>'
                     f'<span class="k">{_e(key)}</span></div>')
    parts.append("</div>")

    for title, key, blurb in (
        ("Tricks you found", "found",
         "Pretty and sound. These are the games worth keeping."),
        ("Tricks you missed", "missed",
         "These were on the board, and nobody would have blamed you for not "
         "seeing them. Worth setting up anyway."),
        ("Gambles that didn't come off", "gambles",
         "Material given up for not quite enough. The instinct is usually "
         "better than the execution."),
    ):
        items = report.get(key) or []
        if not items:
            continue
        parts.append(f"<h2>{_e(title)}</h2><p>{_e(blurb)}</p>")
        parts.extend(_trick_card(t) for t in items[:limit])

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Tricks on the board: {_e(hero)}</title>"
        f"<style>{_CSS}{_EXTRA}</style></head><body><div class=\"wrap\">"
        + "".join(parts)
        + "</div></body></html>"
    )
