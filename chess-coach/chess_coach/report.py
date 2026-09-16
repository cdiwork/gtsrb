"""Rendering a profile for a human being.

Order matters more than content here. The findings come first because they
are what you act on; the tables come after because they are what you check
the findings against; the caveats are near the top, not buried at the end,
because a reader who does not know the sample is thin will over-trust the
rest.
"""
from __future__ import annotations

from typing import Dict, List, Optional

STRENGTH_NOTE = {
    "strong": "consistent across the sample",
    "moderate": "visible, worth testing",
    "weak": "suggestive only",
    "anecdote": "too few cases to trust",
}


def _table(headers: List[str], rows: List[List[str]]) -> List[str]:
    if not rows:
        return []
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return out + [""]


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def render_markdown(profile: Dict) -> str:
    overview = profile.get("overview", {})
    hero = profile.get("hero", "you")
    out: List[str] = [f"# Chess weakness profile: {hero}", ""]

    record = (
        f"{overview.get('wins', 0)}W / {overview.get('draws', 0)}D / "
        f"{overview.get('losses', 0)}L"
    )
    if overview.get("unfinished"):
        record += f" / {overview['unfinished']} with no recorded result"
    out += [
        f"**{overview.get('games', 0)} games** ({record}), "
        f"{overview.get('moves_judged', 0)} of your moves judged by Stockfish. "
        f"Mean accuracy **{overview.get('accuracy')}%**, "
        f"**{overview.get('blunders_per_100')}** blunders and "
        f"**{overview.get('errors_per_100')}** serious errors per 100 moves.",
        "",
    ]
    if overview.get("rating_range"):
        low, high = overview["rating_range"]
        out += [f"Rating over this sample: {low}-{high}.", ""]

    if profile.get("caveats"):
        out += ["> **Read this first.** " + " ".join(profile["caveats"]), ""]

    # --- findings ---------------------------------------------------------
    out += ["## What keeps costing you points", ""]
    findings = profile.get("findings", [])
    if not findings:
        out += [
            "No habit crossed the evidence threshold in this sample. That is a "
            "real result, not a failure: your errors look like ordinary "
            "one-off oversights rather than a systematic bias. Add more games "
            "and re-run before concluding anything.",
            "",
        ]
    for index, finding in enumerate(findings, 1):
        out += [
            f"### {index}. {finding['title']}",
            "",
            f"*{finding['definition']}*",
            "",
            f"- **Evidence** ({finding['strength']} — "
            f"{STRENGTH_NOTE.get(finding['strength'], '')}, n={finding['n']}): "
            f"{finding['evidence']}",
            f"- **How it was measured**: {finding['signal']}",
            f"- **What to do**: {finding['drill']}",
            "",
        ]

    # --- where it happens -------------------------------------------------
    out += ["## Where the points go", "", "### By phase of the game", ""]
    out += _table(
        ["Phase", "Moves", "Accuracy", "Errors/100", "Blunders/100"],
        [
            [name, row["moves"], row["accuracy"], row["errors_per_100"],
             row["blunders_per_100"]]
            for name, row in (profile.get("by_phase") or {}).items()
        ],
    )

    out += [
        "### By how the position stood",
        "",
        "One thing to read carefully: accuracy usually *rises* in losing "
        "positions and that is an artefact, not a skill. Win probability is "
        "already near zero there, so there is very little left to throw away "
        "and every move scores well. Compare the `equal`, `better` and "
        "`winning` rows against each other; ignore `losing` except as a "
        "reminder of how often you get there.",
        "",
    ]
    out += _table(
        ["Position", "Moves", "Accuracy", "Errors/100"],
        [
            [name, row["moves"], row["accuracy"], row["errors_per_100"]]
            for name, row in (profile.get("by_position_state") or {}).items()
        ],
    )

    out += ["### By colour", ""]
    out += _table(
        ["Colour", "Moves", "Accuracy", "Errors/100"],
        [
            [name, row["moves"], row["accuracy"], row["errors_per_100"]]
            for name, row in (profile.get("by_color") or {}).items()
        ],
    )

    time_data = profile.get("time") or {}
    if time_data.get("by_time_spent"):
        out += ["### By time spent on the move", ""]
        out += _table(
            ["Thinking time", "Moves", "Accuracy", "Blunders/100"],
            [
                [name, row["moves"], row["accuracy"], row["blunders_per_100"]]
                for name, row in time_data["by_time_spent"].items()
            ],
        )
    if time_data.get("by_clock_left"):
        out += ["### By clock remaining", ""]
        out += _table(
            ["Clock left", "Moves", "Accuracy", "Blunders/100"],
            [
                [name, row["moves"], row["accuracy"], row["blunders_per_100"]]
                for name, row in time_data["by_clock_left"].items()
            ],
        )

    # --- tactics ----------------------------------------------------------
    motifs = profile.get("motifs") or {}
    if motifs.get("missed") or motifs.get("allowed"):
        out += [
            "## Tactics",
            "",
            "Two different problems. *Missed* is the tactic in the move you "
            "failed to play; *allowed* is the tactic your opponent used to "
            "punish you. Work on whichever column is longer.",
            "",
        ]
        keys = sorted(set(motifs.get("missed", {})) | set(motifs.get("allowed", {})))
        out += _table(
            ["Motif", "You missed it", "It was played on you"],
            [
                [key.replace("_", " "), motifs.get("missed", {}).get(key, 0),
                 motifs.get("allowed", {}).get(key, 0)]
                for key in sorted(
                    keys,
                    key=lambda k: -(motifs.get("missed", {}).get(k, 0)
                                    + motifs.get("allowed", {}).get(k, 0)),
                )
            ],
        )

    pieces = profile.get("pieces") or {}
    if pieces.get("opponent_punishing_piece") or pieces.get("your_moving_piece"):
        out += ["### Which pieces are involved", ""]
        out += [
            f"- You were moving: {_counter_line(pieces.get('your_moving_piece'))}",
            f"- Punished by their: {_counter_line(pieces.get('opponent_punishing_piece'))}",
            f"- Your pieces that got hit: {_counter_line(pieces.get('your_pieces_hit'))}",
            "",
        ]

    # --- openings ---------------------------------------------------------
    openings = [row for row in (profile.get("openings") or []) if row["games"] >= 1]
    if openings:
        out += [
            "## Openings",
            "",
            "`First error` is the median move number of your first serious "
            "mistake — roughly where your understanding of the position runs out.",
            "",
        ]
        out += _table(
            ["Opening", "Games", "Score", "Accuracy", "Loss/move", "First error"],
            [
                [row["opening"][:40], row["games"], f"{row['score_pct']}%",
                 row["accuracy"], row["mean_loss"], row["first_error_move"]]
                for row in openings[:12]
            ],
        )

    # --- conversion -------------------------------------------------------
    conversion = profile.get("conversion") or {}
    if conversion.get("games_reaching_winning"):
        out += [
            "## Converting and saving",
            "",
            f"- You reached a winning position in "
            f"{conversion['games_reaching_winning']} games and won "
            f"{conversion['converted']} of them ({_pct(conversion.get('conversion_pct'))}).",
            f"- You were losing at some point in "
            f"{conversion['games_reaching_losing']} games and salvaged "
            f"{conversion['saved']} ({_pct(conversion.get('save_pct'))}).",
            "",
        ]
        if conversion.get("thrown_away"):
            out += ["Games you had won and did not win:", ""]
            for game in conversion["thrown_away"]:
                out.append(
                    f"- {game['opening']} ({game.get('date', '')}): reached "
                    f"{game['peak_win_pct']}% winning chances, ended in a "
                    f"{game['result']}."
                )
            out.append("")

    # --- context ----------------------------------------------------------
    context = profile.get("context") or {}
    if context.get("games_with_context"):
        out += [
            "## What was going on before the game",
            "",
            f"{context['games_with_context']} of {context['games_total']} games "
            "have context logged. These are the only numbers in this report "
            "written down *before* the first move, so unlike everything else "
            "they cannot have been coloured by how the game went.",
            "",
        ]
        if context.get("correlations"):
            out += [
                "`r` runs from -1 to +1. `Need` is roughly the size a "
                "correlation has to reach at this sample size before it means "
                "anything; below that, the honest reading is *nothing here yet*.",
                "",
            ]
            out += _table(
                ["Context", "Metric", "n", "r", "Need", ""],
                [
                    [row["column"].replace("_", " "),
                     row["metric"].replace("_", " "), row["n"],
                     f"{row['r']:+.2f}",
                     row["needed_for_significance"] or "-",
                     "**notable**" if row["notable"] else ""]
                    for row in context["correlations"]
                ],
            )
        if context.get("groups"):
            out += ["Context values that are words rather than numbers:", ""]
            for row in context["groups"]:
                bits = ", ".join(
                    f"{k.replace('_', ' ')} {v}" for k, v in row.items()
                    if k not in ("column", "value", "n")
                )
                out.append(
                    f"- {row['column'].replace('_', ' ')} = **{row['value']}** "
                    f"(n={row['n']}): {bits}"
                )
            out.append("")
        for caveat in context.get("caveats", []):
            out += [f"> {caveat}", ""]

    # --- positions --------------------------------------------------------
    moments = profile.get("key_moments") or []
    if moments:
        out += [
            "## The positions worth studying",
            "",
            "Set these up on a board. Find the move before reading the answer.",
            "",
        ]
        for index, moment in enumerate(moments, 1):
            out += [
                f"**{index}. Move {moment['move_number']} as {moment['hero_color']}** "
                f"({moment['opening']}, {moment['severity']}, "
                f"-{moment['loss']} win%)",
                "",
                f"`{moment['fen']}`",
                "",
                f"- You played {moment['played']}; {moment['best']} was right"
                + (f" — {moment['best_line']}" if moment.get("best_line") else ""),
            ]
            if moment.get("refutation"):
                out.append(f"- Punished by: {moment['refutation']}")
            if moment.get("time_spent") is not None:
                out.append(f"- You spent {moment['time_spent']}s on it")
            out.append("")

    out += [
        "---",
        "",
        "Generated by chess-coach. Accuracy and win-probability figures use the "
        "Lichess logistic scale, so they are comparable to a Lichess game "
        "report. A move is an *inaccuracy* at 5 win% lost, a *mistake* at 11, "
        "a *blunder* at 20.",
    ]
    return "\n".join(out)


def _counter_line(counter: Optional[Dict]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{name} x{count}" for name, count in list(counter.items())[:6])


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

import html as _html
from urllib.parse import quote as _quote

from .diagram import board_svg, legend_html

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --rule: rgba(11,11,11,0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --good: #0ca30c;
  --warning: #fab219;
  --serious: #ec835a;
  --critical: #d03b3b;
  --track: rgba(11,11,11,0.06);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --rule: rgba(255,255,255,0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --track: rgba(255,255,255,0.08);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --rule: rgba(255,255,255,0.10);
  --series-1: #3987e5;
  --series-2: #d95926;
  --track: rgba(255,255,255,0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 16px; padding-block: 40px; }
h1 { font-size: 1.55rem; line-height: 1.2; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 {
  font-size: 1.05rem; margin: 40px 0 14px; padding-bottom: 7px;
  border-bottom: 1px solid var(--rule); letter-spacing: 0.01em;
}
h3 { font-size: 1rem; margin: 0 0 6px; }
p { margin: 0 0 12px; }
.sub { color: var(--ink-2); margin-bottom: 24px; }
.tiles { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; }
.tile {
  flex: 1 1 120px; background: var(--surface); border: 1px solid var(--rule);
  border-radius: 10px; padding: 12px 14px;
}
.tile .v { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; }
.tile .k { color: var(--muted); font-size: 0.78rem; text-transform: uppercase;
  letter-spacing: 0.06em; margin-top: 2px; }
.note {
  background: var(--surface); border: 1px solid var(--rule);
  border-left: 3px solid var(--warning); border-radius: 8px;
  padding: 12px 14px; color: var(--ink-2); font-size: 0.9rem; margin-bottom: 20px;
}
.card {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
  padding: 16px 18px; margin-bottom: 14px;
}
.card .def { color: var(--ink-2); font-style: italic; margin-bottom: 10px; }
.row { display: flex; gap: 8px; align-items: baseline; margin-bottom: 7px; }
.row .lab {
  color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.06em; flex: 0 0 76px; padding-top: 2px;
}
.row .val { flex: 1; }
.badge {
  display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
  border-radius: 20px; text-transform: uppercase; letter-spacing: 0.06em;
  border: 1px solid currentColor;
}
.s-strong { color: var(--critical); }
.s-moderate { color: var(--serious); }
.s-weak, .s-anecdote { color: var(--muted); }
.chart { margin: 0 0 22px; }
.chart .title { font-weight: 600; margin-bottom: 3px; }
.chart .hint { color: var(--muted); font-size: 0.82rem; margin-bottom: 12px; }
.bars { display: grid; grid-template-columns: minmax(80px, 150px) 1fr; gap: 7px 12px;
  align-items: center; }
.bars .name { color: var(--ink-2); font-size: 0.87rem; text-align: right;
  overflow-wrap: anywhere; }
.lane { display: flex; align-items: center; gap: 8px; }
.track { flex: 1; background: var(--track); border-radius: 4px; height: 100%; }
.bar { height: 14px; border-radius: 0 4px 4px 0; background: var(--series-1);
  transition: filter 0.12s; }
/* A zero value draws nothing: a stub bar reads as a small quantity. */
.bar:not(.zero) { min-width: 2px; }
.bar.s2 { background: var(--series-2); }
.lane:hover .bar { filter: brightness(1.12); }
.num { font-variant-numeric: tabular-nums; font-size: 0.85rem; color: var(--ink-2);
  flex: 0 0 auto; min-width: 34px; }
.legend { display: flex; gap: 14px; margin-bottom: 10px; font-size: 0.84rem;
  color: var(--ink-2); }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 5px; vertical-align: baseline; }
.board { margin: 12px 0; max-width: 340px; }
.board svg { width: 100%; height: auto; display: block; border-radius: 4px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-arrow { display: inline-block; width: 16px; height: 4px; border-radius: 2px; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); }
th { color: var(--muted); font-size: 0.74rem; text-transform: uppercase;
  letter-spacing: 0.06em; font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.pos { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
  padding: 14px 16px; margin-bottom: 12px; }
.pos .head { display: flex; justify-content: space-between; gap: 10px;
  flex-wrap: wrap; margin-bottom: 8px; }
.pos .cost { color: var(--critical); font-weight: 600; font-variant-numeric: tabular-nums; }
code, .fen {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem;
  background: var(--track); padding: 2px 5px; border-radius: 4px;
  overflow-wrap: anywhere;
}
a { color: var(--series-1); }
footer { margin-top: 44px; padding-top: 14px; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: 0.82rem; }
"""


def _e(value) -> str:
    return _html.escape("" if value is None else str(value))


def _bar_chart(
    title: str,
    hint: str,
    rows: List[tuple],
    suffix: str = "",
    series: int = 1,
    legend: Optional[List[str]] = None,
) -> str:
    """Horizontal bars with the value printed on every row.

    Values are always labelled, so the colour is decoration rather than the
    only channel carrying the number.
    """
    usable = [r for r in rows if r[1] is not None]
    if not usable:
        return ""
    top = max(max(float(v) for v in r[1:] if v is not None) for r in usable) or 1.0
    out = [f'<div class="chart"><div class="title">{_e(title)}</div>']
    if hint:
        out.append(f'<div class="hint">{_e(hint)}</div>')
    if legend:
        out.append('<div class="legend">')
        for index, name in enumerate(legend, 1):
            colour = "var(--series-1)" if index == 1 else "var(--series-2)"
            out.append(
                f'<span><span class="swatch" style="background:{colour}"></span>'
                f"{_e(name)}</span>"
            )
        out.append("</div>")
    out.append('<div class="bars">')
    for row in usable:
        name = row[0]
        out.append(f'<div class="name">{_e(name)}</div><div>')
        for index, value in enumerate(row[1:], 1):
            if value is None:
                continue
            width = 100 * float(value) / top
            cls = "bar" if index == 1 else "bar s2"
            if not float(value):
                cls += " zero"
            label = f"{value}{suffix}"
            out.append(
                f'<div class="lane" title="{_e(name)}: {_e(label)}">'
                f'<div class="track"><div class="{cls}" style="width:{width:.1f}%">'
                f"</div></div>"
                f'<div class="num">{_e(label)}</div></div>'
            )
        out.append("</div>")
    out.append("</div></div>")
    return "".join(out)


def _html_table(headers: List[str], rows: List[List], numeric: Optional[set] = None) -> str:
    if not rows:
        return ""
    numeric = numeric or set()
    head = "".join(
        f'<th class="{"n" if index in numeric else ""}">{_e(h)}</th>'
        for index, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="{"n" if index in numeric else ""}">{_e(c)}</td>'
            for index, c in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def render_html(profile: Dict) -> str:
    overview = profile.get("overview", {})
    hero = profile.get("hero", "you")
    parts: List[str] = []

    record = (
        f"{overview.get('wins', 0)}W / {overview.get('draws', 0)}D / "
        f"{overview.get('losses', 0)}L"
    )
    if overview.get("unfinished"):
        record += f" / {overview['unfinished']} with no recorded result"
    parts.append(f"<h1>How {_e(hero)} loses at chess</h1>")
    parts.append(
        f'<p class="sub">{overview.get("games", 0)} games ({record}), '
        f'{overview.get("moves_judged", 0)} moves judged by Stockfish.</p>'
    )
    parts.append('<div class="tiles">')
    for value, key in (
        (f"{overview.get('accuracy')}%", "mean accuracy"),
        (overview.get("blunders_per_100"), "blunders / 100 moves"),
        (overview.get("errors_per_100"), "errors / 100 moves"),
        (overview.get("mean_loss_per_move"), "win% lost per move"),
    ):
        parts.append(f'<div class="tile"><div class="v">{_e(value)}</div>'
                     f'<div class="k">{_e(key)}</div></div>')
    parts.append("</div>")

    if profile.get("caveats"):
        parts.append(
            '<div class="note"><strong>Read this first.</strong> '
            + " ".join(_e(c) for c in profile["caveats"]) + "</div>"
        )

    # --- findings ---
    parts.append("<h2>What keeps costing you points</h2>")
    findings = profile.get("findings", [])
    if not findings:
        parts.append(
            '<div class="card">No habit crossed the evidence threshold in this '
            "sample. That is a real result rather than a failure: these errors "
            "look like ordinary one-off oversights, not a systematic bias. Add "
            "more games and re-run.</div>"
        )
    for finding in findings:
        strength = finding["strength"]
        parts.append(
            f'<div class="card"><div class="head">'
            f'<span class="badge s-{_e(strength)}">{_e(strength)} · n={_e(finding["n"])}'
            f"</span></div>"
            f"<h3>{_e(finding['title'])}</h3>"
            f'<div class="def">{_e(finding["definition"])}</div>'
            f'<div class="row"><div class="lab">Evidence</div>'
            f'<div class="val">{_e(finding["evidence"])}</div></div>'
            f'<div class="row"><div class="lab">Measured</div>'
            f'<div class="val">{_e(finding["signal"])}</div></div>'
            f'<div class="row"><div class="lab">Fix</div>'
            f'<div class="val">{_e(finding["drill"])}</div></div></div>'
        )

    # --- charts ---
    parts.append("<h2>Where the points go</h2>")
    phases = profile.get("by_phase") or {}
    parts.append(_bar_chart(
        "Errors per 100 moves, by phase",
        "Mistakes and blunders only. The phase with the highest rate is where "
        "study pays back fastest.",
        [(name, row["errors_per_100"]) for name, row in phases.items()],
    ))
    states = profile.get("by_position_state") or {}
    parts.append(_bar_chart(
        "Errors per 100 moves, by how the position stood",
        "Compare winning/better/equal against each other. The 'losing' row "
        "always looks good and it is an artefact: with win probability near "
        "zero there is nothing left to throw away, so every move scores well.",
        [(name, row["errors_per_100"]) for name, row in states.items()],
    ))
    time_data = profile.get("time") or {}
    if time_data.get("by_time_spent"):
        parts.append(_bar_chart(
            "Blunders per 100 moves, by thinking time",
            "If the quick moves are the bad ones, the problem is impulse, not "
            "knowledge.",
            [(name, row["blunders_per_100"])
             for name, row in time_data["by_time_spent"].items()],
        ))
    if time_data.get("by_clock_left"):
        parts.append(_bar_chart(
            "Blunders per 100 moves, by clock remaining",
            "",
            [(name, row["blunders_per_100"])
             for name, row in time_data["by_clock_left"].items()],
        ))

    motifs = profile.get("motifs") or {}
    missed, allowed = motifs.get("missed", {}), motifs.get("allowed", {})
    if missed or allowed:
        keys = sorted(
            set(missed) | set(allowed),
            key=lambda k: -(missed.get(k, 0) + allowed.get(k, 0)),
        )[:10]
        parts.append("<h2>Tactics</h2>")
        parts.append(
            "<p>Two different problems. <em>Missed</em> is the tactic in the "
            "move you failed to find; <em>allowed</em> is the one your opponent "
            "used against you. Work on whichever column is longer.</p>"
        )
        parts.append(_bar_chart(
            "Tactical motifs in your mistakes",
            "",
            [(key.replace("_", " "), missed.get(key, 0), allowed.get(key, 0))
             for key in keys],
            series=2,
            legend=["You missed it", "Played on you"],
        ))

    # --- context ---
    context = profile.get("context") or {}
    if context.get("games_with_context"):
        parts.append("<h2>What was going on before the game</h2>")
        parts.append(
            f'<p>{context["games_with_context"]} of {context["games_total"]} '
            "games have context logged. These are the only numbers on this page "
            "written down <em>before</em> the first move, so unlike everything "
            "else they cannot have been coloured by how the game went.</p>"
        )
        if context.get("correlations"):
            parts.append(_html_table(
                ["Context", "Metric", "n", "r", "Need", ""],
                [[row["column"].replace("_", " "),
                  row["metric"].replace("_", " "), row["n"],
                  f"{row['r']:+.2f}", row["needed_for_significance"] or "-",
                  "notable" if row["notable"] else ""]
                 for row in context["correlations"]],
            ))
        for row in context.get("groups", []):
            bits = ", ".join(
                f"{k.replace('_', ' ')} {v}" for k, v in row.items()
                if k not in ("column", "value", "n")
            )
            parts.append(
                f'<p>{_e(row["column"].replace("_", " "))} = '
                f'<strong>{_e(row["value"])}</strong> (n={row["n"]}): {_e(bits)}</p>'
            )
        for caveat in context.get("caveats", []):
            parts.append(f'<div class="note">{_e(caveat)}</div>')

    # --- openings ---
    openings = profile.get("openings") or []
    if openings:
        parts.append("<h2>Openings</h2>")
        parts.append(
            "<p><em>First error</em> is the median move number of your first "
            "serious mistake — roughly where your understanding of the position "
            "runs out.</p>"
        )
        parts.append(_html_table(
            ["Opening", "Games", "Score", "Accuracy", "Loss/move", "First error"],
            [[row["opening"][:44], row["games"], f"{row['score_pct']}%",
              row["accuracy"], row["mean_loss"], row["first_error_move"]]
             for row in openings[:12]],
            numeric={1, 2, 3, 4, 5},
        ))

    conversion = profile.get("conversion") or {}
    if conversion.get("games_reaching_winning"):
        parts.append("<h2>Converting and saving</h2>")
        parts.append(
            f"<p>You reached a winning position in "
            f"<strong>{conversion['games_reaching_winning']}</strong> games and "
            f"won <strong>{conversion['converted']}</strong> of them "
            f"({_pct(conversion.get('conversion_pct'))}). You were losing at some "
            f"point in {conversion['games_reaching_losing']} games and salvaged "
            f"{conversion['saved']} ({_pct(conversion.get('save_pct'))}).</p>"
        )

    # --- positions ---
    moments = profile.get("key_moments") or []
    if moments:
        parts.append("<h2>The positions worth studying</h2>")
        parts.append(
            "<p>Find the move before you read the answer. Boards are drawn from "
            "the side you were playing.</p>"
        )
        parts.append(legend_html())
        for index, moment in enumerate(moments, 1):
            link = "https://lichess.org/analysis/" + _quote(moment["fen"], safe="")
            tags = ", ".join(
                (moment.get("missed_motifs") or []) + (moment.get("allowed_motifs") or [])
            )
            try:
                board = board_svg(
                    moment["fen"],
                    played=moment.get("played"),
                    better=moment.get("best"),
                    orientation=moment.get("hero_color") != "black",
                    size=320,
                )
            except Exception:
                board = ""
            parts.append(
                f'<div class="pos"><div class="head">'
                f"<strong>{index}. Move {_e(moment['move_number'])} as "
                f"{_e(moment['hero_color'])}</strong>"
                f'<span class="cost">-{_e(moment["loss"])} win%</span></div>'
                f"<p>{_e(moment['opening'])} · {_e(moment['severity'])}"
                + (f" · {_e(moment['time_spent'])}s spent"
                   if moment.get("time_spent") is not None else "")
                + (f" · {_e(tags)}" if tags else "")
                + "</p>"
                f"<p>You played <code>{_e(moment['played'])}</code>; "
                f"<code>{_e(moment['best'])}</code> was right"
                + (f" — {_e(moment['best_line'])}" if moment.get("best_line") else "")
                + "</p>"
                + (f"<p>Punished by: {_e(moment['refutation'])}</p>"
                   if moment.get("refutation") else "")
                + (f'<div class="board">{board}</div>' if board else "")
                + f'<p><span class="fen">{_e(moment["fen"])}</span> '
                f'<a href="{_e(link)}" target="_blank" rel="noopener">open board</a></p>'
                "</div>"
            )

    parts.append(
        "<footer>Generated by chess-coach. Win probability and accuracy use the "
        "Lichess logistic scale, so figures are comparable to a Lichess game "
        "report: a move is an <em>inaccuracy</em> at 5 win% lost, a "
        "<em>mistake</em> at 11, a <em>blunder</em> at 20. Every mistake was "
        "re-checked at higher depth before being counted.</footer>"
    )

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>Chess weakness profile: {_e(hero)}</title>"
        f"<style>{_CSS}</style></head><body><div class=\"wrap\">"
        + "".join(parts)
        + "</div></body></html>"
    )


# --------------------------------------------------------------------------
# Single game review
# --------------------------------------------------------------------------

_MARK = {"blunder": "??", "mistake": "?", "inaccuracy": "?!"}


def render_game_review(report: Dict, width: int = 88) -> str:
    """A terminal-readable annotation of one game.

    Only the moves that cost something are listed. A full move-by-move dump
    reads like a log file and hides the three moves that decided the game.
    """
    out: List[str] = []
    header = (
        f"{report['white']} vs {report['black']}"
        f"{' (' + report['opening'] + ')' if report.get('opening') else ''}"
    )
    out.append(header)
    out.append("-" * min(width, len(header)))
    counts = report["counts"]
    out.append(
        f"You were {report['hero_color']}, result: {report['result']}. "
        f"Accuracy {report['accuracy']}%, {report['mean_loss']} win% lost per move."
    )
    def plural(count: int, word: str) -> str:
        return f"{count} {word}{'' if count == 1 else 's'}"

    out.append(
        f"{plural(counts['blunder'], 'blunder')}, "
        f"{plural(counts['mistake'], 'mistake')}, "
        f"{plural(counts['inaccuracy'], 'inaccuracy').replace('inaccuracys', 'inaccuracies')}"
        f" over {report['moves_analysed']} moves."
    )
    out.append("")

    flawed = [m for m in report["moves"] if m.get("severity")]
    if not flawed:
        out.append("Nothing above the inaccuracy threshold. Clean game.")
        return "\n".join(out)

    out.append("Move       Cost   Win%     Engine preferred")
    for move in flawed:
        mark = _MARK.get(move["severity"], "")
        label = f"{move['move_number']}.{'' if move['side'] == 'white' else '..'}{move['san']}{mark}"
        best = move.get("best_san") or "-"
        extra = []
        if move.get("time_spent") is not None:
            extra.append(f"{move['time_spent']:.0f}s")
        if move.get("allowed_motifs"):
            extra.append("allowed " + "/".join(move["allowed_motifs"][:2]))
        elif move.get("missed_motifs"):
            extra.append("missed " + "/".join(move["missed_motifs"][:2]))
        suffix = f"   [{', '.join(extra)}]" if extra else ""
        out.append(
            f"{label:<11}{-move['loss']:>6.1f}  "
            f"{move['win_before']:>4.0f}->{move['win_after']:<4.0f} {best:<8}{suffix}"
        )
    out.append("")

    worst = sorted(flawed, key=lambda m: -m["loss"])[:3]
    out.append("Worth setting up on a board:")
    for move in worst:
        out.append("")
        out.append(
            f"  Move {move['move_number']} ({move['severity']}, "
            f"-{move['loss']:.1f} win%): you played {move['san']}, "
            f"{move.get('best_san') or '?'} was right."
        )
        if move.get("best_line_san"):
            out.append(f"    Engine line: {move['best_line_san']}")
        if move.get("refutation_san"):
            out.append(f"    Punished by: {move['refutation_san']}")
        out.append(f"    {move['fen_before']}")
    return "\n".join(out)
