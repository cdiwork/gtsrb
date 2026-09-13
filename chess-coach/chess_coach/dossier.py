"""The handoff to a coach.

Stockfish can prove a move was wrong. It cannot tell you what the position
wanted, and that is the half you learn from. So this module writes a dossier:
the position, the engine's lines, the structural facts, the tactic involved
and what you actually did -- arranged so a strong reader (a coach, or a
language model) can explain the plans without needing the engine again.

The prompt block at the top is deliberate. The quality of an explanation
depends mostly on being told what not to do: no restating the engine line as
prose, no generic advice that would fit any position.
"""
from __future__ import annotations

from typing import Dict, List, Optional

BRIEF = """\
# Coaching dossier

You are looking at the most expensive decisions from a batch of one player's
games, already measured by Stockfish. Your job is the part the engine cannot
do. For each position below, write:

1. **What the position is about** -- the structure, who has what, which side
   of the board each player should be playing on, and what the correct plan
   was. Two or three sentences of real chess content.
2. **Why the played move is wrong in terms of that plan** -- not "it loses a
   knight" (the evaluation already says so) but which feature of the position
   it misunderstood.
3. **What it would have taken to find the better move.** Be concrete: how many
   forcing replies were there to look at, was the move that refutes it a
   capture, was the better move quiet or backward, how long did they spend.

Then, across all the positions, name the recurring habit in one paragraph and
give one drill that would actually address it.

## The rule that matters most

**An engine cannot tell a blunder from a plan from a hunch, and those three
need completely different fixes.** The same -18 win% is produced by a player
who never looked, a player who calculated a line and got it wrong, and a
player who had a correct instinct and never verified it. Nothing in the data
below distinguishes them.

So do not write "the player probably saw..." as though it were a finding. Ask
the player what they were trying to do, and say plainly that your reading is
provisional until they answer. Each position has a slot for their account;
when it is empty, the honest output is an explanation of the chess plus a
question, not a confident diagnosis of their thinking. When it is filled in,
that testimony outranks your inference -- rebuild the explanation around it
rather than defending the first reading.

Other rules: do not paraphrase engine lines as if they were insight. Do not
give advice that would be true of any position ("develop your pieces",
"control the centre") unless it is specifically the point here. If the
positions do not share a pattern, say so -- one honest observation is worth
more than five invented ones.
"""


def _describe_structure(structure: Optional[Dict], color: str) -> List[str]:
    if not structure:
        return []
    mine = structure.get(color, {})
    theirs = structure.get("black" if color == "white" else "white", {})
    lines = [
        f"- Phase / centre: {structure.get('phase')} / {structure.get('centre')} centre",
        f"- Material balance: {structure.get('material_balance_cp', 0) / 100:+.1f} "
        "(positive = white)",
    ]
    files = structure.get("files", {})
    if files.get("open"):
        lines.append(f"- Open files: {', '.join(files['open'])}")
    for label, side in (("You", mine), ("Opponent", theirs)):
        king = side.get("king", {})
        pawns = side.get("pawns", {})
        bits = [
            f"king {king.get('square')} ({king.get('side')}, shield "
            f"{king.get('pawn_shield')}, {king.get('ring_squares_attacked')} "
            "squares around it attacked)",
            f"mobility {side.get('mobility')}",
        ]
        if pawns.get("passed"):
            bits.append(f"passed pawns {', '.join(pawns['passed'])}")
        if pawns.get("isolated"):
            bits.append(f"isolated {', '.join(pawns['isolated'])}")
        if pawns.get("doubled"):
            bits.append(f"{pawns['doubled']} doubled")
        if side.get("outposts"):
            bits.append(f"outposts {', '.join(side['outposts'])}")
        if side.get("rooks_active"):
            bits.append(f"rooks on useful files {', '.join(side['rooks_active'])}")
        if side.get("bishop_pair"):
            bits.append("bishop pair")
        lines.append(f"- {label}: " + "; ".join(bits))
    if structure.get("opposite_castling"):
        lines.append("- Kings castled on opposite wings (pawn-storm race)")
    return lines


def render_dossier(profile: Dict, limit: int = 6) -> str:
    moments = profile.get("key_moments", [])[:limit]
    out = [BRIEF, ""]
    hero = profile.get("hero", "the player")
    overview = profile.get("overview", {})
    out.append(f"## The player: {hero}")
    out.append("")
    out.append(
        f"{overview.get('games', 0)} games, {overview.get('moves_judged', 0)} moves "
        f"judged, mean accuracy {overview.get('accuracy')}, "
        f"{overview.get('blunders_per_100')} blunders per 100 moves, rating range "
        f"{overview.get('rating_range')}."
    )
    out.append("")

    findings = profile.get("findings", [])
    if findings:
        out.append("### Already measured (do not re-derive these, build on them)")
        out.append("")
        for finding in findings:
            out.append(
                f"- **{finding['title']}** [{finding['strength']}, n={finding['n']}]: "
                f"{finding['evidence']}"
            )
        out.append("")

    motifs = profile.get("motifs", {})
    if motifs.get("missed") or motifs.get("allowed"):
        out.append(
            f"Tactics missed: {motifs.get('missed')}. "
            f"Tactics allowed: {motifs.get('allowed')}."
        )
        out.append("")

    out.append("## Positions")
    out.append("")
    for index, moment in enumerate(moments, 1):
        out.append(
            f"### {index}. Move {moment['move_number']} as "
            f"{moment['hero_color']}, {moment['opening']} "
            f"({moment['severity']}, -{moment['loss']} win%)"
        )
        out.append("")
        out.append(f"```\nFEN: {moment['fen']}\n```")
        out.append("")
        out.append(f"- Played: **{moment['played']}** "
                   f"(win probability {moment['win_before']}% -> {moment['win_after']}%)")
        out.append(f"- Engine wanted: **{moment['best']}** — {moment.get('best_line') or ''}")
        if moment.get("refutation"):
            out.append(f"- What punished it: {moment['refutation']}")
        if moment.get("missed_motifs"):
            out.append(f"- Tactic missed: {', '.join(moment['missed_motifs'])}")
        if moment.get("allowed_motifs"):
            out.append(f"- Tactic allowed: {', '.join(moment['allowed_motifs'])}")
        if moment.get("shape_tags"):
            out.append(f"- The move played was: {', '.join(moment['shape_tags'])}")
        if moment.get("time_spent") is not None:
            out.append(f"- Time spent on it: {moment['time_spent']}s")
        out.append(
            "- **What the player says they were thinking:** _(unanswered -- "
            "ask before diagnosing)_"
        )
        if moment.get("only_move"):
            out.append("- There was only one good move here")
        elif moment.get("complexity") is not None:
            out.append(
                f"- Position sharpness: {moment['complexity']} win% between best and "
                "third-best move"
            )
        out.extend(_describe_structure(moment.get("structure"), moment["hero_color"]))
        out.append("")
    return "\n".join(out)
