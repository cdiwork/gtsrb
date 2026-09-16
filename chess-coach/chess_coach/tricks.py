"""The same games, looking for the good stuff.

The rest of this package asks "what did you get wrong, and what did it
cost". This module asks a different and, for a lot of players, a better
question: **what were the fun moves on the board, and did you find them?**

It runs on an existing analysis file and needs no engine, because the
interesting part is not evaluation -- that is already recorded -- but
aesthetics. Beauty gets scored from the same features a human would point
at when calling a move pretty: material given up, whether the move is quiet
rather than forcing, whether it moves backwards, which motifs fire, whether
it was the only move that worked.

The scoring is a taste judgement made explicit so you can argue with it.
Taking a hanging queen scores near nothing; a quiet retreat that wins scores
a lot. That is the whole point.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import chess

from .motifs import classify_line
from .see import see

# What a sacrifice is worth, aesthetically, by what you gave up.
SAC_TIERS = [
    (800, 40, "queen sacrifice"),
    (450, 30, "rook sacrifice"),
    (200, 22, "piece sacrifice"),
    (90, 12, "pawn or exchange sacrifice"),
]

# Not all tactics are equally delightful. Taking something undefended is
# arithmetic; a zwischenzug is a joke at your opponent's expense.
MOTIF_BEAUTY = {
    "back_rank_mate": 16,
    "zwischenzug": 16,
    "removal_of_the_guard": 16,
    "mating_attack": 12,
    "discovered_attack": 12,
    "skewer": 12,
    "trapped_piece": 10,
    "fork": 14,
    "absolute_pin": 8,
    "pin": 7,
    "undefended_piece": 2,
    "wins_material": 1,
}

# A bare knight fork lands exactly on "neat": it is the most satisfying
# elementary tactic there is, and a scale that excludes it is measuring
# connoisseurship rather than fun.
GRADES = [(45, "brilliant"), (28, "sparkling"), (12, "neat")]


@dataclass
class Trick:
    """One move worth noticing, and why it is worth noticing."""

    kind: str  # played / missed / gamble
    game: str
    date: str
    opening: str
    move_number: int
    side: str
    fen: str
    san: str
    beauty: int
    grade: str
    reasons: List[str] = field(default_factory=list)
    motifs: List[str] = field(default_factory=list)
    sacrificed_cp: int = 0
    loss: float = 0.0
    line: Optional[str] = None
    instead_of: Optional[str] = None


def grade_of(beauty: int) -> Optional[str]:
    for threshold, name in GRADES:
        if beauty >= threshold:
            return name
    return None


def beauty_score(
    fen: str,
    san: str,
    *,
    only_move: bool = False,
    leads_to_mate: bool = False,
    extra_motifs: Optional[List[str]] = None,
) -> Tuple[int, List[str], List[str], int]:
    """Score one move for prettiness.

    Returns (score, reasons, motifs, material_sacrificed_cp). The reasons are
    returned so the report can say *why* rather than just handing over a
    number nobody can argue with.

    `extra_motifs` carries motifs the analysis found across a whole variation
    -- removal of the guard and zwischenzugs only exist over several moves,
    so re-deriving them from a single move would silently lose the two
    cleverest things this tool can spot.
    """
    try:
        board = chess.Board(fen)
        move = board.parse_san(san)
    except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
        return 0, [], [], 0

    found = classify_line(board, [move])
    shape = found.get("shape")
    if shape is None:
        return 0, [], [], 0
    motifs = list(found.get("motifs", []))
    for motif in extra_motifs or []:
        if motif not in motifs:
            motifs.append(motif)

    exchange = see(board, move)
    sacrificed = -exchange if exchange < 0 else 0
    underpromotion = bool(move.promotion and move.promotion != chess.QUEEN)

    # Beauty requires content. A quiet move is only lovely if it is quietly
    # *doing* something -- otherwise it is just a move, and most moves in a
    # game are just moves. Without this gate every developing move in the
    # database scores as a trick.
    if not (sacrificed >= 90 or motifs or only_move or underpromotion):
        return 0, [], motifs, sacrificed

    score = 0
    reasons: List[str] = []

    # Material given up -- the single biggest component of a pretty move.
    for threshold, points, label in SAC_TIERS:
        if sacrificed >= threshold:
            score += points
            reasons.append(f"{label} ({sacrificed / 100:.1f} pawns given up)")
            break

    # A quiet move that does tactical work is the classic beautiful move:
    # no check, no capture, nothing announcing itself.
    if shape.is_quiet:
        score += 16
        reasons.append("a quiet move, with nothing forcing about it")

    # Only count retreats that are genuinely hard to see. A backward capture
    # or check is forcing, and forcing moves announce themselves.
    if shape.direction == "backward" and shape.is_quiet:
        score += 8
        reasons.append("moves backwards, which is where humans never look")

    if shape.span >= 5 and not shape.is_capture:
        score += 4
        reasons.append("travels the length of the board")

    if underpromotion:
        score += 25
        reasons.append("underpromotion")

    for motif in motifs:
        points = MOTIF_BEAUTY.get(motif, 0)
        if points:
            score += points
            reasons.append(motif.replace("_", " "))

    if only_move:
        score += 10
        reasons.append("the only move that works")

    if leads_to_mate:
        score += 14
        reasons.append("forces mate")

    return score, reasons, motifs, sacrificed


# Above this win percentage the position is already won, and an ordinary good
# move is not a trick anybody had to find. Without this guard a long endgame of
# queen-chases-knight scores a "trick" on nearly every move, and a game is
# rewarded for going on a long time rather than for containing anything.
WON_ENOUGH = 90.0


def _worth_finding(move: Dict, san: str, sacrificed: int,
                   line: str = "") -> bool:
    """Was there actually something to find here?"""
    win_before = move.get("win_before")
    if win_before is None or win_before < WON_ENOUGH:
        return True
    # In a won position only the finish counts: mate, or a real sacrifice.
    return san.endswith("#") or "#" in (line or "") or sacrificed >= 90


def find_tricks(games: List[Dict], min_beauty: int = 16) -> Dict:
    """Sort every judged move into found, missed, and didn't-come-off.

    A trick only counts as *found* if it actually worked. An unsound
    sacrifice is not a brilliancy, but it is not nothing either, so those go
    in their own pile rather than being quietly dropped.
    """
    played: List[Trick] = []
    missed: List[Trick] = []
    gambles: List[Trick] = []

    for game in games:
        label = f"{game.get('white', '?')} vs {game.get('black', '?')}"
        common = {
            "game": label,
            "date": game.get("date", ""),
            "opening": game.get("opening", ""),
        }
        for move in game.get("moves", []):
            fen = move.get("fen_before")
            if not fen:
                continue
            loss = move.get("loss") or 0.0
            only_move = bool(move.get("only_move"))

            # --- what you played ---
            score, reasons, motifs, sacrificed = beauty_score(
                fen, move["san"], only_move=only_move
            )
            grade = grade_of(score)
            if grade and loss < 5.0 and _worth_finding(move, move["san"],
                                                       sacrificed):
                played.append(Trick(
                    kind="played", **common, move_number=move["move_number"],
                    side=move["side"], fen=fen, san=move["san"], beauty=score,
                    grade=grade, reasons=reasons, motifs=motifs,
                    sacrificed_cp=sacrificed, loss=loss,
                ))
            elif sacrificed >= 90 and loss >= 11.0:
                gambles.append(Trick(
                    kind="gamble", **common, move_number=move["move_number"],
                    side=move["side"], fen=fen, san=move["san"],
                    beauty=max(score, 0), grade=grade or "audacious",
                    reasons=reasons, motifs=motifs, sacrificed_cp=sacrificed,
                    loss=loss,
                ))

            # --- what was there instead ---
            best = move.get("best_san")
            if not best or best == move["san"] or loss < 3.0:
                continue
            score, reasons, motifs, sacrificed = beauty_score(
                fen, best, only_move=only_move,
                extra_motifs=move.get("missed_motifs"),
            )
            grade = grade_of(score)
            if grade and _worth_finding(move, best, sacrificed,
                                        move.get("best_line_san") or ""):
                missed.append(Trick(
                    kind="missed", **common, move_number=move["move_number"],
                    side=move["side"], fen=fen, san=best, beauty=score,
                    grade=grade, reasons=reasons, motifs=motifs,
                    sacrificed_cp=sacrificed, loss=loss,
                    line=move.get("best_line_san"), instead_of=move["san"],
                ))

    for pile in (played, missed, gambles):
        pile.sort(key=lambda t: -t.beauty)

    found_n, missed_n = len(played), len(missed)
    total = found_n + missed_n
    return {
        "games": len(games),
        "found": [asdict(t) for t in played],
        "missed": [asdict(t) for t in missed],
        "gambles": [asdict(t) for t in gambles],
        "counts": {
            "found": found_n,
            "missed": missed_n,
            "gambles": len(gambles),
            "on_the_board": total,
            "found_share": round(found_n / total, 3) if total else None,
            "per_game": round(total / len(games), 1) if games else None,
        },
        "min_beauty": min_beauty,
    }


def render_tricks(report: Dict, limit: int = 8) -> str:
    """The fun report, for a terminal."""
    counts = report["counts"]
    out = ["Tricks on the board", "=" * 19, ""]
    out.append(
        f"{report['games']} games. {counts['on_the_board']} moves worth "
        f"noticing -- you played {counts['found']} of them and walked past "
        f"{counts['missed']}."
    )
    if counts["found_share"] is not None:
        out.append(f"Hit rate: {counts['found_share']:.0%}. "
                   f"{counts['per_game']} chances per game.")
    out.append("")

    def block(title: str, items: List[Dict], blurb: str) -> None:
        if not items:
            return
        out.append(title)
        out.append("-" * len(title))
        out.append(blurb)
        out.append("")
        for trick in items[:limit]:
            head = (
                f"  {trick['grade'].upper()} ({trick['beauty']}) - move "
                f"{trick['move_number']} as {trick['side']}: {trick['san']}"
            )
            if trick.get("instead_of"):
                head += f"   (you played {trick['instead_of']})"
            out.append(head)
            if trick["reasons"]:
                out.append(f"      why: {', '.join(trick['reasons'])}")
            if trick.get("line"):
                out.append(f"      line: {trick['line']}")
            out.append(f"      {trick['fen']}")
            out.append("")

    block("Tricks you found", report["found"],
          "Moves you actually played that were both pretty and sound.")
    block("Tricks you missed", report["missed"],
          "These were on the board. Ranked by how good they'd have felt, "
          "not by how much they were worth.")
    block("Gambles that didn't come off", report["gambles"],
          "Material given up for not enough. Worth seeing -- the instinct is "
          "right more often than the execution.")
    return "\n".join(out)
