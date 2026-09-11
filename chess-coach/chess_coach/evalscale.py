"""Centipawns are the wrong unit for coaching.

Going from +9.0 to +7.0 means nothing; going from 0.0 to -0.7 can decide
the game. So every "how bad was that move" number in this package is
expressed in *win percentage*, which is close to linear in the thing we
actually care about: how much the move cost.
"""
from __future__ import annotations

import math
from typing import Optional

import chess.engine

# Logistic fit of centipawns -> win probability, from Lichess' regression
# over millions of games. Keeping their constant means our numbers are
# comparable to the accuracy figures you see on a Lichess game report.
_K = 0.00368208
CP_CLAMP = 1000

# How much win% a move has to throw away before we call it something.
# These are thresholds on a *single move's* cost.
INACCURACY = 5.0
MISTAKE = 11.0
BLUNDER = 20.0


def cp_to_win(cp: float) -> float:
    """Centipawns (from the mover's point of view) -> win% in [0, 100]."""
    cp = max(-CP_CLAMP, min(CP_CLAMP, cp))
    return 50 + 50 * (2 / (1 + math.exp(-_K * cp)) - 1)


def score_to_win(score: chess.engine.Score) -> float:
    """A POV-relative engine score -> win% in [0, 100]."""
    if score.is_mate():
        mate = score.mate()
        if mate is None:
            return 50.0
        return 100.0 if mate > 0 else 0.0
    cp = score.score()
    return 50.0 if cp is None else cp_to_win(cp)


def accuracy(win_before: float, win_after: float) -> float:
    """Lichess' per-move accuracy curve, in percent.

    A move that holds its win% is ~100; one that sheds 20 points is ~40.
    """
    drop = max(0.0, win_before - win_after)
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * drop) - 3.1669))


def severity(loss: float) -> Optional[str]:
    """Name the size of a mistake, or None if the move was fine."""
    if loss >= BLUNDER:
        return "blunder"
    if loss >= MISTAKE:
        return "mistake"
    if loss >= INACCURACY:
        return "inaccuracy"
    return None


def eval_band(win: float) -> str:
    """Coarse description of who stands better, from the mover's side."""
    if win >= 80:
        return "winning"
    if win >= 62:
        return "better"
    if win > 38:
        return "equal"
    if win > 20:
        return "worse"
    return "losing"


def format_score(score: chess.engine.Score) -> str:
    """Human-readable eval, e.g. '+1.34' or '#4'."""
    if score.is_mate():
        return f"#{score.mate()}"
    cp = score.score()
    return "0.00" if cp is None else f"{cp / 100:+.2f}"
