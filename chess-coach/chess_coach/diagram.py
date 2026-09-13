"""Drawing the position.

A FEN is not a board to anyone who has not spent years reading them, and the
whole point of this tool is to be useful before that point. So every position
worth studying gets drawn, with the move that was played and the move that
should have been played marked on it.

Colour carries a meaning here (red = what you played, blue = what was better,
orange = the consequence), so every diagram is emitted with a caption naming
the arrows: the colour is a convenience, never the only way to read it.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import chess
import chess.svg

# Red and blue rather than the traditional red and green: the red/green pair
# is the one a sizeable fraction of readers cannot separate.
PLAYED = "#d03b3b"     # the move actually chosen
BETTER = "#2a78d6"     # the move the engine wanted
CONSEQUENCE = "#eb6834"  # what the opponent then does
SUPPORT = "#8a6fd4"    # a defensive relationship worth seeing

BOARD_COLORS = {
    "square light": "#eeeade",
    "square dark": "#9aa892",
    "square light lastmove": "#dcd79a",
    "square dark lastmove": "#b6ba7a",
    "margin": "#3a3a38",
    "coord": "#e8e8e2",
    "inner border": "#3a3a38",
    "outer border": "#3a3a38",
}

MoveLike = Union[str, chess.Move, Tuple[int, int], None]


def _to_move(board: chess.Board, move: MoveLike) -> Optional[chess.Move]:
    """Accept SAN, UCI or a Move, so callers can use whichever they have."""
    if move is None or isinstance(move, tuple):
        return None
    if isinstance(move, chess.Move):
        return move
    text = move.strip()
    for parse in (board.parse_san, chess.Move.from_uci):
        try:
            return parse(text)
        except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
            continue
    return None


def _square(name: Union[str, int]) -> Optional[int]:
    if isinstance(name, int):
        return name
    try:
        return chess.parse_square(name.strip())
    except ValueError:
        return None


def board_svg(
    fen: str,
    *,
    played: MoveLike = None,
    better: MoveLike = None,
    consequence: MoveLike = None,
    support: MoveLike = None,
    mark: Iterable[Union[str, int]] = (),
    mark_color: str = CONSEQUENCE,
    orientation: Optional[bool] = None,
    size: int = 360,
    show_check: bool = True,
) -> str:
    """One position, drawn, with the moves that matter marked on it.

    `support` draws a defending relationship rather than a move -- "this
    knight is what holds that pawn up" -- which is usually the fact a player
    was missing when they went wrong.

    `mark` fills individual squares -- the square that quietly lost its
    defender, the king that cannot castle -- which is usually the thing the
    arrows alone do not explain.
    """
    board = chess.Board(fen)
    if orientation is None:
        orientation = board.turn

    arrows: List[chess.svg.Arrow] = []
    for move_like, colour in (
        (support, SUPPORT),
        (better, BETTER),
        (consequence, CONSEQUENCE),
        (played, PLAYED),
    ):
        move = _to_move(board, move_like)
        if move is not None:
            arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=colour))
        elif isinstance(move_like, tuple):
            arrows.append(chess.svg.Arrow(move_like[0], move_like[1], color=colour))

    fill: Dict[int, str] = {}
    for name in mark:
        square = _square(name)
        if square is not None:
            fill[square] = mark_color

    check_square = None
    if show_check and board.is_check():
        check_square = board.king(board.turn)

    return chess.svg.board(
        board,
        arrows=arrows,
        fill=fill,
        orientation=orientation,
        size=size,
        check=check_square,
        coordinates=True,
        borders=True,
        colors=BOARD_COLORS,
    )


def after(fen: str, moves: Sequence[str]) -> str:
    """The FEN you reach by playing `moves` (SAN) from `fen`.

    Handy for drawing the position *after* the refutation, which is usually
    where the point becomes obvious.
    """
    board = chess.Board(fen)
    for san in moves:
        move = _to_move(board, san)
        if move is None or move not in board.legal_moves:
            break
        board.push(move)
    return board.fen()


def legend_html() -> str:
    """The key for the arrow colours, so they are never the only signal."""
    rows = [
        (PLAYED, "what you played"),
        (BETTER, "what was better"),
        (CONSEQUENCE, "what happens next / the target square"),
        (SUPPORT, "a piece defending a square"),
    ]
    items = "".join(
        f'<span class="legend-item"><span class="legend-arrow" '
        f'style="background:{colour}"></span>{label}</span>'
        for colour, label in rows
    )
    return f'<div class="legend">{items}</div>'
