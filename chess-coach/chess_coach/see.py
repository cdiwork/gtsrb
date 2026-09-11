"""Static exchange evaluation.

python-chess will tell you a square is attacked, but not whether taking
there actually wins anything. Most of the "you left a piece hanging"
detection in this package needs that distinction, so we resolve capture
sequences by hand: always recapture with the least valuable attacker, and
let either side stop when continuing would lose material.

Working on a real board rather than with attack masks means x-rays fall
out for free -- removing a piece exposes whatever was behind it.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import chess

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def value_of(piece: Optional[chess.Piece]) -> int:
    return 0 if piece is None else PIECE_VALUE[piece.piece_type]


def _captures_to(board: chess.Board, square: int) -> List[chess.Move]:
    """Legal moves by the side to move that land on `square`."""
    return [m for m in board.legal_moves if m.to_square == square]


def _attacker_cost(board: chess.Board, move: chess.Move) -> int:
    """Value of the piece doing the capturing; promotions arrive upgraded."""
    cost = value_of(board.piece_at(move.from_square))
    if move.promotion:
        cost -= PIECE_VALUE[move.promotion] - PIECE_VALUE[chess.PAWN]
    return cost


def _recapture_gain(board: chess.Board, square: int, depth: int = 0) -> int:
    """Best material the side to move can win by recapturing on `square`.

    Never negative: recapturing is optional, so a losing exchange is simply
    declined.
    """
    occupant = board.piece_at(square)
    if occupant is None or depth > 12:
        return 0
    candidates = _captures_to(board, square)
    if not candidates:
        return 0
    candidates.sort(key=lambda m: _attacker_cost(board, m))
    move = candidates[0]
    gain = PIECE_VALUE[occupant.piece_type]
    if move.promotion:
        gain += PIECE_VALUE[move.promotion] - PIECE_VALUE[chess.PAWN]
    board.push(move)
    try:
        net = gain - _recapture_gain(board, square, depth + 1)
    finally:
        board.pop()
    return max(0, net)


def see(board: chess.Board, move: chess.Move) -> int:
    """Material the mover nets, in centipawns, after the dust settles.

    Positive means the exchange on `move.to_square` is good for whoever
    plays `move`. Works for quiet moves too, in which case it reports what
    the moved piece is worth if it can simply be taken.
    """
    work = board.copy(stack=False)
    target = move.to_square
    gain = value_of(work.piece_at(target))
    if work.is_en_passant(move):
        gain = PIECE_VALUE[chess.PAWN]
    if move.promotion:
        gain += PIECE_VALUE[move.promotion] - PIECE_VALUE[chess.PAWN]
    work.push(move)
    return gain - _recapture_gain(work, target)


def best_capture(board: chess.Board) -> Tuple[Optional[chess.Move], int]:
    """The most profitable capture available to the side to move."""
    best_move, best_gain = None, 0
    for move in board.legal_moves:
        if not (board.is_capture(move) or move.promotion):
            continue
        gain = see(board, move)
        if gain > best_gain:
            best_move, best_gain = move, gain
    return best_move, best_gain


def loose_pieces(board: chess.Board, color: chess.Color) -> List[Tuple[int, int]]:
    """Pieces of `color` the opponent can profitably take, with the gain.

    Evaluated as if it were the opponent's move, so this answers "what did
    I leave lying around" regardless of whose turn it actually is.
    """
    probe = board.copy(stack=False)
    probe.turn = not color
    found = []
    for move in probe.legal_moves:
        if not probe.is_capture(move):
            continue
        victim = probe.piece_at(move.to_square)
        if victim is None or victim.color != color:
            continue
        gain = see(probe, move)
        if gain >= PIECE_VALUE[chess.PAWN]:
            found.append((move.to_square, gain))
    # Keep the worst assessment per square.
    worst: dict[int, int] = {}
    for square, gain in found:
        worst[square] = max(worst.get(square, 0), gain)
    return sorted(worst.items(), key=lambda kv: -kv[1])
