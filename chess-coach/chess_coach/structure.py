"""Positional features, for talking about plans rather than mistakes.

Stockfish says +0.6 but never says why, and "why" is the part you can
actually learn from. These are the structural facts a coach would point at
-- who owns the open file, which knight has an outpost, whether the centre
is closed and therefore whether the plan should be a wing attack. They are
deliberately plain measurements; the narrative is built on top of them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import chess

from .see import PIECE_VALUE, value_of

CENTRE = [chess.D4, chess.E4, chess.D5, chess.E5]


def _pawn_files(board: chess.Board, color: chess.Color) -> Dict[int, List[int]]:
    files: Dict[int, List[int]] = {}
    for square in board.pieces(chess.PAWN, color):
        files.setdefault(chess.square_file(square), []).append(
            chess.square_rank(square)
        )
    return files


def non_pawn_material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        PIECE_VALUE[pt] * len(board.pieces(pt, color))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def phase_of(board: chess.Board) -> str:
    """Opening, middlegame or endgame -- by material, with a move-number floor."""
    material = non_pawn_material(board, chess.WHITE) + non_pawn_material(
        board, chess.BLACK
    )
    if material <= 1600:
        return "endgame"
    if board.fullmove_number <= 10 and material >= 5800:
        return "opening"
    return "middlegame"


def pawn_report(board: chess.Board, color: chess.Color) -> Dict:
    """Doubled, isolated, backward and passed pawns for one side."""
    own = _pawn_files(board, color)
    enemy = _pawn_files(board, not color)
    doubled = sum(len(ranks) - 1 for ranks in own.values() if len(ranks) > 1)
    isolated, passed, backward = [], [], []
    forward = 1 if color == chess.WHITE else -1
    for file_idx, ranks in own.items():
        neighbours = [f for f in (file_idx - 1, file_idx + 1) if 0 <= f < 8]
        if not any(f in own for f in neighbours):
            isolated.append(chess.square_name(chess.square(file_idx, ranks[0])))
        for rank in ranks:
            ahead = [
                r
                for f in [file_idx] + neighbours
                for r in enemy.get(f, [])
                if (r - rank) * forward > 0
            ]
            if not ahead:
                passed.append(chess.square_name(chess.square(file_idx, rank)))
            own_support = [
                r for f in neighbours for r in own.get(f, []) if (r - rank) * forward < 0
            ]
            if not own_support and any(
                (r - rank) * forward < 0 for f in neighbours for r in own.get(f, [])
            ):
                backward.append(chess.square_name(chess.square(file_idx, rank)))
    return {
        "count": len(board.pieces(chess.PAWN, color)),
        "islands": _count_islands(sorted(own)),
        "doubled": doubled,
        "isolated": isolated,
        "passed": passed,
        "backward": backward,
    }


def _count_islands(files: List[int]) -> int:
    if not files:
        return 0
    islands = 1
    for a, b in zip(files, files[1:]):
        if b - a > 1:
            islands += 1
    return islands


def centre_state(board: chess.Board) -> str:
    """Open, closed, semi-open or fluid -- this is what picks the plan.

    Closed centre means play on the wings and pawn storms are sound; an open
    centre means piece activity and king safety dominate.
    """
    tension = 0
    blocked = 0
    for square in board.pieces(chess.PAWN, chess.WHITE):
        file_idx, rank = chess.square_file(square), chess.square_rank(square)
        if rank + 1 < 8 and board.piece_at(chess.square(file_idx, rank + 1)) == chess.Piece(
            chess.PAWN, chess.BLACK
        ):
            blocked += 1
        for df in (-1, 1):
            if 0 <= file_idx + df < 8 and rank + 1 < 8:
                target = board.piece_at(chess.square(file_idx + df, rank + 1))
                if target == chess.Piece(chess.PAWN, chess.BLACK):
                    tension += 1
    # Judge by who still owns the d- and e-files, not by which of the four
    # centre squares happen to be occupied: at move one those squares are
    # empty, and the centre is certainly not open.
    white_files = _pawn_files(board, chess.WHITE)
    black_files = _pawn_files(board, chess.BLACK)
    central = [
        f for f in (3, 4) if f in white_files or f in black_files
    ]
    if blocked >= 2:
        return "closed"
    if not central:
        return "open"
    if tension >= 1:
        return "tense"
    contested = sum(1 for f in (3, 4) if f in white_files and f in black_files)
    return "fluid" if contested == 2 else "semi-open"


def open_files(board: chess.Board) -> Dict[str, List[str]]:
    """Files with no pawns at all, and files half-open for each side."""
    white = _pawn_files(board, chess.WHITE)
    black = _pawn_files(board, chess.BLACK)
    fully, half_white, half_black = [], [], []
    for file_idx in range(8):
        name = chess.FILE_NAMES[file_idx]
        if file_idx not in white and file_idx not in black:
            fully.append(name)
        elif file_idx not in white:
            half_white.append(name)
        elif file_idx not in black:
            half_black.append(name)
    return {"open": fully, "half_open_white": half_white, "half_open_black": half_black}


def king_safety(board: chess.Board, color: chess.Color) -> Dict:
    """Where the king lives, how much pawn cover it has, who is attacking it."""
    king_sq = board.king(color)
    if king_sq is None:
        return {}
    file_idx, rank = chess.square_file(king_sq), chess.square_rank(king_sq)
    home = 0 if color == chess.WHITE else 7
    side = "queenside" if file_idx <= 2 else ("kingside" if file_idx >= 5 else "centre")
    shield = 0
    step = 1 if color == chess.WHITE else -1
    for df in (-1, 0, 1):
        f = file_idx + df
        if not 0 <= f < 8:
            continue
        for dr in (1, 2):
            r = rank + step * dr
            if 0 <= r < 8 and board.piece_at(chess.square(f, r)) == chess.Piece(
                chess.PAWN, color
            ):
                shield += 1
                break
    ring = [
        chess.square(f, r)
        for f in range(max(0, file_idx - 1), min(8, file_idx + 2))
        for r in range(max(0, rank - 1), min(8, rank + 2))
    ]
    attackers = sum(1 for sq in ring if board.attackers(not color, sq))
    return {
        "square": chess.square_name(king_sq),
        "side": side,
        "on_home_rank": rank == home,
        "pawn_shield": shield,
        "ring_squares_attacked": attackers,
    }


def outposts(board: chess.Board, color: chess.Color) -> List[str]:
    """Knights on squares in enemy territory no enemy pawn can challenge."""
    found = []
    enemy_pawn_files = _pawn_files(board, not color)
    forward = 1 if color == chess.WHITE else -1
    for square in board.pieces(chess.KNIGHT, color):
        file_idx, rank = chess.square_file(square), chess.square_rank(square)
        depth = rank if color == chess.WHITE else 7 - rank
        if depth < 4:
            continue
        challenged = any(
            any((r - rank) * forward > 0 for r in enemy_pawn_files.get(f, []))
            for f in (file_idx - 1, file_idx + 1)
            if 0 <= f < 8
        )
        supported = bool(
            board.attackers(color, square) & board.pieces(chess.PAWN, color)
        )
        if not challenged and supported:
            found.append(chess.square_name(square))
    return found


def mobility(board: chess.Board, color: chess.Color) -> int:
    """Legal move count for `color`, regardless of whose turn it is."""
    probe = board.copy(stack=False)
    probe.turn = color
    try:
        return probe.legal_moves.count()
    except Exception:
        return 0


def rooks_on_open_files(board: chess.Board, color: chess.Color) -> List[str]:
    files = open_files(board)
    relevant = set(files["open"]) | set(
        files["half_open_white" if color == chess.WHITE else "half_open_black"]
    )
    return [
        chess.square_name(sq)
        for sq in board.pieces(chess.ROOK, color)
        if chess.FILE_NAMES[chess.square_file(sq)] in relevant
    ]


def describe(board: chess.Board) -> Dict:
    """The full structural dossier for one position."""
    out = {
        "fen": board.fen(),
        "phase": phase_of(board),
        "centre": centre_state(board),
        "files": open_files(board),
        "material_balance_cp": sum(
            value_of(board.piece_at(sq)) * (1 if board.piece_at(sq).color else -1)
            for sq in chess.SQUARES
            if board.piece_at(sq) is not None
            and board.piece_at(sq).piece_type != chess.KING
        ),
    }
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        out[name] = {
            "pawns": pawn_report(board, color),
            "king": king_safety(board, color),
            "outposts": outposts(board, color),
            "rooks_active": rooks_on_open_files(board, color),
            "mobility": mobility(board, color),
            "bishop_pair": len(board.pieces(chess.BISHOP, color)) >= 2,
        }
    wk = out["white"]["king"].get("side")
    bk = out["black"]["king"].get("side")
    out["opposite_castling"] = bool(
        wk and bk and wk != bk and "centre" not in (wk, bk)
    )
    return out
