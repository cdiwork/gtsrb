"""Naming the tactic.

Knowing a move cost 30% win probability is not coaching. Knowing that it
cost it *to a knight fork you didn't look for* is. This module labels two
different things, and the distinction matters more than any single label:

  missed  -- the tactic in the move you failed to find
  allowed -- the tactic in your opponent's refutation

A player who misses forks and a player who walks into them have different
problems and need different homework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import chess

from .see import PIECE_VALUE, see, value_of

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

_DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_ORTH = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _slider_dirs(piece_type: int) -> List[tuple]:
    if piece_type == chess.BISHOP:
        return _DIAG
    if piece_type == chess.ROOK:
        return _ORTH
    if piece_type == chess.QUEEN:
        return _DIAG + _ORTH
    return []


def _scan(board: chess.Board, square: int, df: int, dr: int, limit: int = 2):
    """First `limit` occupied squares along one direction, nearest first."""
    f, r = chess.square_file(square), chess.square_rank(square)
    found = []
    while len(found) < limit:
        f, r = f + df, r + dr
        if not (0 <= f < 8 and 0 <= r < 8):
            break
        sq = chess.square(f, r)
        piece = board.piece_at(sq)
        if piece is not None:
            found.append((sq, piece))
    return found


# --------------------------------------------------------------------------
# Surface properties of a move
# --------------------------------------------------------------------------


@dataclass
class MoveShape:
    """What a move looks like, before asking whether it was any good.

    These are the features human pattern recognition is known to be uneven
    about -- backward moves and long quiet moves are measurably harder to
    see than forward captures -- so they are worth recording separately
    from the tactic itself.
    """

    san: str
    uci: str
    piece: str
    is_capture: bool
    is_check: bool
    is_promotion: bool
    is_castling: bool
    is_forcing: bool
    is_quiet: bool
    direction: str
    span: int
    see: int
    is_sacrifice: bool
    enters_enemy_half: bool

    def tags(self) -> List[str]:
        out = [f"piece:{self.piece}"]
        if self.is_capture:
            out.append("capture")
        if self.is_check:
            out.append("check")
        if self.is_promotion:
            out.append("promotion")
        if self.is_castling:
            out.append("castling")
        if self.is_quiet:
            out.append("quiet_move")
        if self.is_sacrifice:
            out.append("sacrifice")
        if self.direction == "backward":
            out.append("backward_move")
        if self.span >= 4 and not self.is_capture:
            out.append("long_quiet_move")
        return out


def shape_of(board: chess.Board, move: chess.Move) -> MoveShape:
    """Describe `move` in the position `board`, from the mover's side."""
    mover = board.turn
    piece = board.piece_at(move.from_square)
    is_capture = board.is_capture(move)
    is_check = board.gives_check(move)
    df = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dr = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    if mover == chess.BLACK:
        dr = -dr
    direction = "forward" if dr > 0 else ("backward" if dr < 0 else "sideways")
    to_rank = chess.square_rank(move.to_square)
    if mover == chess.BLACK:
        to_rank = 7 - to_rank
    exchange = see(board, move)
    return MoveShape(
        san=board.san(move),
        uci=move.uci(),
        piece=PIECE_NAMES[piece.piece_type] if piece else "?",
        is_capture=is_capture,
        is_check=is_check,
        is_promotion=move.promotion is not None,
        is_castling=board.is_castling(move),
        is_forcing=is_capture or is_check or move.promotion is not None,
        is_quiet=not (is_capture or is_check or move.promotion is not None),
        direction=direction,
        span=max(abs(df), abs(dr)),
        see=exchange,
        is_sacrifice=exchange <= -150,
        enters_enemy_half=to_rank >= 4,
    )


# --------------------------------------------------------------------------
# Tactical motifs
# --------------------------------------------------------------------------


def _valuable_targets(board: chess.Board, square: int, color: chess.Color):
    """Enemy pieces the piece on `square` hits that are actually worth hitting.

    "Worth hitting" means: the king, something worth more than the attacker,
    or something nobody is defending.
    """
    attacker = board.piece_at(square)
    if attacker is None:
        return []
    hits = []
    for target_sq in board.attacks(square):
        victim = board.piece_at(target_sq)
        if victim is None or victim.color == color:
            continue
        if victim.piece_type == chess.KING:
            hits.append((target_sq, victim))
            continue
        undefended = not board.attackers(victim.color, target_sq)
        if undefended or value_of(victim) > value_of(attacker):
            hits.append((target_sq, victim))
    return hits


def detect_fork(board: chess.Board, move: chess.Move) -> Optional[Dict]:
    """One piece, two or more real threats, after `move` is played."""
    mover = board.turn
    after = board.copy(stack=False)
    after.push(move)
    hits = _valuable_targets(after, move.to_square, mover)
    if len(hits) < 2:
        return None
    piece = after.piece_at(move.to_square)
    return {
        "motif": "fork",
        "by": PIECE_NAMES[piece.piece_type] if piece else "?",
        "targets": [
            f"{PIECE_NAMES[p.piece_type]} {chess.square_name(s)}" for s, p in hits
        ],
    }


def _lines_held(board: chess.Board, color: bool) -> set:
    """Every (front, back) pair `color`'s sliders already have lined up.

    Used to tell a pin that a move *creates* from one that was already
    standing. Without this a queen shuffling around a lone enemy knight
    scores a fresh "absolute pin" on every move of a long endgame.
    """
    held = set()
    for piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in board.pieces(piece_type, color):
            for df, dr in _slider_dirs(piece_type):
                found = _scan(board, square, df, dr)
                if len(found) == 2:
                    (sq_a, piece_a), (sq_b, piece_b) = found
                    if piece_a.color != color and piece_b.color != color:
                        held.add((sq_a, sq_b))
    return held


def detect_pin_or_skewer(board: chess.Board, move: chess.Move) -> Optional[Dict]:
    """A slider lining up two enemy pieces that were not lined up before."""
    mover = board.turn
    after = board.copy(stack=False)
    after.push(move)
    piece = after.piece_at(move.to_square)
    if piece is None:
        return None
    already = _lines_held(board, mover)
    for df, dr in _slider_dirs(piece.piece_type):
        found = _scan(after, move.to_square, df, dr)
        if len(found) != 2:
            continue
        (sq_a, piece_a), (sq_b, piece_b) = found
        if piece_a.color == mover or piece_b.color == mover:
            continue
        if (sq_a, sq_b) in already:
            continue  # the line was already there; this move did not make it
        front, back = value_of(piece_a), value_of(piece_b)
        if piece_b.piece_type == chess.KING:
            motif = "absolute_pin"
        elif back > front:
            motif = "pin"
        elif front > back and front >= PIECE_VALUE[chess.ROOK]:
            motif = "skewer"
        else:
            continue
        return {
            "motif": motif,
            "by": PIECE_NAMES[piece.piece_type],
            "targets": [
                f"{PIECE_NAMES[piece_a.piece_type]} {chess.square_name(sq_a)}",
                f"{PIECE_NAMES[piece_b.piece_type]} {chess.square_name(sq_b)}",
            ],
        }
    return None


def detect_discovered_attack(board: chess.Board, move: chess.Move) -> Optional[Dict]:
    """`move` steps out of the way and uncovers a friendly slider."""
    mover = board.turn

    def slider_threats(position: chess.Board, exclude: int) -> set:
        threats = set()
        for square in position.pieces(chess.BISHOP, mover) | position.pieces(
            chess.ROOK, mover
        ) | position.pieces(chess.QUEEN, mover):
            if square == exclude:
                continue
            for target_sq, _ in _valuable_targets(position, square, mover):
                threats.add((square, target_sq))
        return threats

    after = board.copy(stack=False)
    after.push(move)
    new = slider_threats(after, move.to_square) - slider_threats(
        board, move.from_square
    )
    # Only count threats revealed along the line the moved piece vacated.
    for source, target in new:
        if move.from_square in chess.SquareSet(chess.between(source, target)) or (
            chess.square_distance(source, move.from_square) >= 0
            and move.from_square in chess.SquareSet(chess.ray(source, target))
        ):
            slider = after.piece_at(source)
            victim = after.piece_at(target)
            if slider is None or victim is None:
                continue
            return {
                "motif": "discovered_attack",
                "by": PIECE_NAMES[slider.piece_type],
                "targets": [
                    f"{PIECE_NAMES[victim.piece_type]} {chess.square_name(target)}"
                ],
            }
    return None


def detect_hanging(board: chess.Board, move: chess.Move) -> Optional[Dict]:
    """`move` simply grabs something that was left undefended."""
    if not board.is_capture(move):
        return None
    gain = see(board, move)
    if gain < PIECE_VALUE[chess.PAWN]:
        return None
    victim = board.piece_at(move.to_square)
    if victim is None:
        return {"motif": "wins_material", "gain_cp": gain, "targets": []}
    defended = bool(board.attackers(victim.color, move.to_square))
    return {
        "motif": "undefended_piece" if not defended else "wins_material",
        "gain_cp": gain,
        "targets": [f"{PIECE_NAMES[victim.piece_type]} {chess.square_name(move.to_square)}"],
    }


def detect_trapped_piece(board: chess.Board, move: chess.Move) -> Optional[Dict]:
    """After `move`, an enemy piece is attacked and has nowhere safe to go."""
    mover = board.turn
    after = board.copy(stack=False)
    after.push(move)
    probe = after.copy(stack=False)
    if probe.turn != (not mover):
        return None
    # If the opponent is in check, "it has no safe square" says nothing about
    # the piece -- every reply is forced to deal with the check instead.
    if probe.is_check():
        return None
    for square in chess.SQUARES:
        piece = probe.piece_at(square)
        if (
            piece is None
            or piece.color == mover
            or piece.piece_type in (chess.PAWN, chess.KING)
        ):
            continue
        if not probe.attackers(mover, square):
            continue
        # A piece frozen by a pin is already described as a pin; calling it
        # trapped as well just double-counts the same tactic.
        if probe.is_pinned(piece.color, square):
            continue
        # Is the threat real, and does the piece have a safe retreat?
        threat = max(
            (see(probe, chess.Move(a, square)) for a in probe.attackers(mover, square)),
            default=0,
        )
        if threat < PIECE_VALUE[chess.KNIGHT] - 100:
            continue
        escapes = [m for m in probe.legal_moves if m.from_square == square]
        if escapes and any(see(probe, m) >= -50 for m in escapes):
            continue
        return {
            "motif": "trapped_piece",
            "targets": [f"{PIECE_NAMES[piece.piece_type]} {chess.square_name(square)}"],
        }
    return None


def detect_back_rank(board: chess.Board, line: Sequence[chess.Move]) -> Optional[Dict]:
    """Mate (or forced mate) delivered on the enemy's own back rank."""
    mover = board.turn
    probe = board.copy(stack=False)
    for move in line[:6]:
        if move not in probe.legal_moves:
            break
        probe.push(move)
        if not probe.is_checkmate():
            continue
        king_sq = probe.king(not mover)
        if king_sq is None:
            break
        home = 0 if (not mover) == chess.WHITE else 7
        if chess.square_rank(king_sq) != home:
            break
        last = probe.peek()
        mating = probe.piece_at(last.to_square)
        if mating and mating.piece_type in (chess.ROOK, chess.QUEEN):
            return {"motif": "back_rank_mate", "targets": [chess.square_name(king_sq)]}
        break
    return None


def detect_intermezzo(board: chess.Board, line: Sequence[chess.Move]) -> Optional[Dict]:
    """An in-between check before the expected recapture -- a zwischenzug."""
    if len(line) < 3:
        return None
    first = line[0]
    if not board.gives_check(first) or board.is_capture(first):
        return None
    probe = board.copy(stack=False)
    probe.push(first)
    if line[1] not in probe.legal_moves:
        return None
    probe.push(line[1])
    if line[2] in probe.legal_moves and probe.is_capture(line[2]):
        return {"motif": "zwischenzug", "targets": []}
    return None


def detect_removal_of_guard(
    board: chess.Board, line: Sequence[chess.Move]
) -> Optional[Dict]:
    """First capture the defender, then take what it was defending."""
    if len(line) < 3 or not board.is_capture(line[0]):
        return None
    guard_sq = line[0].to_square
    guard = board.piece_at(guard_sq)
    if guard is None:
        return None
    defended = {
        sq
        for sq in board.attacks(guard_sq)
        if (p := board.piece_at(sq)) is not None and p.color == guard.color
    }
    probe = board.copy(stack=False)
    for move in line[:4]:
        if move not in probe.legal_moves:
            return None
        capture = probe.is_capture(move)
        target = move.to_square
        probe.push(move)
        if capture and target in defended and target != guard_sq:
            return {
                "motif": "removal_of_the_guard",
                "targets": [chess.square_name(target)],
            }
    return None


def classify_line(
    board: chess.Board,
    line: Sequence[chess.Move],
    score_is_mate: bool = False,
) -> Dict:
    """Everything we can say about the tactic in `line` from `board`.

    Returns the move's shape plus every motif that fires. Motifs are not
    mutually exclusive on purpose -- a fork that also wins a hanging piece
    is genuinely both, and collapsing them would lose information.
    """
    if not line:
        return {"shape": None, "motifs": [], "detail": []}
    first = line[0]
    if first not in board.legal_moves:
        return {"shape": None, "motifs": [], "detail": []}

    shape = shape_of(board, first)
    detail = []
    for detector in (
        detect_hanging,
        detect_fork,
        detect_pin_or_skewer,
        detect_discovered_attack,
        detect_trapped_piece,
    ):
        try:
            found = detector(board, first)
        except Exception:
            found = None
        if found:
            detail.append(found)
    for line_detector in (detect_back_rank, detect_intermezzo, detect_removal_of_guard):
        try:
            found = line_detector(board, line)
        except Exception:
            found = None
        if found:
            detail.append(found)
    if score_is_mate and not any(d["motif"] == "back_rank_mate" for d in detail):
        detail.append({"motif": "mating_attack", "targets": []})

    motifs = [d["motif"] for d in detail]
    return {
        "shape": shape,
        "motifs": motifs,
        "detail": detail,
        "tags": shape.tags() + [f"motif:{m}" for m in motifs],
    }
