"""Turning one game into a list of judged decisions.

Every move you made gets a cost in win percentage, and every move that cost
something real gets a second, deeper look before it goes on your record --
plus the two things that make a mistake teachable: what you missed, and what
your opponent used to punish you.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn

from . import evalscale as ev
from .engine import Analyst, Judgement, line_san
from .motifs import classify_line
from .structure import describe, phase_of


@dataclass
class AnalysisConfig:
    """Knobs for the speed/confidence trade-off."""

    fast_depth: int = 14
    deep_depth: int = 20
    multipv: int = 3
    # A shallow loss this big earns a deep second opinion. Deliberately below
    # the inaccuracy threshold so borderline cases get the better search
    # rather than being judged on the cheap pass.
    flag_threshold: float = 3.0
    movetime: Optional[float] = None
    deep_movetime: Optional[float] = None
    max_deep_per_game: int = 24
    structure_for_errors: bool = True


@dataclass
class MoveRecord:
    ply: int
    move_number: int
    side: str
    san: str
    uci: str
    fen_before: str
    win_before: float
    win_after: float
    loss: float
    accuracy: float
    severity: Optional[str]
    band_before: str
    phase: str
    best_san: Optional[str] = None
    best_line_san: Optional[str] = None
    played_rank: Optional[int] = None
    was_best: bool = False
    refutation_san: Optional[str] = None
    clock: Optional[float] = None
    time_spent: Optional[float] = None
    complexity: Optional[float] = None
    close_choices: Optional[int] = None
    only_move: bool = False
    deep: bool = False
    shape_tags: List[str] = field(default_factory=list)
    best_shape_tags: List[str] = field(default_factory=list)
    missed_motifs: List[str] = field(default_factory=list)
    allowed_motifs: List[str] = field(default_factory=list)
    allowed_shape_tags: List[str] = field(default_factory=list)
    missed_detail: List[Dict] = field(default_factory=list)
    allowed_detail: List[Dict] = field(default_factory=list)
    # What the available moves looked like in this position. Without this
    # there is no control group: "you miss backward moves" means nothing
    # until you know what share of the legal moves were backward.
    legal_mix: Optional[Dict] = None
    # The same control group for the opponent's position after your move:
    # "you don't check your opponent's forcing replies" needs to know how
    # many forcing replies there were.
    legal_mix_after: Optional[Dict] = None
    structure: Optional[Dict] = None


def _terminal_win(board: chess.Board) -> Optional[float]:
    """Win% for the side to move when the game is already over."""
    if board.is_checkmate():
        return 0.0
    if board.is_game_over(claim_draw=False):
        return 50.0
    return None


def parse_time_control(value: str) -> Tuple[Optional[int], int]:
    """'180+2' -> (180, 2). Returns (None, 0) for untimed or odd formats."""
    if not value:
        return None, 0
    match = re.match(r"^(\d+)(?:\+(\d+))?$", value.strip())
    if not match:
        return None, 0
    return int(match.group(1)), int(match.group(2) or 0)


def hero_color(game: chess.pgn.Game, user) -> Optional[chess.Color]:
    """Which side the user played, matched case-insensitively.

    `user` may be several names: people rarely have the same handle on
    Chess.com and Lichess, and a combined corpus has to work anyway.
    """
    names = [user] if isinstance(user, str) else list(user)
    targets = {n.strip().lower() for n in names if n and n.strip()}
    if game.headers.get("White", "").strip().lower() in targets:
        return chess.WHITE
    if game.headers.get("Black", "").strip().lower() in targets:
        return chess.BLACK
    return None


def _opening_name(headers: chess.pgn.Headers) -> str:
    name = headers.get("Opening", "").strip()
    if name and name != "?":
        return name
    url = headers.get("ECOUrl", "")
    if url:
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        return slug.replace("-", " ")
    return headers.get("ECO", "").strip() or "unknown"


def _result_for(headers: chess.pgn.Headers, color: chess.Color) -> str:
    result = headers.get("Result", "*")
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if color == chess.WHITE else "loss"
    if result == "0-1":
        return "loss" if color == chess.WHITE else "win"
    return "unknown"


def analyse_game(
    game: chess.pgn.Game,
    user,
    analyst: Analyst,
    config: Optional[AnalysisConfig] = None,
    progress=None,
) -> Optional[Dict]:
    """Judge every move the user made in `game`.

    Returns None when the user did not play in this game, so callers can
    filter a mixed PGN export without special-casing.
    """
    config = config or AnalysisConfig()
    color = hero_color(game, user)
    if color is None:
        return None

    base_time, increment = parse_time_control(game.headers.get("TimeControl", ""))

    # --- collect the game ------------------------------------------------
    board = game.board()
    steps = []
    for node in game.mainline():
        move = node.move
        if move is None:
            break
        steps.append(
            {
                "board": board.copy(stack=False),
                "move": move,
                "clock": node.clock(),
                "san": board.san(move),
            }
        )
        board.push(move)
    final_board = board
    if not steps:
        return None

    # --- pass one: a cheap opinion on every position ---------------------
    positions = [s["board"] for s in steps] + [final_board]
    shallow: List[Optional[Judgement]] = []
    for index, position in enumerate(positions):
        terminal = _terminal_win(position)
        if terminal is not None:
            shallow.append(Judgement(chess.engine.Cp(0), terminal, [], 0))
            continue
        found = analyst.judge(
            position, depth=config.fast_depth, movetime=config.movetime
        )
        shallow.append(found[0] if found else None)
        if progress:
            progress(index, len(positions))

    def win_at(index: int) -> Optional[float]:
        judgement = shallow[index]
        return None if judgement is None else judgement.win

    # --- the user's moves, with losses from the cheap pass ---------------
    records: List[MoveRecord] = []
    candidates: List[int] = []
    for index, step in enumerate(steps):
        position = step["board"]
        if position.turn != color:
            continue
        before = win_at(index)
        after_raw = win_at(index + 1)
        if before is None or after_raw is None:
            continue
        after = 100.0 - after_raw
        loss = max(0.0, before - after)
        shallow_best = shallow[index].best if shallow[index] else None
        record = MoveRecord(
            ply=index,
            move_number=position.fullmove_number,
            side="white" if color == chess.WHITE else "black",
            san=step["san"],
            uci=step["move"].uci(),
            fen_before=position.fen(),
            win_before=round(before, 2),
            win_after=round(after, 2),
            loss=round(loss, 2),
            accuracy=round(ev.accuracy(before, after), 1),
            severity=ev.severity(loss),
            band_before=ev.eval_band(before),
            phase=phase_of(position),
            clock=step["clock"],
            was_best=shallow_best == step["move"],
            best_san=position.san(shallow_best) if shallow_best else None,
        )
        records.append(record)
        if loss >= config.flag_threshold:
            candidates.append(len(records) - 1)

    # --- time spent per move, from the clock comments --------------------
    _attach_times(records, base_time, increment)

    # --- pass two: a deep second opinion where it matters ----------------
    candidates.sort(key=lambda i: -records[i].loss)
    for position_in_list in candidates[: config.max_deep_per_game]:
        _deepen(records[position_in_list], steps, positions, analyst, config, color)

    # --- game-level summary ----------------------------------------------
    curve = []
    for index in range(len(positions)):
        win = win_at(index)
        if win is None:
            curve.append(None)
            continue
        mover = positions[index].turn
        curve.append(round(win if mover == color else 100.0 - win, 1))

    losses = [r.loss for r in records]
    counts = {"blunder": 0, "mistake": 0, "inaccuracy": 0}
    for record in records:
        if record.severity:
            counts[record.severity] += 1

    return {
        "id": game.headers.get("Site", "") or game.headers.get("Link", ""),
        "date": game.headers.get("UTCDate", game.headers.get("Date", "")),
        "time": game.headers.get("UTCTime", ""),
        "white": game.headers.get("White", ""),
        "black": game.headers.get("Black", ""),
        "white_elo": game.headers.get("WhiteElo", ""),
        "black_elo": game.headers.get("BlackElo", ""),
        "hero": game.headers.get("White" if color == chess.WHITE else "Black", ""),
        "hero_color": "white" if color == chess.WHITE else "black",
        "hero_elo": game.headers.get(
            "WhiteElo" if color == chess.WHITE else "BlackElo", ""
        ),
        "opponent_elo": game.headers.get(
            "BlackElo" if color == chess.WHITE else "WhiteElo", ""
        ),
        "result": _result_for(game.headers, color),
        "termination": game.headers.get("Termination", ""),
        "eco": game.headers.get("ECO", ""),
        "opening": _opening_name(game.headers),
        "time_control": game.headers.get("TimeControl", ""),
        "base_time": base_time,
        "increment": increment,
        "moves_analysed": len(records),
        "mean_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "accuracy": round(
            sum(r.accuracy for r in records) / len(records), 1
        ) if records else 0.0,
        "counts": counts,
        "curve": curve,
        "moves": [asdict(r) for r in records],
    }


def legal_move_mix(board: chess.Board) -> Dict[str, float]:
    """What fraction of the legal moves here are captures, checks, backward...

    This is the control group for every "you are blind to X moves" claim. If
    18% of your legal moves are backward and 17% of the moves you miss are
    backward, you are not blind to backward moves.
    """
    legal = list(board.legal_moves)
    if not legal:
        return {}
    mover = board.turn
    captures = checks = backward = quiet = 0
    for move in legal:
        is_capture = board.is_capture(move)
        is_check = board.gives_check(move)
        if is_capture:
            captures += 1
        if is_check:
            checks += 1
        if not (is_capture or is_check or move.promotion):
            quiet += 1
        delta = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
        if mover == chess.BLACK:
            delta = -delta
        if delta < 0:
            backward += 1
    total = float(len(legal))
    return {
        "legal_moves": len(legal),
        "captures": round(captures / total, 3),
        "checks": round(checks / total, 3),
        "backward": round(backward / total, 3),
        "quiet": round(quiet / total, 3),
    }


def _attach_times(
    records: List[MoveRecord], base_time: Optional[int], increment: int
) -> None:
    """Work out seconds spent per move from the remaining-clock comments."""
    previous: Optional[float] = float(base_time) if base_time else None
    for record in records:
        if record.clock is None:
            continue
        if previous is not None:
            spent = previous - record.clock + increment
            if -1 <= spent <= 3600:
                record.time_spent = round(max(0.0, spent), 1)
        previous = record.clock


def _deepen(
    record: MoveRecord,
    steps: List[Dict],
    positions: List[chess.Board],
    analyst: Analyst,
    config: AnalysisConfig,
    color: chess.Color,
) -> None:
    """Re-judge one suspicious move properly, and name the tactics involved."""
    index = record.ply
    before_board = positions[index]
    after_board = positions[index + 1]
    played = steps[index]["move"]

    deep_before = analyst.judge(
        before_board,
        depth=config.deep_depth,
        movetime=config.deep_movetime,
        multipv=config.multipv,
    )
    if not deep_before:
        return

    terminal = _terminal_win(after_board)
    if terminal is not None:
        after_win, refutation = terminal, None
    else:
        deep_after = analyst.judge(
            after_board, depth=config.deep_depth, movetime=config.deep_movetime
        )
        if not deep_after:
            return
        after_win, refutation = deep_after[0].win, deep_after[0]

    best = deep_before[0]
    win_before = best.win
    win_after = 100.0 - after_win
    loss = max(0.0, win_before - win_after)

    record.deep = True
    record.win_before = round(win_before, 2)
    record.win_after = round(win_after, 2)
    record.loss = round(loss, 2)
    record.accuracy = round(ev.accuracy(win_before, win_after), 1)
    record.severity = ev.severity(loss)
    record.band_before = ev.eval_band(win_before)
    record.best_san = before_board.san(best.best) if best.best else None
    record.best_line_san = line_san(before_board, best.pv, 8)
    record.was_best = best.best == played
    record.played_rank = next(
        (n for n, j in enumerate(deep_before, 1) if j.best == played), None
    )

    # How sharp was the position? A big gap to the third choice means there
    # was one move to find; lots of near-equal options means it was murky.
    if len(deep_before) >= 2:
        record.complexity = round(win_before - deep_before[-1].win, 1)
        record.close_choices = sum(
            1 for j in deep_before if win_before - j.win <= 5.0
        )
        record.only_move = (win_before - deep_before[1].win) >= 15.0

    record.shape_tags = classify_line(before_board, [played]).get("tags", [])
    record.legal_mix = legal_move_mix(before_board)
    record.legal_mix_after = legal_move_mix(after_board)

    if not record.was_best and best.pv:
        missed = classify_line(before_board, best.pv, best.score.is_mate())
        record.missed_motifs = missed["motifs"]
        record.missed_detail = missed["detail"]
        record.best_shape_tags = missed.get("tags", [])

    if refutation is not None and refutation.pv:
        allowed = classify_line(
            after_board, refutation.pv, refutation.score.is_mate()
        )
        record.allowed_motifs = allowed["motifs"]
        record.allowed_detail = allowed["detail"]
        record.allowed_shape_tags = allowed.get("tags", [])
        record.refutation_san = line_san(after_board, refutation.pv, 6)

    if config.structure_for_errors and record.severity:
        record.structure = describe(before_board)
