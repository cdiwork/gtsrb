"""Talking to Stockfish.

Two passes, for a reason. A shallow sweep over every position is cheap and
finds the candidates; a deep multi-PV look at just those candidates is what
you need before telling somebody they blundered. Judging a whole game at
depth 20 wastes minutes on moves nobody was ever going to question, and
judging it at depth 12 accuses people of blunders that aren't there.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional, Sequence

import chess
import chess.engine

from .evalscale import score_to_win

# Places Stockfish actually lives, in the order worth trying.
_CANDIDATES = [
    "stockfish",
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "/usr/bin/stockfish",
]

_INSTALL_HINT = """Stockfish was not found. Install it with one of:

  Debian/Ubuntu   sudo apt install stockfish
  macOS           brew install stockfish
  Windows/any     download from https://stockfishchess.org/download/

Then either put it on your PATH or point at it explicitly:

  export CHESS_COACH_STOCKFISH=/path/to/stockfish
  chess-coach analyse games.pgn --user you --engine /path/to/stockfish
"""


def find_engine(explicit: Optional[str] = None) -> str:
    """Locate a Stockfish binary, or explain how to get one."""
    for candidate in [explicit, os.environ.get("CHESS_COACH_STOCKFISH")] + _CANDIDATES:
        if not candidate:
            continue
        resolved = shutil.which(candidate) or (
            candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None
        )
        if resolved:
            return resolved
    raise FileNotFoundError(_INSTALL_HINT)


@dataclass
class Judgement:
    """One engine opinion, from the point of view of the side to move."""

    score: chess.engine.Score
    win: float
    pv: List[chess.Move]
    depth: int

    @property
    def best(self) -> Optional[chess.Move]:
        return self.pv[0] if self.pv else None


class Analyst:
    """A Stockfish process with the settings this tool wants."""

    def __init__(
        self,
        path: Optional[str] = None,
        threads: Optional[int] = None,
        hash_mb: int = 256,
    ):
        self.path = find_engine(path)
        self.threads = threads or max(1, (os.cpu_count() or 2) - 1)
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def __enter__(self) -> "Analyst":
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        options = self._engine.options
        wanted = {}
        if "Threads" in options:
            wanted["Threads"] = self.threads
        if "Hash" in options:
            wanted["Hash"] = self.hash_mb
        if wanted:
            self._engine.configure(wanted)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except chess.engine.EngineTerminatedError:
                pass
            self._engine = None

    @property
    def id(self) -> str:
        if self._engine is None:
            return "stockfish (not started)"
        return self._engine.id.get("name", "stockfish")

    def judge(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
        movetime: Optional[float] = None,
        nodes: Optional[int] = None,
        multipv: int = 1,
    ) -> List[Judgement]:
        """Analyse `board`, best line first, scores from the mover's side."""
        if self._engine is None:
            raise RuntimeError("Analyst must be used as a context manager")
        if board.is_game_over():
            return []
        limit = chess.engine.Limit(
            depth=depth, time=movetime, nodes=nodes
        )
        raw = self._engine.analyse(board, limit, multipv=multipv)
        if isinstance(raw, dict):
            raw = [raw]
        out = []
        for info in raw:
            score = info.get("score")
            if score is None:
                continue
            relative = score.relative
            out.append(
                Judgement(
                    score=relative,
                    win=score_to_win(relative),
                    pv=list(info.get("pv", [])),
                    depth=info.get("depth", 0),
                )
            )
        return out


def line_san(board: chess.Board, moves: Sequence[chess.Move], limit: int = 8) -> str:
    """Render a principal variation as readable SAN from `board`."""
    legal = []
    probe = board.copy(stack=False)
    for move in moves[:limit]:
        if move not in probe.legal_moves:
            break
        legal.append(move)
        probe.push(move)
    return board.variation_san(legal) if legal else ""
