import io
import shutil

import chess
import chess.pgn
import pytest

from chess_coach.analyse import (
    AnalysisConfig, _attach_times, MoveRecord, analyse_game, hero_color,
    legal_move_mix, parse_time_control,
)

TRAP_GAME = """\
[Event "Test"]
[White "retsekoj"]
[Black "trapper"]
[Result "0-1"]
[TimeControl "300+0"]
[Opening "Blackburne Shilling Gambit"]

1. e4 {[%clk 0:04:58]} e5 {[%clk 0:04:59]} 2. Nf3 {[%clk 0:04:55]} Nc6 {[%clk 0:04:57]}
3. Bc4 {[%clk 0:04:52]} Nd4 {[%clk 0:04:55]} 4. Nxe5 {[%clk 0:04:48]} Qg5 {[%clk 0:04:53]}
5. Nxf7 {[%clk 0:04:40]} Qxg2 {[%clk 0:04:51]} 6. Rf1 {[%clk 0:04:31]} Qxe4+ {[%clk 0:04:49]}
7. Be2 {[%clk 0:04:20]} Nf3# {[%clk 0:04:47]} 0-1
"""


def _game():
    return chess.pgn.read_game(io.StringIO(TRAP_GAME))


@pytest.mark.parametrize("value,expected", [
    ("180+2", (180, 2)), ("600", (600, 0)), ("", (None, 0)),
    ("1/259200", (None, 0)), ("-", (None, 0)),
])
def test_parse_time_control(value, expected):
    assert parse_time_control(value) == expected


def test_hero_color_is_case_insensitive():
    game = _game()
    assert hero_color(game, "RetseKoj") == chess.WHITE
    assert hero_color(game, "trapper") == chess.BLACK
    assert hero_color(game, "someone else") is None


def test_legal_move_mix_of_the_opening_position():
    mix = legal_move_mix(chess.Board())
    assert mix["legal_moves"] == 20
    assert mix["captures"] == 0.0
    assert mix["checks"] == 0.0
    assert mix["backward"] == 0.0
    assert mix["quiet"] == 1.0


def test_legal_move_mix_counts_backward_moves_for_black():
    board = chess.Board("4k3/8/8/4r3/8/8/8/6K1 b - - 0 1")
    mix = legal_move_mix(board)
    # The rook can go to e6, e7, e8 (backward for black) among others.
    assert 0 < mix["backward"] < 1


def _record(clock, **kwargs):
    return MoveRecord(
        ply=0, move_number=1, side="white", san="e4", uci="e2e4",
        fen_before="", win_before=50, win_after=50, loss=0, accuracy=100,
        severity=None, band_before="equal", phase="opening", clock=clock, **kwargs
    )


def test_attach_times_uses_the_clock_difference_and_increment():
    records = [_record(298.0), _record(295.0), _record(290.0)]
    _attach_times(records, base_time=300, increment=2)
    assert records[0].time_spent == 4.0   # 300 -> 298, plus 2 increment
    assert records[1].time_spent == 5.0
    assert records[2].time_spent == 7.0


def test_attach_times_ignores_missing_clocks():
    records = [_record(None), _record(None)]
    _attach_times(records, base_time=300, increment=0)
    assert all(r.time_spent is None for r in records)


def test_attach_times_rejects_impossible_values():
    """A clock that jumps upwards means the PGN is odd, not that time was
    spent; better to report nothing than nonsense."""
    records = [_record(100.0), _record(7000.0)]
    _attach_times(records, base_time=300, increment=0)
    assert records[1].time_spent is None


# --------------------------------------------------------------------------
# Engine integration
# --------------------------------------------------------------------------

def _engine_available():
    from chess_coach.engine import find_engine
    try:
        find_engine()
        return True
    except FileNotFoundError:
        return False


needs_engine = pytest.mark.skipif(
    not _engine_available(), reason="Stockfish not installed"
)


@needs_engine
def test_analyse_game_finds_the_trap():
    """5.Nxf7 in the Blackburne Shilling trap must come out as a blunder, with
    the tactic named and the clock read."""
    from chess_coach.engine import Analyst

    with Analyst(threads=2, hash_mb=64) as analyst:
        report = analyse_game(
            _game(), "retsekoj", analyst,
            AnalysisConfig(fast_depth=12, deep_depth=16),
        )
    assert report is not None
    assert report["hero_color"] == "white"
    assert report["result"] == "loss"
    assert report["base_time"] == 300

    by_san = {m["san"]: m for m in report["moves"]}
    assert by_san["Nxf7"]["severity"] == "blunder"
    assert by_san["Nxf7"]["deep"] is True
    assert by_san["Nxf7"]["loss"] > 20
    # The refutation is a queen fork; the detectors should have named something.
    assert by_san["Nxf7"]["allowed_motifs"]
    # Clocks present, so thinking time must be populated.
    assert by_san["e4"]["time_spent"] == 2.0
    # The eval curve runs from roughly level to lost.
    curve = [c for c in report["curve"] if c is not None]
    assert curve[0] > 40 and curve[-1] < 10


@needs_engine
def test_analyse_game_returns_none_for_a_stranger():
    from chess_coach.engine import Analyst

    with Analyst(threads=1, hash_mb=32) as analyst:
        assert analyse_game(_game(), "nobody", analyst) is None
