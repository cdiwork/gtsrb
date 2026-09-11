import chess.engine
import pytest

from chess_coach import evalscale as ev


def test_zero_is_even():
    assert ev.cp_to_win(0) == pytest.approx(50.0)


def test_win_percentage_is_monotonic_and_bounded():
    values = [ev.cp_to_win(cp) for cp in range(-2000, 2001, 100)]
    assert values == sorted(values)
    assert 0 <= values[0] < 5 and 95 < values[-1] <= 100


def test_mate_scores_saturate():
    assert ev.score_to_win(chess.engine.Mate(3)) == 100.0
    assert ev.score_to_win(chess.engine.Mate(-3)) == 0.0


def test_accuracy_falls_with_the_size_of_the_error():
    assert ev.accuracy(60, 60) == pytest.approx(100, abs=0.5)
    assert ev.accuracy(60, 50) < 70
    assert ev.accuracy(60, 20) < 25


def test_severity_thresholds():
    assert ev.severity(1) is None
    assert ev.severity(6) == "inaccuracy"
    assert ev.severity(12) == "mistake"
    assert ev.severity(25) == "blunder"


def test_bands_describe_the_right_side():
    assert ev.eval_band(90) == "winning"
    assert ev.eval_band(50) == "equal"
    assert ev.eval_band(5) == "losing"
