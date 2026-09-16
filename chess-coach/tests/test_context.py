"""Tests for the pre-game context join.

The point of this module is to be hard to fool yourself with, so most of
these tests check that it *declines* to say something.
"""
import csv

from chess_coach.context import (
    TEMPLATE_COLUMNS, correlate, load_context, write_template,
)
from test_profile import make_game, make_move


def _games(n, score_by_index):
    out = []
    for index in range(n):
        game = make_game([make_move()], result=score_by_index(index))
        game["id"] = f"g{index}"
        game["accuracy"] = 80.0
        game["mean_loss"] = 4.0
        out.append(game)
    return out


def _context(n, value_by_index, column="hours_slept"):
    return {f"g{index}": {column: str(value_by_index(index))} for index in range(n)}


def test_template_has_one_row_per_game_and_blank_columns(tmp_path):
    path = tmp_path / "context.csv"
    written = write_template(_games(3, lambda i: "win"), str(path))
    assert written == 3
    rows = list(csv.DictReader(path.open()))
    assert [r["game_id"] for r in rows] == ["g0", "g1", "g2"]
    assert set(rows[0]) == set(TEMPLATE_COLUMNS)
    assert rows[0]["hours_slept"] == ""


def test_round_trip_through_the_csv(tmp_path):
    path = tmp_path / "context.csv"
    write_template(_games(2, lambda i: "win"), str(path))
    text = path.read_text().replace("g0,,,,", "g0,8,morning,0,4")
    path.write_text(text)
    loaded = load_context(str(path))
    assert loaded["g0"]["hours_slept"] == "8"
    assert loaded["g0"]["time_of_day"] == "morning"
    assert "game_id" not in loaded["g0"]


def test_rows_without_a_game_id_are_dropped(tmp_path):
    path = tmp_path / "context.csv"
    path.write_text("game_id,hours_slept\n,7\ng1,8\n")
    assert list(load_context(str(path))) == ["g1"]


def test_unmatched_ids_are_reported_not_silently_ignored():
    report = correlate(_games(3, lambda i: "win"), _context(3, lambda i: 7))
    assert report["games_total"] == 3
    report = correlate(_games(3, lambda i: "win"), {"elsewhere": {"hours_slept": "7"}})
    assert report["games_with_context"] == 0


def test_a_constant_column_produces_no_correlation():
    """Everything logged as 8 hours: there is no variation to correlate."""
    report = correlate(_games(6, lambda i: "win"), _context(6, lambda i: 8))
    assert report["correlations"] == []


def test_three_games_is_too_few_to_correlate():
    report = correlate(_games(3, lambda i: "win"), _context(3, lambda i: i))
    assert report["correlations"] == []
    assert report["games_with_context"] == 3


def test_a_perfect_relationship_is_found_and_flagged():
    games = _games(8, lambda i: "win" if i >= 4 else "loss")
    report = correlate(games, _context(8, lambda i: 4 + i))
    row = next(r for r in report["correlations"] if r["metric"] == "score")
    assert row["r"] > 0.8
    assert row["notable"] is True


def test_a_weak_relationship_is_shown_but_not_flagged():
    """r has to clear the bar for the sample size, not just be non-zero."""
    games = _games(6, lambda i: "win" if i in (0, 3, 4) else "loss")
    report = correlate(games, _context(6, lambda i: i))
    row = next(r for r in report["correlations"] if r["metric"] == "score")
    assert abs(row["r"]) < row["needed_for_significance"]
    assert row["notable"] is False


def test_word_valued_columns_are_grouped_instead_of_correlated():
    games = _games(4, lambda i: "win" if i < 2 else "loss")
    context = _context(4, lambda i: "morning" if i < 2 else "night",
                       column="time_of_day")
    report = correlate(games, context)
    assert report["correlations"] == []
    values = {row["value"]: row for row in report["groups"]}
    assert values["morning"]["score"] == 1.0
    assert values["night"]["score"] == 0.0


def test_a_lone_group_member_is_not_reported_as_a_pattern():
    games = _games(4, lambda i: "win")
    context = _context(4, lambda i: "night" if i == 0 else "morning",
                       column="time_of_day")
    report = correlate(games, context)
    assert [row["value"] for row in report["groups"]] == ["morning"]


def test_small_samples_always_carry_the_warning():
    report = correlate(_games(6, lambda i: "win"), _context(6, lambda i: i))
    assert any("6 games carry context" in c for c in report["caveats"])


def test_many_comparisons_carry_the_multiple_testing_warning():
    games = _games(8, lambda i: "win" if i % 2 else "loss")
    for index, game in enumerate(games):  # give every metric something to vary
        game["accuracy"] = 70.0 + index
        game["mean_loss"] = 10.0 - index
        game["counts"]["blunder"] = index % 3
    context = {
        f"g{index}": {"hours_slept": str(index), "games_already_today": str(8 - index),
                      "sharpness_1_5": str(index % 5)}
        for index in range(8)
    }
    report = correlate(games, context)
    assert report["comparisons"] >= 10
    assert any("chance alone" in c for c in report["caveats"])


def test_the_before_the_game_point_is_always_made():
    report = correlate(_games(6, lambda i: "win"), _context(6, lambda i: i))
    assert any("before" in c for c in report["caveats"])
