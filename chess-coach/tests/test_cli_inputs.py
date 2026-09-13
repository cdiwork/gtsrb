"""Tests for the three ways people hand this tool something it cannot use.

All three used to produce the same unhelpful message ("no games found"), or
in one case no message at all.
"""
from pathlib import Path

from chess_coach.cli import _load_corpus

PGN = """\
[Event "Test"]
[White "retsekoj"]
[Black "trapper"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""


def test_a_real_pgn_loads(tmp_path):
    path = tmp_path / "g.pgn"
    path.write_text(PGN)
    assert len(_load_corpus(path, ["retsekoj"])) == 1


def test_a_bare_fen_is_rejected_with_an_explanation(tmp_path, capsys):
    """A FEN is a position, not a game: there are no moves to judge."""
    path = tmp_path / "pos.fen"
    path.write_text("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n")
    assert _load_corpus(path, ["retsekoj"]) is None
    err = capsys.readouterr().err
    assert "no games with any moves" in err
    assert "FEN" in err


def test_a_game_with_no_moves_is_rejected(tmp_path, capsys):
    path = tmp_path / "empty.pgn"
    path.write_text('[Event "x"]\n[White "retsekoj"]\n[Result "*"]\n\n*\n')
    assert _load_corpus(path, ["retsekoj"]) is None
    assert "no games with any moves" in capsys.readouterr().err


def test_the_wrong_handle_lists_who_actually_played(tmp_path, capsys):
    path = tmp_path / "g.pgn"
    path.write_text(PGN)
    assert _load_corpus(path, ["someone_else"]) is None
    err = capsys.readouterr().err
    assert "nobody called 'someone_else'" in err
    assert "retsekoj" in err and "trapper" in err


def test_multiple_handles_are_named_readably(tmp_path, capsys):
    path = tmp_path / "g.pgn"
    path.write_text(PGN)
    _load_corpus(path, ["alias_one", "alias_two"])
    assert "alias_one / alias_two" in capsys.readouterr().err
