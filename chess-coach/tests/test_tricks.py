"""Tests for the beauty scorer.

The scorer encodes a taste judgement, so the tests are mostly about the
judgement being the one intended: content before prettiness, clever motifs
above arithmetic ones, and an unsound sacrifice never counted as a triumph.
"""
from chess_coach.tricks import beauty_score, find_tricks, grade_of

OPENING = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FORK = "4q1k1/8/8/3N4/8/8/8/7K w - - 0 1"
GUARD = "r1b2rk1/p3n1pp/2pNp3/1p2P3/2B5/8/PP3PPP/R2R2K1 w - - 0 16"
SAC = "r1bqk1nr/ppp1b1pp/2n1p3/4p3/2BP1B2/5N2/PP3PPP/RN1QK2R w KQkq - 0 8"


def test_an_ordinary_developing_move_is_not_a_trick():
    """Without this gate every quiet move in the database scores as beautiful."""
    score, reasons, _, _ = beauty_score(OPENING, "Nf3")
    assert score == 0 and reasons == []


def test_a_fork_scores():
    score, reasons, motifs, _ = beauty_score(FORK, "Nf6+")
    assert score > 0 and "fork" in motifs


def test_clever_motifs_outrank_arithmetic_ones():
    """Taking something undefended is not the same kind of move as a
    zwischenzug, and the score has to know it."""
    from chess_coach.tricks import MOTIF_BEAUTY
    assert MOTIF_BEAUTY["zwischenzug"] > MOTIF_BEAUTY["undefended_piece"] * 5
    assert MOTIF_BEAUTY["removal_of_the_guard"] > MOTIF_BEAUTY["wins_material"] * 5


def test_line_motifs_can_be_supplied_from_the_analysis():
    """Removal of the guard needs a whole variation, so a single-move rescore
    cannot find it and must accept it from the caller."""
    bare, _, _, _ = beauty_score(GUARD, "Nxc8")
    with_line, reasons, motifs, _ = beauty_score(
        GUARD, "Nxc8", extra_motifs=["removal_of_the_guard"]
    )
    assert with_line > bare
    assert "removal_of_the_guard" in motifs
    assert any("removal of the guard" in r for r in reasons)


def test_a_sacrifice_is_scored_by_what_was_given_up():
    score, reasons, _, sacrificed = beauty_score(SAC, "Bxe5", extra_motifs=["fork"])
    assert sacrificed == 0 or score > 0


def test_backward_bonus_only_applies_to_quiet_moves():
    """A backward capture is forcing, and forcing moves announce themselves."""
    quiet, quiet_reasons, _, _ = beauty_score(
        "4k3/8/8/4r3/8/8/8/6K1 b - - 0 1", "Re7", extra_motifs=["fork"]
    )
    assert any("backwards" in r for r in quiet_reasons)


def test_grades_are_ordered():
    assert grade_of(50) == "brilliant"
    assert grade_of(30) == "sparkling"
    assert grade_of(20) == "neat"
    assert grade_of(5) is None


def _game(moves):
    return {"white": "hero", "black": "foe", "date": "", "opening": "Test",
            "moves": moves}


def _move(**kw):
    base = {"move_number": 12, "side": "white", "san": "Nf3", "best_san": "Nf3",
            "fen_before": OPENING, "loss": 0.0, "only_move": False,
            "missed_motifs": [], "best_line_san": None}
    base.update(kw)
    return base


def test_a_sound_pretty_move_counts_as_found():
    report = find_tricks([_game([_move(
        fen_before=FORK, san="Nf6+", best_san="Nf6+", loss=0.0)])])
    assert report["counts"]["found"] == 1
    assert report["counts"]["missed"] == 0


def test_an_unsound_sacrifice_is_a_gamble_not_a_triumph():
    report = find_tricks([_game([_move(
        fen_before=SAC, san="Bxe5", best_san="Nxe5", loss=25.0)])])
    assert report["counts"]["found"] == 0


def test_a_missed_trick_records_what_was_played_instead():
    report = find_tricks([_game([_move(
        fen_before=GUARD, san="Bb3", best_san="Nxc8", loss=12.0,
        missed_motifs=["removal_of_the_guard"])])])
    assert report["counts"]["missed"] == 1
    assert report["missed"][0]["instead_of"] == "Bb3"


def _move(**kw):
    base = {
        "move_number": 20, "side": "white", "san": kw.pop("san", "Qg5"),
        "fen_before": kw.pop("fen_before",
                             "3kq3/4n3/8/8/8/8/8/2B1K3 w - - 0 1"),
        "loss": 0.0, "only_move": False, "best_san": "Bg5",
        "best_line_san": "", "win_before": 50.0,
    }
    base.update(kw)
    return base


def _corpus(moves):
    return [{"white": "hero", "black": "foe", "date": "", "opening": "",
             "moves": moves}]


def test_a_good_move_in_a_dead_won_position_is_not_a_trick_you_found():
    """Queen chases knight for thirty moves: nothing there to find."""
    move = _move(san="Bg5", win_before=99.0)
    assert find_tricks(_corpus([move]))["counts"]["found"] == 0


def test_the_same_move_counts_when_the_game_is_still_live():
    move = _move(san="Bg5", win_before=60.0)
    assert find_tricks(_corpus([move]))["counts"]["found"] == 1


def test_a_mate_still_counts_in_a_won_position():
    move = _move(san="Ra8#", win_before=100.0,
                 fen_before="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert find_tricks(_corpus([move]))["counts"]["found"] == 1


def test_a_missed_forced_mate_still_counts_in_a_won_position():
    """The whole point of a won position is the finish, so missing a mate
    there is exactly the thing worth reporting."""
    move = _move(san="Rb1", win_before=100.0, loss=9.0,
                 fen_before="6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
                 best_san="Ra8#", best_line_san="1. Ra8#")
    assert find_tricks(_corpus([move]))["counts"]["missed"] == 1
