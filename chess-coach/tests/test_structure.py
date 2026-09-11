import chess

from chess_coach.structure import (
    centre_state, describe, king_safety, outposts, pawn_report, phase_of,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"


def test_the_opening_position_is_not_an_open_centre():
    assert centre_state(chess.Board(START)) == "fluid"


def test_a_kings_indian_wall_is_closed():
    board = chess.Board(
        "r1bq1rk1/pp2ppbp/2np1np1/2pPp3/2P1P3/2N2N1P/PP2BPP1/R1BQ1RK1 w - - 0 9"
    )
    assert centre_state(board) == "closed"


def test_phase_follows_material():
    assert phase_of(chess.Board(START)) == "opening"
    assert phase_of(chess.Board("8/5ppp/8/3k4/3P4/8/5PPP/6K1 w - - 0 40")) == "endgame"


def test_passed_and_isolated_pawns():
    report = pawn_report(chess.Board("8/8/8/3P4/8/8/8/4K2k w - - 0 1"), chess.WHITE)
    assert report["passed"] == ["d5"]
    assert report["isolated"] == ["d5"]
    assert report["islands"] == 1


def test_king_safety_counts_the_shield():
    safety = king_safety(chess.Board("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"), chess.WHITE)
    assert safety["square"] == "g1"
    assert safety["side"] == "kingside"
    assert safety["pawn_shield"] == 3


def test_outpost_needs_pawn_support_and_no_pawn_challenge():
    # Knight on d5 supported by the e4 pawn, with no black c- or e-pawn.
    board = chess.Board("4k3/pp6/8/3N4/4P3/8/8/4K3 w - - 0 1")
    assert outposts(board, chess.WHITE) == ["d5"]
    # Give black an e-pawn that can challenge it and it is no longer an outpost.
    challenged = chess.Board("4k3/pp2p3/8/3N4/4P3/8/8/4K3 w - - 0 1")
    assert outposts(challenged, chess.WHITE) == []


def test_describe_covers_both_sides():
    data = describe(chess.Board(START))
    assert data["phase"] == "opening"
    assert data["material_balance_cp"] == 0
    assert data["white"]["mobility"] == data["black"]["mobility"] == 20
