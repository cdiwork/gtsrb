import chess
import pytest

from chess_coach.see import best_capture, loose_pieces, see


@pytest.mark.parametrize("fen,uci,expected", [
    # A pawn nobody is defending is simply free.
    ("4k3/8/8/4p3/8/5N2/8/4K3 w - - 0 1", "f3e5", 100),
    # The same pawn, defended: knight for pawn loses 220.
    ("4k3/8/3p4/4p3/8/5N2/8/4K3 w - - 0 1", "f3e5", -220),
    # A second attacker does not help when the cheapest recapture still wins.
    ("4k3/8/3p4/4p3/8/5N2/8/4K1R1 w - - 0 1", "f3e5", -220),
    # Rook takes a rook with nothing to recapture.
    ("4k3/8/8/4r3/8/4R3/4R3/4K3 w - - 0 1", "e3e5", 500),
])
def test_see_values(fen, uci, expected):
    assert see(chess.Board(fen), chess.Move.from_uci(uci)) == expected


def test_see_sees_through_xrays():
    """Two rooks behind each other beat one defender."""
    board = chess.Board("3rk3/8/8/8/8/3R4/3R4/4K3 w - - 0 1")
    # Rxd8 Rxd8 Rxd8 wins a rook.
    assert see(board, chess.Move.from_uci("d3d8")) == 500


def test_loose_pieces_reports_nothing_in_a_sound_position():
    board = chess.Board()
    assert loose_pieces(board, chess.WHITE) == []
    assert loose_pieces(board, chess.BLACK) == []


def test_loose_pieces_finds_the_free_piece():
    # Rook on e1, black knight on e4, nothing in between.
    board = chess.Board("4k3/8/8/8/4n3/8/8/4R1K1 w - - 0 1")
    found = loose_pieces(board, chess.BLACK)
    assert [chess.square_name(sq) for sq, _ in found] == ["e4"]


def test_best_capture_prefers_the_profitable_one():
    board = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 0 1")
    move, gain = best_capture(board)
    assert gain == 100 and move is not None
