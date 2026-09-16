import chess
import pytest

from chess_coach.motifs import classify_line, shape_of


@pytest.mark.parametrize("fen,san,motif", [
    ("4q1k1/8/8/3N4/8/8/8/7K w - - 0 1", "Nf6+", "fork"),
    ("3qk3/4n3/8/8/8/8/8/2B1K3 w - - 0 1", "Bg5", "pin"),
    ("3kq3/4n3/8/8/8/8/8/2B1K3 w - - 0 1", "Bg5", "absolute_pin"),
    ("4k3/8/8/4p3/8/5N2/8/4K3 w - - 0 1", "Nxe5", "undefended_piece"),
    ("4q3/4k3/8/8/8/8/8/R5K1 w - - 0 1", "Re1+", "skewer"),
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Ra8#", "back_rank_mate"),
    ("4q1k1/8/8/8/4N3/8/8/4R1K1 w - - 0 1", "Nd6", "discovered_attack"),
    ("4k3/8/8/8/8/8/8/5K1n w - - 0 1", "Kg2", "trapped_piece"),
])
def test_motif_is_detected(fen, san, motif):
    board = chess.Board(fen)
    result = classify_line(board, [board.parse_san(san)])
    assert motif in result["motifs"]


def test_the_ruy_lopez_bishop_is_not_a_pin():
    """Bb5 famously looks like a pin and is not one: d7 blocks the line to e8.

    A detector that reports this would produce false motifs in nearly every
    Spanish game, so it is worth pinning down as a test.
    """
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 0 1"
    )
    result = classify_line(board, [board.parse_san("Bb5")])
    assert "pin" not in result["motifs"]
    assert "absolute_pin" not in result["motifs"]


def test_a_piece_frozen_by_a_pin_is_not_also_called_trapped():
    board = chess.Board("3kq3/4n3/8/8/8/8/8/2B1K3 w - - 0 1")
    result = classify_line(board, [board.parse_san("Bg5")])
    assert "trapped_piece" not in result["motifs"]


def test_no_trapped_claim_while_the_opponent_is_in_check():
    """With the king in check every reply is forced, so "no safe square" is
    not evidence about any other piece."""
    board = chess.Board("4q1k1/8/8/3N4/8/8/8/7K w - - 0 1")
    result = classify_line(board, [board.parse_san("Nf6+")])
    assert "trapped_piece" not in result["motifs"]


def test_shape_reads_direction_from_the_movers_side():
    board = chess.Board("4k3/8/8/4r3/8/8/8/6K1 b - - 0 1")
    shape = shape_of(board, board.parse_san("Re7"))
    # Black retreating up the board towards its own side is moving backwards.
    assert shape.direction == "backward"
    assert shape.is_quiet and not shape.is_capture


def test_shape_flags_a_sacrifice():
    board = chess.Board("4k3/8/3p4/4p3/8/5N2/8/4K3 w - - 0 1")
    shape = shape_of(board, board.parse_san("Nxe5"))
    assert shape.is_capture and shape.is_sacrifice and shape.see < 0


def test_a_pin_that_was_already_standing_is_not_credited_again():
    """A queen shuffling around a lone knight must not score a fresh pin
    on every move of a long endgame."""
    board = chess.Board("3kq3/4n3/8/8/7B/8/8/4K3 w - - 0 1")
    # The bishop on h4 already pins e7 along h4-d8; Bg5 only steps closer.
    result = classify_line(board, [board.parse_san("Bg5")])
    assert "absolute_pin" not in result["motifs"]
