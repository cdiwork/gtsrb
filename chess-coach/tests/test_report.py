"""Tests for diagram rendering."""
import chess

from chess_coach.diagram import after, board_svg, legend_html

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TRAP = "r1bqk1nr/ppp1b1pp/2n1p3/4p3/2BP1B2/5N2/PP3PPP/RN1QK2R w KQkq - 0 8"


def test_board_renders_svg():
    svg = board_svg(START)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")


def test_arrows_use_distinct_colours_for_played_and_better():
    svg = board_svg(TRAP, played="Nxe5", better="Bxe5")
    from chess_coach.diagram import BETTER, PLAYED
    assert PLAYED in svg and BETTER in svg


def test_san_and_uci_are_both_accepted():
    assert board_svg(TRAP, played="Nxe5") == board_svg(TRAP, played="f3e5")


def test_an_illegal_move_is_skipped_rather_than_raising():
    """Bad SAN in a report must not take the whole page down."""
    svg = board_svg(TRAP, played="Qxh7", better="nonsense")
    assert svg.startswith("<svg")


def test_marked_squares_are_filled():
    plain = board_svg(TRAP)
    marked = board_svg(TRAP, mark=["d4"])
    assert len(marked) > len(plain)


def test_after_plays_the_moves_out():
    fen = after(TRAP, ["Nxe5", "Qxd4"])
    board = chess.Board(fen)
    assert board.piece_at(chess.D4) == chess.Piece(chess.QUEEN, chess.BLACK)
    assert board.piece_at(chess.E5) == chess.Piece(chess.KNIGHT, chess.WHITE)


def test_after_stops_at_the_first_illegal_move():
    assert after(TRAP, ["Nxe5", "Qxh8"]) == after(TRAP, ["Nxe5"])


def test_orientation_follows_the_player():
    white = board_svg(TRAP, orientation=True)
    black = board_svg(TRAP, orientation=False)
    assert white != black


def test_legend_names_every_arrow_colour():
    html = legend_html()
    for label in ("what you played", "what was better"):
        assert label in html
