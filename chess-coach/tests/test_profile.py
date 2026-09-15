"""Tests for the part that makes claims about a person.

The interesting assertions here are the negative ones. A detector that fires
on data matching its own base rate is worse than no detector at all, because
its output is indistinguishable from insight.
"""
import json

from chess_coach.profile import build_profile


def make_move(**kwargs):
    move = {
        "ply": kwargs.get("ply", 10),
        "move_number": kwargs.get("move_number", 12),
        "side": kwargs.get("side", "white"),
        "san": kwargs.get("san", "Nf3"),
        "uci": "g1f3",
        "fen_before": kwargs.get(
            "fen_before", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        ),
        "win_before": kwargs.get("win_before", 50.0),
        "win_after": kwargs.get("win_after", 50.0),
        "loss": kwargs.get("loss", 0.0),
        "accuracy": kwargs.get("accuracy", 99.0),
        "severity": kwargs.get("severity"),
        "band_before": kwargs.get("band_before", "equal"),
        "phase": kwargs.get("phase", "middlegame"),
        "best_san": kwargs.get("best_san", "Nc3"),
        "best_line_san": "1. Nc3",
        "played_rank": None,
        "was_best": False,
        "refutation_san": kwargs.get("refutation_san"),
        "clock": kwargs.get("clock"),
        "time_spent": kwargs.get("time_spent"),
        "complexity": 5.0,
        "close_choices": 2,
        "only_move": False,
        "deep": kwargs.get("deep", True),
        "shape_tags": kwargs.get("shape_tags", ["piece:knight", "quiet_move"]),
        "best_shape_tags": kwargs.get("best_shape_tags", []),
        "allowed_shape_tags": kwargs.get("allowed_shape_tags", []),
        "missed_motifs": kwargs.get("missed_motifs", []),
        "allowed_motifs": kwargs.get("allowed_motifs", []),
        "missed_detail": [],
        "allowed_detail": kwargs.get("allowed_detail", []),
        "legal_mix": kwargs.get(
            "legal_mix",
            {"legal_moves": 30, "captures": 0.2, "checks": 0.05,
             "backward": 0.2, "quiet": 0.75},
        ),
        "legal_mix_after": kwargs.get(
            "legal_mix_after",
            {"legal_moves": 30, "captures": 0.2, "checks": 0.05,
             "backward": 0.2, "quiet": 0.75},
        ),
        "structure": None,
    }
    return move


def make_game(moves, result="loss", **kwargs):
    return {
        "id": "g1", "date": "2026.09.01", "time": "12:00",
        "white": "hero", "black": "foe", "white_elo": "1500", "black_elo": "1500",
        "hero": "hero", "hero_color": kwargs.get("hero_color", "white"),
        "hero_elo": "1500", "opponent_elo": "1500",
        "result": result, "termination": "", "eco": "C50",
        "opening": kwargs.get("opening", "Italian Game"),
        "time_control": "300+0", "base_time": 300, "increment": 0,
        "moves_analysed": len(moves), "mean_loss": 3.0, "accuracy": 85.0,
        "counts": {"blunder": 0, "mistake": 0, "inaccuracy": 0},
        "curve": kwargs.get("curve", [50.0] * (len(moves) + 1)),
        "moves": moves,
    }


def _finding(profile, key):
    return next((f for f in profile["findings"] if f["key"] == key), None)


def test_capture_reflex_is_silent_at_the_base_rate():
    """20% of mistakes are captures and 20% of legal moves were captures.

    There is nothing here but the base rate, and the report must say nothing.
    """
    moves = []
    for index in range(30):
        is_capture = index % 5 == 0
        moves.append(make_move(
            severity="mistake", loss=13.0, accuracy=50.0,
            shape_tags=["piece:knight"] + (["capture"] if is_capture else ["quiet_move"]),
        ))
    profile = build_profile([make_game(moves)])
    assert _finding(profile, "capture_reflex") is None


def test_capture_reflex_is_reported_when_well_above_the_base_rate():
    moves = []
    for index in range(30):
        is_capture = index % 10 != 0  # 90%, against a 20% base rate
        moves.append(make_move(
            severity="mistake", loss=13.0, accuracy=50.0,
            shape_tags=["piece:knight"] + (["capture"] if is_capture else ["quiet_move"]),
        ))
    profile = build_profile([make_game(moves)])
    found = _finding(profile, "capture_reflex")
    assert found is not None
    assert found["strength"] == "strong"
    assert found["numbers"]["lift"] > 4


def test_retreat_blindness_is_silent_when_retreats_were_simply_rare():
    """Backward moves are 20% of the missed best moves AND 20% of legal moves."""
    moves = []
    for index in range(30):
        backward = index % 5 == 0
        moves.append(make_move(
            severity="mistake", loss=13.0, accuracy=50.0,
            best_shape_tags=["piece:bishop"] + (["backward_move"] if backward else []),
        ))
    profile = build_profile([make_game(moves)])
    assert _finding(profile, "retreat_blindness") is None


def test_opponent_blindness_needs_more_than_a_majority():
    """Half the refutations being forcing is not a finding when half the
    opponent's legal moves were forcing."""
    moves = []
    for index in range(20):
        moves.append(make_move(
            severity="blunder", loss=25.0, accuracy=20.0,
            allowed_shape_tags=["capture"] if index % 2 == 0 else ["quiet_move"],
            legal_mix_after={"legal_moves": 30, "captures": 0.45, "checks": 0.05,
                             "backward": 0.2, "quiet": 0.5},
        ))
    profile = build_profile([make_game(moves)])
    assert _finding(profile, "opponent_blindness") is None


def test_opponent_blindness_is_reported_when_refutations_are_mostly_forcing():
    moves = []
    for index in range(20):
        moves.append(make_move(
            severity="blunder", loss=25.0, accuracy=20.0,
            allowed_shape_tags=["capture"] if index % 10 != 0 else ["quiet_move"],
        ))
    profile = build_profile([make_game(moves)])
    found = _finding(profile, "opponent_blindness")
    assert found is not None and found["numbers"]["lift"] > 2


def test_a_clean_player_gets_no_findings():
    games = [make_game([make_move() for _ in range(40)], result="win")
             for _ in range(12)]
    profile = build_profile(games)
    assert profile["findings"] == []


def test_missing_clock_data_is_called_out():
    profile = build_profile([make_game([make_move() for _ in range(40)])])
    assert any("clock" in c for c in profile["caveats"])


def test_small_samples_are_called_out():
    profile = build_profile([make_game([make_move() for _ in range(20)])])
    assert any("games analysed" in c for c in profile["caveats"])


def test_hanging_pieces_finding():
    moves = [
        make_move(severity="blunder", loss=30.0, accuracy=10.0,
                  allowed_motifs=["undefended_piece"])
        for _ in range(8)
    ]
    profile = build_profile([make_game(moves)])
    found = _finding(profile, "board_vision")
    assert found is not None and found["numbers"]["hits"] == 8


def test_key_moments_prefer_positions_that_were_still_playable():
    """Throwing away a playable game teaches more than move 40 of a lost one."""
    lost_already = make_move(
        severity="blunder", loss=40.0, win_before=8.0, san="Kh1", best_san="Kg1",
    )
    still_playable = make_move(
        severity="blunder", loss=25.0, win_before=55.0, san="Nd5", best_san="Nf5",
    )
    profile = build_profile([make_game([lost_already, still_playable])])
    assert profile["key_moments"][0]["played"] == "Nd5"


def test_conversion_spots_a_thrown_away_win():
    game = make_game(
        [make_move() for _ in range(10)], result="loss",
        curve=[50.0, 60.0, 88.0, 70.0, 20.0, 5.0] + [5.0] * 5,
    )
    profile = build_profile([game])
    assert profile["conversion"]["games_reaching_winning"] == 1
    assert profile["conversion"]["converted"] == 0
    assert profile["conversion"]["thrown_away"][0]["peak_win_pct"] == 88.0


def test_profile_survives_json_round_trip():
    """The corpus holds back-references to parent games while it works; if any
    survive into the output, json.dumps recurses forever."""
    games = [make_game([make_move() for _ in range(20)]) for _ in range(3)]
    profile = build_profile(games)
    assert json.loads(json.dumps(profile))["overview"]["games"] == 3


def test_profile_with_clock_data_survives_and_reports_time():
    """Regression: the cleanup of parent-game back-references used to run
    before the time analysis that needs them, so any corpus with clocks --
    which is to say every real one -- crashed."""
    moves = [
        make_move(time_spent=1.0 if i % 7 == 0 else 9.0, clock=250 - i * 5,
                  severity="blunder" if i % 7 == 0 else None,
                  loss=25.0 if i % 7 == 0 else 1.0,
                  accuracy=20.0 if i % 7 == 0 else 97.0)
        for i in range(40)
    ]
    profile = build_profile([make_game(list(moves)) for _ in range(10)])
    assert profile["time"]["by_time_spent"]
    assert profile["time"]["by_clock_left"]
    assert profile["overview"]["clock_data"] is True
    json.dumps(profile)


def test_shared_move_dicts_do_not_leak_back_references():
    """Two games built from the same move objects must still serialise."""
    shared = [make_move() for _ in range(10)]
    profile = build_profile([make_game(shared), make_game(shared)])
    json.dumps(profile)


def test_key_moments_do_not_repeat_a_position():
    moves = [make_move(severity="blunder", loss=25.0, win_before=55.0) for _ in range(6)]
    profile = build_profile([make_game(moves)])
    fens = [m["fen"] for m in profile["key_moments"]]
    assert len(fens) == len(set(fens)) == 1


def _pawn_game(n_pawn_best, n_piece_best):
    """Positions where the best move is a pawn push or a piece move."""
    # A position with a healthy mix of pawn and piece moves available.
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    moves = []
    for _ in range(n_pawn_best):
        moves.append(make_move(fen_before=fen, san="Nc3", best_san="d3",
                               severity="mistake", loss=13.0, accuracy=50.0))
    for _ in range(n_piece_best):
        moves.append(make_move(fen_before=fen, san="Nc3", best_san="Ng5",
                               severity="mistake", loss=13.0, accuracy=50.0))
    return make_game(moves)


def test_pawn_blindness_fires_when_pawn_moves_dominate_the_misses():
    profile = build_profile([_pawn_game(16, 2)])
    found = _finding(profile, "pawn_moves")
    assert found is not None
    assert found["numbers"]["lift"] > 1.3


def test_pawn_blindness_stays_quiet_at_the_base_rate():
    """Roughly a quarter of legal moves are pawn moves in that position, so a
    quarter of the misses being pawn moves is no finding at all."""
    profile = build_profile([_pawn_game(4, 14)])
    assert _finding(profile, "pawn_moves") is None


def test_pawn_blindness_needs_a_sample():
    profile = build_profile([_pawn_game(4, 1)])
    assert _finding(profile, "pawn_moves") is None


# A middlegame position with a black king castled kingside and a white queen
# that can swing to h5 among many other moves.
SWING = "r1bq1rk1/ppp2ppp/2n5/3np3/2B5/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 1"


def _swing_game(n_swing_best, n_other_best):
    moves = []
    for _ in range(n_swing_best):
        moves.append(make_move(fen_before=SWING, san="Nc3", best_san="Qd2",
                               severity="mistake", loss=13.0, accuracy=50.0))
    for _ in range(n_other_best):
        moves.append(make_move(fen_before=SWING, san="Nc3", best_san="a3",
                               severity="mistake", loss=13.0, accuracy=50.0))
    return make_game(moves)


def test_queen_swing_detector_needs_a_real_excess():
    """Two or three swings among many misses is the base rate, not a habit."""
    profile = build_profile([_swing_game(2, 16)])
    assert _finding(profile, "queen_swing") is None


def test_queen_swing_detector_needs_a_sample():
    profile = build_profile([_swing_game(6, 2)])
    # 8 mistakes is below the floor regardless of how lopsided it looks.
    assert _finding(profile, "queen_swing") is None


def test_queen_swing_is_not_the_same_as_any_queen_move():
    """The point is the square she arrives on, not that she moved.

    Qd2 eyes h6 and counts; Qe2 is just as much a queen move and reaches
    nothing near the king, so it must not.
    """
    import chess
    from chess_coach.profile import _is_queen_swing
    board = chess.Board(SWING)
    assert _is_queen_swing(board, board.parse_san("Qd2")) is True
    assert _is_queen_swing(board, board.parse_san("Qe2")) is False
    assert _is_queen_swing(board, board.parse_san("Nc3")) is False


def test_queen_swing_fires_on_a_real_excess():
    profile = build_profile([_swing_game(14, 4)])
    found = _finding(profile, "queen_swing")
    assert found is not None and found["numbers"]["lift"] > 1.5
