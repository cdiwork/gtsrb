"""Generate a synthetic game corpus for testing the pipeline.

Not a coaching tool -- a fixture generator. Two deliberately weakened
Stockfish instances play each other out of real opening lines, so the corpus
contains genuine engine-verified blunders across varied structures. Clock
comments are synthetic and exist only to exercise the time-analysis code
paths; no conclusion about time management should ever be drawn from them.
"""
from __future__ import annotations

import argparse
import random
import sys

import chess
import chess.engine
import chess.pgn

OPENINGS = [
    ("Italian Game", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6"),
    ("Sicilian Najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"),
    ("French Defence", "e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7"),
    ("Queen's Gambit Declined", "d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7"),
    ("King's Indian Defence", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O"),
    ("Caro-Kann", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7"),
    ("Ruy Lopez", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7"),
    ("London System", "d4 d5 Nf3 Nf6 Bf4 e6 e3 Bd6"),
    ("Scandinavian", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6"),
    ("English Opening", "c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5"),
    ("Scotch Game", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5"),
    ("Slav Defence", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4"),
    ("Pirc Defence", "e4 d6 d4 Nf6 Nc3 g6 Be3 Bg7"),
    ("Vienna Game", "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4"),
]


def play_game(path, hero, hero_skill, foe_skill, opening, hero_white, rng, base=300):
    name, line = opening
    board = chess.Board()
    game = chess.pgn.Game()
    for san in line.split():
        board.push_san(san)
    game.setup(chess.Board())
    node = game
    replay = chess.Board()
    for san in line.split():
        move = replay.parse_san(san)
        node = node.add_variation(move)
        replay.push(move)

    engines = {}
    for color, skill in ((chess.WHITE, hero_skill if hero_white else foe_skill),
                         (chess.BLACK, foe_skill if hero_white else hero_skill)):
        engine = chess.engine.SimpleEngine.popen_uci(path)
        engine.configure({"Skill Level": skill, "Threads": 1, "Hash": 32})
        engines[color] = engine

    clocks = {chess.WHITE: float(base), chess.BLACK: float(base)}
    try:
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < 80:
            engine = engines[board.turn]
            result = engine.play(board, chess.engine.Limit(time=0.03))
            if result.move is None:
                break
            spent = min(clocks[board.turn] - 1, max(0.4, rng.lognormvariate(1.1, 0.8)))
            clocks[board.turn] = max(1.0, clocks[board.turn] - spent)
            node = node.add_variation(result.move)
            node.set_clock(clocks[board.turn])
            board.push(result.move)
    finally:
        for engine in engines.values():
            engine.quit()

    white = hero if hero_white else "sparring_bot"
    black = "sparring_bot" if hero_white else hero
    game.headers.update({
        "Event": "Synthetic sparring (engine self-play fixture)",
        "Site": "synthetic",
        "Date": "2026.09.01",
        "White": white,
        "Black": black,
        "Result": board.result(claim_draw=True),
        "WhiteElo": "1400",
        "BlackElo": "1400",
        "TimeControl": f"{base}+0",
        "Opening": name,
        "ECO": "A00",
        "Termination": "synthetic",
    })
    game.headers["Result"] = board.result(claim_draw=True)
    return game


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="/usr/games/stockfish")
    parser.add_argument("--games", type=int, default=14)
    parser.add_argument("--hero", default="retsekoj_demo")
    parser.add_argument("--hero-skill", type=int, default=3)
    parser.add_argument("--foe-skill", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="demo/games.pgn")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.out, "w") as handle:
        for index in range(args.games):
            opening = OPENINGS[index % len(OPENINGS)]
            game = play_game(
                args.engine, args.hero, args.hero_skill, args.foe_skill,
                opening, index % 2 == 0, rng,
            )
            print(game, file=handle, end="\n\n")
            print(f"game {index + 1}/{args.games}: {opening[0]} -> "
                  f"{game.headers['Result']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
