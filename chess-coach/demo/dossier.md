# Coaching dossier

You are looking at the most expensive decisions from a batch of one player's
games, already measured by Stockfish. Your job is the part the engine cannot
do. For each position below, write:

1. **What the position is about** -- the structure, who has what, which side
   of the board each player should be playing on, and what the correct plan
   was. Two or three sentences of real chess content.
2. **Why the played move is wrong in terms of that plan** -- not "it loses a
   knight" (the evaluation already says so) but which feature of the position
   it misunderstood.
3. **What the player probably saw and did not see.** Be concrete and use the
   evidence given: the clock time, whether the refutation was a check or
   capture, whether the move they missed was quiet or backward.

Then, across all the positions, name the recurring habit in one paragraph and
give one drill that would actually address it.

Rules: do not paraphrase engine lines as if they were insight. Do not give
advice that would be true of any position ("develop your pieces", "control
the centre") unless it is specifically the point here. If the positions do
not share a pattern, say so -- a profile of one honest observation is worth
more than five invented ones.


## The player: retsekoj_demo

14 games, 769 moves judged, mean accuracy 89.75, 4.16 blunders per 100 moves, rating range [1400, 1400].

### Already measured (do not re-derive these, build on them)

- **The middlegame is where you lose points** [strong, n=430]: middlegame: 10.5 errors per 100 moves (430 moves) versus endgame: 3.8 (209 moves), 2.7x.
- **One mistake becomes three** [strong, n=294]: 3.7 errors per 100 moves before the first blunder, 11.9 after it (3.2x), across 10 games.
- **You relax when you are winning** [strong, n=108]: 15.7 errors per 100 moves when ahead versus 8.6 when equal (1.8x), over 108 winning-side moves.
- **You calculate your plan, not their reply** [strong, n=61]: 37 of 61 mistakes (61%) were punished by a check or capture, while only 17% of their legal replies were checks or captures (3.5x the base rate).
- **The capture gets played because it is a capture** [strong, n=61]: 16 of 61 mistakes (26%) were captures, but only 7% of your legal moves were (3.6x).
- **Pieces left undefended** [weak, n=32]: 8 of 32 blunders (25%) handed over an undefended piece.

Tactics missed: {'undefended_piece': 9, 'wins_material': 7, 'fork': 6, 'absolute_pin': 3, 'zwischenzug': 2, 'skewer': 2, 'pin': 1, 'trapped_piece': 1, 'discovered_attack': 1}. Tactics allowed: {'undefended_piece': 15, 'fork': 9, 'skewer': 7, 'wins_material': 5, 'pin': 5, 'mating_attack': 4, 'zwischenzug': 3, 'absolute_pin': 3}.

## Positions

### 1. Move 40 as white, Italian Game (blunder, -68.47 win%)

```
FEN: 1r4k1/b1Q2pp1/2p3qp/R7/p2P4/2P4P/1P4P1/6BK w - - 1 40
```

- Played: **Qxa7** (win probability 80.09% -> 11.62%)
- Engine wanted: **Rxa7** — 40. Rxa7 Rxb2 41. Qc8+ Kh7 42. Qg4 Qxg4 43. hxg4 Rb3
- What punished it: 40...Rxb2 41. Bf2 Rxf2 42. Qb8+ Kh7 43. Qh2
- Tactic missed: undefended_piece
- Tactic allowed: undefended_piece
- The move played was: piece:queen, capture, motif:undefended_piece
- Time spent on it: 8.0s
- Position sharpness: 30.1 win% between best and third-best move
- Phase / centre: middlegame / semi-open centre
- Material balance: +0.0 (positive = white)
- Open files: e
- You: king h1 (kingside, shield 2, 1 squares around it attacked); mobility 36; rooks on useful files a5
- Opponent: king g8 (kingside, shield 3, 1 squares around it attacked); mobility 36; isolated a4, c6; rooks on useful files b8

### 2. Move 33 as white, King's Indian Defence (blunder, -50.69 win%)

```
FEN: r1b2r2/1p4k1/2p4p/2P3pq/p1Q1R3/5N1P/PP4K1/5R2 w - - 0 33
```

- Played: **Qd4+** (win probability 87.48% -> 36.79%)
- Engine wanted: **Re7+** — 33. Re7+ Qf7 34. Qd4+ Kg8 35. Rxf7 Rxf7 36. Ne5 Rxf1
- What punished it: 33...Kg8 34. Qc4+ Qf7 35. Ne5 Qxc4 36. Rxf8+
- The move played was: piece:queen, check
- Time spent on it: 4.2s
- There was only one good move here
- Phase / centre: middlegame / open centre
- Material balance: -1.1 (positive = white)
- Open files: d, e, f
- You: king g2 (kingside, shield 1, 2 squares around it attacked); mobility 51; isolated h3; rooks on useful files f1, e4
- Opponent: king g7 (kingside, shield 2, 2 squares around it attacked); mobility 33; rooks on useful files f8
