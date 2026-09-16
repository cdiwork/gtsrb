# Chess weakness profile: retsekoj

**14 games** (8W / 1D / 5L), 517 of your moves judged by Stockfish. Mean accuracy **91.08%**, **2.71** blunders and **6.0** serious errors per 100 moves.

> **Read this first.** No clock data in these games, so nothing about time management could be measured.

## What keeps costing you points

### 1. The capture gets played because it is a capture

*Acquisitiveness, a cousin of loss aversion: taking material is concrete and immediate, so it gets chosen over quiet moves without being compared to them.*

- **Evidence** (strong — consistent across the sample, n=31): 9 of 31 mistakes (29%) were captures, but only 8% of your legal moves were (3.5x).
- **How it was measured**: Share of your mistakes that were captures, against the share of your legal moves that were captures.
- **What to do**: When a capture looks obvious, force yourself to write down one non-capturing candidate and compare them. The habit is not 'stop taking' -- it is 'take on purpose'.

### 2. Pawn moves are where you go wrong

*Not a tactical gap. Pawn moves are irreversible and are usually about structure rather than a concrete threat, so they are judged rather than calculated -- and judgement is the slower thing to build. They are also what decides balanced positions, which is where the rest of your errors live.*

- **Evidence** (moderate — visible, worth testing, n=31): 9 of 31 missed best moves (29%) were pawn moves, against a 21% base rate (1.37x).
- **How it was measured**: Share of the best moves you missed that were pawn moves, against the share of pawn moves available in those same positions.
- **What to do**: In any position where nothing is forced, ask what your pawns should be doing before you ask what your pieces should be doing. Name the break you are playing for and the wing it is on -- that one question covers the pawn push you keep not making and the wrong-wing break in equal measure.

## Where the points go

### By phase of the game

| Phase | Moves | Accuracy | Errors/100 | Blunders/100 |
|---|---|---|---|---|
| opening | 123 | 91.86 | 4.07 | 1.63 |
| middlegame | 302 | 88.58 | 8.61 | 3.97 |
| endgame | 92 | 98.26 | 0.0 | 0.0 |

### By how the position stood

One thing to read carefully: accuracy usually *rises* in losing positions and that is an artefact, not a skill. Win probability is already near zero there, so there is very little left to throw away and every move scores well. Compare the `equal`, `better` and `winning` rows against each other; ignore `losing` except as a reminder of how often you get there.

| Position | Moves | Accuracy | Errors/100 |
|---|---|---|---|
| winning | 252 | 95.04 | 1.59 |
| better | 50 | 82.09 | 18.0 |
| equal | 161 | 87.86 | 10.56 |
| worse | 19 | 88.17 | 5.26 |
| losing | 35 | 91.83 | 0.0 |

### By colour

| Colour | Moves | Accuracy | Errors/100 |
|---|---|---|---|
| white | 268 | 91.73 | 5.22 |
| black | 249 | 90.38 | 6.83 |

## Tactics

Two different problems. *Missed* is the tactic in the move you failed to play; *allowed* is the tactic your opponent used to punish you. Work on whichever column is longer.

| Motif | You missed it | It was played on you |
|---|---|---|
| pin | 4 | 3 |
| fork | 3 | 3 |
| wins material | 4 | 1 |
| zwischenzug | 2 | 3 |
| absolute pin | 2 | 1 |
| undefended piece | 0 | 2 |
| discovered attack | 0 | 1 |
| mating attack | 1 | 0 |
| removal of the guard | 1 | 0 |
| skewer | 0 | 1 |

### Which pieces are involved

- You were moving: pawn x9, bishop x7, rook x7, knight x6, queen x2
- Punished by their: queen x7, rook x2
- Your pieces that got hit: bishop x5, rook x5, pawn x4, king x3, queen x2, knight x1

## Openings

`First error` is the median move number of your first serious mistake — roughly where your understanding of the position runs out.

| Opening | Games | Score | Accuracy | Loss/move | First error |
|---|---|---|---|---|---|
| unknown | 14 | 60.7% | 91.08 | 2.7 | 11 |

## Converting and saving

- You reached a winning position in 10 games and won 8 of them (80%).
- You were losing at some point in 5 games and salvaged 1 (20%).

Games you had won and did not win:

- unknown (????.??.??): reached 82.9% winning chances, ended in a loss.
- unknown (????.??.??): reached 92.0% winning chances, ended in a loss.

## What was going on before the game

14 of 14 games have context logged. These are the only numbers in this report written down *before* the first move, so unlike everything else they cannot have been coloured by how the game went.

`r` runs from -1 to +1. `Need` is roughly the size a correlation has to reach at this sample size before it means anything; below that, the honest reading is *nothing here yet*.

| Context | Metric | n | r | Need |  |
|---|---|---|---|---|---|
| hours since 6am | score | 14 | -0.73 | 0.53 | **notable** |
| games already today | score | 14 | -0.63 | 0.53 | **notable** |
| hours since 6am | accuracy | 14 | -0.47 | 0.53 |  |
| games already today | accuracy | 14 | -0.43 | 0.53 |  |
| hours since previous | trick hit rate | 12 | -0.41 | 0.57 |  |
| hours since 6am | mean loss | 14 | +0.40 | 0.53 |  |
| games already today | mean loss | 14 | +0.40 | 0.53 |  |
| games already today | blunders | 14 | +0.39 | 0.53 |  |
| hours since 6am | trick hit rate | 14 | -0.34 | 0.53 |  |
| hours since 6am | blunders | 14 | +0.24 | 0.53 |  |
| hours since previous | score | 12 | -0.17 | 0.57 |  |
| hours since previous | blunders | 12 | -0.17 | 0.57 |  |
| hours since previous | accuracy | 12 | -0.07 | 0.57 |  |
| games already today | trick hit rate | 14 | -0.05 | 0.53 |  |
| hours since previous | mean loss | 12 | +0.01 | 0.57 |  |

Context values that are words rather than numbers:

- after 8pm = **no** (n=8): score 0.9, accuracy 92.7, blunders 0.5, trick hit rate 76.1, mean loss 2.1
- after 8pm = **yes** (n=6): score 0.2, accuracy 87.5, blunders 1.7, trick hit rate 55.0, mean loss 3.9

> 15 correlations were run, so chance alone would be expected to decorate about 0.8 of them. Treat the largest one as a hypothesis to test on new games, not as a result.

> These fields are the only numbers in this report recorded *before* the game, so unlike everything else they cannot have been coloured by how it went. That is what makes them worth logging.

## The positions worth studying

Set these up on a board. Find the move before reading the answer.

**1. Move 44 as white** (unknown, blunder, -55.09 win%)

`4r3/8/3k2p1/2p1pr1p/p1PpR2P/P2N3K/1P2R1P1/8 w - - 9 44`

- You played g4; Kh2 was right — 44. Kh2 g5 45. hxg5 Rxg5 46. Kg1 Re7 47. Kf1 Re6
- Punished by: 44...Rf3+ 45. Kg2 Rxd3 46. Rf2 Rd1 47. gxh5

**2. Move 21 as black** (unknown, blunder, -51.98 win%)

`r1b2rk1/p1p2pp1/5q2/5N2/3RP1Q1/5P2/P1P3PP/5RK1 b - - 0 21`

- You played Ba6; Qxd4+ was right — 21...Qxd4+ 22. Nxd4 Bxg4 23. fxg4 Rfe8 24. Rf4 Rab8 25. a3
- Punished by: 22. Rfd1 Bc8 23. Qh4 Qxh4 24. Nxh4 Rb8

**3. Move 13 as black** (unknown, blunder, -43.68 win%)

`r1b2rk1/pppp1ppp/8/2b1pq2/4N1n1/3P1B2/PPP3PP/R1BQ1K1R b - - 2 13`

- You played Be3; Ne3+ was right — 13...Ne3+ 14. Bxe3 Bxe3 15. Qe2 Bb6 16. g4 Qe6 17. Re1
- Punished by: 14. h3 Nh2+ 15. Rxh2 Bb6 16. c4 d5

**4. Move 27 as white** (unknown, blunder, -40.24 win%)

`r4rk1/1b4b1/p2p1pQ1/5P2/2p5/7R/Pq4PP/4R2K w - - 4 27`

- You played Rg3; Qh7+ was right — 27. Qh7+ Kf7 28. Qg6+ Kg8 29. Qh7+
- Punished by: 27...Rf7 28. Rge3 c3 29. Qg3 Qb5 30. Rxc3

**5. Move 23 as white** (unknown, blunder, -39.74 win%)

`r2q1rk1/1b4b1/p2p1p2/5P1Q/2p5/7R/PP4PP/R5K1 w - - 0 23`

- You played Re1; Qh7+ was right — 23. Qh7+ Kf7 24. Qg6+ Kg8
- Punished by: 23...Re8 24. Re6 Bd5 25. Rxe8+ Qxe8 26. Qxe8+

**6. Move 33 as black** (unknown, blunder, -39.64 win%)

`6k1/2qb1rr1/p5RQ/3pPp2/3P4/2P5/P4P2/6RK b - - 4 33`

- You played Be6; Bb5 was right — 33...Bb5 34. e6 Re7 35. Qh5 Rxg6 36. Rxg6+ Rg7 37. Rh6
- Punished by: 34. Rxg7+ Rxg7 35. Rxg7+ Qxg7 36. Qxe6+ Kh8

**7. Move 21 as white** (unknown, blunder, -34.38 win%)

`r2q1r1k/1b3pb1/p2p4/5P2/2p3Q1/5R2/PP4PP/R5K1 w - - 0 21`

- You played Rh3+; f6 was right — 21. f6 Qxf6 22. Rxf6 Bxf6 23. Re1 Rae8 24. Rxe8 Rxe8
- Punished by: 21...Kg8 22. Rd1 Qf6 23. Rg3 Rfd8 24. h4

**8. Move 21 as white** (unknown, blunder, -31.41 win%)

`r4rk1/5pp1/p5np/1p1pPR2/3P2Qq/P1P5/6PP/R1B3K1 w - - 5 21`

- You played Qf3; Qd1 was right — 21. Qd1 Qe4 22. Rf2 Rae8 23. a4 Re6 24. Re2 Qf5
- Punished by: 21...Qe1+ 22. Qf1 Qxc3 23. Rb1 Qxd4+ 24. Rf2

**9. Move 18 as black** (unknown, blunder, -28.86 win%)

`r1bq1rk1/4b1pp/pn6/1p1pPp2/3P4/2N2N2/PPQ2P1P/R1B3RK b - - 2 18`

- You played Rf7; f4 was right — 18...f4 19. Ne1 Bf5 20. Qd1 f3 21. a3 Rc8 22. Nd3
- Punished by: 19. Bf4 Be6 20. Ne2 Rc8 21. Qd3 Nd7

**10. Move 19 as black** (unknown, blunder, -25.7 win%)

`r1bq2k1/4brpp/pn5B/1p1pPp2/3P4/2N2N2/PPQ2P1P/R5RK b - - 4 19`

- You played g6; f4 was right — 19...f4 20. Rae1 Bf5 21. Rxg7+ Kh8 22. Rxf7 Bxc2 23. e6
- Punished by: 20. Ne2 Ra7 21. Nf4 Bf8 22. Bxf8 Qxf8

---

Generated by chess-coach. Accuracy and win-probability figures use the Lichess logistic scale, so they are comparable to a Lichess game report. A move is an *inaccuracy* at 5 win% lost, a *mistake* at 11, a *blunder* at 20.