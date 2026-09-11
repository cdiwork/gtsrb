# Chess weakness profile: retsekoj_demo

**14 games** (2W / 2D / 7L / 3 with no recorded result), 769 of your moves judged by Stockfish. Mean accuracy **89.75%**, **4.16** blunders and **7.93** serious errors per 100 moves.

Rating over this sample: 1400-1400.

> **Read this first.** Your openings barely repeat, so per-opening numbers rest on one or two games each.

## What keeps costing you points

### 1. The middlegame is where you lose points

*Not a bias but a skill distribution: the phases of the game draw on different abilities, and yours are uneven.*

- **Evidence** (strong — consistent across the sample, n=430): middlegame: 10.5 errors per 100 moves (430 moves) versus endgame: 3.8 (209 moves), 2.7x.
- **How it was measured**: Error rate per 100 moves by game phase.
- **What to do**: Middlegame work means candidate-move discipline and tactical pattern volume -- puzzles that are positions, not 'white to play and win'.

### 2. One mistake becomes three

*Two mechanisms the data cannot separate: tilt (the emotional cost of the error eats the attention the position now needs) and sunk cost (you keep pushing the idea that failed instead of re-assessing a changed position). Either way the first mistake is not what loses the game.*

- **Evidence** (strong — consistent across the sample, n=294): 3.7 errors per 100 moves before the first blunder, 11.9 after it (3.2x), across 10 games.
- **How it was measured**: Your error rate before your first blunder of a game against your error rate after it.
- **What to do**: Build a reset ritual: after any move you know was bad, take one deliberate pause and re-evaluate the position from scratch, as if you had just sat down at it. The previous plan is evidence, not an obligation.

### 3. You relax when you are winning

*Overconfidence plus reduced effort: once the position feels won, calculation quietly stops and moves get played on general impressions.*

- **Evidence** (strong — consistent across the sample, n=108): 15.7 errors per 100 moves when ahead versus 8.6 when equal (1.8x), over 108 winning-side moves.
- **How it was measured**: Your error rate in better/winning positions against equal ones.
- **What to do**: Treat reaching a winning position as a trigger to slow down, not speed up. Winning positions have one job: trade pieces and remove counterplay. Name your opponent's only source of activity before each move.

### 4. You calculate your plan, not their reply

*Egocentric calculation: attention goes to your own intention, so the opponent's most obvious answers never get examined. The best-documented single cause of amateur blunders.*

- **Evidence** (strong — consistent across the sample, n=61): 37 of 61 mistakes (61%) were punished by a check or capture, while only 17% of their legal replies were checks or captures (3.5x the base rate).
- **How it was measured**: Share of your mistakes refuted by a check or a capture -- the two move types that are cheapest to check -- against how often such replies were available at all.
- **What to do**: Adopt a fixed trigger: after choosing a move, and before playing it, list every check and every capture your opponent then has. Only then play. It costs ten seconds.

### 5. The capture gets played because it is a capture

*Acquisitiveness, a cousin of loss aversion: taking material is concrete and immediate, so it gets chosen over quiet moves without being compared to them.*

- **Evidence** (strong — consistent across the sample, n=61): 16 of 61 mistakes (26%) were captures, but only 7% of your legal moves were (3.6x).
- **How it was measured**: Share of your mistakes that were captures, against the share of your legal moves that were captures.
- **What to do**: When a capture looks obvious, force yourself to write down one non-capturing candidate and compare them. The habit is not 'stop taking' -- it is 'take on purpose'.

### 6. Pieces left undefended

*Not a cognitive bias -- a board-vision gap. The move was punished by a capture of something nothing was defending.*

- **Evidence** (weak — suggestive only, n=32): 8 of 32 blunders (25%) handed over an undefended piece.
- **How it was measured**: Share of your blunders whose refutation simply takes an undefended piece.
- **What to do**: Before every move, name every one of your pieces that nothing defends. Most players can do this in five seconds and it removes the majority of these.

## Where the points go

### By phase of the game

| Phase | Moves | Accuracy | Errors/100 | Blunders/100 |
|---|---|---|---|---|
| opening | 130 | 90.59 | 6.15 | 3.08 |
| middlegame | 430 | 86.91 | 10.47 | 5.12 |
| endgame | 209 | 95.06 | 3.83 | 2.87 |

### By how the position stood

One thing to read carefully: accuracy usually *rises* in losing positions and that is an artefact, not a skill. Win probability is already near zero there, so there is very little left to throw away and every move scores well. Compare the `equal`, `better` and `winning` rows against each other; ignore `losing` except as a reminder of how often you get there.

| Position | Moves | Accuracy | Errors/100 |
|---|---|---|---|
| winning | 55 | 86.9 | 10.91 |
| better | 53 | 81.05 | 20.75 |
| equal | 374 | 89.02 | 8.56 |
| worse | 87 | 87.23 | 9.2 |
| losing | 200 | 95.28 | 2.0 |

### By colour

| Colour | Moves | Accuracy | Errors/100 |
|---|---|---|---|
| white | 302 | 87.55 | 9.6 |
| black | 467 | 91.17 | 6.85 |

### By time spent on the move

| Thinking time | Moves | Accuracy | Blunders/100 |
|---|---|---|---|
| under 2s | 227 | 89.25 | 2.64 |
| 2-5s | 298 | 89.37 | 5.03 |
| 5-10s | 137 | 89.69 | 5.11 |
| 10-25s | 45 | 83.4 | 8.89 |
| over 25s | 3 | 96.63 | 0.0 |

### By clock remaining

| Clock left | Moves | Accuracy | Blunders/100 |
|---|---|---|---|
| over 50% left | 458 | 86.48 | 5.46 |
| 25-50% left | 144 | 91.1 | 4.86 |
| 10-25% left | 50 | 96.44 | 0.0 |
| under 10% left | 58 | 97.83 | 0.0 |

## Tactics

Two different problems. *Missed* is the tactic in the move you failed to play; *allowed* is the tactic your opponent used to punish you. Work on whichever column is longer.

| Motif | You missed it | It was played on you |
|---|---|---|
| undefended piece | 9 | 15 |
| fork | 6 | 9 |
| wins material | 7 | 5 |
| skewer | 2 | 7 |
| absolute pin | 3 | 3 |
| pin | 1 | 5 |
| zwischenzug | 2 | 3 |
| mating attack | 0 | 4 |
| discovered attack | 1 | 0 |
| trapped piece | 1 | 0 |

### Which pieces are involved

- You were moving: king x15, queen x12, pawn x11, bishop x9, rook x8, knight x6
- Punished by their: queen x13, rook x8, bishop x1, pawn x1, knight x1
- Your pieces that got hit: pawn x30, king x14, knight x7, bishop x7, queen x5, rook x5

## Openings

`First error` is the median move number of your first serious mistake — roughly where your understanding of the position runs out.

| Opening | Games | Score | Accuracy | Loss/move | First error |
|---|---|---|---|---|---|
| Italian Game | 1 | 0.0% | 80.38 | 7.08 | 9 |
| Slav Defence | 1 | 0.0% | 82.98 | 5.67 | 8 |
| King's Indian Defence | 1 | 0.0% | 84.39 | 4.98 | 29 |
| Pirc Defence | 1 | 0.0% | 85.53 | 4.61 | 9 |
| French Defence | 1 | 0.0% | 88.83 | 3.46 | 7 |
| Vienna Game | 1 | 0.0% | 89.39 | 3.39 | 5 |
| Queen's Gambit Declined | 1 | 0.0% | 89.41 | 3.1 | 23 |
| Scotch Game | 1 | 50.0% | 89.6 | 2.98 | 39 |
| Caro-Kann | 1 | 0.0% | 90.66 | 2.45 | 9 |
| Ruy Lopez | 1 | 50.0% | 94.28 | 2.12 | 9 |
| Scandinavian | 1 | 100.0% | 92.65 | 1.86 | 14 |
| Sicilian Najdorf | 1 | 0.0% | 94.55 | 1.67 | 46 |

## Converting and saving

- You reached a winning position in 5 games and won 2 of them (40%).
- You were losing at some point in 10 games and salvaged 3 (30%).

Games you had won and did not win:

- Italian Game (2026.09.01): reached 83.4% winning chances, ended in a loss.
- King's Indian Defence (2026.09.01): reached 86.0% winning chances, ended in a loss.
- Scotch Game (2026.09.01): reached 93.4% winning chances, ended in a draw.

## The positions worth studying

Set these up on a board. Find the move before reading the answer.

**1. Move 40 as white** (Italian Game, blunder, -68.47 win%)

`1r4k1/b1Q2pp1/2p3qp/R7/p2P4/2P4P/1P4P1/6BK w - - 1 40`

- You played Qxa7; Rxa7 was right — 40. Rxa7 Rxb2 41. Qc8+ Kh7 42. Qg4 Qxg4 43. hxg4 Rb3
- Punished by: 40...Rxb2 41. Bf2 Rxf2 42. Qb8+ Kh7 43. Qh2
- You spent 8.0s on it

**2. Move 33 as white** (King's Indian Defence, blunder, -50.69 win%)

`r1b2r2/1p4k1/2p4p/2P3pq/p1Q1R3/5N1P/PP4K1/5R2 w - - 0 33`

- You played Qd4+; Re7+ was right — 33. Re7+ Qf7 34. Qd4+ Kg8 35. Rxf7 Rxf7 36. Ne5 Rxf1
- Punished by: 33...Kg8 34. Qc4+ Qf7 35. Ne5 Qxc4 36. Rxf8+
- You spent 4.2s on it

**3. Move 28 as black** (Vienna Game, blunder, -49.58 win%)

`5rk1/1pQ2p2/1p4p1/1B3b1p/P2q4/2P5/6PP/5R1K b - - 0 28`

- You played Rc8; Qe4 was right — 28...Qe4 29. Rf3 Qe6 30. h3 Rc8 31. Qg3 Qd5 32. Re3
- Punished by: 29. Qxc8+ Bxc8 30. cxd4 Be6 31. Be2 Bd5
- You spent 1.8s on it

**4. Move 16 as black** (Slav Defence, blunder, -44.31 win%)

`4k2r/p1qn2pp/Q1pbp3/7b/3pN3/4B3/PP3PPP/2R1K2R b Kk - 1 16`

- You played Nc5; dxe3 was right — 16...dxe3 17. Nxd6+ Qxd6 18. O-O e2 19. Rfe1 O-O 20. Qxc6
- Punished by: 17. Nxc5 dxe3 18. O-O e2 19. Rfe1 Bxh2+
- You spent 11.3s on it

**5. Move 34 as white** (Italian Game, blunder, -42.39 win%)

`r4k2/b1Q1Rpp1/2p4p/p4q2/p2P4/2P4P/1P4P1/2B3K1 w - - 11 34`

- You played Be3; Qb7 was right — 34. Qb7 g5 35. Qxa8+ Kxe7 36. Qxa7+ Kf6 37. Qc7 Qb1
- Punished by: 34...Qb1+ 35. Kf2 Qxb2+ 36. Kf3 Qb3 37. Qd6
- You spent 3.2s on it

**6. Move 32 as white** (French Defence, blunder, -39.47 win%)

`2b2r2/5p1k/3Qp2p/p6P/q1p1P3/8/1PKR2P1/5B2 w - - 4 32`

- You played Kc3; Kb1 was right — 32. Kb1 Qb4 33. Qxb4 axb4 34. Bxc4 Bb7 35. e5 Rc8
- Punished by: 32...Qb3+ 33. Kd4 e5+ 34. Qxe5 Rd8+ 35. Qd5
- You spent 2.1s on it

**7. Move 53 as black** (Slav Defence, blunder, -37.66 win%)

`4R3/8/2p3Pk/3b3P/8/p4p2/5K2/8 b - - 6 53`

- You played Kxh5; a2 was right — 53...a2 54. Rh8+ Kg5 55. Ra8 Kxh5 56. g7 Kh6 57. g8=B
- Punished by: 54. g7 a2 55. Rh8+ Kg4 56. g8=Q+ Bxg8
- You spent 0.4s on it

**8. Move 23 as black** (Queen's Gambit Declined, blunder, -37.1 win%)

`3bk2r/1q3p1p/2b1p1p1/3p4/1p6/1N1BP3/PP1Q1PPP/2R3K1 b k - 5 23`

- You played Qb6; O-O was right — 23...O-O 24. Be2 Qb6 25. Qd4 Qxd4 26. Nxd4 Bd7 27. g3
- Punished by: 24. Rxc6 Qb7 25. Rc2 O-O 26. Nc5 Qa7
- You spent 2.3s on it

**9. Move 24 as white** (Pirc Defence, blunder, -33.19 win%)

`2r1r1k1/1p2Bpbp/1q3pp1/pB6/P3Q3/2Pp3P/1P5P/R3K2R w KQ - 1 24`

- You played Kf1; Bxe8 was right — 24. Bxe8 Rxe8 25. O-O-O Bf8 26. Qd4 Qxd4 27. cxd4 Rxe7
- Punished by: 24...f5 25. Qe1 Qc7 26. Bxe8 Rxe8 27. Qg3
- You spent 8.4s on it

**10. Move 23 as white** (Pirc Defence, blunder, -30.11 win%)

`2rr2k1/1p2Bpbp/1q3pp1/pB6/P3Q3/3p3P/1PP4P/R3K2R w KQ - 0 23`

- You played c3; Bxd8 was right — 23. Bxd8 Rxd8 24. O-O-O d2+ 25. Kb1 f5 26. Qf4 Rd4
- Punished by: 23...Rd5 24. Qxd5 Qe3+ 25. Kf1 Qe2+ 26. Kg1
- You spent 3.2s on it

---

Generated by chess-coach. Accuracy and win-probability figures use the Lichess logistic scale, so they are comparable to a Lichess game report. A move is an *inaccuracy* at 5 win% lost, a *mistake* at 11, a *blunder* at 20.