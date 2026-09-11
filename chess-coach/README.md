# chess-coach

Play your games against people, then find out *how* you lose rather than just
that you lost. This downloads your games from Chess.com or Lichess, has
Stockfish judge every move you played, names the tactic in each mistake, and
aggregates the result into a profile of your recurring habits — including the
cognitive ones.

It deliberately splits the work in two:

- **Measurement is code.** Evaluations, tactical motifs, clock usage,
  structural features, base rates. Reproducible, testable, no opinions.
- **Explanation is a language model's job.** What the position was about, what
  the plan should have been, what you probably saw and did not. The
  `dossier` command packages the measurements for exactly that.

Stockfish can prove your move was wrong. It cannot tell you why you played it.

## Install

```bash
./setup.sh                     # virtualenv + dependencies + checks for Stockfish
```

Stockfish itself:

| Platform | Command |
|---|---|
| Debian/Ubuntu | `sudo apt install stockfish` |
| macOS | `brew install stockfish` |
| Anything | [stockfishchess.org/download](https://stockfishchess.org/download/) |

If it lives somewhere unusual: `export CHESS_COACH_STOCKFISH=/path/to/stockfish`.

## Use it

The short version — fetch, analyse, profile and report in one command:

```bash
./.venv/bin/chess-coach coach --site chess.com --user YOUR_HANDLE --max 20
```

That writes `coach-out/report.html` (open it in a browser) plus the JSON behind
it. The four steps are also separate, which is what you want once you are
iterating:

```bash
chess-coach fetch   --site chess.com --user YOUR_HANDLE --max 30 -o games.pgn
chess-coach analyse games.pgn --user YOUR_HANDLE -o analysis.json
chess-coach profile analysis.json -o profile.json
chess-coach report  profile.json --html report.html --md report.md
```

Already have a PGN (any site's export, or a single game you want to look at)?
Skip the fetch and point `analyse` at the file. `--user` has to match the name
in the PGN headers exactly enough to identify which side was you —
case-insensitively.

**Playing on both sites?** Fetch each, concatenate, and pass both handles —
`--user` is repeatable on `analyse` and `review`, so a mixed corpus works even
when your names differ:

```bash
chess-coach fetch --site chess.com --user retsekoj   -o cc.pgn
chess-coach fetch --site lichess   --user other_name -o li.pgn
cat cc.pgn li.pgn > games.pgn
chess-coach analyse games.pgn --user retsekoj --user other_name -o analysis.json
```

**Just want one game looked at?** That is the common case after a session that
annoyed you:

```bash
chess-coach review game.pgn --user YOUR_HANDLE
```

It prints only the moves that cost something — with the engine's preference,
the tactic involved and the time you spent — then the three positions worth
setting up on a board.

Useful flags: `--time-class blitz` (repeatable; bullet is excluded by default),
`--include-unrated`, `--limit N`, and on Lichess `--token` if you want more than
the anonymous rate limit allows.

### Then ask for the plans

```bash
chess-coach dossier profile.json -o dossier.md
```

`dossier.md` contains your worst decisions with the position, the engine's
lines, the structural facts, the tactic, your clock time — and a brief telling a
strong reader what to produce. Hand it to Claude, or to a coach. That is where
"what were the important plans in this position" gets answered, and it is the
part of this whole pipeline that actually teaches you something.

A reasonable weekly loop: play, `coach`, read the findings, `dossier`, work
through the positions, then re-run in a month and see whether the numbers moved.

## What it measures

**Win probability, not centipawns.** +9.0 to +7.0 is irrelevant; 0.0 to −0.7
decides games. Every cost in the report is in win percentage on the Lichess
logistic scale, so a move is an *inaccuracy* at 5 points lost, a *mistake* at
11, a *blunder* at 20 — and the accuracy figures are comparable to a Lichess
game report.

**Two passes.** A cheap sweep (depth 14) over every position finds the
candidates; a deep multi-PV look (depth 20) re-judges only those before
anything goes on your record. Judging a whole game deep wastes minutes on moves
nobody would question; judging it shallow accuses you of blunders that are not
there.

**Two different tactical failures**, which need different homework:

- *missed* — the tactic in the move you failed to find
- *allowed* — the tactic your opponent used to punish you

Detected motifs: forks, absolute/relative pins, skewers, discovered attacks,
undefended pieces, trapped pieces, removal of the guard, zwischenzugs, back-rank
mates, mating attacks.

**Context for every mistake**: phase, whether you stood better or worse, the
clock, how sharp the position was (gap between best and third-best), the pawn
structure, king safety, open files, outposts, mobility.

## How the habits are inferred

This is the part that is easy to fake, so two rules are enforced in code:

1. **Every claim carries its sample size**, and thin samples are labelled
   `weak` or `anecdote` rather than rounded up into confidence.
2. **Every "you are blind to X" claim is compared against how often X was
   available.** If 18% of your legal moves were backward and 17% of the moves
   you missed were backward, there is no finding — that is the base rate. The
   analysis records the legal-move composition of every position it judges
   precisely so this comparison is possible.

What gets looked for:

| Finding | The measurement behind it |
|---|---|
| You calculate your plan, not their reply | share of mistakes refuted by a check or capture, against how many such replies existed |
| The capture gets played because it is a capture | share of your mistakes that were captures, against the share of legal moves that were captures |
| Your candidate list is all forcing moves | share of missed best moves that were quiet, against the quiet-move base rate |
| Backward moves don't occur to you | same comparison for backward moves |
| You relax when winning | error rate when ahead against when equal |
| One mistake becomes three | error rate after your first blunder of a game against before it |
| Your mistakes are the moves you didn't stop for | error rate on moves under half your median think time |
| The clock is beating you | error rate in the last 15% of your clock |
| You ride a plan past the point it stopped working | share of total loss occurring in runs of 3+ consecutive costly moves |
| The punishment arrives at your king | share of mistakes refuted by a mating attack |
| Pieces left undefended | share of blunders whose refutation just takes something loose |
| A phase or an opening that costs you | error rate by phase; loss per move by opening, with the median move number of your first mistake |

Note what that table is: a set of **measurable signals with bias names attached
as interpretation**. "You calculate your plan, not their reply" is a real,
well-documented pattern and the measurement is a fair proxy for it — but it is a
proxy. Where two mechanisms produce the same data (tilt and sunk-cost both look
like "errors cluster after the first one"), the report says so instead of
picking the more flattering story. An empty findings list is a legitimate,
common result and the report says that too.

## Speed

Roughly **1–2 minutes per game** at default depths on four cores. Twenty games
is a coffee break, and that is about the right sample for the habit detectors.
To go faster:

```bash
chess-coach analyse games.pgn --user you --fast-depth 12 --deep-depth 16
# or use time instead of depth:
chess-coach analyse games.pgn --user you --movetime 0.15 --deep-movetime 0.8
```

Findings need volume: under 10 games the report will tell you not to trust it.

## Output

| File | What is in it |
|---|---|
| `games.pgn` | the downloaded games |
| `analysis.json` | every move you played, judged, with motifs and context |
| `profile.json` | the aggregates, the findings, the key positions |
| `report.html` | the readable report — theme-aware, works on a phone |
| `report.md` | the same thing as Markdown |
| `dossier.md` | key positions packaged for a coach or an LLM |

## Limitations

- Engine evaluation is not human evaluation. A "blunder" at depth 20 may have
  been unfindable over the board; that is why the positions that were *still
  playable* are ranked above ones where the game was already lost.
- Bullet games are excluded by default and should stay that way — the mistakes
  are mouse speed, not thinking.
- Opening names come from the site's own tags, so per-opening rows are only as
  good as those tags.
- Clock-based findings need clock data in the PGN. Both sites provide it; some
  exports strip it, and the report will say so.
- Motif detection is heuristic and errs towards silence. It will miss exotic
  tactics; it tries hard not to invent them.

## Development

```bash
.venv/bin/python -m pytest tests -q        # 59 tests, ~4s
```

The tests worth reading first are in `tests/test_profile.py`: most of them
assert that a detector stays **silent** on data that only matches its own base
rate. `tools/simulate.py` generates a synthetic corpus from weakened engine
self-play for exercising the pipeline end to end.

| Module | Job |
|---|---|
| `evalscale.py` | centipawns → win probability, accuracy, thresholds |
| `see.py` | static exchange evaluation (is taking there actually good?) |
| `motifs.py` | naming the tactic; move shape |
| `structure.py` | pawn structure, king safety, files, outposts, phase |
| `engine.py` | Stockfish process and two-pass limits |
| `analyse.py` | one game → judged moves with context |
| `profile.py` | many games → aggregates, base rates, findings |
| `report.py` | Markdown and HTML rendering |
| `dossier.py` | key positions packaged for explanation |
