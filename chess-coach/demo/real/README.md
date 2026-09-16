# Real games — 14 blitz games against Maia

Unlike `../games.pgn`, which is engine self-play, these are one human's actual
5-minute games (handle `retsekoj`) against Maia 1100–1500 over four days in
September 2026, plus the context file that turned them into the only finding
in this repository worth acting on.

## Why this directory exists

It is the worked example for `profile --context`. Everything else the package
measures is derived from the game itself and is therefore contaminated by how
the game went: you cannot tell whether low accuracy caused the loss or the
lost position caused the low accuracy. Context fields are written down before
the first move, so they are the one set of numbers immune to that.

```bash
.venv/bin/chess-coach analyse demo/real/games.pgn --user retsekoj \
    -o demo/real/analysis.json
.venv/bin/chess-coach profile demo/real/analysis.json \
    --context demo/real/context.csv -o demo/real/profile.json
.venv/bin/chess-coach report demo/real/profile.json \
    --html demo/real/report.html --md demo/real/report.md
.venv/bin/chess-coach tricks demo/real/analysis.json --html demo/real/tricks.html
```

Analysing the 14 games took about 6½ minutes on three threads.

## What it found

```
after_8pm = no  (n=8):  score 0.9   accuracy 92.7   blunders 0.5   tricks 76%
after_8pm = yes (n=6):  score 0.2   accuracy 87.5   blunders 1.7   tricks 55%
```

Every game played between 10:25 and 18:07 was a win or a draw. Five of the six
games played after 20:45 were losses. Fisher exact on that split is p = 0.003,
and it survives restricting to a single opponent strength (Maia 1400: 5½/6 by
day, 1/3 at night) and to a single colour.

Neither of the two behavioural findings the profiler reports — a capture
reflex and pawn-move blindness — is anywhere near that strong. A player
looking for the single highest-value change here should change *when they
play*, not *how*.

## How this context file was made, and why that is a caveat

The honest version: it was reconstructed, not recorded. The player sent each
game to an assistant immediately after finishing it, so the delivery timestamp
of each message stands in for the game's end time. The columns were derived
from those timestamps in Europe/London:

| column | meaning |
|---|---|
| `hours_since_6am` | hours into a day that begins at 06:00 local, so a session running past midnight stays on one day |
| `hours_since_previous` | gap since the previous game, blank if over 12 hours |
| `games_already_today` | games already played in that session |
| `after_8pm` | the day/night split, a word-valued column so it gets grouped rather than correlated |

Three consequences worth stating plainly:

- **It only counts games that were sent.** If the player played others and did
  not submit them, `games_already_today` understates the true count.
- **The 06:00 day boundary is a choice.** It happens not to matter here — no
  game falls between 03:06 and 10:25 — but on a different corpus it would.
- **`hours_since_6am` and `games_already_today` are collinear.** Later in the
  day means more games already played. They are one finding reported twice,
  not two independent ones.

The template `chess-coach context-template` writes asks for hours slept, time
of day, games already played and a 1–5 sharpness guess instead. Those are
better, because "late" and "tired" are separable only if you record both — and
this corpus contains a direct counterexample to their being the same thing
(see game 13 below).

## The two games worth opening

**Game 13** (`cb525715-441`, 03:06) is the best game in the set: 96.0%
accuracy, no blunders, no mistakes, a sound knight sacrifice on move 10 — and
it was played at the latest hour recorded, 27 minutes after a loss. Every one
of its five inaccuracies came from a position already at 90–100%, which is
what separates it from the loss that preceded it: the same error *count*, but
in a position where being imprecise costs nothing. It is the reason this
directory does not claim a curfew.

**Game 14** (`7ca20af2-23b`, 03:32) is the best illustration of what the
beauty scorer is for. `11...Nf2` scores 62 — the only *brilliant* in the whole
corpus — and it was missed; the move played instead, and the blunder two moves
later, were both attempts at the same fork by preparation rather than by
force. The game was lost on time from a +88% position.

## On the numbers moving

Two scoring bugs were fixed after these games were first analysed, both of
which had inflated the trick rate in long endgames: a pin was credited even
when the move did not create it, and ordinary good moves counted as "tricks"
in positions already won. The figures here are post-fix. Earlier numbers
quoted in conversation for game 13 (89% rather than 78%) were wrong.
