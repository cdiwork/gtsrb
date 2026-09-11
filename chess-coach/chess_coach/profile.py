"""From a pile of judged moves to a profile of how you think.

The hard part of this file is not the arithmetic, it is resisting the
temptation to be interesting. "You suffer from loss aversion" is a
satisfying sentence and a worthless one unless the data distinguishes it
from chance. So two rules are enforced throughout:

  1. Every claim carries its sample size, and small samples are labelled
     rather than rounded into confidence.
  2. Anything of the form "you are blind to X moves" is compared against how
     often X moves were *available*. If 18% of your legal moves were
     backward and 17% of the ones you missed were backward, there is no
     finding -- just the base rate.

What survives those two rules is usually less exciting and actually true.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, List, Optional, Tuple

from . import evalscale as ev

SEVERE = ("blunder", "mistake")


@dataclass
class Finding:
    """One claim about a habit, with the evidence that earned it."""

    key: str
    title: str
    # What the bias actually is, in plain language.
    definition: str
    # The measurement standing in for it -- stated so you can disagree.
    signal: str
    evidence: str
    n: int
    strength: str
    drill: str
    numbers: Dict = field(default_factory=dict)


def _mean(values: Iterable[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def _rate(count: int, moves: int, per: int = 100) -> Optional[float]:
    return round(per * count / moves, 2) if moves else None


def _share(part: int, whole: int) -> Optional[float]:
    return round(part / whole, 3) if whole else None


def _strength(n: int, lift: Optional[float], gap: Optional[float] = None) -> str:
    """Grade a finding by sample size and effect size, conservatively."""
    if n < 5:
        return "anecdote"
    big = (lift is not None and lift >= 1.6) or (gap is not None and gap >= 0.18)
    medium = (lift is not None and lift >= 1.3) or (gap is not None and gap >= 0.10)
    if n >= 12 and big:
        return "strong"
    if n >= 6 and (big or medium):
        return "moderate"
    return "weak"


# --------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------


class Corpus:
    """All the judged moves, with the handful of views the detectors need."""

    def __init__(self, games: List[Dict]):
        self.games = games
        self.moves: List[Dict] = []
        self.by_game: List[List[Dict]] = []
        for game in games:
            moves = game.get("moves", [])
            for move in moves:
                move["_game"] = game
            self.by_game.append(moves)
            self.moves.extend(moves)
        self.deep = [m for m in self.moves if m.get("deep")]
        self.errors = [m for m in self.moves if m.get("severity") in SEVERE]
        self.blunders = [m for m in self.moves if m.get("severity") == "blunder"]
        self.timed = [m for m in self.moves if m.get("time_spent") is not None]

    @property
    def n_moves(self) -> int:
        return len(self.moves)

    def accuracy(self, moves: Optional[List[Dict]] = None) -> Optional[float]:
        return _mean(m["accuracy"] for m in (self.moves if moves is None else moves))

    def blunder_rate(self, moves: Optional[List[Dict]] = None) -> Optional[float]:
        pool = self.moves if moves is None else moves
        return _rate(sum(1 for m in pool if m.get("severity") == "blunder"), len(pool))

    def error_rate(self, moves: Optional[List[Dict]] = None) -> Optional[float]:
        pool = self.moves if moves is None else moves
        return _rate(sum(1 for m in pool if m.get("severity") in SEVERE), len(pool))


def _tagged(move: Dict, key: str, field_name: str = "shape_tags") -> bool:
    return key in (move.get(field_name) or [])


def _expected_share(moves: List[Dict], key: str) -> Optional[float]:
    """Average availability of a move type across these positions."""
    shares = [
        m["legal_mix"][key]
        for m in moves
        if m.get("legal_mix") and key in m["legal_mix"]
    ]
    return round(statistics.mean(shares), 3) if shares else None


# --------------------------------------------------------------------------
# Descriptive breakdowns
# --------------------------------------------------------------------------


def overview(corpus: Corpus) -> Dict:
    games = corpus.games
    results = Counter(g["result"] for g in games)
    colors = Counter(g["hero_color"] for g in games)
    elos = [int(g["hero_elo"]) for g in games if str(g.get("hero_elo", "")).isdigit()]
    return {
        "games": len(games),
        "moves_judged": corpus.n_moves,
        "wins": results.get("win", 0),
        "draws": results.get("draw", 0),
        "losses": results.get("loss", 0),
        "as_white": colors.get("white", 0),
        "as_black": colors.get("black", 0),
        "accuracy": corpus.accuracy(),
        "mean_loss_per_move": _mean(m["loss"] for m in corpus.moves),
        "blunders_per_100": corpus.blunder_rate(),
        "errors_per_100": corpus.error_rate(),
        "rating_range": [min(elos), max(elos)] if elos else None,
        "time_controls": dict(Counter(g["time_control"] for g in games)),
        "openings_seen": len({g["opening"] for g in games}),
        "clock_data": bool(corpus.timed),
    }


def by_bucket(corpus: Corpus, key: str, order: Optional[List[str]] = None) -> Dict:
    """Accuracy and error rates split by any per-move label."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for move in corpus.moves:
        groups[move.get(key) or "unknown"].append(move)
    keys = order or sorted(groups)
    out = {}
    for name in keys:
        pool = groups.get(name)
        if not pool:
            continue
        out[name] = {
            "moves": len(pool),
            "accuracy": corpus.accuracy(pool),
            "mean_loss": _mean(m["loss"] for m in pool),
            "blunders_per_100": corpus.blunder_rate(pool),
            "errors_per_100": corpus.error_rate(pool),
        }
    return out


def time_buckets(corpus: Corpus) -> Dict:
    """Error rates by how long you thought, and by how much clock was left."""
    if not corpus.timed:
        return {}
    edges = [(0, 2), (2, 5), (5, 10), (10, 25), (25, 10**6)]
    labels = ["under 2s", "2-5s", "5-10s", "10-25s", "over 25s"]
    spent: Dict[str, List[Dict]] = {label: [] for label in labels}
    for move in corpus.timed:
        value = move["time_spent"]
        for (low, high), label in zip(edges, labels):
            if low <= value < high:
                spent[label].append(move)
                break
    out = {"by_time_spent": {}}
    for label in labels:
        pool = spent[label]
        if not pool:
            continue
        out["by_time_spent"][label] = {
            "moves": len(pool),
            "accuracy": corpus.accuracy(pool),
            "blunders_per_100": corpus.blunder_rate(pool),
        }
    # Remaining clock, as a fraction of the starting time.
    pressure: Dict[str, List[Dict]] = defaultdict(list)
    for move in corpus.moves:
        base = move["_game"].get("base_time")
        clock = move.get("clock")
        if not base or clock is None:
            continue
        fraction = clock / base
        label = (
            "under 10% left" if fraction < 0.10
            else "10-25% left" if fraction < 0.25
            else "25-50% left" if fraction < 0.50
            else "over 50% left"
        )
        pressure[label].append(move)
    out["by_clock_left"] = {
        label: {
            "moves": len(pool),
            "accuracy": corpus.accuracy(pool),
            "blunders_per_100": corpus.blunder_rate(pool),
        }
        for label, pool in pressure.items()
        if pool
    }
    out["median_time_per_move"] = _mean([statistics.median(
        [m["time_spent"] for m in corpus.timed]
    )])
    return out


def motif_tally(corpus: Corpus) -> Dict:
    """Which tactics you fail to find, and which ones get played against you."""
    missed: Counter = Counter()
    allowed: Counter = Counter()
    for move in corpus.errors:
        missed.update(move.get("missed_motifs") or [])
        allowed.update(move.get("allowed_motifs") or [])
    return {
        "missed": dict(missed.most_common()),
        "allowed": dict(allowed.most_common()),
        "errors_considered": len(corpus.errors),
    }


def piece_tally(corpus: Corpus) -> Dict:
    """Which piece you were moving when it went wrong, and what punished you."""
    moving: Counter = Counter()
    punisher: Counter = Counter()
    victims: Counter = Counter()
    for move in corpus.errors:
        for tag in move.get("shape_tags") or []:
            if tag.startswith("piece:"):
                moving[tag.split(":", 1)[1]] += 1
        for detail in move.get("allowed_detail") or []:
            if detail.get("by"):
                punisher[detail["by"]] += 1
            for target in detail.get("targets") or []:
                victims[target.split()[0]] += 1
    return {
        "your_moving_piece": dict(moving.most_common()),
        "opponent_punishing_piece": dict(punisher.most_common()),
        "your_pieces_hit": dict(victims.most_common()),
    }


def opening_tally(corpus: Corpus) -> List[Dict]:
    """Per-opening scoring, accuracy, and where your knowledge runs out."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for game in corpus.games:
        groups[game.get("opening") or "unknown"].append(game)
    rows = []
    for name, games in groups.items():
        moves = [m for g in games for m in g.get("moves", [])]
        first_errors = []
        for game in games:
            errors = [
                m["move_number"]
                for m in game.get("moves", [])
                if m.get("severity") in SEVERE
            ]
            if errors:
                first_errors.append(min(errors))
        points = sum(
            1.0 if g["result"] == "win" else 0.5 if g["result"] == "draw" else 0.0
            for g in games
        )
        rows.append({
            "opening": name,
            "games": len(games),
            "score_pct": round(100 * points / len(games), 1),
            "accuracy": _mean(m["accuracy"] for m in moves),
            "mean_loss": _mean(m["loss"] for m in moves),
            "first_error_move": round(statistics.median(first_errors), 1)
            if first_errors else None,
            "colors": dict(Counter(g["hero_color"] for g in games)),
        })
    return sorted(rows, key=lambda r: -(r["mean_loss"] or 0))


def conversion(corpus: Corpus) -> Dict:
    """Do you finish won games, and do you save lost ones?"""
    reached_winning = won_from_winning = 0
    reached_losing = saved_from_losing = 0
    thrown = []
    for game in corpus.games:
        curve = [c for c in (game.get("curve") or []) if c is not None]
        if not curve:
            continue
        peak, trough = max(curve), min(curve)
        if peak >= 80:
            reached_winning += 1
            if game["result"] == "win":
                won_from_winning += 1
            elif game["result"] != "win":
                thrown.append({
                    "opening": game["opening"],
                    "peak_win_pct": peak,
                    "result": game["result"],
                    "date": game.get("date", ""),
                })
        if trough <= 20:
            reached_losing += 1
            if game["result"] != "loss":
                saved_from_losing += 1
    return {
        "games_reaching_winning": reached_winning,
        "converted": won_from_winning,
        "conversion_pct": _share(won_from_winning, reached_winning),
        "games_reaching_losing": reached_losing,
        "saved": saved_from_losing,
        "save_pct": _share(saved_from_losing, reached_losing),
        "thrown_away": thrown[:8],
    }


# --------------------------------------------------------------------------
# Habit detectors
#
# Each one answers a single question and returns either a Finding or None.
# None is the common and correct outcome: most players do not have most of
# these problems, and a profile that fires every detector is a horoscope.
# --------------------------------------------------------------------------


def _against_base_rate(
    moves: List[Dict],
    tag: str,
    mix_key: str,
    tag_field: str = "shape_tags",
    mix_field: str = "legal_mix",
) -> Optional[Tuple[int, int, float, float, float]]:
    """Compare how often a move type shows up against how often it was there.

    Returns (hits, n, observed share, expected share, lift) or None when
    there is nothing to compare.
    """
    pool = [m for m in moves if m.get(mix_field) and m.get(tag_field) is not None]
    if len(pool) < 5:
        return None
    hits = sum(1 for m in pool if tag in (m.get(tag_field) or []))
    expected_values = [
        m[mix_field][mix_key] for m in pool if mix_key in m[mix_field]
    ]
    if not expected_values:
        return None
    observed = hits / len(pool)
    expected = statistics.mean(expected_values)
    lift = observed / expected if expected > 0 else 0.0
    return hits, len(pool), observed, expected, lift


def detect_hanging_pieces(corpus: Corpus) -> Optional[Finding]:
    """The plainest fault of all: you left something to be taken."""
    pool = corpus.blunders
    if len(pool) < 4:
        return None
    hits = sum(
        1 for m in pool
        if "undefended_piece" in (m.get("allowed_motifs") or [])
    )
    share = hits / len(pool)
    if share < 0.25:
        return None
    return Finding(
        key="board_vision",
        title="Pieces left undefended",
        definition=(
            "Not a cognitive bias -- a board-vision gap. The move was "
            "punished by a capture of something nothing was defending."
        ),
        signal="Share of your blunders whose refutation simply takes an undefended piece.",
        evidence=f"{hits} of {len(pool)} blunders ({share:.0%}) handed over an undefended piece.",
        n=len(pool),
        strength=_strength(len(pool), None, gap=share - 0.25),
        drill=(
            "Before every move, name every one of your pieces that nothing "
            "defends. Most players can do this in five seconds and it removes "
            "the majority of these."
        ),
        numbers={"hits": hits, "blunders": len(pool), "share": round(share, 3)},
    )


def detect_opponent_reply_blindness(corpus: Corpus) -> Optional[Finding]:
    """Did the punishment come from a move you could have seen coming?"""
    pool = [m for m in corpus.errors if m.get("legal_mix_after")]
    if len(pool) < 5:
        return None
    forcing = sum(
        1 for m in pool
        if {"capture", "check"} & set(m.get("allowed_shape_tags") or [])
    )
    expected_values = [
        m["legal_mix_after"].get("captures", 0) + m["legal_mix_after"].get("checks", 0)
        for m in pool
    ]
    expected = statistics.mean(expected_values)
    observed = forcing / len(pool)
    lift = observed / expected if expected else 0.0
    if observed < 0.45 or lift < 1.2:
        return None
    return Finding(
        key="opponent_blindness",
        title="You calculate your plan, not their reply",
        definition=(
            "Egocentric calculation: attention goes to your own intention, so "
            "the opponent's most obvious answers never get examined. The "
            "best-documented single cause of amateur blunders."
        ),
        signal=(
            "Share of your mistakes refuted by a check or a capture -- the two "
            "move types that are cheapest to check -- against how often such "
            "replies were available at all."
        ),
        evidence=(
            f"{forcing} of {len(pool)} mistakes ({observed:.0%}) were punished by a "
            f"check or capture, while only {expected:.0%} of their legal replies "
            f"were checks or captures ({lift:.1f}x the base rate)."
        ),
        n=len(pool),
        strength=_strength(len(pool), lift),
        drill=(
            "Adopt a fixed trigger: after choosing a move, and before playing "
            "it, list every check and every capture your opponent then has. "
            "Only then play. It costs ten seconds."
        ),
        numbers={
            "forcing_refutations": forcing,
            "mistakes": len(pool),
            "observed": round(observed, 3),
            "expected": round(expected, 3),
            "lift": round(lift, 2),
        },
    )


def detect_forcing_move_tunnel(corpus: Corpus) -> Optional[Finding]:
    """Do your own errors cluster in checks and captures?"""
    found = _against_base_rate(corpus.errors, "capture", "captures")
    if not found:
        return None
    hits, n, observed, expected, lift = found
    if lift < 1.3 or hits < 4:
        return None
    return Finding(
        key="capture_reflex",
        title="The capture gets played because it is a capture",
        definition=(
            "Acquisitiveness, a cousin of loss aversion: taking material is "
            "concrete and immediate, so it gets chosen over quiet moves "
            "without being compared to them."
        ),
        signal=(
            "Share of your mistakes that were captures, against the share of "
            "your legal moves that were captures."
        ),
        evidence=(
            f"{hits} of {n} mistakes ({observed:.0%}) were captures, but only "
            f"{expected:.0%} of your legal moves were ({lift:.1f}x)."
        ),
        n=n,
        strength=_strength(n, lift),
        drill=(
            "When a capture looks obvious, force yourself to write down one "
            "non-capturing candidate and compare them. The habit is not 'stop "
            "taking' -- it is 'take on purpose'."
        ),
        numbers={"hits": hits, "n": n, "observed": round(observed, 3),
                 "expected": round(expected, 3), "lift": round(lift, 2)},
    )


def detect_quiet_move_blindness(corpus: Corpus) -> Optional[Finding]:
    """Are the moves you fail to find the non-forcing ones?"""
    missed = [m for m in corpus.errors if m.get("best_shape_tags")]
    found = _against_base_rate(missed, "quiet_move", "quiet", tag_field="best_shape_tags")
    if not found:
        return None
    hits, n, observed, expected, lift = found
    if observed < 0.5 or lift < 1.1:
        return None
    return Finding(
        key="quiet_move_blindness",
        title="Your candidate list is all forcing moves",
        definition=(
            "Availability bias in move generation: checks and captures "
            "announce themselves, quiet improving moves have to be looked "
            "for, so they never enter the comparison."
        ),
        signal=(
            "Share of the best moves you missed that were quiet (no check, no "
            "capture), against the share of quiet moves available."
        ),
        evidence=(
            f"{hits} of {n} missed best moves ({observed:.0%}) were quiet moves "
            f"(base rate {expected:.0%}, {lift:.1f}x)."
        ),
        n=n,
        strength=_strength(n, lift),
        drill=(
            "In any position you think is critical, require one candidate that "
            "is neither a check nor a capture before you start calculating -- "
            "a piece improvement, a prophylactic move, a pawn lever."
        ),
        numbers={"hits": hits, "n": n, "observed": round(observed, 3),
                 "expected": round(expected, 3), "lift": round(lift, 2)},
    )


def detect_retreat_blindness(corpus: Corpus) -> Optional[Finding]:
    """Backward moves are measurably harder for humans to see. Are they for you?"""
    missed = [m for m in corpus.errors if m.get("best_shape_tags")]
    found = _against_base_rate(
        missed, "backward_move", "backward", tag_field="best_shape_tags"
    )
    if not found:
        return None
    hits, n, observed, expected, lift = found
    if lift < 1.4 or hits < 4:
        return None
    return Finding(
        key="retreat_blindness",
        title="Backward moves don't occur to you",
        definition=(
            "Directional bias: attention follows the direction of attack, so "
            "retreats, regroupings and backward defensive moves are "
            "under-generated as candidates."
        ),
        signal=(
            "Share of missed best moves that moved a piece backwards, against "
            "the share of legal moves that were backward."
        ),
        evidence=(
            f"{hits} of {n} missed best moves ({observed:.0%}) were backward "
            f"moves, against a {expected:.0%} base rate ({lift:.1f}x)."
        ),
        n=n,
        strength=_strength(n, lift),
        drill=(
            "When a piece is attacked or a plan stalls, explicitly consider "
            "the ugliest retreat on the board. Strong players' moves look "
            "backwards far more often than amateurs'."
        ),
        numbers={"hits": hits, "n": n, "observed": round(observed, 3),
                 "expected": round(expected, 3), "lift": round(lift, 2)},
    )


def detect_winning_complacency(corpus: Corpus) -> Optional[Finding]:
    """Do you get worse once you are ahead?"""
    ahead = [m for m in corpus.moves if m.get("band_before") in ("winning", "better")]
    level = [m for m in corpus.moves if m.get("band_before") == "equal"]
    if len(ahead) < 30 or len(level) < 30:
        return None
    ahead_rate = corpus.error_rate(ahead) or 0.0
    level_rate = corpus.error_rate(level) or 0.0
    if level_rate <= 0:
        return None
    lift = ahead_rate / level_rate
    if lift < 1.25:
        return None
    return Finding(
        key="winning_complacency",
        title="You relax when you are winning",
        definition=(
            "Overconfidence plus reduced effort: once the position feels won, "
            "calculation quietly stops and moves get played on general "
            "impressions."
        ),
        signal="Your error rate in better/winning positions against equal ones.",
        evidence=(
            f"{ahead_rate:.1f} errors per 100 moves when ahead versus "
            f"{level_rate:.1f} when equal ({lift:.1f}x), over "
            f"{len(ahead)} winning-side moves."
        ),
        n=len(ahead),
        strength=_strength(len(ahead) // 5, lift),
        drill=(
            "Treat reaching a winning position as a trigger to slow down, not "
            "speed up. Winning positions have one job: trade pieces and remove "
            "counterplay. Name your opponent's only source of activity before "
            "each move."
        ),
        numbers={"ahead_error_rate": ahead_rate, "equal_error_rate": level_rate,
                 "lift": round(lift, 2), "moves_ahead": len(ahead)},
    )


def detect_post_error_collapse(corpus: Corpus) -> Optional[Finding]:
    """After the first real mistake, does the rest of the game fall apart?"""
    before_moves = after_moves = 0
    before_errors = after_errors = 0
    games_with = 0
    for moves in corpus.by_game:
        first = next(
            (i for i, m in enumerate(moves) if m.get("severity") == "blunder"), None
        )
        if first is None:
            continue
        games_with += 1
        for index, move in enumerate(moves):
            if index < first:
                before_moves += 1
                before_errors += move.get("severity") in SEVERE
            elif index > first:
                after_moves += 1
                after_errors += move.get("severity") in SEVERE
    if before_moves < 25 or after_moves < 25 or games_with < 4:
        return None
    before_rate = 100 * before_errors / before_moves
    after_rate = 100 * after_errors / after_moves
    if before_rate <= 0:
        return None
    lift = after_rate / before_rate
    if lift < 1.4:
        return None
    return Finding(
        key="post_error_collapse",
        title="One mistake becomes three",
        definition=(
            "Two mechanisms the data cannot separate: tilt (the emotional "
            "cost of the error eats the attention the position now needs) and "
            "sunk cost (you keep pushing the idea that failed instead of "
            "re-assessing a changed position). Either way the first mistake is "
            "not what loses the game."
        ),
        signal=(
            "Your error rate before your first blunder of a game against your "
            "error rate after it."
        ),
        evidence=(
            f"{before_rate:.1f} errors per 100 moves before the first blunder, "
            f"{after_rate:.1f} after it ({lift:.1f}x), across {games_with} games."
        ),
        n=after_moves,
        strength=_strength(games_with * 2, lift),
        drill=(
            "Build a reset ritual: after any move you know was bad, take one "
            "deliberate pause and re-evaluate the position from scratch, as if "
            "you had just sat down at it. The previous plan is evidence, not an "
            "obligation."
        ),
        numbers={"before_rate": round(before_rate, 2), "after_rate": round(after_rate, 2),
                 "lift": round(lift, 2), "games": games_with},
    )


def detect_time_impulsivity(corpus: Corpus) -> Optional[Finding]:
    """Do your mistakes happen on the moves you barely thought about?"""
    timed = corpus.timed
    if len(timed) < 60:
        return None
    median = statistics.median(m["time_spent"] for m in timed)
    if median <= 0:
        return None
    fast = [m for m in timed if m["time_spent"] < 0.5 * median]
    slow = [m for m in timed if m["time_spent"] >= median]
    if len(fast) < 20 or len(slow) < 20:
        return None
    fast_rate = corpus.error_rate(fast) or 0.0
    slow_rate = corpus.error_rate(slow) or 0.0
    if slow_rate <= 0:
        return None
    lift = fast_rate / slow_rate
    if lift < 1.3:
        return None
    return Finding(
        key="impulsivity",
        title="Your mistakes are the moves you didn't stop for",
        definition=(
            "Impulsive responding: the first plausible move gets played before "
            "the position has been checked, usually in positions that looked "
            "familiar rather than positions that were simple."
        ),
        signal=(
            "Error rate on moves played in under half your median thinking "
            "time, against moves where you took at least the median."
        ),
        evidence=(
            f"{fast_rate:.1f} errors per 100 moves on quick moves (under "
            f"{0.5 * median:.1f}s) versus {slow_rate:.1f} on considered ones "
            f"({lift:.1f}x), median think time {median:.1f}s."
        ),
        n=len(fast),
        strength=_strength(len(fast) // 4, lift),
        drill=(
            "Recognise that 'obvious' is a feeling, not a fact. Spend three "
            "seconds on every recapture and every move that feels forced -- "
            "that is where this leaks points."
        ),
        numbers={"median_s": round(median, 1), "fast_rate": fast_rate,
                 "slow_rate": slow_rate, "lift": round(lift, 2)},
    )


def detect_time_trouble(corpus: Corpus) -> Optional[Finding]:
    """How much of the damage happens in the last of the clock?"""
    low, high = [], []
    for move in corpus.moves:
        base = move["_game"].get("base_time")
        clock = move.get("clock")
        if not base or clock is None:
            continue
        (low if clock / base < 0.15 else high if clock / base > 0.5 else []).append(move)
    if len(low) < 25 or len(high) < 25:
        return None
    low_rate = corpus.error_rate(low) or 0.0
    high_rate = corpus.error_rate(high) or 0.0
    if high_rate <= 0:
        return None
    lift = low_rate / high_rate
    if lift < 1.4:
        return None
    return Finding(
        key="time_trouble",
        title="The clock, not the position, is beating you",
        definition=(
            "Time mismanagement: effort is spent early on positions that did "
            "not need it, leaving none for the positions that did."
        ),
        signal=(
            "Error rate with under 15% of your clock left, against moves with "
            "over half the clock left."
        ),
        evidence=(
            f"{low_rate:.1f} errors per 100 moves in the last 15% of your clock "
            f"versus {high_rate:.1f} early ({lift:.1f}x), over {len(low)} moves."
        ),
        n=len(low),
        strength=_strength(len(low) // 4, lift),
        drill=(
            "Budget the clock: decide before the game roughly how many seconds "
            "a move you get, and spend the surplus only on positions you have "
            "identified as critical -- not on the first complicated thing you see."
        ),
        numbers={"low_clock_rate": low_rate, "healthy_clock_rate": high_rate,
                 "lift": round(lift, 2)},
    )


def detect_plan_persistence(corpus: Corpus) -> Optional[Finding]:
    """Do your losses come in runs rather than singly?"""
    streaks = []
    for moves in corpus.by_game:
        run: List[Dict] = []
        for move in moves:
            if (move.get("loss") or 0) >= 4.0:
                run.append(move)
            else:
                if len(run) >= 3:
                    streaks.append(run)
                run = []
        if len(run) >= 3:
            streaks.append(run)
    if len(streaks) < 3:
        return None
    in_streak = sum(m.get("loss") or 0 for run in streaks for m in run)
    total = sum(m.get("loss") or 0 for m in corpus.moves)
    share = in_streak / total if total else 0.0
    if share < 0.3:
        return None
    lengths = [len(run) for run in streaks]
    return Finding(
        key="plan_persistence",
        title="You ride a plan past the point it stopped working",
        definition=(
            "The Einstellung effect, the most studied cognitive trap in chess: "
            "once a plan is in mind it suppresses the search for better ideas, "
            "and players keep following it while the position argues against it. "
            "Measured here as runs of consecutive losing moves rather than "
            "isolated oversights."
        ),
        signal=(
            "Share of all the win probability you lost that was lost in runs of "
            "three or more consecutive costly moves."
        ),
        evidence=(
            f"{len(streaks)} runs of 3+ consecutive costly moves (longest "
            f"{max(lengths)}), accounting for {share:.0%} of everything you lost."
        ),
        n=len(streaks),
        strength=_strength(len(streaks) * 3, None, gap=share - 0.3),
        drill=(
            "Re-evaluate from scratch whenever your opponent makes a move you "
            "did not expect. Ask what changed, not how to continue. A plan is a "
            "hypothesis, and your opponent keeps supplying evidence."
        ),
        numbers={"streaks": len(streaks), "longest": max(lengths),
                 "share_of_loss": round(share, 3)},
    )


def detect_king_safety_neglect(corpus: Corpus) -> Optional[Finding]:
    """How often does the punishment come at your king?"""
    pool = corpus.errors
    if len(pool) < 6:
        return None
    hits = sum(
        1 for m in pool
        if {"mating_attack", "back_rank_mate"} & set(m.get("allowed_motifs") or [])
    )
    share = hits / len(pool)
    if share < 0.2 or hits < 3:
        return None
    return Finding(
        key="king_safety",
        title="The punishment arrives at your king",
        definition=(
            "Attention asymmetry: material is countable and king safety is "
            "not, so the side of the board where the game is actually decided "
            "gets the least checking."
        ),
        signal="Share of your mistakes whose refutation was a mating attack.",
        evidence=f"{hits} of {len(pool)} mistakes ({share:.0%}) let a mating attack in.",
        n=len(pool),
        strength=_strength(len(pool), None, gap=share - 0.2),
        drill=(
            "Count attackers and defenders around your own king every time "
            "your opponent brings a new piece towards it. Material can wait; "
            "mate cannot."
        ),
        numbers={"hits": hits, "errors": len(pool), "share": round(share, 3)},
    )


def detect_phase_gap(corpus: Corpus) -> Optional[Finding]:
    """Which third of the game costs you most?"""
    phases = by_bucket(corpus, "phase", ["opening", "middlegame", "endgame"])
    usable = {k: v for k, v in phases.items() if v["moves"] >= 25}
    if len(usable) < 2:
        return None
    worst = max(usable, key=lambda k: usable[k]["errors_per_100"] or 0)
    best = min(usable, key=lambda k: usable[k]["errors_per_100"] or 0)
    worst_rate = usable[worst]["errors_per_100"] or 0
    best_rate = usable[best]["errors_per_100"] or 0
    if best_rate <= 0 or worst_rate / best_rate < 1.5:
        return None
    advice = {
        "opening": (
            "Your openings need understanding rather than more moves: for each "
            "one you play, learn the typical structure, whose plan goes where, "
            "and the three most common tactical accidents."
        ),
        "middlegame": (
            "Middlegame work means candidate-move discipline and tactical "
            "pattern volume -- puzzles that are positions, not 'white to play "
            "and win'."
        ),
        "endgame": (
            "Endgames reward study more efficiently than anything else at this "
            "level: king activity, pawn races, rook endings, and the basic "
            "theoretical draws."
        ),
    }
    return Finding(
        key=f"phase_gap_{worst}",
        title=f"The {worst} is where you lose points",
        definition=(
            "Not a bias but a skill distribution: the phases of the game draw "
            "on different abilities, and yours are uneven."
        ),
        signal="Error rate per 100 moves by game phase.",
        evidence=(
            f"{worst}: {worst_rate:.1f} errors per 100 moves "
            f"({usable[worst]['moves']} moves) versus {best}: {best_rate:.1f} "
            f"({usable[best]['moves']} moves), {worst_rate / best_rate:.1f}x."
        ),
        n=usable[worst]["moves"],
        strength=_strength(usable[worst]["moves"] // 5, worst_rate / best_rate),
        drill=advice[worst],
        numbers={"phases": usable},
    )


def detect_opening_cliff(corpus: Corpus) -> Optional[Finding]:
    """Is there one opening where things reliably go wrong, and when?"""
    rows = [r for r in opening_tally(corpus) if r["games"] >= 2]
    if len(rows) < 2:
        return None
    overall = _mean(m["loss"] for m in corpus.moves) or 0.0
    worst = rows[0]
    if not worst["mean_loss"] or overall <= 0:
        return None
    lift = worst["mean_loss"] / overall
    if lift < 1.4:
        return None
    when = worst["first_error_move"]
    return Finding(
        key="opening_cliff",
        title=f"Your knowledge runs out in the {worst['opening']}",
        definition=(
            "Anchoring on memorised moves: preparation carries you to a "
            "position you recognise, and the first move you have to invent is "
            "where the game turns."
        ),
        signal=(
            "Mean win probability lost per move, per opening, with the median "
            "move number of the first real mistake."
        ),
        evidence=(
            f"{worst['opening']}: {worst['mean_loss']} win% lost per move across "
            f"{worst['games']} games versus {overall} overall ({lift:.1f}x); "
            f"first mistake typically around move {when}."
        ),
        n=worst["games"],
        strength=_strength(worst["games"] * 3, lift),
        drill=(
            f"Take the {worst['opening']} to move {when} or so and work out what "
            "the position actually wants -- the pawn structure, the piece "
            "placements it implies, the plan for both sides. Memorising two more "
            "moves will not help; understanding the resulting structure will."
        ),
        numbers=worst,
    )


DETECTORS = (
    detect_hanging_pieces,
    detect_opponent_reply_blindness,
    detect_forcing_move_tunnel,
    detect_quiet_move_blindness,
    detect_retreat_blindness,
    detect_winning_complacency,
    detect_post_error_collapse,
    detect_time_impulsivity,
    detect_time_trouble,
    detect_plan_persistence,
    detect_king_safety_neglect,
    detect_phase_gap,
    detect_opening_cliff,
)

_RANK = {"strong": 0, "moderate": 1, "weak": 2, "anecdote": 3}


def key_moments(corpus: Corpus, limit: int = 10) -> List[Dict]:
    """The most expensive, most instructive positions in the corpus."""
    scored = [
        m for m in corpus.moves
        if m.get("deep") and m.get("severity") and m.get("best_san")
    ]
    # Worth learning from means costly *and* from a position that was still
    # playable -- throwing away a won game teaches more than move 40 of a
    # position that was already lost.
    scored.sort(key=lambda m: -(m["loss"] * (1.0 if m["win_before"] >= 30 else 0.4)))
    out = []
    seen_positions = set()
    for move in scored:
        if len(out) >= limit:
            break
        # The same position reached twice teaches the same lesson twice.
        if move["fen_before"] in seen_positions:
            continue
        seen_positions.add(move["fen_before"])
        game = move["_game"]
        out.append({
            "game": f"{game['white']} vs {game['black']}",
            "game_id": game.get("id", ""),
            "date": game.get("date", ""),
            "opening": game.get("opening", ""),
            "hero_color": game.get("hero_color", ""),
            "move_number": move["move_number"],
            "played": move["san"],
            "best": move["best_san"],
            "best_line": move.get("best_line_san"),
            "refutation": move.get("refutation_san"),
            "loss": move["loss"],
            "win_before": move["win_before"],
            "win_after": move["win_after"],
            "severity": move["severity"],
            "phase": move["phase"],
            "fen": move["fen_before"],
            "time_spent": move.get("time_spent"),
            "complexity": move.get("complexity"),
            "only_move": move.get("only_move"),
            "missed_motifs": move.get("missed_motifs"),
            "allowed_motifs": move.get("allowed_motifs"),
            "shape_tags": move.get("shape_tags"),
            "structure": move.get("structure"),
        })
    return out


def build_profile(games: List[Dict], moments: int = 10) -> Dict:
    """The whole picture: what you do, and where the evidence is thin."""
    corpus = Corpus(games)
    findings = []
    for detector in DETECTORS:
        try:
            found = detector(corpus)
        except Exception as error:  # a broken detector must not kill the report
            findings.append(Finding(
                key=f"error_{detector.__name__}",
                title="Detector failed",
                definition="", signal="", evidence=str(error), n=0,
                strength="anecdote", drill="",
            ))
            continue
        if found:
            findings.append(found)
    findings.sort(key=lambda f: (_RANK.get(f.strength, 9), -f.n))

    caveats = []
    if len(games) < 10:
        caveats.append(
            f"Only {len(games)} games analysed. Treat everything here as a "
            "hypothesis; habits need 20-30 games before the numbers settle."
        )
    if corpus.n_moves < 300:
        caveats.append(
            f"{corpus.n_moves} of your moves judged. Rates per 100 moves are "
            "noisy below about 400."
        )
    if not corpus.timed:
        caveats.append(
            "No clock data in these games, so nothing about time management "
            "could be measured."
        )
    if len({g["opening"] for g in games}) > len(games) * 0.7:
        caveats.append(
            "Your openings barely repeat, so per-opening numbers rest on one "
            "or two games each."
        )

    result = {
        "overview": overview(corpus),
        "by_phase": by_bucket(corpus, "phase", ["opening", "middlegame", "endgame"]),
        "by_position_state": by_bucket(
            corpus, "band_before", ["winning", "better", "equal", "worse", "losing"]
        ),
        "by_color": by_bucket(corpus, "side", ["white", "black"]),
        "time": time_buckets(corpus),
        "motifs": motif_tally(corpus),
        "pieces": piece_tally(corpus),
        "openings": opening_tally(corpus),
        "conversion": conversion(corpus),
        "findings": [asdict(f) for f in findings],
        "caveats": caveats,
        "key_moments": key_moments(corpus, moments),
    }

    # Only now that every section has been computed: drop the back-references
    # to the parent games, which would otherwise make the move dicts cyclic
    # and json.dumps recurse forever. Doing this any earlier breaks whichever
    # section happens to be evaluated afterwards.
    for move in corpus.moves:
        move.pop("_game", None)
    return result
