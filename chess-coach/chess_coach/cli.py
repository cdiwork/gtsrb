"""Command line entry points."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import chess.pgn

from . import __version__
from .analyse import AnalysisConfig, analyse_game, hero_color
from .engine import Analyst
from .fetch import DEFAULT_CLASSES, FetchError, fetch
from .profile import build_profile


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_games(path: Path) -> List[chess.pgn.Game]:
    games = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games.append(game)
    return games


# --------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> int:
    try:
        pgn = fetch(
            args.site,
            args.user,
            max_games=args.max,
            time_classes=args.time_class or DEFAULT_CLASSES,
            rated_only=not args.include_unrated,
            token=args.token,
            progress=lambda url, have: _log(f"  ... {url.rsplit('/', 2)[-2]}-"
                                           f"{url.rsplit('/', 1)[-1]}: {have} games"),
        )
    except FetchError as error:
        _log(f"error: {error}")
        return 2
    Path(args.out).write_text(pgn, encoding="utf-8")
    count = pgn.count("[Event ")
    _log(f"wrote {count} games to {args.out}")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    games = _read_games(Path(args.pgn))
    mine = [g for g in games if hero_color(g, args.user) is not None]
    if not mine:
        names = sorted({g.headers.get(c, "") for g in games for c in ("White", "Black")})
        _log(f"error: no games in {args.pgn} were played by {args.user!r}.")
        _log(f"players present: {', '.join(n for n in names if n)[:300]}")
        return 2
    if args.limit:
        mine = mine[: args.limit]

    config = AnalysisConfig(
        fast_depth=args.fast_depth,
        deep_depth=args.deep_depth,
        multipv=args.multipv,
        movetime=args.movetime,
        deep_movetime=args.deep_movetime,
    )
    started = time.time()
    reports = []
    with Analyst(path=args.engine, threads=args.threads, hash_mb=args.hash) as analyst:
        _log(f"engine: {analyst.id} ({analyst.threads} threads)")
        for index, game in enumerate(mine, 1):
            label = (
                f"[{index}/{len(mine)}] {game.headers.get('White', '?')} vs "
                f"{game.headers.get('Black', '?')}"
            )
            report = analyse_game(game, args.user, analyst, config)
            if report is None:
                _log(f"{label}: skipped")
                continue
            reports.append(report)
            counts = report["counts"]
            _log(
                f"{label} ({report['opening'][:28]}) {report['result']}: "
                f"accuracy {report['accuracy']}, "
                f"{counts['blunder']}B/{counts['mistake']}M/{counts['inaccuracy']}I"
            )
    payload = {
        "tool": f"chess-coach {__version__}",
        "hero": args.user,
        "games": reports,
        "config": vars(config) if hasattr(config, "__dict__") else {},
        "seconds": round(time.time() - started, 1),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    _log(f"analysed {len(reports)} games in {payload['seconds']}s -> {args.out}")
    return 0


def _load_analysis(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "games" not in data:
        raise SystemExit(f"{path} does not look like a chess-coach analysis file")
    return data


def cmd_profile(args: argparse.Namespace) -> int:
    data = _load_analysis(args.analysis)
    profile = build_profile(data["games"], moments=args.moments)
    profile["hero"] = data.get("hero", "")
    Path(args.out).write_text(json.dumps(profile, indent=1), encoding="utf-8")
    findings = profile["findings"]
    _log(f"profile for {profile['hero']}: {len(findings)} findings -> {args.out}")
    for finding in findings:
        _log(f"  [{finding['strength']:8s}] {finding['title']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import render_html, render_markdown

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    if args.md:
        Path(args.md).write_text(render_markdown(profile), encoding="utf-8")
        _log(f"wrote {args.md}")
    if args.html:
        Path(args.html).write_text(render_html(profile), encoding="utf-8")
        _log(f"wrote {args.html}")
    if not (args.md or args.html):
        print(render_markdown(profile))
    return 0


def cmd_dossier(args: argparse.Namespace) -> int:
    from .dossier import render_dossier

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    text = render_dossier(profile, limit=args.limit)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        _log(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_coach(args: argparse.Namespace) -> int:
    """fetch -> analyse -> profile -> report, in one go."""
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pgn_path = workdir / "games.pgn"
    analysis_path = workdir / "analysis.json"
    profile_path = workdir / "profile.json"

    if not args.pgn:
        fetch_args = argparse.Namespace(
            site=args.site, user=args.user, max=args.max,
            time_class=args.time_class, include_unrated=args.include_unrated,
            token=args.token, out=str(pgn_path),
        )
        if cmd_fetch(fetch_args) != 0:
            return 2
    else:
        pgn_path = Path(args.pgn)

    analyse_args = argparse.Namespace(
        pgn=str(pgn_path), user=args.user, out=str(analysis_path), limit=args.max,
        engine=args.engine, threads=args.threads, hash=args.hash,
        fast_depth=args.fast_depth, deep_depth=args.deep_depth,
        multipv=3, movetime=args.movetime, deep_movetime=args.deep_movetime,
    )
    if cmd_analyse(analyse_args) != 0:
        return 2
    if cmd_profile(argparse.Namespace(
        analysis=str(analysis_path), out=str(profile_path), moments=args.moments
    )) != 0:
        return 2
    return cmd_report(argparse.Namespace(
        profile=str(profile_path),
        md=str(workdir / "report.md"),
        html=str(workdir / "report.html"),
    ))


# --------------------------------------------------------------------------


def _add_engine_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", help="path to the Stockfish binary")
    parser.add_argument("--threads", type=int, help="engine threads (default: cores-1)")
    parser.add_argument("--hash", type=int, default=256, help="engine hash in MB")
    parser.add_argument("--fast-depth", type=int, default=14,
                        help="depth for the sweep over every position")
    parser.add_argument("--deep-depth", type=int, default=20,
                        help="depth for the second look at suspicious moves")
    parser.add_argument("--movetime", type=float,
                        help="seconds per position instead of a fixed depth")
    parser.add_argument("--deep-movetime", type=float,
                        help="seconds per suspicious position instead of a depth")


def _add_fetch_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--site", default="chess.com",
                        choices=["chess.com", "lichess"], help="where you play")
    parser.add_argument("--max", type=int, default=30, help="how many games")
    parser.add_argument("--time-class", action="append",
                        choices=["bullet", "blitz", "rapid", "classical", "daily"],
                        help="repeatable; default blitz+rapid+classical")
    parser.add_argument("--include-unrated", action="store_true")
    parser.add_argument("--token", help="Lichess API token (optional, for more games)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chess-coach",
        description="Analyse your own games and find out how you lose.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subs.add_parser("fetch", help="download your recent games")
    fetch_parser.add_argument("--user", required=True)
    _add_fetch_flags(fetch_parser)
    fetch_parser.add_argument("-o", "--out", default="games.pgn")
    fetch_parser.set_defaults(func=cmd_fetch)

    analyse_parser = subs.add_parser("analyse", help="judge every move you played")
    analyse_parser.add_argument("pgn")
    analyse_parser.add_argument("--user", required=True)
    analyse_parser.add_argument("--limit", type=int, help="only the first N games")
    analyse_parser.add_argument("--multipv", type=int, default=3)
    _add_engine_flags(analyse_parser)
    analyse_parser.add_argument("-o", "--out", default="analysis.json")
    analyse_parser.set_defaults(func=cmd_analyse)

    profile_parser = subs.add_parser("profile", help="aggregate into a weakness profile")
    profile_parser.add_argument("analysis")
    profile_parser.add_argument("--moments", type=int, default=10)
    profile_parser.add_argument("-o", "--out", default="profile.json")
    profile_parser.set_defaults(func=cmd_profile)

    report_parser = subs.add_parser("report", help="render the profile for humans")
    report_parser.add_argument("profile")
    report_parser.add_argument("--md")
    report_parser.add_argument("--html")
    report_parser.set_defaults(func=cmd_report)

    dossier_parser = subs.add_parser(
        "dossier", help="dump key positions for a coach (or an LLM) to explain"
    )
    dossier_parser.add_argument("profile")
    dossier_parser.add_argument("--limit", type=int, default=6)
    dossier_parser.add_argument("-o", "--out")
    dossier_parser.set_defaults(func=cmd_dossier)

    coach_parser = subs.add_parser("coach", help="fetch, analyse, profile and report")
    coach_parser.add_argument("--user", required=True)
    coach_parser.add_argument("--pgn", help="use this PGN instead of downloading")
    _add_fetch_flags(coach_parser)
    _add_engine_flags(coach_parser)
    coach_parser.add_argument("--moments", type=int, default=10)
    coach_parser.add_argument("--workdir", default="coach-out")
    coach_parser.set_defaults(func=cmd_coach)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
