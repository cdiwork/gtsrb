"""Getting your games off Lichess and Chess.com.

Both sites publish your own games through a public API, so no login and no
password is needed -- just the handle. Bullet is excluded by default: at one
minute a game the mistakes are mostly mouse speed, and that teaches you very
little about how you think.
"""
from __future__ import annotations

import io
import time
from typing import Dict, Iterable, List, Optional

import requests

# Both APIs ask for a descriptive User-Agent, and Chess.com returns 403
# without one.
USER_AGENT = "chess-coach/0.1 (personal game analysis; https://github.com/)"
DEFAULT_CLASSES = ("blitz", "rapid", "classical")


class FetchError(RuntimeError):
    pass


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


# --------------------------------------------------------------------------
# Lichess
# --------------------------------------------------------------------------


def fetch_lichess(
    user: str,
    max_games: int = 40,
    time_classes: Iterable[str] = DEFAULT_CLASSES,
    rated_only: bool = True,
    since: Optional[int] = None,
    token: Optional[str] = None,
) -> str:
    """Download a user's recent games from Lichess as one PGN string.

    Clock comments are requested explicitly -- the time-pressure half of the
    analysis is worthless without them.
    """
    params = {
        "max": max_games,
        "clocks": "true",
        "evals": "false",
        "opening": "true",
        "moves": "true",
        "tags": "true",
        "perfType": ",".join(time_classes),
    }
    if rated_only:
        params["rated"] = "true"
    if since:
        params["since"] = since
    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://lichess.org/api/games/user/{user}"
    response = _session().get(url, params=params, headers=headers, timeout=120)
    if response.status_code == 404:
        raise FetchError(f"Lichess has no user called {user!r}")
    if response.status_code == 429:
        raise FetchError("Lichess rate limit hit -- wait a minute and retry")
    response.raise_for_status()
    if not response.text.strip():
        raise FetchError(
            f"No games found for {user!r} on Lichess with those filters "
            f"(time classes: {', '.join(time_classes)}, rated_only={rated_only})"
        )
    return response.text


# --------------------------------------------------------------------------
# Chess.com
# --------------------------------------------------------------------------


def fetch_chesscom(
    user: str,
    max_games: int = 40,
    time_classes: Iterable[str] = DEFAULT_CLASSES,
    rated_only: bool = True,
    progress=None,
) -> str:
    """Download a user's recent games from Chess.com as one PGN string.

    Chess.com publishes one archive per month, so we walk months backwards
    from the present until we have enough games.
    """
    session = _session()
    wanted = set(time_classes)
    index = session.get(
        f"https://api.chess.com/pub/player/{user}/games/archives", timeout=60
    )
    if index.status_code == 404:
        raise FetchError(f"Chess.com has no user called {user!r}")
    index.raise_for_status()
    archives: List[str] = index.json().get("archives", [])
    if not archives:
        raise FetchError(f"{user!r} has no published games on Chess.com")

    collected: List[str] = []
    for archive_url in reversed(archives):
        if len(collected) >= max_games:
            break
        if progress:
            progress(archive_url, len(collected))
        response = session.get(archive_url, timeout=120)
        if response.status_code == 429:
            time.sleep(5)
            response = session.get(archive_url, timeout=120)
        response.raise_for_status()
        games = response.json().get("games", [])
        # Newest first within the month, so "recent games" means recent.
        for game in reversed(games):
            if len(collected) >= max_games:
                break
            if game.get("time_class") not in wanted:
                continue
            if rated_only and not game.get("rated", False):
                continue
            if game.get("rules", "chess") != "chess":
                continue
            pgn = game.get("pgn")
            if pgn:
                collected.append(pgn.strip())
    if not collected:
        raise FetchError(
            f"No games found for {user!r} on Chess.com with those filters "
            f"(time classes: {', '.join(sorted(wanted))}, rated_only={rated_only})"
        )
    return "\n\n".join(collected) + "\n"


def fetch(
    site: str,
    user: str,
    max_games: int = 40,
    time_classes: Iterable[str] = DEFAULT_CLASSES,
    rated_only: bool = True,
    token: Optional[str] = None,
    progress=None,
) -> str:
    site = site.lower()
    if site in ("lichess", "lichess.org"):
        return fetch_lichess(
            user, max_games, time_classes, rated_only, token=token
        )
    if site in ("chesscom", "chess.com", "chess_com"):
        return fetch_chesscom(
            user, max_games, time_classes, rated_only, progress=progress
        )
    raise FetchError(f"Unknown site {site!r}; use 'lichess' or 'chess.com'")
