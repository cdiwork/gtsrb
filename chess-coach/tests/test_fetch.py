"""Tests for the download layer.

These use a fake HTTP session rather than the real APIs: the point is that
the URL, the filters and the error messages are right, and none of that
needs the network (which CI generally will not have anyway).
"""
import json

import pytest

from chess_coach import fetch as fetch_module
from chess_coach.fetch import FetchError, fetch, fetch_chesscom, fetch_lichess


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected HTTP {self.status_code}")


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        for pattern, response in self.routes.items():
            if pattern in url:
                return response
        return FakeResponse(404, {}, "")


def _install(monkeypatch, session):
    monkeypatch.setattr(fetch_module, "_session", lambda: session)
    return session


def _game(time_class="blitz", rated=True, rules="chess", pgn="[Event \"x\"]\n1. e4 *"):
    return {"time_class": time_class, "rated": rated, "rules": rules, "pgn": pgn}


# --- Chess.com ------------------------------------------------------------


def test_chesscom_filters_by_time_class_rated_and_variant(monkeypatch):
    session = _install(monkeypatch, FakeSession({
        "games/archives": FakeResponse(200, {"archives": ["https://x/2026/08"]}),
        "2026/08": FakeResponse(200, {"games": [
            _game(pgn="KEEP blitz rated"),
            _game(time_class="bullet", pgn="DROP bullet"),
            _game(rated=False, pgn="DROP unrated"),
            _game(rules="chess960", pgn="DROP variant"),
        ]}),
    }))
    result = fetch_chesscom("retsekoj", max_games=10)
    assert "KEEP blitz rated" in result
    assert "DROP" not in result


def test_chesscom_respects_the_game_limit_and_walks_months_backwards(monkeypatch):
    _install(monkeypatch, FakeSession({
        "games/archives": FakeResponse(200, {"archives": [
            "https://x/2026/07", "https://x/2026/08",
        ]}),
        "2026/08": FakeResponse(200, {"games": [
            _game(pgn='[Event "a"] newest-1'), _game(pgn='[Event "b"] newest-2'),
        ]}),
        "2026/07": FakeResponse(200, {"games": [_game(pgn='[Event "c"] older')]}),
    }))
    result = fetch_chesscom("retsekoj", max_games=2)
    # The most recent month is consulted first and the limit stops us there.
    assert "older" not in result
    assert result.count("[Event") == 2


def test_chesscom_unknown_user_is_a_clear_error(monkeypatch):
    _install(monkeypatch, FakeSession({
        "games/archives": FakeResponse(404, {}),
    }))
    with pytest.raises(FetchError, match="no user called"):
        fetch_chesscom("definitely-not-a-user")


def test_chesscom_no_matching_games_explains_the_filters(monkeypatch):
    _install(monkeypatch, FakeSession({
        "games/archives": FakeResponse(200, {"archives": ["https://x/2026/08"]}),
        "2026/08": FakeResponse(200, {"games": [_game(time_class="bullet")]}),
    }))
    with pytest.raises(FetchError, match="time classes"):
        fetch_chesscom("retsekoj")


# --- Lichess --------------------------------------------------------------


def test_lichess_asks_for_clocks_and_openings(monkeypatch):
    session = _install(monkeypatch, FakeSession({
        "lichess.org/api/games/user": FakeResponse(200, None, "[Event \"x\"]\n1. e4 *"),
    }))
    fetch_lichess("someone", max_games=5, time_classes=["blitz", "rapid"])
    call = session.calls[0]
    assert call["url"].endswith("/someone")
    assert call["params"]["clocks"] == "true"      # time analysis needs these
    assert call["params"]["opening"] == "true"
    assert call["params"]["max"] == 5
    assert call["params"]["perfType"] == "blitz,rapid"
    assert call["params"]["rated"] == "true"
    assert call["headers"]["Accept"] == "application/x-chess-pgn"


def test_lichess_token_is_sent_as_a_bearer(monkeypatch):
    session = _install(monkeypatch, FakeSession({
        "lichess.org": FakeResponse(200, None, "[Event \"x\"]"),
    }))
    fetch_lichess("someone", token="secret")
    assert session.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_lichess_unknown_user_and_rate_limit_are_named(monkeypatch):
    _install(monkeypatch, FakeSession({"lichess.org": FakeResponse(404, None, "")}))
    with pytest.raises(FetchError, match="no user"):
        fetch_lichess("nobody")
    _install(monkeypatch, FakeSession({"lichess.org": FakeResponse(429, None, "")}))
    with pytest.raises(FetchError, match="rate limit"):
        fetch_lichess("someone")


def test_lichess_empty_response_is_not_silently_accepted(monkeypatch):
    _install(monkeypatch, FakeSession({"lichess.org": FakeResponse(200, None, "  ")}))
    with pytest.raises(FetchError, match="No games found"):
        fetch_lichess("someone")


# --- dispatch -------------------------------------------------------------


@pytest.mark.parametrize("name", ["lichess", "Lichess", "lichess.org"])
def test_fetch_dispatches_lichess_aliases(monkeypatch, name):
    _install(monkeypatch, FakeSession({"lichess.org": FakeResponse(200, None, "[Event")}))
    assert fetch(name, "someone")


@pytest.mark.parametrize("name", ["chess.com", "chesscom", "CHESS.COM"])
def test_fetch_dispatches_chesscom_aliases(monkeypatch, name):
    _install(monkeypatch, FakeSession({
        "games/archives": FakeResponse(200, {"archives": ["https://x/2026/08"]}),
        "2026/08": FakeResponse(200, {"games": [_game()]}),
    }))
    assert fetch(name, "someone")


def test_fetch_rejects_an_unknown_site():
    with pytest.raises(FetchError, match="Unknown site"):
        fetch("playchess", "someone")
