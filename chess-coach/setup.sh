#!/usr/bin/env bash
# One-time setup: a virtualenv, the Python deps, and a check for Stockfish.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip setuptools wheel
./.venv/bin/pip install --quiet -e ".[dev]"

echo "Python environment ready."

if command -v stockfish >/dev/null 2>&1; then
  echo "Stockfish found: $(command -v stockfish)"
elif [ -x /usr/games/stockfish ]; then
  echo "Stockfish found: /usr/games/stockfish"
else
  cat <<'MSG'

Stockfish is NOT installed. Install it with one of:

  Debian/Ubuntu   sudo apt install stockfish
  macOS           brew install stockfish
  Windows/other   https://stockfishchess.org/download/

If you put it somewhere unusual, point at it with:
  export CHESS_COACH_STOCKFISH=/path/to/stockfish
MSG
fi

echo
echo "Try it:  ./.venv/bin/chess-coach coach --site chess.com --user YOUR_HANDLE --max 20"
