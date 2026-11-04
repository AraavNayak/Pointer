#!/usr/bin/env bash
# Set up (first run only) and start Pointer at http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -r requirements.txt
fi
[ -f .env ] || { cp .env.example .env; echo "Created .env. Add your MODEL_API_KEY to it, then run ./run.sh again."; exit 1; }

echo "Pointer running at http://localhost:8000"
exec .venv/bin/python -m uvicorn server.app:app --port 8000
