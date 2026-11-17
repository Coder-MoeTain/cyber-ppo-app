#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
PORT="${PORT:-8787}"
HOST="127.0.0.1"

if ! "$PYTHON" -c 'import fastapi, uvicorn, torch' >/dev/null 2>&1; then
  echo "Dashboard dependencies missing. Install dashboard/requirements-server.txt in this environment." >&2
  exit 2
fi

if [[ -z "${SENTINEL_DASHBOARD_TOKEN:-}" ]]; then
  TOKEN_FILE="$ROOT/.dashboard_token"
  if [[ ! -s "$TOKEN_FILE" ]]; then
    umask 077
    "$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(36))' > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
    echo "New dashboard access token (save it securely; shown only once):"
    cat "$TOKEN_FILE"
  else
    echo "Using existing dashboard token from $TOKEN_FILE"
  fi
  export SENTINEL_DASHBOARD_TOKEN="$(cat "$TOKEN_FILE")"
fi

export PYTHONPATH="$ROOT/src:$ROOT/vendor/CybORG:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
echo "Starting Sentinel MARL GPU dashboard at http://$HOST:$PORT"
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]]; then
  echo "WARNING: Non-loopback binding. Put this behind HTTPS, firewall and a trusted reverse proxy." >&2
fi
exec "$PYTHON" -m uvicorn dashboard.server:app --host "$HOST" --port "$PORT"
