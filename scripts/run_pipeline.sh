#!/usr/bin/env bash
# Run the full pipeline from the project's virtualenv. Used by cron/launchd.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/smp run-all "$@"
