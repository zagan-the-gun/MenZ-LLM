#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Assume venv is already activated by user

exec python -m app.main