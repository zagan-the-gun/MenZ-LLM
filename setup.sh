#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Assume virtualenv is created and activated manually by the user
#   python -m venv .venv
#   source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

:

echo "Setup complete. Edit config.ini as needed."