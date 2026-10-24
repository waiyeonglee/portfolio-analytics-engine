#!/bin/bash
set -euo pipefail

cd /home/ubuntu/moomoo_scripts
exec /home/ubuntu/.local/bin/uv run --locked --no-dev python -u -m scripts.main --env "simulate" --live
