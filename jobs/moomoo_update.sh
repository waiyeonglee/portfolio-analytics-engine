#!/bin/bash
set -euo pipefail

cd /home/ubuntu/portfolio_analytics_engine
export GIT_TERMINAL_PROMPT=0
git pull --ff-only origin main
/home/ubuntu/.local/bin/uv sync --locked --no-dev
