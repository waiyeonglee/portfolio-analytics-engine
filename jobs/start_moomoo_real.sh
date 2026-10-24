#!/bin/bash
set -euo pipefail

# Copy the application and its locked environment configuration to EC2
aws s3 sync "s3://moomoos3/scripts/" "/home/ubuntu/moomoo_scripts/scripts/" \
    --exclude "*" \
    --include "*.py"
aws s3 cp "s3://moomoos3/pyproject.toml" "/home/ubuntu/moomoo_scripts/pyproject.toml"
aws s3 cp "s3://moomoos3/uv.lock" "/home/ubuntu/moomoo_scripts/uv.lock"

# Find and delete log files older than 30 days
# find /home/ubuntu/moomoo_scripts/logs/ -type f -mtime +30 -delete
# Run the application from its uv-managed environment
cd /home/ubuntu/moomoo_scripts
uv sync --locked
uv run --locked python -u -m scripts.main --env "real" --live