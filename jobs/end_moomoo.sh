#!/bin/bash
set -euo pipefail

ENV="$1"

case "$ENV" in
    real|simulate) ;;
    *)
        echo "Usage: $0 {real|simulate}" >&2
        exit 2
        ;;
esac

DIR="/home/ubuntu/portfolio_analytics_engine/logs"
today=$(date +%Y-%m-%d)

SERVICE="moomoo_${ENV}"

mkdir -p "$DIR"

# Export only this service's journal
journalctl -u "$SERVICE.service" -b 0 --no-pager \
    > "$DIR/$today - ${SERVICE}_journal.txt"

# Upload only this environment's logs
find "$DIR" -maxdepth 1 -type f \
    -name "*$today*${ENV}*" \
    -print0 |
while IFS= read -r -d '' file; do
    aws s3 cp "$file" s3://moomoos3/logs/
done