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

# Export and upload the boot updater journal
journalctl -u moomoo_update.service -b 0 --no-pager \
    > "$DIR/$today - moomoo_update_journal.txt"
aws s3 cp "$DIR/$today - moomoo_update_journal.txt" s3://moomoos3/logs/

# Upload only this environment's logs
find "$DIR" -maxdepth 1 -type f \
    -name "*$today*${ENV}*" \
    -print0 |
while IFS= read -r -d '' file; do
    aws s3 cp "$file" s3://moomoos3/logs/
done