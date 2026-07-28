#!/bin/bash

# Directory where your files are
DIR="/home/ubuntu/moomoo_scripts/logs"

# Get today's date (format must match your file timestamps)
today=$(date +%Y-%m-%d)

# Save all logs of the service to a file
journalctl -u moomoo_simulate -b 0 --no-pager > "$DIR/$today - moomoo_simulate_journal.txt"
journalctl -u moomoo_real -b 0 --no-pager > "$DIR/$today - moomoo_real_journal.txt"

# Find files with today's date
find "$DIR" -type f -name "*$today*" | while IFS= read -r file; do
    aws s3 cp "$file" s3://moomoos3/logs/
done