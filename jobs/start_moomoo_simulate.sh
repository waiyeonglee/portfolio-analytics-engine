#!/bin/bash

# Copy latest scripts to EC2
aws s3 sync "s3://moomoos3/scripts/" "/home/ubuntu/moomoo_scripts/scripts/" \
    --exclude "*" \
    --include "*.py"

# Find and delete log files older than 30 days
# find /home/ubuntu/moomoo_scripts/logs/ -type f -mtime +30 -delete
# Run main.py
/home/ubuntu/moomoo/bin/python3 -u /home/ubuntu/moomoo_scripts/main.py --live