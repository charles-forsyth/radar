#!/bin/bash
# RADAR Daily Intelligence Sweep
# Scheduled to run twice a day

# Project Path
PROJECT_DIR="/home/chuck/Projects/radar"
TOPICS_FILE="$PROJECT_DIR/sweep_targets.txt"
LOG_FILE="$PROJECT_DIR/daily_sweep.log"
UV_BIN="/home/chuck/.local/bin/uv"

# Load environment variables (optional if .env is in the folder)
# export GOOGLE_API_KEY="AIza..." 

cd "$PROJECT_DIR" || exit

echo "--- Sweep started at $(date) ---" >> "$LOG_FILE"

# Run the sweep
# Using uv run directly. No --voice flag to avoid issues with audio in cron.
"$UV_BIN" run radar sweep "$TOPICS_FILE" >> "$LOG_FILE" 2>&1

echo "--- Sweep finished at $(date) ---" >> "$LOG_FILE"
