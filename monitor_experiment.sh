#!/bin/bash
# Monitor experiment progress

EXPERIMENT_DIR="/home/PRD-Mobility-Change-Pattern/outputs/full_dual_year_2021vs2024_20260117_094341"
LOG_FILE="$EXPERIMENT_DIR/logs/experiment.log"

echo "=== Experiment Monitor ==="
echo "Experiment: full_dual_year_2021vs2024_20260117_094341"
echo "Log file: $LOG_FILE"
echo ""

# Check if process is running
if ps aux | grep "python3 run_dual_year_experiment.py" | grep -v grep > /dev/null; then
    echo "✓ Experiment is RUNNING"
    echo ""
else
    echo "✗ Experiment is NOT running"
    echo ""
fi

# Show last 30 lines of log
echo "=== Latest Log (last 30 lines) ==="
tail -30 "$LOG_FILE" 2>/dev/null || echo "Log file not found yet"

echo ""
echo "=== Training Progress ==="
# Count epochs completed
EPOCHS=$(grep -c "Epoch [0-9]*/100" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Epochs completed: $EPOCHS / 100"

# Show latest accuracy if available
LATEST_ACC=$(grep "Val   - Loss:" "$LOG_FILE" 2>/dev/null | tail -1)
if [ ! -z "$LATEST_ACC" ]; then
    echo "Latest validation: $LATEST_ACC"
fi

echo ""
echo "=== To monitor in real-time, run: ==="
echo "tail -f $LOG_FILE"
