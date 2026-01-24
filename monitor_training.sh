#!/bin/bash
# Comprehensive training monitoring script

echo "================================================================================"
echo "Training Monitor - Real-time Status"
echo "================================================================================"
echo ""

# Check if training is running
if ps aux | grep -q "[t]rain_hierarchical_simple.py"; then
    echo "✓ Training process is RUNNING"
    echo ""

    # Get process info
    echo "Process Information:"
    ps aux | grep "[t]rain_hierarchical_simple.py" | awk '{printf "  PID: %s, CPU: %s%%, Memory: %s%%, Runtime: %s\n", $2, $3, $4, $10}'

    # Calculate elapsed time
    START_TIME="06:43:34"
    CURRENT_TIME=$(date +"%H:%M:%S")
    echo "  Started: $START_TIME"
    echo "  Current: $CURRENT_TIME"

    # Estimate progress
    echo ""
    echo "Progress Estimation:"

    # Check if still in preprocessing
    if grep -q "Aggregating grid flows" training_output.log 2>/dev/null; then
        LAST_LINE=$(tail -1 training_output.log | grep -oP '\d+/10000' | head -1)
        if [ ! -z "$LAST_LINE" ]; then
            CURRENT=$(echo $LAST_LINE | cut -d'/' -f1)
            TOTAL=10000
            PERCENT=$(echo "scale=1; $CURRENT * 100 / $TOTAL" | bc)
            echo "  Stage: Data Preprocessing - Aggregating grids"
            echo "  Progress: $CURRENT/$TOTAL grids ($PERCENT%)"

            REMAINING=$((TOTAL - CURRENT))
            EST_MINUTES=$((REMAINING / 8 / 60))
            echo "  Estimated remaining: ~$EST_MINUTES minutes for this stage"
        fi
    elif grep -q "Building spatial graphs" training_output.log 2>/dev/null; then
        echo "  Stage: Building spatial graphs"
    elif grep -q "Epoch" training_output.log 2>/dev/null; then
        LAST_EPOCH=$(grep -oP "Epoch \d+/\d+" training_output.log | tail -1)
        echo "  Stage: Model Training"
        echo "  Progress: $LAST_EPOCH"

        # Get latest metrics
        LATEST_ACC=$(grep "Val Acc:" training_output.log | tail -1 | grep -oP "Val Acc: \d+\.\d+")
        if [ ! -z "$LATEST_ACC" ]; then
            echo "  Latest: $LATEST_ACC"
        fi
    else
        echo "  Stage: Initializing..."
    fi

    echo ""
    echo "Latest Log Output (last 20 lines):"
    echo "--------------------------------------------------------------------------------"
    tail -20 training_output.log 2>/dev/null | grep -v "Processing grids:" | tail -15
    echo "--------------------------------------------------------------------------------"

else
    echo "✗ Training process NOT RUNNING"
    echo ""

    # Check if completed
    if [ -f training_output.log ]; then
        if grep -q "Training completed" training_output.log 2>/dev/null; then
            echo "✓ Training COMPLETED successfully!"
            echo ""
            echo "Final Results:"
            echo "--------------------------------------------------------------------------------"
            grep -A 30 "Final Test Results" training_output.log 2>/dev/null | head -35
            echo "--------------------------------------------------------------------------------"

            # Show output directory
            OUTPUT_DIR=$(grep "Output directory:" training_output.log | tail -1 | awk '{print $NF}')
            if [ ! -z "$OUTPUT_DIR" ]; then
                echo ""
                echo "Results saved to: $OUTPUT_DIR"
                echo ""
                echo "Available files:"
                ls -lh "$OUTPUT_DIR" 2>/dev/null | tail -n +2
            fi

        elif grep -q "Error\|Exception\|Traceback" training_output.log 2>/dev/null; then
            echo "✗ Training FAILED with errors"
            echo ""
            echo "Error Information:"
            echo "--------------------------------------------------------------------------------"
            grep -A 10 "Error\|Exception\|Traceback" training_output.log | tail -20
            echo "--------------------------------------------------------------------------------"
        else
            echo "⚠ Training stopped unexpectedly"
            echo ""
            echo "Last 30 lines of log:"
            echo "--------------------------------------------------------------------------------"
            tail -30 training_output.log
            echo "--------------------------------------------------------------------------------"
        fi
    else
        echo "No training log found (training_output.log)"
    fi
fi

echo ""
echo "================================================================================"
echo "Commands:"
echo "  Watch live: tail -f training_output.log"
echo "  Check again: bash monitor_training.sh"
echo "  View cache: ls -lht data/cache/"
echo "================================================================================"
