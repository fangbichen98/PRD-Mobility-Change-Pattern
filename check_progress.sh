#!/bin/bash
echo "=========================================="
echo "实验进度监控"
echo "=========================================="
echo ""

# 检查进程
if ps aux | grep "python3 run_dual_year_experiment.py" | grep -v grep > /dev/null; then
    RUNTIME=$(ps -p $(pgrep -f "python3 run_dual_year_experiment.py") -o etime= | tr -d ' ')
    echo "✓ 实验正在运行中 (运行时间: $RUNTIME)"
else
    echo "✗ 实验未运行"
fi

echo ""
echo "当前阶段:"
LAST_LINE=$(tail -1 /home/PRD-Mobility-Change-Pattern/experiment_output.log 2>/dev/null)
echo "$LAST_LINE"

echo ""
echo "训练进度:"
EPOCHS=$(grep -c "Epoch [0-9]*/100" /home/PRD-Mobility-Change-Pattern/experiment_output.log 2>/dev/null)
echo "已完成 Epoch: $EPOCHS / 100"

echo ""
echo "最近的训练指标:"
grep -E "(Train - Loss|Val   - Loss)" /home/PRD-Mobility-Change-Pattern/experiment_output.log 2>/dev/null | tail -4

echo ""
echo "=========================================="
echo "实时监控命令:"
echo "tail -f /home/PRD-Mobility-Change-Pattern/experiment_output.log"
echo "=========================================="
