#!/bin/bash
# 实时显示实验进度

clear
echo "=========================================="
echo "   实验进度实时监控"
echo "=========================================="
echo ""

# 检查进程状态
if ps aux | grep "python3 run_dual_year_experiment.py" | grep -v grep > /dev/null; then
    PID=$(pgrep -f "python3 run_dual_year_experiment.py")
    RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o %mem= | tr -d ' ')

    echo "✓ 实验正在运行"
    echo "  进程ID: $PID"
    echo "  运行时间: $RUNTIME"
    echo "  CPU使用: ${CPU}%"
    echo "  内存使用: ${MEM}%"
else
    echo "✗ 实验未运行"
    exit 1
fi

echo ""
echo "=========================================="
echo "   当前阶段"
echo "=========================================="

# 获取最新的进度条
LATEST_PROGRESS=$(tail -1 /home/PRD-Mobility-Change-Pattern/experiment_output.log 2>/dev/null)

if echo "$LATEST_PROGRESS" | grep -q "Processing grids:"; then
    echo "📊 数据预处理阶段 - 聚合网格流量"
    echo "$LATEST_PROGRESS"

    # 提取进度百分比
    PERCENT=$(echo "$LATEST_PROGRESS" | grep -oP '\d+%' | head -1)
    CURRENT=$(echo "$LATEST_PROGRESS" | grep -oP '\d+/9977' | cut -d'/' -f1)
    SPEED=$(echo "$LATEST_PROGRESS" | grep -oP '\d+\.\d+it/s')

    echo ""
    echo "  完成: $CURRENT / 9,977 网格 ($PERCENT)"
    echo "  速度: $SPEED"

    # 估算剩余时间
    if [ ! -z "$SPEED" ]; then
        REMAINING=$((9977 - CURRENT))
        SPEED_NUM=$(echo $SPEED | grep -oP '\d+\.\d+')
        TIME_SEC=$(echo "scale=0; $REMAINING / $SPEED_NUM" | bc)
        TIME_MIN=$((TIME_SEC / 60))
        echo "  预计剩余: ~${TIME_MIN} 分钟"
    fi

elif echo "$LATEST_PROGRESS" | grep -q "Epoch"; then
    echo "🚀 模型训练阶段"

    # 统计已完成的epoch
    EPOCHS=$(grep -c "Epoch [0-9]*/100" /home/PRD-Mobility-Change-Pattern/experiment_output.log 2>/dev/null)
    echo "  已完成 Epoch: $EPOCHS / 100"

    # 显示最近的训练指标
    echo ""
    echo "  最近训练指标:"
    grep -E "(Train - Loss|Val   - Loss)" /home/PRD-Mobility-Change-Pattern/experiment_output.log 2>/dev/null | tail -2 | sed 's/^/    /'

else
    echo "📥 数据加载阶段"
    echo "$LATEST_PROGRESS"
fi

echo ""
echo "=========================================="
echo "   监控命令"
echo "=========================================="
echo "实时查看日志:"
echo "  tail -f /home/PRD-Mobility-Change-Pattern/experiment_output.log"
echo ""
echo "再次查看进度:"
echo "  /home/PRD-Mobility-Change-Pattern/show_progress.sh"
echo "=========================================="
