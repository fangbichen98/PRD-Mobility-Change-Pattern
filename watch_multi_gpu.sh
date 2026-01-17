#!/bin/bash
# 实时监控多GPU实验进度

echo "=========================================="
echo "   多GPU实验实时监控"
echo "=========================================="
echo ""
echo "实验名称: multi_gpu_dual_year_2021vs2024"
echo "开始时间: 2026-01-17 11:27:24"
echo ""

# 检查进程
if ps aux | grep "python3 run_dual_year_experiment.py" | grep -v grep > /dev/null; then
    PID=$(pgrep -f "python3 run_dual_year_experiment.py")
    RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')

    echo "✓ 实验正在运行"
    echo "  进程ID: $PID"
    echo "  运行时间: $RUNTIME"
    echo "  CPU使用: ${CPU}%"
    echo "  内存使用: $MEM"
else
    echo "✗ 实验已完成或未运行"
fi

echo ""
echo "=========================================="
echo "   当前阶段"
echo "=========================================="

# 检查是否在训练阶段
if grep -q "Epoch 1/100" /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null; then
    echo "🚀 模型训练阶段"

    # 统计完成的epoch
    EPOCHS=$(grep -c "Epoch [0-9]*/100" /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null)
    echo "  已完成 Epoch: $EPOCHS / 100"

    # 显示最近的训练指标
    echo ""
    echo "  最近训练指标:"
    grep -E "(Train - Loss|Val   - Loss)" /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null | tail -2 | sed 's/^/    /'

elif grep -q "Processing grids:" /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null; then
    echo "📊 数据预处理 - 聚合网格流量"

    # 提取最新进度
    python3 << 'PYEOF'
import re
try:
    with open('/home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log', 'r') as f:
        content = f.read()
        matches = re.findall(r'Processing grids:\s+(\d+)%.*?(\d+)/9977.*?\[([\d:]+)<([\d:]+),\s*([\d.]+)it/s\]', content)
        if matches:
            last = matches[-1]
            percent, current, elapsed, remaining, speed = last
            print(f"  进度: {percent}% ({current}/9,977)")
            print(f"  已用时间: {elapsed}")
            print(f"  预计剩余: {remaining}")
            print(f"  处理速度: {speed} 网格/秒")
except Exception as e:
    print(f"  无法获取详细进度: {e}")
PYEOF

else
    echo "📥 数据加载阶段"
    tail -1 /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null
fi

echo ""
echo "=========================================="
echo "   多GPU配置"
echo "=========================================="

# 检查是否已配置多GPU
if grep -q "Available GPUs: 8" /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null; then
    echo "✓ 检测到 8个 NVIDIA A100-SXM4-40GB GPU"

    if grep -q "Wrapping model with DataParallel" /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null; then
        echo "✓ 模型已用DataParallel包装"
        BATCH_INFO=$(grep "Adjusted batch size" /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null | tail -1)
        if [ ! -z "$BATCH_INFO" ]; then
            echo "✓ $BATCH_INFO"
        fi
    else
        echo "⏳ 等待模型初始化..."
    fi
else
    echo "⏳ 等待GPU配置..."
fi

echo ""
echo "=========================================="
echo "   输出目录"
echo "=========================================="
LATEST_DIR=$(ls -td /home/PRD-Mobility-Change-Pattern/outputs/multi_gpu_dual_year_2021vs2024_* 2>/dev/null | head -1)
if [ ! -z "$LATEST_DIR" ]; then
    echo "$LATEST_DIR"
else
    echo "尚未创建输出目录"
fi

echo ""
echo "=========================================="
echo "实时日志: tail -f /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log"
echo "=========================================="
