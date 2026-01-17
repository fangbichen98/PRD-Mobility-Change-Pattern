#!/bin/bash
# Monitor multi-GPU experiment

clear
echo "=========================================="
echo "   多GPU实验监控"
echo "=========================================="
echo ""

# Check process
if ps aux | grep "python3 run_dual_year_experiment.py" | grep -v grep > /dev/null; then
    PID=$(pgrep -f "python3 run_dual_year_experiment.py")
    RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
    echo "✓ 实验正在运行"
    echo "  进程ID: $PID"
    echo "  运行时间: $RUNTIME"
else
    echo "✗ 实验未运行"
fi

echo ""
echo "=========================================="
echo "   GPU使用情况"
echo "=========================================="

# Try to get GPU info
python3 << 'EOF'
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpu_id, name, util, mem_used, mem_total = parts[:5]
                print(f"GPU {gpu_id}: {util}% 利用率, {mem_used}/{mem_total} MB 显存")
    else:
        print("无法获取GPU信息")
except Exception as e:
    print(f"GPU监控失败: {e}")
EOF

echo ""
echo "=========================================="
echo "   训练进度"
echo "=========================================="

# Check progress
LATEST=$(tail -1 /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null)
echo "$LATEST"

# Count epochs
EPOCHS=$(grep -c "Epoch [0-9]*/100" /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null)
if [ "$EPOCHS" -gt 0 ]; then
    echo ""
    echo "已完成 Epoch: $EPOCHS / 100"

    # Show recent metrics
    echo ""
    echo "最近训练指标:"
    grep -E "(Train - Loss|Val   - Loss)" /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log 2>/dev/null | tail -2
fi

echo ""
echo "=========================================="
echo "实时监控: tail -f /home/PRD-Mobility-Change-Pattern/multi_gpu_experiment.log"
echo "=========================================="
