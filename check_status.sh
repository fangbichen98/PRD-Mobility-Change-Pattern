#!/bin/bash
# 实验状态监控脚本

clear
echo "=========================================="
echo "   实验运行状态监控"
echo "=========================================="
echo ""

# 1. 检查进程状态
echo "【1. 进程状态】"
if ps aux | grep "python3 run_improved_dual_year_experiment.py" | grep -v grep > /dev/null; then
    PID=$(pgrep -f "python3 run_improved_dual_year_experiment.py")
    RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')

    echo "✓ 实验正在运行"
    echo "  进程ID: $PID"
    echo "  运行时间: $RUNTIME"
    echo "  CPU使用: ${CPU}%"
    echo "  内存使用: $MEM"
else
    echo "✗ 实验未运行"
fi

echo ""
echo "【2. 当前阶段】"
# 检查最新日志
LATEST=$(tail -1 /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log 2>/dev/null)

if echo "$LATEST" | grep -q "Processing grids:"; then
    echo "📊 数据预处理 - 聚合网格流量"
    # 提取进度
    python3 << 'EOF'
import re
try:
    with open('/home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log', 'r') as f:
        content = f.read()
        matches = re.findall(r'Processing grids:\s+(\d+)%.*?(\d+)/9977', content)
        if matches:
            last = matches[-1]
            print(f"  进度: {last[0]}% ({last[1]}/9,977)")
except:
    print("  无法获取详细进度")
EOF

elif grep -q "Epoch.*Train" /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log 2>/dev/null; then
    echo "🚀 模型训练阶段"
    EPOCHS=$(grep -c "Epoch [0-9]*/100" /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log 2>/dev/null)
    echo "  已完成 Epoch: $EPOCHS / 100"

    # 显示最近的训练指标
    echo ""
    echo "  最近训练指标:"
    grep -E "(Train - Loss|Val   - Loss)" /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log 2>/dev/null | tail -2 | sed 's/^/    /'

elif grep -q "Loading OD flow data" /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log 2>/dev/null; then
    echo "📥 数据加载阶段"
    tail -1 /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log 2>/dev/null | sed 's/^/  /'
else
    echo "⏳ 初始化阶段"
fi

echo ""
echo "【3. 日志统计】"
LOG_LINES=$(wc -l < /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log 2>/dev/null || echo "0")
LOG_SIZE=$(du -h /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log 2>/dev/null | cut -f1 || echo "0")
echo "  日志行数: $LOG_LINES"
echo "  日志大小: $LOG_SIZE"

echo ""
echo "【4. 缓存状态】"
if [ -d "data/cache" ] && [ "$(ls -A data/cache 2>/dev/null)" ]; then
    CACHE_COUNT=$(ls data/cache/*.pkl 2>/dev/null | wc -l)
    CACHE_SIZE=$(du -sh data/cache 2>/dev/null | cut -f1)
    echo "  缓存文件: $CACHE_COUNT 个"
    echo "  缓存大小: $CACHE_SIZE"

    # 显示缓存信息
    if [ -f data/cache/*_info.txt ]; then
        echo ""
        echo "  最新缓存信息:"
        head -5 data/cache/*_info.txt 2>/dev/null | sed 's/^/    /'
    fi
else
    echo "  缓存: 尚未创建（首次运行）"
fi

echo ""
echo "【5. GPU使用情况】"
python3 << 'EOF'
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines[:1]:  # 只显示GPU 0
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpu_id, util, mem_used, mem_total = parts[:4]
                print(f"  GPU {gpu_id}: {util}% 利用率, {mem_used}/{mem_total} MB 显存")
    else:
        print("  无法获取GPU信息")
except:
    print("  GPU监控不可用")
EOF

echo ""
echo "【6. 输出目录】"
LATEST_OUTPUT=$(ls -td outputs/improved_full_dual_year_2021vs2024_* 2>/dev/null | head -1)
if [ ! -z "$LATEST_OUTPUT" ]; then
    echo "  $LATEST_OUTPUT"

    # 检查是否有结果文件
    if [ -f "$LATEST_OUTPUT/metrics/test_results.json" ]; then
        echo "  ✓ 实验已完成，结果已保存"
    else
        echo "  ⏳ 实验进行中..."
    fi
else
    echo "  尚未创建输出目录"
fi

echo ""
echo "=========================================="
echo "【快捷命令】"
echo "=========================================="
echo "实时查看日志:"
echo "  tail -f /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log"
echo ""
echo "查看最近20行:"
echo "  tail -20 /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log"
echo ""
echo "搜索关键信息:"
echo "  grep -E 'Epoch|Accuracy|F1' improved_experiment_with_cache.log"
echo ""
echo "再次运行此脚本:"
echo "  /home/PRD-Mobility-Change-Pattern/check_status.sh"
echo "=========================================="
