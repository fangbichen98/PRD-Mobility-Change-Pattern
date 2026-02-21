#!/bin/bash
echo "监控训练进度..."
echo ""

# 等待数据加载完成
echo "等待数据加载完成..."
while ! grep -q "Step 2: Creating model" train_labels1.log 2>/dev/null; do
    sleep 10
    echo -n "."
done
echo ""
echo "✓ 数据加载完成"

# 等待训练完成
echo ""
echo "等待训练完成..."
while ! grep -q "Test Results:" train_labels1.log 2>/dev/null; do
    sleep 30
    current_epoch=$(grep "Epoch [0-9]*/100" train_labels1.log 2>/dev/null | tail -1)
    if [ -n "$current_epoch" ]; then
        echo "当前进度: $current_epoch"
    fi
done

echo ""
echo "✓ 训练完成!"
echo ""
echo "=== 最终结果 ==="
tail -30 train_labels1.log
