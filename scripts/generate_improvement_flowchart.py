#!/usr/bin/env python3
"""
生成改进流程图 - 可视化整个改进过程

Usage:
    python scripts/generate_improvement_flowchart.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_flowchart():
    """创建改进流程图"""
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # 标题
    ax.text(5, 13.5, '标注质量改进流程图', fontsize=24, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2))

    # ============ 第一部分：问题诊断 ============
    y_start = 12.5

    # 问题诊断标题
    ax.text(1, y_start, '1. 问题诊断', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', edgecolor='black'))

    # 问题列表
    problems = [
        '整体标注质量：40.2%',
        '方向维度错误率：54.4%',
        'L5 错误率：83.2%',
        'L9 错误率：86.7%'
    ]

    y_pos = y_start - 0.5
    for i, problem in enumerate(problems):
        y_pos -= 0.4
        ax.text(1, y_pos, f'• {problem}', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', linewidth=1))

    # 根本原因
    y_pos -= 0.6
    ax.text(1, y_pos, '根本原因:', fontsize=12, fontweight='bold')

    causes = [
        '过度依赖 Shannon 熵（3% 阈值）',
        '缺乏流量集中度测量',
        '单一特征决策'
    ]

    for cause in causes:
        y_pos -= 0.35
        ax.text(1.2, y_pos, f'→ {cause}', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffe6e6', edgecolor='gray'))

    # 箭头：问题诊断 → 改进方案
    arrow1 = FancyArrowPatch((2.5, y_pos - 0.3), (2.5, y_pos - 0.8),
                             arrowstyle='->', mutation_scale=30, linewidth=3, color='blue')
    ax.add_patch(arrow1)

    # ============ 第二部分：改进方案 ============
    y_start = y_pos - 1.2

    # 改进方案标题
    ax.text(1, y_start, '2. 改进方案', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc', edgecolor='black'))

    # 改进1：新增特征
    y_pos = y_start - 0.5
    ax.text(1, y_pos, '改进 1: 新增图结构特征', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e6ffe6', edgecolor='green', linewidth=2))

    features = [
        '加权度比例（Top-3 流量占比）',
        'Gini 系数（流量集中度）'
    ]

    for feature in features:
        y_pos -= 0.35
        ax.text(1.2, y_pos, f'✓ {feature}', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='green'))

    # 改进2：多特征融合
    y_pos -= 0.5
    ax.text(1, y_pos, '改进 2: 多特征融合决策', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e6ffe6', edgecolor='green', linewidth=2))

    voting = [
        '熵变化（权重 ×1）',
        '目的地变化（权重 ×1）',
        '加权比例（权重 ×2）',
        'Gini 系数（权重 ×2）'
    ]

    for vote in voting:
        y_pos -= 0.35
        ax.text(1.2, y_pos, f'→ {vote}', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='green'))

    # 改进3：调整阈值
    y_pos -= 0.5
    ax.text(1, y_pos, '改进 3: 调整阈值', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e6ffe6', edgecolor='green', linewidth=2))

    thresholds = [
        '熵阈值：3% → 10%',
        '目的地阈值：30% → 20%',
        '置信度阈值：15% → 20%'
    ]

    for threshold in thresholds:
        y_pos -= 0.35
        ax.text(1.2, y_pos, f'• {threshold}', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='green'))

    # 箭头：改进方案 → 实现
    arrow2 = FancyArrowPatch((2.5, y_pos - 0.3), (2.5, y_pos - 0.8),
                             arrowstyle='->', mutation_scale=30, linewidth=3, color='blue')
    ax.add_patch(arrow2)

    # ============ 第三部分：代码实现 ============
    y_start = y_pos - 1.2

    # 代码实现标题
    ax.text(1, y_start, '3. 代码实现', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccccff', edgecolor='black'))

    # 核心脚本
    y_pos = y_start - 0.5
    scripts = [
        'auto_label_sgh_improved.py（改进版标注）',
        'test_improved_labeler.py（单元测试）',
        'compare_labels.py（标注对比）',
        'visualize_improvement.py（可视化）'
    ]

    for script in scripts:
        y_pos -= 0.35
        ax.text(1.2, y_pos, f'✓ {script}', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='blue'))

    # 测试结果
    y_pos -= 0.5
    ax.text(1, y_pos, '✅ 单元测试：全部通过', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e6f3ff', edgecolor='blue', linewidth=2))

    # ============ 右侧：执行流程 ============
    x_right = 6

    # 执行流程标题
    ax.text(x_right, 12.5, '执行流程', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='black'))

    # 步骤1
    y_pos = 11.8
    ax.text(x_right, y_pos, '步骤 1: 运行改进版标注', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', edgecolor='orange', linewidth=2))

    y_pos -= 0.4
    ax.text(x_right, y_pos, './标注/quick_start.sh', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

    y_pos -= 0.3
    ax.text(x_right, y_pos, '⏱ 预期时间：10-20 分钟', fontsize=9, style='italic')

    # 箭头
    arrow3 = FancyArrowPatch((x_right, y_pos - 0.2), (x_right, y_pos - 0.5),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='orange')
    ax.add_patch(arrow3)

    # 步骤2
    y_pos -= 0.8
    ax.text(x_right, y_pos, '步骤 2: 对比原始和改进标注', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', edgecolor='orange', linewidth=2))

    y_pos -= 0.4
    ax.text(x_right, y_pos, 'compare_labels.py', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

    y_pos -= 0.3
    ax.text(x_right, y_pos, '📊 查看标签变化统计', fontsize=9, style='italic')

    # 箭头
    arrow4 = FancyArrowPatch((x_right, y_pos - 0.2), (x_right, y_pos - 0.5),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='orange')
    ax.add_patch(arrow4)

    # 步骤3
    y_pos -= 0.8
    ax.text(x_right, y_pos, '步骤 3: 可视化改进效果', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', edgecolor='orange', linewidth=2))

    y_pos -= 0.4
    ax.text(x_right, y_pos, 'visualize_improvement.py', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

    y_pos -= 0.3
    ax.text(x_right, y_pos, '📈 生成对比图表', fontsize=9, style='italic')

    # 箭头
    arrow5 = FancyArrowPatch((x_right, y_pos - 0.2), (x_right, y_pos - 0.5),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='orange')
    ax.add_patch(arrow5)

    # 步骤4
    y_pos -= 0.8
    ax.text(x_right, y_pos, '步骤 4: 重新训练模型', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', edgecolor='orange', linewidth=2))

    y_pos -= 0.4
    ax.text(x_right, y_pos, 'train_multiscale_temporal.py', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

    y_pos -= 0.3
    ax.text(x_right, y_pos, '🎯 使用改进标注训练', fontsize=9, style='italic')

    # 箭头
    arrow6 = FancyArrowPatch((x_right, y_pos - 0.2), (x_right, y_pos - 0.5),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='orange')
    ax.add_patch(arrow6)

    # 步骤5
    y_pos -= 0.8
    ax.text(x_right, y_pos, '步骤 5: 评估模型性能', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9e6', edgecolor='orange', linewidth=2))

    y_pos -= 0.4
    ax.text(x_right, y_pos, '对比验证准确率和混淆矩阵', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))

    y_pos -= 0.3
    ax.text(x_right, y_pos, '✅ 验证改进效果', fontsize=9, style='italic')

    # ============ 底部：预期效果 ============
    y_bottom = 1.5

    # 预期效果标题
    ax.text(5, y_bottom + 0.5, '预期改进效果', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffff', edgecolor='black'))

    # 效果对比表格
    metrics = [
        ('标注质量', '40.2%', '60-70%', '+20-30%'),
        ('方向错误率', '54.4%', '<30%', '-24%'),
        ('L5 准确率', '~17%', '>60%', '+43%'),
        ('L9 准确率', '~13%', '>60%', '+47%'),
        ('模型准确率', '72.71%', '75-80%', '+2-7%')
    ]

    # 表头
    y_pos = y_bottom
    headers = ['指标', '改进前', '改进后', '提升']
    x_positions = [2, 4, 6, 8]

    for i, header in enumerate(headers):
        ax.text(x_positions[i], y_pos, header, fontsize=11, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', edgecolor='black'))

    # 数据行
    for metric, before, after, improvement in metrics:
        y_pos -= 0.35
        ax.text(x_positions[0], y_pos, metric, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='gray'))
        ax.text(x_positions[1], y_pos, before, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffcccc', edgecolor='gray'))
        ax.text(x_positions[2], y_pos, after, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#ccffcc', edgecolor='gray'))
        ax.text(x_positions[3], y_pos, improvement, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#ccffff', edgecolor='gray'))

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='#ffcccc', edgecolor='black', label='问题诊断'),
        mpatches.Patch(facecolor='#ccffcc', edgecolor='black', label='改进方案'),
        mpatches.Patch(facecolor='#ccccff', edgecolor='black', label='代码实现'),
        mpatches.Patch(facecolor='#ffffcc', edgecolor='black', label='执行流程'),
        mpatches.Patch(facecolor='#ccffff', edgecolor='black', label='预期效果')
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    return fig


def main():
    print("="*80)
    print("生成改进流程图")
    print("="*80)

    print("\n生成流程图...")
    fig = create_flowchart()

    output_file = 'output_analysis/improvement_flowchart.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 保存流程图: {output_file}")

    print("\n" + "="*80)
    print("✅ 流程图生成完成！")
    print("="*80)
    print(f"\n输出文件: {output_file}")

    plt.close()


if __name__ == "__main__":
    main()
