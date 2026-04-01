#!/usr/bin/env python3
"""
可视化改进效果 - 对比原始标注和改进标注的特征分布

Usage:
    python scripts/visualize_improvement.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 标签映射
LABEL_MAP = {
    1: "稳定静态", 2: "稳定聚集", 3: "稳定扩散",
    4: "增长静态", 5: "增长聚集", 6: "增长扩散",
    7: "衰减静态", 8: "衰减聚集", 9: "衰减扩散"
}


def load_detailed_labels(filepath):
    """加载详细标注文件"""
    df = pd.read_csv(filepath)
    print(f"  ✓ 加载了 {len(df):,} 个标注")
    return df


def visualize_feature_comparison(old_detailed, new_detailed, output_dir):
    """可视化特征对比"""
    print("\n生成特征对比可视化...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 合并数据
    merged = pd.merge(
        old_detailed[['grid_id', 'label', 'entropy_change_ratio', 'dest_change_ratio']],
        new_detailed[['grid_id', 'label', 'entropy_change', 'dest_change',
                      'weighted_ratio_change', 'gini_change']],
        on='grid_id',
        suffixes=('_old', '_new')
    )

    # 创建图表
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('原始标注 vs 改进标注 - 特征分布对比', fontsize=18, fontweight='bold')

    # 1. 标签分布对比（饼图）
    ax1 = axes[0, 0]
    old_dist = old_detailed['label'].value_counts().sort_index()
    new_dist = new_detailed['label'].value_counts().sort_index()

    labels_old = [f"L{i}\n{LABEL_MAP[i]}" for i in old_dist.index]
    ax1.pie(old_dist.values, labels=labels_old, autopct='%1.1f%%', startangle=90)
    ax1.set_title('原始标注分布', fontsize=14, fontweight='bold')

    ax2 = axes[0, 1]
    labels_new = [f"L{i}\n{LABEL_MAP[i]}" for i in new_dist.index]
    ax2.pie(new_dist.values, labels=labels_new, autopct='%1.1f%%', startangle=90)
    ax2.set_title('改进标注分布', fontsize=14, fontweight='bold')

    # 2. 标签变化统计
    ax3 = axes[0, 2]
    changed = merged[merged['label_old'] != merged['label_new']]
    unchanged = merged[merged['label_old'] == merged['label_new']]

    ax3.bar(['未改变', '已改变'], [len(unchanged), len(changed)],
            color=['#2ecc71', '#e74c3c'], alpha=0.8)
    ax3.set_ylabel('样本数', fontsize=12)
    ax3.set_title(f'标签变化统计\n(改变: {len(changed):,} / {len(merged):,})',
                  fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, v in enumerate([len(unchanged), len(changed)]):
        ax3.text(i, v + max(len(unchanged), len(changed))*0.02,
                f'{v:,}\n({v/len(merged)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    # 3. 熵变化分布对比
    ax4 = axes[1, 0]

    # 原始标注的熵变化
    old_entropy = old_detailed['entropy_change_ratio'].dropna()
    ax4.hist(old_entropy, bins=50, alpha=0.6, label='原始标注', color='blue', edgecolor='black')
    ax4.axvline(0.03, color='red', linestyle='--', linewidth=2, label='原阈值 (3%)')
    ax4.axvline(-0.03, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('熵变化率', fontsize=12)
    ax4.set_ylabel('样本数', fontsize=12)
    ax4.set_title('原始标注 - 熵变化分布\n(阈值: 3%)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 改进标注的熵变化
    ax5 = axes[1, 1]
    new_entropy = new_detailed['entropy_change'].dropna()
    ax5.hist(new_entropy, bins=50, alpha=0.6, label='改进标注', color='green', edgecolor='black')
    ax5.axvline(0.10, color='red', linestyle='--', linewidth=2, label='新阈值 (10%)')
    ax5.axvline(-0.10, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('熵变化率', fontsize=12)
    ax5.set_ylabel('样本数', fontsize=12)
    ax5.set_title('改进标注 - 熵变化分布\n(阈值: 10%)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 4. 新增特征：加权比例变化
    ax6 = axes[1, 2]
    weighted_ratio = new_detailed['weighted_ratio_change'].dropna()
    ax6.hist(weighted_ratio, bins=50, alpha=0.6, color='purple', edgecolor='black')
    ax6.axvline(0.15, color='red', linestyle='--', linewidth=2, label='阈值 (15%)')
    ax6.axvline(-0.15, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('加权比例变化率', fontsize=12)
    ax6.set_ylabel('样本数', fontsize=12)
    ax6.set_title('新增特征 - 加权比例变化\n(Top-3流量占比)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)

    # 5. 新增特征：Gini系数变化
    ax7 = axes[2, 0]
    gini_change = new_detailed['gini_change'].dropna()
    ax7.hist(gini_change, bins=50, alpha=0.6, color='orange', edgecolor='black')
    ax7.axvline(0.10, color='red', linestyle='--', linewidth=2, label='阈值 (10%)')
    ax7.axvline(-0.10, color='red', linestyle='--', linewidth=2)
    ax7.set_xlabel('Gini系数变化率', fontsize=12)
    ax7.set_ylabel('样本数', fontsize=12)
    ax7.set_title('新增特征 - Gini系数变化\n(流量集中度)', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3)

    # 6. 关键类别对比：L5
    ax8 = axes[2, 1]

    # L5在原始标注和改进标注中的数量
    l5_old = len(old_detailed[old_detailed['label'] == 5])
    l5_new = len(new_detailed[new_detailed['label'] == 5])

    # L5→L4的转换数量
    l5_to_l4 = len(changed[(changed['label_old'] == 5) & (changed['label_new'] == 4)])

    categories = ['原始标注\nL5数量', '改进标注\nL5数量', 'L5→L4\n转换数量']
    values = [l5_old, l5_new, l5_to_l4]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax8.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax8.set_ylabel('样本数', fontsize=12)
    ax8.set_title('L5（增长聚集型）修正效果', fontsize=14, fontweight='bold')
    ax8.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 7. 关键类别对比：L9
    ax9 = axes[2, 2]

    # L9在原始标注和改进标注中的数量
    l9_old = len(old_detailed[old_detailed['label'] == 9])
    l9_new = len(new_detailed[new_detailed['label'] == 9])

    # L9→L7的转换数量
    l9_to_l7 = len(changed[(changed['label_old'] == 9) & (changed['label_new'] == 7)])

    categories = ['原始标注\nL9数量', '改进标注\nL9数量', 'L9→L7\n转换数量']
    values = [l9_old, l9_new, l9_to_l7]
    colors = ['#9b59b6', '#1abc9c', '#e67e22']

    bars = ax9.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax9.set_ylabel('样本数', fontsize=12)
    ax9.set_title('L9（衰减扩散型）修正效果', fontsize=14, fontweight='bold')
    ax9.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()

    output_file = output_dir / 'improvement_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存可视化图: {output_file}")

    plt.close()


def visualize_voting_analysis(new_detailed, output_dir):
    """可视化投票分析"""
    print("\n生成投票分析可视化...")

    output_dir = Path(output_dir)

    # 解析投票结果（从字符串转换为字典）
    # 注意：votes列可能是字符串格式，需要解析

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('多特征投票分析', fontsize=18, fontweight='bold')

    # 1. 置信度分布
    ax1 = axes[0, 0]
    if 'overall_conf' in new_detailed.columns:
        conf = new_detailed['overall_conf'].dropna()
        ax1.hist(conf, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(0.20, color='red', linestyle='--', linewidth=2, label='置信度阈值 (0.20)')
        ax1.set_xlabel('置信度', fontsize=12)
        ax1.set_ylabel('样本数', fontsize=12)
        ax1.set_title('置信度分布', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 统计
        high_conf = len(conf[conf >= 0.20])
        low_conf = len(conf[conf < 0.20])
        ax1.text(0.5, 0.95, f'高置信: {high_conf:,} ({high_conf/len(conf)*100:.1f}%)\n低置信: {low_conf:,} ({low_conf/len(conf)*100:.1f}%)',
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=11, fontweight='bold')

    # 2. 模糊样本分布
    ax2 = axes[0, 1]
    if 'is_ambiguous' in new_detailed.columns:
        ambiguous_counts = new_detailed['is_ambiguous'].value_counts()
        labels = ['清晰', '模糊']
        values = [ambiguous_counts.get(False, 0), ambiguous_counts.get(True, 0)]
        colors = ['#2ecc71', '#e74c3c']

        ax2.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title(f'样本清晰度分布\n(模糊: {values[1]:,} / {sum(values):,})',
                     fontsize=14, fontweight='bold')

    # 3. 各标签的平均置信度
    ax3 = axes[1, 0]
    if 'overall_conf' in new_detailed.columns:
        label_conf = new_detailed.groupby('label')['overall_conf'].mean().sort_index()

        bars = ax3.bar(range(len(label_conf)), label_conf.values,
                      color=plt.cm.viridis(np.linspace(0, 1, len(label_conf))),
                      alpha=0.8, edgecolor='black')
        ax3.set_xticks(range(len(label_conf)))
        ax3.set_xticklabels([f'L{i}' for i in label_conf.index])
        ax3.set_ylabel('平均置信度', fontsize=12)
        ax3.set_xlabel('标签', fontsize=12)
        ax3.set_title('各标签的平均置信度', fontsize=14, fontweight='bold')
        ax3.axhline(0.20, color='red', linestyle='--', linewidth=2, label='阈值')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, val in zip(bars, label_conf.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. 特征重要性（基于投票权重）
    ax4 = axes[1, 1]
    features = ['熵变化\n(权重×1)', '目的地变化\n(权重×1)',
                '加权比例\n(权重×2)', 'Gini系数\n(权重×2)']
    weights = [1, 1, 2, 2]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    bars = ax4.bar(features, weights, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_ylabel('投票权重', fontsize=12)
    ax4.set_title('特征投票权重分配', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # 添加说明
    ax4.text(0.5, 0.95, '加权比例和Gini系数更可靠\n因此赋予双倍权重',
            transform=ax4.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'voting_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存投票分析图: {output_file}")

    plt.close()


def generate_summary_report(old_detailed, new_detailed, output_dir):
    """生成改进效果摘要报告"""
    print("\n生成改进效果摘要报告...")

    output_dir = Path(output_dir)
    report_file = output_dir / 'improvement_summary.txt'

    # 合并数据
    merged = pd.merge(
        old_detailed[['grid_id', 'label']],
        new_detailed[['grid_id', 'label']],
        on='grid_id',
        suffixes=('_old', '_new')
    )

    changed = merged[merged['label_old'] != merged['label_new']]

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("改进版标注 - 效果摘要报告\n")
        f.write("="*80 + "\n\n")

        # 1. 基本统计
        f.write("1. 基本统计\n")
        f.write("-"*80 + "\n")
        f.write(f"总样本数: {len(merged):,}\n")
        f.write(f"标签改变: {len(changed):,} ({len(changed)/len(merged)*100:.1f}%)\n")
        f.write(f"标签未变: {len(merged) - len(changed):,} ({(len(merged) - len(changed))/len(merged)*100:.1f}%)\n\n")

        # 2. 关键改进
        f.write("2. 关键改进（解决过度预测问题）\n")
        f.write("-"*80 + "\n")

        # L5→L4
        l5_to_l4 = len(changed[(changed['label_old'] == 5) & (changed['label_new'] == 4)])
        l5_old = len(old_detailed[old_detailed['label'] == 5])
        l5_new = len(new_detailed[new_detailed['label'] == 5])

        f.write(f"\nL5（增长聚集型）修正:\n")
        f.write(f"  原始标注: {l5_old:,} 个样本\n")
        f.write(f"  改进标注: {l5_new:,} 个样本\n")
        f.write(f"  L5→L4转换: {l5_to_l4:,} 个样本 ({l5_to_l4/l5_old*100:.1f}%)\n")

        # L9→L7
        l9_to_l7 = len(changed[(changed['label_old'] == 9) & (changed['label_new'] == 7)])
        l9_old = len(old_detailed[old_detailed['label'] == 9])
        l9_new = len(new_detailed[new_detailed['label'] == 9])

        f.write(f"\nL9（衰减扩散型）修正:\n")
        f.write(f"  原始标注: {l9_old:,} 个样本\n")
        f.write(f"  改进标注: {l9_new:,} 个样本\n")
        f.write(f"  L9→L7转换: {l9_to_l7:,} 个样本 ({l9_to_l7/l9_old*100:.1f}%)\n")

        # 所有聚集→静态
        agg_to_static = len(changed[
            (changed['label_old'].isin([2, 5, 8])) &
            (changed['label_new'].isin([1, 4, 7]))
        ])

        # 所有扩散→静态
        diff_to_static = len(changed[
            (changed['label_old'].isin([3, 6, 9])) &
            (changed['label_new'].isin([1, 4, 7]))
        ])

        f.write(f"\n方向维度修正:\n")
        f.write(f"  聚集→静态: {agg_to_static:,} 个样本\n")
        f.write(f"  扩散→静态: {diff_to_static:,} 个样本\n")
        f.write(f"  合计修正: {agg_to_static + diff_to_static:,} 个样本\n\n")

        # 3. 新增特征统计
        f.write("3. 新增特征统计\n")
        f.write("-"*80 + "\n")

        if 'weighted_ratio_change' in new_detailed.columns:
            weighted_ratio = new_detailed['weighted_ratio_change'].dropna()
            f.write(f"\n加权比例变化:\n")
            f.write(f"  平均值: {weighted_ratio.mean():.4f}\n")
            f.write(f"  标准差: {weighted_ratio.std():.4f}\n")
            f.write(f"  范围: [{weighted_ratio.min():.4f}, {weighted_ratio.max():.4f}]\n")

        if 'gini_change' in new_detailed.columns:
            gini_change = new_detailed['gini_change'].dropna()
            f.write(f"\nGini系数变化:\n")
            f.write(f"  平均值: {gini_change.mean():.4f}\n")
            f.write(f"  标准差: {gini_change.std():.4f}\n")
            f.write(f"  范围: [{gini_change.min():.4f}, {gini_change.max():.4f}]\n")

        # 4. 置信度统计
        if 'overall_conf' in new_detailed.columns:
            f.write("\n4. 置信度统计\n")
            f.write("-"*80 + "\n")

            conf = new_detailed['overall_conf'].dropna()
            high_conf = len(conf[conf >= 0.20])
            low_conf = len(conf[conf < 0.20])

            f.write(f"\n整体置信度:\n")
            f.write(f"  平均置信度: {conf.mean():.4f}\n")
            f.write(f"  高置信样本 (≥0.20): {high_conf:,} ({high_conf/len(conf)*100:.1f}%)\n")
            f.write(f"  低置信样本 (<0.20): {low_conf:,} ({low_conf/len(conf)*100:.1f}%)\n")

            # 各标签置信度
            f.write(f"\n各标签平均置信度:\n")
            label_conf = new_detailed.groupby('label')['overall_conf'].mean().sort_index()
            for label, conf_val in label_conf.items():
                f.write(f"  L{label} ({LABEL_MAP[label]}): {conf_val:.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("报告生成完成\n")
        f.write("="*80 + "\n")

    print(f"  ✓ 保存摘要报告: {report_file}")


def main():
    print("="*80)
    print("可视化改进效果")
    print("="*80)

    # 检查文件是否存在
    old_detailed_path = 'data/label_sgh.csv'  # 原始标注（简化版）
    new_detailed_path = '标注/labels/label_sgh_improved_detailed.csv'

    if not Path(old_detailed_path).exists():
        print(f"❌ 错误: 找不到原始标注文件: {old_detailed_path}")
        return 1

    if not Path(new_detailed_path).exists():
        print(f"❌ 错误: 找不到改进标注文件: {new_detailed_path}")
        print(f"   请先运行: cd 标注 && python auto_label_sgh_improved.py")
        return 1

    # 加载数据
    print("\n[1/4] 加载标注文件...")
    print(f"  原始标注: {old_detailed_path}")
    old_detailed = pd.read_csv(old_detailed_path)
    print(f"  ✓ 加载了 {len(old_detailed):,} 个标注")

    print(f"  改进标注: {new_detailed_path}")
    new_detailed = load_detailed_labels(new_detailed_path)

    # 输出目录
    output_dir = 'output_analysis/improvement_visualization'

    # 生成可视化
    print("\n[2/4] 生成特征对比可视化...")
    visualize_feature_comparison(old_detailed, new_detailed, output_dir)

    print("\n[3/4] 生成投票分析可视化...")
    visualize_voting_analysis(new_detailed, output_dir)

    print("\n[4/4] 生成改进效果摘要报告...")
    generate_summary_report(old_detailed, new_detailed, output_dir)

    print("\n" + "="*80)
    print("✅ 可视化完成！")
    print("="*80)
    print(f"\n输出文件:")
    print(f"  - {output_dir}/improvement_visualization.png")
    print(f"  - {output_dir}/voting_analysis.png")
    print(f"  - {output_dir}/improvement_summary.txt")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
