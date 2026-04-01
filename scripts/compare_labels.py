#!/usr/bin/env python3
"""
标注对比分析脚本

对比原始标注和改进标注的差异，生成详细报告

Usage:
    python scripts/compare_labels.py --old data/label_sgh.csv --new labels/label_sgh_improved.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import Counter

# 标签映射
LABEL_MAP = {
    1: "稳定静态型", 2: "稳定聚集型", 3: "稳定扩散型",
    4: "增长静态型", 5: "增长聚集型", 6: "增长扩散型",
    7: "衰减静态型", 8: "衰减聚集型", 9: "衰减扩散型"
}

# 维度映射
def get_intensity(label):
    """获取强度维度"""
    if label in [1, 2, 3]:
        return "稳定"
    elif label in [4, 5, 6]:
        return "增长"
    elif label in [7, 8, 9]:
        return "衰减"
    return "未知"

def get_direction(label):
    """获取方向维度"""
    if label in [1, 4, 7]:
        return "静态"
    elif label in [2, 5, 8]:
        return "聚集"
    elif label in [3, 6, 9]:
        return "扩散"
    return "未知"


def load_labels(filepath):
    """加载标注文件"""
    df = pd.read_csv(filepath)
    print(f"  ✓ 加载了 {len(df):,} 个标注")
    return df


def compare_distributions(old_df, new_df):
    """对比标签分布"""
    print("\n" + "="*80)
    print("标签分布对比")
    print("="*80)

    old_dist = old_df['label'].value_counts().sort_index()
    new_dist = new_df['label'].value_counts().sort_index()

    print(f"\n{'标签':<6} {'类型':<12} {'原始标注':<12} {'改进标注':<12} {'变化':<12}")
    print("-"*80)

    for label in range(1, 10):
        old_count = old_dist.get(label, 0)
        new_count = new_dist.get(label, 0)
        change = new_count - old_count
        change_pct = (change / old_count * 100) if old_count > 0 else 0

        print(f"L{label:<5} {LABEL_MAP[label]:<12} {old_count:<12} {new_count:<12} {change:+5d} ({change_pct:+.1f}%)")

    print("-"*80)
    print(f"{'总计':<6} {'':<12} {len(old_df):<12} {len(new_df):<12}")


def compare_dimensions(old_df, new_df):
    """对比维度分布"""
    print("\n" + "="*80)
    print("维度分布对比")
    print("="*80)

    # 强度维度
    old_df['intensity'] = old_df['label'].apply(get_intensity)
    new_df['intensity'] = new_df['label'].apply(get_intensity)

    old_intensity = old_df['intensity'].value_counts()
    new_intensity = new_df['intensity'].value_counts()

    print("\n强度维度（Intensity）:")
    print(f"{'维度':<10} {'原始标注':<12} {'改进标注':<12} {'变化':<12}")
    print("-"*60)
    for intensity in ["稳定", "增长", "衰减"]:
        old_count = old_intensity.get(intensity, 0)
        new_count = new_intensity.get(intensity, 0)
        change = new_count - old_count
        change_pct = (change / old_count * 100) if old_count > 0 else 0
        print(f"{intensity:<10} {old_count:<12} {new_count:<12} {change:+5d} ({change_pct:+.1f}%)")

    # 方向维度
    old_df['direction'] = old_df['label'].apply(get_direction)
    new_df['direction'] = new_df['label'].apply(get_direction)

    old_direction = old_df['direction'].value_counts()
    new_direction = new_df['direction'].value_counts()

    print("\n方向维度（Direction）:")
    print(f"{'维度':<10} {'原始标注':<12} {'改进标注':<12} {'变化':<12}")
    print("-"*60)
    for direction in ["静态", "聚集", "扩散"]:
        old_count = old_direction.get(direction, 0)
        new_count = new_direction.get(direction, 0)
        change = new_count - old_count
        change_pct = (change / old_count * 100) if old_count > 0 else 0
        print(f"{direction:<10} {old_count:<12} {new_count:<12} {change:+5d} ({change_pct:+.1f}%)")


def analyze_label_changes(old_df, new_df):
    """分析标签变化"""
    print("\n" + "="*80)
    print("标签变化分析")
    print("="*80)

    # 合并数据
    merged = pd.merge(old_df[['grid_id', 'label']],
                      new_df[['grid_id', 'label']],
                      on='grid_id',
                      suffixes=('_old', '_new'))

    # 统计变化
    changed = merged[merged['label_old'] != merged['label_new']]
    unchanged = merged[merged['label_old'] == merged['label_new']]

    print(f"\n总样本数: {len(merged):,}")
    print(f"标签未变: {len(unchanged):,} ({len(unchanged)/len(merged)*100:.1f}%)")
    print(f"标签改变: {len(changed):,} ({len(changed)/len(merged)*100:.1f}%)")

    # 分析变化模式
    if len(changed) > 0:
        print(f"\n主要变化模式（Top 10）:")
        print(f"{'原标签':<10} {'新标签':<10} {'数量':<10} {'占比':<10}")
        print("-"*60)

        change_patterns = changed.groupby(['label_old', 'label_new']).size().sort_values(ascending=False)

        for (old_label, new_label), count in change_patterns.head(10).items():
            old_name = LABEL_MAP.get(old_label, f"L{old_label}")
            new_name = LABEL_MAP.get(new_label, f"L{new_label}")
            pct = count / len(changed) * 100
            print(f"L{old_label} → L{new_label:<3} {count:<10} {pct:.1f}%")
            print(f"  ({old_name} → {new_name})")

    return merged, changed


def analyze_dimension_changes(changed_df):
    """分析维度变化"""
    print("\n" + "="*80)
    print("维度变化分析")
    print("="*80)

    changed_df['intensity_old'] = changed_df['label_old'].apply(get_intensity)
    changed_df['intensity_new'] = changed_df['label_new'].apply(get_intensity)
    changed_df['direction_old'] = changed_df['label_old'].apply(get_direction)
    changed_df['direction_new'] = changed_df['label_new'].apply(get_direction)

    # 强度维度变化
    intensity_changed = changed_df[changed_df['intensity_old'] != changed_df['intensity_new']]
    print(f"\n强度维度改变: {len(intensity_changed):,} ({len(intensity_changed)/len(changed_df)*100:.1f}%)")

    if len(intensity_changed) > 0:
        print(f"\n强度变化模式:")
        intensity_patterns = intensity_changed.groupby(['intensity_old', 'intensity_new']).size().sort_values(ascending=False)
        for (old, new), count in intensity_patterns.items():
            print(f"  {old} → {new}: {count:,} 个")

    # 方向维度变化
    direction_changed = changed_df[changed_df['direction_old'] != changed_df['direction_new']]
    print(f"\n方向维度改变: {len(direction_changed):,} ({len(direction_changed)/len(changed_df)*100:.1f}%)")

    if len(direction_changed) > 0:
        print(f"\n方向变化模式:")
        direction_patterns = direction_changed.groupby(['direction_old', 'direction_new']).size().sort_values(ascending=False)
        for (old, new), count in direction_patterns.items():
            print(f"  {old} → {new}: {count:,} 个")

    # 关键变化：聚集/扩散 → 静态
    agg_to_static = changed_df[(changed_df['direction_old'] == '聚集') & (changed_df['direction_new'] == '静态')]
    diff_to_static = changed_df[(changed_df['direction_old'] == '扩散') & (changed_df['direction_new'] == '静态')]

    print(f"\n关键改进（解决过度预测问题）:")
    print(f"  聚集 → 静态: {len(agg_to_static):,} 个")
    print(f"  扩散 → 静态: {len(diff_to_static):,} 个")
    print(f"  合计: {len(agg_to_static) + len(diff_to_static):,} 个")

    # 具体到L5和L9
    l5_to_l4 = changed_df[(changed_df['label_old'] == 5) & (changed_df['label_new'] == 4)]
    l9_to_l7 = changed_df[(changed_df['label_old'] == 9) & (changed_df['label_new'] == 7)]

    print(f"\n针对问题类别的修正:")
    print(f"  L5（增长聚集）→ L4（增长静态）: {len(l5_to_l4):,} 个")
    print(f"  L9（衰减扩散）→ L7（衰减静态）: {len(l9_to_l7):,} 个")


def visualize_comparison(old_df, new_df, changed_df, output_dir):
    """可视化对比"""
    print("\n" + "="*80)
    print("生成可视化对比图")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('标注对比分析', fontsize=16, fontweight='bold')

    # 1. 标签分布对比（柱状图）
    ax1 = axes[0, 0]
    old_dist = old_df['label'].value_counts().sort_index()
    new_dist = new_df['label'].value_counts().sort_index()

    x = np.arange(1, 10)
    width = 0.35

    ax1.bar(x - width/2, [old_dist.get(i, 0) for i in range(1, 10)], width, label='原始标注', alpha=0.8)
    ax1.bar(x + width/2, [new_dist.get(i, 0) for i in range(1, 10)], width, label='改进标注', alpha=0.8)
    ax1.set_xlabel('标签')
    ax1.set_ylabel('样本数')
    ax1.set_title('标签分布对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{i}' for i in range(1, 10)])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. 强度维度对比
    ax2 = axes[0, 1]
    old_df['intensity'] = old_df['label'].apply(get_intensity)
    new_df['intensity'] = new_df['label'].apply(get_intensity)

    old_intensity = old_df['intensity'].value_counts()
    new_intensity = new_df['intensity'].value_counts()

    intensities = ["稳定", "增长", "衰减"]
    x_int = np.arange(len(intensities))

    ax2.bar(x_int - width/2, [old_intensity.get(i, 0) for i in intensities], width, label='原始标注', alpha=0.8)
    ax2.bar(x_int + width/2, [new_intensity.get(i, 0) for i in intensities], width, label='改进标注', alpha=0.8)
    ax2.set_xlabel('强度维度')
    ax2.set_ylabel('样本数')
    ax2.set_title('强度维度分布对比')
    ax2.set_xticks(x_int)
    ax2.set_xticklabels(intensities)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. 方向维度对比
    ax3 = axes[0, 2]
    old_df['direction'] = old_df['label'].apply(get_direction)
    new_df['direction'] = new_df['label'].apply(get_direction)

    old_direction = old_df['direction'].value_counts()
    new_direction = new_df['direction'].value_counts()

    directions = ["静态", "聚集", "扩散"]
    x_dir = np.arange(len(directions))

    ax3.bar(x_dir - width/2, [old_direction.get(i, 0) for i in directions], width, label='原始标注', alpha=0.8)
    ax3.bar(x_dir + width/2, [new_direction.get(i, 0) for i in directions], width, label='改进标注', alpha=0.8)
    ax3.set_xlabel('方向维度')
    ax3.set_ylabel('样本数')
    ax3.set_title('方向维度分布对比（关键改进）')
    ax3.set_xticks(x_dir)
    ax3.set_xticklabels(directions)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. 标签变化统计
    ax4 = axes[1, 0]
    changed_count = len(changed_df)
    unchanged_count = len(old_df) - changed_count

    ax4.pie([unchanged_count, changed_count],
            labels=['未改变', '已改变'],
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            startangle=90)
    ax4.set_title(f'标签变化比例\n(改变: {changed_count:,} / {len(old_df):,})')

    # 5. Top变化模式
    ax5 = axes[1, 1]
    if len(changed_df) > 0:
        change_patterns = changed_df.groupby(['label_old', 'label_new']).size().sort_values(ascending=False).head(10)

        y_pos = np.arange(len(change_patterns))
        labels = [f"L{old}→L{new}" for (old, new) in change_patterns.index]

        ax5.barh(y_pos, change_patterns.values, alpha=0.8)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(labels)
        ax5.set_xlabel('样本数')
        ax5.set_title('Top 10 标签变化模式')
        ax5.grid(axis='x', alpha=0.3)

    # 6. 关键改进统计
    ax6 = axes[1, 2]

    # 统计关键变化
    l5_to_l4 = len(changed_df[(changed_df['label_old'] == 5) & (changed_df['label_new'] == 4)])
    l9_to_l7 = len(changed_df[(changed_df['label_old'] == 9) & (changed_df['label_new'] == 7)])
    agg_to_static = len(changed_df[(changed_df['direction_old'] == '聚集') & (changed_df['direction_new'] == '静态')])
    diff_to_static = len(changed_df[(changed_df['direction_old'] == '扩散') & (changed_df['direction_new'] == '静态')])

    categories = ['L5→L4\n(增长聚集→静态)', 'L9→L7\n(衰减扩散→静态)',
                  '所有聚集→静态', '所有扩散→静态']
    values = [l5_to_l4, l9_to_l7, agg_to_static, diff_to_static]

    ax6.bar(range(len(categories)), values, alpha=0.8, color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c'])
    ax6.set_xticks(range(len(categories)))
    ax6.set_xticklabels(categories, rotation=15, ha='right')
    ax6.set_ylabel('样本数')
    ax6.set_title('关键改进：过度预测修正')
    ax6.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, v in enumerate(values):
        ax6.text(i, v + max(values)*0.02, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'label_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存可视化图: {output_file}")

    plt.close()


def generate_report(old_df, new_df, merged_df, changed_df, output_dir):
    """生成详细报告"""
    print("\n" + "="*80)
    print("生成详细报告")
    print("="*80)

    output_dir = Path(output_dir)
    report_file = output_dir / 'label_comparison_report.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("标注对比分析报告\n")
        f.write("="*80 + "\n\n")

        # 基本统计
        f.write("1. 基本统计\n")
        f.write("-"*80 + "\n")
        f.write(f"原始标注样本数: {len(old_df):,}\n")
        f.write(f"改进标注样本数: {len(new_df):,}\n")
        f.write(f"标签未改变: {len(merged_df[merged_df['label_old'] == merged_df['label_new']]):,} ({len(merged_df[merged_df['label_old'] == merged_df['label_new']])/len(merged_df)*100:.1f}%)\n")
        f.write(f"标签已改变: {len(changed_df):,} ({len(changed_df)/len(merged_df)*100:.1f}%)\n\n")

        # 标签分布
        f.write("2. 标签分布对比\n")
        f.write("-"*80 + "\n")
        old_dist = old_df['label'].value_counts().sort_index()
        new_dist = new_df['label'].value_counts().sort_index()

        f.write(f"{'标签':<6} {'类型':<15} {'原始':<10} {'改进':<10} {'变化':<15}\n")
        f.write("-"*80 + "\n")
        for label in range(1, 10):
            old_count = old_dist.get(label, 0)
            new_count = new_dist.get(label, 0)
            change = new_count - old_count
            change_pct = (change / old_count * 100) if old_count > 0 else 0
            f.write(f"L{label:<5} {LABEL_MAP[label]:<15} {old_count:<10} {new_count:<10} {change:+5d} ({change_pct:+6.1f}%)\n")
        f.write("\n")

        # 维度分布
        f.write("3. 维度分布对比\n")
        f.write("-"*80 + "\n")

        old_df['intensity'] = old_df['label'].apply(get_intensity)
        new_df['intensity'] = new_df['label'].apply(get_intensity)
        old_df['direction'] = old_df['label'].apply(get_direction)
        new_df['direction'] = new_df['label'].apply(get_direction)

        f.write("\n强度维度:\n")
        old_intensity = old_df['intensity'].value_counts()
        new_intensity = new_df['intensity'].value_counts()
        for intensity in ["稳定", "增长", "衰减"]:
            old_count = old_intensity.get(intensity, 0)
            new_count = new_intensity.get(intensity, 0)
            change = new_count - old_count
            f.write(f"  {intensity}: {old_count:,} → {new_count:,} ({change:+,})\n")

        f.write("\n方向维度:\n")
        old_direction = old_df['direction'].value_counts()
        new_direction = new_df['direction'].value_counts()
        for direction in ["静态", "聚集", "扩散"]:
            old_count = old_direction.get(direction, 0)
            new_count = new_direction.get(direction, 0)
            change = new_count - old_count
            f.write(f"  {direction}: {old_count:,} → {new_count:,} ({change:+,})\n")
        f.write("\n")

        # 关键改进
        f.write("4. 关键改进（解决过度预测问题）\n")
        f.write("-"*80 + "\n")

        changed_df['direction_old'] = changed_df['label_old'].apply(get_direction)
        changed_df['direction_new'] = changed_df['label_new'].apply(get_direction)

        agg_to_static = changed_df[(changed_df['direction_old'] == '聚集') & (changed_df['direction_new'] == '静态')]
        diff_to_static = changed_df[(changed_df['direction_old'] == '扩散') & (changed_df['direction_new'] == '静态')]

        f.write(f"聚集 → 静态: {len(agg_to_static):,} 个样本\n")
        f.write(f"扩散 → 静态: {len(diff_to_static):,} 个样本\n")
        f.write(f"合计修正: {len(agg_to_static) + len(diff_to_static):,} 个样本\n\n")

        l5_to_l4 = changed_df[(changed_df['label_old'] == 5) & (changed_df['label_new'] == 4)]
        l9_to_l7 = changed_df[(changed_df['label_old'] == 9) & (changed_df['label_new'] == 7)]

        f.write("针对问题类别:\n")
        f.write(f"  L5（增长聚集）→ L4（增长静态）: {len(l5_to_l4):,} 个\n")
        f.write(f"  L9（衰减扩散）→ L7（衰减静态）: {len(l9_to_l7):,} 个\n\n")

        # Top变化模式
        f.write("5. Top 10 标签变化模式\n")
        f.write("-"*80 + "\n")
        change_patterns = changed_df.groupby(['label_old', 'label_new']).size().sort_values(ascending=False)
        for i, ((old_label, new_label), count) in enumerate(change_patterns.head(10).items(), 1):
            pct = count / len(changed_df) * 100
            f.write(f"{i}. L{old_label} → L{new_label}: {count:,} 个 ({pct:.1f}%)\n")
            f.write(f"   {LABEL_MAP[old_label]} → {LABEL_MAP[new_label]}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("报告生成完成\n")
        f.write("="*80 + "\n")

    print(f"  ✓ 保存报告: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='标注对比分析')
    parser.add_argument('--old', type=str, required=True, help='原始标注文件路径')
    parser.add_argument('--new', type=str, required=True, help='改进标注文件路径')
    parser.add_argument('--output', type=str, default='output_analysis/label_comparison',
                        help='输出目录（默认: output_analysis/label_comparison）')
    args = parser.parse_args()

    print("="*80)
    print("标注对比分析")
    print("="*80)

    # 加载数据
    print("\n[1/6] 加载标注文件...")
    print(f"  原始标注: {args.old}")
    old_df = load_labels(args.old)
    print(f"  改进标注: {args.new}")
    new_df = load_labels(args.new)

    # 对比分布
    print("\n[2/6] 对比标签分布...")
    compare_distributions(old_df, new_df)

    # 对比维度
    print("\n[3/6] 对比维度分布...")
    compare_dimensions(old_df, new_df)

    # 分析变化
    print("\n[4/6] 分析标签变化...")
    merged_df, changed_df = analyze_label_changes(old_df, new_df)

    # 分析维度变化
    print("\n[5/6] 分析维度变化...")
    analyze_dimension_changes(changed_df)

    # 可视化
    print("\n[6/6] 生成可视化和报告...")
    visualize_comparison(old_df, new_df, changed_df, args.output)
    generate_report(old_df, new_df, merged_df, changed_df, args.output)

    print("\n" + "="*80)
    print("✅ 对比分析完成！")
    print("="*80)
    print(f"\n输出文件:")
    print(f"  - {args.output}/label_comparison.png")
    print(f"  - {args.output}/label_comparison_report.txt")


if __name__ == "__main__":
    main()
