#!/usr/bin/env python3
"""
应用重新标注建议
Apply re-labeling suggestions

Usage:
    python scripts/apply_relabeling.py [--confidence THRESHOLD] [--dry-run]

Options:
    --confidence THRESHOLD  只应用置信度 >= THRESHOLD 的建议 (默认: 0.75)
    --dry-run              只显示将要修改的内容,不实际修改

Output:
    - data/label_sgh_relabeled.csv: 新的标注文件
    - relabeling_report.txt: 重新标注报告
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='应用重新标注建议')
    parser.add_argument('--confidence', type=float, default=0.75,
                        help='置信度阈值 (默认: 0.75)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示将要修改的内容,不实际修改')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式模式,逐个确认每个修改')
    args = parser.parse_args()

    print("="*80)
    print("应用重新标注建议")
    print("="*80)

    # 读取数据
    print("\n[1/4] 读取数据...")

    labels_df = pd.read_csv('data/label_sgh.csv')
    relabel_df = pd.read_csv('relabel_suggestions.csv')

    print(f"  ✓ 原始标注: {len(labels_df)} 个样本")
    print(f"  ✓ 重新标注建议: {len(relabel_df)} 个样本")

    # 过滤建议
    print(f"\n[2/4] 过滤建议 (置信度 >= {args.confidence})...")

    # 只保留需要修改且置信度足够的建议
    filtered_suggestions = relabel_df[
        (relabel_df['needs_change'] == True) &
        (relabel_df['confidence'] >= args.confidence)
    ]

    print(f"  ✓ 符合条件的建议: {len(filtered_suggestions)} 个")

    if len(filtered_suggestions) == 0:
        print("\n  没有符合条件的建议,退出.")
        return

    # 显示统计
    print(f"\n  标签转换统计:")
    transitions = filtered_suggestions.groupby(['current_label', 'suggested_label']).size().sort_values(ascending=False)
    for (curr, sugg), count in transitions.items():
        print(f"    L{curr} → L{sugg}: {count} 个样本")

    # 交互式确认
    if args.interactive:
        print(f"\n[3/4] 交互式确认...")
        confirmed_changes = []

        for idx, row in filtered_suggestions.iterrows():
            print(f"\n  Grid {row['grid_id']}:")
            print(f"    当前标签: L{row['current_label']} ({row['current_intensity']} {row['current_direction']})")
            print(f"    建议标签: L{row['suggested_label']} ({row['suggested_intensity']} {row['suggested_direction']})")
            print(f"    置信度: {row['confidence']:.2f}")
            print(f"    增长率: {row['growth_rate']:.4f}")
            print(f"    加权比例变化: {row['weighted_ratio_change']:.4f}")
            print(f"    度比例变化: {row['degree_ratio_change']:.4f}")

            response = input("    接受此建议? (y/n/q): ").strip().lower()

            if response == 'q':
                print("  用户取消,退出.")
                return
            elif response == 'y':
                confirmed_changes.append(row)

        if len(confirmed_changes) == 0:
            print("\n  没有确认的修改,退出.")
            return

        filtered_suggestions = pd.DataFrame(confirmed_changes)
        print(f"\n  确认修改: {len(filtered_suggestions)} 个样本")

    # 应用修改
    print(f"\n[3/4] 应用修改...")

    new_labels_df = labels_df.copy()
    changes_made = 0

    for idx, row in filtered_suggestions.iterrows():
        grid_id = row['grid_id']
        new_label = row['suggested_label']

        # 找到对应的行并修改
        mask = new_labels_df['grid_id'] == grid_id
        if mask.any():
            old_label = new_labels_df.loc[mask, 'label'].values[0]
            new_labels_df.loc[mask, 'label'] = new_label
            changes_made += 1

            if args.dry_run:
                print(f"  [DRY-RUN] Grid {grid_id}: L{old_label} → L{new_label}")

    print(f"  ✓ 修改了 {changes_made} 个样本的标签")

    # 保存结果
    if not args.dry_run:
        print(f"\n[4/4] 保存结果...")

        output_file = 'data/label_sgh_relabeled.csv'
        new_labels_df.to_csv(output_file, index=False)
        print(f"  ✓ 新标注已保存: {output_file}")

        # 生成报告
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("重新标注报告")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"置信度阈值: {args.confidence}")
        report_lines.append(f"修改样本数: {changes_made}")
        report_lines.append("")
        report_lines.append("标签转换统计:")
        for (curr, sugg), count in transitions.items():
            report_lines.append(f"  L{curr} → L{sugg}: {count} 个样本")
        report_lines.append("")
        report_lines.append("新标注分布:")
        new_dist = new_labels_df['label'].value_counts().sort_index()
        for label, count in new_dist.items():
            report_lines.append(f"  L{label}: {count} 个样本")
        report_lines.append("")
        report_lines.append("="*80)

        report_text = "\n".join(report_lines)

        with open('relabeling_report.txt', 'w') as f:
            f.write(report_text)

        print(f"  ✓ 报告已保存: relabeling_report.txt")

        print("\n" + "="*80)
        print("✅ 完成!")
        print("="*80)

        print("\n下一步:")
        print("  1. 查看 relabeling_report.txt")
        print("  2. 使用新标注重新训练模型:")
        print("     python train_multiscale_temporal.py --label-file data/label_sgh_relabeled.csv")
        print("  3. 评估新模型性能")

    else:
        print("\n" + "="*80)
        print("✅ DRY-RUN 完成!")
        print("="*80)
        print("\n  这是一次试运行,没有实际修改任何文件.")
        print("  如果确认要应用这些修改,请去掉 --dry-run 参数重新运行.")


if __name__ == '__main__':
    main()
