#!/usr/bin/env python3
"""
计算图结构特征用于重新标注
Calculate graph structural features for re-labeling

Usage:
    python scripts/calculate_graph_features.py

Output:
    - graph_features.csv: 所有网格的图结构特征
    - relabel_suggestions.csv: 可疑样本的重新标注建议
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))


def gini_coefficient(weights):
    """计算Gini系数 (集中度指标)"""
    if len(weights) == 0:
        return 0.0

    weights = np.array(weights, dtype=float)
    weights = weights[weights > 0]  # 只考虑正权重

    if len(weights) == 0:
        return 0.0

    if len(weights) == 1:
        return 1.0  # 完全集中

    weights = np.sort(weights)
    n = len(weights)
    index = np.arange(1, n + 1)

    gini = (2 * np.sum(index * weights)) / (n * np.sum(weights)) - (n + 1) / n

    return gini


def calculate_graph_features_for_grid(grid_id, edge_index_2021, edge_weights_2021,
                                      edge_index_2024, edge_weights_2024,
                                      grid_id_to_idx):
    """计算单个网格的图结构特征"""

    features = {}

    # 获取网格的索引
    if grid_id not in grid_id_to_idx:
        return None

    node_idx = grid_id_to_idx[grid_id]

    # === 2021年特征 ===

    # 入边 (流入该网格)
    in_mask_2021 = edge_index_2021[1] == node_idx
    in_edges_2021 = edge_index_2021[:, in_mask_2021]
    in_weights_2021 = edge_weights_2021[in_mask_2021]

    # 出边 (从该网格流出)
    out_mask_2021 = edge_index_2021[0] == node_idx
    out_edges_2021 = edge_index_2021[:, out_mask_2021]
    out_weights_2021 = edge_weights_2021[out_mask_2021]

    # 度中心性
    features['in_degree_2021'] = len(in_weights_2021)
    features['out_degree_2021'] = len(out_weights_2021)

    # 加权度 (总流量)
    features['weighted_in_2021'] = float(np.sum(in_weights_2021))
    features['weighted_out_2021'] = float(np.sum(out_weights_2021))

    # 集中度 (Gini系数)
    features['in_concentration_2021'] = gini_coefficient(in_weights_2021)
    features['out_concentration_2021'] = gini_coefficient(out_weights_2021)

    # === 2024年特征 ===

    # 入边
    in_mask_2024 = edge_index_2024[1] == node_idx
    in_edges_2024 = edge_index_2024[:, in_mask_2024]
    in_weights_2024 = edge_weights_2024[in_mask_2024]

    # 出边
    out_mask_2024 = edge_index_2024[0] == node_idx
    out_edges_2024 = edge_index_2024[:, out_mask_2024]
    out_weights_2024 = edge_weights_2024[out_mask_2024]

    # 度中心性
    features['in_degree_2024'] = len(in_weights_2024)
    features['out_degree_2024'] = len(out_weights_2024)

    # 加权度
    features['weighted_in_2024'] = float(np.sum(in_weights_2024))
    features['weighted_out_2024'] = float(np.sum(out_weights_2024))

    # 集中度
    features['in_concentration_2024'] = gini_coefficient(in_weights_2024)
    features['out_concentration_2024'] = gini_coefficient(out_weights_2024)

    # === 变化特征 ===

    # 度变化
    features['in_degree_change'] = features['in_degree_2024'] - features['in_degree_2021']
    features['out_degree_change'] = features['out_degree_2024'] - features['out_degree_2021']

    # 度比例
    features['degree_ratio_2021'] = features['in_degree_2021'] / (features['out_degree_2021'] + 1)
    features['degree_ratio_2024'] = features['in_degree_2024'] / (features['out_degree_2024'] + 1)
    features['degree_ratio_change'] = features['degree_ratio_2024'] - features['degree_ratio_2021']

    # 加权度变化
    features['weighted_in_change'] = features['weighted_in_2024'] - features['weighted_in_2021']
    features['weighted_out_change'] = features['weighted_out_2024'] - features['weighted_out_2021']

    # 加权度比例
    features['weighted_ratio_2021'] = features['weighted_in_2021'] / (features['weighted_out_2021'] + 1)
    features['weighted_ratio_2024'] = features['weighted_in_2024'] / (features['weighted_out_2024'] + 1)
    features['weighted_ratio_change'] = features['weighted_ratio_2024'] - features['weighted_ratio_2021']

    # 集中度变化
    features['in_concentration_change'] = features['in_concentration_2024'] - features['in_concentration_2021']
    features['out_concentration_change'] = features['out_concentration_2024'] - features['out_concentration_2021']

    return features


def suggest_label(growth_rate, graph_features):
    """基于特征建议标签"""

    # 1. 判断强度维度
    if growth_rate > 0.1:
        intensity = 'Growth'
        intensity_code = 1  # Growth: 4, 5, 6
    elif growth_rate < -0.1:
        intensity = 'Decline'
        intensity_code = 2  # Decline: 7, 8, 9
    else:
        intensity = 'Stable'
        intensity_code = 0  # Stable: 1, 2, 3

    # 2. 判断方向维度
    weighted_ratio_change = graph_features['weighted_ratio_change']
    degree_ratio_change = graph_features['degree_ratio_change']
    in_conc_change = graph_features['in_concentration_change']
    out_conc_change = graph_features['out_concentration_change']

    # Static条件 (4个条件)
    static_conditions = [
        abs(weighted_ratio_change) <= 0.2,
        abs(degree_ratio_change) <= 0.2,
        abs(in_conc_change) <= 0.1,
        abs(out_conc_change) <= 0.1
    ]
    static_score = sum(static_conditions)

    # Aggregation条件 (4个条件)
    agg_conditions = [
        weighted_ratio_change > 0.2,
        degree_ratio_change > 0.2,
        in_conc_change > 0.1,
        out_conc_change < -0.1
    ]
    agg_score = sum(agg_conditions)

    # Diffusion条件 (4个条件)
    diff_conditions = [
        weighted_ratio_change < -0.2,
        degree_ratio_change < -0.2,
        out_conc_change > 0.1,
        in_conc_change < -0.1
    ]
    diff_score = sum(diff_conditions)

    # 选择得分最高的方向
    if static_score >= 3:
        direction = 'Static'
        direction_code = 0
    elif agg_score >= 2 and agg_score > diff_score:
        direction = 'Aggregation'
        direction_code = 1
    elif diff_score >= 2:
        direction = 'Diffusion'
        direction_code = 2
    else:
        # 默认Static
        direction = 'Static'
        direction_code = 0

    # 组合标签
    label_map = {
        (0, 0): 1,  # Stable Static
        (0, 1): 2,  # Stable Aggregation
        (0, 2): 3,  # Stable Diffusion
        (1, 0): 4,  # Growth Static
        (1, 1): 5,  # Growth Aggregation
        (1, 2): 6,  # Growth Diffusion
        (2, 0): 7,  # Decline Static
        (2, 1): 8,  # Decline Aggregation
        (2, 2): 9,  # Decline Diffusion
    }

    suggested_label = label_map[(intensity_code, direction_code)]

    return {
        'suggested_label': suggested_label,
        'intensity': intensity,
        'direction': direction,
        'static_score': static_score,
        'agg_score': agg_score,
        'diff_score': diff_score,
        'confidence': max(static_score, agg_score, diff_score) / 4.0  # 归一化到0-1
    }


def main():
    print("="*80)
    print("计算图结构特征并生成重新标注建议")
    print("="*80)

    # 读取数据
    print("\n[1/5] 读取数据...")

    labels_df = pd.read_csv('data/label_sgh.csv')
    suspicious_df = pd.read_csv('suspicious_labels.csv')

    cache_file = 'data/cache/dual_year_data_all_grids_correct.pkl'
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    print(f"  ✓ 标注数据: {len(labels_df)} 个样本")
    print(f"  ✓ 可疑样本: {len(suspicious_df)} 个样本")

    # 提取图数据
    print("\n[2/5] 提取图数据...")

    graphs_2021 = data['graphs_2021']
    graphs_2024 = data['graphs_2024']
    grid_id_to_idx = data['grid_id_to_idx']

    # graphs_2021/2024 是 (edge_index, edge_attr) 的元组列表
    # 我们只需要第一个图 (完整图)
    edge_index_2021, edge_weights_2021 = graphs_2021[0]
    edge_index_2024, edge_weights_2024 = graphs_2024[0]

    # 转换为numpy数组
    edge_index_2021 = edge_index_2021.numpy()
    edge_weights_2021 = edge_weights_2021.numpy()
    edge_index_2024 = edge_index_2024.numpy()
    edge_weights_2024 = edge_weights_2024.numpy()

    print(f"  ✓ 2021年图: {edge_index_2021.shape[1]} 条边")
    print(f"  ✓ 2024年图: {edge_index_2024.shape[1]} 条边")

    # 计算图结构特征
    print("\n[3/5] 计算图结构特征...")

    all_features = []

    for grid_id in tqdm(suspicious_df['grid_id'].unique(), desc="  处理网格"):
        features = calculate_graph_features_for_grid(
            grid_id,
            edge_index_2021, edge_weights_2021,
            edge_index_2024, edge_weights_2024,
            grid_id_to_idx
        )

        if features is not None:
            features['grid_id'] = grid_id
            all_features.append(features)

    graph_features_df = pd.DataFrame(all_features)

    # 保存图结构特征
    graph_features_df.to_csv('graph_features.csv', index=False)
    print(f"  ✓ 图结构特征已保存: graph_features.csv")
    print(f"  ✓ 计算了 {len(graph_features_df)} 个网格的特征")

    # 生成重新标注建议
    print("\n[4/5] 生成重新标注建议...")

    relabel_suggestions = []

    for idx, row in tqdm(suspicious_df.iterrows(), total=len(suspicious_df), desc="  生成建议"):
        grid_id = row['grid_id']
        current_label = row['label']
        growth_rate = row['growth_rate']

        # 获取图特征
        graph_feat = graph_features_df[graph_features_df['grid_id'] == grid_id]

        if len(graph_feat) == 0:
            continue

        graph_feat = graph_feat.iloc[0].to_dict()

        # 生成建议
        suggestion = suggest_label(growth_rate, graph_feat)

        # 判断是否需要修改
        current_label_num = int(current_label[1])  # 从 'L4' 提取 4
        suggested_label_num = suggestion['suggested_label']

        needs_change = (current_label_num != suggested_label_num)

        relabel_suggestions.append({
            'grid_id': grid_id,
            'current_label': current_label_num,
            'suggested_label': suggested_label_num,
            'needs_change': needs_change,
            'current_intensity': row['intensity_expected'],
            'current_direction': row['direction_expected'],
            'suggested_intensity': suggestion['intensity'],
            'suggested_direction': suggestion['direction'],
            'confidence': suggestion['confidence'],
            'static_score': suggestion['static_score'],
            'agg_score': suggestion['agg_score'],
            'diff_score': suggestion['diff_score'],
            'growth_rate': growth_rate,
            'weighted_ratio_change': graph_feat['weighted_ratio_change'],
            'degree_ratio_change': graph_feat['degree_ratio_change'],
            'in_concentration_change': graph_feat['in_concentration_change'],
            'out_concentration_change': graph_feat['out_concentration_change'],
            'error_type': row['error_type']
        })

    relabel_df = pd.DataFrame(relabel_suggestions)

    # 保存建议
    relabel_df.to_csv('relabel_suggestions.csv', index=False)
    print(f"  ✓ 重新标注建议已保存: relabel_suggestions.csv")

    # 统计分析
    print("\n[5/5] 统计分析...")

    total_suspicious = len(relabel_df)
    needs_change = relabel_df['needs_change'].sum()
    no_change = total_suspicious - needs_change

    print(f"\n  可疑样本总数: {total_suspicious}")
    print(f"  建议修改标签: {needs_change} ({needs_change/total_suspicious*100:.1f}%)")
    print(f"  建议保持不变: {no_change} ({no_change/total_suspicious*100:.1f}%)")

    # 按当前标签统计
    print(f"\n  各类别建议修改数:")
    for label in ['L4', 'L5', 'L7', 'L9']:
        label_num = int(label[1])
        label_data = relabel_df[relabel_df['current_label'] == label_num]
        if len(label_data) > 0:
            change_count = label_data['needs_change'].sum()
            print(f"    {label}: {change_count}/{len(label_data)} ({change_count/len(label_data)*100:.1f}%)")

    # 标签转换统计
    print(f"\n  主要标签转换:")
    changes = relabel_df[relabel_df['needs_change']]
    if len(changes) > 0:
        transitions = changes.groupby(['current_label', 'suggested_label']).size().sort_values(ascending=False)
        for (curr, sugg), count in transitions.head(10).items():
            print(f"    L{curr} → L{sugg}: {count} 个样本")

    # 高置信度建议
    high_conf = relabel_df[relabel_df['confidence'] >= 0.75]
    print(f"\n  高置信度建议 (≥0.75): {len(high_conf)} 个样本")
    print(f"    其中需要修改: {high_conf['needs_change'].sum()} 个")

    print("\n" + "="*80)
    print("✅ 完成!")
    print("="*80)

    print("\n生成的文件:")
    print("  1. graph_features.csv - 所有可疑样本的图结构特征")
    print("  2. relabel_suggestions.csv - 重新标注建议")

    print("\n下一步:")
    print("  1. 查看 relabel_suggestions.csv")
    print("  2. 按 confidence 排序,优先处理高置信度样本")
    print("  3. 人工审核并决定是否接受建议")
    print("  4. 使用 scripts/apply_relabeling.py 应用新标签")


if __name__ == '__main__':
    main()
