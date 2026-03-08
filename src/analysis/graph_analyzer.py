"""
Graph analysis tool for diagnosing isolated nodes and graph structure issues

This tool provides comprehensive analysis of spatial graphs built from OD flow data,
with focus on identifying isolated nodes that can negatively impact GAT performance.
"""

import numpy as np
import pandas as pd
import argparse
import logging
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.dual_year_processor import prepare_dual_year_experiment_data
from preprocessing.graph_builder import SpatialGraphBuilder
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_degrees(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    Compute node degrees from edge_index

    Args:
        edge_index: Edge index array (2, num_edges)
        num_nodes: Total number of nodes

    Returns:
        degrees: Array of node degrees (num_nodes,)
    """
    if edge_index.shape[1] == 0:
        return np.zeros(num_nodes, dtype=int)

    # Count occurrences of each node in edge_index
    all_nodes = np.concatenate([edge_index[0], edge_index[1]])
    degrees = np.bincount(all_nodes, minlength=num_nodes)
    return degrees


def find_isolated_nodes(edge_index: np.ndarray, num_nodes: int) -> Tuple[List[int], float]:
    """
    Find isolated nodes (nodes with degree = 0)

    Args:
        edge_index: Edge index array (2, num_edges)
        num_nodes: Total number of nodes

    Returns:
        isolated_nodes: List of isolated node indices
        isolated_ratio: Ratio of isolated nodes to total nodes
    """
    degrees = compute_degrees(edge_index, num_nodes)
    isolated_mask = degrees == 0
    isolated_nodes = np.where(isolated_mask)[0].tolist()
    isolated_ratio = len(isolated_nodes) / num_nodes if num_nodes > 0 else 0.0

    return isolated_nodes, isolated_ratio


def find_connected_components(edge_index: np.ndarray, num_nodes: int) -> List[Set[int]]:
    """
    Find connected components in the graph using BFS

    Args:
        edge_index: Edge index array (2, num_edges)
        num_nodes: Total number of nodes

    Returns:
        components: List of connected components (each is a set of node indices)
    """
    if edge_index.shape[1] == 0:
        # No edges, each node is its own component
        return [{i} for i in range(num_nodes)]

    # Build adjacency list
    adj_list = defaultdict(set)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_list[src].add(dst)
        adj_list[dst].add(src)

    # BFS to find connected components
    visited = set()
    components = []

    for node in range(num_nodes):
        if node not in visited:
            component = set()
            queue = [node]
            visited.add(node)

            while queue:
                current = queue.pop(0)
                component.add(current)

                for neighbor in adj_list.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

    return components


def analyze_graph_structure(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    num_nodes: int,
    node_labels: Dict[int, int] = None
) -> Dict:
    """
    Comprehensive analysis of graph structure

    Args:
        edge_index: Edge index array (2, num_edges)
        edge_attr: Edge attribute array (num_edges,)
        num_nodes: Total number of nodes
        node_labels: Optional mapping from node index to class label (0-8)

    Returns:
        analysis: Dictionary containing comprehensive graph statistics
    """
    num_edges = edge_index.shape[1]

    # Basic statistics
    degrees = compute_degrees(edge_index, num_nodes)
    isolated_nodes, isolated_ratio = find_isolated_nodes(edge_index, num_nodes)
    components = find_connected_components(edge_index, num_nodes)

    # Degree statistics
    avg_degree = np.mean(degrees) if len(degrees) > 0 else 0.0
    median_degree = np.median(degrees) if len(degrees) > 0 else 0.0
    max_degree = np.max(degrees) if len(degrees) > 0 else 0
    min_degree = np.min(degrees) if len(degrees) > 0 else 0

    # Graph density
    max_possible_edges = num_nodes * (num_nodes - 1) / 2
    graph_density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0

    # Edge weight statistics
    if len(edge_attr) > 0:
        avg_weight = np.mean(edge_attr)
        median_weight = np.median(edge_attr)
        max_weight = np.max(edge_attr)
        min_weight = np.min(edge_attr)
    else:
        avg_weight = median_weight = max_weight = min_weight = 0.0

    # Connected component statistics
    component_sizes = sorted([len(c) for c in components], reverse=True)
    num_components = len(components)
    largest_component_size = component_sizes[0] if component_sizes else 0
    largest_component_ratio = largest_component_size / num_nodes if num_nodes > 0 else 0.0

    # Per-class isolated node analysis (if labels provided)
    class_isolated_stats = None
    if node_labels:
        class_isolated_stats = defaultdict(lambda: {'total': 0, 'isolated': 0})
        for node_idx in range(num_nodes):
            if node_idx in node_labels:
                label = node_labels[node_idx]
                class_isolated_stats[label]['total'] += 1
                if node_idx in isolated_nodes:
                    class_isolated_stats[label]['isolated'] += 1

        # Convert to dict and compute ratios
        class_isolated_stats = dict(class_isolated_stats)
        for label in class_isolated_stats:
            stats = class_isolated_stats[label]
            stats['isolated_ratio'] = stats['isolated'] / stats['total'] if stats['total'] > 0 else 0.0

    analysis = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'median_degree': median_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'isolated_nodes': isolated_nodes,
        'num_isolated': len(isolated_nodes),
        'isolated_ratio': isolated_ratio,
        'graph_density': graph_density,
        'num_components': num_components,
        'component_sizes': component_sizes[:10],  # Top 10 components
        'largest_component_size': largest_component_size,
        'largest_component_ratio': largest_component_ratio,
        'edge_weight_stats': {
            'avg': avg_weight,
            'median': median_weight,
            'max': max_weight,
            'min': min_weight
        },
        'class_isolated_stats': class_isolated_stats
    }

    return analysis


def print_analysis_report(analysis: Dict, graph_name: str = "Graph"):
    """
    Print formatted analysis report

    Args:
        analysis: Analysis dictionary from analyze_graph_structure
        graph_name: Name of the graph for reporting
    """
    print(f"\n{'='*80}")
    print(f"Graph Analysis Report: {graph_name}")
    print(f"{'='*80}\n")

    print("Basic Structure:")
    print(f"  Nodes: {analysis['num_nodes']:,}")
    print(f"  Edges: {analysis['num_edges']:,}")
    print(f"  Graph Density: {analysis['graph_density']:.6f}")

    print("\nDegree Statistics:")
    print(f"  Average Degree: {analysis['avg_degree']:.2f}")
    print(f"  Median Degree: {analysis['median_degree']:.2f}")
    print(f"  Min Degree: {analysis['min_degree']}")
    print(f"  Max Degree: {analysis['max_degree']}")

    print("\nIsolated Nodes (CRITICAL ISSUE):")
    print(f"  Isolated Nodes: {analysis['num_isolated']:,} / {analysis['num_nodes']:,}")
    print(f"  Isolated Ratio: {analysis['isolated_ratio']*100:.2f}%")

    if analysis['num_isolated'] > 0:
        print(f"  ⚠️  WARNING: {analysis['num_isolated']} nodes have NO connections!")
        print(f"  This severely limits GAT's ability to learn spatial patterns.")

    print("\nConnected Components:")
    print(f"  Number of Components: {analysis['num_components']:,}")
    print(f"  Largest Component: {analysis['largest_component_size']:,} nodes ({analysis['largest_component_ratio']*100:.2f}%)")

    if len(analysis['component_sizes']) > 1:
        print(f"  Top 10 Component Sizes: {analysis['component_sizes'][:10]}")

    print("\nEdge Weight Statistics:")
    ew = analysis['edge_weight_stats']
    print(f"  Average Weight: {ew['avg']:.4f}")
    print(f"  Median Weight: {ew['median']:.4f}")
    print(f"  Min Weight: {ew['min']:.4f}")
    print(f"  Max Weight: {ew['max']:.4f}")

    if analysis['class_isolated_stats']:
        print("\nIsolated Nodes by Class:")
        print(f"  {'Class':<10} {'Total':<10} {'Isolated':<10} {'Ratio':<10}")
        print(f"  {'-'*40}")
        for label in sorted(analysis['class_isolated_stats'].keys()):
            stats = analysis['class_isolated_stats'][label]
            print(f"  {label:<10} {stats['total']:<10} {stats['isolated']:<10} {stats['isolated_ratio']*100:.2f}%")

    print(f"\n{'='*80}\n")


def analyze_flow_graph_from_data(
    od_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    label_df: pd.DataFrame,
    threshold: float
) -> Dict:
    """
    Analyze flow graph built from OD data

    Args:
        od_df: OD flow DataFrame
        metadata_df: Grid metadata DataFrame
        label_df: Label DataFrame
        threshold: Flow threshold for edge creation

    Returns:
        analysis: Graph analysis dictionary
    """
    logger.info(f"Building flow graph with threshold={threshold}")

    # Build graph
    graph_builder = SpatialGraphBuilder(metadata_df, k_neighbors=8)
    edge_index, edge_weights = graph_builder.build_flow_graph(
        od_df,
        threshold=threshold,
        include_neighbors=True
    )

    # Get number of nodes
    num_nodes = len(graph_builder.grid_id_to_idx)

    # Create label mapping (grid_id -> label)
    label_map = {}
    for _, row in label_df.iterrows():
        grid_id = row['grid_id']
        label = row['label'] - 1  # Convert 1-9 to 0-8
        if grid_id in graph_builder.grid_id_to_idx:
            node_idx = graph_builder.grid_id_to_idx[grid_id]
            label_map[node_idx] = label

    # Analyze graph
    analysis = analyze_graph_structure(
        edge_index,
        edge_weights,
        num_nodes,
        label_map
    )

    return analysis, graph_builder


def main():
    parser = argparse.ArgumentParser(description='Analyze graph structure for isolated nodes')
    parser.add_argument('--label-path', type=str, default=config.LABEL_PATH,
                        help='Path to label CSV file')
    parser.add_argument('--threshold', type=float, default=config.FLOW_THRESHOLD,
                        help='Flow threshold for edge creation')
    parser.add_argument('--output-dir', type=str, default='outputs/graph_analysis',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading data and building graphs for analysis...")

    try:
        # Load data using dual_year_processor
        data = prepare_dual_year_experiment_data(
            label_path=args.label_path,
            use_cache=True
        )

        # Analyze 2021 graph
        logger.info("\n" + "="*80)
        logger.info("Analyzing 2021 Flow Graph")
        logger.info("="*80)

        graphs_2021 = data['graphs_2021']
        if len(graphs_2021) > 0:
            edge_index_2021, edge_weights_2021 = graphs_2021[0]

            # Convert torch tensors to numpy if needed
            if hasattr(edge_index_2021, 'cpu'):
                edge_index_2021 = edge_index_2021.cpu().numpy()
                edge_weights_2021 = edge_weights_2021.cpu().numpy()

            # Get label mapping
            labels = data['labels']
            label_map = {idx: label-1 for idx, label in enumerate(labels.values())}

            # Get number of nodes from all_features
            all_features_2021 = data.get('all_features_2021')
            if all_features_2021 is not None:
                num_nodes = all_features_2021.shape[0]
            else:
                num_nodes = edge_index_2021.max() + 1 if edge_index_2021.shape[1] > 0 else 0

            analysis_2021 = analyze_graph_structure(
                edge_index_2021,
                edge_weights_2021,
                num_nodes,
                label_map
            )

            print_analysis_report(analysis_2021, "2021 Flow Graph")

        # Analyze 2024 graph
        logger.info("\n" + "="*80)
        logger.info("Analyzing 2024 Flow Graph")
        logger.info("="*80)

        graphs_2024 = data['graphs_2024']
        if len(graphs_2024) > 0:
            edge_index_2024, edge_weights_2024 = graphs_2024[0]

            # Convert torch tensors to numpy if needed
            if hasattr(edge_index_2024, 'cpu'):
                edge_index_2024 = edge_index_2024.cpu().numpy()
                edge_weights_2024 = edge_weights_2024.cpu().numpy()

            all_features_2024 = data.get('all_features_2024')
            if all_features_2024 is not None:
                num_nodes = all_features_2024.shape[0]
            else:
                num_nodes = edge_index_2024.max() + 1 if edge_index_2024.shape[1] > 0 else 0

            analysis_2024 = analyze_graph_structure(
                edge_index_2024,
                edge_weights_2024,
                num_nodes,
                label_map
            )

            print_analysis_report(analysis_2024, "2024 Flow Graph")

        # Summary and recommendations
        print("\n" + "="*80)
        print("SUMMARY AND RECOMMENDATIONS")
        print("="*80 + "\n")

        if analysis_2021['isolated_ratio'] > 0.05:
            print("⚠️  CRITICAL ISSUE DETECTED:")
            print(f"   {analysis_2021['isolated_ratio']*100:.2f}% of nodes are isolated!")
            print("\n📋 RECOMMENDED ACTIONS:")
            print("   1. Enable self-loops: ADD_SELF_LOOPS = True in config.py")
            print("   2. Enable KNN fallback: USE_KNN_FALLBACK = True in config.py")
            print("   3. Consider lowering FLOW_THRESHOLD (current: {})".format(args.threshold))
        elif analysis_2021['isolated_ratio'] > 0.01:
            print("⚠️  MODERATE ISSUE DETECTED:")
            print(f"   {analysis_2021['isolated_ratio']*100:.2f}% of nodes are isolated")
            print("\n📋 RECOMMENDED ACTIONS:")
            print("   1. Enable self-loops: ADD_SELF_LOOPS = True in config.py")
            print("   2. Monitor if KNN fallback is needed")
        else:
            print("✅ Graph structure is healthy!")
            print(f"   Only {analysis_2021['isolated_ratio']*100:.2f}% isolated nodes")

        print("\n" + "="*80 + "\n")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
