"""
Quick test script to verify graph optimizations work correctly
"""
import sys
import numpy as np
import pandas as pd
from src.preprocessing.graph_builder import SpatialGraphBuilder

# Load metadata
metadata_df = pd.read_csv('data/grid_metadata/sgh_grid_metadata.csv')

# Create a small test OD dataframe
test_od_data = {
    'o_grid_500': [1, 2, 3, 1, 2],
    'd_grid_500': [2, 3, 1, 3, 1],
    'num_total': [15, 20, 12, 18, 25]
}
test_od_df = pd.DataFrame(test_od_data)

print("Testing graph optimizations...")
print(f"Metadata: {len(metadata_df)} grids")
print(f"Test OD data: {len(test_od_df)} rows\n")

# Initialize graph builder
graph_builder = SpatialGraphBuilder(metadata_df, k_neighbors=8)

# Test 1: Build flow graph without optimizations (baseline)
print("="*80)
print("Test 1: Baseline (no optimizations)")
print("="*80)
import config

# Temporarily disable optimizations
original_add_self_loops = config.ADD_SELF_LOOPS
original_use_knn_fallback = config.USE_KNN_FALLBACK

config.ADD_SELF_LOOPS = False
config.USE_KNN_FALLBACK = False

edge_index_baseline, edge_weights_baseline = graph_builder.build_flow_graph(
    test_od_df, threshold=10.0, include_neighbors=True
)

from src.analysis.graph_analyzer import compute_degrees, find_isolated_nodes
num_nodes_baseline = edge_index_baseline.shape[1] + 1 if edge_index_baseline.shape[1] > 0 else len(metadata_df)
degrees_baseline = compute_degrees(edge_index_baseline, num_nodes_baseline)
isolated_baseline, ratio_baseline = find_isolated_nodes(edge_index_baseline, num_nodes_baseline)

print(f"Nodes: {num_nodes_baseline}")
print(f"Edges: {edge_index_baseline.shape[1]}")
print(f"Isolated nodes: {len(isolated_baseline)} ({ratio_baseline*100:.2f}%)")
print()

# Test 2: With self-loops only
print("="*80)
print("Test 2: With self-loops (Phase 2.1)")
print("="*80)
config.ADD_SELF_LOOPS = True
config.USE_KNN_FALLBACK = False

# Reset graph builder to use original grid_id_to_idx
graph_builder2 = SpatialGraphBuilder(metadata_df, k_neighbors=8)
edge_index_loops, edge_weights_loops = graph_builder2.build_flow_graph(
    test_od_df, threshold=10.0, include_neighbors=True
)

num_nodes_loops = edge_index_loops.shape[1] + 1 if edge_index_loops.shape[1] > 0 else len(metadata_df)
degrees_loops = compute_degrees(edge_index_loops, num_nodes_loops)
isolated_loops, ratio_loops = find_isolated_nodes(edge_index_loops, num_nodes_loops)

print(f"Nodes: {num_nodes_loops}")
print(f"Edges: {edge_index_loops.shape[1]}")
print(f"Isolated nodes: {len(isolated_loops)} ({ratio_loops*100:.2f}%)")
print(f"Edges added by self-loops: {edge_index_loops.shape[1] - edge_index_baseline.shape[1]}")
print()

# Test 3: With both self-loops and KNN fallback
print("="*80)
print("Test 3: With self-loops + KNN fallback (Phase 2.2)")
print("="*80)
config.ADD_SELF_LOOPS = True
config.USE_KNN_FALLBACK = True

graph_builder3 = SpatialGraphBuilder(metadata_df, k_neighbors=8)
edge_index_full, edge_weights_full = graph_builder3.build_flow_graph(
    test_od_df, threshold=10.0, include_neighbors=True
)

num_nodes_full = edge_index_full.shape[1] + 1 if edge_index_full.shape[1] > 0 else len(metadata_df)
degrees_full = compute_degrees(edge_index_full, num_nodes_full)
isolated_full, ratio_full = find_isolated_nodes(edge_index_full, num_nodes_full)

print(f"Nodes: {num_nodes_full}")
print(f"Edges: {edge_index_full.shape[1]}")
print(f"Isolated nodes: {len(isolated_full)} ({ratio_full*100:.2f}%)")
print(f"Edges added by optimizations: {edge_index_full.shape[1] - edge_index_baseline.shape[1]}")
print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Baseline isolated nodes: {len(isolated_baseline)} ({ratio_baseline*100:.2f}%)")
print(f"With self-loops: {len(isolated_loops)} ({ratio_loops*100:.2f}%)")
print(f"With self-loops + KNN: {len(isolated_full)} ({ratio_full*100:.2f}%)")
print()

if len(isolated_full) == 0:
    print("✅ SUCCESS: All optimizations working correctly!")
    print("   - Self-loops added")
    print("   - KNN fallback eliminating remaining isolated nodes")
elif len(isolated_full) < len(isolated_baseline):
    print(f"⚠️  PARTIAL SUCCESS: Reduced isolated nodes from {len(isolated_baseline)} to {len(isolated_full)}")
else:
    print("❌ FAILURE: Optimizations not working as expected")

# Restore config
config.ADD_SELF_LOOPS = original_add_self_loops
config.USE_KNN_FALLBACK = original_use_knn_fallback
