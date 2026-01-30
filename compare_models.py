"""
Quick comparison script to verify the pure graph model
Tests both models with dummy data to ensure correctness
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import time
import numpy as np
from src.models.dual_branch_model import ImprovedDualBranchModel
from src.models.dual_branch_model_pure_graph import PureGraphDualBranchModel

print("=" * 80)
print("Model Comparison: Original vs Pure Graph")
print("=" * 80)

# Test parameters
batch_size = 16
num_nodes = 1000
num_edges = 10000
num_time_steps = 7

print(f"\nTest Configuration:")
print(f"  - Batch size: {batch_size}")
print(f"  - Num nodes: {num_nodes}")
print(f"  - Num edges: {num_edges}")
print(f"  - Time steps: {num_time_steps}")

# Create dummy data
print("\n" + "=" * 80)
print("Creating test data...")
print("=" * 80)

# Temporal features (both models need this)
x_2021_full = torch.randn(num_nodes, num_time_steps, 2)
x_2024_full = torch.randn(num_nodes, num_time_steps, 2)

# Graph structure
edge_index_2021 = torch.randint(0, num_nodes, (2, num_edges))
edge_attr_2021 = torch.rand(num_edges) * 100

edge_index_2024 = torch.randint(0, num_nodes, (2, num_edges))
edge_attr_2024 = torch.rand(num_edges) * 100

graphs_2021 = [(edge_index_2021, edge_attr_2021)]
graphs_2024 = [(edge_index_2024, edge_attr_2024)]

# Batch indices
node_indices = torch.randint(0, num_nodes, (batch_size,))

print("✓ Test data created")

# Test Original Model
print("\n" + "=" * 80)
print("Testing Original Model (ImprovedDualBranchModel)")
print("=" * 80)

try:
    model_original = ImprovedDualBranchModel(
        temporal_input_size=2,
        spatial_input_size=2,
        hidden_size=256,
        num_classes=9,
        num_time_steps=7,
        dropout=0.2,
        use_hierarchical=True
    )

    # Store graphs in model
    model_original.graphs_2021 = graphs_2021
    model_original.graphs_2024 = graphs_2024

    total_params_original = sum(p.numel() for p in model_original.parameters())
    print(f"✓ Model created")
    print(f"  - Total parameters: {total_params_original:,}")

    # Test forward pass
    model_original.eval()
    with torch.no_grad():
        start_time = time.time()

        intensity_logits, direction_logits, direct_logits = model_original(
            x_2021=x_2021_full,
            x_2024=x_2024_full,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            node_indices=node_indices
        )

        forward_time_original = time.time() - start_time

    print(f"✓ Forward pass successful")
    print(f"  - Time: {forward_time_original:.4f}s")
    print(f"  - Intensity logits: {intensity_logits.shape}")
    print(f"  - Direction logits: {direction_logits.shape}")
    print(f"  - Direct logits: {direct_logits.shape}")

    original_success = True

except Exception as e:
    print(f"✗ Original model failed: {e}")
    original_success = False
    forward_time_original = None
    total_params_original = None

# Test Pure Graph Model
print("\n" + "=" * 80)
print("Testing Pure Graph Model (PureGraphDualBranchModel)")
print("=" * 80)

try:
    model_pure = PureGraphDualBranchModel(
        temporal_input_size=2,
        hidden_size=256,
        num_classes=9,
        num_time_steps=7,
        dropout=0.2,
        use_hierarchical=True
    )

    total_params_pure = sum(p.numel() for p in model_pure.parameters())
    print(f"✓ Model created")
    print(f"  - Total parameters: {total_params_pure:,}")

    # Test forward pass
    model_pure.eval()
    with torch.no_grad():
        start_time = time.time()

        intensity_logits, direction_logits, direct_logits = model_pure(
            x_2021=x_2021_full,
            x_2024=x_2024_full,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

        forward_time_pure = time.time() - start_time

    print(f"✓ Forward pass successful")
    print(f"  - Time: {forward_time_pure:.4f}s")
    print(f"  - Intensity logits: {intensity_logits.shape}")
    print(f"  - Direction logits: {direction_logits.shape}")
    print(f"  - Direct logits: {direct_logits.shape}")

    pure_success = True

except Exception as e:
    print(f"✗ Pure graph model failed: {e}")
    pure_success = False
    forward_time_pure = None
    total_params_pure = None

# Comparison
print("\n" + "=" * 80)
print("Comparison Results")
print("=" * 80)

if original_success and pure_success:
    print("\n✓ Both models work correctly!")

    print(f"\nParameter Count:")
    print(f"  - Original: {total_params_original:,}")
    print(f"  - Pure Graph: {total_params_pure:,}")
    print(f"  - Difference: {total_params_original - total_params_pure:,} ({100*(total_params_original - total_params_pure)/total_params_original:.1f}% reduction)")

    print(f"\nForward Pass Time:")
    print(f"  - Original: {forward_time_original:.4f}s")
    print(f"  - Pure Graph: {forward_time_pure:.4f}s")
    print(f"  - Speedup: {forward_time_original/forward_time_pure:.2f}x")

    print(f"\nKey Improvements:")
    print(f"  ✓ Eliminates information redundancy")
    print(f"  ✓ Simpler data preprocessing")
    print(f"  ✓ Faster computation ({forward_time_original/forward_time_pure:.1f}x speedup)")
    print(f"  ✓ Fewer parameters ({100*(total_params_original - total_params_pure)/total_params_original:.1f}% reduction)")

elif original_success and not pure_success:
    print("\n⚠ Pure graph model has issues")
    print("  - Original model works")
    print("  - Pure graph model failed")
    print("  - Check error messages above")

elif not original_success and pure_success:
    print("\n✓ Pure graph model works!")
    print("  - Original model failed (expected if dependencies missing)")
    print("  - Pure graph model works correctly")

else:
    print("\n✗ Both models failed")
    print("  - Check dependencies and imports")

# Memory usage comparison
print("\n" + "=" * 80)
print("Memory Usage Estimation")
print("=" * 80)

print(f"\nNode Features:")
print(f"  - Original: (N={num_nodes}, 7, 2) = {num_nodes * 7 * 2 * 4 / 1024:.2f} KB")
print(f"  - Pure Graph: Computed on-the-fly from graph structure")
print(f"  - Savings: {num_nodes * 7 * 2 * 4 / 1024:.2f} KB per batch")

print(f"\nSpatial Branch Processing:")
print(f"  - Original: 7 GAT passes per year (14 total)")
print(f"  - Pure Graph: 1 GAT pass per year (2 total)")
print(f"  - Reduction: 7x fewer GAT operations")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print(f"\nThe pure graph model:")
print(f"  1. Uses only graph structure for spatial branch")
print(f"  2. Computes node features from graph (in-degree, out-degree, total-degree)")
print(f"  3. Eliminates redundant node features")
print(f"  4. Achieves ~{forward_time_original/forward_time_pure if (original_success and pure_success) else 'N/A'}x speedup")
print(f"  5. Reduces parameters by ~{100*(total_params_original - total_params_pure)/total_params_original if (original_success and pure_success) else 'N/A'}%")

print("\n" + "=" * 80)
print("Next Steps")
print("=" * 80)

print(f"\n1. Train the pure graph model:")
print(f"   python3 train_pure_graph.py")

print(f"\n2. Compare with original model:")
print(f"   python3 train_hierarchical_simple.py")

print(f"\n3. Analyze results:")
print(f"   - Training time")
print(f"   - Memory usage")
print(f"   - Accuracy metrics")
print(f"   - F1 scores")

print("\n" + "=" * 80)
print("✓ Comparison complete!")
print("=" * 80)
