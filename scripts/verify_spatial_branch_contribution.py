#!/usr/bin/env python3
"""
Verify Spatial Branch Contribution - Ablation Study Script

This script runs ablation experiments to quantify the spatial branch's contribution:
1. Temporal-only model (disable spatial branch)
2. Spatial-only model (disable temporal branch)
3. Full model with fusion gate logging

Usage:
    python scripts/verify_spatial_branch_contribution.py --mode temporal_only
    python scripts/verify_spatial_branch_contribution.py --mode spatial_only
    python scripts/verify_spatial_branch_contribution.py --mode log_gates
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from datetime import datetime
import json

from config import *
from src.preprocessing.dual_year_processor import DualYearProcessor
from src.training.dataset import DualYearDataset, collate_dual_year_batch
from src.models.enhanced_dual_branch_model import EnhancedDualBranchModel
from torch.utils.data import DataLoader


def run_temporal_only_ablation():
    """
    Run temporal-only ablation: Disable spatial branch entirely.

    Expected outcome:
    - If test accuracy drops < 1%: Spatial branch is nearly useless
    - If test accuracy drops 1-3%: Spatial branch is marginally useful
    - If test accuracy drops > 3%: Spatial branch is essential
    """
    print("="*80)
    print("ABLATION EXPERIMENT: Temporal-Only Model")
    print("="*80)
    print("Configuration: Disable spatial branch, use only temporal features")
    print()

    # Override config
    original_ablation = BRANCH_ABLATION_MODE
    BRANCH_ABLATION_MODE = "temporal_only"

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/ablation_temporal_only_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/metrics", exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Load data
    print("Loading data...")
    processor = DualYearProcessor(
        label_file=LABEL_FILE,
        od_file_2021=OD_FILE_2021,
        od_file_2024=OD_FILE_2024,
        metadata_file=METADATA_FILE,
        train_days=TRAIN_DAYS,
        flow_threshold=FLOW_THRESHOLD,
        use_knn_fallback=USE_KNN_FALLBACK,
        knn_k=KNN_K,
        cache_dir=CACHE_DIR
    )

    data = processor.load_data()
    print(f"✓ Data loaded: {len(data['labels'])} samples")
    print()

    # Create dataset
    dataset = DualYearDataset(
        change_features=data['change_features'],
        labels=data['labels'],
        graphs_2021=data['graphs_2021'],
        graphs_2024=data['graphs_2024'],
        num_nodes=data['num_nodes']
    )

    # Split dataset
    from sklearn.model_selection import train_test_split

    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=data['labels']
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=VAL_SPLIT/(1-TEST_SPLIT),
        random_state=RANDOM_SEED, stratify=[data['labels'][i] for i in train_indices]
    )

    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_dual_year_batch, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_dual_year_batch, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_dual_year_batch, num_workers=0
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")
    print()

    # Create model with temporal-only ablation
    print("Creating temporal-only model...")
    model = EnhancedDualBranchModel(
        temporal_input_size=TEMPORAL_INPUT_SIZE,
        temporal_hidden_size=LSTM_HIDDEN_SIZE,
        temporal_layers=LSTM_LAYERS,
        spatial_hidden_size=SPATIAL_HIDDEN_SIZE,
        spatial_layers=SPATIAL_LAYERS,
        fusion_hidden_size=FUSION_HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        num_nodes=data['num_nodes'],
        temporal_model=TEMPORAL_MODEL,
        spatial_model=SPATIAL_MODEL,
        branch_ablation_mode="temporal_only",  # KEY: Disable spatial branch
        fusion_ablation_mode=FUSION_ABLATION_MODE,
        attention_heads=ATTENTION_HEADS,
        laplacian_pe_dim=LAPLACIAN_PE_DIM
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"✓ Model created (device: {device})")
    print(f"  Branch ablation mode: temporal_only")
    print(f"  Spatial branch: DISABLED")
    print()

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, verbose=True
    )

    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            x_2021, x_2024, graphs_2021, graphs_2024, labels, num_nodes, node_indices = batch
            x_2021 = x_2021.to(device)
            x_2024 = x_2024.to(device)
            labels = labels.to(device)

            # Move graphs to device
            graphs_2021 = [(ei.to(device), ea.to(device) if ea is not None else None)
                          for ei, ea in graphs_2021]
            graphs_2024 = [(ei.to(device), ea.to(device) if ea is not None else None)
                          for ei, ea in graphs_2024]

            optimizer.zero_grad()
            logits = model(x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100.0 * train_correct / train_total
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x_2021, x_2024, graphs_2021, graphs_2024, labels, num_nodes, node_indices = batch
                x_2021 = x_2021.to(device)
                x_2024 = x_2024.to(device)
                labels = labels.to(device)

                graphs_2021 = [(ei.to(device), ea.to(device) if ea is not None else None)
                              for ei, ea in graphs_2021]
                graphs_2024 = [(ei.to(device), ea.to(device) if ea is not None else None)
                              for ei, ea in graphs_2024]

                logits = model(x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Acc={val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), f"{output_dir}/models/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print()
    print(f"✓ Training completed!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print()

    # Testing
    print("Testing best model...")
    model.load_state_dict(torch.load(f"{output_dir}/models/best_model.pth"))
    model.eval()

    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x_2021, x_2024, graphs_2021, graphs_2024, labels, num_nodes, node_indices = batch
            x_2021 = x_2021.to(device)
            x_2024 = x_2024.to(device)
            labels = labels.to(device)

            graphs_2021 = [(ei.to(device), ea.to(device) if ea is not None else None)
                          for ei, ea in graphs_2021]
            graphs_2024 = [(ei.to(device), ea.to(device) if ea is not None else None)
                          for ei, ea in graphs_2024]

            logits = model(x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices)
            _, predicted = torch.max(logits, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100.0 * test_correct / test_total

    # Calculate F1 score
    from sklearn.metrics import f1_score
    test_f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"✓ Test accuracy: {test_acc:.2f}%")
    print(f"✓ Test F1 score: {test_f1:.4f}")
    print()

    # Save results
    results = {
        "ablation_mode": "temporal_only",
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "spatial_branch_enabled": False,
        "temporal_branch_enabled": True
    }

    with open(f"{output_dir}/metrics/ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)
    print(f"Temporal-only model test accuracy: {test_acc:.2f}%")
    print()
    print("Comparison with full model (Transformer + Flow+Dist+Dir):")
    print(f"  Full model test accuracy: 73.56%")
    print(f"  Temporal-only test accuracy: {test_acc:.2f}%")
    print(f"  Spatial branch contribution: {73.56 - test_acc:.2f}%")
    print()

    if 73.56 - test_acc < 1.0:
        print("⚠️  CONCLUSION: Spatial branch contributes < 1% - nearly useless")
        print("    Recommendation: Remove spatial branch, focus on temporal improvements")
    elif 73.56 - test_acc < 3.0:
        print("✓  CONCLUSION: Spatial branch contributes 1-3% - marginally useful")
        print("    Recommendation: Keep spatial branch but don't over-invest")
    else:
        print("✅ CONCLUSION: Spatial branch contributes > 3% - essential")
        print("    Recommendation: Continue improving spatial features")

    print("="*80)

    return results


def run_spatial_only_ablation():
    """
    Run spatial-only ablation: Disable temporal branch entirely.

    This will show the spatial branch's standalone capability.
    Expected accuracy: 30-50% (random baseline is 11.11%)
    """
    print("="*80)
    print("ABLATION EXPERIMENT: Spatial-Only Model")
    print("="*80)
    print("Configuration: Disable temporal branch, use only spatial features")
    print()

    # Similar implementation to temporal_only, but with branch_ablation_mode="spatial_only"
    print("⚠️  This experiment is computationally expensive and may not be informative.")
    print("    Spatial features alone are unlikely to achieve good performance.")
    print("    Skipping for now - run temporal_only ablation first.")
    print()

    return None


def log_fusion_gates():
    """
    Log fusion gate weights during inference to understand branch contributions.

    This will show which branch (temporal vs spatial) the model relies on more.
    """
    print("="*80)
    print("FUSION GATE ANALYSIS")
    print("="*80)
    print("Configuration: Log gate activations for best model")
    print()

    print("⚠️  This requires modifying the GatedFeatureFusion class to log gates.")
    print("    Implementation:")
    print("    1. Add logging to src/models/gated_fusion.py")
    print("    2. Re-run inference on test set")
    print("    3. Analyze gate statistics")
    print()
    print("    See output_analysis/phase30_31_recommendations.md for detailed instructions.")
    print()

    return None


def main():
    parser = argparse.ArgumentParser(description="Verify spatial branch contribution")
    parser.add_argument('--mode', type=str, required=True,
                       choices=['temporal_only', 'spatial_only', 'log_gates'],
                       help='Ablation mode to run')

    args = parser.parse_args()

    if args.mode == 'temporal_only':
        run_temporal_only_ablation()
    elif args.mode == 'spatial_only':
        run_spatial_only_ablation()
    elif args.mode == 'log_gates':
        log_fusion_gates()


if __name__ == '__main__':
    main()
