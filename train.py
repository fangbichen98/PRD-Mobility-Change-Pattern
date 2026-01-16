"""
Main training script for mobility pattern classification
"""
import torch
import numpy as np
import pandas as pd
import logging
import os
import json
from datetime import datetime

import config
from src.preprocessing.data_processor import (
    ODFlowProcessor, GridMetadataProcessor, LabelProcessor,
    build_temporal_features, aggregate_grid_flows
)
from src.preprocessing.graph_builder import SpatialGraphBuilder
from src.models.dual_branch_model import DualBranchSTModel, BaselineLSTM, BaselineGAT
from src.training.dataset import MobilityDataset
from src.training.trainer import Trainer, create_data_loaders
from src.evaluation.evaluator import Evaluator, compare_models
from src.visualization.visualizer import SpatialVisualizer, TemporalVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data(year=2021, sample_size=None):
    """
    Prepare data for training

    Args:
        year: Year of data to use (2021 or 2024)
        sample_size: Number of rows to sample (None for all)

    Returns:
        Tuple of processed data components
    """
    logger.info("=" * 50)
    logger.info("Step 1: Data Preparation")
    logger.info("=" * 50)

    # Load metadata
    logger.info("Loading grid metadata...")
    metadata_processor = GridMetadataProcessor()
    metadata_df = metadata_processor.load_and_validate()
    valid_grid_ids = metadata_processor.get_valid_grid_ids(metadata_df)

    # Load labels
    logger.info("Loading labels...")
    label_processor = LabelProcessor()
    label_df = label_processor.load_and_validate(valid_grid_ids)

    # Create label dictionary (0-indexed)
    labels = dict(zip(label_df['grid_id'], label_df['label_idx']))

    # Load OD flow data
    logger.info(f"Loading OD flow data for {year}...")
    od_processor = ODFlowProcessor(year)
    od_df = od_processor.load_and_preprocess(nrows=sample_size)
    od_df = od_processor.filter_training_period(od_df)
    od_df = od_processor.validate_grid_ids(od_df, valid_grid_ids)
    od_df, norm_params = od_processor.normalize_flow(od_df)

    # Build temporal features
    logger.info("Building temporal features...")
    od_df = build_temporal_features(od_df)

    # Aggregate grid flows
    logger.info("Aggregating grid flows...")
    labeled_grid_ids = list(labels.keys())
    grid_flows = aggregate_grid_flows(od_df, labeled_grid_ids)

    # Build spatial graph
    logger.info("Building spatial graph...")
    graph_builder = SpatialGraphBuilder(metadata_df, k_neighbors=8)
    edge_index, edge_weights = graph_builder.build_hybrid_graph(od_df)

    # Create grid_id to index mapping
    grid_id_to_idx = {gid: idx for idx, gid in enumerate(labeled_grid_ids)}

    logger.info(f"Data preparation completed:")
    logger.info(f"  - Total grids: {len(metadata_df)}")
    logger.info(f"  - Labeled grids: {len(labels)}")
    logger.info(f"  - OD records: {len(od_df)}")
    logger.info(f"  - Graph edges: {edge_index.shape[1]}")

    return {
        'metadata_df': metadata_df,
        'labels': labels,
        'grid_flows': grid_flows,
        'edge_index': edge_index,
        'edge_weights': edge_weights,
        'grid_id_to_idx': grid_id_to_idx,
        'norm_params': norm_params
    }


def train_model(data, model_type='dual_branch', device='cuda'):
    """
    Train model

    Args:
        data: Prepared data dictionary
        model_type: Type of model ('dual_branch', 'lstm', 'gat')
        device: Device to use

    Returns:
        Trained model and results
    """
    logger.info("=" * 50)
    logger.info(f"Step 2: Training {model_type.upper()} Model")
    logger.info("=" * 50)

    # Create dataset
    dataset = MobilityDataset(
        grid_flows=data['grid_flows'],
        labels=data['labels'],
        grid_ids=list(data['labels'].keys()),
        metadata_df=data['metadata_df']
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        edge_index=data['edge_index'],
        edge_attr=data['edge_weights'],
        grid_id_to_idx=data['grid_id_to_idx']
    )

    # Initialize model
    if model_type == 'dual_branch':
        model = DualBranchSTModel()
    elif model_type == 'lstm':
        model = BaselineLSTM()
    elif model_type == 'gat':
        model = BaselineGAT()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    best_acc, best_f1 = trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluator = Evaluator(model, test_loader, device=device)
    results = evaluator.evaluate()

    # Generate report
    evaluator.generate_report(results, model_name=model_type)

    return model, results, test_loader


def run_experiments(data, device='cuda'):
    """
    Run all experiments including baseline comparisons

    Args:
        data: Prepared data dictionary
        device: Device to use

    Returns:
        Dictionary of results for all models
    """
    logger.info("=" * 50)
    logger.info("Running All Experiments")
    logger.info("=" * 50)

    all_results = {}

    # Train dual-branch model
    logger.info("\n" + "=" * 50)
    logger.info("Experiment 1: Dual-Branch Model")
    logger.info("=" * 50)
    model_dual, results_dual, test_loader = train_model(data, 'dual_branch', device)
    all_results['Dual-Branch'] = results_dual

    # Train baseline LSTM
    logger.info("\n" + "=" * 50)
    logger.info("Experiment 2: Baseline LSTM")
    logger.info("=" * 50)
    model_lstm, results_lstm, _ = train_model(data, 'lstm', device)
    all_results['LSTM'] = results_lstm

    # Train baseline GAT
    logger.info("\n" + "=" * 50)
    logger.info("Experiment 3: Baseline GAT")
    logger.info("=" * 50)
    model_gat, results_gat, _ = train_model(data, 'gat', device)
    all_results['GAT'] = results_gat

    # Compare models
    logger.info("\n" + "=" * 50)
    logger.info("Model Comparison")
    logger.info("=" * 50)

    comparison_path = os.path.join(config.FIGURE_DIR, 'model_comparison.jpg')
    compare_models(all_results, save_path=comparison_path)

    # Print comparison table
    logger.info("\nComparison Table:")
    logger.info("-" * 70)
    logger.info(f"{'Model':<15} {'Accuracy':<12} {'F1 (Macro)':<12} {'Precision':<12} {'Recall':<12}")
    logger.info("-" * 70)
    for model_name, results in all_results.items():
        logger.info(
            f"{model_name:<15} "
            f"{results['accuracy']:<12.4f} "
            f"{results['f1_macro']:<12.4f} "
            f"{results['precision']:<12.4f} "
            f"{results['recall']:<12.4f}"
        )
    logger.info("-" * 70)

    return all_results, model_dual, test_loader


def generate_visualizations(data, results, model):
    """
    Generate visualization outputs

    Args:
        data: Prepared data dictionary
        results: Evaluation results
        model: Trained model
    """
    logger.info("=" * 50)
    logger.info("Step 3: Generating Visualizations")
    logger.info("=" * 50)

    # Spatial visualizations
    logger.info("Creating spatial visualizations...")
    spatial_viz = SpatialVisualizer(data['metadata_df'])

    # Plot true label distribution
    spatial_viz.plot_spatial_distribution(
        grid_labels=data['labels'],
        title='True Mobility Pattern Distribution',
        save_path=os.path.join(config.FIGURE_DIR, 'true_label_distribution.jpg')
    )

    # Plot predicted label distribution
    predictions = dict(zip(results['grid_ids'], results['predictions']))
    spatial_viz.plot_spatial_distribution(
        grid_labels=predictions,
        title='Predicted Mobility Pattern Distribution',
        save_path=os.path.join(config.FIGURE_DIR, 'predicted_label_distribution.jpg')
    )

    # Temporal visualizations
    logger.info("Creating temporal visualizations...")
    temporal_viz = TemporalVisualizer()

    # Plot class temporal patterns
    temporal_viz.plot_class_temporal_patterns(
        grid_flows=data['grid_flows'],
        grid_labels=data['labels'],
        save_path=os.path.join(config.FIGURE_DIR, 'class_temporal_patterns.jpg')
    )

    # Plot sample temporal series for each class
    for class_idx in range(config.NUM_CLASSES):
        # Find grids of this class
        class_grids = [gid for gid, label in data['labels'].items() if label == class_idx]

        if len(class_grids) > 0:
            # Select up to 3 representative grids
            sample_grids = class_grids[:min(3, len(class_grids))]

            temporal_viz.plot_temporal_series(
                grid_flows=data['grid_flows'],
                grid_ids=sample_grids,
                labels=data['labels'],
                save_path=os.path.join(config.FIGURE_DIR, f'temporal_series_class_{class_idx + 1}.jpg')
            )

    logger.info(f"Visualizations saved to {config.FIGURE_DIR}")


def main():
    """Main execution function"""
    logger.info("=" * 50)
    logger.info("Mobility Pattern Classification")
    logger.info("=" * 50)

    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Prepare data
    # NOTE: For testing, use sample_size parameter to limit data
    # For full training, set sample_size=None
    data = prepare_data(year=2021, sample_size=5000000)  # 5M rows for faster testing

    # Run experiments
    all_results, best_model, test_loader = run_experiments(data, device=device)

    # Generate visualizations
    generate_visualizations(data, all_results['Dual-Branch'], best_model)

    # Save final results
    results_path = os.path.join(config.OUTPUT_DIR, 'final_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, results in all_results.items():
            json_results[model_name] = {
                'accuracy': results['accuracy'],
                'f1_macro': results['f1_macro'],
                'f1_weighted': results['f1_weighted'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_per_class': results['f1_per_class']
            }
        json.dump(json_results, f, indent=2)

    logger.info(f"Final results saved to {results_path}")
    logger.info("=" * 50)
    logger.info("Training completed successfully!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
