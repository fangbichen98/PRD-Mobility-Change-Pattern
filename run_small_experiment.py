"""
Small-scale experiment: Sample 100 grids per class for quick testing
"""
import pandas as pd
import numpy as np
import torch
import logging
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import config
from src.preprocessing.data_processor import (
    ODFlowProcessor, GridMetadataProcessor, LabelProcessor,
    build_temporal_features, aggregate_grid_flows
)
from src.preprocessing.graph_builder import SpatialGraphBuilder
from src.models.dual_branch_model import DualBranchSTModel, BaselineLSTM, BaselineGAT
from src.training.dataset import MobilityDataset
from src.training.trainer import Trainer, create_data_loaders
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import SpatialVisualizer, TemporalVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sample_balanced_labels(label_path, samples_per_class=100, output_path='data/labels_sampled.csv'):
    """
    Sample balanced labels: 100 samples per class

    Args:
        label_path: Path to original label file
        samples_per_class: Number of samples per class
        output_path: Path to save sampled labels

    Returns:
        DataFrame with sampled labels
    """
    logger.info("=" * 70)
    logger.info("Step 1: Sampling Balanced Labels")
    logger.info("=" * 70)

    # Read labels
    df = pd.read_csv(label_path)
    logger.info(f"Total labels: {len(df)}")

    # Check label distribution
    label_counts = df['label'].value_counts().sort_index()
    logger.info(f"\nOriginal label distribution:")
    for label, count in label_counts.items():
        logger.info(f"  Class {label}: {count} samples")

    # Sample from each class
    sampled_dfs = []
    for label in range(1, config.NUM_CLASSES + 1):
        class_df = df[df['label'] == label]

        if len(class_df) >= samples_per_class:
            sampled = class_df.sample(n=samples_per_class, random_state=config.RANDOM_SEED)
        else:
            logger.warning(f"Class {label} has only {len(class_df)} samples, using all")
            sampled = class_df

        sampled_dfs.append(sampled)

    # Combine
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    # Save
    sampled_df.to_csv(output_path, index=False)
    logger.info(f"\nSampled {len(sampled_df)} total labels")
    logger.info(f"Saved to: {output_path}")

    # Show sampled distribution
    sampled_counts = sampled_df['label'].value_counts().sort_index()
    logger.info(f"\nSampled label distribution:")
    for label, count in sampled_counts.items():
        logger.info(f"  Class {label}: {count} samples")

    return sampled_df


def prepare_small_data(year=2021, label_path='data/labels_sampled.csv'):
    """
    Prepare data for small experiment

    Args:
        year: Year of data to use
        label_path: Path to sampled label file

    Returns:
        Dictionary with prepared data
    """
    logger.info("=" * 70)
    logger.info("Step 2: Data Preparation")
    logger.info("=" * 70)

    # Load metadata
    logger.info("Loading grid metadata...")
    metadata_processor = GridMetadataProcessor()
    metadata_df = metadata_processor.load_and_validate()
    valid_grid_ids = metadata_processor.get_valid_grid_ids(metadata_df)

    # Load sampled labels
    logger.info(f"Loading sampled labels from {label_path}...")
    label_df = pd.read_csv(label_path)

    # Validate labels
    label_df = label_df[label_df['grid_id'].isin(valid_grid_ids)]
    label_df = label_df[label_df['label'].between(1, 9)]
    label_df['label_idx'] = label_df['label'] - 1

    labels = dict(zip(label_df['grid_id'], label_df['label_idx']))
    sampled_grid_ids = set(labels.keys())

    logger.info(f"Valid sampled grids: {len(labels)}")

    # Load OD flow data - only for sampled grids
    logger.info(f"Loading OD flow data for {year}...")
    od_processor = ODFlowProcessor(year)

    # Load in chunks and filter for sampled grids
    chunks = []
    chunksize = 500000

    for chunk in pd.read_csv(od_processor.data_path, chunksize=chunksize):
        # Filter for sampled grids only
        chunk = chunk[
            chunk['o_grid_500'].isin(sampled_grid_ids) |
            chunk['d_grid_500'].isin(sampled_grid_ids)
        ]

        if len(chunk) > 0:
            # Convert date
            chunk['date_dt'] = pd.to_datetime(chunk['date_dt'], format='%Y%m%d')

            # Validate time
            chunk = chunk[chunk['time'].between(0, 23)]

            # Validate num_total
            chunk = chunk[chunk['num_total'] > 0]

            chunks.append(chunk)

    od_df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded {len(od_df)} OD records for sampled grids")

    # Filter training period
    od_df = od_processor.filter_training_period(od_df)
    logger.info(f"After filtering to {config.TRAIN_DAYS} days: {len(od_df)} records")

    # Validate grid IDs
    od_df = od_processor.validate_grid_ids(od_df, valid_grid_ids)

    # Normalize
    od_df, norm_params = od_processor.normalize_flow(od_df)

    # Build temporal features
    logger.info("Building temporal features...")
    od_df = build_temporal_features(od_df)

    # Aggregate grid flows
    logger.info("Aggregating grid flows...")
    labeled_grid_ids = list(labels.keys())
    grid_flows = aggregate_grid_flows(od_df, labeled_grid_ids)

    logger.info(f"Aggregated flows for {len(grid_flows)} grids")

    # Filter metadata to sampled grids
    metadata_sampled = metadata_df[metadata_df['grid_id'].isin(sampled_grid_ids)].copy()

    # Build spatial graph
    logger.info("Building spatial graph...")
    graph_builder = SpatialGraphBuilder(metadata_sampled, k_neighbors=8)
    edge_index, edge_weights = graph_builder.build_hybrid_graph(od_df)

    # Create grid_id to index mapping
    grid_id_to_idx = {gid: idx for idx, gid in enumerate(labeled_grid_ids)}

    logger.info(f"\nData preparation completed:")
    logger.info(f"  - Sampled grids: {len(labels)}")
    logger.info(f"  - OD records: {len(od_df)}")
    logger.info(f"  - Graph edges: {edge_index.shape[1]}")

    return {
        'metadata_df': metadata_sampled,
        'labels': labels,
        'grid_flows': grid_flows,
        'edge_index': edge_index,
        'edge_weights': edge_weights,
        'grid_id_to_idx': grid_id_to_idx,
        'norm_params': norm_params,
        'label_df': label_df
    }


def train_small_model(data, model_type='dual_branch', device='cuda', num_epochs=50):
    """
    Train model on small dataset

    Args:
        data: Prepared data dictionary
        model_type: Type of model
        device: Device to use
        num_epochs: Number of epochs

    Returns:
        Trained model and results
    """
    logger.info("=" * 70)
    logger.info(f"Step 3: Training {model_type.upper()} Model")
    logger.info("=" * 70)

    # Create dataset
    dataset = MobilityDataset(
        grid_flows=data['grid_flows'],
        labels=data['labels'],
        grid_ids=list(data['labels'].keys()),
        metadata_df=data['metadata_df']
    )

    logger.info(f"Dataset size: {len(dataset)} samples")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        edge_index=data['edge_index'],
        edge_attr=data['edge_weights'],
        grid_id_to_idx=data['grid_id_to_idx'],
        batch_size=16  # Smaller batch size for small dataset
    )

    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Initialize model
    if model_type == 'dual_branch':
        model = DualBranchSTModel()
    elif model_type == 'lstm':
        model = BaselineLSTM()
    elif model_type == 'gat':
        model = BaselineGAT()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    best_acc, best_f1 = trainer.train(num_epochs=num_epochs, early_stopping_patience=10)

    # Evaluate on test set
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Evaluation on Test Set")
    logger.info("=" * 70)

    evaluator = Evaluator(model, test_loader, device=device)
    results = evaluator.evaluate()

    # Generate report
    evaluator.generate_report(results, model_name=f'{model_type}_small')

    return model, results, test_loader


def visualize_results(data, results, model_name='dual_branch_small'):
    """
    Generate visualizations for small experiment

    Args:
        data: Prepared data dictionary
        results: Evaluation results
        model_name: Name of the model
    """
    logger.info("=" * 70)
    logger.info("Step 5: Generating Visualizations")
    logger.info("=" * 70)

    output_dir = os.path.join(config.FIGURE_DIR, 'small_experiment')
    os.makedirs(output_dir, exist_ok=True)

    # Spatial visualizations
    logger.info("Creating spatial visualizations...")
    spatial_viz = SpatialVisualizer(data['metadata_df'], output_dir=output_dir)

    # Plot true labels
    spatial_viz.plot_spatial_distribution(
        grid_labels=data['labels'],
        title='True Mobility Pattern Distribution (Sampled)',
        save_path=os.path.join(output_dir, 'true_labels.jpg')
    )

    # Plot predicted labels
    predictions = dict(zip(results['grid_ids'], results['predictions']))
    spatial_viz.plot_spatial_distribution(
        grid_labels=predictions,
        title='Predicted Mobility Pattern Distribution',
        save_path=os.path.join(output_dir, 'predicted_labels.jpg')
    )

    # Temporal visualizations
    logger.info("Creating temporal visualizations...")
    temporal_viz = TemporalVisualizer(output_dir=output_dir)

    # Plot class temporal patterns
    temporal_viz.plot_class_temporal_patterns(
        grid_flows=data['grid_flows'],
        grid_labels=data['labels'],
        save_path=os.path.join(output_dir, 'class_temporal_patterns.jpg')
    )

    # Plot sample time series for each class
    for class_idx in range(config.NUM_CLASSES):
        class_grids = [gid for gid, label in data['labels'].items() if label == class_idx]

        if len(class_grids) > 0:
            sample_grids = class_grids[:min(3, len(class_grids))]
            temporal_viz.plot_temporal_series(
                grid_flows=data['grid_flows'],
                grid_ids=sample_grids,
                labels=data['labels'],
                save_path=os.path.join(output_dir, f'temporal_class_{class_idx + 1}.jpg')
            )

    logger.info(f"Visualizations saved to {output_dir}")


def print_summary(results):
    """Print experiment summary"""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n{'Metric':<25} {'Value':<15}")
    logger.info("-" * 40)
    logger.info(f"{'Accuracy':<25} {results['accuracy']:<15.4f}")
    logger.info(f"{'F1 Score (Macro)':<25} {results['f1_macro']:<15.4f}")
    logger.info(f"{'F1 Score (Weighted)':<25} {results['f1_weighted']:<15.4f}")
    logger.info(f"{'Precision (Macro)':<25} {results['precision']:<15.4f}")
    logger.info(f"{'Recall (Macro)':<25} {results['recall']:<15.4f}")

    logger.info(f"\n{'Class':<10} {'F1 Score':<15}")
    logger.info("-" * 25)
    for i, f1 in enumerate(results['f1_per_class']):
        logger.info(f"{'Class ' + str(i+1):<10} {f1:<15.4f}")

    logger.info("\n" + "=" * 70)


def main():
    """Main execution for small experiment"""
    logger.info("=" * 70)
    logger.info("SMALL-SCALE MOBILITY PATTERN CLASSIFICATION EXPERIMENT")
    logger.info("=" * 70)

    start_time = datetime.now()

    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}\n")

    # Step 1: Sample balanced labels
    sampled_labels = sample_balanced_labels(
        label_path='data/labels_1w.csv',
        samples_per_class=200,
        output_path='data/labels_sampled200.csv'
    )

    # Step 2: Prepare data
    data = prepare_small_data(year=2021, label_path='data/labels_sampled.csv')

    # Step 3 & 4: Train and evaluate
    model, results, test_loader = train_small_model(
        data=data,
        model_type='dual_branch',
        device=device,
        num_epochs=50
    )

    # Step 5: Visualize
    visualize_results(data, results, model_name='dual_branch_small')

    # Print summary
    print_summary(results)

    # Save results
    results_path = os.path.join(config.OUTPUT_DIR, 'small_experiment_results.json')
    with open(results_path, 'w') as f:
        json_results = {
            'accuracy': results['accuracy'],
            'f1_macro': results['f1_macro'],
            'f1_weighted': results['f1_weighted'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_per_class': results['f1_per_class'],
            'num_samples': len(data['labels']),
            'num_train': int(len(data['labels']) * 0.7),
            'num_val': int(len(data['labels']) * 0.1),
            'num_test': int(len(data['labels']) * 0.2)
        }
        json.dump(json_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"\nTotal experiment time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
