"""
Evaluation utilities for model assessment
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import os
import logging
import config

logger = logging.getLogger(__name__)


class Evaluator:
    """Model evaluator"""

    def __init__(self, model, test_loader, device='cuda', output_dir=config.OUTPUT_DIR):
        """
        Initialize evaluator

        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to use
            output_dir: Output directory for results
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir

        # Check if model is wrapped with DataParallel
        self.is_data_parallel = isinstance(model, nn.DataParallel)

        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self):
        """
        Evaluate model on test set

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_grid_ids = []

        with torch.no_grad():
            for batch in self.test_loader:
                # Move data to device
                temporal = batch['temporal'].to(self.device)
                spatial = batch['spatial'].to(self.device)
                labels = batch['labels'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_attr = batch['edge_attr'].to(self.device) if batch['edge_attr'] is not None else None
                node_indices = batch['node_indices'].to(self.device) if batch['node_indices'] is not None else None
                all_spatial_features = batch['all_spatial_features'].to(self.device) if batch['all_spatial_features'] is not None else spatial
                grid_ids = batch['grid_ids']

                # Forward pass
                from src.models.dual_branch_model import BaselineLSTM, BaselineGAT

                if isinstance(self.model, BaselineLSTM):
                    logits = self.model(temporal)
                elif isinstance(self.model, BaselineGAT):
                    # For GAT, use all spatial features
                    all_logits = self.model(all_spatial_features, edge_index, edge_attr)
                    # Select batch nodes
                    if node_indices is not None:
                        logits = all_logits[node_indices]
                    else:
                        logits = all_logits[:len(labels)]
                else:
                    logits = self.model(temporal, all_spatial_features, edge_index, edge_attr, node_indices)

                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_grid_ids.extend(grid_ids)

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=[f'Class {i+1}' for i in range(config.NUM_CLASSES)],
            digits=4
        )

        # Log results
        logger.info("=" * 50)
        logger.info("Evaluation Results")
        logger.info("=" * 50)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
        logger.info(f"F1 Score (Weighted): {f1_weighted:.4f}")
        logger.info(f"Precision (Macro): {precision:.4f}")
        logger.info(f"Recall (Macro): {recall:.4f}")
        logger.info("\nPer-class F1 Scores:")
        for i, f1 in enumerate(f1_per_class):
            logger.info(f"  Class {i+1}: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{report}")

        # Save results
        results = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_per_class': f1_per_class.tolist(),
            'precision': float(precision),
            'recall': float(recall),
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
            'grid_ids': all_grid_ids
        }

        return results

    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[f'Class {i+1}' for i in range(config.NUM_CLASSES)],
            yticklabels=[f'Class {i+1}' for i in range(config.NUM_CLASSES)],
            cbar_kws={'label': 'Count'}
        )

        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT)
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.close()

    def plot_f1_scores(self, f1_per_class, save_path=None):
        """
        Plot per-class F1 scores

        Args:
            f1_per_class: Array of F1 scores per class
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 6))

        classes = [f'Class {i+1}' for i in range(len(f1_per_class))]
        colors = plt.cm.viridis(np.linspace(0, 1, len(f1_per_class)))

        bars = plt.bar(classes, f1_per_class, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.xlabel('Class', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT)
            logger.info(f"Saved F1 scores plot to {save_path}")

        plt.close()

    def generate_report(self, results, model_name='model', output_dir=None):
        """
        Generate comprehensive evaluation report

        Args:
            results: Evaluation results dictionary
            model_name: Name of the model
            output_dir: Optional directory to override self.output_dir
        """
        base_dir = output_dir if output_dir is not None else self.output_dir
        report_dir = os.path.join(base_dir, 'evaluation_reports')
        os.makedirs(report_dir, exist_ok=True)

        # Save metrics to file
        metrics_path = os.path.join(report_dir, f'{model_name}_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write(f"Evaluation Report: {model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"F1 Score (Macro): {results['f1_macro']:.4f}\n")
            f.write(f"F1 Score (Weighted): {results['f1_weighted']:.4f}\n")
            f.write(f"Precision (Macro): {results['precision']:.4f}\n")
            f.write(f"Recall (Macro): {results['recall']:.4f}\n\n")
            f.write("Per-class F1 Scores:\n")
            for i, f1 in enumerate(results['f1_per_class']):
                f.write(f"  Class {i+1}: {f1:.4f}\n")

        logger.info(f"Saved metrics report to {metrics_path}")

        # Plot confusion matrix
        cm = np.array(results['confusion_matrix'])
        cm_path = os.path.join(report_dir, f'{model_name}_confusion_matrix.{config.FIGURE_FORMAT}')
        self.plot_confusion_matrix(cm, cm_path)

        # Plot F1 scores
        f1_path = os.path.join(report_dir, f'{model_name}_f1_scores.{config.FIGURE_FORMAT}')
        self.plot_f1_scores(np.array(results['f1_per_class']), f1_path)

        logger.info(f"Generated evaluation report for {model_name}")


def compare_models(results_dict, save_path=None):
    """
    Compare multiple models

    Args:
        results_dict: Dictionary mapping model names to results
        save_path: Path to save comparison figure
    """
    model_names = list(results_dict.keys())
    metrics = ['accuracy', 'f1_macro', 'precision', 'recall']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [results_dict[name][metric] for name in model_names]

        ax = axes[idx]
        bars = ax.bar(model_names, values, alpha=0.8, edgecolor='black')

        # Color bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
        logger.info(f"Saved model comparison to {save_path}")

    plt.close()
