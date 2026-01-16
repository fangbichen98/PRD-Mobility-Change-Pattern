"""
Visualization utilities for spatial and temporal analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import os
import logging
import config

logger = logging.getLogger(__name__)


class SpatialVisualizer:
    """Visualizer for spatial patterns"""

    def __init__(self, metadata_df, output_dir=config.FIGURE_DIR):
        """
        Initialize spatial visualizer

        Args:
            metadata_df: Grid metadata DataFrame
            output_dir: Output directory for figures
        """
        self.metadata_df = metadata_df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_spatial_distribution(self, grid_labels, title='Spatial Distribution', save_path=None):
        """
        Plot spatial distribution of labels

        Args:
            grid_labels: Dictionary mapping grid_id to label
            title: Plot title
            save_path: Path to save figure
        """
        # Merge labels with metadata
        label_df = pd.DataFrame(list(grid_labels.items()), columns=['grid_id', 'label'])
        plot_df = self.metadata_df.merge(label_df, on='grid_id', how='inner')

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Create colormap for 9 classes
        colors = plt.cm.tab10(np.linspace(0, 1, config.NUM_CLASSES))
        cmap = ListedColormap(colors)

        # Scatter plot
        scatter = ax.scatter(
            plot_df['lon'],
            plot_df['lat'],
            c=plot_df['label'],
            cmap=cmap,
            s=20,
            alpha=0.7,
            edgecolors='none',
            vmin=0,
            vmax=config.NUM_CLASSES - 1
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(config.NUM_CLASSES))
        cbar.set_label('Mobility Pattern Class', fontsize=12)
        cbar.ax.set_yticklabels([f'Class {i+1}' for i in range(config.NUM_CLASSES)])

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved spatial distribution plot to {save_path}")

        plt.close()

    def plot_heatmap(self, grid_values, title='Heatmap', cmap=config.HEATMAP_CMAP, save_path=None):
        """
        Plot heatmap of grid values

        Args:
            grid_values: Dictionary mapping grid_id to value
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
        """
        # Merge values with metadata
        value_df = pd.DataFrame(list(grid_values.items()), columns=['grid_id', 'value'])
        plot_df = self.metadata_df.merge(value_df, on='grid_id', how='inner')

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Scatter plot with values
        scatter = ax.scatter(
            plot_df['lon'],
            plot_df['lat'],
            c=plot_df['value'],
            cmap=cmap,
            s=30,
            alpha=0.8,
            edgecolors='none'
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Value', fontsize=12)

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")

        plt.close()

    def plot_class_distribution_map(self, predictions, labels, save_path=None):
        """
        Plot comparison of predicted vs true labels

        Args:
            predictions: Dictionary mapping grid_id to predicted label
            labels: Dictionary mapping grid_id to true label
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # True labels
        self._plot_on_axis(axes[0], labels, 'True Labels')

        # Predicted labels
        self._plot_on_axis(axes[1], predictions, 'Predicted Labels')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved class distribution map to {save_path}")

        plt.close()

    def _plot_on_axis(self, ax, grid_labels, title):
        """Helper function to plot on a specific axis"""
        label_df = pd.DataFrame(list(grid_labels.items()), columns=['grid_id', 'label'])
        plot_df = self.metadata_df.merge(label_df, on='grid_id', how='inner')

        colors = plt.cm.tab10(np.linspace(0, 1, config.NUM_CLASSES))
        cmap = ListedColormap(colors)

        scatter = ax.scatter(
            plot_df['lon'],
            plot_df['lat'],
            c=plot_df['label'],
            cmap=cmap,
            s=20,
            alpha=0.7,
            edgecolors='none',
            vmin=0,
            vmax=config.NUM_CLASSES - 1
        )

        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')


class TemporalVisualizer:
    """Visualizer for temporal patterns"""

    def __init__(self, output_dir=config.FIGURE_DIR):
        """
        Initialize temporal visualizer

        Args:
            output_dir: Output directory for figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_temporal_series(self, grid_flows, grid_ids, labels=None, save_path=None):
        """
        Plot temporal series for selected grids

        Args:
            grid_flows: Dictionary mapping grid_id to temporal flow array
            grid_ids: List of grid IDs to plot
            labels: Optional dictionary mapping grid_id to label
            save_path: Path to save figure
        """
        n_grids = len(grid_ids)
        fig, axes = plt.subplots(n_grids, 1, figsize=(14, 3 * n_grids))

        if n_grids == 1:
            axes = [axes]

        for idx, grid_id in enumerate(grid_ids):
            ax = axes[idx]
            flow_data = grid_flows[grid_id]  # Shape: (time_steps, 2)

            time_steps = np.arange(flow_data.shape[0])

            # Plot inflow and outflow
            ax.plot(time_steps, flow_data[:, 0], label='Outflow', linewidth=1.5, alpha=0.8)
            ax.plot(time_steps, flow_data[:, 1], label='Inflow', linewidth=1.5, alpha=0.8)

            # Add label info if available
            title = f'Grid {grid_id}'
            if labels and grid_id in labels:
                title += f' (Class {labels[grid_id] + 1})'

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (hours)', fontsize=10)
            ax.set_ylabel('Normalized Flow', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved temporal series plot to {save_path}")

        plt.close()

    def plot_class_temporal_patterns(self, grid_flows, grid_labels, save_path=None):
        """
        Plot average temporal patterns for each class

        Args:
            grid_flows: Dictionary mapping grid_id to temporal flow array
            grid_labels: Dictionary mapping grid_id to label
            save_path: Path to save figure
        """
        # Group flows by class
        class_flows = {i: [] for i in range(config.NUM_CLASSES)}

        for grid_id, label in grid_labels.items():
            if grid_id in grid_flows:
                class_flows[label].append(grid_flows[grid_id])

        # Calculate average for each class
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.flatten()

        for class_idx in range(config.NUM_CLASSES):
            ax = axes[class_idx]

            if len(class_flows[class_idx]) > 0:
                flows = np.array(class_flows[class_idx])  # Shape: (n_samples, time_steps, 2)
                mean_flow = flows.mean(axis=0)  # Shape: (time_steps, 2)
                std_flow = flows.std(axis=0)

                time_steps = np.arange(mean_flow.shape[0])

                # Plot outflow
                ax.plot(time_steps, mean_flow[:, 0], label='Outflow', linewidth=2, color='#1f77b4')
                ax.fill_between(
                    time_steps,
                    mean_flow[:, 0] - std_flow[:, 0],
                    mean_flow[:, 0] + std_flow[:, 0],
                    alpha=0.2,
                    color='#1f77b4'
                )

                # Plot inflow
                ax.plot(time_steps, mean_flow[:, 1], label='Inflow', linewidth=2, color='#ff7f0e')
                ax.fill_between(
                    time_steps,
                    mean_flow[:, 1] - std_flow[:, 1],
                    mean_flow[:, 1] + std_flow[:, 1],
                    alpha=0.2,
                    color='#ff7f0e'
                )

                ax.set_title(f'Class {class_idx + 1} (n={len(class_flows[class_idx])})',
                           fontsize=11, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Class {class_idx + 1} (n=0)', fontsize=11, fontweight='bold')

            ax.set_xlabel('Time (hours)', fontsize=9)
            ax.set_ylabel('Normalized Flow', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.suptitle('Average Temporal Patterns by Class', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved class temporal patterns to {save_path}")

        plt.close()

    def plot_daily_patterns(self, grid_flows, grid_id, label=None, save_path=None):
        """
        Plot daily patterns for a specific grid

        Args:
            grid_flows: Dictionary mapping grid_id to temporal flow array
            grid_id: Grid ID to plot
            label: Optional label for the grid
            save_path: Path to save figure
        """
        flow_data = grid_flows[grid_id]  # Shape: (168, 2) for 7 days

        # Reshape to (7 days, 24 hours, 2 features)
        n_days = flow_data.shape[0] // 24
        daily_data = flow_data[:n_days * 24].reshape(n_days, 24, 2)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        hours = np.arange(24)

        # Outflow
        ax = axes[0]
        for day in range(n_days):
            ax.plot(hours, daily_data[day, :, 0], label=f'Day {day + 1}', alpha=0.7, linewidth=1.5)
        ax.set_title('Outflow Patterns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=10)
        ax.set_ylabel('Normalized Outflow', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Inflow
        ax = axes[1]
        for day in range(n_days):
            ax.plot(hours, daily_data[day, :, 1], label=f'Day {day + 1}', alpha=0.7, linewidth=1.5)
        ax.set_title('Inflow Patterns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=10)
        ax.set_ylabel('Normalized Inflow', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

        title = f'Daily Patterns for Grid {grid_id}'
        if label is not None:
            title += f' (Class {label + 1})'
        plt.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved daily patterns plot to {save_path}")

        plt.close()
