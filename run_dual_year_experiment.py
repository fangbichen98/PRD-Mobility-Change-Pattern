"""
Dual-year experiment: Compare 2021 vs 2024 mobility patterns
This correctly implements the project's core idea of temporal change classification
"""
import pandas as pd
import numpy as np
import torch
import logging
import os
import json
from datetime import datetime

import config
from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data
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


class DualYearDataset(MobilityDataset):
    """Dataset for dual-year mobility change classification"""

    def __init__(self,
                 change_features,
                 labels,
                 grid_ids,
                 metadata_df=None):
        """
        Initialize dual-year dataset

        Args:
            change_features: Dictionary mapping grid_id to change features (time_steps, 10)
                            [2021_inflow, 2021_outflow, 2024_inflow, 2024_outflow,
                             diff_inflow, diff_outflow, rel_change_inflow, rel_change_outflow,
                             total_2021, total_2024]
            labels: Dictionary mapping grid_id to label (0-8)
            grid_ids: List of grid IDs to include
            metadata_df: Grid metadata DataFrame
        """
        self.change_features = change_features
        # Alias for compatibility with create_data_loaders
        self.grid_flows = change_features
        self.labels = labels
        self.grid_ids = [gid for gid in grid_ids if gid in labels and gid in change_features]
        self.metadata_df = metadata_df

        logger.info(f"Created dual-year dataset with {len(self.grid_ids)} samples")

    def __len__(self):
        return len(self.grid_ids)

    def __getitem__(self, idx):
        """
        Get item by index

        Returns:
            temporal_features: Change features over time (seq_len, 10)
            spatial_features: Flattened change features (seq_len * 10,)
            label: Class label (0-8)
            grid_id: Grid ID
        """
        grid_id = self.grid_ids[idx]

        # Get change features (168, 10)
        temporal_features = self.change_features[grid_id]

        # Create spatial features (flattened)
        spatial_features = temporal_features.flatten()

        # Get label
        label = self.labels[grid_id]

        return {
            'temporal': torch.FloatTensor(temporal_features),
            'spatial': torch.FloatTensor(spatial_features),
            'label': torch.LongTensor([label])[0],
            'grid_id': grid_id
        }


def run_dual_year_experiment(
    experiment_name="dual_year_comparison",
    model_type='dual_branch',
    samples_per_class=100,
    num_epochs=50,
    batch_size=16,
    device='cuda'
):
    """
    Run dual-year comparison experiment

    Args:
        experiment_name: Name of the experiment
        model_type: Type of model ('dual_branch', 'lstm', 'gat')
        samples_per_class: Number of samples per class
        num_epochs: Number of training epochs
        batch_size: Batch size
        device: Device to use
    """
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join('outputs', f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    dirs = {
        'root': experiment_dir,
        'models': os.path.join(experiment_dir, 'models'),
        'logs': os.path.join(experiment_dir, 'logs'),
        'figures': os.path.join(experiment_dir, 'figures'),
        'metrics': os.path.join(experiment_dir, 'metrics'),
        'data': os.path.join(experiment_dir, 'data')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Setup logging
    log_file = os.path.join(dirs['logs'], 'experiment.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Save configuration
    config_dict = {
        'experiment_type': 'dual_year_comparison',
        'year1': 2021,
        'year2': 2024,
        'experiment_name': experiment_name,
        'model_type': model_type,
        'samples_per_class': samples_per_class,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'device': device,
        'num_classes': config.NUM_CLASSES,
        'train_days': config.TRAIN_DAYS,
        'lstm_layers': config.LSTM_LAYERS,
        'lstm_hidden_size': config.LSTM_HIDDEN_SIZE,
        'dysat_layers': config.DYSAT_LAYERS,
        'dysat_hidden_size': config.DYSAT_HIDDEN_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'random_seed': config.RANDOM_SEED,
        'feature_description': '10 features: [2021_inflow, 2021_outflow, 2024_inflow, 2024_outflow, diff_inflow, diff_outflow, rel_change_inflow, rel_change_outflow, total_2021, total_2024]',
        'feature_note': 'Both temporal branch (LSTM) and spatial branch (DySAT) use the same 10 features. Temporal branch processes them as time series (168, 10), spatial branch uses flattened version (1680,)'
    }

    config_path = os.path.join(dirs['root'], 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Check device
    device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    logger.info(f"Using device: {device}\n")

    # Prepare dual-year data
    logger.info("=" * 80)
    logger.info("Preparing Dual-Year Data (2021 vs 2024)")
    logger.info("=" * 80)

    data = prepare_dual_year_experiment_data(
        label_path='data/labels_1w.csv',
        samples_per_class=samples_per_class
    )

    # Save sampled labels
    label_output_path = os.path.join(dirs['data'], 'labels_sampled.csv')
    data['label_df'].to_csv(label_output_path, index=False)
    logger.info(f"Saved sampled labels to {label_output_path}")

    # Create dataset
    logger.info("=" * 80)
    logger.info(f"Creating Dual-Year Dataset")
    logger.info("=" * 80)

    dataset = DualYearDataset(
        change_features=data['change_features'],
        labels=data['labels'],
        grid_ids=list(data['labels'].keys()),
        metadata_df=data['metadata_df']
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        edge_index=data['edge_index'],
        edge_attr=data['edge_weights'],
        grid_id_to_idx=data['grid_id_to_idx'],
        batch_size=batch_size
    )

    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Initialize model
    logger.info("=" * 80)
    logger.info(f"Initializing {model_type.upper()} Model")
    logger.info("=" * 80)

    if model_type == 'dual_branch':
        model = DualBranchSTModel(
            temporal_input_size=10,
            spatial_input_size=168 * 10
        )
    elif model_type == 'lstm':
        model = BaselineLSTM(input_size=10)
    elif model_type == 'gat':
        model = BaselineGAT(input_size=168 * 10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Input features: 10 (2021_in, 2021_out, 2024_in, 2024_out, diff_in, diff_out, rel_in, rel_out, total_2021, total_2024)")
    logger.info(f"Feature usage: Both temporal (LSTM) and spatial (DySAT) branches use the same 10 features")

    # Train
    logger.info("=" * 80)
    logger.info("Training Model")
    logger.info("=" * 80)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=dirs['models']
    )

    best_acc, best_f1 = trainer.train(num_epochs=num_epochs, early_stopping_patience=10)

    # Save training history
    history = {
        'train_loss': trainer.train_losses,
        'train_acc': trainer.train_accs,
        'train_f1': trainer.train_f1s,
        'val_loss': trainer.val_losses,
        'val_acc': trainer.val_accs,
        'val_f1': trainer.val_f1s,
    }

    history_path = os.path.join(dirs['metrics'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 80)

    evaluator = Evaluator(model, test_loader, device=device, output_dir=dirs['figures'])
    results = evaluator.evaluate()

    # Save results
    results_path = os.path.join(dirs['metrics'], 'test_results.json')
    with open(results_path, 'w') as f:
        json_results = {
            'accuracy': results['accuracy'],
            'f1_macro': results['f1_macro'],
            'f1_weighted': results['f1_weighted'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_per_class': results['f1_per_class']
        }
        json.dump(json_results, f, indent=2)

    evaluator.generate_report(results, model_name=model_type, output_dir=dirs['metrics'])

    # Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Generating Visualizations")
    logger.info("=" * 80)

    # Spatial visualizations
    spatial_viz = SpatialVisualizer(data['metadata_df'], output_dir=dirs['figures'])

    spatial_viz.plot_spatial_distribution(
        grid_labels=data['labels'],
        title='True Mobility Change Pattern Distribution (2021 vs 2024)',
        save_path=os.path.join(dirs['figures'], 'true_labels.jpg')
    )

    predictions = dict(zip(results['grid_ids'], results['predictions']))
    spatial_viz.plot_spatial_distribution(
        grid_labels=predictions,
        title='Predicted Mobility Change Pattern Distribution',
        save_path=os.path.join(dirs['figures'], 'predicted_labels.jpg')
    )

    # Temporal visualizations
    temporal_viz = TemporalVisualizer(output_dir=dirs['figures'])

    temporal_viz.plot_class_temporal_patterns(
        grid_flows=data['flows_2021'],
        grid_labels=data['labels'],
        save_path=os.path.join(dirs['figures'], 'class_temporal_patterns_2021.jpg')
    )

    temporal_viz.plot_class_temporal_patterns(
        grid_flows=data['flows_2024'],
        grid_labels=data['labels'],
        save_path=os.path.join(dirs['figures'], 'class_temporal_patterns_2024.jpg')
    )

    # Generate summary report
    summary_path = os.path.join(dirs['root'], 'EXPERIMENT_SUMMARY.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("双年份对比实验总结 (2021 vs 2024)\n")
        f.write("=" * 80 + "\n\n")

        f.write("## 实验设计\n\n")
        f.write("本实验正确实现了项目的核心思路：\n")
        f.write("- 对比2021年和2024年的流动模式\n")
        f.write("- 标签表示人群流动模式的变化类型（9类）\n")
        f.write("- 模型输入包含两年的数据及其变化特征\n\n")

        f.write("## 特征设计\n\n")
        f.write("每个网格的时间序列特征（168小时 × 10特征）：\n")
        f.write("1. 2021年流入量\n")
        f.write("2. 2021年流出量\n")
        f.write("3. 2024年流入量\n")
        f.write("4. 2024年流出量\n")
        f.write("5. 流入量变化（2024-2021）\n")
        f.write("6. 流出量变化（2024-2021）\n")
        f.write("7. 流入量相对变化率\n")
        f.write("8. 流出量相对变化率\n")
        f.write("9. 2021年总流量（流入+流出）\n")
        f.write("10. 2024年总流量（流入+流出）\n\n")

        f.write("## 双分支特征使用\n\n")
        f.write("时间序列分支（LSTM）和动态图分支（DySAT）使用相同的10个特征：\n")
        f.write("- 时间序列分支：接收 (168, 10) 的时间序列数据\n")
        f.write("- 动态图分支：接收 (1680,) 的展平特征向量\n")
        f.write("- 两个分支通过注意力机制融合，学习时空特征的自适应权重\n\n")

        f.write("## 数据统计\n\n")
        f.write(f"- 分析网格数: {len(data['labels'])}\n")
        f.write(f"- 类别数: {config.NUM_CLASSES}\n\n")

        f.write("## 模型性能\n\n")
        f.write(f"- 最佳验证准确率: {max(history['val_acc']):.4f}\n")
        f.write(f"- 测试准确率: {results['accuracy']:.4f}\n")
        f.write(f"- 测试F1分数: {results['f1_macro']:.4f}\n\n")

        f.write("## 与单年份实验的区别\n\n")
        f.write("单年份实验（之前的run_small_experiment.py）：\n")
        f.write("- ❌ 只使用2021年数据\n")
        f.write("- ❌ 特征：2个（流入、流出）\n")
        f.write("- ❌ 无法捕捉时间变化\n\n")

        f.write("双年份实验（本实验）：\n")
        f.write("- ✅ 使用2021和2024年数据\n")
        f.write("- ✅ 特征：10个（包含变化量、变化率和总流量）\n")
        f.write("- ✅ 正确实现了变化模式分类\n\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Summary saved to {summary_path}")

    logger.info("\n" + "=" * 80)
    logger.info("DUAL-YEAR EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info(f"All results saved to: {experiment_dir}")
    logger.info("=" * 80)

    return experiment_dir, results


if __name__ == "__main__":
    # Run dual-year experiment
    experiment_dir, results = run_dual_year_experiment(
        experiment_name="dual_year_2021vs2024",
        model_type='dual_branch',
        samples_per_class=100,
        num_epochs=50,
        batch_size=16,
        device='cpu'
    )

    print(f"\n✅ Dual-year experiment completed!")
    print(f"📁 Results saved to: {experiment_dir}")
    print(f"📊 Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"📈 Test F1 Score: {results['f1_macro']:.4f}")
    print(f"\n💡 This experiment correctly compares 2021 vs 2024 mobility patterns!")
    print(f"💡 Both temporal (LSTM) and spatial (DySAT) branches use the same 10 features!")
