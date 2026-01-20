#!/usr/bin/env python3
"""
训练改进的双年度模型
"""
import torch
import logging
import json
import os
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data
from src.training.dataset import ImprovedDualYearDataset, ImprovedGraphBatchCollator
from src.models.dual_branch_model import ImprovedDualBranchModel
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from src.evaluation.evaluator import Evaluator
from src.visualization.visualizer import SpatialVisualizer, TemporalVisualizer
import config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_output_directories(experiment_name='improved_dual_year'):
    """创建输出目录结构"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join('outputs', f"{experiment_name}_{timestamp}")

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

    logger.info(f"Created output directory: {experiment_dir}")
    return dirs


def prepare_data(label_path='data/labels.csv', samples_per_class=None,
                 use_cache=True, cache_dir='data/cache', force_regenerate=False):
    """准备数据"""
    logger.info("=" * 80)
    logger.info("Step 1: Loading and preprocessing data")
    logger.info("=" * 80)
    logger.info(f"Label path: {label_path}")
    logger.info(f"Samples per class: {samples_per_class if samples_per_class else 'ALL'}")
    logger.info(f"Use cache: {use_cache}")
    logger.info(f"Cache directory: {cache_dir}")

    data = prepare_dual_year_experiment_data(
        label_path=label_path,
        samples_per_class=samples_per_class,
        use_cache=use_cache and not force_regenerate,
        cache_dir=cache_dir
    )

    logger.info(f"✓ Loaded {len(data['labels'])} labeled grids")
    logger.info(f"✓ Created {len(data['graphs_2021'])} dynamic graphs per year")

    return data


def create_dataloaders(data):
    """创建数据加载器"""
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating dataloaders")
    logger.info("=" * 80)

    # 创建数据集
    grid_ids = list(data['labels'].keys())
    dataset = ImprovedDualYearDataset(
        change_features=data['change_features'],
        labels=data['labels'],
        grid_ids=grid_ids,
        metadata_df=data['metadata_df']
    )

    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    logger.info(f"✓ Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # 准备全图特征
    all_features_2021 = []
    all_features_2024 = []

    for grid_id in data['grid_id_to_idx'].keys():
        if grid_id in data['change_features']:
            features = data['change_features'][grid_id]
            x_2021 = features[:, [0, 2]]  # [total_log, net_flow_log]
            x_2024 = features[:, [1, 3]]
            all_features_2021.append(x_2021)
            all_features_2024.append(x_2024)

    all_features_2021 = np.stack(all_features_2021)
    all_features_2024 = np.stack(all_features_2024)

    # 创建collator
    collator = ImprovedGraphBatchCollator(
        graphs_2021=data['graphs_2021'],
        graphs_2024=data['graphs_2024'],
        grid_id_to_idx=data['grid_id_to_idx'],
        all_features_2021=all_features_2021,
        all_features_2024=all_features_2024
    )

    # 创建dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )

    logger.info(f"✓ Created dataloaders with batch_size={config.BATCH_SIZE}")

    return train_loader, val_loader, test_loader, data['class_weights']


def create_model(device):
    """创建模型"""
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Initializing model")
    logger.info("=" * 80)

    model = ImprovedDualBranchModel(
        temporal_input_size=config.TEMPORAL_INPUT_SIZE,
        spatial_input_size=config.SPATIAL_INPUT_SIZE,
        num_time_steps=config.TIME_STEPS
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model initialized with {num_params:,} parameters")
    logger.info(f"✓ Device: {device}")

    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, batch in enumerate(train_loader):
        # 移动到设备
        x_2021 = batch['all_features_2021'].to(device)
        x_2024 = batch['all_features_2024'].to(device)
        labels = batch['labels'].to(device)
        node_indices = batch['node_indices'].to(device)

        # 移动图数据到设备
        graphs_2021 = [(edge_index.to(device), edge_attr.to(device))
                       for edge_index, edge_attr in batch['graphs_2021']]
        graphs_2024 = [(edge_index.to(device), edge_attr.to(device))
                       for edge_index, edge_attr in batch['graphs_2024']]

        # 前向传播
        outputs = model(x_2021, x_2024, graphs_2021, graphs_2024, node_indices)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        # 进度日志
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                       f"Loss: {loss.item():.4f}")

    train_acc = 100. * train_correct / train_total
    avg_loss = train_loss / len(train_loader)

    return avg_loss, train_acc


def validate(model, val_loader, device):
    """验证"""
    model.eval()
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x_2021 = batch['all_features_2021'].to(device)
            x_2024 = batch['all_features_2024'].to(device)
            labels = batch['labels'].to(device)
            node_indices = batch['node_indices'].to(device)

            # 移动图数据到设备
            graphs_2021 = [(edge_index.to(device), edge_attr.to(device))
                           for edge_index, edge_attr in batch['graphs_2021']]
            graphs_2024 = [(edge_index.to(device), edge_attr.to(device))
                           for edge_index, edge_attr in batch['graphs_2024']]

            outputs = model(x_2021, x_2024, graphs_2021, graphs_2024, node_indices)
            _, predicted = outputs.max(1)

            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = 100. * val_correct / val_total
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    return val_acc, f1_macro


def train_model(model, train_loader, val_loader, class_weights, device, dirs):
    """完整训练流程"""
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Training model")
    logger.info("=" * 80)

    # 训练配置
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    best_val_acc = 0.0
    best_f1 = 0.0
    patience_counter = 0

    # 训练历史记录
    train_losses = []
    train_accs = []
    val_accs = []
    val_f1s = []

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        val_acc, f1_macro = validate(model, val_loader, device)

        val_accs.append(val_acc)
        val_f1s.append(f1_macro)

        # 学习率调度
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        # 日志
        logger.info(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Acc: {val_acc:.2f}% | F1-Macro: {f1_macro:.4f}")
        logger.info(f"  LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_f1 = f1_macro
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'f1_macro': f1_macro,
            }, os.path.join(dirs['models'], 'best_model.pth'))

            logger.info(f"  ✓ Saved best model (val_acc={val_acc:.2f}%, f1={f1_macro:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"\n✓ Early stopping at epoch {epoch}")
                break

    logger.info(f"\n✓ Training completed!")
    logger.info(f"  Best Val Acc: {best_val_acc:.2f}%")
    logger.info(f"  Best F1-Macro: {best_f1:.4f}")

    # 保存训练历史
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'val_f1': val_f1s
    }

    history_path = os.path.join(dirs['metrics'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    return best_val_acc, best_f1, history


def test_model(model, test_loader, device, dirs):
    """测试模型"""
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Testing model")
    logger.info("=" * 80)

    # 加载最佳模型
    checkpoint_path = os.path.join(dirs['models'], 'best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded best model from epoch {checkpoint['epoch']}")

    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    all_grid_ids = []

    with torch.no_grad():
        for batch in test_loader:
            x_2021 = batch['all_features_2021'].to(device)
            x_2024 = batch['all_features_2024'].to(device)
            labels = batch['labels'].to(device)
            node_indices = batch['node_indices'].to(device)

            # 移动图数据到设备
            graphs_2021 = [(edge_index.to(device), edge_attr.to(device))
                           for edge_index, edge_attr in batch['graphs_2021']]
            graphs_2024 = [(edge_index.to(device), edge_attr.to(device))
                           for edge_index, edge_attr in batch['graphs_2024']]

            outputs = model(x_2021, x_2024, graphs_2021, graphs_2024, node_indices)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_grid_ids.extend(batch['grid_ids'])

    test_acc = 100. * test_correct / test_total
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    logger.info(f"\n{'='*80}")
    logger.info("FINAL TEST RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"F1-Macro: {f1_macro:.4f}")
    logger.info(f"F1-Weighted: {f1_weighted:.4f}")
    logger.info(f"{'='*80}")

    # 详细分类报告
    logger.info("\nClassification Report:")
    report = classification_report(
        all_labels, all_preds,
        target_names=[f'Class {i+1}' for i in range(config.NUM_CLASSES)]
    )
    logger.info(f"\n{report}")

    # 保存分类报告
    report_path = os.path.join(dirs['metrics'], 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i+1}' for i in range(config.NUM_CLASSES)],
                yticklabels=[f'Class {i+1}' for i in range(config.NUM_CLASSES)])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(dirs['figures'], 'confusion_matrix.jpg')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")

    # 保存测试结果
    results = {
        'accuracy': float(test_acc / 100),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'predictions': [int(p) for p in all_preds],
        'labels': [int(l) for l in all_labels],
        'grid_ids': all_grid_ids
    }

    results_path = os.path.join(dirs['metrics'], 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test results saved to {results_path}")

    return test_acc, f1_macro, results


def main(label_path='data/labels.csv', samples_per_class=None,
         use_cache=True, cache_dir='data/cache', force_regenerate=False,
         experiment_name='improved_dual_year'):
    """主函数"""
    # 创建输出目录
    dirs = create_output_directories(experiment_name)

    # 设置文件日志
    log_file = os.path.join(dirs['logs'], 'experiment.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("\n" + "=" * 80)
    logger.info("IMPROVED DUAL-YEAR MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Experiment directory: {dirs['root']}")
    logger.info(f"Label path: {label_path}")
    logger.info(f"Samples per class: {samples_per_class if samples_per_class else 'ALL'}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # 保存配置
    config_dict = {
        'experiment_name': experiment_name,
        'label_path': label_path,
        'samples_per_class': samples_per_class,
        'use_cache': use_cache,
        'cache_dir': cache_dir,
        'device': str(device),
        'num_classes': config.NUM_CLASSES,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'num_epochs': config.NUM_EPOCHS,
        'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
        'random_seed': config.RANDOM_SEED,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    config_path = os.path.join(dirs['root'], 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Configuration saved to {config_path}")

    try:
        # 1. 准备数据
        data = prepare_data(label_path, samples_per_class, use_cache,
                           cache_dir, force_regenerate)

        # 保存标签数据
        if 'label_df' in data:
            label_output_path = os.path.join(dirs['data'], 'labels_used.csv')
            data['label_df'].to_csv(label_output_path, index=False)
            logger.info(f"Labels saved to {label_output_path}")

        # 2. 创建数据加载器
        train_loader, val_loader, test_loader, class_weights = create_dataloaders(data)

        # 3. 创建模型
        model = create_model(device)

        # 4. 训练模型
        best_val_acc, best_f1, history = train_model(
            model, train_loader, val_loader, class_weights, device, dirs
        )

        # 5. 测试模型
        test_acc, test_f1, results = test_model(model, test_loader, device, dirs)

        # 总结
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        logger.info(f"Best Validation F1-Macro: {best_f1:.4f}")
        logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
        logger.info(f"Final Test F1-Macro: {test_f1:.4f}")
        logger.info("=" * 80)
        logger.info(f"\n✓ All results saved to: {dirs['root']}")
        logger.info("\n✓ Training completed successfully!")

        # 生成总结报告
        summary_path = os.path.join(dirs['root'], 'EXPERIMENT_SUMMARY.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("改进的双年份对比实验总结 (2021 vs 2024)\n")
            f.write("=" * 80 + "\n\n")

            f.write("## 实验配置\n\n")
            f.write(f"- 实验名称: {experiment_name}\n")
            f.write(f"- 标签文件: {label_path}\n")
            f.write(f"- 样本数量: {samples_per_class if samples_per_class else 'ALL'}\n")
            f.write(f"- 设备: {device}\n")
            f.write(f"- 批次大小: {config.BATCH_SIZE}\n")
            f.write(f"- 学习率: {config.LEARNING_RATE}\n")
            f.write(f"- 训练轮数: {config.NUM_EPOCHS}\n\n")

            f.write("## 数据统计\n\n")
            f.write(f"- 总样本数: {len(data['labels'])}\n")
            f.write(f"- 训练集: {len(train_loader.dataset)}\n")
            f.write(f"- 验证集: {len(val_loader.dataset)}\n")
            f.write(f"- 测试集: {len(test_loader.dataset)}\n\n")

            f.write("## 模型性能\n\n")
            f.write(f"- 最佳验证准确率: {best_val_acc:.2f}%\n")
            f.write(f"- 最佳验证F1分数: {best_f1:.4f}\n")
            f.write(f"- 测试准确率: {test_acc:.2f}%\n")
            f.write(f"- 测试F1分数: {test_f1:.4f}\n\n")

            f.write("## 输出文件\n\n")
            f.write(f"- 模型: {dirs['models']}/best_model.pth\n")
            f.write(f"- 训练历史: {dirs['metrics']}/training_history.json\n")
            f.write(f"- 测试结果: {dirs['metrics']}/test_results.json\n")
            f.write(f"- 分类报告: {dirs['metrics']}/classification_report.txt\n")
            f.write(f"- 混淆矩阵: {dirs['figures']}/confusion_matrix.jpg\n")
            f.write(f"- 日志文件: {dirs['logs']}/experiment.log\n\n")

            f.write("=" * 80 + "\n")

        logger.info(f"Summary report saved to {summary_path}")

        return dirs['root'], results

    except Exception as e:
        logger.error(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='训练改进的双年度模型')
    parser.add_argument('--label-path', type=str, default='data/labels.csv',
                        help='标签文件路径 (默认: data/labels.csv)')
    parser.add_argument('--samples-per-class', type=int, default=None,
                        help='每类采样数量，None表示使用全部数据 (默认: None)')
    parser.add_argument('--cache-dir', type=str, default='data/cache',
                        help='缓存目录 (默认: data/cache)')
    parser.add_argument('--no-cache', action='store_true',
                        help='不使用缓存')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='强制重新生成缓存（忽略已有缓存）')
    parser.add_argument('--experiment-name', type=str, default='improved_dual_year',
                        help='实验名称 (默认: improved_dual_year)')

    args = parser.parse_args()

    # 运行主函数
    experiment_dir, results = main(
        label_path=args.label_path,
        samples_per_class=args.samples_per_class,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        force_regenerate=args.force_regenerate,
        experiment_name=args.experiment_name
    )

    # 打印最终结果
    print(f"\n{'='*80}")
    print(f"✅ 训练完成！")
    print(f"{'='*80}")
    print(f"📁 结果目录: {experiment_dir}")
    print(f"📊 测试准确率: {results['accuracy']*100:.2f}%")
    print(f"📈 测试F1分数: {results['f1_macro']:.4f}")
    print(f"{'='*80}\n")
