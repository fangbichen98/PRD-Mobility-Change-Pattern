#!/usr/bin/env python3
"""
层次化分类训练脚本 (3×3 Hierarchical Classification)

输出三种预测结果:
1. 流量强度分类 (3类): 稳定/增长/衰减
2. 空间方向分类 (3类): 均衡/聚集/扩散
3. 直接9分类: 用于对比

使用方法:
python3 train_hierarchical_simple.py
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
import config
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_labels_to_hierarchical(labels):
    """
    将9分类标签转换为两个3分类标签

    Args:
        labels: (batch_size,) 值为0-8

    Returns:
        intensity_labels: (batch_size,) 值为0,1,2 (稳定/增长/衰减)
        direction_labels: (batch_size,) 值为0,1,2 (均衡/聚集/扩散)
    """
    intensity_labels = labels // 3  # 整除3
    direction_labels = labels % 3   # 取余3
    return intensity_labels, direction_labels


def combine_hierarchical_predictions(intensity_pred, direction_pred):
    """
    将两个3分类预测组合成9分类预测

    Args:
        intensity_pred: (batch_size,) 值为0,1,2
        direction_pred: (batch_size,) 值为0,1,2

    Returns:
        final_pred: (batch_size,) 值为0-8
    """
    return intensity_pred * 3 + direction_pred


def train_epoch_hierarchical(model, train_loader, criterion_intensity, criterion_direction, criterion_direct, optimizer, device, epoch):
    """训练一个epoch (层次化版本)"""
    model.train()

    # 统计指标
    total_loss = 0.0
    total_intensity_loss = 0.0
    total_direction_loss = 0.0
    total_direct_loss = 0.0

    # 准确率统计
    intensity_correct = 0
    direction_correct = 0
    hierarchical_correct = 0  # 组合预测的准确率
    direct_correct = 0
    total_samples = 0

    # Gradient accumulation settings
    accumulation_steps = 4

    for batch_idx, batch in enumerate(train_loader):
        # Get batch-level features for temporal branch
        x_2021_batch = batch['x_2021'].to(device)  # (batch_size, 7, 2)
        x_2024_batch = batch['x_2024'].to(device)  # (batch_size, 7, 2)
        labels = batch['labels'].to(device)
        node_indices = batch['node_indices'].to(device)

        # Get full graph features for spatial branch
        x_2021_full = batch['all_features_2021_temporal'].to(device)  # (num_nodes, 7, 2)
        x_2024_full = batch['all_features_2024_temporal'].to(device)  # (num_nodes, 7, 2)

        # Get graphs from model (already stored)
        graphs_2021 = model.graphs_2021
        graphs_2024 = model.graphs_2024

        # Move graphs to device
        graphs_2021 = [(edge_index.to(device), edge_attr.to(device))
                       for edge_index, edge_attr in graphs_2021]
        graphs_2024 = [(edge_index.to(device), edge_attr.to(device))
                       for edge_index, edge_attr in graphs_2024]

        # 转换标签
        intensity_labels, direction_labels = convert_labels_to_hierarchical(labels)

        # 前向传播 - temporal uses batch features, spatial uses full graph
        # We need to modify the model forward to accept both
        # For now, use a workaround: pass full features and node_indices
        intensity_logits, direction_logits, direct_logits = model(
            x_2021_full, x_2024_full, graphs_2021, graphs_2024, node_indices=node_indices
        )

        # 计算三个损失
        loss_intensity = criterion_intensity(intensity_logits, intensity_labels)
        loss_direction = criterion_direction(direction_logits, direction_labels)
        loss_direct = criterion_direct(direct_logits, labels)

        # 组合损失 (可以调整权重)
        loss = (loss_intensity + loss_direction + 0.5 * loss_direct) / accumulation_steps

        # 反向传播 (accumulate gradients)
        loss.backward()

        # 只在累积步数达到时更新参数
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # 统计 (注意loss已经除以accumulation_steps)
        total_loss += loss.item() * accumulation_steps
        total_intensity_loss += loss_intensity.item()
        total_direction_loss += loss_direction.item()
        total_direct_loss += loss_direct.item()

        # 预测
        intensity_pred = torch.argmax(intensity_logits, dim=1)
        direction_pred = torch.argmax(direction_logits, dim=1)
        hierarchical_pred = combine_hierarchical_predictions(intensity_pred, direction_pred)
        direct_pred = torch.argmax(direct_logits, dim=1)

        # 准确率
        batch_size = labels.size(0)
        total_samples += batch_size
        intensity_correct += (intensity_pred == intensity_labels).sum().item()
        direction_correct += (direction_pred == direction_labels).sum().item()
        hierarchical_correct += (hierarchical_pred == labels).sum().item()
        direct_correct += (direct_pred == labels).sum().item()

        # 进度日志
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                       f"Loss: {loss.item():.4f} "
                       f"(I:{loss_intensity.item():.3f} "
                       f"D:{loss_direction.item():.3f} "
                       f"9C:{loss_direct.item():.3f})")

    # 计算平均指标
    avg_loss = total_loss / len(train_loader)
    avg_intensity_loss = total_intensity_loss / len(train_loader)
    avg_direction_loss = total_direction_loss / len(train_loader)
    avg_direct_loss = total_direct_loss / len(train_loader)

    intensity_acc = 100. * intensity_correct / total_samples
    direction_acc = 100. * direction_correct / total_samples
    hierarchical_acc = 100. * hierarchical_correct / total_samples
    direct_acc = 100. * direct_correct / total_samples

    return {
        'loss': avg_loss,
        'intensity_loss': avg_intensity_loss,
        'direction_loss': avg_direction_loss,
        'direct_loss': avg_direct_loss,
        'intensity_acc': intensity_acc,
        'direction_acc': direction_acc,
        'hierarchical_acc': hierarchical_acc,
        'direct_acc': direct_acc
    }


def validate_hierarchical(model, val_loader, device):
    """验证 (层次化版本)"""
    model.eval()

    # 收集所有预测和标签
    all_intensity_preds = []
    all_direction_preds = []
    all_hierarchical_preds = []
    all_direct_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x_2021_batch = batch['x_2021'].to(device)  # (batch_size, 7, 2)
            x_2024_batch = batch['x_2024'].to(device)  # (batch_size, 7, 2)
            labels = batch['labels'].to(device)
            node_indices = batch['node_indices'].to(device)

            # Get full graph features for spatial branch
            x_2021_full = batch['all_features_2021_temporal'].to(device)  # (num_nodes, 7, 2)
            x_2024_full = batch['all_features_2024_temporal'].to(device)  # (num_nodes, 7, 2)

            # Get graphs from model (already stored)
            graphs_2021 = model.graphs_2021
            graphs_2024 = model.graphs_2024

            # Move graphs to device
            graphs_2021 = [(edge_index.to(device), edge_attr.to(device))
                           for edge_index, edge_attr in graphs_2021]
            graphs_2024 = [(edge_index.to(device), edge_attr.to(device))
                           for edge_index, edge_attr in graphs_2024]

            # 前向传播 (use full graph features with node_indices)
            intensity_logits, direction_logits, direct_logits = model(
                x_2021_full, x_2024_full, graphs_2021, graphs_2024, node_indices=node_indices
            )

            # 预测
            intensity_pred = torch.argmax(intensity_logits, dim=1)
            direction_pred = torch.argmax(direction_logits, dim=1)
            hierarchical_pred = combine_hierarchical_predictions(intensity_pred, direction_pred)
            direct_pred = torch.argmax(direct_logits, dim=1)

            # 收集结果
            all_intensity_preds.append(intensity_pred.cpu())
            all_direction_preds.append(direction_pred.cpu())
            all_hierarchical_preds.append(hierarchical_pred.cpu())
            all_direct_preds.append(direct_pred.cpu())
            all_labels.append(labels.cpu())

    # 拼接所有结果
    all_intensity_preds = torch.cat(all_intensity_preds).numpy()
    all_direction_preds = torch.cat(all_direction_preds).numpy()
    all_hierarchical_preds = torch.cat(all_hierarchical_preds).numpy()
    all_direct_preds = torch.cat(all_direct_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # 转换标签
    intensity_labels = all_labels // 3
    direction_labels = all_labels % 3

    # 计算准确率
    intensity_acc = 100. * (all_intensity_preds == intensity_labels).sum() / len(all_labels)
    direction_acc = 100. * (all_direction_preds == direction_labels).sum() / len(all_labels)
    hierarchical_acc = 100. * (all_hierarchical_preds == all_labels).sum() / len(all_labels)
    direct_acc = 100. * (all_direct_preds == all_labels).sum() / len(all_labels)

    # 计算F1
    hierarchical_f1 = f1_score(all_labels, all_hierarchical_preds, average='macro')
    direct_f1 = f1_score(all_labels, all_direct_preds, average='macro')

    return {
        'intensity_acc': intensity_acc,
        'direction_acc': direction_acc,
        'hierarchical_acc': hierarchical_acc,
        'hierarchical_f1': hierarchical_f1,
        'direct_acc': direct_acc,
        'direct_f1': direct_f1,
        'all_hierarchical_preds': all_hierarchical_preds,
        'all_direct_preds': all_direct_preds,
        'all_labels': all_labels
    }


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("Hierarchical Classification Training (3×3)")
    logger.info("=" * 80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/hierarchical_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/metrics", exist_ok=True)
    
    # 添加文件日志处理器，将日志保存到文件
    file_handler = logging.FileHandler(f"{output_dir}/training.log", encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {output_dir}/training.log")

    # 1. 加载数据
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading data")
    logger.info("=" * 80)

    # Load data using prepare_dual_year_experiment_data (with cache support)
    from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data

    data = prepare_dual_year_experiment_data(
        label_path='data/labels.csv',
        samples_per_class=None,  # Use all samples
        use_cache=True
    )

    logger.info(f"✓ Loaded {len(data['labels'])} labeled grids from cache")

    # 打印类别分布
    from collections import Counter
    label_counts = Counter(data['labels'].values())
    logger.info("Class distribution:")
    for cls in range(9):
        count = label_counts.get(cls, 0)
        logger.info(f"  Class {cls+1}: {count} samples")

    # 确保有metadata_df
    if 'metadata_df' not in data:
        logger.warning("metadata_df not in cache, creating empty one")
        import pandas as pd
        data['metadata_df'] = pd.DataFrame()

    # 2. 创建数据集
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating datasets")
    logger.info("=" * 80)

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

    # First, check the feature dimension from the first grid
    first_grid_id = next(iter(data['grid_id_to_idx'].keys()))
    if first_grid_id in data['change_features']:
        feature_dim = data['change_features'][first_grid_id].shape[1]  # 4 or 8
        logger.info(f"Feature dimension detected: (7, {feature_dim})")
    else:
        feature_dim = 4  # Default to flow features only
        logger.warning("Could not detect feature dimension, using default (7, 4)")

    for grid_id in data['grid_id_to_idx'].keys():
        if grid_id in data['change_features']:
            features = data['change_features'][grid_id]  # (7, 4) or (7, 8)

            # Ensure consistent feature dimension
            if features.shape[1] == feature_dim:
                all_features_2021.append(features)
                all_features_2024.append(features)
            else:
                # Pad or truncate to match expected dimension
                if features.shape[1] < feature_dim:
                    # Pad with zeros (missing ellipse features)
                    padded = np.zeros((7, feature_dim))
                    padded[:, :features.shape[1]] = features
                    all_features_2021.append(padded)
                    all_features_2024.append(padded)
                else:
                    # Truncate (shouldn't happen, but handle it)
                    all_features_2021.append(features[:, :feature_dim])
                    all_features_2024.append(features[:, :feature_dim])

    all_features_2021 = np.stack(all_features_2021)  # (N, 7, 4 or 8)
    all_features_2024 = np.stack(all_features_2024)  # (N, 7, 4 or 8)

    logger.info(f"Prepared all_features with shape: {all_features_2021.shape}")

    # 创建collator
    collator = ImprovedGraphBatchCollator(
        graphs_2021=data['graphs_2021'],
        graphs_2024=data['graphs_2024'],
        grid_id_to_idx=data['grid_id_to_idx'],
        all_features_2021=all_features_2021,
        all_features_2024=all_features_2024
    )

    # 创建dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator)

    logger.info(f"✓ Created dataloaders with batch_size={config.BATCH_SIZE}")

    # 3. 创建模型
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Creating model")
    logger.info("=" * 80)

    model = ImprovedDualBranchModel(
        temporal_input_size=config.TEMPORAL_INPUT_SIZE,
        spatial_input_size=config.SPATIAL_INPUT_SIZE,
        num_time_steps=config.TIME_STEPS,
        use_hierarchical=True  # 使用层次化分类
    )

    # Store graphs in model (for easy access)
    # Convert numpy arrays to tensors
    model.graphs_2021 = [
        (torch.LongTensor(edge_index), torch.FloatTensor(edge_attr))
        for edge_index, edge_attr in data['graphs_2021']
    ]
    model.graphs_2024 = [
        (torch.LongTensor(edge_index), torch.FloatTensor(edge_attr))
        for edge_index, edge_attr in data['graphs_2024']
    ]
    logger.info("✓ Stored graph data in model (converted to tensors)")

    # Single GPU with gradient accumulation (avoid OOM and DataParallel issues)
    logger.info("Using single GPU with gradient accumulation")
    logger.info(f"Physical batch size: {config.BATCH_SIZE}, Accumulation steps: 4")
    logger.info(f"Effective batch size: {config.BATCH_SIZE * 4}")

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model initialized with {num_params:,} parameters")
    logger.info(f"✓ Model type: Hierarchical (3×3)")

    # 4. 训练配置
    # 为层次化分类创建权重
    # 原始权重是9个类别的，需要转换为3个类别的权重
    original_weights = data['class_weights']

    # 计算流量强度维度的权重 (合并Classes 1-3, 4-6, 7-9)
    intensity_weights = torch.tensor([
        (original_weights[0] + original_weights[1] + original_weights[2]) / 3,  # 稳定
        (original_weights[3] + original_weights[4] + original_weights[5]) / 3,  # 增长
        (original_weights[6] + original_weights[7] + original_weights[8]) / 3,  # 衰减
    ])

    # 计算空间方向维度的权重 (合并Classes 1,4,7 / 2,5,8 / 3,6,9)
    direction_weights = torch.tensor([
        (original_weights[0] + original_weights[3] + original_weights[6]) / 3,  # 均衡
        (original_weights[1] + original_weights[4] + original_weights[7]) / 3,  # 聚集
        (original_weights[2] + original_weights[5] + original_weights[8]) / 3,  # 扩散
    ])

    logger.info(f"Original class weights: {original_weights.tolist()}")
    logger.info(f"Intensity weights: {intensity_weights.tolist()}")
    logger.info(f"Direction weights: {direction_weights.tolist()}")

    # 创建三个损失函数
    criterion_intensity = torch.nn.CrossEntropyLoss(weight=intensity_weights.to(device))
    criterion_direction = torch.nn.CrossEntropyLoss(weight=direction_weights.to(device))
    criterion_direct = torch.nn.CrossEntropyLoss(weight=original_weights.to(device))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    # 5. 训练
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Training")
    logger.info("=" * 80)

    best_hierarchical_acc = 0.0
    best_direct_acc = 0.0
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # 训练
        train_metrics = train_epoch_hierarchical(
            model, train_loader, criterion_intensity, criterion_direction, criterion_direct, optimizer, device, epoch
        )

        # 验证
        val_metrics = validate_hierarchical(model, val_loader, device)

        # 学习率调度
        scheduler.step(val_metrics['hierarchical_acc'])
        current_lr = optimizer.param_groups[0]['lr']

        # 日志
        logger.info(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"    - Intensity: {train_metrics['intensity_acc']:.2f}%")
        logger.info(f"    - Direction: {train_metrics['direction_acc']:.2f}%")
        logger.info(f"    - Hierarchical (3×3): {train_metrics['hierarchical_acc']:.2f}%")
        logger.info(f"    - Direct (9-class): {train_metrics['direct_acc']:.2f}%")
        logger.info(f"  Val Accuracy:")
        logger.info(f"    - Intensity: {val_metrics['intensity_acc']:.2f}%")
        logger.info(f"    - Direction: {val_metrics['direction_acc']:.2f}%")
        logger.info(f"    - Hierarchical (3×3): {val_metrics['hierarchical_acc']:.2f}% | F1: {val_metrics['hierarchical_f1']:.4f}")
        logger.info(f"    - Direct (9-class): {val_metrics['direct_acc']:.2f}% | F1: {val_metrics['direct_f1']:.4f}")
        logger.info(f"  LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_metrics['hierarchical_acc'] > best_hierarchical_acc:
            best_hierarchical_acc = val_metrics['hierarchical_acc']
            best_direct_acc = val_metrics['direct_acc']
            patience_counter = 0

            # Save model state dict (single GPU, no DataParallel)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hierarchical_acc': val_metrics['hierarchical_acc'],
                'hierarchical_f1': val_metrics['hierarchical_f1'],
                'direct_acc': val_metrics['direct_acc'],
                'direct_f1': val_metrics['direct_f1'],
            }, f"{output_dir}/models/best_model.pth")

            logger.info(f"  ✓ Saved best model")
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"\n✓ Early stopping at epoch {epoch}")
                break

    logger.info(f"\n✓ Training completed!")
    logger.info(f"  Best Hierarchical Acc: {best_hierarchical_acc:.2f}%")
    logger.info(f"  Best Direct Acc: {best_direct_acc:.2f}%")

    # 6. 测试
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Testing")
    logger.info("=" * 80)

    # 加载最佳模型 (single GPU, no DataParallel)
    checkpoint = torch.load(f"{output_dir}/models/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded best model from epoch {checkpoint['epoch']}")

    # 测试
    test_metrics = validate_hierarchical(model, test_loader, device)

    logger.info(f"\n" + "=" * 80)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Intensity Accuracy: {test_metrics['intensity_acc']:.2f}%")
    logger.info(f"Direction Accuracy: {test_metrics['direction_acc']:.2f}%")
    logger.info(f"Hierarchical (3×3) Accuracy: {test_metrics['hierarchical_acc']:.2f}%")
    logger.info(f"Hierarchical (3×3) F1-Macro: {test_metrics['hierarchical_f1']:.4f}")
    logger.info(f"Direct (9-class) Accuracy: {test_metrics['direct_acc']:.2f}%")
    logger.info(f"Direct (9-class) F1-Macro: {test_metrics['direct_f1']:.4f}")
    logger.info("=" * 80)

    # 保存分类报告
    logger.info("\nHierarchical (3×3) Classification Report:")
    report_hierarchical = classification_report(
        test_metrics['all_labels'],
        test_metrics['all_hierarchical_preds'],
        target_names=[f'Class {i+1}' for i in range(9)],
        digits=2
    )
    logger.info("\n" + report_hierarchical)

    logger.info("\nDirect (9-class) Classification Report:")
    report_direct = classification_report(
        test_metrics['all_labels'],
        test_metrics['all_direct_preds'],
        target_names=[f'Class {i+1}' for i in range(9)],
        digits=2
    )
    logger.info("\n" + report_direct)

    # 保存结果
    results = {
        'test_intensity_acc': float(test_metrics['intensity_acc']),
        'test_direction_acc': float(test_metrics['direction_acc']),
        'test_hierarchical_acc': float(test_metrics['hierarchical_acc']),
        'test_hierarchical_f1': float(test_metrics['hierarchical_f1']),
        'test_direct_acc': float(test_metrics['direct_acc']),
        'test_direct_f1': float(test_metrics['direct_f1']),
    }

    with open(f"{output_dir}/metrics/test_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
