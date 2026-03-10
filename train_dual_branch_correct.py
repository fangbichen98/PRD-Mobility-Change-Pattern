"""
完整的双分支模型实现 - 参考 train_multiscale_temporal.py

关键改进：
1. 使用 PureGraphDualYearDataset + PureGraphBatchCollator（与参考文件一致）
2. 使用 LSTMOnlyBranch 作为时序分支（68.84% baseline）
3. 使用 PureGraphDualYearGCN 作为空间分支
4. 正确的融合策略：6个特征 → GatedFusion → 分类器
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
from datetime import datetime
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data
from src.training.dataset_pure_graph import PureGraphDualYearDataset, PureGraphBatchCollator

# 导入模型组件
from src.models.temporal_branch_ablation import LSTMOnlyBranch
from src.models.spatial_branch_pure_graph import PureGraphDualYearGCN
from src.models.gated_fusion import GatedFeatureFusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_time(seconds):
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class CorrectDualBranchModel(nn.Module):
    """
    正确实现的双分支模型

    架构：
    1. 时序分支: LSTMOnlyBranch → temporal_2021, temporal_2024, temporal_diff
    2. 空间分支: PureGraphDualYearGCN → spatial_2021, spatial_2024, spatial_diff
    3. 融合: 6个特征 → GatedFeatureFusion → 256维
    4. 分类器: 256 → 128 → 9类
    """

    def __init__(self, hidden_size=256, num_classes=9, dropout=0.4):
        super().__init__()

        # 时序分支：使用LSTMOnlyBranch（68.84% baseline）
        self.temporal_branch = LSTMOnlyBranch(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            dropout=dropout,
            output_size=hidden_size
        )

        # 空间分支：PureGraphDualYearGCN
        self.spatial_branch = PureGraphDualYearGCN(
            hidden_size=128,
            num_layers=3,
            dropout=dropout,
            output_size=hidden_size
        )

        # 融合层：6个特征 → 256维
        self.fusion = GatedFeatureFusion(
            feature_size=hidden_size,
            num_features=6,  # temporal_2021, temporal_2024, temporal_diff, spatial_2021, spatial_2024, spatial_diff
            dropout=dropout
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices):
        """
        前向传播

        Args:
            x_2021: 全图2021时序特征 (num_nodes, 168, 1)
            x_2024: 全图2024时序特征 (num_nodes, 168, 1)
            graphs_2021: [(edge_index, edge_attr)] for 2021
            graphs_2024: [(edge_index, edge_attr)] for 2024
            num_nodes: 总节点数
            node_indices: batch节点的索引 (batch,)

        Returns:
            logits: (batch, 9)
        """
        # ===== 时序分支 =====
        # 提取batch节点的时序特征
        x_2021_batch = x_2021[node_indices]  # (batch, 168, 1)
        x_2024_batch = x_2024[node_indices]  # (batch, 168, 1)

        # LSTMOnlyBranch返回 (batch, 3, hidden_size): [h_2021, h_2024, h_diff]
        temporal_features = self.temporal_branch(x_2021_batch, x_2024_batch)

        # 提取3个时序特征
        temporal_2021 = temporal_features[:, 0, :]  # (batch, hidden_size)
        temporal_2024 = temporal_features[:, 1, :]  # (batch, hidden_size)
        temporal_diff = temporal_features[:, 2, :]  # (batch, hidden_size)

        # ===== 空间分支 =====
        spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
            graphs_2021, graphs_2024, num_nodes, node_indices
        )  # 每个: (batch, hidden_size)

        # ===== 融合 =====
        # Stack 6个特征: (batch, 6, hidden_size)
        all_features = torch.stack([
            temporal_2021, temporal_2024, temporal_diff,
            spatial_2021, spatial_2024, spatial_diff
        ], dim=1)

        # Gated fusion
        fused = self.fusion(all_features)  # (batch, hidden_size)

        # ===== 分类 =====
        logits = self.classifier(fused)  # (batch, 9)

        return logits


def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=4, grad_clip_norm=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    correct = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        node_indices = batch['node_indices'].to(device)
        labels = batch['labels'].to(device)
        num_nodes = batch['num_nodes']

        # Get full temporal features
        all_temporal_2021 = batch['all_temporal_2021'].to(device)
        all_temporal_2024 = batch['all_temporal_2024'].to(device)

        # Move graphs to device
        graphs_2021 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2021']]
        graphs_2024 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2024']]

        # Forward pass
        logits = model(
            x_2021=all_temporal_2021,
            x_2024=all_temporal_2024,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass with gradient accumulation
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Compute predictions
        pred = logits.argmax(dim=1)

        # Update metrics
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct += (pred == labels).sum().item()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Epoch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # Final gradient update if needed
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()

    # Compute average metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for batch in data_loader:
        # Move data to device
        node_indices = batch['node_indices'].to(device)
        labels = batch['labels'].to(device)
        num_nodes = batch['num_nodes']

        # Get full temporal features
        all_temporal_2021 = batch['all_temporal_2021'].to(device)
        all_temporal_2024 = batch['all_temporal_2024'].to(device)

        # Move graphs to device
        graphs_2021 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2021']]
        graphs_2024 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2024']]

        # Forward pass
        logits = model(
            x_2021=all_temporal_2021,
            x_2024=all_temporal_2024,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

        # Compute loss
        loss = criterion(logits, labels)

        # Update metrics
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Compute predictions
        pred = logits.argmax(dim=1)

        # Collect predictions
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * (all_preds == all_labels).sum() / total_samples
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1, all_labels, all_preds


def main():
    """主训练函数"""
    start_time = time.time()
    start_datetime = datetime.now()

    logger.info("=" * 80)
    logger.info("完整的双分支模型训练")
    logger.info("=" * 80)
    logger.info(f"训练开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n架构:")
    logger.info("  - 时序分支: LSTMOnlyBranch (68.84% baseline)")
    logger.info("  - 空间分支: PureGraphDualYearGCN")
    logger.info("  - 融合: GatedFeatureFusion (6 features)")
    logger.info("  - 分类器: MLP (256 → 128 → 9)")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/dual_branch_correct_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)

    file_handler = logging.FileHandler(f"{output_dir}/training.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # 加载数据
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: 加载数据")
    logger.info("=" * 80)

    data = prepare_dual_year_experiment_data(
        label_path=config.LABEL_PATH,
        samples_per_class=None,
        use_cache=True
    )

    logger.info(f"✓ 数据加载成功")
    logger.info(f"  - 总网格数: {len(data['labels'])}")

    # 准备时序特征
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: 准备时序特征")
    logger.info("=" * 80)

    temporal_features_2021 = {}
    temporal_features_2024 = {}

    for grid_id, features in data['change_features'].items():
        temporal_features_2021[grid_id] = features[:, [0]]  # (168, 1) - total_log only
        temporal_features_2024[grid_id] = features[:, [1]]  # (168, 1)

    logger.info(f"✓ 时序特征准备完成 (单特征: total_log)")

    # 创建数据集
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: 创建数据集")
    logger.info("=" * 80)

    grid_ids = list(data['labels'].keys())
    dataset = PureGraphDualYearDataset(
        temporal_features_2021=temporal_features_2021,
        temporal_features_2024=temporal_features_2024,
        labels=data['labels'],
        grid_ids=grid_ids
    )

    logger.info(f"✓ 数据集创建成功: {len(dataset)} 个样本")

    # 划分数据集
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = int(config.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    logger.info(f"  - 训练集: {len(train_dataset)}")
    logger.info(f"  - 验证集: {len(val_dataset)}")
    logger.info(f"  - 测试集: {len(test_dataset)}")

    # 准备全图时序特征
    num_nodes = len(data['grid_id_to_idx'])
    all_temporal_2021 = torch.zeros(num_nodes, 168, 1)
    all_temporal_2024 = torch.zeros(num_nodes, 168, 1)

    for grid_id, idx in data['grid_id_to_idx'].items():
        if grid_id in temporal_features_2021:
            all_temporal_2021[idx] = torch.tensor(temporal_features_2021[grid_id], dtype=torch.float32)
            all_temporal_2024[idx] = torch.tensor(temporal_features_2024[grid_id], dtype=torch.float32)

    # 创建collator
    collator = PureGraphBatchCollator(
        graphs_2021=data['graphs_2021'],
        graphs_2024=data['graphs_2024'],
        grid_id_to_idx=data['grid_id_to_idx'],
        all_temporal_2021=all_temporal_2021,
        all_temporal_2024=all_temporal_2024
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=collator, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=collator, num_workers=0
    )

    logger.info(f"✓ 数据加载器创建成功")

    # 创建模型
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: 创建模型")
    logger.info("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    model = CorrectDualBranchModel(hidden_size=256, num_classes=9, dropout=0.4)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ 模型创建成功")
    logger.info(f"  - 总参数: {total_params:,}")
    logger.info(f"  - 时序分支: LSTMOnlyBranch (128 hidden)")
    logger.info(f"  - 空间分支: PureGraphDualYearGCN (128 hidden)")
    logger.info(f"  - 融合: GatedFeatureFusion (6 features)")
    logger.info(f"  - 分类器: MLP (256 → 128 → 9)")

    # 损失函数和优化器
    class_weights = data['class_weights'].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    logger.info(f"✓ 训练设置完成")
    logger.info(f"  - 优化器: Adam (lr={config.LEARNING_RATE})")
    logger.info(f"  - Scheduler: ReduceLROnPlateau (patience=5)")
    logger.info(f"  - Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")

    # 训练循环
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: 训练")
    logger.info("=" * 80)

    best_accuracy = 0
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, accumulation_steps=4
        )

        # 验证
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        # Log metrics
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Train Accuracy: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Accuracy: {val_acc:.2f}% | F1: {val_f1:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  LR: {current_lr:.6f}")

        scheduler.step(val_acc)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'f1': val_f1
            }, f"{output_dir}/models/best_model.pth")

            logger.info(f"  ✓ 新的最佳模型! (Acc: {best_accuracy:.2f}%)")
        else:
            patience_counter += 1

        logger.info(f"  Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"\n✓ Early stopping at epoch {epoch + 1}")
            break

    logger.info(f"\n✓ 训练完成! 最佳准确率: {best_accuracy:.2f}%")

    # 测试
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: 测试")
    logger.info("=" * 80)

    checkpoint = torch.load(f"{output_dir}/models/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ 加载最佳模型 (epoch {checkpoint['epoch'] + 1})")

    test_loss, test_acc, test_f1, all_labels, all_preds = evaluate(model, test_loader, criterion, device)

    logger.info(f"\n测试结果:")
    logger.info(f"  - 准确率: {test_acc:.2f}%")
    logger.info(f"  - F1 Score: {test_f1:.4f}")

    # 保存详细测试结果
    test_results = {
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'best_val_accuracy': float(best_accuracy),
        'lstm_baseline': 68.84,
        'improvement_over_baseline': float(best_accuracy - 68.84),
        'model_architecture': {
            'temporal_branch': 'LSTMOnlyBranch',
            'spatial_branch': 'PureGraphDualYearGCN',
            'fusion': 'GatedFeatureFusion (6 features)',
            'classifier': 'MLP (256 → 128 → 9)'
        }
    }

    os.makedirs(f"{output_dir}/metrics", exist_ok=True)
    with open(f"{output_dir}/metrics/test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    # 保存分类报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[f'Class {i+1}' for i in range(9)],
        zero_division=0
    )

    with open(f"{output_dir}/metrics/classification_report.txt", 'w') as f:
        f.write("9-Class Classification Report - Correct Dual Branch Model\n")
        f.write("=" * 80 + "\n\n")

        # 写入数据信息
        f.write("Data Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Label File: {data.get('label_file_name', 'N/A')}\n")
        f.write(f"  Label File Hash: {data.get('label_file_hash', 'N/A')}\n")
        f.write(f"  Total Samples: {len(data['labels'])}\n")
        f.write("\n")

        # 写入类别分布
        f.write("Class Distribution:\n")
        f.write("-" * 80 + "\n")
        class_dist = data.get('class_distribution', {})
        for i in range(config.NUM_CLASSES):
            count = class_dist.get(i, 0)
            weight = data['class_weights'][i].item()
            f.write(f"  Class {i+1}: {count} samples (weight: {weight:.4f})\n")
        f.write("\n")

        # 写入数据配置
        f.write("Data Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Flow Threshold: {config.FLOW_THRESHOLD}\n")
        f.write(f"  Time Steps: {config.TIME_STEPS} ({config.TIME_STEPS // 24} days × 24 hours)\n")
        f.write(f"  Graph 2021 Edges: {int(data['graphs_2021'][0][0].shape[1])}\n")
        f.write(f"  Graph 2024 Edges: {int(data['graphs_2024'][0][0].shape[1])}\n")
        f.write("\n")

        # 写入模型架构
        f.write("Model Architecture:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Temporal Branch:\n")
        f.write(f"    - Type: LSTMOnlyBranch\n")
        f.write(f"    - LSTM Layers: {config.LSTM_LAYERS}\n")
        f.write(f"    - LSTM Hidden Size: 128\n")
        f.write(f"    - LSTM Dropout: 0.4\n")
        f.write(f"    - Temporal Input Size: 1 (total_log only)\n")
        f.write(f"  Spatial Branch:\n")
        f.write(f"    - Type: PureGraphDualYearGCN\n")
        f.write(f"    - GCN Layers: 3\n")
        f.write(f"    - GCN Hidden Size: 128\n")
        f.write(f"    - GCN Heads: 4\n")
        f.write(f"  Fusion:\n")
        f.write(f"    - Type: GatedFeatureFusion\n")
        f.write(f"    - Num Features: 6 (temporal_2021, temporal_2024, temporal_diff,\n")
        f.write(f"                        spatial_2021, spatial_2024, spatial_diff)\n")
        f.write(f"    - Hidden Size: 256\n")
        f.write(f"  Classifier:\n")
        f.write(f"    - Architecture: MLP (256 → 128 → 9)\n")
        f.write(f"    - Dropout: 0.4\n")
        f.write("\n")

        # 写入训练配置
        f.write("Training Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"  Weight Decay: {config.WEIGHT_DECAY}\n")
        f.write(f"  Max Epochs: {config.NUM_EPOCHS}\n")
        f.write(f"  Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}\n")
        f.write(f"  Gradient Accumulation Steps: 4\n")
        f.write(f"  Train/Val/Test Split: {config.TRAIN_SPLIT}/{config.VAL_SPLIT}/{config.TEST_SPLIT}\n")
        f.write(f"  Random Seed: {config.RANDOM_SEED}\n")
        f.write("\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)

    # 保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    np.save(f"{output_dir}/metrics/confusion_matrix.npy", cm)

    logger.info(f"\n✓ 所有结果已保存到 {output_dir}/metrics/")
    logger.info("  - test_results.json")
    logger.info("  - classification_report.txt")
    logger.info("  - confusion_matrix.npy")

    # 计算并记录总训练时间
    end_time = time.time()
    end_datetime = datetime.now()
    total_time = end_time - start_time

    logger.info("\n" + "=" * 80)
    logger.info("训练时间统计")
    logger.info("=" * 80)
    logger.info(f"开始时间:      {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结束时间:      {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总时长:        {format_time(total_time)} ({total_time:.2f} 秒)")
    logger.info(f"               ({total_time/60:.2f} 分钟, {total_time/3600:.2f} 小时)")
    logger.info("=" * 80)

    # 保存时间信息到文件
    timing_info = {
        "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        "total_time_seconds": total_time,
        "total_time_formatted": format_time(total_time),
        "total_time_minutes": total_time / 60,
        "total_time_hours": total_time / 3600
    }

    with open(f"{output_dir}/metrics/timing_info.json", 'w') as f:
        json.dump(timing_info, f, indent=2)

    logger.info(f"✓ 时间信息已保存到 {output_dir}/metrics/timing_info.json")
    logger.info("=" * 80)

    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("实验总结")
    logger.info("=" * 80)
    logger.info(f"LSTM时序baseline: 68.84%")
    logger.info(f"LSTM+GCN融合准确率: {best_accuracy:.2f}%")
    logger.info(f"提升幅度: {best_accuracy - 68.84:+.2f}%")
    if best_accuracy >= 70.0:
        logger.info(f"🎉 成功突破70%!")
    else:
        logger.info(f"距离70%目标还有: {70.0 - best_accuracy:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
