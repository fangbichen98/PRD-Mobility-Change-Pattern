# 小型实验运行指南

## 实验概述

本实验从每个类别（共9类）中抽取100个样本，进行快速的模型训练、测试和可视化。

## 运行步骤

### 1. 安装依赖

```bash
cd D:\Code_File\mobility_analysis
pip install -r requirements.txt
```

### 2. 运行小型实验

```bash
python run_small_experiment.py
```

## 实验流程

### 步骤1: 数据采样
- 从 `data/labels_1w.csv` 中每个类别随机抽取100个样本
- 生成 `data/labels_sampled.csv`（约900个样本）
- 显示原始和采样后的类别分布

### 步骤2: 数据预处理
- 加载格网元数据
- 加载2021年OD流量数据（仅针对采样的格网）
- 过滤前7天（168小时）数据
- Z-score标准化流量
- 构建时序特征（inflow/outflow）
- 构建空间图（K-NN + 流量图）

### 步骤3: 模型训练
- 模型：双分支时空模型（LSTM-SPP + DySAT）
- 数据划分：70% 训练 / 10% 验证 / 20% 测试
- Batch size: 16
- Epochs: 50（早停patience=10）
- 优化器：Adam (lr=0.001)
- 损失函数：CrossEntropyLoss

### 步骤4: 模型评估
- 在测试集上评估
- 计算指标：
  - 总体准确率
  - F1分数（宏平均和加权平均）
  - 每个类别的F1分数
  - 精确率和召回率
  - 混淆矩阵（9×9）

### 步骤5: 可视化
生成以下可视化结果（保存在 `outputs/figures/small_experiment/`）：

1. **空间分布图**
   - `true_labels.jpg` - 真实标签的地理分布
   - `predicted_labels.jpg` - 预测标签的地理分布

2. **时序模式图**
   - `class_temporal_patterns.jpg` - 9个类别的平均时序模式（3×3网格）
   - `temporal_class_X.jpg` - 每个类别的样本时序曲线

3. **评估图表**
   - `dual_branch_small_confusion_matrix.jpg` - 混淆矩阵
   - `dual_branch_small_f1_scores.jpg` - 每类F1分数柱状图

## 预期输出

### 控制台输出示例

```
======================================================================
SMALL-SCALE MOBILITY PATTERN CLASSIFICATION EXPERIMENT
======================================================================
Using device: cuda

======================================================================
Step 1: Sampling Balanced Labels
======================================================================
Total labels: 10000

Original label distribution:
  Class 1: 1234 samples
  Class 2: 1456 samples
  Class 3: 987 samples
  ...
  Class 9: 1123 samples

Sampled 900 total labels
Saved to: data/labels_sampled.csv

Sampled label distribution:
  Class 1: 100 samples
  Class 2: 100 samples
  ...
  Class 9: 100 samples

======================================================================
Step 2: Data Preparation
======================================================================
Loading grid metadata...
Loaded 123456 valid grid cells

Loading sampled labels from data/labels_sampled.csv...
Valid sampled grids: 900

Loading OD flow data for 2021...
Loaded 2345678 OD records for sampled grids
After filtering to 7 days: 1234567 records

Building temporal features...
Aggregating grid flows...
Aggregated flows for 900 grids

Building spatial graph...

Data preparation completed:
  - Sampled grids: 900
  - OD records: 1234567
  - Graph edges: 7200

======================================================================
Step 3: Training DUAL_BRANCH Model
======================================================================
Dataset size: 900 samples
Train: 630, Val: 90, Test: 180
Model parameters: 1234567

Epoch 1/50
  Train - Loss: 2.1234, Acc: 0.2345, F1: 0.2123
  Val   - Loss: 2.0123, Acc: 0.2567, F1: 0.2345
  Saved best model (Acc: 0.2567, F1: 0.2345)

...

Epoch 25/50
  Train - Loss: 0.5678, Acc: 0.8234, F1: 0.8123
  Val   - Loss: 0.6789, Acc: 0.7890, F1: 0.7678
  Saved best model (Acc: 0.7890, F1: 0.7678)

Early stopping triggered after 35 epochs
Training completed
Best validation accuracy: 0.7890
Best validation F1: 0.7678

======================================================================
Step 4: Evaluation on Test Set
======================================================================
==================================================
Evaluation Results
==================================================
Accuracy: 0.7667
F1 Score (Macro): 0.7523
F1 Score (Weighted): 0.7589
Precision (Macro): 0.7612
Recall (Macro): 0.7498

Per-class F1 Scores:
  Class 1: 0.7234
  Class 2: 0.7890
  Class 3: 0.7123
  Class 4: 0.7456
  Class 5: 0.7678
  Class 6: 0.7345
  Class 7: 0.7567
  Class 8: 0.7789
  Class 9: 0.7234

======================================================================
Step 5: Generating Visualizations
======================================================================
Creating spatial visualizations...
Saved spatial distribution plot to outputs/figures/small_experiment/true_labels.jpg
Saved spatial distribution plot to outputs/figures/small_experiment/predicted_labels.jpg

Creating temporal visualizations...
Saved class temporal patterns to outputs/figures/small_experiment/class_temporal_patterns.jpg
...

Visualizations saved to outputs/figures/small_experiment

======================================================================
EXPERIMENT SUMMARY
======================================================================

Metric                    Value
----------------------------------------
Accuracy                  0.7667
F1 Score (Macro)          0.7523
F1 Score (Weighted)       0.7589
Precision (Macro)         0.7612
Recall (Macro)            0.7498

Class      F1 Score
-------------------------
Class 1    0.7234
Class 2    0.7890
Class 3    0.7123
Class 4    0.7456
Class 5    0.7678
Class 6    0.7345
Class 7    0.7567
Class 8    0.7789
Class 9    0.7234

======================================================================

Results saved to: outputs/small_experiment_results.json

Total experiment time: 1234.56 seconds (20.58 minutes)
======================================================================
EXPERIMENT COMPLETED SUCCESSFULLY!
======================================================================
```

## 输出文件

### 1. 采样标签
- `data/labels_sampled.csv` - 采样后的标签文件（900条）

### 2. 模型检查点
- `checkpoints/best_model.pth` - 最佳模型权重
- `checkpoints/confusion_matrix.npy` - 混淆矩阵

### 3. 训练日志
- `outputs/logs/run_YYYYMMDD_HHMMSS/` - TensorBoard日志

### 4. 评估报告
- `outputs/evaluation_reports/dual_branch_small_metrics.txt` - 文本格式指标
- `outputs/evaluation_reports/dual_branch_small_confusion_matrix.jpg` - 混淆矩阵图
- `outputs/evaluation_reports/dual_branch_small_f1_scores.jpg` - F1分数图

### 5. 可视化结果
- `outputs/figures/small_experiment/true_labels.jpg` - 真实标签空间分布
- `outputs/figures/small_experiment/predicted_labels.jpg` - 预测标签空间分布
- `outputs/figures/small_experiment/class_temporal_patterns.jpg` - 类别时序模式
- `outputs/figures/small_experiment/temporal_class_1.jpg` ~ `temporal_class_9.jpg` - 各类样本时序

### 6. 结果摘要
- `outputs/small_experiment_results.json` - JSON格式的完整结果

## 预期性能

基于900个样本的小型实验，预期性能：

- **准确率**: 70-80%
- **F1分数（宏平均）**: 68-78%
- **训练时间**: 15-30分钟（GPU）/ 1-2小时（CPU）

注意：
- 小样本量可能导致某些类别性能不稳定
- 实际性能取决于数据质量和类别平衡性
- GPU训练会显著加快速度

## 查看TensorBoard

```bash
tensorboard --logdir outputs/logs
# 访问 http://localhost:6006
```

可以实时查看：
- 训练/验证损失曲线
- 准确率曲线
- F1分数曲线
- 每个类别的F1分数

## 故障排除

### 问题1: 内存不足
**解决方案**:
- 减少batch_size（在run_small_experiment.py中修改为8或4）
- 减少每类样本数（修改samples_per_class=50）

### 问题2: CUDA不可用
**解决方案**:
- 模型会自动切换到CPU
- 训练时间会更长

### 问题3: 数据加载慢
**解决方案**:
- 这是正常的，因为OD数据文件很大（12GB+）
- 脚本会自动过滤只加载相关格网的数据

## 手动采样（如果自动脚本失败）

如果自动脚本无法运行，可以手动采样：

```bash
python sample_labels_simple.py
```

这会生成 `data/labels_sampled.csv`，然后可以手动运行后续步骤。

## 联系与支持

如有问题，请检查：
1. Python版本 >= 3.8
2. 所有依赖已安装（requirements.txt）
3. 数据文件完整（2021.csv, labels_1w.csv, grid_metadata）
4. 有足够的磁盘空间（至少20GB）
