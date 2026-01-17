# 两个实验文件的详细对比

## 文件对比：run_dual_year_experiment.py vs run_improved_dual_year_experiment.py

### 📊 基本信息

| 特性 | run_dual_year_experiment.py | run_improved_dual_year_experiment.py |
|------|----------------------------|-------------------------------------|
| 文件行数 | 440 行 | 532 行 |
| 创建时间 | 原始版本 | 改进版本 |
| 状态 | 已修改（添加了GPU检测但禁用了DataParallel） | 最新改进版本 |

---

## 🔍 关键区别

### 1. **Dataset类实现**

#### run_dual_year_experiment.py
```python
class DualYearDataset(MobilityDataset):
    """Dataset for dual-year mobility change classification"""

    def __getitem__(self, idx):
        # ...
        temporal_features = self.change_features[grid_id]

        # ❌ 错误：将空间特征展平
        spatial_features = temporal_features.flatten()  # (1680,)
```

#### run_improved_dual_year_experiment.py
```python
class ImprovedDualYearDataset(MobilityDataset):
    """
    Improved dataset for dual-year mobility change classification
    CRITICAL FIX: Does NOT flatten spatial features
    """

    def __getitem__(self, idx):
        # ...
        temporal_features = self.change_features[grid_id]

        # ✅ 正确：保持2D结构
        spatial_features = temporal_features  # (168, 10) - NOT FLATTENED
```

**影响**：
- ❌ 原版：展平破坏了时间结构，DySAT无法使用时间注意力
- ✅ 改进版：保留时间结构，DySAT可以正确应用时间注意力

---

### 2. **GPU配置**

#### run_dual_year_experiment.py
```python
# Check device and GPU count
if torch.cuda.is_available() and device == 'cuda':
    device = 'cuda'
    gpu_count = torch.cuda.device_count()
    logger.info(f"Available GPUs: {gpu_count}")
    # 显示所有GPU信息

# ❌ DataParallel被禁用（因为GNN不兼容）
if device == 'cuda' and gpu_count > 1 and use_multi_gpu:
    logger.info("Note: Multi-GPU training with DataParallel is not compatible with GNN")
    logger.info("Using single GPU for GNN training")
    # model = nn.DataParallel(model)  # Disabled
```

#### run_improved_dual_year_experiment.py
```python
# Check device and GPU count
if torch.cuda.is_available() and device == 'cuda':
    device = 'cuda'
    gpu_count = torch.cuda.device_count()
    logger.info(f"Available GPUs: {gpu_count}")
    # 显示所有GPU信息
    logger.info("Note: Graph Neural Networks (GNN) don't support standard DataParallel")
    logger.info("Using single GPU for GNN training (standard practice)")
```

**影响**：
- 两者都使用单GPU训练（GNN限制）
- 改进版有更清晰的说明

---

### 3. **类别权重支持**

#### run_dual_year_experiment.py
```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    checkpoint_dir=dirs['models'],
    class_weights=data['class_weights']  # ✅ 有类别权重
)
```

#### run_improved_dual_year_experiment.py
```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    checkpoint_dir=dirs['models'],
    log_dir=dirs['logs'],
    class_weights=data['class_weights']  # ✅ 有类别权重
)
```

**影响**：
- ✅ 两者都支持类别权重
- 改进版额外指定了log_dir

---

### 4. **日志输出**

#### run_dual_year_experiment.py
```python
logger.info(f"Training completed!")
logger.info(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
logger.info(f"Best validation F1 score: {best_f1:.4f}")

# 测试结果
logger.info(f"Test Results:")
logger.info(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
logger.info(f"  F1 Score (Macro): {results['f1_macro']:.4f}")
# ... 详细指标
```

#### run_improved_dual_year_experiment.py
```python
logger.info(f"Training completed!")
logger.info(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
logger.info(f"Best validation F1 score: {best_f1:.4f}")

# 测试结果
logger.info(f"Test Results:")
logger.info(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
logger.info(f"  F1 Score (Macro): {results['f1_macro']:.4f}")
# ... 详细指标
logger.info(f"Per-class F1 scores:")
for i, f1 in enumerate(results['f1_per_class']):
    logger.info(f"  Class {i+1}: {f1:.4f}")
```

**影响**：
- ✅ 两者都有增强的日志
- 改进版有更详细的per-class F1输出

---

### 5. **默认配置**

#### run_dual_year_experiment.py (main函数)
```python
experiment_name="multi_gpu_dual_year_2021vs2024",  # ❌ 名称误导（实际单GPU）
model_type='dual_branch',
samples_per_class=None,  # ✅ 使用全部样本
num_epochs=100,
batch_size=64,  # ✅ 适合单GPU
device='cuda',
use_multi_gpu=True  # ❌ 实际被忽略
```

#### run_improved_dual_year_experiment.py (main函数)
```python
experiment_name="improved_full_dual_year_2021vs2024",  # ✅ 名称准确
model_type='dual_branch',
samples_per_class=None,  # ✅ 使用全部样本
num_epochs=100,
batch_size=64,  # ✅ 适合单GPU
device='cuda',
label_path='data/labels.csv'  # ✅ 明确指定标签文件
```

**影响**：
- 改进版配置更清晰
- 没有误导性的multi_gpu参数

---

### 6. **特征维度处理**

#### run_dual_year_experiment.py
```python
# 模型初始化
model = DualBranchSTModel(
    temporal_input_size=10,  # 10 features per time step
    spatial_input_size=10    # 10 features per time step (NOT 168*10)
)

# ❌ 但Dataset返回的是展平的 (1680,)
# 导致维度不匹配！
```

#### run_improved_dual_year_experiment.py
```python
# 模型初始化
model = DualBranchSTModel(
    temporal_input_size=10,  # 10 features per time step
    spatial_input_size=10    # 10 features per time step (for 3D input)
)

# ✅ Dataset返回 (168, 10)
# 维度匹配！
```

**影响**：
- ❌ 原版：维度不匹配可能导致错误
- ✅ 改进版：维度正确匹配

---

## 📋 功能对比表

| 功能 | run_dual_year_experiment.py | run_improved_dual_year_experiment.py |
|------|----------------------------|-------------------------------------|
| **空间特征处理** | ❌ 展平 (1680,) | ✅ 保持2D (168, 10) |
| **GPU检测** | ✅ 有 | ✅ 有 |
| **多GPU支持** | ❌ 禁用（GNN限制） | ❌ 不支持（GNN限制） |
| **类别权重** | ✅ 支持 | ✅ 支持 |
| **增强日志** | ✅ 有 | ✅ 有（更详细） |
| **全量数据** | ✅ 支持 | ✅ 支持 |
| **批次大小** | 64 | 64 |
| **Epochs** | 100 | 100 |
| **维度一致性** | ❌ 不一致 | ✅ 一致 |
| **代码清晰度** | 中等 | ✅ 高 |
| **文档说明** | 基本 | ✅ 详细 |

---

## 🎯 推荐使用

### ✅ 推荐：run_improved_dual_year_experiment.py

**原因**：
1. ✅ **正确的特征处理**：不展平空间特征，保留时间结构
2. ✅ **维度一致性**：Dataset输出与模型输入匹配
3. ✅ **更好的文档**：详细的注释和说明
4. ✅ **清晰的配置**：没有误导性的参数
5. ✅ **更详细的日志**：包含per-class F1分数

### ⚠️ 不推荐：run_dual_year_experiment.py

**问题**：
1. ❌ **特征展平错误**：破坏时间结构
2. ❌ **维度不匹配**：Dataset返回(1680,)但模型期望(168, 10)
3. ⚠️ **误导性命名**：名称包含"multi_gpu"但实际单GPU
4. ⚠️ **use_multi_gpu参数**：存在但被忽略

---

## 🚀 运行建议

### 使用改进版本
```bash
# 运行改进的实验（推荐）
python3 run_improved_dual_year_experiment.py

# 或后台运行
nohup python3 run_improved_dual_year_experiment.py > improved_exp.log 2>&1 &
```

### 如果要修复原版本
需要修改 `run_dual_year_experiment.py` 的 `DualYearDataset.__getitem__` 方法：
```python
# 将这行：
spatial_features = temporal_features.flatten()

# 改为：
spatial_features = temporal_features  # 不展平
```

---

## 📊 预期性能对比

| 指标 | run_dual_year_experiment.py | run_improved_dual_year_experiment.py |
|------|----------------------------|-------------------------------------|
| **准确率** | 可能较低（特征处理错误） | 预期更高 |
| **训练稳定性** | 可能不稳定 | 更稳定 |
| **DySAT效果** | ❌ 无法使用时间注意力 | ✅ 正确使用时间注意力 |
| **运行时间** | 2.5-3.5小时 | 2.5-3.5小时 |

---

## 总结

**关键区别**：
1. **最重要**：`run_improved_dual_year_experiment.py` 不展平空间特征，保留时间结构
2. 两者都支持类别权重和增强日志
3. 两者都使用单GPU（GNN限制）
4. 改进版有更好的代码组织和文档

**建议**：使用 `run_improved_dual_year_experiment.py` 进行实验！
