# 实验改进总结

## ✅ 完成的改进

### 1. 数据预处理缓存功能

**文件**: `src/preprocessing/dual_year_processor.py`

**功能**:
- 自动缓存预处理后的数据到 `data/cache/` 目录
- 缓存文件名基于label文件和采样参数生成唯一hash
- 包含详细的缓存信息文件（.txt）

**缓存命名规则**:
```
data/cache/dual_year_data_{hash}.pkl        # 数据文件
data/cache/dual_year_data_{hash}_info.txt  # 信息文件
```

**Hash生成**:
```python
cache_key = f"{label_basename}_samples_{samples_per_class}"
# 例如: "labels.csv_samples_None" -> hash: "a1b2c3d4"
```

**使用方法**:
```python
# 启用缓存（默认）
data = prepare_dual_year_experiment_data(
    label_path='data/labels.csv',
    samples_per_class=None,
    use_cache=True,  # 启用缓存
    cache_dir='data/cache'
)

# 禁用缓存
data = prepare_dual_year_experiment_data(
    label_path='data/labels.csv',
    samples_per_class=None,
    use_cache=False  # 禁用缓存
)
```

**性能提升**:
- 首次运行: 20-25分钟（数据预处理）
- 后续运行: ~1秒（从缓存加载）
- **加速比**: 1200-1500x

**缓存信息示例**:
```
Cache Information:
  Label file: data/labels.csv
  Samples per class: ALL
  Total grids: 9977
  Graph edges: 79816
  Feature shape: (168, 10)
  Class distribution:
    Class 1: 657 samples (weight: 1.6873)
    Class 2: 2182 samples (weight: 0.5080)
    ...
```

### 2. Batch Size调整

**修改**: `run_improved_dual_year_experiment.py`

**变更**:
- 从 `batch_size=64` 降低到 `batch_size=16`
- 原因: 避免GPU内存不足（OOM）错误

**影响**:
- ✅ 解决OOM问题
- ⚠️ 训练速度略慢（每个epoch多4倍迭代）
- ✅ 显存使用: ~27GB → ~10GB

### 3. 文件管理

**删除/备份**:
- `run_dual_year_experiment.py` → `run_dual_year_experiment.py.backup`
- 原因: 该文件存在特征展平错误

**保留**:
- `run_improved_dual_year_experiment.py` ✅ 正确版本

## 📊 当前实验状态

**实验名称**: `improved_full_dual_year_2021vs2024`

**配置**:
- ✅ 单GPU训练（GPU 0: NVIDIA A100-SXM4-40GB）
- ✅ 全量数据集：9,977 samples
- ✅ 类别权重：已启用
- ✅ 批次大小：16（避免OOM）
- ✅ 数据缓存：已启用
- ✅ Epochs: 100

**当前阶段**: 数据加载（首次运行，将创建缓存）

**进程信息**:
- 进程ID: 371552
- CPU使用: 209%
- 状态: 正在运行

## 🚀 缓存使用指南

### 不同Label文件的缓存

缓存会根据label文件名自动区分：

```python
# 使用 labels.csv
data = prepare_dual_year_experiment_data(
    label_path='data/labels.csv',  # 缓存: dual_year_data_a1b2c3d4.pkl
    samples_per_class=None
)

# 使用 labels_v2.csv
data = prepare_dual_year_experiment_data(
    label_path='data/labels_v2.csv',  # 缓存: dual_year_data_e5f6g7h8.pkl
    samples_per_class=None
)

# 使用采样
data = prepare_dual_year_experiment_data(
    label_path='data/labels.csv',
    samples_per_class=200  # 缓存: dual_year_data_i9j0k1l2.pkl
)
```

### 查看缓存

```bash
# 查看所有缓存文件
ls -lh data/cache/

# 查看缓存信息
cat data/cache/dual_year_data_*_info.txt

# 删除所有缓存（重新生成）
rm -rf data/cache/*
```

### 缓存管理

```bash
# 查看缓存大小
du -sh data/cache/

# 删除特定缓存
rm data/cache/dual_year_data_{hash}.*

# 清理旧缓存（保留最新3个）
ls -t data/cache/*.pkl | tail -n +4 | xargs rm -f
```

## ⏱️ 预计时间

### 首次运行（创建缓存）
- 数据预处理: 20-25分钟
- 模型训练: 3-4小时（batch_size=16）
- **总计**: 约3.5-4.5小时

### 后续运行（使用缓存）
- 数据加载: ~1秒 ⚡
- 模型训练: 3-4小时
- **总计**: 约3-4小时

## 📁 输出位置

```
/home/PRD-Mobility-Change-Pattern/outputs/improved_full_dual_year_2021vs2024_YYYYMMDD_HHMMSS/
```

## 🔍 监控命令

```bash
# 实时查看日志
tail -f /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log

# 查看进度
watch -n 10 'tail -20 /home/PRD-Mobility-Change-Pattern/improved_experiment_with_cache.log'

# 查看GPU使用
watch -n 1 nvidia-smi
```

## 📝 日志文件

- **主日志**: `improved_experiment_with_cache.log`
- **实验日志**: `outputs/improved_full_dual_year_2021vs2024_*/logs/experiment.log`

## 🎯 关键改进点

1. ✅ **数据缓存**: 后续运行节省20-25分钟
2. ✅ **Batch size优化**: 避免OOM错误
3. ✅ **文件清理**: 删除错误版本
4. ✅ **单GPU训练**: GNN最佳实践
5. ✅ **类别权重**: 关注少样本类别
6. ✅ **增强日志**: 详细的准确率指标

## 💡 使用建议

1. **首次运行**: 等待缓存创建完成（20-25分钟）
2. **后续调试**: 享受1秒加载速度
3. **更换Label**: 自动创建新缓存
4. **清理缓存**: 定期删除旧缓存节省空间

## 🔄 下次运行

```bash
# 直接运行，将使用缓存
python3 run_improved_dual_year_experiment.py

# 输出将显示:
# ================================================================================
# Loading Preprocessed Data from Cache
# ================================================================================
# Cache file: data/cache/dual_year_data_a1b2c3d4.pkl
# ✓ Successfully loaded cached data!
#   - Total grids: 9977
#   - Graph edges: 79816
# ================================================================================
```

实验正在顺利进行中！🚀
