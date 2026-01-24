# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a deep learning project for mobility pattern classification in the Pearl River Delta (PRD) region of China. It implements a dual-branch spatiotemporal model combining LSTM-SPP (temporal) and DySAT (spatial) networks with attention-based fusion to classify 9 types of mobility change patterns.
研究构建了人群流动模式变化检测（Human Mobility Pattern Change Detection, HMP-CD）的概念与方法框架。该框架实现了从原始移动数据到模式变化类型识别的端到端检测
【概念框架图】HMP-CD框架包含：
（1）变化检测定义
【变化检测定义层】该层构建了人群流动模式变化的定义。对于两个时期（T1和T2）的移动数据，分别提取流动连接强度（Flow Intensity）和空间方向分布（Spatial Distribution）两个维度，形成各时期的"时空稳态快照"作为两种基线状态（Baseline State），作为人群流动模式的定义。在此基础上，基于流动强度趋势（增长/稳定/衰减）与空间组织方向（聚集/均衡/扩散）的组合，构建3×3模式变化分类体系（Change Type），将模式变化系统性地划分为A-I共九种类型。
（2）时空深度学习方法
【时空深度学习方法层】该层实现了从时空数据到模式变化类型的智能识别。方法包含两个并行的分支：时间序列分支：采用LSTM网络（LSTM-net）和空间金字塔池化网络（SPP-net）分别处理T1和T2时期的时序输入，学习每个空间单元在时间维度上的模式演化趋势，捕捉长期依赖关系并识别关键转变节点。动态图分支：基于T1和T2时期进行动态图构建，采用动态自注意力网络（DySAT-net）建模空间单元间的依赖关系与传播机制，识别变化的空间分异规律与扩散路径。两个通道提取的时空特征经过特征融合（Feature Fusion）后，输入流动模式变化检测模块（Mobility Pattern Change Detection），端到端输出九类模式变化类型。

## Project Structure

```
mobility_analysis/
├── config.py                    # Configuration parameters
├── train.py                     # Main training script (original)
├── train_improved.py            # Improved training script (dual-year)
├── preprocess_data.py           # Standalone preprocessing script
├── test_cache_key.py            # Cache key generation test script
├── ablation_study.py            # Ablation experiments
├── requirements.txt             # Python dependencies
├── data/                        # Data directory
│   ├── 2021.csv                # OD flow data 2021 (~12.7 GB)
│   ├── 2024.csv                # OD flow data 2024 (~12.9 GB)
│   ├── labels_1w.csv           # Grid labels (9 classes, 10K samples)
│   ├── cache/                  # Preprocessed data cache
│   │   ├── dual_year_data_{hash}.pkl      # Cached preprocessed data
│   │   └── dual_year_data_{hash}_info.txt # Cache metadata
│   └── grid_metadata/
│       └── PRD_grid_metadata.csv
├── src/
│   ├── preprocessing/
│   │   ├── data_processor.py       # Data loading and preprocessing
│   │   ├── dual_year_processor.py  # Dual-year data processor (with cache v3)
│   │   └── graph_builder.py        # Spatial graph construction
│   ├── models/
│   │   ├── temporal_branch.py      # LSTM + SPP network
│   │   ├── spatial_branch.py       # DySAT network
│   │   └── dual_branch_model.py    # Complete model + baselines
│   ├── training/
│   │   ├── dataset.py              # PyTorch dataset
│   │   └── trainer.py              # Training pipeline
│   ├── evaluation/
│   │   └── evaluator.py            # Evaluation metrics
│   └── visualization/
│       └── visualizer.py           # Spatial/temporal visualization
├── outputs/                         # Training outputs
│   ├── models/                     # Saved models
│   ├── logs/                       # TensorBoard logs
│   └── figures/                    # Generated visualizations
├── checkpoints/                     # Model checkpoints
├── CACHE_IMPROVEMENT_SUMMARY.md     # Cache system documentation
├── CACHE_BEHAVIOR_EXPLANATION.md    # Cache behavior guide
└── CLAUDE.md                        # This file
```

## Data Structure

### Mobility Data Files
- `data/2021.csv` - Mobility data from 2021 (~12.7 GB)
- `data/2024.csv` - Mobility data from 2024 (~12.9 GB)

**Schema:**
- `date_dt` - Date in YYYYMMDD format
- `time` - Hour of day (0-23)
- `o_grid_500` - Origin grid ID (500m grid cell)
- `d_grid_500` - Destination grid ID (500m grid cell)
- `num_total` - Total number of trips between origin and destination

### Grid Metadata
- `data/grid_metadata/PRD_grid_metadata.csv` - Spatial reference for grid cells (~13 MB)

**Schema:**
- `OBJECTID` - Unique object identifier
- `area_name` - District/area name (Chinese)
- `city_name` - City name (Chinese)
- `grid_id` - Grid cell identifier
- `lon` - Longitude (WGS84)
- `lat` - Latitude (WGS84)

### Labels
- `data/labels_1w_20251228.csv` - Labeled grid cells with 9-class classifications

**Schema:**
- `grid_id` - Grid cell identifier
- `lon` - Longitude
- `lat` - Latitude
- `label` - Classification label (1-9, converted to 0-8 internally)
- `remark` - Additional notes (optional)

## Model Architecture

### Dual-Branch Structure

The model uses two parallel branches that process different aspects of mobility data:

1. **Temporal Branch (LSTM-SPP)**
   - 2-layer bidirectional LSTM (128 hidden units each)
   - Spatial Pyramid Pooling with 3 levels (1×1, 2×2, 4×4)
   - Processes time series of inflow/outflow (168 hours × 2 features)
   - Captures temporal dynamics and multi-scale patterns

2. **Spatial Branch (DySAT)**
   - 3-layer Dynamic Self-Attention Network
   - Graph Attention with 4 heads per layer
   - K-nearest neighbor graph (k=8) + flow-based edges
   - Captures spatial relationships and geographic dependencies

3. **Attention Fusion**
   - Multi-head attention (4 heads) to combine temporal and spatial features
   - Learns adaptive weighting between branches
   - 256-dimensional fused representation

4. **Classification Head**
   - 2-layer MLP with ReLU activation
   - Outputs 9-class predictions

### Baseline Models

- **LSTM**: Temporal-only baseline (2-layer LSTM)
- **GAT**: Spatial-only baseline (2-layer Graph Attention Network)

## Commands

### Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories (automatically created by scripts)
mkdir -p outputs/models outputs/logs outputs/figures checkpoints data/cache
```

### Data Preprocessing (Recommended Workflow)

```bash
# Preprocess data with caching (recommended for dual-year experiments)
python3 preprocess_data.py --label-path data/labels_1w.csv

# With custom cache directory
python3 preprocess_data.py --label-path data/labels_1w.csv --cache-dir data/cache

# Force regenerate cache (bypass existing cache)
python3 preprocess_data.py --label-path data/labels_1w.csv --force-regenerate

# With sampling (for testing)
python3 preprocess_data.py --label-path data/labels_1w.csv --samples-per-class 100

# Test cache key generation
python3 test_cache_key.py
```

### Training

```bash
# Dual-year training (improved model, uses preprocessed cache)
python3 train_improved.py

# Original single-year training
python3 train.py

# For testing with smaller dataset (faster)
# Edit train.py line: data = prepare_data(year=2021, sample_size=1000000)
```

### Cache Management

```bash
# View cache files
ls -lht data/cache/

# View cache information
cat data/cache/dual_year_data_*_info.txt

# Clean old caches (keep only latest)
ls -t data/cache/dual_year_data_*.pkl | tail -n +2 | xargs rm -f
ls -t data/cache/dual_year_data_*_info.txt | tail -n +2 | xargs rm -f

# Clean caches older than 7 days
find data/cache/ -name "dual_year_data_*.pkl" -mtime +7 -delete

# Remove all caches
rm data/cache/dual_year_data_*.*
```

### Ablation Study

```bash
# Run ablation experiments (without SPP, without DySAT)
python ablation_study.py
```

### Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir outputs/logs

# View at http://localhost:6006
```

### Testing Individual Components

```bash
# Test data preprocessing
python src/preprocessing/data_processor.py

# Test dual-year preprocessing
python src/preprocessing/dual_year_processor.py

# Test temporal branch
python src/models/temporal_branch.py

# Test spatial branch
python src/models/spatial_branch.py

# Test complete model
python src/models/dual_branch_model.py
```

## Key Configuration Parameters

All parameters are defined in `config.py`:

### Data Parameters
- `TRAIN_DAYS = 7` - Use first 7 days (168 hours) for training
- `NUM_CLASSES = 9` - 9 mobility pattern classes
- `LABEL_RANGE = (1, 9)` - Label values in data

### Model Architecture
- `LSTM_LAYERS = 2`, `LSTM_HIDDEN_SIZE = 128`
- `SPP_LEVELS = [1, 2, 4]` - Pyramid pooling levels
- `DYSAT_LAYERS = 3`, `DYSAT_HIDDEN_SIZE = 128`, `DYSAT_HEADS = 4`
- `TIME_WINDOW = 24` - 24-hour sliding window for dynamic graphs

### Training
- `BATCH_SIZE = 32`
- `LEARNING_RATE = 0.001`
- `NUM_EPOCHS = 100`
- `EARLY_STOPPING_PATIENCE = 15`

## Data Processing Pipeline

### 1. Data Loading
- OD flow data loaded in chunks (100K rows) due to large size
- Date/time validation and conversion
- Grid ID consistency checking with metadata

### 2. Preprocessing
- Filter to first 7 days (168 hours) for training
- Z-score normalization of flow volumes
- Temporal feature engineering (hour, day_of_week, is_weekend)

### 3. Feature Aggregation
- Aggregate inflow/outflow for each grid cell over time
- Create temporal sequences: (168 hours, 2 features)
- Create spatial features: flattened temporal data (336 features)

### 4. Graph Construction
- **Spatial graph**: K-nearest neighbors (k=8) based on geographic distance
- **Flow graph**: Edges based on OD flow volume (threshold-based)
- **Hybrid graph**: Weighted combination of spatial + flow graphs

### 5. Dataset Creation
- Train/Val/Test split: 70%/10%/20%
- Custom collator for batching with graph structure
- Random seed (42) for reproducibility

### 6. Cache Management (v3)

**Improved Cache Key Generation (2026-01-19):**

The preprocessing pipeline uses an intelligent caching system that automatically detects data changes:

**Cache Key Components:**
- Label file **content hash** (MD5): Detects any changes to label file content
- OD data file **modification times**: Detects updates to 2021.csv and 2024.csv
- `samples_per_class` parameter: Different sampling creates different caches
- Version number (v3): Ensures compatibility with code updates

**Cache Invalidation (Automatic):**
```bash
# Cache automatically regenerates when:
- Label file content changes (any modification to labels)
- OD data files are updated (2021.csv or 2024.csv)
- samples_per_class parameter changes
- Code version upgrades
```

**Cache Behavior:**
- **Same parameters**: Overwrites existing cache (same filename)
- **Different parameters**: Creates new cache (different filename, old cache preserved)
- **Location**: `data/cache/dual_year_data_{hash}.pkl`
- **Size**: ~100-200 MB per cache file

**Cache File Naming:**
```
dual_year_data_ff8dc58099d9.pkl       # Cache file (12-char hash)
dual_year_data_ff8dc58099d9_info.txt  # Cache metadata
```

**Cache Info File Contents:**
```
Cache Information:
  Label file: data/labels_1w.csv
  Label file hash: 6e999bb1
  Data 2021 mtime: 1749545823
  Data 2024 mtime: 1749545691
  Samples per class: ALL
  Total grids: 10000
  Class distribution: [...]
```

**Manual Cache Management:**
```bash
# View cache files
ls -lht data/cache/

# View cache information
cat data/cache/dual_year_data_*_info.txt

# Clean old caches (manual)
rm data/cache/dual_year_data_444c24a7.*

# Clean all caches (use with caution)
rm data/cache/dual_year_data_*.*

# Clean caches older than 7 days
find data/cache/ -name "dual_year_data_*.pkl" -mtime +7 -delete
```

**Typical Workflow:**
```bash
# 1. Modify labels
vim data/labels_1w.csv

# 2. Run preprocessing (detects change, regenerates cache)
python3 preprocess_data.py --label-path data/labels_1w.csv
# Output: Cache file: data/cache/dual_year_data_ff8dc58099d9.pkl

# 3. Train model (uses new cache)
python3 train_improved.py

# 4. Clean old caches (optional)
ls -lht data/cache/  # Check which caches exist
rm data/cache/dual_year_data_444c24a7.*  # Remove old cache
```

**Cache Validation:**
- Automatically validates cache integrity on load
- Checks for required keys: labels, change_features, graphs_2021, graphs_2024, class_weights
- Regenerates cache if validation fails

**Performance:**
- Label file hash calculation: < 0.1 seconds
- OD data mtime reading: < 0.01 seconds
- Total overhead: Negligible

**Important Notes:**
- Old cache files are NOT automatically deleted
- Multiple caches can coexist (different parameters)
- Cache directory: `data/cache/` (configurable via `--cache-dir`)
- Use `--force-regenerate` to bypass cache and regenerate

## Training Pipeline

### Training Loop
1. Forward pass through dual branches
2. Attention-based fusion
3. Classification with cross-entropy loss
4. Gradient clipping (max_norm=1.0)
5. Learning rate scheduling (ReduceLROnPlateau)

### Logging
- TensorBoard metrics: loss, accuracy, F1 (overall + per-class)
- Model checkpoints: best model + periodic saves (every 10 epochs)
- Confusion matrix saved at best validation accuracy

### Early Stopping
- Monitors validation accuracy
- Patience: 15 epochs without improvement

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score (Macro)**: Unweighted average across 9 classes
- **F1 Score (Weighted)**: Weighted by class support

### Per-Class Metrics
- F1 score for each of the 9 classes
- Precision and recall (macro-averaged)

### Outputs
- Confusion matrix (9×9)
- Classification report with per-class metrics
- Model comparison plots (dual-branch vs baselines)

## Visualization Outputs

All visualizations saved as JPG (300 DPI) in `outputs/figures/`:

### Spatial Visualizations
- `true_label_distribution.jpg` - Geographic distribution of true labels
- `predicted_label_distribution.jpg` - Geographic distribution of predictions
- Scatter plots with lon/lat coordinates, color-coded by class

### Temporal Visualizations
- `class_temporal_patterns.jpg` - Average temporal patterns for each class (3×3 grid)
- `temporal_series_class_X.jpg` - Sample time series for each class
- Shows inflow/outflow patterns over 168 hours with mean ± std

### Evaluation Visualizations
- `model_comparison.jpg` - Bar charts comparing all models
- `{model}_confusion_matrix.jpg` - Confusion matrices
- `{model}_f1_scores.jpg` - Per-class F1 score bar charts

## Working with Large Data

### Memory Management
- OD files are 12+ GB each - always use chunked reading
- Default chunk size: 100,000 rows
- Filter data early (date range, valid grid IDs) to reduce memory
- Use `sample_size` parameter in `prepare_data()` for testing

### GPU Requirements
- Model fits on GPUs with 8GB+ VRAM
- Batch size can be reduced if OOM occurs
- CPU training is supported but much slower

### Data Sampling for Development
```python
# In train.py, use sample_size for faster iteration
data = prepare_data(year=2021, sample_size=1000000)  # 1M rows
```

## Common Issues and Solutions

### Issue: Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Use smaller `sample_size` in data loading
- Reduce `LSTM_HIDDEN_SIZE` or `DYSAT_HIDDEN_SIZE`

### Issue: Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce data size with `sample_size` parameter
- Use fewer epochs or early stopping

### Issue: Poor Performance
- Check label distribution (may be imbalanced)
- Verify data preprocessing (normalization, time filtering)
- Try different learning rates or batch sizes
- Run ablation study to identify weak components

### Issue: Cache Not Updating After Label Changes
**Symptom**: Modified labels but model still uses old data

**Solution**: The improved cache system (v3) automatically detects label changes. If you're still seeing old data:
```bash
# 1. Verify you're using the new cache system
python3 test_cache_key.py  # Should show label hash

# 2. Force regenerate cache
python3 preprocess_data.py --label-path data/labels_1w.csv --force-regenerate

# 3. Clean old caches
rm data/cache/dual_year_data_*.pkl
```

### Issue: Too Many Cache Files
**Symptom**: `data/cache/` directory has many old cache files

**Solution**: Cache files accumulate when parameters change. Clean periodically:
```bash
# View all caches with timestamps
ls -lht data/cache/

# Keep only the latest cache
ls -t data/cache/dual_year_data_*.pkl | tail -n +2 | xargs rm -f
ls -t data/cache/dual_year_data_*_info.txt | tail -n +2 | xargs rm -f

# Or clean caches older than 7 days
find data/cache/ -name "dual_year_data_*.pkl" -mtime +7 -delete
find data/cache/ -name "dual_year_data_*_info.txt" -mtime +7 -delete
```

### Issue: Disk Space Running Low
**Symptom**: Each cache file is ~100-200 MB

**Solution**:
```bash
# Check cache directory size
du -sh data/cache/

# Remove all caches (will regenerate when needed)
rm data/cache/dual_year_data_*.*

# Or keep only the most recent cache
ls -t data/cache/dual_year_data_*.pkl | tail -n +2 | xargs rm -f
```

### Issue: Which Cache Is Being Used?
**Symptom**: Multiple cache files exist, unsure which is active

**Solution**:
```bash
# Method 1: Run preprocessing to see cache file name
python3 preprocess_data.py --label-path data/labels_1w.csv
# Output shows: Cache file: data/cache/dual_year_data_ff8dc58099d9.pkl

# Method 2: Use test script
python3 test_cache_key.py
# Shows: Cache hash: ff8dc58099d9

# Method 3: Check cache info files
cat data/cache/dual_year_data_*_info.txt
```

## Coordinate System
All coordinates are in WGS84 (EPSG:4326) decimal degrees.

## Expected Outputs

After running `train.py`, you should have:

1. **Trained Models**
   - `checkpoints/best_model.pth` - Best model checkpoint
   - `checkpoints/confusion_matrix.npy` - Confusion matrix

2. **Training Logs**
   - `outputs/logs/run_YYYYMMDD_HHMMSS/` - TensorBoard logs

3. **Evaluation Reports**
   - `outputs/evaluation_reports/{model}_metrics.txt` - Text metrics
   - `outputs/evaluation_reports/{model}_confusion_matrix.jpg`
   - `outputs/evaluation_reports/{model}_f1_scores.jpg`

4. **Visualizations**
   - `outputs/figures/` - All spatial and temporal visualizations

5. **Results Summary**
   - `outputs/final_results.json` - JSON with all model metrics
