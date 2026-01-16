# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a deep learning project for mobility pattern classification in the Pearl River Delta (PRD) region of China. It implements a dual-branch spatiotemporal model combining LSTM-SPP (temporal) and DySAT (spatial) networks with attention-based fusion to classify 9 types of mobility change patterns.

## Project Structure

```
mobility_analysis/
├── config.py                    # Configuration parameters
├── train.py                     # Main training script
├── ablation_study.py           # Ablation experiments
├── requirements.txt            # Python dependencies
├── data/                       # Data directory
│   ├── 2021.csv               # OD flow data 2021 (~12.7 GB)
│   ├── 2024.csv               # OD flow data 2024 (~12.9 GB)
│   ├── labels_1w_20251228.csv # Grid labels (9 classes)
│   └── grid_metadata/
│       └── PRD_grid_metadata.csv
├── src/
│   ├── preprocessing/
│   │   ├── data_processor.py  # Data loading and preprocessing
│   │   └── graph_builder.py   # Spatial graph construction
│   ├── models/
│   │   ├── temporal_branch.py # LSTM + SPP network
│   │   ├── spatial_branch.py  # DySAT network
│   │   └── dual_branch_model.py # Complete model + baselines
│   ├── training/
│   │   ├── dataset.py         # PyTorch dataset
│   │   └── trainer.py         # Training pipeline
│   ├── evaluation/
│   │   └── evaluator.py       # Evaluation metrics
│   └── visualization/
│       └── visualizer.py      # Spatial/temporal visualization
├── outputs/                    # Training outputs
│   ├── models/                # Saved models
│   ├── logs/                  # TensorBoard logs
│   └── figures/               # Generated visualizations
└── checkpoints/               # Model checkpoints
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
mkdir -p outputs/models outputs/logs outputs/figures checkpoints
```

### Training

```bash
# Full training with all experiments (dual-branch + baselines)
python train.py

# For testing with smaller dataset (faster)
# Edit train.py line: data = prepare_data(year=2021, sample_size=1000000)
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
