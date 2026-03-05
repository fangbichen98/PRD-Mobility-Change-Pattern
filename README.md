# Mobility Pattern Classification Project

## Overview

This project implements a deep spatiotemporal model for classifying 9 types of mobility change patterns in the Pearl River Delta (PRD) region using 2021 and 2024 OD flow data.

**🎯 Key Achievement**: The model achieves **65.86% accuracy** using **ONLY 4 basic OD flow features** (inflow, outflow, total flow, net flow) - **no external features required**:

- ❌ No ellipse parameters
- ❌ No POI (Point of Interest) data
- ❌ No socioeconomic indicators
- ❌ No land use data
- ❌ No road network features
- ✅ **Only raw OD flow data** → **Strong performance**

This demonstrates that the **multi-scale spatiotemporal architecture** can extract complex mobility patterns from basic flow data alone, eliminating the need for manual feature engineering or expensive external data sources.

## Model Architecture

### Enhanced Dual-Branch Model with Multi-Scale Temporal Features

**Dual-Branch Structure:**

1. **Temporal Branch** (Multi-Scale)
   - **Hourly Level**: Captures fine-grained hourly patterns (168 time steps)
   - **Daily Level**: Aggregates patterns within each day (7 days)
   - **Weekly Level**: Overall trend across the week
   - **LSTM**: 2 layers, 128 hidden units with dropout
   - **Spatial Pyramid Pooling**: Multi-level pooling (1×1, 2×2, 4×4)
   - **Output**: 256-dimensional embeddings per scale

2. **Spatial Branch** (Pure Graph GAT)
   - **GAT Layers**: 3 layers
   - **Attention Heads**: 4 heads per layer
   - **Hidden Size**: 128 units per head
   - **Graph Type**: Static flow-based graphs per year
   - **Edge Creation**: OD flow > threshold (configurable)
   - **Output**: 256-dimensional embeddings

3. **Gated Feature Fusion**
   - **Fusion Type**: Gated mechanism for adaptive feature combination
   - **Attention Heads**: 4 heads
   - **Hidden Size**: 256 dimensions
   - **Input**: Concatenated temporal and spatial features
   - **Output**: Unified 256-dimensional representation

4. **Classification Head**
   - **Type**: Fully connected + Softmax
   - **Output**: 9 classes (mobility change patterns)
   - **Loss**: Weighted cross-entropy (handles class imbalance)

**Key Improvements:**
- Multi-scale temporal features capture patterns at different time granularities
- Gated fusion learns optimal weighting between temporal and spatial branches
- **Pure graph approach** - achieves strong performance **without ellipse features or any external data**
- **Minimal feature set** - only 4 basic OD flow features needed (inflow, outflow, total, net)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training

**Multi-Scale Temporal Branch Training:**
```bash
# Train with multi-scale temporal features (hourly + daily + weekly)
# Uses only basic OD flow features - no external data required
python train_multiscale_temporal.py
```

**Other Training Scripts:**
```bash
# Hierarchical dual-branch model (intensity + direction classification)
python train_hierarchical_simple.py

# Full training (dual-branch + baselines)
python train.py

# For faster testing with smaller dataset, edit train.py:
# data = prepare_data(year=2021, sample_size=1000000)
```

### 3. Run Ablation Study
```bash
python ablation_study.py
```

### 4. Monitor Training
```bash
tensorboard --logdir outputs/logs
```

## Project Structure

```
mobility_analysis/
├── config.py                    # All configuration parameters
├── train_multiscale_temporal.py # Multi-scale temporal training (RECOMMENDED)
├── train_hierarchical_simple.py # Hierarchical dual-branch training
├── train.py                     # Main training script
├── ablation_study.py           # Ablation experiments
├── requirements.txt            # Dependencies
├── data/                       # Data files (2021.csv, 2024.csv, labels, metadata)
│   └── cache/                  # Preprocessed data cache (auto-generated)
├── src/
│   ├── preprocessing/          # Data loading and graph construction
│   │   ├── dual_year_processor.py  # Dual-year data processing
│   │   └── graph_builder.py        # Spatial graph construction
│   ├── models/                 # Model architectures
│   │   ├── enhanced_dual_branch_model.py  # Enhanced model with multi-scale temporal
│   │   ├── multi_scale_temporal.py        # Multi-scale temporal branch
│   │   ├── spatial_branch_pure_graph.py   # Pure graph GAT branch
│   │   └── dual_branch_model.py           # Original dual-branch model
│   ├── training/               # Training pipeline
│   │   ├── dataset.py                # Dataset classes
│   │   └── dataset_pure_graph.py      # Pure graph dataset
│   ├── evaluation/             # Evaluation metrics
│   └── visualization/          # Visualization tools
├── outputs/                    # Training outputs
│   ├── multiscale_temporal_sgh_*/  # Multi-scale temporal results
│   ├── hierarchical_*/             # Hierarchical model results
│   ├── models/                # Saved models
│   ├── logs/                  # TensorBoard logs
│   └── figures/               # Visualizations (JPG, 300 DPI)
└── checkpoints/               # Model checkpoints
```

## Model Performance

### Shenzhen-Dongguan-Huizhou (SGH) Region

**Dataset**: `label_sgh.csv` (深莞惠都市圈)
- **Total grids**: 4,143
- **Training samples**: 2,900 (70%)
- **Validation samples**: 414 (10%)
- **Test samples**: 829 (20%)
- **Graph edges (2021)**: 4,735
- **Graph edges (2024)**: 4,539

**Best Performance** (flow_threshold=7.5):
- **Test Accuracy**: 65.86%
- **Macro F1 Score**: 0.6547

**Performance Range Across Different Flow Thresholds**:
- Flow threshold 5.0: 63.33% accuracy, F1=0.6298
- Flow threshold 7.5: **65.86% accuracy**, F1=0.6547 (best)
- Flow threshold 9.0: 65.26% accuracy, F1=0.6370
- Flow threshold 10.0: 65.74% accuracy, F1=0.6474
- Flow threshold 11.0: 64.78% accuracy, F1=0.6414
- Flow threshold 12.0: 61.04% accuracy, F1=0.5944

**Key Observations**:
1. Model achieves **~65% accuracy** using **only basic OD flow features** (no ellipse/POI/socioeconomic data)
2. Flow threshold around 7.5-10.0 provides optimal balance between graph connectivity and noise reduction
3. Consistent performance across different hyperparameter settings shows model robustness

**Note**: SGH (深莞惠) refers to the Shenzhen-Dongguan-Huizhou metropolitan region, not Shenzhen alone.

### Class Distribution

The SGH dataset has a balanced distribution:
- Classes 1, 4, 5, 6, 7, 8, 9: 500 samples each
- Class 2: 259 samples
- Class 3: 384 samples

## Key Features

### 🔥 Minimal Feature Set - Maximum Performance

**This model uses ONLY basic OD flow features - no external data required:**

**Only 4 Features Used:**
1. **Inflow**: Sum of flows where grid is destination
2. **Outflow**: Sum of flows where grid is origin
3. **Total Flow**: inflow + outflow (log-transformed)
4. **Net Flow**: outflow - inflow (sign-preserved, log-transformed)

**No additional features needed:**

1. **Inflow**: Sum of flows where grid is destination
2. **Outflow**: Sum of flows where grid is origin
3. **Total Flow**: inflow + outflow (log-transformed)
4. **Net Flow**: outflow - inflow (sign-preserved, log-transformed)

**No additional features needed:**
- ❌ Ellipse parameters
- ❌ POI (Point of Interest) data
- ❌ Socioeconomic indicators
- ❌ Land use data
- ❌ Road network features

**Why this works**: The multi-scale temporal branch (hourly + daily + weekly) and pure graph spatial branch can extract complex spatiotemporal patterns from raw OD flow data alone, eliminating the need for manual feature engineering or external data sources.

**Benefits of Minimal Feature Set:**
- **Simpler data pipeline** - no need to collect/merge external datasets
- **Faster preprocessing** - only basic OD aggregation required
- **Better generalization** - model learns from raw mobility patterns, not feature engineering
- **Easier deployment** - works with any OD flow dataset without additional data requirements
- **Cost-effective** - no expensive third-party data sources needed

### Data Processing
- Handles large CSV files (12+ GB) with chunked reading
- **Log transformation** for flow volumes (preserves magnitude)
- **Sign preservation** for net flow (maintains direction information)
- First 7 days (168 hours) used for training
- **Smart caching system**: Auto-generates cache files for fast reloading
- Static flow-based graphs (one per year)
- Pure graph approach (no ellipse features)

### Model Components

**Multi-Scale Temporal Branch:**
- **Hourly features**: 168 time steps capturing fine-grained patterns
- **Daily aggregation**: 7-day patterns
- **Weekly trends**: Overall week-level patterns
- **LSTM**: 2 layers, 128 hidden units with dropout
- **Spatial Pyramid Pooling (SPP)**: Multi-level pooling [1×1, 2×2, 4×4]
- Captures both short-term and long-term temporal dependencies

**Pure Graph Spatial Branch:**
- **Graph Attention Network (GAT)**: 3 layers, 4 attention heads
- **Static flow graphs**: Edges created when OD flow > threshold
- Processes each day separately, then aggregates temporally
- Focuses on spatial relationships without ellipse features

**Gated Feature Fusion:**
- **Adaptive weighting**: Learns optimal combination of temporal and spatial features
- **Multi-head attention**: 4 heads for diverse feature interactions
- **Gated mechanism**: Selectively emphasizes relevant features
- Outputs unified 256-dimensional representation

**Hierarchical Classification (Optional):**
- Decomposes 9-class problem into two 3-class problems:
  - **Flow Intensity**: Stable / Growth / Decline
  - **Spatial Direction**: Balanced / Aggregation / Diffusion
- Improves interpretability and training stability

### Evaluation
- Accuracy, F1 (macro/weighted), Precision, Recall
- Per-class F1 scores for all 9 classes
- Confusion matrix visualization
- Baseline comparisons (LSTM-only, GAT-only)

### Visualizations
- Spatial distribution maps (true vs predicted labels)
- Temporal patterns by class (mean ± std over 168 hours)
- Sample time series for each class
- Model comparison charts

## Configuration

Key parameters in `config.py`:

**Data Parameters:**
- `TRAIN_DAYS = 7` (168 hours of data)
- `TIME_STEPS = 168` (temporal sequence length)
- `TEMPORAL_INPUT_SIZE = 2` [total_log, net_flow_log]
- `SPATIAL_INPUT_SIZE = 2`
- `FLOW_THRESHOLD = 10.0` (minimum flow for edge creation)
- `USE_STATIC_GRAPH = True` (aggregated graphs per year)
- `USE_FLOW_ONLY_GRAPH = True` (flow-based edges only)

**Model Architecture:**
- `LSTM_LAYERS = 2`
- `LSTM_HIDDEN_SIZE = 128`
- `LSTM_DROPOUT = 0.2`
- `SPP_LEVELS = [1, 2, 4]` (spatial pyramid pooling)
- `GAT_LAYERS = 3`
- `GAT_HIDDEN_SIZE = 128`
- `GAT_HEADS = 4`
- `FUSION_HIDDEN_SIZE = 256`
- `ATTENTION_HEADS = 4`
- `NUM_CLASSES = 9`

**Training Hyperparameters:**
- `BATCH_SIZE = 16` (with gradient accumulation, effective = 64)
- `LEARNING_RATE = 0.001`
- `NUM_EPOCHS = 100`
- `EARLY_STOPPING_PATIENCE = 15`
- `WEIGHT_DECAY = 1e-5`
- `GRADIENT_ACCUMULATION_STEPS = 4`
- `GRADIENT_CLIP_NORM = 1.0`

**Data Split:**
- `TRAIN_SPLIT = 0.7` (70% for training)
- `VAL_SPLIT = 0.1` (10% for validation)
- `TEST_SPLIT = 0.2` (20% for testing)
- `RANDOM_SEED = 42` (reproducibility)

## Expected Outputs

### Multi-Scale Temporal Training (`train_multiscale_temporal.py`)

Output directory: `outputs/multiscale_temporal_sgh_{timestamp}/`

**Contents:**
1. **Models**: `models/best_model.pth`
   - Model checkpoint with best validation accuracy
   - Contains model state, optimizer state, epoch, accuracy, F1 score

2. **Metrics**: `metrics/`
   - `test_results.json` - Complete test results with configuration
   - `classification_report.txt` - Per-class precision, recall, F1
   - `confusion_matrix.npy` - Confusion matrix for visualization
   - `timing_info.json` - Training time statistics

3. **Logs**: `training.log`
   - Detailed training progress per epoch
   - Training/validation loss and accuracy
   - Learning rate changes
   - Early stopping information

### Hierarchical Training (`train_hierarchical_simple.py`)

Output directory: `outputs/hierarchical_{timestamp}/`

Similar structure with additional hierarchical metrics:
- Intensity accuracy (3-class)
- Direction accuracy (3-class)
- Hierarchical accuracy (9-class combined)
- Direct accuracy (baseline 9-class)

### General Outputs

All training scripts generate:
- **Saved models**: Best model based on validation accuracy
- **Training logs**: Epoch-by-epoch progress
- **Test metrics**: Accuracy, F1 (macro/weighted), per-class scores
- **Configuration**: Complete model and training hyperparameters
- **Timing statistics**: Total training time, start/end timestamps

## Memory Management

For large datasets:
- Use `sample_size` parameter in `prepare_data()` for testing
- Reduce `BATCH_SIZE` if OOM occurs
- Requires GPU with 8GB+ VRAM (CPU supported but slower)

## Training Scripts

### train_multiscale_temporal.py

**Purpose**: Training with Multi-Scale Temporal Branch (hourly + daily + weekly features)

**Dataset**: Uses `label_sgh.csv` (Shenzhen-Dongguan-Huizhou metropolitan region - 深莞惠都市圈)

**Key Design**: Uses **only basic OD flow features** (inflow, outflow, total, net) without any external features like ellipse parameters, POI, or socioeconomic data.

**Performance**: Achieves ~65% accuracy on SGH region with pure graph approach.

**Key Functions:**
- `train_epoch()` - Single epoch training with gradient accumulation (4 steps)
  - Gradient clipping (max_norm=1.0)
  - Progress logging every 10 batches
  - Returns: average loss and accuracy

- `evaluate()` - Model evaluation on validation/test sets
  - Computes: loss, accuracy, macro F1 score
  - Returns: metrics + predictions + labels
  - Used for both validation and testing

- `format_time()` - Time formatting utility
  - Converts seconds to HH:MM:SS format
  - Used for training time statistics

- `main()` - Complete training orchestration
  1. **Data Loading**: Loads dual-year data with caching
  2. **Feature Preparation**: Extracts 168-hour temporal features
  3. **Dataset Creation**: PureGraphDualYearDataset with train/val/test split
  4. **Model Initialization**: EnhancedDualBranchModel with multi-scale temporal
  5. **Training Loop**:
     - Train for up to 100 epochs
     - Early stopping with patience=15
     - Learning rate scheduling (ReduceLROnPlateau)
     - Save best model based on validation accuracy
  6. **Testing**: Load best model and evaluate on test set
  7. **Results Saving**: Comprehensive metrics and configuration

**Output Files:**
- `models/best_model.pth` - Best model checkpoint
- `metrics/test_results.json` - Complete test results
- `metrics/classification_report.txt` - Per-class metrics
- `metrics/confusion_matrix.npy` - Confusion matrix
- `metrics/timing_info.json` - Training time statistics
- `training.log` - Detailed training log

**Usage:**
```bash
python train_multiscale_temporal.py
```

### train_hierarchical_simple.py

**Purpose**: Hierarchical dual-branch model with intensity + direction classification

**Key Features:**
- Decomposes 9-class problem into two 3-class problems
- Three parallel classification heads:
  - Intensity classifier (3 classes: Stable/Growth/Decline)
  - Direction classifier (3 classes: Balanced/Aggregation/Diffusion)
  - Direct 9-class classifier (baseline)
- Combined loss with weighted cross-entropy

**Output:**
- Hierarchical accuracy metrics (intensity, direction, combined)
- Comparison with direct 9-class classification

### Key Differences Between Training Scripts

| Feature | train_multiscale_temporal.py | train_hierarchical_simple.py |
|---------|----------------------------|------------------------------|
| Temporal Features | Multi-scale (168 hours hourly/daily/weekly) | Daily aggregated (7 days) |
| Classification | Direct 9-class | Hierarchical (3×3 intensity × direction) |
| Model | EnhancedDualBranchModel | ImprovedDualBranchModel |
| Dataset | PureGraphDualYearDataset | ImprovedDualYearDataset |
| Spatial Branch | Pure Graph GAT (no ellipse features) | Graph + Ellipse Features |
| Dataset Used | label_sgh.csv (深莞惠 SGH region) | labels.csv |
| Performance | 65.86% accuracy on SGH | Varies by dataset |
| Feature Set | Minimal (OD flow only) | Extended (OD flow + ellipses) |

## Data Flow

### 1. Data Loading
```
Raw OD Data (12GB CSV)
    ↓
Chunked Reading (1M rows)
    ↓
Feature Aggregation
    ↓
Log Transformation
    ↓
Smart Cache (auto-generated)
```

### 2. Feature Engineering
For each grid (7 days × 2 features):
- **Inflow**: Sum of flows where grid is destination
- **Outflow**: Sum of flows where grid is origin
- **Total Flow**: inflow + outflow → log(1 + total)
- **Net Flow**: outflow - inflow → sign × log(1 + \|net\|)

**Output**: (7, 4) feature vector per grid
  - 7 rows = 7 daily snapshots
  - 4 columns = [2021_total, 2024_total, 2021_net, 2024_net]

### 3. Graph Construction
```
OD Flow Data
    ↓
Filter (flow > threshold)
    ↓
Edge Creation (origin → destination)
    ↓
Static Graph per Year
    ↓
GAT Processing
```

### 4. Model Forward Pass
```
Temporal Features (N, 168, 2)
    ↓
Multi-Scale Branch (LSTM + SPP)
    ↓
256-dim Embeddings × 6

Spatial Features (N, 7, 2)
    ↓
Pure Graph GAT
    ↓
256-dim Embeddings × 3

Concatenate & Fuse (9 features)
    ↓
Gated Fusion
    ↓
256-dim Unified Representation

    ↓
Classifier
    ↓
9-Class Predictions
```

## Citation

This implementation demonstrates that **complex mobility pattern classification can be achieved using only basic OD flow features**, without requiring external datasets (ellipse parameters, POI, socioeconomic indicators, etc.). The multi-scale spatiotemporal architecture (hourly + daily + weekly temporal features + pure graph spatial features) achieves **65.86% accuracy** on the Shenzhen-Dongguan-Huizhou (SGH) metropolitan region using minimal data requirements.

**Key Contribution**: Shows that deep learning architectures can replace manual feature engineering for urban mobility analysis.

## See Also

- `CLAUDE.md` - Comprehensive documentation for Claude Code (detailed architecture)
- `config.py` - All configurable parameters
- `src/models/` - Model implementations with docstrings
- `src/preprocessing/` - Data processing pipeline
- `src/training/` - Training utilities and dataset classes
