# Mobility Pattern Classification Project

## Overview

This project implements a deep spatiotemporal model for classifying 9 types of mobility change patterns in the Pearl River Delta (PRD) region using 2021 and 2024 OD flow data.

## Model Architecture

**Dual-Branch Structure:**
- **Temporal Branch**: LSTM (2 layers, 128 units) + Spatial Pyramid Pooling (1×1, 2×2, 4×4)
- **Spatial Branch**: DySAT (3 layers, 4 attention heads) for dynamic graph learning
- **Fusion**: Multi-head attention (4 heads) to combine temporal and spatial features
- **Output**: 9-class classification

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
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
├── train.py                     # Main training script
├── ablation_study.py           # Ablation experiments
├── requirements.txt            # Dependencies
├── data/                       # Data files (2021.csv, 2024.csv, labels, metadata)
├── src/
│   ├── preprocessing/          # Data loading and graph construction
│   ├── models/                 # Model architectures
│   ├── training/               # Training pipeline
│   ├── evaluation/             # Evaluation metrics
│   └── visualization/          # Visualization tools
├── outputs/                    # Training outputs
│   ├── models/                # Saved models
│   ├── logs/                  # TensorBoard logs
│   └── figures/               # Visualizations (JPG, 300 DPI)
└── checkpoints/               # Model checkpoints
```

## Key Features

### Data Processing
- Handles large CSV files (12+ GB) with chunked reading
- Z-score normalization of flow volumes
- First 7 days (168 hours) used for training
- Hybrid graph construction (spatial + flow-based)

### Model Components
- **LSTM-SPP**: Captures temporal dynamics with multi-scale pooling
- **DySAT**: Models spatial relationships with graph attention
- **Attention Fusion**: Adaptively combines temporal and spatial features

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
- `TRAIN_DAYS = 7` (168 hours)
- `NUM_CLASSES = 9`
- `BATCH_SIZE = 32`
- `LEARNING_RATE = 0.001`
- `NUM_EPOCHS = 100`
- `EARLY_STOPPING_PATIENCE = 15`

## Expected Outputs

After training:
1. **Models**: `checkpoints/best_model.pth`
2. **Logs**: `outputs/logs/` (TensorBoard)
3. **Reports**: `outputs/evaluation_reports/` (metrics, confusion matrix, F1 scores)
4. **Figures**: `outputs/figures/` (spatial maps, temporal patterns)
5. **Results**: `outputs/final_results.json`

## Memory Management

For large datasets:
- Use `sample_size` parameter in `prepare_data()` for testing
- Reduce `BATCH_SIZE` if OOM occurs
- Requires GPU with 8GB+ VRAM (CPU supported but slower)

## Citation

This implementation follows the task specification for mobility pattern classification using dual-branch spatiotemporal modeling with LSTM-SPP and DySAT networks.

## See Also

- `CLAUDE.md` - Comprehensive documentation for Claude Code
- `config.py` - All configurable parameters
- `src/` - Source code with detailed docstrings
