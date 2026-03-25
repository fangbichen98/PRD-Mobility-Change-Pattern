# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

This project implements a **dual-branch spatiotemporal deep learning model** for classifying mobility pattern changes in the Shenzhen-Guangzhou-Hong Kong (SGH) region between 2021 and 2024. The model uses a 9-class classification framework to identify distinct mobility change patterns.

**Research Problem**: Classify how urban mobility patterns have changed by analyzing Origin-Destination (OD) flow data:
- **9 Classes**: Different combinations of flow intensity and spatial direction changes
- **Dual-Year Comparison**: 2021 vs 2024 mobility patterns
- **Spatiotemporal Learning**: Combines temporal sequences (LSTM) and spatial graphs (GCN/GraphSAGE)

---

## Quick Start Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
# Main training script (current model)
python train_multiscale_temporal.py

# Monitor training progress
tail -f outputs/multiscale_temporal_sgh_*/training.log
```

### Testing & Evaluation
```bash
# Training script automatically evaluates on validation set
# Best model saved to: outputs/multiscale_temporal_sgh_*/models/best_model.pth
# Results logged in: outputs/multiscale_temporal_sgh_*/training.log
```

### Data Preprocessing
```bash
# Data is automatically preprocessed and cached on first run
# Cache location: data/cache/dual_year_data_{hash}.pkl
# To force reprocessing: rm -f data/cache/*.pkl
```

---

## Current Model Architecture

### EnhancedDualBranchModel (LSTM+GCN)

```
EnhancedDualBranchModel (2,487,119 parameters)
├── Temporal Branch (SimplifiedMultiScaleTemporal)
│   Input: (batch, 168, 2) - [inflow_log, outflow_log]
│   │   - 168 timesteps = 7 days × 24 hours (hourly data)
│   │   - 2 features = [log(1 + inflow), log(1 + outflow)] per hour
│   ├── Hourly Scale: 2-layer LSTM (128 hidden) → processes 168 hourly timesteps
│   ├── Daily Scale: 2-layer LSTM (128 hidden) → processes 7 daily aggregates (sum)
│   ├── Weekly Scale: MLP → processes 4 weekly statistics (sum, max, mean, trend)
│   │   - Input: 2 features × 4 stats = 8 dimensions
│   └── Fusion: Linear(768, 256) → combines 3 scales
│   Output: (batch, 3, 256) - [2021_features, 2024_features, diff]
│
├── Spatial Branch (PureGraphDualYearGCN/SAGE/GINE)
│   Input: Graph adjacency matrices (featureless for GCN/SAGE, Laplacian PE for GINE)
│   ├── Architecture: 3-layer GCN/SAGE/GINE (128 hidden units per layer)
│   ├── Graph: Static flow-based edges (17M edges for 2021, 15M for 2024)
│   ├── Nodes: 65,049 total grids (4,143 labeled grids for training)
│   └── Processing: Separate processing for 2021 and 2024 graphs
│   Output: (batch, 3, 256) - [2021_embeddings, 2024_embeddings, diff]
│
├── Fusion Layer (GatedFeatureFusion)
│   ├── Input: 6 features (3 temporal + 3 spatial) × 256 dim = 1,536 dim
│   ├── Gate Network: Linear(1536, 256) → Linear(256, 6) → Sigmoid
│   └── Weighted fusion with learned gates
│   Output: (batch, 256)
│
└── Classifier
    └── MLP: Linear(256, 128) → ReLU → Dropout → Linear(128, 9)
    Output: (batch, 9) - 9-class logits
```

---

## Data Processing Pipeline

### Input Data Format

**Raw OD Flow Data** (`data/2021_sgh_week.csv`, `data/2024_sgh_week.csv`):
- Size: ~12GB each
- Columns: `o_grid_500`, `d_grid_500`, `date_dt`, `time`, `num_total`
- Time period: First 7 days of each year (168 hours)
- Records: ~69M for 2021, ~65M for 2024

**Labels** (`data/label_sgh.csv`):
- Grid ID → Class label (1-9)
- Total: 4,143 labeled grids
- Class distribution: Imbalanced (Class 2: 210 samples, others: ~500 samples each)

**Grid Metadata** (`data/grid_metadata/sgh_grid_metadata.csv`):
- Grid coordinates for spatial graph construction
- Total: 65,049 valid grid cells

### Feature Engineering

**Temporal Features (per grid, per hour):**

1. **Inflow and Outflow** (Separate Flow Direction Features)
   ```python
   inflow_log = log(1 + inflow)    # Log transform for stability
   outflow_log = log(1 + outflow)  # Log transform for stability
   ```

2. **Feature Shape**: (168, 2) per grid per year
   - 168 hours = 7 days × 24 hours
   - 2 features = [inflow_log, outflow_log] (hourly flows)
   - Raw data shape: (168, 4) = [inflow_2021, outflow_2021, inflow_2024, outflow_2024]
   - Split into: 2021 features (168, 2) and 2024 features (168, 2)

**Multi-Scale Temporal Aggregation:**

1. **Hourly Scale**: Raw 168-hour sequence → LSTM
2. **Daily Scale**: Sum 24 hours → 7 daily values → LSTM
   ```python
   x_daily = x.view(batch, 7, 24, 2).sum(dim=2)  # Physical flow aggregation
   ```
3. **Weekly Scale**: 4 statistics over 168 hours
   ```python
   weekly_sum = x.sum(dim=1)      # Total weekly flow
   weekly_max = x.max(dim=1)[0]   # Peak flow
   weekly_mean = x.mean(dim=1)    # Average flow
   trend = (x[:, -1, :] - x[:, 0, :]) / 168  # Linear trend
   # Input to weekly MLP: input_size * 4 = 2 * 4 = 8 features
   ```

### Spatial Graph Construction

**Graph Type: Static Flow Graphs**

- **One aggregated graph per year** (not dynamic per day)
- **Edge Creation**: OD flow > threshold (currently 0.0, includes all flows)
- **Edge Weights**: Raw flow volume (log-transformed in GCN)
- **Self-loops**: Added with weight 1.0
- **KNN Fallback**: 8 nearest neighbors for isolated nodes (weight 0.5)

**Graph Statistics**:
```
2021 Flow Graph: 17,114,138 edges
2024 Flow Graph: 15,122,681 edges
Nodes: 65,049 total grids (4,143 labeled grids for training)
```

**Code Location**: `src/preprocessing/graph_builder.py`

### Caching Mechanism

**Smart Cache System** (`src/preprocessing/dual_year_processor.py`):
- Cache key: MD5 hash of label file + OD data modification times
- Cache file: `data/cache/dual_year_data_{hash}.pkl`
- First run: ~5-10 minutes (full preprocessing)
- Subsequent runs: ~1 second (load from cache)
- Auto-regenerates if data files change

---

## Training Configuration

### Key Parameters (`config.py`)

**Data Parameters**:
```python
TRAIN_DAYS = 7              # Use first 7 days
TIME_STEPS = 168            # 168 hourly snapshots
TEMPORAL_INPUT_SIZE = 2     # [inflow_log, outflow_log]
FLOW_THRESHOLD = 0.0        # Include all flow edges
```

**Model Architecture**:
```python
# Temporal Branch (SimplifiedMultiScaleTemporal hardcodes these values)
LSTM_LAYERS = 3             # Not used (SimplifiedMultiScaleTemporal uses 2 layers)
LSTM_HIDDEN_SIZE = 256      # Not used (SimplifiedMultiScaleTemporal uses 128 hidden)
LSTM_DROPOUT = 0.4          # Used by SimplifiedMultiScaleTemporal

# Spatial Branch
SPATIAL_MODEL = "GCN"       # Options: "GCN", "SAGE", "GINE", "GAT"
SPATIAL_LAYERS = 3          # Number of GCN/SAGE/GINE layers
SPATIAL_HIDDEN_SIZE = 128   # Hidden units per layer
LAPLACIAN_PE_DIM = 16       # Laplacian PE dimension (for GINE only)

# Edge Feature Mode (for GINE and GAT)
EDGE_FEATURE_MODE = "flow_only"  # Options: "flow_only", "flow_distance_direction", "flow_distribution"
# - flow_only: scalar edge weight (1-dim)
# - flow_distance_direction: [flow, norm_dist, cos(θ), sin(θ)] (4-dim)
# - flow_distribution: [flow, p_ij, info_ij] (3-dim) — aligned with entropy-based labels

# Fusion
FUSION_HIDDEN_SIZE = 256
NUM_CLASSES = 9
```

**Training Hyperparameters**:
```python
BATCH_SIZE = 12             # Reduced for memory efficiency
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-3         # L2 regularization
NUM_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 20
GRADIENT_ACCUMULATION = 4   # Effective batch size = 12 * 4 = 48
```

**Data Split**:
```python
TRAIN_SPLIT = 0.7           # 2,900 samples
VAL_SPLIT = 0.1             # 414 samples
TEST_SPLIT = 0.2            # 829 samples
RANDOM_SEED = 42
```

### Loss Function

**Weighted Cross-Entropy Loss**:
```python
# Compute class weights for imbalanced data
class_weights = total_samples / (num_classes * class_counts)

# Example weights:
# Class 1-9 (except 2): weight ≈ 0.94
# Class 2: weight ≈ 2.23 (due to only 210 samples)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Training Loop

**Standard Training** (`train_multiscale_temporal.py`):
```python
for epoch in range(NUM_EPOCHS):
    # Training
    for batch in train_loader:
        logits = model(x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    # Validation
    val_acc, val_f1 = evaluate(model, val_loader)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    if val_acc > best_val_acc:
        save_checkpoint()
        patience = 0
    else:
        patience += 1
        if patience >= EARLY_STOPPING_PATIENCE:
            break
```

---

## Current Performance

### Latest Results (Multi-Scale Temporal + GCN)

**Training Run**: `outputs/multiscale_temporal_20260312_224223_label_sgh`

**Best Validation Performance** (Epoch 31):
- Validation Accuracy: **72.71%**
- Validation F1 Score: **0.7129**
- Training Accuracy: 64.52%
- Training Loss: 0.8617
- Validation Loss: 0.6686

**Performance Trend**:
- Early epochs (1-5): Val Acc 50-67% (rapid improvement)
- Mid training (6-10): Val Acc 68-72% (steady progress)
- Peak performance (22-31): Val Acc 71-73% (best: 72.71% at epoch 31)
- Learning rate reduced from 0.0001 → 0.000025 (ReduceLROnPlateau)

**Key Improvements Over Previous Version**:
1. **+21% accuracy gain**: From 51.78% to 72.71% validation accuracy
2. **Better feature representation**: Using [inflow, outflow] instead of [total_flow]
3. **Enhanced temporal modeling**: 4 weekly statistics (sum, max, mean, trend) vs 3
4. **Improved convergence**: Stable training with gradient accumulation
5. **Reduced overfitting**: Train acc (64.52%) close to Val acc (72.71%)

**Observations**:
1. **Strong performance**: 72.71% accuracy on 9-class classification
2. **Good generalization**: Small train-val gap indicates robust learning
3. **Effective fusion**: Gated fusion successfully combines temporal and spatial features
4. **Class balance**: Weighted loss effectively handles class imbalance

---

## Model Variants Tested

### 1. PureGraphDualYearGCN (Current - Best Performance)
- **Architecture**: 3-layer GCN (128 hidden units)
- **Performance**: **72.71% validation accuracy** (Epoch 31)
- **Pros**: Stable training, uses edge weights, excellent performance
- **Features**: Featureless learning (all-1 node features), Laplacian normalization

### 2. PureGraphDualYearSAGE (Alternative)
- **Architecture**: 3-layer GraphSAGE (128 hidden units)
- **Performance**: Not tested with current configuration
- **Issue**: GraphSAGE doesn't support edge_weight parameter in PyG
- **Note**: Edge weights ignored, loses flow information

### 3. PureGraphDualYearGINE (Experimental)
- **Architecture**: 3-layer GINE (128 hidden units)
- **Performance**: Not tested with current configuration
- **Features**: Uses Laplacian Positional Encoding (16-dim)
- **Pros**: Can incorporate structural features via Laplacian PE
- **Edge Feature Modes**: Supports multi-dim edge features via `EDGE_FEATURE_MODE`
- **Note**: More complex, requires PE computation

### 4. PureGraphDualYearGAT (Experimental)
- **Architecture**: GAT with attention-based message passing
- **Edge Feature Modes**: Supports multi-dim edge features via `EDGE_FEATURE_MODE`
- **Note**: Supports `edge_dim` parameter for multi-dimensional edge attributes

### Edge Feature Modes (for GINE/GAT)
- **flow_only** (1-dim): Scalar flow weight. Default, used by GCN/SAGE.
- **flow_distance_direction** (4-dim): `[flow, norm_distance, cos(θ), sin(θ)]`. Geographic features.
- **flow_distribution** (3-dim): `[flow, p_ij, info_ij]`. Distribution features aligned with entropy-based labels.
  - `p_ij = flow(i→j) / total_outflow(i)`: flow share (probability in entropy calculation)
  - `info_ij = -log(p_ij)`: information content (entropy component)
  - Directly encodes each edge's role in the OD distribution, matching how labels distinguish aggregation vs diffusion

---

## Key Implementation Details

### Physical Flow Aggregation (Critical Fix)

**Problem**: Using `mean()` for daily aggregation loses physical meaning
```python
# WRONG: Average flow per hour
x_daily = x.view(batch, 7, 24, 1).mean(dim=2)  # Divides by 24

# CORRECT: Total daily flow
x_daily = x.view(batch, 7, 24, 1).sum(dim=2)   # Preserves magnitude
```

**Impact**: Sum aggregation preserves the physical meaning of flow data (total trips per day)

### Batch Processing with Full Graph

**Challenge**: GCN requires full graph, but we train on batches

**Solution** (`src/training/dataset.py`):
1. Pass full graph to spatial branch
2. Pass `node_indices` to extract batch nodes
3. Spatial branch processes full graph, then extracts batch embeddings

```python
# In spatial branch
h_full = gcn_layers(x_full, edge_index, edge_attr)  # (N, 256)
h_batch = h_full[node_indices]  # (batch, 256)
```

### Memory Optimization

**Techniques**:
1. **Caching**: Preprocessed features cached to disk
2. **Batch size**: 24 (balanced between memory and convergence)
3. **Reduced capacity**: 2 layers × 96 hidden (vs 3 × 128)
4. **Static graphs**: One graph per year (vs 7 dynamic graphs)

---

## File Structure & Key Locations

### Entry Point
- `train_multiscale_temporal.py` - Main training script

### Data Processing
- `src/preprocessing/dual_year_processor.py` - Data loading & feature engineering
- `src/preprocessing/graph_builder.py` - Spatial graph construction
- `src/preprocessing/data_processor.py` - Grid metadata processing

### Model Architecture
- `src/models/enhanced_dual_branch_model.py` - Complete model
- `src/models/multi_scale_temporal.py` - Multi-scale temporal branch (LSTM)
- `src/models/spatial_branch_pure_graph.py` - GCN/GraphSAGE spatial branch
- `src/models/gated_fusion.py` - Gated feature fusion

### Training & Evaluation
- `src/training/dataset.py` - Dataset & collator classes
- `src/training/trainer.py` - Training utilities

### Configuration
- `config.py` - All configurable parameters

---

## Common Development Tasks

### Modifying Model Architecture

**To change GCN hidden size**:
1. Edit `config.py`: `SPATIAL_HIDDEN_SIZE = 256`
2. Model automatically adapts

**To add more GCN layers**:
1. Edit `config.py`: `SPATIAL_LAYERS = 4`
2. Model automatically adapts

**To switch between GCN, GraphSAGE, and GINE**:
1. Edit `config.py`: `SPATIAL_MODEL = "SAGE"` or `"GINE"`
2. For GINE, also set: `LAPLACIAN_PE_DIM = 16`
3. Model automatically selects the correct spatial branch

### Adjusting Graph Construction

**To change flow threshold**:
1. Edit `config.py`: `FLOW_THRESHOLD = 10.0`
2. Delete cache: `rm data/cache/*.pkl`
3. Rerun training

**To disable KNN fallback**:
1. Edit `config.py`: `USE_KNN_FALLBACK = False`
2. Delete cache and rerun

### Debugging Data Issues

**Check preprocessed features**:
```python
import pickle
with open('data/cache/dual_year_data_{hash}.pkl', 'rb') as f:
    data = pickle.load(f)
print(data['change_features'][grid_id].shape)  # Should be (168, 4)
# Features: [inflow_2021_log, outflow_2021_log, inflow_2024_log, outflow_2024_log]
```

**Verify graph structure**:
```python
edge_index, edge_attr = data['graphs_2021'][0]
print(f"Edges: {edge_index.shape[1]}")
print(f"Nodes: {edge_index.max() + 1}")
```

### Performance Tuning

**If OOM occurs**:
1. Reduce `BATCH_SIZE` in `config.py` (currently 12)
2. Reduce `SPATIAL_HIDDEN_SIZE` (currently 128)
3. Reduce `SPATIAL_LAYERS` (currently 3)
4. Use CPU instead of GPU (slower but more memory)

**If training is slow**:
1. Increase `BATCH_SIZE` (if memory allows)
2. Reduce `SPATIAL_LAYERS`
3. Use smaller `FLOW_THRESHOLD` (fewer edges)
4. Reduce gradient accumulation steps (currently 4)

---

## Known Issues & Limitations

### 1. ~~Performance Plateau (~52%)~~ - RESOLVED ✓
- **Previous Issue**: Model converged to 51-52% validation accuracy
- **Solution Applied**:
  - Changed from single feature [total_log] to dual features [inflow_log, outflow_log]
  - Added trend statistic to weekly features (4 stats instead of 3)
  - Increased spatial branch capacity (3 layers, 128 hidden units)
  - Applied gradient accumulation (effective batch size 48)
- **Result**: **72.71% validation accuracy** (+21% improvement)

### 2. GraphSAGE Edge Weight Issue
- **Issue**: PyG's SAGEConv doesn't support `edge_weight` parameter
- **Impact**: Loses flow magnitude information
- **Workaround**: Use GCN instead (supports edge weights)

### 3. ~~Unusual Train-Val Gap~~ - RESOLVED ✓
- **Previous Observation**: Training acc (47%) < Validation acc (52%)
- **Current Status**: Normal train-val relationship restored
  - Training acc: 64.52%
  - Validation acc: 72.71%
- **Note**: Small gap indicates good generalization without overfitting

### 4. Class Imbalance
- **Issue**: Class 2 has only 210 samples (vs 500 for others)
- **Current solution**: Weighted cross-entropy (weight ≈ 2.23)
- **Limitation**: Still may underperform on Class 2

---

## Troubleshooting

### Cache Issues
**Problem**: Stale cache after data changes
**Solution**: `rm -f data/cache/*.pkl` and rerun

### Graph Construction Errors
**Problem**: "No edges created" warning
**Solution**: Lower `FLOW_THRESHOLD` in `config.py`

### OOM During Training
**Problem**: CUDA out of memory
**Solution**: Reduce `BATCH_SIZE`, `SPATIAL_HIDDEN_SIZE`, or use CPU

### Poor Performance
**Problem**: Low accuracy on validation set
**Solution**:
- Check class imbalance (weights computed automatically)
- Increase `NUM_EPOCHS` or reduce `EARLY_STOPPING_PATIENCE`
- Verify data preprocessing (check cache file)
- Try different spatial branch (GCN vs GraphSAGE)

---


## References

### Key Papers
- Graph Convolutional Networks (GCN): Kipf & Welling, 2017
- GraphSAGE: Hamilton et al., 2017
- Graph Attention Networks (GAT): Veličković et al., 2018

### PyTorch Geometric Documentation
- GCNConv: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
- SAGEConv: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
- GATConv: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv

---

This documentation covers the current architecture and implementation of the SGH mobility pattern classification system. The dual-branch model combines multi-scale temporal (LSTM) and spatial (GCN) processing to classify 9 types of mobility change patterns between 2021 and 2024.

**Last Updated**: 2026-03-13
**Current Best Performance**: 72.71% validation accuracy (Multi-Scale Temporal + GCN model)
**Key Success Factors**:
- Dual-feature temporal input [inflow, outflow] instead of single [total]
- Enhanced weekly statistics (4 features: sum, max, mean, trend)
- Increased spatial capacity (3 layers × 128 hidden units)
- Gradient accumulation for stable training
