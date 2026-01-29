# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

This project implements a **hierarchical dual-branch spatiotemporal deep learning model** for classifying mobility pattern changes in the Pearl River Delta (PRD) region between 2021 and 2024. The model uses a 3×3 classification framework to identify 9 distinct mobility change patterns.

**Research Problem**: Classify how urban mobility patterns have changed by analyzing Origin-Destination (OD) flow data across two dimensions:
- **Flow Intensity**: Stable / Growth / Decline (3 classes)
- **Spatial Direction**: Balanced / Aggregation / Diffusion (3 classes)
- **Combined**: 9 classes (3 × 3 hierarchical structure)

---

## Quick Start Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
# Main training script (hierarchical model)
python train_hierarchical_simple.py

# Monitor training with TensorBoard
tensorboard --logdir outputs/logs
```

### Testing & Evaluation
```bash
# The training script automatically evaluates on test set after training
# Results saved to: outputs/test_results_hierarchical.json
```

### Data Preprocessing
```bash
# Data is automatically preprocessed and cached on first run
# Cache location: data/cache/dual_year_data_{hash}.pkl
# To force reprocessing, delete the cache file
```

---

## Architecture Overview

### Model Structure

```
ImprovedDualBranchModel
├── Temporal Branch (ParallelTemporalBranch)
│   ├── LSTM Branch: Captures long-term temporal dependencies
│   │   └── 2-layer LSTM (128 hidden units) → 256-dim embeddings
│   └── SPP Branch: Multi-scale pattern extraction
│       └── Spatial Pyramid Pooling [1×1, 2×2, 4×4] → 256-dim embeddings
│   Output: 6 features (2021, 2024, diff) × 2 branches
│
├── Spatial Branch (SimplifiedDualYearGAT)
│   └── 3-layer GAT (4 attention heads per layer)
│       ├── Processes each of 7 days separately
│       ├── Temporal aggregation (average across days)
│       └── Output: 3 features (2021, 2024, diff)
│
├── Fusion Layer (MultiFeatureAttentionFusion)
│   └── Multi-head self-attention (4 heads) over 9 features
│       └── Output: 256-dim fused representation
│
└── Classification Heads (3 parallel heads)
    ├── Intensity Classifier: 3 classes (Stable/Growth/Decline)
    ├── Direction Classifier: 3 classes (Balanced/Aggregation/Diffusion)
    └── Direct Classifier: 9 classes (for comparison)
```

---

## Data Processing Pipeline

### Input Data Format

**Raw OD Flow Data** (`data/2021.csv`, `data/2024.csv`):
- Size: ~12GB each
- Columns: `o_grid_500`, `d_grid_500`, `date_dt`, `time`, `num_total`
- Time period: First 7 days of each year (168 hours)

**Labels** (`data/labels.csv`):
- Grid ID → Class label (1-9)
- ~1,000+ labeled grids

**Grid Metadata** (`data/PRD_grid_metadata.csv`):
- Grid coordinates for spatial graph construction

### Feature Engineering

**For each grid and each day, the system computes:**

1. **Inflow & Outflow Aggregation**
   - Inflow: Sum of flows where grid is destination
   - Outflow: Sum of flows where grid is origin
   - Aggregated daily (not hourly)

2. **Total Flow** (Flow Intensity Indicator)
   ```python
   total = inflow + outflow
   total_log = log(1 + total)  # Log transform preserves magnitude
   ```

3. **Net Flow** (Spatial Direction Indicator)
   ```python
   net_flow = outflow - inflow
   net_flow_log = sign(net_flow) × log(1 + |net_flow|)  # Preserves sign
   ```

4. **Final Feature Vector per Grid**: Shape (7, 4)
   ```
   [2021_total_log, 2024_total_log, 2021_net_flow_log, 2024_net_flow_log]
   ```
   - 7 rows = 7 daily snapshots
   - 4 columns = 2 years × 2 features

### Caching Mechanism

**Smart Cache System** (`src/preprocessing/dual_year_processor.py`):
- Cache key: MD5 hash of label file + OD data modification times
- Cache file: `data/cache/dual_year_data_{hash}.pkl`
- First run: ~5-10 minutes (full preprocessing)
- Subsequent runs: ~1 second (load from cache)
- Auto-regenerates if data files change

---

## Spatial Graph Construction

### Graph Type: Static Flow Graphs

**Current Implementation**:
- **One aggregated graph per year** (not dynamic per day)
- Edges created when OD flow > threshold
- Edge weights: Raw flow volume
- Threshold: `FLOW_THRESHOLD = 10.0` (configurable in `config.py`)

**Graph Statistics**:
```
2021 Flow Graph: ~10,956 edges
2024 Flow Graph: ~10,956 edges
Nodes: ~1,500 grids
```

**Code Location**: `src/preprocessing/graph_builder.py`

---

## Data Flow Through Model

### Step-by-Step Processing

1. **Data Loading** (`train_hierarchical_simple.py:301-346`)
   ```python
   data = prepare_dual_year_experiment_data(
       label_path='data/labels.csv',
       use_cache=True
   )
   # Returns: change_features, labels, graphs_2021, graphs_2024, grid_id_to_idx
   ```

2. **Dataset Creation** (`src/training/dataset.py`)
   ```python
   dataset = ImprovedDualYearDataset(
       change_features=data['change_features'],  # {grid_id: (7, 4)}
       labels=data['labels']                     # {grid_id: 0-8}
   )
   # Each sample: x_2021 (7, 2), x_2024 (7, 2), label
   ```

3. **Batch Collation** (`src/training/dataset.py`)
   ```python
   collator = ImprovedGraphBatchCollator(
       graphs_2021=data['graphs_2021'],
       graphs_2024=data['graphs_2024'],
       grid_id_to_idx=data['grid_id_to_idx'],
       all_features_2021=all_features_2021,  # (N, 7, 4)
       all_features_2024=all_features_2024
   )
   # Returns batch with node_indices for graph extraction
   ```

4. **Model Forward Pass** (`src/models/dual_branch_model.py:343-535`)
   ```python
   intensity_logits, direction_logits, direct_logits = model(
       x_2021_full,      # (N, 7, 2) - full graph features
       x_2024_full,      # (N, 7, 2)
       graphs_2021,      # [(edge_index, edge_attr)]
       graphs_2024,      # [(edge_index, edge_attr)]
       node_indices      # (batch,) - indices to extract
   )
   ```

### Temporal Branch Processing

**Input**: `x_2021` (batch, 7, 2), `x_2024` (batch, 7, 2)

**LSTM Component** (`src/models/temporal_branch.py:308-372`):
```python
# Shared LSTM processes both years
_, (h_2021, _) = lstm(x_2021)  # (batch, 128)
_, (h_2024, _) = lstm(x_2024)  # (batch, 128)
diff_lstm = h_2024 - h_2021    # Captures change

# Project to 256-dim
h_2021 = projection(h_2021)    # (batch, 256)
h_2024 = projection(h_2024)    # (batch, 256)
diff_lstm = projection(diff_lstm)
```

**SPP Component**:
```python
# Spatial Pyramid Pooling with levels [1, 2, 4]
# Input: (batch, 2, 7) - features × time
# Output: (batch, 14) - concatenated pooled features
spp_out = spp(x)
h_spp = projection(spp_out)  # (batch, 256)
```

**Output**: Stack 6 features → (batch, 6, 256)

### Spatial Branch Processing

**Input**: `x_2021` (N, 7, 2), `x_2024` (N, 7, 2), static graphs

**GAT Processing** (`src/models/spatial_branch.py:444-589`):
```python
for t in range(7):  # Process each day
    x_t = x[:, t, :]  # (N, 2)

    # Apply 3 GAT layers
    for gat_layer in gat_layers:
        h = gat_layer(h, edge_index, edge_attr)
        h = elu(h) + dropout(h)
    # h: (N, 512)

    daily_embeddings.append(h)

# Temporal aggregation
h_aggregated = mean(daily_embeddings, dim=0)  # (N, 512)
h_out = output_proj(h_aggregated)  # (N, 256)

# Extract batch nodes
h_batch = h_out[node_indices]  # (batch, 256)
```

**Output**: 3 features (2021, 2024, diff) → (batch, 3, 256)

### Fusion & Classification

**Fusion** (`src/models/dual_branch_model.py`):
```python
# Concatenate 9 features (6 temporal + 3 spatial)
all_features = concat([temporal_features, spatial_features])  # (batch, 9, 256)

# Multi-head self-attention
fused = fusion_layer(all_features)  # (batch, 256)
```

**Classification Heads**:
```python
intensity_logits = intensity_classifier(fused)  # (batch, 3)
direction_logits = direction_classifier(fused)  # (batch, 3)
direct_logits = direct_classifier(fused)        # (batch, 9)
```

---

## Hierarchical Label System

### Label Conversion

**From 9-class to hierarchical** (`train_hierarchical_simple.py:34-61`):
```python
# Original label: 0-8 (representing classes 1-9)
intensity_label = label // 3  # 0, 1, 2 (Stable, Growth, Decline)
direction_label = label % 3   # 0, 1, 2 (Balanced, Aggregation, Diffusion)
```

**Example**:
- Label 5 (Class 6) → intensity=1 (Growth), direction=2 (Diffusion)
- Label 0 (Class 1) → intensity=0 (Stable), direction=0 (Balanced)

**Combining predictions**:
```python
hierarchical_pred = intensity_pred * 3 + direction_pred
```

---

## Training Configuration

### Key Parameters (`config.py`)

**Data Parameters**:
```python
TRAIN_DAYS = 7              # Use first 7 days
TIME_STEPS = 7              # 7 daily snapshots
TEMPORAL_INPUT_SIZE = 2     # [total_log, net_flow_log]
SPATIAL_INPUT_SIZE = 2
```

**Model Architecture**:
```python
LSTM_LAYERS = 2
LSTM_HIDDEN_SIZE = 128
LSTM_DROPOUT = 0.2
SPP_LEVELS = [1, 2, 4]

GAT_LAYERS = 3
GAT_HIDDEN_SIZE = 128
GAT_HEADS = 4
FUSION_HIDDEN_SIZE = 256
ATTENTION_HEADS = 4
```

**Graph Configuration**:
```python
USE_STATIC_GRAPH = True      # Static aggregated graphs (not dynamic)
USE_FLOW_ONLY_GRAPH = True   # Flow-based edges only
FLOW_THRESHOLD = 10.0        # Minimum flow for edge creation
```

**Training Hyperparameters**:
```python
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-5
```

**Data Split**:
```python
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
RANDOM_SEED = 42
```

### Loss Function

**Three weighted cross-entropy losses** (`train_hierarchical_simple.py:438-473`):
```python
# Compute class weights for imbalanced data
class_weights = total_samples / (num_classes * class_counts)

# Three loss functions
criterion_intensity = CrossEntropyLoss(weight=intensity_weights)
criterion_direction = CrossEntropyLoss(weight=direction_weights)
criterion_direct = CrossEntropyLoss(weight=original_weights)

# Combined loss
loss = (loss_intensity + loss_direction + 0.5 * loss_direct) / 4
```

### Training Loop

**Gradient Accumulation** (`train_hierarchical_simple.py:64-180`):
```python
for batch_idx, batch in enumerate(train_loader):
    # Forward pass
    intensity_logits, direction_logits, direct_logits = model(...)

    # Compute losses
    loss_intensity = criterion_intensity(intensity_logits, intensity_labels)
    loss_direction = criterion_direction(direction_logits, direction_labels)
    loss_direct = criterion_direct(direct_logits, labels)

    # Combined loss
    loss = (loss_intensity + loss_direction + 0.5 * loss_direct) / 4

    # Gradient accumulation (effective batch size = 16 × 4 = 64)
    loss.backward()
    if (batch_idx + 1) % 4 == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## Evaluation Metrics

### Four Types of Accuracy

1. **Intensity Accuracy**: Correct flow intensity predictions (3 classes)
2. **Direction Accuracy**: Correct spatial direction predictions (3 classes)
3. **Hierarchical Accuracy**: Combined intensity × direction (9 classes)
4. **Direct Accuracy**: Direct 9-class predictions (baseline)

### F1 Scores

```python
hierarchical_f1 = f1_score(labels, hierarchical_pred, average='macro')
direct_f1 = f1_score(labels, direct_pred, average='macro')
```

### Output Format

**Training Progress**:
```
Epoch 10/100
  Train Loss: 1.2345
    - Intensity: 85.23%
    - Direction: 78.45%
    - Hierarchical (3×3): 72.34%
    - Direct (9-class): 68.91%
  Val Accuracy:
    - Intensity: 82.10%
    - Direction: 75.32%
    - Hierarchical (3×3): 70.15% | F1: 0.6823
    - Direct (9-class): 66.78% | F1: 0.6421
```

**Test Results** (`outputs/test_results_hierarchical.json`):
```json
{
  "intensity_accuracy": 0.8210,
  "direction_accuracy": 0.7532,
  "hierarchical_accuracy": 0.7015,
  "hierarchical_f1": 0.6823,
  "direct_accuracy": 0.6678,
  "direct_f1": 0.6421,
  "confusion_matrix": [[...]]
}
```

---

## Key Implementation Details

### Static vs Dynamic Graphs

**Current Implementation: Static Graphs**
- One aggregated graph per year (all 7 days combined)
- Stored as: `graphs_2021 = [(edge_index, edge_attr)]`
- Memory efficient, simpler computation

**Alternative: Dynamic Graphs** (Not currently used)
- One graph per day (7 graphs per year)
- Would be: `graphs_2021 = [(edge_index_t, edge_attr_t) for t in range(7)]`
- More expressive but higher memory cost

### Batch Processing with Full Graph

**Challenge**: GAT requires full graph, but we train on batches

**Solution** (`src/training/dataset.py`):
1. Pass full graph features to spatial branch: `(N, 7, 2)`
2. Pass `node_indices` to extract batch nodes: `(batch,)`
3. Spatial branch processes full graph, then extracts batch embeddings

```python
# In spatial branch
h_full = gat_layers(x_full, edge_index, edge_attr)  # (N, 256)
h_batch = h_full[node_indices]  # (batch, 256)
```

### Memory Optimization

**Gradient Accumulation**:
- Effective batch size: 16 × 4 = 64
- Reduces memory usage while maintaining large batch benefits

**Chunked Data Loading**:
- OD data loaded in chunks (1M rows at a time)
- Prevents OOM with 12GB+ CSV files

**Caching**:
- Preprocessed features cached to disk
- Avoids reprocessing on every run

---

## File Structure & Key Locations

### Entry Point
- `train_hierarchical_simple.py` - Main training script

### Data Processing
- `src/preprocessing/dual_year_processor.py` - Data loading & feature engineering
- `src/preprocessing/graph_builder.py` - Spatial graph construction
- `src/preprocessing/data_processor.py` - Grid metadata processing

### Model Architecture
- `src/models/dual_branch_model.py` - Complete model (lines 343-535)
- `src/models/temporal_branch.py` - LSTM + SPP (lines 308-372)
- `src/models/spatial_branch.py` - GAT (lines 444-589)

### Training & Evaluation
- `src/training/dataset.py` - Dataset & collator classes
- `src/training/trainer.py` - Training utilities
- `src/evaluation/evaluator.py` - Evaluation metrics

### Configuration
- `config.py` - All configurable parameters

---

## Common Development Tasks

### Modifying Model Architecture

**To change LSTM hidden size**:
1. Edit `config.py`: `LSTM_HIDDEN_SIZE = 256`
2. Model automatically adapts

**To add more GAT layers**:
1. Edit `config.py`: `GAT_LAYERS = 4`
2. Model automatically adapts

**To change attention heads**:
1. Edit `config.py`: `GAT_HEADS = 8` or `ATTENTION_HEADS = 8`

### Adjusting Graph Construction

**To change flow threshold**:
1. Edit `config.py`: `FLOW_THRESHOLD = 20.0`
2. Delete cache: `rm data/cache/*.pkl`
3. Rerun training

**To use dynamic graphs** (not recommended due to memory):
1. Edit `config.py`: `USE_STATIC_GRAPH = False`
2. Modify `graph_builder.py` to return list of graphs per day

### Debugging Data Issues

**Check preprocessed features**:
```python
import pickle
with open('data/cache/dual_year_data_{hash}.pkl', 'rb') as f:
    data = pickle.load(f)
print(data['change_features'][grid_id])  # (7, 4)
```

**Verify graph structure**:
```python
edge_index, edge_attr = data['graphs_2021'][0]
print(f"Edges: {edge_index.shape[1]}")
print(f"Nodes: {edge_index.max() + 1}")
```

### Performance Tuning

**If OOM occurs**:
1. Reduce `BATCH_SIZE` in `config.py`
2. Reduce `GAT_HIDDEN_SIZE` or `LSTM_HIDDEN_SIZE`
3. Use CPU instead of GPU (slower but more memory)

**If training is slow**:
1. Increase `BATCH_SIZE` (if memory allows)
2. Reduce `GAT_LAYERS` or `LSTM_LAYERS`
3. Use smaller `FLOW_THRESHOLD` (fewer edges)

---

## Important Notes

### Label Indexing
- CSV labels: 1-9 (user-facing)
- Internal labels: 0-8 (zero-indexed for PyTorch)
- Conversion happens in `dual_year_processor.py`

### Graph Storage
- Graphs stored in model: `model.graphs_2021`, `model.graphs_2024`
- Moved to device automatically during training
- Static graphs shared across all batches

### Feature Normalization
- Log transformation applied to handle skewed flow distributions
- Sign preserved for net flow (important for direction classification)
- No additional normalization (log transform sufficient)

### Reproducibility
- Set `RANDOM_SEED = 42` in `config.py`
- PyTorch, NumPy, and Python random seeds all set
- Data split deterministic with fixed seed

---

## Architecture Rationale

### Why Hierarchical Classification?
- Decomposes complex 9-class problem into two simpler 3-class problems
- Intensity and direction are conceptually independent dimensions
- Improves interpretability and training stability

### Why Dual-Branch Design?
- Temporal patterns (LSTM+SPP) and spatial patterns (GAT) require different inductive biases
- Separate branches allow specialized processing
- Fusion layer learns optimal combination

### Why Static Graphs?
- Dynamic graphs (one per day) cause memory issues with large graphs
- Static aggregated graphs capture overall spatial structure
- Temporal information still captured by processing each day's features separately

### Why Log Transformation?
- Flow data highly skewed (few high-flow edges, many low-flow edges)
- Log transform stabilizes variance and improves model training
- Sign preservation for net flow maintains directional information

---

## Troubleshooting

### Cache Issues
**Problem**: Stale cache after data changes
**Solution**: Delete `data/cache/*.pkl` and rerun

### Graph Construction Errors
**Problem**: "No edges created" warning
**Solution**: Lower `FLOW_THRESHOLD` in `config.py`

### OOM During Training
**Problem**: CUDA out of memory
**Solution**: Reduce `BATCH_SIZE`, `GAT_HIDDEN_SIZE`, or use CPU

### Poor Performance
**Problem**: Low accuracy on validation set
**Solution**:
- Check class imbalance (weights computed automatically)
- Increase `NUM_EPOCHS` or reduce `EARLY_STOPPING_PATIENCE`
- Verify data preprocessing (check cache file)

---

This documentation covers the complete architecture and data flow of the PRD mobility pattern classification system. The hierarchical dual-branch model combines temporal (LSTM+SPP) and spatial (GAT) processing to classify 9 types of mobility change patterns between 2021 and 2024.
