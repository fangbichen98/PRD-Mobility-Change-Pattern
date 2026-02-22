# Model Optimization Implementation Summary

**Date**: 2026-02-21  
**Objective**: Improve accuracy from 45.28% to 50%+ on labels2 dataset

---

## Changes Implemented

### 1. Enhanced Temporal Features ✅

**File**: `src/preprocessing/dual_year_processor.py` (lines 250-283)

**Before**: 1 feature
- `total_flow_log`

**After**: 5 features  
- `total_flow_log` - total inflow + outflow
- `inflow_log` - inflow only
- `outflow_log` - outflow only  
- `net_flow_log` - outflow - inflow (sign preserved)
- `flow_variance_log` - variance across 7 days

**Shape per grid**: (7, 10) = [5×2021, 5×2024]

**Config change**: `TEMPORAL_INPUT_SIZE: 1 → 5`

**Expected impact**: +3-5% accuracy improvement

---

### 2. Upgraded GAT Architecture ✅

**File**: `config.py`

**Before**:
```python
GAT_LAYERS = 3
GAT_HEADS = 4
```

**After**:
```python
GAT_LAYERS = 4  # +1 layer
GAT_HEADS = 6   # +2 heads
```

**Model parameters**:
- Before: ~3.9M (estimated)
- After: 2.59M (actual, different architecture)
- Spatial branch config verified: 4 layers, 6 heads, 128 hidden

**Expected impact**: +2-4% accuracy improvement

---

### 3. Extended Training ✅

**File**: `config.py`

**Changes**:
```python
NUM_EPOCHS: 100 → 150
EARLY_STOPPING_PATIENCE: 15 → 20
```

**Rationale**: Larger model needs more time to converge

**Expected impact**: +1-2% accuracy improvement

---

### 4. Dataset Configuration ✅

**File**: `train_pure_graph.py` (line 244)

**Changed**:
```python
label_path='data/labels3.csv' → 'data/labels2.csv'
```

**Dataset**: labels2.csv
- Samples: 10,113 grids
- Baseline accuracy: 45.28%
- Baseline F1: 0.3193

**Feature extraction updated** (lines 263-267):
```python
# Before: features[:, [0]] and features[:, [1]]  # (7, 1)
# After:  features[:, 0:5] and features[:, 5:10]  # (7, 5)
```

---

## Model Architecture Summary

```
PureGraphDualBranchModel
├── Temporal Branch (ParallelTemporalBranch)
│   ├── Input: (batch, 7, 5) - 5 temporal features
│   ├── LSTM Branch: 2-layer LSTM (128 hidden)
│   └── SPP Branch: Spatial Pyramid Pooling [1×1, 2×2, 4×4]
│   Output: 6 features (2021, 2024, diff) × 2 branches
│
├── Spatial Branch (PureGraphDualYearGAT)
│   ├── Input: 3 structural features (in_degree, out_degree, total_degree)
│   ├── Architecture: 4 layers × 6 attention heads
│   ├── Hidden size: 128 per head
│   └── Output: 3 features (2021, 2024, diff)
│
├── Fusion Layer (MultiFeatureAttentionFusion)
│   ├── Input: 9 features (6 temporal + 3 spatial)
│   ├── Multi-head self-attention: 4 heads
│   └── Output: 256-dim fused representation
│
└── Classifier
    └── 9-class output (mobility patterns)
```

**Total parameters**: 2,590,985  
**Trainable parameters**: 2,590,985

---

## Validation Results

✅ All configuration changes validated  
✅ Model initializes correctly with new config  
✅ Feature extraction logic updated  
✅ Cache cleared (ready for fresh preprocessing)  
✅ Label file exists: data/labels2.csv (10,113 samples)

---

## Expected Performance

### Conservative Estimate
- Accuracy: 48-50% (+3-5%)
- F1 Score: 0.35-0.38
- Minority class recall: 5-10%

### Expected Estimate
- Accuracy: 50-53% (+5-8%)
- F1 Score: 0.38-0.42
- Minority class recall: 10-15%

### Ideal Target
- Accuracy: 53-55% (+8-10%)
- F1 Score: 0.42-0.45
- Minority class recall: 15-20%

---

## Training Command

```bash
# Run optimized training on labels2 dataset
CUDA_VISIBLE_DEVICES=3 nohup python train_pure_graph.py \
    > train_optimized_labels2.log 2>&1 &

# Monitor training
tail -f train_optimized_labels2.log

# Check tensorboard
tensorboard --logdir outputs/logs
```

**Expected training time**: 3-4 hours (150 epochs)

---

## Key Improvement Mechanisms

1. **Richer temporal features**: 5× more information at each time step
2. **Deeper GAT**: Better capture of complex spatial patterns
3. **More attention heads**: Richer multi-scale representations
4. **Longer training**: Full convergence of larger model

---

## Verification Checklist

- [x] Temporal features: 1D → 5D
- [x] GAT architecture: 3×4 → 4×6
- [x] Training epochs: 100 → 150
- [x] Early stopping patience: 15 → 20
- [x] Label path: labels3 → labels2
- [x] Cache cleared
- [x] Model initializes correctly
- [x] All imports work

**Status**: Ready to train ✅

---

## Files Modified

1. `config.py` - Core configuration
2. `src/preprocessing/dual_year_processor.py` - Feature extraction
3. `train_pure_graph.py` - Training script

## Files Created

1. `OPTIMIZATION_CHANGES.md` - This summary

---

**Next step**: Start training and monitor for 3-4 hours
