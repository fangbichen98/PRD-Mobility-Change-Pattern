# Phase 30-31: Actionable Recommendations & Next Experiments

**Date**: 2026-03-23
**Focus**: Concrete action items to improve model performance and understand spatial branch contribution

---

## Executive Summary

### Current Best Model
- **Configuration**: Transformer + Flow+Dist+Dir (GINE spatial branch)
- **Test Accuracy**: 73.56%
- **Test F1**: 0.7304
- **Key Strength**: Balanced performance across most classes
- **Key Weakness**: Poor performance on Classes 5, 7, 9 (recall < 0.50)

### Critical Findings
1. **Temporal branch dominates**: 11x more impactful than spatial features
2. **Edge features help marginally**: +0.44% test accuracy (Transformer only)
3. **Persistent confusions**: Class 4↔5 (30-35 samples), Class 7↔9 (26-30 samples)
4. **Validation-test discrepancy**: Edge features hurt validation (-3.56%) but help test (+0.44%)

---

## Priority 1: Verify Spatial Branch Contribution (CRITICAL)

### Experiment 1A: Temporal-Only Ablation
**Goal**: Quantify spatial branch's actual contribution

**Implementation**:
```python
# config.py
BRANCH_ABLATION_MODE = "temporal_only"  # Disable spatial branch entirely
TEMPORAL_MODEL = "TRANSFORMER"
```

**Expected Results**:
- If test accuracy drops < 1%: **Spatial branch is nearly useless**
- If test accuracy drops 1-3%: **Spatial branch is marginally useful** (current hypothesis)
- If test accuracy drops > 3%: **Spatial branch is essential** (unlikely)

**Decision Tree**:
```
Test Accuracy Drop:
├── < 1%: Remove spatial branch, focus on temporal improvements
├── 1-3%: Keep spatial branch, but don't over-invest in it
└── > 3%: Spatial branch is valuable, invest in improving it
```

### Experiment 1B: Spatial-Only Ablation
**Goal**: Understand spatial branch's standalone capability

**Implementation**:
```python
# config.py
BRANCH_ABLATION_MODE = "spatial_only"  # Disable temporal branch
SPATIAL_MODEL = "GINE"
EDGE_FEATURE_MODE = "flow_distance_direction"
```

**Expected Results**:
- Likely test accuracy: 30-50% (random baseline is 11.11%)
- Will reveal if spatial features capture any meaningful patterns

### Experiment 1C: Log Fusion Gate Weights
**Goal**: Understand how gated fusion balances temporal vs spatial features

**Implementation**:
```python
# src/models/gated_fusion.py
class GatedFeatureFusion(nn.Module):
    def forward(self, temporal_features, spatial_features):
        # ... existing code ...
        gates = torch.sigmoid(self.gate_network(combined))  # (batch, 6)

        # Log gate statistics (add this)
        temporal_gate_mean = gates[:, :3].mean().item()
        spatial_gate_mean = gates[:, 3:].mean().item()

        if self.training:
            # Log to tensorboard or file
            self.log_gates(temporal_gate_mean, spatial_gate_mean)

        return fused
```

**Expected Results**:
- If temporal_gate_mean >> spatial_gate_mean: Confirms temporal dominance
- If spatial_gate_mean ≈ 0: Spatial branch is being ignored
- If gates vary by class: Some classes benefit more from spatial features

**Analysis**:
```python
# After training, analyze gate logs
import pandas as pd
gates_df = pd.read_csv("fusion_gates.log")

print("Average gate weights:")
print(f"  Temporal: {gates_df['temporal_gate'].mean():.3f}")
print(f"  Spatial:  {gates_df['spatial_gate'].mean():.3f}")

# Per-class analysis
for class_id in range(1, 10):
    class_gates = gates_df[gates_df['true_label'] == class_id]
    print(f"Class {class_id}: Temporal={class_gates['temporal_gate'].mean():.3f}, "
          f"Spatial={class_gates['spatial_gate'].mean():.3f}")
```

---

## Priority 2: Investigate Label Quality (HIGH PRIORITY)

### Problem: Persistent Confusion Patterns Suggest Label Issues

**Evidence**:
- Class 5 → Class 4: 21 samples (45% of Class 5) - **WORST**
- Class 4 → Class 5: 12 samples (20% of Class 4)
- Class 9 → Class 7: 15 samples (33% of Class 9)
- Class 7 → Class 9: 14 samples (30% of Class 7)

**These confusions are UNCHANGED across all 4 experiments**, suggesting:
1. Classes may be inherently overlapping
2. Labels may be noisy or ambiguous
3. Current features cannot distinguish these classes

### Experiment 2A: Visualize Confused Samples
**Goal**: Understand what makes confused samples different

**Implementation**:
```python
# scripts/analyze_confusions.py
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.dual_year_processor import DualYearProcessor

# Load data and predictions
processor = DualYearProcessor(...)
data = processor.load_data()

# Load best model predictions
predictions = np.load("outputs/.../predictions.npy")
true_labels = np.load("outputs/.../true_labels.npy")

# Find confused samples
class5_to_class4 = np.where((true_labels == 4) & (predictions == 3))[0]  # 0-indexed
class4_to_class5 = np.where((true_labels == 3) & (predictions == 4))[0]

# Visualize temporal patterns
fig, axes = plt.subplots(4, 5, figsize=(20, 12))
for i, idx in enumerate(class5_to_class4[:5]):
    grid_id = test_indices[idx]
    features = data['change_features'][grid_id]  # (168, 4)

    # Plot inflow/outflow for 2021 and 2024
    axes[0, i].plot(features[:, 0], label='2021 inflow')
    axes[1, i].plot(features[:, 1], label='2021 outflow')
    axes[2, i].plot(features[:, 2], label='2024 inflow')
    axes[3, i].plot(features[:, 3], label='2024 outflow')
    axes[0, i].set_title(f"Class 5→4 (Grid {grid_id})")

plt.savefig("output_analysis/class5_to_class4_patterns.png")

# Repeat for other confusions
```

**Expected Insights**:
- Visual inspection may reveal if confused samples are truly similar
- May identify labeling errors (e.g., Class 5 sample that looks like Class 4)

### Experiment 2B: Statistical Analysis of Confused Classes
**Goal**: Quantify feature distribution overlap

**Implementation**:
```python
# scripts/analyze_class_overlap.py
import numpy as np
from scipy.stats import ks_2samp
from src.preprocessing.dual_year_processor import DualYearProcessor

processor = DualYearProcessor(...)
data = processor.load_data()
labels = data['labels']

# Extract features for Class 4 and Class 5
class4_features = [data['change_features'][i] for i in range(len(labels)) if labels[i] == 3]
class5_features = [data['change_features'][i] for i in range(len(labels)) if labels[i] == 4]

# Compute summary statistics
class4_stats = {
    'mean_inflow_2021': np.mean([f[:, 0].mean() for f in class4_features]),
    'mean_outflow_2021': np.mean([f[:, 1].mean() for f in class4_features]),
    'mean_inflow_2024': np.mean([f[:, 2].mean() for f in class4_features]),
    'mean_outflow_2024': np.mean([f[:, 3].mean() for f in class4_features]),
    'flow_change': np.mean([f[:, 2:].mean() - f[:, :2].mean() for f in class4_features])
}

class5_stats = {
    'mean_inflow_2021': np.mean([f[:, 0].mean() for f in class5_features]),
    'mean_outflow_2021': np.mean([f[:, 1].mean() for f in class5_features]),
    'mean_inflow_2024': np.mean([f[:, 2].mean() for f in class5_features]),
    'mean_outflow_2024': np.mean([f[:, 3].mean() for f in class5_features]),
    'flow_change': np.mean([f[:, 2:].mean() - f[:, :2].mean() for f in class5_features])
}

print("Class 4 vs Class 5 Feature Statistics:")
for key in class4_stats:
    print(f"  {key}:")
    print(f"    Class 4: {class4_stats[key]:.3f}")
    print(f"    Class 5: {class5_stats[key]:.3f}")
    print(f"    Difference: {abs(class4_stats[key] - class5_stats[key]):.3f}")

# Kolmogorov-Smirnov test for distribution similarity
class4_flow_change = [f[:, 2:].mean() - f[:, :2].mean() for f in class4_features]
class5_flow_change = [f[:, 2:].mean() - f[:, :2].mean() for f in class5_features]
ks_stat, p_value = ks_2samp(class4_flow_change, class5_flow_change)

print(f"\nKS Test: statistic={ks_stat:.3f}, p-value={p_value:.3f}")
if p_value > 0.05:
    print("  → Classes 4 and 5 have SIMILAR distributions (may need merging)")
else:
    print("  → Classes 4 and 5 have DIFFERENT distributions (labels are valid)")
```

**Expected Outcomes**:
- If p-value > 0.05: Classes are statistically similar → **Consider merging**
- If p-value < 0.05: Classes are distinct → **Improve feature engineering**

### Experiment 2C: Check Label File for Errors
**Goal**: Verify label consistency and quality

**Implementation**:
```bash
# Check label distribution
python3 << 'EOF'
import pandas as pd
labels_df = pd.read_csv("data/label_sgh.csv")
print("Label Distribution:")
print(labels_df['label'].value_counts().sort_index())

# Check for duplicate grid IDs
duplicates = labels_df[labels_df.duplicated(subset=['grid_id'], keep=False)]
if len(duplicates) > 0:
    print(f"\nWARNING: {len(duplicates)} duplicate grid IDs found!")
    print(duplicates)

# Check for invalid labels
invalid = labels_df[~labels_df['label'].isin(range(1, 10))]
if len(invalid) > 0:
    print(f"\nWARNING: {len(invalid)} invalid labels found!")
    print(invalid)
EOF
```

---

## Priority 3: Improve Edge Features (MEDIUM PRIORITY)

### Current Edge Features (4D)
```python
edge_attr = [
    log(flow),        # (E, 1) - Flow magnitude
    direction_x,      # (E, 1) - Normalized direction vector X
    direction_y,      # (E, 1) - Normalized direction vector Y
    log(distance)     # (E, 1) - Spatial distance
]
```

### Experiment 3A: Add Flow Ratio Feature (5D)
**Goal**: Capture flow change magnitude on edges

**Implementation**:
```python
# src/preprocessing/graph_builder.py
def build_flow_graph_with_features(self, od_df_2021, od_df_2024, metadata_df):
    # ... existing code to build edge_index, flow_2021, flow_2024 ...

    # Compute flow ratio (change magnitude)
    flow_ratio = np.log1p(flow_2024) - np.log1p(flow_2021)  # Log difference

    # Combine all edge features
    edge_attr = np.concatenate([
        np.log1p(flow_2024).reshape(-1, 1),  # Current flow
        directions,                           # (E, 2)
        np.log1p(distances).reshape(-1, 1),  # Distance
        flow_ratio.reshape(-1, 1)            # Flow change (NEW)
    ], axis=1)  # (E, 5)

    return edge_index, edge_attr
```

**Expected Impact**:
- May help distinguish Class 4 vs Class 5 (different flow change magnitudes)
- May help distinguish Class 8 (extreme change) from Class 2 (moderate change)

### Experiment 3B: Add Directional Change Feature (6D)
**Goal**: Capture how flow direction changes between years

**Implementation**:
```python
# src/preprocessing/graph_builder.py
def compute_directional_change(self, od_df_2021, od_df_2024):
    # For each edge, compute direction vector in 2021 and 2024
    # Then compute angle between them

    # Get flow vectors for 2021
    flow_2021_by_edge = od_df_2021.groupby(['o_grid', 'd_grid'])['num_total'].sum()

    # Get flow vectors for 2024
    flow_2024_by_edge = od_df_2024.groupby(['o_grid', 'd_grid'])['num_total'].sum()

    # Compute angle change (simplified: use flow magnitude ratio as proxy)
    # More sophisticated: compute actual direction change in flow space
    direction_change = np.arctan2(flow_2024, flow_2021 + 1e-8)

    return direction_change
```

**Expected Impact**:
- May help distinguish Class 7 vs Class 9 (different directional changes)

### Experiment 3C: Add Inflow/Outflow Ratio (7D)
**Goal**: Capture flow directionality at node level

**Implementation**:
```python
# src/preprocessing/graph_builder.py
def compute_inflow_outflow_ratio(self, edge_index, edge_weights):
    # For each node, compute inflow/outflow ratio
    num_nodes = edge_index.max() + 1
    inflow = np.zeros(num_nodes)
    outflow = np.zeros(num_nodes)

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        weight = edge_weights[i]
        outflow[src] += weight
        inflow[dst] += weight

    # Compute ratio for each edge (use source node's ratio)
    io_ratio = np.log1p(inflow / (outflow + 1e-8))
    edge_io_ratio = io_ratio[edge_index[0]]  # Use source node's ratio

    return edge_io_ratio
```

**Expected Impact**:
- May help distinguish classes with different flow directionality patterns

---

## Priority 4: Increase Model Capacity (LOW PRIORITY)

### Current Spatial Branch Configuration
```python
SPATIAL_MODEL = "GINE"
SPATIAL_LAYERS = 2
SPATIAL_HIDDEN_SIZE = 128
LAPLACIAN_PE_DIM = 16
```

### Experiment 4A: Deeper GINE (3-4 layers)
**Goal**: Test if spatial branch is under-parameterized

**Implementation**:
```python
# config.py
SPATIAL_LAYERS = 4  # From 2 to 4
```

**Expected Results**:
- If accuracy improves: Spatial branch needs more depth
- If accuracy unchanged: Depth is not the bottleneck

### Experiment 4B: Wider GINE (256 hidden units)
**Goal**: Test if spatial branch needs more capacity

**Implementation**:
```python
# config.py
SPATIAL_HIDDEN_SIZE = 256  # From 128 to 256
```

**Expected Results**:
- If accuracy improves: Spatial branch needs more capacity
- If accuracy unchanged: Width is not the bottleneck

### Experiment 4C: Increase Laplacian PE Dimension
**Goal**: Test if positional encoding is insufficient

**Implementation**:
```python
# config.py
LAPLACIAN_PE_DIM = 32  # From 16 to 32
```

**Expected Results**:
- If accuracy improves: PE dimension was limiting
- If accuracy unchanged: PE dimension is sufficient

---

## Priority 5: Alternative Architectures (RESEARCH DIRECTION)

### Experiment 5A: Attention-Based Fusion
**Goal**: Replace gated fusion with cross-attention

**Implementation**:
```python
# src/models/attention_fusion.py
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, temporal_features, spatial_features):
        # temporal_features: (batch, 3, 256)
        # spatial_features: (batch, 3, 256)

        # Use temporal as query, spatial as key/value
        fused, attention_weights = self.cross_attention(
            query=temporal_features,
            key=spatial_features,
            value=spatial_features
        )

        # Log attention weights to understand spatial contribution
        self.log_attention_weights(attention_weights)

        return fused.mean(dim=1)  # (batch, 256)
```

**Expected Benefits**:
- Explicit attention weights show which spatial features are used
- May improve fusion quality

### Experiment 5B: Hierarchical Classification
**Goal**: Two-stage classification to handle confused classes

**Implementation**:
```python
# Stage 1: Classify into 3 super-classes
super_class_mapping = {
    1: 0,  # Stable/high-flow classes
    6: 0,
    8: 0,
    2: 1,  # Moderate change classes
    3: 1,
    4: 2,  # Complex change classes
    5: 2,
    7: 2,
    9: 2
}

# Stage 2: Within-super-class classification
# Train separate models for each super-class
# Super-class 2 model uses more spatial features
```

**Expected Benefits**:
- Better handling of confused classes (4, 5, 7, 9)
- Specialized models for different change patterns

### Experiment 5C: Temporal Graph Networks (TGN)
**Goal**: Model temporal evolution of graph structure

**Implementation**:
```python
# Instead of static graphs, use dynamic graphs
# One graph per day (7 graphs total)
# Use TGN or TGAT to model temporal graph evolution
```

**Expected Benefits**:
- Captures how spatial structure changes over time
- May better model mobility pattern changes

---

## Recommended Experiment Sequence

### Week 1: Verify Spatial Branch Contribution
1. **Experiment 1A**: Temporal-only ablation (1 run, ~30 min)
2. **Experiment 1B**: Spatial-only ablation (1 run, ~30 min)
3. **Experiment 1C**: Log fusion gate weights (modify existing code, re-run best model)

**Decision Point**: If spatial branch contributes < 1%, **stop investing in spatial features**

### Week 2: Investigate Label Quality
1. **Experiment 2A**: Visualize confused samples (analysis only, ~2 hours)
2. **Experiment 2B**: Statistical analysis of class overlap (analysis only, ~1 hour)
3. **Experiment 2C**: Check label file for errors (analysis only, ~30 min)

**Decision Point**: If classes are statistically similar, **consider merging** Class 4+5 and Class 7+9

### Week 3: Improve Edge Features (if spatial branch is useful)
1. **Experiment 3A**: Add flow ratio feature (1 run, ~30 min)
2. **Experiment 3B**: Add directional change feature (1 run, ~30 min)
3. **Experiment 3C**: Add inflow/outflow ratio (1 run, ~30 min)

**Decision Point**: If edge features improve accuracy > 1%, **continue with spatial features**

### Week 4: Increase Model Capacity (if needed)
1. **Experiment 4A**: Deeper GINE (1 run, ~45 min)
2. **Experiment 4B**: Wider GINE (1 run, ~45 min)
3. **Experiment 4C**: Larger Laplacian PE (1 run, ~30 min)

---

## Expected Outcomes & Success Criteria

### Scenario 1: Spatial Branch is Useful (Temporal-only drops > 2%)
**Action Plan**:
- Continue with Experiments 3A-3C (improve edge features)
- Then try Experiments 4A-4C (increase capacity)
- Target: 75-76% test accuracy

### Scenario 2: Spatial Branch is Marginal (Temporal-only drops 1-2%)
**Action Plan**:
- Focus on improving temporal branch (e.g., larger Transformer)
- Keep spatial branch but don't over-invest
- Target: 74-75% test accuracy

### Scenario 3: Spatial Branch is Useless (Temporal-only drops < 1%)
**Action Plan**:
- Remove spatial branch entirely
- Focus all efforts on temporal modeling
- Consider alternative temporal architectures (e.g., TCN, Mamba)
- Target: 73-74% test accuracy (similar to current)

### Scenario 4: Label Quality Issues (Classes are statistically similar)
**Action Plan**:
- Merge confused classes (4+5, 7+9)
- Retrain with 7-class classification
- Target: 80-85% test accuracy (easier task)

---

## Key Metrics to Track

### For Each Experiment:
1. **Overall Metrics**:
   - Test accuracy
   - Test F1 score
   - Training time
   - Best epoch

2. **Per-Class Metrics**:
   - Recall for Classes 4, 5, 7, 9 (currently problematic)
   - Precision for Classes 4, 5, 7, 9
   - Confusion matrix

3. **Fusion Analysis** (if applicable):
   - Temporal gate mean
   - Spatial gate mean
   - Per-class gate statistics

4. **Convergence Analysis**:
   - Epochs to best model
   - Validation-test gap
   - Overfitting indicators

---

## Summary: What to Do Next

### Immediate Actions (This Week):
1. ✅ **Run Experiment 1A** (Temporal-only ablation) - **HIGHEST PRIORITY**
2. ✅ **Run Experiment 1C** (Log fusion gates) - **HIGH PRIORITY**
3. ✅ **Run Experiment 2B** (Statistical class overlap analysis) - **HIGH PRIORITY**

### Based on Results:
- **If spatial branch is useful**: Proceed with edge feature improvements (Experiments 3A-3C)
- **If spatial branch is useless**: Focus on temporal branch improvements
- **If classes overlap**: Consider merging classes and retraining

### Long-Term Goals:
- **Target accuracy**: 75-76% (if spatial branch is useful)
- **Target accuracy**: 80-85% (if classes are merged)
- **Research contribution**: Understand when spatial features help in spatiotemporal tasks

---

**Generated**: 2026-03-23
**Status**: Ready for implementation
**Estimated Time**: 3-4 weeks for full experiment sequence
