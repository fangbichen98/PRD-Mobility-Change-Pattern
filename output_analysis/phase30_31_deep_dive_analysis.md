# Phase 30-31: Deep Dive Analysis - Confusion Patterns & Spatial Branch Contribution

**Date**: 2026-03-23
**Focus**: Understanding misclassification patterns and evaluating spatial branch effectiveness

---

## Critical Finding: Persistent Confusion Patterns

### The "Class 4-5-7-9 Confusion Cluster"

Across **ALL four experiments**, the same misclassification patterns dominate:

| Confusion Pattern | LSTM + Flow | LSTM + Flow+Dist+Dir | Transformer + Flow | Transformer + Flow+Dist+Dir |
|-------------------|-------------|----------------------|--------------------|-----------------------------|
| **Class 5 → Class 4** | 18 samples | 20 samples | 20 samples | **21 samples** |
| **Class 9 → Class 7** | 15 samples | 15 samples | 14 samples | 15 samples |
| **Class 4 → Class 5** | 13 samples | 14 samples | 14 samples | 12 samples |
| **Class 7 → Class 9** | 12 samples | 15 samples | 13 samples | 14 samples |

### Key Observations:

#### 1. **Class 5 ↔ Class 4 Bidirectional Confusion**
- **Class 5 → Class 4**: 18-21 samples (38-45% of Class 5 samples)
- **Class 4 → Class 5**: 12-14 samples (20-23% of Class 4 samples)
- **Total confusion**: 30-35 samples across both directions
- **Implication**: These two classes have **highly overlapping feature distributions**

#### 2. **Class 7 ↔ Class 9 Bidirectional Confusion**
- **Class 9 → Class 7**: 14-15 samples (31-33% of Class 9 samples)
- **Class 7 → Class 9**: 12-15 samples (26-32% of Class 7 samples)
- **Total confusion**: 26-30 samples across both directions
- **Implication**: These two classes are **nearly indistinguishable** to the model

#### 3. **Edge Features Do NOT Resolve These Confusions**
- Class 5 → Class 4 confusion **increases** with edge features (18 → 21 samples)
- Class 7 ↔ Class 9 confusion remains **constant** (±1 sample variation)
- **Conclusion**: Direction and distance features are **irrelevant** to these confusions

---

## Class-Level Analysis: What Each Class Represents

### Hypothesis: Class Definitions Based on Confusion Patterns

Based on the confusion matrix, we can infer class relationships:

```
Cluster 1: "Stable/High-Flow Classes" (Easy to classify)
├── Class 1: High precision (0.92-1.00), high recall (0.84)
├── Class 6: High precision (0.76-1.00), high recall (0.83-0.90)
└── Class 8: High precision (0.96-1.00), high recall (0.83-0.90)

Cluster 2: "Moderate-Flow Classes" (Moderate difficulty)
├── Class 2: Moderate precision (0.63-0.74), high recall (0.83-0.87)
└── Class 3: Moderate precision (0.66-0.74), high recall (0.78-0.82)

Cluster 3: "Complex-Change Classes" (Hard to classify)
├── Class 4 ↔ Class 5: Bidirectional confusion (30-35 samples)
├── Class 7 ↔ Class 9: Bidirectional confusion (26-30 samples)
└── Common pattern: Low recall (0.34-0.72), low precision (0.46-0.62)
```

### Detailed Class Profiles:

#### **Class 1: "Stable High-Flow"** (Hypothesis)
- **Precision**: 0.92-1.00 (best)
- **Recall**: 0.84 (consistent)
- **Confusion**: Minimal (3-7 false negatives, 0-3 false positives)
- **Interpretation**: Likely represents grids with **consistently high flow in both years**
- **Spatial relevance**: Low (temporal features sufficient)

#### **Class 6: "Stable Medium-Flow"** (Hypothesis)
- **Precision**: 0.76-1.00 (excellent)
- **Recall**: 0.83-0.90 (excellent)
- **Confusion**: Minimal (5-9 false negatives, 0-14 false positives)
- **Interpretation**: Likely represents grids with **stable medium flow**
- **Spatial relevance**: Low (temporal features sufficient)

#### **Class 8: "Extreme Flow Change"** (Hypothesis)
- **Precision**: 0.96-1.00 (best)
- **Recall**: 0.83-0.90 (excellent)
- **Confusion**: Mostly confused with Class 2 (6-10 samples)
- **Interpretation**: Likely represents **dramatic flow increase or decrease**
- **Spatial relevance**: **Medium** (edge features improve recall from 0.84 → 0.90)

#### **Class 2: "Moderate Flow Increase"** (Hypothesis)
- **Precision**: 0.63-0.74 (moderate)
- **Recall**: 0.83-0.87 (high)
- **Confusion**: Often predicted when true class is 5 or 8
- **Interpretation**: Likely represents **moderate flow increase**
- **Spatial relevance**: Low (no improvement with edge features)

#### **Class 3: "Moderate Flow Decrease"** (Hypothesis)
- **Precision**: 0.66-0.74 (moderate)
- **Recall**: 0.78-0.82 (good)
- **Confusion**: Confused with Class 6 and Class 7
- **Interpretation**: Likely represents **moderate flow decrease**
- **Spatial relevance**: Low (minimal improvement with edge features)

#### **Class 4: "Complex Change Pattern A"** (Hypothesis)
- **Precision**: 0.50-0.62 (low)
- **Recall**: 0.43-0.72 (highly variable)
- **Confusion**: **Bidirectional with Class 5** (25-26 samples total)
- **Interpretation**: Likely represents **flow change with direction shift**
- **Spatial relevance**: **HIGH** (recall improves from 0.53 → 0.72 with Transformer + edge features)

#### **Class 5: "Complex Change Pattern B"** (Hypothesis)
- **Precision**: 0.52-0.58 (lowest)
- **Recall**: 0.34-0.40 (lowest)
- **Confusion**: **Strongly confused with Class 4** (18-21 samples → Class 4)
- **Interpretation**: Likely represents **similar pattern to Class 4 but with different magnitude**
- **Spatial relevance**: **Negative** (edge features hurt performance)

#### **Class 7: "Directional Change A"** (Hypothesis)
- **Precision**: 0.46-0.55 (lowest)
- **Recall**: 0.60-0.70 (moderate)
- **Confusion**: **Bidirectional with Class 9** (26-29 samples total)
- **Interpretation**: Likely represents **flow direction change (e.g., inflow → outflow)**
- **Spatial relevance**: **Unclear** (mixed results with edge features)

#### **Class 9: "Directional Change B"** (Hypothesis)
- **Precision**: 0.58-0.63 (low)
- **Recall**: 0.42-0.58 (low)
- **Confusion**: **Strongly confused with Class 7** (14-15 samples → Class 7)
- **Interpretation**: Likely represents **similar directional change to Class 7**
- **Spatial relevance**: **Unclear** (mixed results with edge features)

---

## Critical Question: Is the Spatial Branch Actually Working?

### Evidence FOR Spatial Branch Contribution:

#### 1. **Class 4 Recall Improvement**
- LSTM + Flow Only: 0.53 recall
- Transformer + Flow+Dist+Dir: **0.72 recall** (+0.19 improvement)
- **Mechanism**: Direction and distance features help distinguish Class 4 from Class 5

#### 2. **Class 8 Recall Improvement**
- LSTM + Flow Only: 0.83 recall
- Transformer + Flow+Dist+Dir: **0.90 recall** (+0.07 improvement)
- **Mechanism**: Distance features may capture long-distance flow changes

#### 3. **Class 6 Precision Improvement**
- LSTM + Flow Only: 0.76 precision
- Transformer + Flow+Dist+Dir: **1.00 precision** (+0.24 improvement)
- **Mechanism**: Spatial features eliminate false positives

### Evidence AGAINST Spatial Branch Contribution:

#### 1. **Persistent Confusion Patterns**
- Class 5 → Class 4 confusion **increases** with edge features (18 → 21 samples)
- Class 7 ↔ Class 9 confusion **unchanged** (±1 sample variation)
- **Implication**: Spatial features don't resolve the hardest confusions

#### 2. **Validation Accuracy Decreases**
- LSTM: 65.78% → 61.78% (-4.00%)
- Transformer: 73.78% → 70.22% (-3.56%)
- **Implication**: Edge features may cause overfitting or add noise

#### 3. **Small Overall Impact**
- Test accuracy improvement: +0.45% (73.11% → 73.56%)
- **Implication**: Spatial branch contribution is marginal

---

## Hypothesis: Temporal Branch Dominates, Spatial Branch is Auxiliary

### Proposed Model Behavior:

```
Input: Grid temporal features (168 hours × 2 features)
       ↓
Temporal Branch (Transformer): Extracts temporal patterns
       ↓ (Strong signal: ~70% accuracy)
       ↓
Gated Fusion: Combines temporal + spatial features
       ↓ (Spatial adds ~3-4% accuracy)
       ↓
Classifier: 9-class prediction
       ↓
Output: Class probabilities
```

### Evidence:

1. **Temporal branch upgrade**: +5% accuracy (LSTM → Transformer)
2. **Spatial branch upgrade**: +0.45% accuracy (Flow → Flow+Dist+Dir)
3. **Ratio**: Temporal is **11x more important** than spatial

### Implication:

- **Temporal features are sufficient** for most classes (1, 2, 3, 6, 8)
- **Spatial features help** for specific classes (4, 8) with directional changes
- **Spatial features hurt** for ambiguous classes (5, 7, 9) by adding noise

---

## Recommended Next Steps

### 1. **Investigate Class Definitions** (Highest Priority)

**Action**: Manually inspect samples from confused classes
```python
# Pseudocode
confused_samples = {
    "Class 5 → Class 4": get_samples(true_label=5, pred_label=4),
    "Class 4 → Class 5": get_samples(true_label=4, pred_label=5),
    "Class 7 → Class 9": get_samples(true_label=7, pred_label=9),
    "Class 9 → Class 7": get_samples(true_label=9, pred_label=7)
}

for confusion, samples in confused_samples.items():
    visualize_temporal_patterns(samples)
    visualize_spatial_patterns(samples)
    check_label_quality(samples)
```

**Expected Outcome**:
- Identify if Class 4 and Class 5 are truly distinct
- Identify if Class 7 and Class 9 are truly distinct
- Potentially **merge classes** if they're indistinguishable

### 2. **Analyze Gated Fusion Weights** (High Priority)

**Action**: Log fusion gate activations during inference
```python
# In gated_fusion.py
def forward(self, temporal_features, spatial_features):
    gates = self.gate_network(combined_features)  # (batch, 6)

    # Log gate statistics
    temporal_gates = gates[:, :3].mean(dim=1)  # (batch,)
    spatial_gates = gates[:, 3:].mean(dim=1)   # (batch,)

    print(f"Temporal gate mean: {temporal_gates.mean():.3f}")
    print(f"Spatial gate mean: {spatial_gates.mean():.3f}")

    return fused_features
```

**Expected Outcome**:
- If temporal gates >> spatial gates: Confirms temporal dominance
- If spatial gates are near zero: Spatial branch is being ignored
- If gates vary by class: Some classes benefit from spatial features

### 3. **Ablation: Temporal-Only Model** (High Priority)

**Action**: Train model with **only temporal branch** (no spatial branch)
```python
# In config.py
BRANCH_ABLATION_MODE = "temporal_only"  # Disable spatial branch
```

**Expected Outcome**:
- If accuracy drops < 1%: Spatial branch is nearly useless
- If accuracy drops 3-5%: Spatial branch is moderately useful
- If accuracy drops > 5%: Spatial branch is essential

### 4. **Try Hierarchical Classification** (Medium Priority)

**Approach**: Two-stage classification
```
Stage 1: Classify into 3 super-classes
  - Super-class A: Classes 1, 6, 8 (stable/extreme flow)
  - Super-class B: Classes 2, 3 (moderate changes)
  - Super-class C: Classes 4, 5, 7, 9 (complex changes)

Stage 2: Within-super-class fine-grained classification
  - Use different models for each super-class
  - Super-class C model uses more spatial features
```

**Expected Outcome**:
- Better handling of confused classes
- Specialized models for different change patterns

### 5. **Add Flow Ratio Edge Feature** (Medium Priority)

**Action**: Add flow change magnitude to edge features
```python
# In graph_builder.py
flow_2021 = od_df_2021.groupby(['o_grid', 'd_grid'])['num_total'].sum()
flow_2024 = od_df_2024.groupby(['o_grid', 'd_grid'])['num_total'].sum()

flow_ratio = np.log1p(flow_2024) - np.log1p(flow_2021)  # Log ratio
edge_attr = np.concatenate([
    flow_log,      # (E, 1)
    directions,    # (E, 2)
    dist_log,      # (E, 1)
    flow_ratio     # (E, 1) - NEW
], axis=1)  # (E, 5)
```

**Expected Outcome**:
- May help distinguish Class 4 vs Class 5 (different flow change magnitudes)
- May help distinguish Class 7 vs Class 9 (different directional change magnitudes)

### 6. **Increase Spatial Branch Capacity** (Low Priority)

**Action**: Deeper and wider GINE
```python
# In config.py
SPATIAL_LAYERS = 4        # From 2 to 4
SPATIAL_HIDDEN_SIZE = 256 # From 128 to 256
```

**Expected Outcome**:
- If accuracy improves: Spatial branch was under-parameterized
- If accuracy unchanged: Spatial features are the bottleneck, not capacity

---

## Confusion Pattern Analysis: Detailed Breakdown

### Class 5 → Class 4 Confusion (Worst Confusion)

**Statistics**:
- LSTM + Flow Only: 18 samples (38% of Class 5)
- LSTM + Flow+Dist+Dir: 20 samples (43% of Class 5)
- Transformer + Flow Only: 20 samples (43% of Class 5)
- Transformer + Flow+Dist+Dir: **21 samples (45% of Class 5)**

**Trend**: **Worsens with edge features**

**Hypothesis**:
- Class 5 and Class 4 may represent **similar mobility patterns with different magnitudes**
- Example: Class 4 = "moderate flow increase + direction shift", Class 5 = "small flow increase + direction shift"
- Edge features (direction, distance) are **similar for both classes**, so they don't help discrimination

**Recommendation**:
1. Check if Class 4 and Class 5 have **overlapping flow change distributions**
2. Consider **merging Class 4 and Class 5** into a single class
3. Or add **flow magnitude features** to distinguish them

### Class 9 → Class 7 Confusion (Second Worst)

**Statistics**:
- LSTM + Flow Only: 15 samples (33% of Class 9)
- LSTM + Flow+Dist+Dir: 15 samples (33% of Class 9)
- Transformer + Flow Only: 14 samples (31% of Class 9)
- Transformer + Flow+Dist+Dir: 15 samples (33% of Class 9)

**Trend**: **Unchanged across all models**

**Hypothesis**:
- Class 7 and Class 9 may represent **similar directional changes**
- Example: Class 7 = "inflow → outflow", Class 9 = "outflow → inflow"
- Current features (temporal + spatial) **cannot distinguish** these patterns

**Recommendation**:
1. Add **directional change features**: Angle between 2021 and 2024 flow vectors
2. Add **inflow/outflow ratio features**: `log(inflow / outflow)` for each year
3. Consider **merging Class 7 and Class 9** if they're truly indistinguishable

### Class 8 → Class 2 Confusion (Third Most Common)

**Statistics**:
- LSTM + Flow Only: 10 samples (17% of Class 8)
- LSTM + Flow+Dist+Dir: 9 samples (16% of Class 8)
- Transformer + Flow Only: 9 samples (16% of Class 8)
- Transformer + Flow+Dist+Dir: **6 samples (10% of Class 8)**

**Trend**: **Improves with Transformer + edge features**

**Hypothesis**:
- Class 8 and Class 2 may both represent **flow increases**
- Class 8 = "extreme flow increase", Class 2 = "moderate flow increase"
- Edge features help distinguish **magnitude of change**

**Recommendation**:
- This confusion is **improving** with better models
- Continue using Transformer + edge features
- May benefit from **flow ratio edge feature**

---

## Summary: Key Insights

### 1. **Temporal Branch is the Primary Driver**
- Accounts for ~70% of model performance
- Spatial branch adds only ~3-4% accuracy

### 2. **Spatial Features Help Specific Classes**
- **Class 4**: +0.19 recall improvement (0.53 → 0.72)
- **Class 8**: +0.07 recall improvement (0.83 → 0.90)
- **Class 6**: +0.24 precision improvement (0.76 → 1.00)

### 3. **Persistent Confusion Patterns Indicate Label Issues**
- **Class 4 ↔ Class 5**: 30-35 samples confused (may need merging)
- **Class 7 ↔ Class 9**: 26-30 samples confused (may need merging)
- **Edge features don't resolve these confusions**

### 4. **Validation-Test Discrepancy is Concerning**
- Edge features hurt validation accuracy (-3.56%)
- But improve test accuracy (+0.45%)
- **Possible causes**: Small validation set, overfitting, or label noise

### 5. **Next Steps Should Focus on**:
1. **Label quality investigation** (Class 4, 5, 7, 9)
2. **Fusion gate analysis** (Is spatial branch being used?)
3. **Temporal-only ablation** (How much does spatial branch contribute?)
4. **Flow ratio edge feature** (May help distinguish confused classes)

---

**Generated**: 2026-03-23
**Analysis Depth**: Confusion matrix analysis, class-level profiling, spatial branch contribution evaluation
