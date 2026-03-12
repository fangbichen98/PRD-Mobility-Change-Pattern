"""
Configuration file for mobility pattern analysis model
Multi-Scale Temporal Branch Training

This config file contains ONLY the parameters used in train_multiscale_temporal.py
Last updated: 2026-03-05
"""
import os

# ==============================================================================
# DATA PATHS
# ==============================================================================
# These paths are used by dual_year_processor.py (called by train_multiscale_temporal.py)
DATA_DIR = "data"
OD_2021_PATH = os.path.join(DATA_DIR, "2021_sgh_week.csv")
OD_2024_PATH = os.path.join(DATA_DIR, "2024_sgh_week.csv")
GRID_METADATA_PATH = os.path.join(DATA_DIR, "grid_metadata", "sgh_grid_metadata.csv")
LABEL_PATH = os.path.join(DATA_DIR, "label_sgh.csv")

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================
TRAIN_DAYS = 7  # Use first 7 days for training
TIME_STEPS = 168  # 168 hourly snapshots (7 days × 24 hours)
FLOW_THRESHOLD = 0.0  # OPTIMAL VALUE - extensively tested [5.0, 7.5, 9.0, 10.0, 11.0, 12.0], 10.0 is best with 65.74% accuracy
SPP_LEVELS = [1, 2, 4]  # Spatial pyramid pooling levels for temporal branch

# Graph construction parameters (used by graph_builder.py)
ADD_SELF_LOOPS = True  # Add self-loops to graph (helps with isolated nodes)
SELF_LOOP_WEIGHT = 1.0  # Weight for self-loops
USE_KNN_FALLBACK = True  # Use KNN fallback for isolated nodes (based on spatial coordinates)
KNN_FALLBACK_K = 8  # Number of nearest neighbors for KNN fallback
KNN_FALLBACK_WEIGHT = 0.5  # Weight for KNN fallback edges (lower priority than flow edges)

# Coordinate validation ranges (used by data_processor.py)
LON_RANGE = (-180, 180)
LAT_RANGE = (-90, 90)

# ==============================================================================
# DATA SPLIT
# ==============================================================================
TRAIN_SPLIT = 0.7  # Training set ratio
VAL_SPLIT = 0.1    # Validation set ratio
TEST_SPLIT = 0.2   # Test set ratio
RANDOM_SEED = 42   # Random seed for reproducibility

# ==============================================================================
# MODEL ARCHITECTURE - TEMPORAL BRANCH
# ==============================================================================
# Multi-Scale Temporal Branch (LSTM-based)
TEMPORAL_INPUT_SIZE = 2  # Input feature dimension: [total_log]
LSTM_LAYERS = 3          # Number of LSTM layers
LSTM_HIDDEN_SIZE = 256   # LSTM hidden units
LSTM_DROPOUT = 0.4       # LSTM dropout rate

# ==============================================================================
# MODEL ARCHITECTURE - SPATIAL BRANCH
# ==============================================================================
# Spatial Branch Options: GCN, GraphSAGE, or GINE
SPATIAL_MODEL = "GCN"        # Options: "GCN", "SAGE", "GINE"
SPATIAL_LAYERS = 3           # Number of spatial layers
SPATIAL_HIDDEN_SIZE = 128    # Spatial branch hidden units
LAPLACIAN_PE_DIM = 16        # Laplacian Positional Encoding dimension (for GINE)
# Note: GCN doesn't use attention heads (unlike GAT)
# Legacy GAT_* parameters are kept for backward compatibility
SPATIAL_HEADS = 4            # Deprecated: Not used by GCN (only for GAT)

# ==============================================================================
# MODEL ARCHITECTURE - FUSION & CLASSIFICATION
# ==============================================================================
# Gated Feature Fusion
FUSION_HIDDEN_SIZE = 256  # Fusion layer hidden size
ATTENTION_HEADS = 4       # Number of attention heads in fusion
NUM_CLASSES = 9           # Number of output classes (9 mobility patterns)

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 24          # Batch size for training
LEARNING_RATE = 0.0001    # Initial learning rate
WEIGHT_DECAY = 1e-3       # L2 regularization
NUM_EPOCHS = 300          # Maximum number of training epochs
EARLY_STOPPING_PATIENCE =20  # Stop if no improvement for N epochs

# ==============================================================================
# CONFIGURATION SUMMARY
# ==============================================================================
"""
Total parameters: 30 (3 new SPATIAL_* params, 3 deprecated GAT_* params)

Used in train_multiscale_temporal.py:
1. LABEL_PATH           - Label file path
2. TIME_STEPS           - Temporal sequence length
3. FLOW_THRESHOLD       - Graph construction threshold
4. TRAIN_SPLIT          - Train/validation/test split
5. VAL_SPLIT
6. TEST_SPLIT
7. RANDOM_SEED
8. BATCH_SIZE           - Training batch size
9. LEARNING_RATE        - Optimizer learning rate
10. WEIGHT_DECAY        - L2 regularization
11. NUM_EPOCHS          - Max training epochs
12. EARLY_STOPPING_PATIENCE
13. TEMPORAL_INPUT_SIZE - Input feature dimension
14. FUSION_HIDDEN_SIZE  - Fusion layer size
15. NUM_CLASSES         - Output classes
16. LSTM_LAYERS         - Temporal branch architecture
17. LSTM_HIDDEN_SIZE
18. LSTM_DROPOUT
19. SPATIAL_LAYERS      - Spatial branch architecture (GCN)
20. SPATIAL_HIDDEN_SIZE
21. GAT_LAYERS          - Deprecated: Use SPATIAL_LAYERS instead
22. GAT_HIDDEN_SIZE     - Deprecated: Use SPATIAL_HIDDEN_SIZE instead
23. GAT_HEADS           - Deprecated: Not used by GCN
24. ATTENTION_HEADS     - Fusion layer architecture
25. ADD_SELF_LOOPS      - Graph construction (self-loops for isolated nodes)
26. SELF_LOOP_WEIGHT    - Weight for self-loop edges
27. USE_KNN_FALLBACK    - Use KNN fallback for isolated nodes
28. KNN_FALLBACK_K      - Number of neighbors for KNN fallback
29. KNN_FALLBACK_WEIGHT - Weight for KNN fallback edges

Paths used by dual_year_processor.py:
- OD_2021_PATH
- OD_2024_PATH
- GRID_METADATA_PATH
"""
