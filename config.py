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
FLOW_THRESHOLD = 10.0  # OPTIMAL VALUE - extensively tested [5.0, 7.5, 9.0, 10.0, 11.0, 12.0], 10.0 is best with 65.74% accuracy

# ==============================================================================
# GRAPH OPTIMIZATION - ISOLATED NODE FIXES
# ==============================================================================
# Phase 2.1: Self-loop addition
ADD_SELF_LOOPS = True  # Add self-loops to all nodes (prevents isolated nodes)
SELF_LOOP_WEIGHT = 1.0  # Weight for self-loops

# Phase 2.2: KNN fallback for isolated nodes
USE_KNN_FALLBACK = True  # Add spatial KNN edges for isolated nodes only
KNN_FALLBACK_K = 8  # Number of nearest neighbors for isolated nodes
KNN_FALLBACK_WEIGHT = 0.5  # Weight multiplier for KNN fallback edges

# Phase 3: Hybrid graph (use only if Phase 2 insufficient)
USE_HYBRID_GRAPH = False  # Use hybrid graph (flow + spatial) - WARNING: memory intensive
HYBRID_SPATIAL_WEIGHT = 0.3  # Weight for spatial edges in hybrid graph
HYBRID_FLOW_WEIGHT = 0.7  # Weight for flow edges in hybrid graph

# Memory optimization (for Phase 3)
MAX_EDGES_PER_NODE = 50  # Maximum average degree per node
MAX_TOTAL_EDGES = 200000  # Hard limit on total edges

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
TEMPORAL_INPUT_SIZE = 1  # Input feature dimension: [total_log]
LSTM_LAYERS = 3          # Number of LSTM layers
LSTM_HIDDEN_SIZE = 256   # LSTM hidden units
LSTM_DROPOUT = 0.2       # LSTM dropout rate

# ==============================================================================
# MODEL ARCHITECTURE - SPATIAL BRANCH
# ==============================================================================
# Pure Graph GAT Branch
GAT_LAYERS = 3           # Number of GAT layers
GAT_HIDDEN_SIZE = 128    # GAT hidden units per head
GAT_HEADS = 4            # Number of attention heads

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
BATCH_SIZE = 16           # Batch size for training
LEARNING_RATE = 0.0001    # Initial learning rate
WEIGHT_DECAY = 5e-4       # L2 regularization
NUM_EPOCHS = 300          # Maximum number of training epochs
EARLY_STOPPING_PATIENCE = 40  # Stop if no improvement for N epochs

# ==============================================================================
# CONFIGURATION SUMMARY
# ==============================================================================
"""
Total parameters: 22

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
19. GAT_LAYERS          - Spatial branch architecture
20. GAT_HIDDEN_SIZE
21. GAT_HEADS
22. ATTENTION_HEADS     - Fusion layer architecture

Paths used by dual_year_processor.py:
- OD_2021_PATH
- OD_2024_PATH
- GRID_METADATA_PATH
"""
