"""
Configuration file for mobility pattern analysis model
"""
import os

# Data paths
DATA_DIR = "data"
OD_2021_PATH = os.path.join(DATA_DIR, "2021_week.csv")
OD_2024_PATH = os.path.join(DATA_DIR, "2024_week.csv")
GRID_METADATA_PATH = os.path.join(DATA_DIR, "grid_metadata", "PRD_grid_metadata.csv")
LABEL_PATH = os.path.join(DATA_DIR, "labels.csv")

# Output paths
OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
CHECKPOINT_DIR = "checkpoints"

# Data preprocessing parameters
TRAIN_DAYS = 7  # Use first 7 days for training
HOURS_PER_DAY = 24
TRAIN_HOURS = TRAIN_DAYS * HOURS_PER_DAY  # 168 hours (for old model)
TIME_STEPS = 168  # NEW: 168 hourly snapshots (hourly granularity)
AGGREGATION_METHOD = 'sum'  # Time aggregation method

# Feature parameters
TEMPORAL_INPUT_SIZE = 1  # [total_log] - only total flow for temporal branch
SPATIAL_INPUT_SIZE = 2   # [eccentricity, log_area] - ellipse features for spatial branch
NORMALIZATION = 'log'  # 'log', 'none', 'zscore'
USE_LOG_TRANSFORM = True

# Coordinate validation ranges
LON_RANGE = (-180, 180)
LAT_RANGE = (-90, 90)

# Label parameters
NUM_CLASSES = 9  # 9 types of mobility change patterns
LABEL_RANGE = (1, 9)

# Model architecture parameters
# Temporal branch (LSTM + SPP)
LSTM_LAYERS = 3
LSTM_HIDDEN_SIZE = 256
LSTM_DROPOUT = 0.3
SPP_LEVELS = [1, 2, 4]  # Spatial Pyramid Pooling levels (1x1, 2x2, 4x4)

# Dynamic graph branch (DySAT)
DYSAT_LAYERS = 3
DYSAT_HIDDEN_SIZE = 64
DYSAT_HEADS = 2  # Number of attention heads
DYSAT_DROPOUT = 0.2
TIME_WINDOW = 24  # 24-hour sliding window (for old model)

# Graph construction mode
USE_STATIC_GRAPH = True  # Use static aggregated graphs instead of dynamic
USE_FLOW_ONLY_GRAPH = True  # Use flow-only graphs, not hybrid k-NN

# Flow graph configuration
FLOW_THRESHOLD = 10.0  # Minimum flow to create edge
                       # Recommended values:
                       # - 5: Aggressive (~28,563 edges)
                       # - 10: Balanced (~10,956 edges) - RECOMMENDED
                       # - 20: Conservative (~3,553 edges)

# Spatial branch configuration
SPATIAL_BRANCH_TYPE = 'gat'  # 'dysat' or 'gat'
GAT_HIDDEN_SIZE = 128
GAT_LAYERS = 3
GAT_HEADS = 4

# Fusion layer
FUSION_HIDDEN_SIZE = 256
ATTENTION_HEADS = 4

# Training parameters
BATCH_SIZE = 16  # Reduced from 32 to avoid GPU OOM with ellipse features
LEARNING_RATE = 0.001
NUM_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
WEIGHT_DECAY = 1e-5

# Evaluation parameters
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Visualization parameters
FIGURE_DPI = 300
FIGURE_FORMAT = "jpg"
HEATMAP_CMAP = "YlOrRd"
