"""
Configuration file for mobility pattern analysis model
"""
import os

# Data paths
DATA_DIR = "data"
OD_2021_PATH = os.path.join(DATA_DIR, "2021.csv")
OD_2024_PATH = os.path.join(DATA_DIR, "2024.csv")
GRID_METADATA_PATH = os.path.join(DATA_DIR, "grid_metadata", "PRD_grid_metadata.csv")
LABEL_PATH = os.path.join(DATA_DIR, "labels_1w_20251228.csv")

# Output paths
OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
CHECKPOINT_DIR = "checkpoints"

# Data preprocessing parameters
TRAIN_DAYS = 7  # Use first 7 days (168 hours) for training
HOURS_PER_DAY = 24
TRAIN_HOURS = TRAIN_DAYS * HOURS_PER_DAY

# Coordinate validation ranges
LON_RANGE = (-180, 180)
LAT_RANGE = (-90, 90)

# Label parameters
NUM_CLASSES = 9  # 9 types of mobility change patterns
LABEL_RANGE = (1, 9)

# Model architecture parameters
# Temporal branch (LSTM + SPP)
LSTM_LAYERS = 2
LSTM_HIDDEN_SIZE = 128
LSTM_DROPOUT = 0.2
SPP_LEVELS = [1, 2, 4]  # Spatial Pyramid Pooling levels (1x1, 2x2, 4x4)

# Dynamic graph branch (DySAT)
DYSAT_LAYERS = 3
DYSAT_HIDDEN_SIZE = 128
DYSAT_HEADS = 4  # Number of attention heads
DYSAT_DROPOUT = 0.2
TIME_WINDOW = 24  # 24-hour sliding window

# Fusion layer
FUSION_HIDDEN_SIZE = 256
ATTENTION_HEADS = 4

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-5

# Evaluation parameters
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Visualization parameters
FIGURE_DPI = 300
FIGURE_FORMAT = "jpg"
HEATMAP_CMAP = "YlOrRd"
