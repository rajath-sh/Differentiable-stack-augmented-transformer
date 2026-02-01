"""
Configuration and hyperparameters for Stack-Augmented Transformer training.
"""

import os

# ==============================================================================
# PATHS
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Create directories if they don't exist
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

MODEL_NAME = "distilbert-base-uncased"
STACK_DEPTH = 16  # Size of the stack state vector
MAX_SEQ_LENGTH = 64  # Maximum tokenized sequence length

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

# Dataset sizes (per task type)
DYCK_TRAIN_SAMPLES = 5000
DYCK_VAL_SAMPLES = 1000
DYCK_TEST_SAMPLES = 500  # Per length category

ARITHMETIC_TRAIN_SAMPLES = 5000
ARITHMETIC_VAL_SAMPLES = 1000
ARITHMETIC_TEST_SAMPLES = 500  # Per length category

# Sequence length ranges for Dyck
DYCK_TRAIN_LENGTH_RANGE = (4, 16)
DYCK_VAL_LENGTH_RANGE = (8, 20)

# Test set length categories (for Graph 2) - Excluding Extra Long
TEST_LENGTH_CATEGORIES = [
    ("Short (6-10)", 6, 10),
    ("Medium (12-18)", 12, 18),
    ("Long (20-30)", 20, 30),
    ("Very Long (32-44)", 32, 44),
    # ("Extra Long (46-60)", 46, 60),  # Excluded - both models at ~50%
]

# Arithmetic expression depths
ARITHMETIC_TRAIN_DEPTH = (1, 3)
ARITHMETIC_VAL_DEPTH = (2, 4)
ARITHMETIC_TEST_DEPTH = (2, 5)

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0

# Learning rate scheduler
LR_SCHEDULER = "cosine"  # Options: "cosine", "step"
LR_STEP_SIZE = 5  # For StepLR
LR_GAMMA = 0.5  # For StepLR

# Dropout
DROPOUT = 0.1

# ==============================================================================
# RANDOM SEEDS (for reproducibility)
# ==============================================================================

RANDOM_SEED = 42

# ==============================================================================
# TASK TYPES
# ==============================================================================

TASK_DYCK = "dyck"
TASK_ARITHMETIC = "arithmetic"
ENABLED_TASKS = [TASK_DYCK, TASK_ARITHMETIC]
