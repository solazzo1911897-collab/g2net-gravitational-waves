import os
import torch

class CFG:
    """
    Configuration class for G2Net Gravitational Wave Detection.
    Handles hyperparameters, file paths, and hardware settings.
    """
    
    # --- General Setup ---
    seed = 42
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Model & Training Hyperparameters ---
    model_name = 'tf_efficientnet_b2_ns'
    img_size = 256
    batch_size = 64
    epochs = 8
    subset_size = 60000
    lr = 1e-3
    weight_decay = 1e-4
    n_fold = 5
    
    # --- Audio / CQT Parameters ---
    sr = 2048           # Sample rate
    fmin = 20           # Min frequency
    fmax = 1024         # Max frequency
    hop_length = 32     # Hop length for CQT
    bins_per_octave = 12
    
    # --- Dynamic Path Configuration ---
    # Locates the project root directory relative to this config file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input Data Paths
    DATA_ROOT = os.path.join(BASE_DIR, "data", "raw", "train")
    LABELS_CSV = os.path.join(BASE_DIR, "data", "raw", "training_labels.csv")
    
    # Processed Data Paths (Generated during runtime)
    SUBSET_CSV = os.path.join(BASE_DIR, "data", "processed", "train_subset_60k.csv")
    
    # Output Paths
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Ensure critical output directories exist
os.makedirs(CFG.LOGS_DIR, exist_ok=True)
os.makedirs(CFG.PLOTS_DIR, exist_ok=True)
os.makedirs(CFG.MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CFG.SUBSET_CSV), exist_ok=True)