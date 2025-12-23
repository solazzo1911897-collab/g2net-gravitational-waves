import torch
from pathlib import Path

# Detect the project root (assumes this file is in src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- TOGGLE THIS TO 'True' FOR A FAST TEST RUN ---
DEBUG = True 
# -------------------------------------------------

class CFG:
    # Model Config
    seed = 42
    model_name = 'tf_efficientnet_b2_ns'
    img_size = 256
    batch_size = 64 if not DEBUG else 16  # Smaller batch for debug
    
    # Optimizer settings (These were missing!)
    lr = 1e-3
    weight_decay = 1e-4

    # Debug Logic: If DEBUG is True, run 2 epochs on 100 images.
    epochs = 2 if DEBUG else 8
    subset_size = 100 if DEBUG else 60000
    n_fold = 2 if DEBUG else 5  # Test only 2 folds in debug mode
    
    # CQT Parameters (Fixed)
    sr = 2048
    fmin = 20
    fmax = 1024
    hop_length = 32
    
    # Device
    # Note: On Mac, this defaults to 'cpu' unless you explicitly set 'mps'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 2 if not DEBUG else 0 # 0 is safer for debugging
    
    # Paths
    train_csv = DATA_DIR / "training_labels.csv"
    train_data_dir = DATA_DIR / "train"