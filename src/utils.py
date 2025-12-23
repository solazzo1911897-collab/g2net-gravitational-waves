import os
import random
import numpy as np
import torch
import datetime

def set_seed(seed=42):
    """Sets the seed for the entire environment to ensure reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[INFO] Seed set to {seed}")

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))