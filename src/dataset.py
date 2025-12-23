import torch
import numpy as np
from scipy import signal
from torch.utils.data import Dataset
from .config import CFG

def apply_bandpass(x, lf=20, hf=500, order=4, sr=2048):
    """Whitening filter to remove low-frequency noise floor."""
    # Safety check for NaN in input
    if np.isnan(x).any():
        x = np.nan_to_num(x)
        
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt(1/2048)
    filtered = signal.sosfiltfilt(sos, x) * normalization
    return filtered

class G2NetDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.file_names = df['id'].values
        self.labels = df['target'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_id = self.file_names[idx]
        # Construct path: data/raw/train/0/1/2/012345.npy
        file_path = CFG.train_data_dir / file_id[0] / file_id[1] / file_id[2] / f"{file_id}.npy"
        
        try:
            waves = np.load(file_path).astype(np.float32)
        except FileNotFoundError:
            # Create a dummy silent wave if file is missing (prevents crash)
            waves = np.zeros((3, 4096), dtype=np.float32)
        
        # Bandpass Filter
        for i in range(3):
            waves[i] = apply_bandpass(waves[i])
            
        # Normalization (Robust against division by zero)
        max_val = np.max(np.abs(waves), axis=1, keepdims=True)
        # If max_val is 0, replace with 1 to avoid NaN. Result will be 0/1 = 0.
        max_val[max_val == 0] = 1.0 
        waves = waves / max_val
        
        return torch.tensor(waves, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)