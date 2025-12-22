import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
from .config import CFG

def apply_bandpass(x, lf=20, hf=500, order=4, sr=2048):
    """
    Applies a Butterworth bandpass filter to whiten the signal data.
    Removes low-frequency noise floor and high-frequency artifacts.
    """
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt(1 / sr)
    return signal.sosfiltfilt(sos, x) * normalization

class G2NetDataset(Dataset):
    """
    Custom Dataset class for G2Net Gravitational Wave Detection.
    Loads raw .npy files, applies signal processing, and returns tensors.
    """
    def __init__(self, df):
        self.df = df
        self.file_names = df['id'].values
        self.labels = df['target'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_id = self.file_names[idx]
        # Construct the file path: data/raw/train/0/1/2/012abc.npy
        file_path = os.path.join(
            CFG.DATA_ROOT, 
            file_id[0], file_id[1], file_id[2], 
            f"{file_id}.npy"
        )
        
        try:
            # Load raw gravitational wave data (3 sensors)
            waves = np.load(file_path).astype(np.float32)
            
            # Apply bandpass filter to each channel
            for i in range(3):
                waves[i] = apply_bandpass(waves[i])
            
            # Global normalization (max absolute scaling)
            waves = waves / (np.max(np.abs(waves), axis=1, keepdims=True) + 1e-7)
            
        except FileNotFoundError:
            # Fallback for missing files during local testing
            print(f"[WARNING] File not found: {file_path}. Returning zero tensor.")
            waves = np.zeros((3, 4096), dtype=np.float32)
            
        return (
            torch.tensor(waves, dtype=torch.float32), 
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )