import sys
import os
import torch
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Suppress library warnings for a cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import CFG
from src.dataset import G2NetDataset
from src.model import G2NetModel

def quick_check():
    print(f"\n[INFO] Python Version: {sys.version.split()[0]}")
    print(f"[INFO] PyTorch Version: {torch.__version__}")
    print(f"[INFO] Device: {CFG.device}")
    
    print("\n[1/3] Checking File System...")
    if os.path.exists(CFG.LABELS_CSV) and os.path.exists(CFG.DATA_ROOT):
        print("✅ Data directories found.")
    else:
        print(f"❌ Data MISSING. Please check {CFG.DATA_ROOT}")
        return

    print("\n[2/3] Testing Data Loading...")
    try:
        # Load first 10 rows for a quick smoke test
        df = pd.read_csv(CFG.LABELS_CSV).head(10)
        ds = G2NetDataset(df)
        loader = DataLoader(ds, batch_size=2)
        
        # Load a batch (Note: Missing files in the subset will print a warning, this is expected)
        waves, targets = next(iter(loader))
        print(f"✅ Batch loaded successfully. Shape: {waves.shape}")
    except Exception as e:
        print(f"❌ Data Loading Error: {e}")
        return

    print("\n[3/3] Testing Model Architecture...")
    try:
        # Initialize model (without pretrained weights for speed)
        model = G2NetModel(CFG, pretrained=False).to(CFG.device)
        waves = waves.to(CFG.device)
        
        # Forward pass
        output = model(waves)
        print(f"✅ Model forward pass successful. Output Shape: {output.shape}")
    except Exception as e:
        print(f"❌ Model Error: {e}")
        return

    print("\n🚀 SYSTEM READY. All systems are functioning correctly.")

if __name__ == "__main__":
    quick_check()