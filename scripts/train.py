import sys
import os
import json
import warnings

# --- SILENCER BLOCK (PULIZIA TERMINALE) ---
# Ignora tutti i warning non critici per avere un output pulito
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", module="nnAudio")
warnings.filterwarnings("ignore", module="torch")
# ------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add project root to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import CFG
from src.dataset import G2NetDataset
from src.model import G2NetModel

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Executes one training epoch with mixed precision.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    
    # Progress bar for training
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for waves, targets in pbar:
        waves = waves.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward Pass
        # Su Mac CPU questo contesto non fa molto, ma lo lasciamo per compatibilità codice
        with amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(waves).squeeze(1)
            loss = criterion(outputs, targets)
        
        # Backward Pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Store metrics
        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    epoch_loss = running_loss / len(loader)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5 # Fallback se batch troppo piccolo o classe unica
    
    return epoch_loss, epoch_auc

def valid_one_epoch(model, loader, criterion, device):
    """
    Executes validation phase (inference only).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for waves, targets in tqdm(loader, desc="Validation", leave=False):
            waves = waves.to(device)
            targets = targets.to(device)
            
            outputs = model(waves).squeeze(1)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5
    
    return epoch_loss, epoch_auc, np.array(all_preds)

def main():
    # Force clean output
    print(f"\n[INFO] Starting Training Pipeline on device: {CFG.device}")
    
    # 1. Prepare Data Subset if not exists
    if not os.path.exists(CFG.SUBSET_CSV):
        print(f"[INFO] Generating subset CSV ({CFG.subset_size} samples)...")
        if not os.path.exists(CFG.LABELS_CSV):
            print(f"[ERROR] Labels file not found at {CFG.LABELS_CSV}")
            return
            
        df = pd.read_csv(CFG.LABELS_CSV)
        # Random sample with fixed seed for reproducibility
        df_subset = df.sample(n=CFG.subset_size, random_state=CFG.seed).reset_index(drop=True)
        df_subset.to_csv(CFG.SUBSET_CSV, index=False)
    else:
        df_subset = pd.read_csv(CFG.SUBSET_CSV)
    
    # 2. Cross-Validation Setup
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    full_history = []
    
    # Eseguiamo solo il primo fold per il test
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_subset, df_subset['target'])):
        print(f"\n{'='*20} FOLD {fold+1}/{CFG.n_fold} {'='*20}")
        
        # Data Loaders
        train_ds = G2NetDataset(df_subset.iloc[train_idx].reset_index(drop=True))
        valid_ds = G2NetDataset(df_subset.iloc[val_idx].reset_index(drop=True))
        
        # NOTE: pin_memory=False removes the MPS warning on Mac
        train_loader = DataLoader(
            train_ds, 
            batch_size=CFG.batch_size, 
            shuffle=True, 
            num_workers=CFG.num_workers,
            pin_memory=False 
        )
        valid_loader = DataLoader(
            valid_ds, 
            batch_size=CFG.batch_size, 
            shuffle=False, 
            num_workers=CFG.num_workers,
            pin_memory=False
        )
        
        # Model Initialization
        model = G2NetModel(CFG).to(CFG.device)
        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        scaler = amp.GradScaler(enabled=(CFG.device.type == 'cuda'))
        
        fold_history = {'t_loss': [], 'v_loss': [], 't_auc': [], 'v_auc': []}
        
        # Epoch Loop
        for epoch in range(CFG.epochs):
            t_loss, t_auc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, CFG.device)
            v_loss, v_auc, _ = valid_one_epoch(model, valid_loader, criterion, CFG.device)
            
            print(f"Epoch {epoch+1}/{CFG.epochs} | "
                  f"Train Loss: {t_loss:.4f} AUC: {t_auc:.4f} | "
                  f"Valid Loss: {v_loss:.4f} AUC: {v_auc:.4f}")
            
            # Record metrics
            fold_history['t_loss'].append(t_loss)
            fold_history['v_loss'].append(v_loss)
            fold_history['t_auc'].append(t_auc)
            fold_history['v_auc'].append(v_auc)
            
        full_history.append(fold_history)
        
        # Save logs checkpoint
        with open(os.path.join(CFG.LOGS_DIR, "training_history.json"), "w") as f:
            json.dump(full_history, f, indent=4)
            
    print("\n[SUCCESS] Training complete. Logs saved.")

if __name__ == "__main__":
    main()