import sys
import warnings
import os

# --- 1. SILENCE ALL WARNINGS (Must be first) ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# -----------------------------------------------

from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.cuda.amp as amp # Not used explicitly in new torch versions
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add src to python path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import CFG, OUTPUT_DIR
from src.utils import set_seed
from src.dataset import G2NetDataset
from src.model import G2NetModel

def train_epoch(train_loader, model, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0
    preds_list = []
    labels_list = []
    
    # Clean progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False, bar_format='{l_bar}{bar:10}{r_bar}')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # CPU/MPS friendly AMP check
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision for CPU/MPS (avoids warnings)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        preds_list.append(torch.sigmoid(outputs).detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    all_preds = np.concatenate(preds_list)
    all_labels = np.concatenate(labels_list)
    return running_loss/len(train_loader), roc_auc_score(all_labels, all_preds)

def validate_epoch(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds_list.append(torch.sigmoid(outputs).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(preds_list)
    all_labels = np.concatenate(labels_list)
    return running_loss/len(valid_loader), roc_auc_score(all_labels, all_preds), all_preds, all_labels

if __name__ == '__main__':
    # --- DATA AVAILABILITY CHECK ---
    if not CFG.train_data_dir.exists() or not any(CFG.train_data_dir.iterdir()):
        print("\n[ERROR] Data not found in 'data/raw/train/'.")
        print("Please download the dataset from the Google Drive link in the README.")
        print("Extract the 'train' folder so it sits inside 'data/raw/'.\n")
        sys.exit(1)
    # -------------------------------

    # Ensure outputs exist
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)

    set_seed(CFG.seed)
    print(f"[INFO] Local Training: EfficientNet-B2 | Epochs: {CFG.epochs} | Device: {CFG.device}")
    
    df = pd.read_csv(CFG.train_csv)
    df_subset = df.sample(n=CFG.subset_size, random_state=CFG.seed).reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    oof_df = df_subset.copy()
    oof_df['pred_b2'] = 0.0
    
    history_data = []

    # Check if we should pin memory (Only True if using CUDA)
    use_pin_memory = (CFG.device.type == 'cuda')

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_subset, df_subset['target'])):
        print(f"\n=== FOLD {fold+1}/{CFG.n_fold} ===")
        
        train_ds = G2NetDataset(df_subset.iloc[train_idx].reset_index(drop=True))
        valid_ds = G2NetDataset(df_subset.iloc[val_idx].reset_index(drop=True))
        
        # FIXED: Dynamic pin_memory
        train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, 
                                num_workers=CFG.num_workers, pin_memory=use_pin_memory)
        valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False, 
                                num_workers=CFG.num_workers, pin_memory=use_pin_memory)
        
        model = G2NetModel().to(CFG.device)
        optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        
        # Scaler is only needed for CUDA
        scaler = torch.amp.GradScaler('cuda', enabled=(CFG.device.type == 'cuda'))
        
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr, 
                                                steps_per_epoch=len(train_loader), epochs=CFG.epochs)
        
        fold_history = {'t_loss': [], 'v_loss': [], 't_auc': [], 'v_auc': []}
        best_auc = 0
        best_preds = None
        
        for epoch in range(CFG.epochs):
            t_loss, t_auc = train_epoch(train_loader, model, nn.BCEWithLogitsLoss(), optimizer, scaler, CFG.device)
            v_loss, v_auc, v_preds, _ = validate_epoch(valid_loader, model, nn.BCEWithLogitsLoss(), CFG.device)
            
            fold_history['t_loss'].append(t_loss)
            fold_history['v_loss'].append(v_loss)
            fold_history['t_auc'].append(t_auc)
            fold_history['v_auc'].append(v_auc)
            
            print(f"Ep {epoch+1}/{CFG.epochs} | Loss: {t_loss:.4f}/{v_loss:.4f} | AUC: {t_auc:.4f}/{v_auc:.4f}")
            
            if v_auc > best_auc:
                best_auc = v_auc
                best_preds = v_preds
            scheduler.step()
            
        oof_df.loc[val_idx, 'pred_b2'] = best_preds
        history_data.append(fold_history)
        
        # Optional: Save Model
        torch.save(model.state_dict(), OUTPUT_DIR / "models" / f"model_fold_{fold+1}.pth")

    print("\n[INFO] Saving Results...")
    oof_df.to_csv(OUTPUT_DIR / "logs" / "oof_predictions_b2.csv", index=False)
    with open(OUTPUT_DIR / "logs" / "training_history.json", 'w') as f:
        json.dump(history_data, f)
    
    print("Done. Run scripts/visualize.py to generate plots.")