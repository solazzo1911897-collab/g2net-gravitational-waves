# --- 1. SILENCE WARNINGS (MUST BE AT THE VERY TOP) ---
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# -----------------------------------------------------

import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
# nnAudio imports must come AFTER warnings are silenced
from nnAudio.Spectrogram import CQT1992v2

# Plot Config
sns.set_style("white")
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.grid'] = False

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import CFG, OUTPUT_DIR
from src.dataset import G2NetDataset
from src.utils import set_seed

# ==============================================================================
# DATA: KAGGLE LOGS
# ==============================================================================
KAGGLE_LOGS = """
=== FOLD 1/5 ===
Ep 1/8 | Loss: 0.9840/0.5859 | AUC: 0.6083/0.7314
Ep 2/8 | Loss: 0.6372/0.5176 | AUC: 0.7343/0.7886
Ep 3/8 | Loss: 0.5610/0.4923 | AUC: 0.7794/0.8114
Ep 4/8 | Loss: 0.5218/0.4839 | AUC: 0.8038/0.8211
Ep 5/8 | Loss: 0.4910/0.4695 | AUC: 0.8243/0.8297
Ep 6/8 | Loss: 0.4728/0.4743 | AUC: 0.8376/0.8324
Ep 7/8 | Loss: 0.4481/0.4703 | AUC: 0.8549/0.8327
Ep 8/8 | Loss: 0.4210/0.4790 | AUC: 0.8741/0.8294
=== FOLD 2/5 ===
Ep 1/8 | Loss: 0.9512/0.5776 | AUC: 0.6215/0.7403
Ep 2/8 | Loss: 0.6304/0.5126 | AUC: 0.7356/0.7945
Ep 3/8 | Loss: 0.5541/0.4913 | AUC: 0.7807/0.8140
Ep 4/8 | Loss: 0.5120/0.4805 | AUC: 0.8087/0.8223
Ep 5/8 | Loss: 0.4854/0.4774 | AUC: 0.8270/0.8299
Ep 6/8 | Loss: 0.4623/0.4753 | AUC: 0.8438/0.8302
Ep 7/8 | Loss: 0.4425/0.4780 | AUC: 0.8577/0.8315
Ep 8/8 | Loss: 0.4219/0.4950 | AUC: 0.8724/0.8296
=== FOLD 3/5 ===
Ep 1/8 | Loss: 0.9642/0.5829 | AUC: 0.6110/0.7308
Ep 2/8 | Loss: 0.6355/0.5267 | AUC: 0.7321/0.7878
Ep 3/8 | Loss: 0.5624/0.5103 | AUC: 0.7775/0.8108
Ep 4/8 | Loss: 0.5199/0.4762 | AUC: 0.8049/0.8218
Ep 5/8 | Loss: 0.4951/0.4707 | AUC: 0.8220/0.8286
Ep 6/8 | Loss: 0.4683/0.4709 | AUC: 0.8394/0.8296
Ep 7/8 | Loss: 0.4465/0.4753 | AUC: 0.8556/0.8315
Ep 8/8 | Loss: 0.4251/0.5073 | AUC: 0.8718/0.8286
=== FOLD 4/5 ===
Ep 1/8 | Loss: 0.9544/0.6243 | AUC: 0.6083/0.7264
Ep 2/8 | Loss: 0.6292/0.5233 | AUC: 0.7319/0.7865
Ep 3/8 | Loss: 0.5513/0.4985 | AUC: 0.7811/0.8057
Ep 4/8 | Loss: 0.5098/0.4871 | AUC: 0.8089/0.8186
Ep 5/8 | Loss: 0.4850/0.4961 | AUC: 0.8269/0.8216
Ep 6/8 | Loss: 0.4623/0.4845 | AUC: 0.8434/0.8230
Ep 7/8 | Loss: 0.4395/0.4904 | AUC: 0.8597/0.8242
Ep 8/8 | Loss: 0.4198/0.5026 | AUC: 0.8736/0.8206
=== FOLD 5/5 ===
Ep 1/8 | Loss: 0.9628/0.6287 | AUC: 0.6211/0.7352
Ep 2/8 | Loss: 0.6291/0.5199 | AUC: 0.7385/0.7861
Ep 3/8 | Loss: 0.5534/0.5147 | AUC: 0.7817/0.8077
Ep 4/8 | Loss: 0.5168/0.4855 | AUC: 0.8072/0.8151
Ep 5/8 | Loss: 0.4865/0.4885 | AUC: 0.8265/0.8215
Ep 6/8 | Loss: 0.4628/0.5019 | AUC: 0.8433/0.8251
Ep 7/8 | Loss: 0.4457/0.4856 | AUC: 0.8554/0.8264
Ep 8/8 | Loss: 0.4211/0.4932 | AUC: 0.8736/0.8256
"""

def parse_logs(log_text):
    data = []
    current_fold = {}
    for line in log_text.strip().split('\n'):
        line = line.strip()
        if not line: continue
        if "=== FOLD" in line:
            if current_fold: data.append(current_fold)
            current_fold = {'t_loss': [], 'v_loss': [], 't_auc': [], 'v_auc': []}
        elif "Loss:" in line:
            parts = line.split('|')
            loss_part = parts[1].split(':')[1].strip().split('/')
            auc_part = parts[2].split(':')[1].strip().split('/')
            current_fold['t_loss'].append(float(loss_part[0]))
            current_fold['v_loss'].append(float(loss_part[1]))
            current_fold['t_auc'].append(float(auc_part[0]))
            current_fold['v_auc'].append(float(auc_part[1]))
    if current_fold: data.append(current_fold)
    return data

# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_grid_results(histories):
    n_folds = len(histories)
    epochs = range(1, len(histories[0]['t_loss']) + 1)
    colors = sns.color_palette("husl", n_folds)
    fig, axes = plt.subplots(2, n_folds, figsize=(20, 8), sharex=True)
    fig.text(0.125, 1.02, 'Results per fold (Kaggle Run)', fontsize=28, fontweight='bold', ha='left')
    
    for i in range(n_folds):
        h = histories[i]
        c = colors[i]
        ax_l = axes[0, i]
        ax_l.plot(epochs, h['t_loss'], label='Train', color=c, marker='o', lw=2)
        ax_l.plot(epochs, h['v_loss'], label='Valid', color='gray', marker='x', ls='--', lw=1.5)
        ax_l.set_title(f'Loss: Fold {i+1}', fontsize=14, fontweight='bold')
        if i == 0: ax_l.set_ylabel('BCE Loss', fontsize=12)
        if i == n_folds - 1: ax_l.legend(loc='upper right')

        ax_a = axes[1, i]
        ax_a.plot(epochs, h['t_auc'], label='Train AUC', color=c, marker='o', lw=2)
        ax_a.plot(epochs, h['v_auc'], label='Valid AUC', color='dimgray', marker='s', lw=2)
        if i == 0: ax_a.set_ylabel('AUC Score', fontsize=12)
        ax_a.set_xlabel('Epoch', fontsize=12)
        if i == n_folds - 1: ax_a.legend(loc='lower right')
        
    plt.tight_layout()
    save_path = OUTPUT_DIR / "plots" / "kaggle_training_grid.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path.name}")

def plot_candidates_hd(device):
    # This requires the local dataset to be present
    try:
        df = pd.read_csv(CFG.train_csv)
        ds = G2NetDataset(df)
    except Exception as e:
        print(f"[WARN] Could not load dataset: {e}")
        return

    target_ids = ['9c13c328bf', '3329ef4849']
    found_indices = [i for i, fname in enumerate(ds.file_names) if fname in target_ids]
    
    if len(found_indices) < 2:
        print("[WARN] Specific Kaggle candidate IDs not found in local dataset. Skipping spectrogram.")
        return

    print("[INFO] Generating HD Spectrograms for candidates...")
    cqt_layer = CQT1992v2(sr=2048, fmin=20, fmax=500, hop_length=32, 
                          bins_per_octave=12, output_format="Magnitude", 
                          verbose=False).to(device)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    plt.subplots_adjust(right=0.9, top=0.85)

    for i, idx in enumerate(found_indices):
        wave, _ = ds[idx]
        file_id = ds.file_names[idx]
        wave_tensor = wave.unsqueeze(0).to(device)
        with torch.no_grad():
            spec = cqt_layer(wave_tensor.view(1*3, -1))
            spec_np = spec.cpu().numpy()
            img = spec_np[0] + spec_np[1]
            img = np.log1p(img)
            
            # --- Robust Normalization Fix ---
            img = np.nan_to_num(img) # Fix any NaNs
            img = img - np.median(img, axis=1, keepdims=True)
            
            # Avoid div/0 if image is flat
            min_val = np.min(img)
            max_val = np.max(img)
            if (max_val - min_val) > 1e-6:
                img = (img - min_val) / (max_val - min_val)
            else:
                img = np.zeros_like(img)
            
            img[img < 0.60] = 0

            ax = axes[i]
            im = ax.imshow(img, aspect='auto', origin='lower', cmap='inferno', interpolation='bicubic')
            ax.set_title(f"ID: {file_id}\nSource: LIGO Hanford + Livingston", fontsize=16, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized Amplitude (Threshold > 0.60)', fontsize=12)
    plt.suptitle("Gravitational Wave Signal Reconstruction (Kaggle Best)", fontsize=22)
    
    save_path = OUTPUT_DIR / "plots" / "kaggle_spectrogram.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved {save_path.name}")

def plot_roc_and_cm(csv_path):
    if not csv_path.exists():
        print(f"[ERROR] Kaggle CSV not found at {csv_path}. Please rename your OOF file to 'kaggle_oof_predictions_b2.csv'.")
        return
        
    df = pd.read_csv(csv_path)
    y_true = df['target']
    y_pred = df['pred_b2']
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    score = roc_auc_score(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color='#8B0000', lw=3, label=f'AUC = {score:.4f}')
    plt.fill_between(fpr, tpr, color='#8B0000', alpha=0.1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve (Global AUC: {score:.5f})', fontsize=20, fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(OUTPUT_DIR / "plots" / "kaggle_roc_curve.png")
    print("Saved kaggle_roc_curve.png")
    
    # CM
    y_pred_bin = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    plt.figure(figsize=(8, 7.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={"size": 14, "weight": "bold"})
    plt.title('Global Confusion Matrix (Kaggle)', fontsize=18, fontweight='bold')
    
    plt.subplots_adjust(bottom=0.25)
    stats_text = (f"Accuracy: {accuracy:.4f}\n"
                  f"Sensitivity: {sensitivity:.4f}\n"
                  f"Specificity: {specificity:.4f}")
    plt.figtext(0.5, 0.08, stats_text, ha="center", fontsize=14, 
                bbox={"facecolor":"orange", "alpha":0.1, "pad":10, "edgecolor":"orange"})
    plt.savefig(OUTPUT_DIR / "plots" / "kaggle_confusion_matrix.png")
    print("Saved kaggle_confusion_matrix.png")

if __name__ == "__main__":
    print("[INFO] Processing Kaggle Logs for Demonstration...")
    history = parse_logs(KAGGLE_LOGS)
    plot_grid_results(history)
    plot_candidates_hd(CFG.device)
    kaggle_csv = OUTPUT_DIR / "logs" / "kaggle_oof_predictions_b2.csv"
    plot_roc_and_cm(kaggle_csv)