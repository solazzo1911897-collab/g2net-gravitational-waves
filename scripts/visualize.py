import sys
import warnings
import os

# --- SILENCE ALL WARNINGS (Must be first) ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# --------------------------------------------

import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from nnAudio.Spectrogram import CQT1992v2

sys.path.append(str(Path(__file__).parent.parent))
from src.config import CFG, OUTPUT_DIR
from src.dataset import G2NetDataset
from src.utils import set_seed

# Plot Config
sns.set_style("white")
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.grid'] = False

def plot_grid_results(histories):
    n_folds = len(histories)
    epochs = range(1, len(histories[0]['t_loss']) + 1)
    colors = sns.color_palette("husl", n_folds)
    fig, axes = plt.subplots(2, n_folds, figsize=(20, 8), sharex=True)
    fig.text(0.125, 1.02, 'Results per fold', fontsize=28, fontweight='bold', ha='left')
    
    for i in range(n_folds):
        h = histories[i]
        c = colors[i]
        ax_l = axes[0, i]
        ax_l.plot(epochs, h['t_loss'], label='Train', color=c, marker='o', lw=2)
        ax_l.plot(epochs, h['v_loss'], label='Valid', color='gray', marker='x', ls='--', lw=1.5)
        ax_l.set_title(f'Loss: Fold {i+1}', fontsize=14, fontweight='bold')
        if i == 0: ax_l.set_ylabel('BCE Loss', fontsize=12)
        
        ax_a = axes[1, i]
        ax_a.plot(epochs, h['t_auc'], label='Train AUC', color=c, marker='o', lw=2)
        ax_a.plot(epochs, h['v_auc'], label='Valid AUC', color='dimgray', marker='s', lw=2)
        if i == 0: ax_a.set_ylabel('AUC Score', fontsize=12)
        ax_a.set_xlabel('Epoch', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "training_grid.png", bbox_inches='tight')
    print("Saved training_grid.png")

def plot_candidates_hd(device):
    # To reproduce this, we need to load the dataset again
    df = pd.read_csv(CFG.train_csv)
    # We don't need the split here, just finding the file in the full 60k is enough
    # But to be safe on paths, let's just use the full dataset class
    ds = G2NetDataset(df)
    
    target_ids = ['9c13c328bf', '3329ef4849']
    found_indices = [i for i, fname in enumerate(ds.file_names) if fname in target_ids]
    
    if len(found_indices) < 2:
        print("[WARN] Could not find specific target IDs for spectrogram plot (Data subset might be too small).")
        return

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
            img = img - np.median(img, axis=1, keepdims=True)
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
            img[img < 0.60] = 0

            ax = axes[i]
            im = ax.imshow(img, aspect='auto', origin='lower', cmap='inferno', interpolation='bicubic')
            ax.set_title(f"ID: {file_id}\nSource: LIGO Hanford + Livingston", fontsize=16, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized Amplitude (Threshold > 0.60)', fontsize=12)
    plt.suptitle("Gravitational Wave Signal Reconstruction", fontsize=22)
    plt.savefig(OUTPUT_DIR / "plots" / "spectrogram.png", bbox_inches='tight')
    print("Saved spectrogram.png")

def plot_roc_and_cm(csv_path):
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
    plt.title('ROC Curve', fontsize=20, fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(OUTPUT_DIR / "plots" / "roc_curve.png")
    print("Saved roc_curve.png")
    
    # CM
    y_pred_bin = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)
    plt.figure(figsize=(8, 7.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Global Confusion Matrix', fontsize=18, fontweight='bold')
    plt.savefig(OUTPUT_DIR / "plots" / "confusion_matrix.png")
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Loading history...")
    with open(OUTPUT_DIR / "logs" / "training_history.json", 'r') as f:
        history = json.load(f)
        
    plot_grid_results(history)
    plot_candidates_hd(CFG.device)
    plot_roc_and_cm(OUTPUT_DIR / "logs" / "oof_predictions_b2.csv")