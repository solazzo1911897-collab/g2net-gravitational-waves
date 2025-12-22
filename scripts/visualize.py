import sys
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from nnAudio.Spectrogram import CQT1992v2

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import CFG
from src.dataset import G2NetDataset

# ==============================================================================
# 1. Spectrogram Analysis (Balanced Version)
# ==============================================================================
def find_loudest_signals(ds, cqt_layer, limit=500):
    """
    Scans for the strongest signals to ensure we plot visible chirps.
    """
    print(f"[INFO] Scanning first {limit} positive files to find strong signals...")
    
    pos_indices = np.where(ds.labels == 1)[0]
    scan_indices = pos_indices[:limit]
    
    scores = []
    
    for idx in tqdm(scan_indices, desc="Hunting for Chirps"):
        wave, _ = ds[idx]
        with torch.no_grad():
            wave_tensor = wave.unsqueeze(0).to(CFG.device)
            spec = cqt_layer(wave_tensor.view(1*3, -1))
            img = (spec[0] + spec[1] + spec[2]).cpu().numpy()
            
            img = np.log1p(img)
            img = img - np.median(img, axis=1, keepdims=True)
            
            # Score based on peak intensity
            score = np.max(img)
            scores.append((score, idx))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    best_indices = [x[1] for x in scores[:2]]
    
    print(f"[SUCCESS] Found strongest candidates at indices: {best_indices}")
    return best_indices

def plot_spectrogram():
    print("[INFO] Generating Balanced Spectrogram (Standard Res + Texture)...")
    
    if not os.path.exists(CFG.SUBSET_CSV):
        df_full = pd.read_csv(CFG.LABELS_CSV)
        df = df_full.sample(n=60000, random_state=CFG.seed).reset_index(drop=True)
        df.to_csv(CFG.SUBSET_CSV, index=False)
    else:
        df = pd.read_csv(CFG.SUBSET_CSV)

    ds = G2NetDataset(df)
    
    # BACK TO STANDARD RESOLUTION (bins_per_octave=12)
    # This creates the "blocky" look that makes the signal pop visually.
    cqt_layer = CQT1992v2(
        sr=CFG.sr, fmin=CFG.fmin, fmax=500, hop_length=32, 
        bins_per_octave=12, 
        output_format="Magnitude", verbose=False
    ).to(CFG.device)

    # Find the loud signals
    indices = find_loudest_signals(ds, cqt_layer)

    # Plot Setup
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    plt.subplots_adjust(right=0.9, top=0.85)

    for i, idx in enumerate(indices):
        wave, target = ds[idx]
        file_id = ds.file_names[idx]
        
        with torch.no_grad():
            wave_tensor = wave.unsqueeze(0).to(CFG.device)
            spec = cqt_layer(wave_tensor.view(1*3, -1))
            img = (spec[0] + spec[1] + spec[2]).cpu().numpy()

        # --- BALANCED PIPELINE ---
        
        # 1. Log Scale
        img = np.log1p(img)
        
        # 2. Median Subtraction (Cleaner background)
        img = img - np.median(img, axis=1, keepdims=True)
        
        # 3. Min-Max Normalization
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        # 4. THRESHOLD ADJUSTMENT
        # lowered from 0.60 to 0.45.
        # This keeps the signal bright but allows some background "dust" to appear,
        # making it look less "fake/empty".
        img[img < 0.45] = 0 

        # Plot
        ax = axes[i]
        im = ax.imshow(img, aspect='auto', origin='lower', cmap='inferno', interpolation='bicubic')
        
        ax.set_title(f"ID: {file_id}\nStrong Chirp (Target=1)", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Frequency")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label('Normalized Amplitude (Threshold > 0.45)', fontsize=12)
    
    plt.suptitle("Gravitational Wave Signal Reconstruction", fontsize=22, y=1.05)
    
    output_path = os.path.join(CFG.PLOTS_DIR, "spectrogram.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Spectrogram saved to {output_path}")

# ==============================================================================
# 2. Training Grid
# ==============================================================================
def plot_grid():
    log_file = os.path.join(CFG.LOGS_DIR, "training_history.json")
    if not os.path.exists(log_file): return

    with open(log_file, 'r') as f: histories = json.load(f)
    n_folds = len(histories)
    epochs = range(1, len(histories[0]['t_loss']) + 1)
    colors = sns.color_palette("husl", n_folds)
    
    sns.set_style("white"); plt.rcParams['axes.spines.right'] = False; plt.rcParams['axes.spines.top'] = False
    fig, axes = plt.subplots(2, n_folds, figsize=(20, 8), sharex=True)
    fig.text(0.125, 1.02, 'Cross-Validation Results', fontsize=28, fontweight='bold', ha='left')

    for i in range(n_folds):
        h = histories[i]; c = colors[i]
        axes[0, i].plot(epochs, h['t_loss'], color=c, marker='o', label='Train')
        axes[0, i].plot(epochs, h['v_loss'], color='gray', marker='x', ls='--', label='Valid')
        axes[0, i].set_title(f'Fold {i+1}', fontweight='bold')
        if i==0: axes[0, i].set_ylabel('BCE Loss'); axes[0, i].legend()
        
        axes[1, i].plot(epochs, h['t_auc'], color=c, marker='o', label='Train')
        axes[1, i].plot(epochs, h['v_auc'], color='dimgray', marker='s', label='Valid')
        if i==0: axes[1, i].set_ylabel('AUC Score')
        axes[1, i].set_xlabel('Epoch')

    plt.tight_layout(); plt.savefig(os.path.join(CFG.PLOTS_DIR, "training_grid.png"), bbox_inches='tight'); plt.close()

# ==============================================================================
# 3. Metrics
# ==============================================================================
def plot_metrics():
    csv_file = os.path.join(CFG.LOGS_DIR, "oof_predictions_b2.csv")
    if not os.path.exists(csv_file): return

    df = pd.read_csv(csv_file)
    y_true, y_pred = df['target'], df['pred_b2']

    plt.figure(figsize=(9, 7)); fpr, tpr, _ = roc_curve(y_true, y_pred); score = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, color='#8B0000', lw=3, label=f'AUC={score:.4f}')
    plt.fill_between(fpr, tpr, color='#8B0000', alpha=0.1); plt.plot([0,1],[0,1], 'navy', ls='--'); plt.legend(loc='lower right')
    plt.title('ROC Curve', fontsize=20, fontweight='bold'); plt.savefig(os.path.join(CFG.PLOTS_DIR, "roc_curve.png")); plt.close()

    y_pred_bin = (y_pred > 0.5).astype(int); cm = confusion_matrix(y_true, y_pred_bin); tn, fp, fn, tp = cm.ravel()
    labels = np.asarray([f"TN\n{tn}", f"FP\n{fp}", f"FN\n{fn}", f"TP\n{tp}"]).reshape(2, 2)
    plt.figure(figsize=(8, 7.5)); sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, annot_kws={"size": 14, "weight":"bold"})
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold'); plt.subplots_adjust(bottom=0.25)
    stats = f"Accuracy: {(tp+tn)/len(y_true):.4f}\nSensitivity: {tp/(tp+fn):.4f}\nSpecificity: {tn/(tn+fp):.4f}"
    plt.figtext(0.5, 0.08, stats, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.1, "pad":10})
    plt.savefig(os.path.join(CFG.PLOTS_DIR, "confusion_matrix.png")); plt.close()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    plot_spectrogram()
    plot_grid()
    plot_metrics()
    print(f"\n[DONE] All plots generated in {CFG.PLOTS_DIR}")