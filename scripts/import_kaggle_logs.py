import json
import os
import sys

# Add project root to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import CFG

# Raw output logs preserved from the Kaggle training session (V6)
RAW_KAGGLE_LOGS = """
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

def parse_logs():
    """
    Parses the raw text logs and converts them into a structured list of dictionaries.
    Each dictionary represents the history of one fold.
    """
    history = []
    current_fold_data = {}
    
    lines = RAW_KAGGLE_LOGS.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect new fold section
        if "=== FOLD" in line:
            # Save previous fold data if exists
            if current_fold_data:
                history.append(current_fold_data)
            # Reset container for new fold
            current_fold_data = {
                't_loss': [], 'v_loss': [], 
                't_auc': [], 'v_auc': []
            }
            continue
            
        # Parse epoch metrics
        # Format: Ep X/X | Loss: T/V | AUC: T/V
        if "Loss:" in line and "AUC:" in line:
            parts = line.split('|')
            
            # Extract Loss values (Train/Valid)
            loss_part = parts[1].split(':')[1].strip().split('/')
            t_loss = float(loss_part[0])
            v_loss = float(loss_part[1])
            
            # Extract AUC values (Train/Valid)
            auc_part = parts[2].split(':')[1].strip().split('/')
            t_auc = float(auc_part[0])
            v_auc = float(auc_part[1])
            
            # Append to current fold history
            current_fold_data['t_loss'].append(t_loss)
            current_fold_data['v_loss'].append(v_loss)
            current_fold_data['t_auc'].append(t_auc)
            current_fold_data['v_auc'].append(v_auc)
            
    # Append the last fold
    if current_fold_data:
        history.append(current_fold_data)
        
    return history

def main():
    print("[INFO] Parsing raw Kaggle logs...")
    history_data = parse_logs()
    
    # Ensure output directory exists
    os.makedirs(CFG.LOGS_DIR, exist_ok=True)
    
    output_path = os.path.join(CFG.LOGS_DIR, "training_history.json")
    
    with open(output_path, "w") as f:
        json.dump(history_data, f, indent=4)
        
    print(f"[SUCCESS] Parsed data for {len(history_data)} folds.")
    print(f"[SUCCESS] Training history saved to: {output_path}")

if __name__ == "__main__":
    main()