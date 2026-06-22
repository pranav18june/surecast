import numpy as np
import pandas as pd
import torch
import logging
import argparse
import sys
import os
import joblib
from phase4_model_architecture import train_dl_branch

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_and_prep_data():
    data_path = "data/engineered_dataset.csv"
    if not os.path.exists(data_path):
        logging.error(f"[ERROR] Required dataset '{data_path}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    target_col = "Sales"
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    cat_group = next((c for c in df.columns if 'category' in c.lower()), None)
    region_group = next((c for c in df.columns if 'region' in c.lower()), None)
    
    ignore_cols = [target_col, cat_group, region_group, date_col, 'YearWeek']
    from pandas.api.types import is_numeric_dtype
    feature_cols = [c for c in df.columns if c not in ignore_cols and is_numeric_dtype(df[c])]
    
    seq_len = 8
    X_seq_all, y_seq_all, dates_all = [], [], []
    
    for _, group in df.groupby([cat_group, region_group]):
        vals = group[feature_cols].values
        targets = group[target_col].values
        dates = group[date_col].values
        
        if len(vals) < seq_len:
            pad_len = seq_len - len(vals)
            pad_vals = np.full((pad_len, vals.shape[1]), 0.0)
            vals = np.vstack([pad_vals, vals])
            pad_target = np.full((pad_len,), 0.0)
            targets = np.concatenate([pad_target, targets])
            pad_dates = np.full((pad_len,), dates[0])
            dates = np.concatenate([pad_dates, dates])
            
        for i in range(len(vals) - seq_len):
            X_seq_all.append(vals[i:i+seq_len])
            y_seq_all.append(targets[i+seq_len])
            dates_all.append(dates[i+seq_len])
            
    sort_idx = np.argsort(np.array(dates_all))
    X_seq_all = np.array(X_seq_all)[sort_idx]
    y_seq_all = np.array(y_seq_all)[sort_idx]
    return X_seq_all, y_seq_all, len(feature_cols)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    logging.info("═══════════════════════════════════════")
    logging.info("ROBUSTNESS AUDIT: Deep Learning Branch")
    logging.info("═══════════════════════════════════════\n")
    
    X_all, y_all, input_size = load_and_prep_data()
    target_scaler = joblib.load("data/target_scaler.pkl")
    
    # Pre-scale all targets for the DL model
    y_all_scaled = target_scaler.transform(y_all.reshape(-1, 1)).flatten()
    
    split_idx = int(len(X_all) * 0.8)
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train_scaled, y_val_scaled = y_all_scaled[:split_idx], y_all_scaled[split_idx:]
    y_val_orig = y_all[split_idx:]
    
    logging.info("1. Initialization Sensitivity (5 Random Seeds)")
    seeds = [42, 100, 2026, 777, 12345]
    maes = []
    
    for s in seeds:
        set_seed(s)
        logging.info(f" -> Training with seed {s}...")
        # Train for fewer epochs just to check stability efficiently in audit
        mae_scaled, preds_scaled = train_dl_branch(X_train, y_train_scaled, X_val, y_val_scaled, input_size, seq_len=8, epochs=30)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        mae_orig = np.mean(np.abs(preds - y_val_orig))
        maes.append(mae_orig)
        logging.info(f"    Validation MAE: {mae_orig:.4f}")
        
    logging.info(f"\n[RESULTS] 5-Seed MAE: Mean = {np.mean(maes):.4f}, Std Dev = {np.std(maes):.4f}")
    
    logging.info("\n2. Walk-Forward Validation (3 Folds)")
    # Split the timeline into 4 chunks: Train1, Val1, Val2, Val3
    chunk_size = int(len(X_all) * 0.25)
    wf_maes = []
    
    for fold in range(1, 4):
        set_seed(42)
        train_end = chunk_size * fold
        val_end = chunk_size * (fold + 1)
        
        X_wf_train = X_all[:train_end]
        y_wf_train = y_all_scaled[:train_end]
        
        X_wf_val = X_all[train_end:val_end]
        y_wf_val = y_all_scaled[train_end:val_end]
        y_wf_val_orig = y_all[train_end:val_end]
        
        logging.info(f" -> Fold {fold}: Train size = {len(X_wf_train)}, Val size = {len(X_wf_val)}")
        mae_scaled, preds_scaled = train_dl_branch(X_wf_train, y_wf_train, X_wf_val, y_wf_val, input_size, seq_len=8, epochs=30)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        mae_orig = np.mean(np.abs(preds - y_wf_val_orig))
        wf_maes.append(mae_orig)
        logging.info(f"    Fold {fold} MAE: {mae_orig:.4f}")
        
    logging.info(f"\n[RESULTS] Walk-Forward MAE: Mean = {np.mean(wf_maes):.4f}, Std Dev = {np.std(wf_maes):.4f}")
    logging.info("\nRobustness Audit Complete.")

if __name__ == "__main__":
    set_seed(42)
    main()
