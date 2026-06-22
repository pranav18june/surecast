import pandas as pd
import numpy as np
import logging
import argparse
import sys
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import random

logging.basicConfig(level=logging.INFO, format='%(message)s')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Random seed set to {seed} for reproducibility.")

def create_domain_interactions(df):
    """
    Creates specific domain interaction features based on supply chain logic.
    Returns the dataframe and a metadata dictionary of formulas.
    """
    metadata = {}
    
    # 1. Sales-Quantity Ratio
    if 'Sales' in df.columns and 'Order Item Quantity' in df.columns:
        # Formula: Sales / (Order Item Quantity + 1e-5)
        df['Sales_Quantity_Ratio'] = df['Sales'] / (df['Order Item Quantity'] + 1e-5)
        metadata['Sales_Quantity_Ratio'] = "Sales / (Order Item Quantity + 1e-5)"
        
    # 2. Shipping-Benefit Ratio
    if 'Benefit per order' in df.columns and 'Days for shipping (real)' in df.columns:
        # Formula: Benefit per order / (Days for shipping (real) + 1e-5)
        df['Shipping_Benefit_Ratio'] = df['Benefit per order'] / (df['Days for shipping (real)'] + 1e-5)
        metadata['Shipping_Benefit_Ratio'] = "Benefit per order / (Days for shipping (real) + 1e-5)"
        
    # 3. Late Delivery Risk Interaction (if present)
    if 'Late_delivery_risk' in df.columns and 'Days for shipment (scheduled)' in df.columns:
        df['Risk_Scheduled_Days'] = df['Late_delivery_risk'] * df['Days for shipment (scheduled)']
        metadata['Risk_Scheduled_Days'] = "Late_delivery_risk * Days for shipment (scheduled)"
        
    return df, metadata

def apply_mathematical_transforms(df, numeric_cols):
    """
    Applies quadratic (x^2) and log(1+|x|) transforms to numeric columns.
    Returns transformed df and a list of the new column names.
    """
    new_cols = []
    for col in numeric_cols:
        if col not in df.columns: continue
        
        # Quadratic
        quad_col = f"{col}_quad"
        df[quad_col] = df[col] ** 2
        new_cols.append(quad_col)
        
        # Log(1 + |x|)
        log_col = f"{col}_log"
        df[log_col] = np.log1p(np.abs(df[col]))
        new_cols.append(log_col)
        
    return df, new_cols

def compute_rolling_features(df, group_cols, target_col, time_col):
    """
    Computes rolling statistical/momentum features (mean, std, z-score)
    WITHIN each time-ordered series.
    """
    df = df.sort_values(by=group_cols + [time_col]).copy()
    
    windows = [4, 12, 50]  # capped at 50 per instructions
    new_features = []
    
    grouped = df.groupby(group_cols)[target_col]
    
    for w in windows:
        # Rolling Mean
        mean_col = f"{target_col}_rolling_mean_{w}"
        df[mean_col] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        
        # Rolling Std
        std_col = f"{target_col}_rolling_std_{w}"
        df[std_col] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        
        # Rolling Z-Score (Momentum)
        z_col = f"{target_col}_rolling_zscore_{w}"
        # (x - rolling_mean) / (rolling_std + 1e-5)
        df[z_col] = (df[target_col] - df[mean_col]) / (df[std_col] + 1e-5)
        
        new_features.extend([mean_col, std_col, z_col])
        
    return df, new_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/cleaned_dataset.csv")
    parser.add_argument("--output_path", type=str, default="data/engineered_dataset.csv")
    parser.add_argument("--target_col", type=str, default="Sales")
    args = parser.parse_args()
    
    set_seed(42)

    if not os.path.exists(args.data_path):
        logging.error(f"Missing data at {args.data_path}. Please complete previous phases.")
        sys.exit(1)

    logging.info("═══════════════════════════════════════")
    logging.info("PHASE 3 — Feature Engineering")
    logging.info("═══════════════════════════════════════\n")

    df = pd.read_csv(args.data_path)
    
    # Identify basic columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if args.target_col in numeric_cols:
        numeric_cols.remove(args.target_col)
        
    all_cat_cols = df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    
    # Filter high cardinality to avoid OOM
    categorical_cols = [c for c in all_cat_cols if c != date_col and df[c].nunique() < 50]
    high_card_cols = [c for c in all_cat_cols if c not in categorical_cols and c != date_col]
    logging.info(f"Dropping high-cardinality categorical columns to prevent OOM: {len(high_card_cols)} columns dropped.")
    df = df.drop(columns=high_card_cols)
    
    if not date_col:
        logging.error("No date column found to ensure temporal ordering.")
        sys.exit(1)
        
    cat_group = 'Category Name' if 'Category Name' in df.columns else next((c for c in df.columns if 'category' in c.lower()), None)
    region_group = 'Order Region' if 'Order Region' in df.columns else next((c for c in df.columns if 'region' in c.lower()), None)
    
    # 1. Domain Interactions
    logging.info("1. Building domain interaction features...")
    df, interaction_metadata = create_domain_interactions(df)
    for k, v in interaction_metadata.items():
        logging.info(f" - {k} = {v}")
        
    # 2. Quadratic & Log Transforms
    logging.info("\n2. Applying quadratic and log(1+|x|) transforms...")
    df, transform_cols = apply_mathematical_transforms(df, numeric_cols)
    logging.info(f" - Generated {len(transform_cols)} transformed features.")
    
    # 3. Rolling / Momentum Features
    logging.info("\n3. Computing rolling statistical/momentum features WITHIN time-ordered series...")
    if cat_group and region_group:
        group_cols = [cat_group, region_group]
        df, rolling_cols = compute_rolling_features(df, group_cols, args.target_col, date_col)
        logging.info(f" - Generated {len(rolling_cols)} rolling features (windows: 4, 12, 50).")
    else:
        logging.warning(" - Could not find Category/Region to group by. Skipping rolling features.")
        rolling_cols = []
        
    # Temporal Split for Permutation Importance (Avoid Leakage)
    df = df.sort_values(by=date_col).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # 4. Preprocessing Pipeline (OneHot, Median Impute, RobustScale)
    logging.info("\n4. Building Scikit-Learn Pipeline (OneHot, Median Imputation, Robust Scaling)...")
    
    # Drop columns that are completely NaN (e.g. Product Description)
    df = df.dropna(axis=1, how='all')
    
    # Update train/val references after dropna
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    from pandas.api.types import is_numeric_dtype
    all_numeric = [c for c in df.columns if is_numeric_dtype(df[c]) and c != args.target_col and c != 'YearWeek' and c not in categorical_cols]
    
    # Fit Numeric Pipeline on TRAIN only
    num_imputer = SimpleImputer(strategy='median')
    num_scaler = RobustScaler(quantile_range=(25.0, 75.0))
    
    train_df[all_numeric] = num_imputer.fit_transform(train_df[all_numeric])
    train_df[all_numeric] = num_scaler.fit_transform(train_df[all_numeric])
    
    val_df[all_numeric] = num_imputer.transform(val_df[all_numeric])
    val_df[all_numeric] = num_scaler.transform(val_df[all_numeric])
    
    # Apply to full df
    df[all_numeric] = num_scaler.transform(num_imputer.transform(df[all_numeric]))
    
    # Fit Categorical Pipeline on TRAIN only
    cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
    train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
    val_df[categorical_cols] = cat_imputer.transform(val_df[categorical_cols])
    df[categorical_cols] = cat_imputer.transform(df[categorical_cols])
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_df[categorical_cols])
    
    # Transform
    X_train_ohe = ohe.transform(train_df[categorical_cols])
    X_val_ohe = ohe.transform(val_df[categorical_cols])
    full_ohe = ohe.transform(df[categorical_cols])
    
    ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    
    # Create X_train and X_val for Permutation Importance
    X_train = np.hstack([train_df[all_numeric].values, X_train_ohe])
    X_val = np.hstack([val_df[all_numeric].values, X_val_ohe])
    y_train = train_df[args.target_col].values
    y_val = val_df[args.target_col].values
    feature_names = all_numeric + ohe_feature_names
    
    # Attach OHE features back to full df for saving
    ohe_df = pd.DataFrame(full_ohe, columns=ohe_feature_names, index=df.index)
    df = pd.concat([df, ohe_df], axis=1)
    
    # Drop original categoricals (except grouping cols needed for Phase 4)
    cols_to_drop = [c for c in categorical_cols if c not in [cat_group, region_group]]
    df = df.drop(columns=cols_to_drop)
    
    # 5. Permutation Importance
    logging.info("\n5. Running Permutation Importance to filter useless transforms...")
    # Using a fast RandomForest on a sample to evaluate feature importance and prevent OOM
    sample_size = min(10000, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_train[sample_idx], y_train[sample_idx])
    
    val_sample_size = min(5000, len(X_val))
    val_sample_idx = np.random.choice(len(X_val), val_sample_size, replace=False)
    result = permutation_importance(rf, X_val[val_sample_idx], y_val[val_sample_idx], n_repeats=3, random_state=42, n_jobs=-1)
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=False)
    
    # Evaluate Quadratic/Log Transforms
    useless_transforms = []
    useful_transforms = []
    for col in transform_cols:
        if col in importances['feature'].values:
            imp_val = importances.loc[importances['feature'] == col, 'importance'].values[0]
            if imp_val <= 0.001:  # Threshold for usefulness
                useless_transforms.append(col)
            else:
                useful_transforms.append(col)
                
    logging.info(f" - Found {len(useful_transforms)} USEFUL mathematical transforms (Importance > 0.001).")
    logging.info(f" - Found {len(useless_transforms)} USELESS mathematical transforms. REMOVING them.")
    
    if len(useless_transforms) > 0:
        logging.info(f" - Dropped e.g.: {useless_transforms[:5]}...")
        df = df.drop(columns=useless_transforms)
        
    # Fit and save Target Scaler as requested
    logging.info("\n6. Fitting Target Scaler (StandardScaler) on the target variable...")
    target_scaler = StandardScaler()
    # Fit on the training portion to prevent data leakage
    train_target = train_df[args.target_col].values.reshape(-1, 1)
    target_scaler.fit(train_target)
    
    scaler_path = "models/target_scaler.pkl"
    joblib.dump(target_scaler, scaler_path)
    logging.info(f" - Target scaler saved to {scaler_path}")
        
    # Save engineered dataset
    df.to_csv(args.output_path, index=False)
    logging.info(f"\n6. OUTPUT: Engineered dataset saved to {args.output_path}")

if __name__ == "__main__":
    set_seed(42)
    main()
