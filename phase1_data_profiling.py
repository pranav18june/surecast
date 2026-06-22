import pandas as pd
import numpy as np
import os
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def set_seed(seed=42):
    np.random.seed(seed)
    # random.seed(seed) if needed
    logging.info(f"Random seed set to {seed} for reproducibility.")

def main(data_path, output_path, output_report_path):
    set_seed(42)
    
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at '{data_path}'.")
        logging.info("Please download the 'DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS' dataset from Kaggle.")
        logging.info("Expected filename: 'DataCoSupplyChainDataset.csv'")
        sys.exit(1)
        
    logging.info("═══════════════════════════════════════")
    logging.info("PHASE 1 — Data Profiling and Cleaning")
    logging.info("═══════════════════════════════════════\n")
    
    try:
        # 1. Load the raw dataset
        # Encoding 'latin1' is usually required for this Kaggle dataset
        df = pd.read_csv(data_path, encoding='latin1')
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        sys.exit(1)
        
    initial_row_count = len(df)
    logging.info(f"1. TOTAL ROW COUNT: {initial_row_count}\n")
    
    # Identify target variable
    # Candidate columns: 'Sales', 'Sales per customer', 'Order Item Quantity'
    target_col = None
    for col in ['Sales', 'Sales per customer', 'Order Item Quantity']:
        if col in df.columns:
            target_col = col
            break
            
    if not target_col:
        logging.error("Target variable not found in columns. Cannot proceed.")
        sys.exit(1)
        
    logging.info(f"\n[ASSUMPTION FLAG] Identified target variable as '{target_col}'. Please confirm this is correct.\n")
    
    # Print columns, dtypes, missing values
    missing_counts = df.isnull().sum()
    col_summary = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_count': missing_counts
    })
    logging.info("COLUMNS, DTYPES, AND MISSING VALUES:")
    logging.info(col_summary.to_string())
    logging.info("\n")
    
    logging.info(f"BASIC SUMMARY STATISTICS FOR '{target_col}':")
    logging.info(df[target_col].describe().to_string())
    logging.info("\n")
    
    # 2. Remove records missing the target variable
    missing_target = df[target_col].isnull()
    num_missing_target = missing_target.sum()
    
    if num_missing_target > 0:
        df_dropped = df[missing_target]
        df_retained = df[~missing_target]
        logging.info(f"2. REMOVAL: Dropping {num_missing_target} records missing target '{target_col}'.")
        
        # Check for systematic bias
        for col in ['Type', 'Market', 'Order Region']:
            if col in df.columns:
                logging.info(f"\nBias Check - Distribution of '{col}' in DROPPED rows:")
                logging.info(df_dropped[col].value_counts(normalize=True).to_string())
                logging.info(f"\nBias Check - Distribution of '{col}' in RETAINED rows:")
                logging.info(df_retained[col].value_counts(normalize=True).to_string())
                
        df = df_retained
    else:
        logging.info(f"2. REMOVAL: No records missing the target variable '{target_col}'. (0 rows dropped)\n")
        
    # 3. Median/Mode imputation for remaining missing predictors (LEAK-FREE)
    date_col = next((c for c in df.columns if 'date' in c.lower() and 'order' in c.lower()), None)
    if not date_col:
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
    else:
        logging.warning("No date column found, falling back to full dataset for imputation stats.")
        train_df = df

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_numeric = df[numeric_cols].isnull().sum()
    cols_to_impute = missing_numeric[missing_numeric > 0].index.tolist()
    
    logging.info("3. IMPUTATION (Fit on Train 80% split):")
    if cols_to_impute:
        for col in cols_to_impute:
            missing_count = df[col].isnull().sum()
            median_val = train_df[col].median()
            df[col] = df[col].fillna(median_val)
            logging.info(f" - Imputed {missing_count} missing values in '{col}' with training median: {median_val}")
    else:
        logging.info(" - No missing values in numeric predictors required median imputation.")
        
    # Categorical mode imputation
    categorical_cols = df.select_dtypes(exclude=[np.number, 'datetime', 'datetime64[ns]']).columns
    missing_cat = df[categorical_cols].isnull().sum()
    cat_cols_to_impute = missing_cat[missing_cat > 0].index.tolist()
    if cat_cols_to_impute:
        for col in cat_cols_to_impute:
            missing_count = df[col].isnull().sum()
            modes = train_df[col].mode(dropna=True)
            mode_val = modes[0] if not modes.empty else 'Missing'
            df[col] = df[col].fillna(mode_val)
            logging.info(f" - Imputed {missing_count} missing values in '{col}' with training mode: '{mode_val}'")
    logging.info("\n")
        
    # 4. Aggregate city-level coordinates to regional/country-level granularity
    logging.info("4. AGGREGATION OF COORDINATES:")
    lat_col = 'Latitude'
    lon_col = 'Longitude'
    country_col = 'Order Country'
    
    # Try alternatives if exact names differ slightly
    if lat_col not in df.columns:
        lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    if lon_col not in df.columns:
        lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    if country_col not in df.columns:
        country_col = next((c for c in df.columns if 'country' in c.lower() and 'order' in c.lower()), None)
        if not country_col:
             country_col = next((c for c in df.columns if 'country' in c.lower()), None)
             
    if lat_col and lon_col and country_col:
        country_coords = df.groupby(country_col)[[lat_col, lon_col]].mean().reset_index()
        country_coords.rename(columns={lat_col: 'Country_Latitude_Mean', lon_col: 'Country_Longitude_Mean'}, inplace=True)
        df = df.merge(country_coords, on=country_col, how='left')
        df = df.drop(columns=[lat_col, lon_col])
        logging.info(f" - Replaced city-level '{lat_col}' and '{lon_col}' with regional averages grouped by '{country_col}'.")
        logging.info(" - Justification: City-level coordinates introduce excessive high-cardinality noise and potential for overfitting, particularly for sequence models. Aggregating to country/regional centroids preserves macroscopic geographical patterns while providing a robust, lower-variance spatial feature for demand forecasting.")
    else:
        logging.warning(" - Expected coordinate/country columns not found. Skipping geographical aggregation.")
    logging.info("\n")
        
    # 5. Output clean dataframe and written summary
    df.to_csv(output_path, index=False)
    logging.info(f"5. OUTPUT: Cleaned dataframe saved to '{output_path}'.\n")
    
    # Generate written report
    report = []
    report.append("═══════════════════════════════════════")
    report.append("SUREcast Phase 1: Data Summary Report")
    report.append("═══════════════════════════════════════")
    report.append(f"Initial Row Count: {initial_row_count}")
    report.append(f"Final Row Count (After Cleaning): {len(df)}")
    
    # Time Range Covered
    date_col = next((c for c in df.columns if 'date' in c.lower() and 'order' in c.lower()), None)
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col])
            report.append(f"Time Range Covered: {dates.min().strftime('%Y-%m-%d %H:%M:%S')} to {dates.max().strftime('%Y-%m-%d %H:%M:%S')} (via {date_col})")
        except:
            report.append(f"Time Range Covered: Parsing failed for '{date_col}'")
    else:
        report.append("Time Range Covered: Unknown (Date column not found)")
        
    report.append(f"Target Variable: '{target_col}'")
    report.append("Target Variable Scale & Units: Typically USD ($) or standard currency for 'Sales'. Represents revenue per order item.")
    report.append("\nTarget Variable Distribution (After Cleaning):")
    report.append(df[target_col].describe().to_string())
    
    report_text = "\n".join(report)
    with open(output_report_path, "w") as f:
        f.write(report_text)
        
    logging.info(f"Output Report saved to '{output_report_path}'.")
    logging.info("Summary Content:")
    logging.info(report_text)


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="SUREcast Phase 1: Data Profiling")
    parser.add_argument("--data_path", type=str, default="data/DataCoSupplyChainDataset.csv", help="Path to raw dataset CSV")
    parser.add_argument("--output_path", type=str, default="data/cleaned_dataset.csv", help="Path to save cleaned CSV")
    parser.add_argument("--output_report_path", type=str, default="reports/phase1_summary.txt", help="Path to save summary report")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_report_path), exist_ok=True)
    
    main(args.data_path, args.output_path, args.output_report_path)
