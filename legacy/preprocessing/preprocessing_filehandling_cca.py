# preprocessing_filehandling_cca.py

import os
import sys
import random
from pathlib import Path
import pandas as pd
import numpy as np
from jenga.corruptions.generic import MissingValues

# Make sure .../project_root/code is on sys.path
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import PREPROCESSING_DIR, experiments_dir, fraction_dir

def simulate_missingness(df_ground_truth, fraction, missingness_type, seed=42):
    """
    Applies missingness to the ground truth dataframe using Jenga.
    Returns only the corrupted dataframe.
    """
    # Force the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Create a copy so we don't alter the original ground truth in memory
    df_corrupted = df_ground_truth.copy()
    
    # Identify feature columns (protecting 'eid' and 'label' from corruption)
    feature_cols = [c for c in df_corrupted.columns if c not in ["eid", "label"]]
    
    if len(feature_cols) == 0:
        print("WARNING: No feature columns to corrupt.")
        return df_corrupted

    # Apply missingness column-wise using Jenga
    for col in feature_cols:
        try:
            # Try passing random_state if Jenga's API supports it directly
            corruptor = MissingValues(
                column=col,
                fraction=fraction,
                missingness=missingness_type,
                random_state=seed
            )
        except TypeError:
            # Fallback if MissingValues doesn't accept random_state natively.
            # The global np.random.seed(seed) above will handle the stochasticity.
            corruptor = MissingValues(
                column=col,
                fraction=fraction,
                missingness=missingness_type
            )
            
        corrupted_col_df = corruptor.transform(df_corrupted)
        df_corrupted[col] = corrupted_col_df[col]

    return df_corrupted


def preprocessing_filehandling(file_path, columns_num, fractions, missingness_types, seed=42):
    """
    Run preprocessing for each combination of fraction and missingness.
    Drops originally missing values to create a true ground truth, then
    simulates and SAVES the corrupted data using the provided seed.
    """
    
    # 1. Load original dataset
    print(f"Loading data from: {file_path} (Seed: {seed})")
    df_orig = pd.read_csv(file_path)
    
    # 2. Process labels: Ensure label exists and remove label == 2
    if "label" in df_orig.columns:
        initial_row_count = df_orig.shape[0]
        df_orig = df_orig[df_orig["label"] != 2].reset_index(drop=True)
        removed_rows = initial_row_count - df_orig.shape[0]
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows where label == 2. Remaining rows: {df_orig.shape[0]}")
    else:
        print("Warning: 'label' column is not present in the data.")

    # 3. Preserve eid and label before slicing columns
    eid_label_columns = [c for c in ["eid", "label"] if c in df_orig.columns]
    eid_label = df_orig[eid_label_columns]
    features = df_orig.drop(columns=eid_label_columns)
    
    # 4. Handle 'columns_num' parameter safely on features only
    if str(columns_num).lower() != "all":
        try:
            col_limit = int(columns_num)
            if col_limit < len(features.columns):
                # Keep only the first N feature columns
                features = features.iloc[:, :col_limit]
        except ValueError:
            print(f"Warning: columns_num '{columns_num}' is not 'all' or a valid integer. Using all columns.")

    # 5. Drop all-NaN columns from features
    all_nan_cols = features.columns[features.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"Dropping {len(all_nan_cols)} columns with all NaN values.")
        features = features.drop(columns=all_nan_cols)

    actual_cols = len(features.columns)

    # Reattach eid and label for Complete Case Analysis
    df_combined = pd.concat([eid_label.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    # 6. Calculate missingness statistics on combined data
    total_rows = len(df_combined)
    total_cells = df_combined.size
    missing_cells = df_combined.isna().sum().sum()
    cell_missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

    # 7. Create True Ground Truth by dropping rows with ANY missing values
    df_ground_truth = df_combined.dropna()
    rows_kept = len(df_ground_truth)
    rows_eliminated = total_rows - rows_kept
    row_eliminated_percentage = (rows_eliminated / total_rows) * 100 if total_rows > 0 else 0

    # 8. Report the statistics
    print("\n" + "="*40)
    print("      DATA PREPROCESSING REPORT")
    print("="*40)
    print(f"Feature Columns used:           {actual_cols} (Param: {columns_num})")
    print(f"Total overall cell missingness: {cell_missing_percentage:.2f}%")
    print(f"Initial total rows:             {total_rows}")
    print(f"Rows eliminated (had NaNs):     {rows_eliminated} ({row_eliminated_percentage:.2f}%)")
    print(f"Rows retained (GROUND TRUTH):   {rows_kept}")
    print("="*40 + "\n")

    if rows_kept == 0:
        raise ValueError("CRITICAL ERROR: All retained rows contain at least one missing value. Complete case analysis is impossible.")

    # Main directory where reprocessed data should be saved
    main_directory = str(experiments_dir(columns_num))
    os.makedirs(main_directory, exist_ok=True)

    # 9. Iterate over each combination of fraction and missingness
    for fraction in fractions:
        for missingness in missingness_types:
            # Define subfolder name for this combination
            full_path = fraction_dir(columns_num, fraction, missingness)
            full_path.mkdir(parents=True, exist_ok=True)

            # --- ADDED SEED TO CORRUPTED FILENAME ---
            filename_corruption = f'Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}_seed_{seed}.csv'
            corrupted_data_path = os.path.join(full_path, filename_corruption)

            filename_normalized_data = f'normalized_data_col_{columns_num}.csv'
            normalized_data_path = os.path.join(full_path, filename_normalized_data)

            # --- SAVE 1: The clean ground truth ---
            # We save the ground truth. It will naturally overwrite during loops, but it is 
            # mathematically identical every time, so this is perfectly fine.
            df_ground_truth.to_csv(normalized_data_path, index=False)

            # --- GENERATE CORRUPTION ---
            df_corrupted = simulate_missingness(df_ground_truth, fraction, missingness, seed=seed)
            
            # --- SAVE 2: The corrupted data ---
            df_corrupted.to_csv(corrupted_data_path, index=False)
            
            print(f"Saved GT and Corrupted Data for: {missingness} at {fraction*100}% missingness (Seed: {seed}).")
            
    print(f"\nPreprocessing and simulation complete for seed {seed}!")