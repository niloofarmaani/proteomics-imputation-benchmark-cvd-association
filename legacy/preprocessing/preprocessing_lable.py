import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from jenga.corruptions.generic import MissingValues


def process_data(file_path, columns_num, normalized_data_path):
    # Load data
    data = pd.read_csv(file_path)

    # Ensure label exists
    if "label" not in data.columns:
        raise ValueError("The 'label' column is not present in the data.")

    # Remove rows where label == 2
    initial_row_count = data.shape[0]
    data = data[data["label"] != 2].reset_index(drop=True)
    removed_rows = initial_row_count - data.shape[0]
    print(f"Removed {removed_rows} rows where label == 2. Remaining rows: {data.shape[0]}")

    if data.shape[0] == 0:
        print("WARNING: No rows left after filtering. Writing empty output.")
        empty_df = pd.DataFrame()
        empty_df.to_csv(normalized_data_path, index=False)
        return empty_df

    # Preserve eid and label
    eid_label_columns = [c for c in ["eid", "label"] if c in data.columns]
    eid_label = data[eid_label_columns]
    data = data.drop(columns=eid_label_columns)

    # Select columns
    if columns_num == "all":
        print("Using all columns.")
    else:
        print(f"Using the first {columns_num} columns.")
        data = data.iloc[:, :int(columns_num)]

    # Drop all-NaN columns
    all_nan_cols = data.columns[data.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"Dropping {len(all_nan_cols)} columns with all NaN values:")
        print(list(all_nan_cols))
        data = data.drop(columns=all_nan_cols)

    # If no feature columns remain, write minimal output and exit
    if data.shape[1] == 0:
        print("WARNING: No usable feature columns remain.")
        minimal_df = eid_label.copy()
        minimal_df.to_csv(normalized_data_path, index=False)
        return minimal_df

    # Mean imputation
    imputer = SimpleImputer(strategy="mean")
    data_imputed = pd.DataFrame(
        imputer.fit_transform(data),
        columns=data.columns
    )

    # Normalization
    scaler = StandardScaler()
    data_normalized = pd.DataFrame(
        scaler.fit_transform(data_imputed),
        columns=data.columns
    )

    # Reattach eid and label
    data_normalized = pd.concat(
        [eid_label.reset_index(drop=True), data_normalized],
        axis=1
    )

    # Save
    data_normalized.to_csv(normalized_data_path, index=False)
    print("Normalized data saved:", normalized_data_path)

    return data_normalized


def introduce_missing_values(normalized_data, fr, miss, corrupted_data_path, missing_mask_path):
    feature_cols = [c for c in normalized_data.columns if c not in ["eid", "label"]]

    # No feature columns case
    if len(feature_cols) == 0:
        print("WARNING: No feature columns to corrupt. Writing empty outputs.")
        normalized_data.to_csv(corrupted_data_path, index=False)
        empty_mask = pd.DataFrame(index=normalized_data.index)
        empty_mask.to_csv(missing_mask_path, index=False)
        return normalized_data, empty_mask

    corrupted_data = normalized_data.copy()

    # Apply missingness column-wise
    for col in feature_cols:
        corrupted_col_df = MissingValues(
            column=col,
            fraction=fr,
            missingness=miss
        ).transform(normalized_data)

        corrupted_data[col] = corrupted_col_df[col]

    # Missing mask
    missing_mask_df = corrupted_data[feature_cols].isna().astype(int)

    # Save outputs
    corrupted_data.to_csv(corrupted_data_path, index=False)
    missing_mask_df.to_csv(missing_mask_path, index=False)

    miss_rate = float(missing_mask_df.to_numpy().mean())
    miss_count = int(missing_mask_df.to_numpy().sum())

    print(
        f"Missing mask saved: {missing_mask_path} "
        f"(shape={missing_mask_df.shape}, missing_count={miss_count}, missing_rate={miss_rate:.6f})"
    )
    print(f"Corrupted data saved: {corrupted_data_path}")

    return corrupted_data, missing_mask_df


if __name__ == "__main__":
    print("Starting the data processing script...")

    file_path = sys.argv[1]
    columns_num = sys.argv[2]
    fraction = float(sys.argv[3])
    missingness = sys.argv[4]
    corrupted_data_path = sys.argv[5]
    missing_mask_path = sys.argv[6]
    normalized_data_path = sys.argv[7]

    print(f"Processing data from: {file_path}")

    normalized_data = process_data(
        file_path,
        columns_num,
        normalized_data_path
    )

    introduce_missing_values(
        normalized_data,
        fraction,
        missingness,
        corrupted_data_path,
        missing_mask_path
    )

    print("Data processing completed.")
