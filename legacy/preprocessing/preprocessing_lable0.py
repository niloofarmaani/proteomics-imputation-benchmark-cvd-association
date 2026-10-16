import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from jenga.corruptions.generic import MissingValues
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import MissingIndicator
from sklearn.impute import KNNImputer
import sys
import os

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import MissingIndicator


def process_data(file_path, columns_num, normalized_data_path, missing_mask_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Ensure 'label' column exists
    if 'label' not in data.columns:
        raise ValueError("The 'label' column is not present in the data.")
    
    # Remove rows where 'label' == 2
    initial_row_count = data.shape[0]
    data = data[data['label'] != 2].reset_index(drop=True)
    removed_rows = initial_row_count - data.shape[0]
    print(f"Removed {removed_rows} rows where label == 2. Remaining rows: {data.shape[0]}")
    
    # Ensure 'eid' and 'label' are included in the data
    eid_label_columns = [col for col in ['eid', 'label'] if col in data.columns]
    
    # Extract 'eid' and 'label' columns
    eid_label = data[eid_label_columns]
    data = data.drop(columns=eid_label_columns)
    
    # Select columns to process
    if columns_num == 'all':
        print("Using all columns.")
    else:
        print(f"Using the first {columns_num} columns.")
        data = data.iloc[:, :int(columns_num)]
    
    # Handle missing values
    indicator = MissingIndicator(missing_values=np.nan, features='missing-only', sparse=False)
    missing_mask = indicator.fit_transform(data)
    
    # Save missing mask
    missing_features = data.columns[indicator.features_]
    missing_mask_df = pd.DataFrame(missing_mask, columns=missing_features)
    missing_mask_df.to_csv(missing_mask_path, index=False)
    print("Missing mask saved.")
    
    # Impute missing values
    num_imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(num_imputer.fit_transform(data), columns=data.columns)
    
    # Normalize data
    #data_normalized = data_imputed
    # Normalize data
    scaler = StandardScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)
    # Reattach 'eid' and 'label' columns
    data_normalized = pd.concat([eid_label.reset_index(drop=True), data_normalized], axis=1)
    
    # Save normalized data
    data_normalized.to_csv(normalized_data_path, index=False)
    print("Normalized data saved.")
    
    return missing_mask_df, data_normalized


def introduce_missing_values(normalized_data, fr, miss,corrupted_data_path):
    # Get the list of columns excluding 'eid' and 'label'
    impacted_columns = [col for col in normalized_data.columns if col not in ['eid', 'label']]
    
    # Create a copy of normalized_data to avoid modifying the original data
    corrupted_data = normalized_data.copy()
    
    # Apply the missingness to each column
    for selected_column in impacted_columns:
        # Apply the missingness transformation using the provided MissingValues function
        corrupted_data[selected_column] = MissingValues(column=selected_column, fraction=fr, missingness=miss).transform(normalized_data)[selected_column]
    
    # Save the corrupted data to a CSV file
    corrupted_data.to_csv(corrupted_data_path, index=False)
    
    return corrupted_data

if __name__ == "__main__":
    print("Starting the data processing script...")

    # Get command line arguments
    file_path = sys.argv[1]
    columns_num = sys.argv[2]
    fraction = float(sys.argv[3])
    missingness = sys.argv[4]
    corrupted_data_path = sys.argv[5]
    missing_mask_path = sys.argv[6]
    normalized_data_path = sys.argv[7]


    print(f"Processing data from: {file_path}")
    
    # Process the data
    missing_mask, normalized_data = process_data(
        file_path, columns_num, normalized_data_path, missing_mask_path
    )
    
    # Introduce missing values
    introduce_missing_values(normalized_data, fraction, missingness, corrupted_data_path)
    print("Data processing completed.")
