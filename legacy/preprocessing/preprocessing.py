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


def process_data(file_path,columns_num,nomalized_data_path,missing_mask_path):
    
    data = pd.read_csv(file_path)
        # Select all columns if 'column' is 'all', otherwise select the first 'column' columns
    if columns_num == 'all':
        print("Using all columns.")
    else:
        print(f"Using the first {columns_num} columns.")
        data = data.iloc[:, :int(columns_num)]  # Select the first 'column' columns


    indicator = MissingIndicator(missing_values=np.nan, features='missing-only', sparse=False)

    missing_mask = indicator.fit_transform(data)
    pd.DataFrame(missing_mask).to_csv(missing_mask_path, index=False)
    print("missing mask saved.")
    num_imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(num_imputer.fit_transform(data), columns=data.columns)

    scaler = StandardScaler()
    data_normalized = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)
    data_normalized.to_csv(nomalized_data_path, index=False)

    return pd.DataFrame(missing_mask), data_normalized

def introduce_missing_values(normalized_data, fraction, missingness, corrupted_data_path):

    print(f"Introducing missing values with fraction={fraction} and missingness={missingness}...")
    
    impacted_column = normalized_data.columns.tolist()
    corrupted_data = pd.DataFrame(np.nan, index=normalized_data.index, columns=impacted_column)

    for i in range(len(impacted_column)):
        selected_column = impacted_column[i]
        corrupted_data[selected_column] = MissingValues(column=selected_column, fraction=fraction, missingness=missingness).transform(normalized_data)[selected_column]

    corrupted_data.to_csv(corrupted_data_path, index=False)

    print("Corrupted data saved.")

    return corrupted_data

if __name__ == "__main__":
    print("Starting the data processing script...")

    # Get command line arguments
    file_path = sys.argv[1]
    columns_num= sys.argv[2]
    fraction = float(sys.argv[3])
    missingness = sys.argv[4]
    corrupted_data_path= sys.argv[5]
    missing_mask_path= sys.argv[6]
    nomalized_data_path= sys.argv[7]

    print(f"Processing data from: {file_path}")
    
    # Process the data
    missing_mask, normalized_data = process_data(file_path,columns_num,nomalized_data_path, missing_mask_path)
    

   
    # Introduce missing values
    introduce_missing_values(normalized_data, fraction, missingness,corrupted_data_path)
    print("Data processing completed.")
