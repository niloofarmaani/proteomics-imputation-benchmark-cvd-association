import os
import subprocess
from pathlib import Path
import sys

# Make sure we can import config_paths from code/
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir, IMPUTATION_DIR


def mean(file_path, columns_num, fractions, missingness_types):
    """
    Preprocesses and handles files for Mean Imputation experiments.

    Parameters
    ----------
    file_path : str
        Path to the original CSV dataset.
    columns_num : str
        Columns specification (e.g., "all" or "10").
    fractions : list of float
        Fractions of data to be corrupted (e.g., [0.01, 0.05, 0.3]).
    missingness_types : list of str
        Types of missingness to introduce (e.g., ["MCAR", "MAR", "MNAR"]).
    """

    # Main experiment directory
    main_directory = str(experiments_dir(columns_num))

    os.makedirs(main_directory, exist_ok=True)

    # Iterate over each combination of fraction and missingness
    for fraction in fractions:
        for missingness in missingness_types:
            # Subfolder based on fraction and missingness
            subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
            full_path = os.path.join(main_directory, subfolder_name)
            os.makedirs(full_path, exist_ok=True)

            # Define file paths
            filename_corruption = f'Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}.csv'
            corrupted_data_path = os.path.join(full_path, filename_corruption)

            filename_normalized_data = f'normalized_data_col_{columns_num}.csv'
            normalized_data_path = os.path.join(full_path, filename_normalized_data)

            # Output file paths for RMSE and mean-imputed data
            filename_rmse_path_mean = f'RMSE_Mean_col_{columns_num}.csv'
            rmse_path_mean = os.path.join(full_path, filename_rmse_path_mean)

            filename_imputed_path_mean = f'imputed_Mean_col_{columns_num}.csv'
            imputed_path_mean = os.path.join(full_path, filename_imputed_path_mean)

            # Check if the required data files exist
            if os.path.exists(corrupted_data_path) and os.path.exists(normalized_data_path):
                print(f"Required input files exist for fraction={fraction}, missingness={missingness}.")
                print("Performing mean imputation...")

                # Run the R script for mean imputation
                subprocess.run([
                    "Rscript",
                    str(IMPUTATION_DIR / "mean_imputer.R"),  # Centralised path from config_paths
                    corrupted_data_path,
                    normalized_data_path,
                    rmse_path_mean,
                    imputed_path_mean
                ],
                stdout=subprocess.DEVNULL  # Suppress normal output
                )
            else:
                print(f"Required data files not found in {full_path}. Please generate corrupted data first.")