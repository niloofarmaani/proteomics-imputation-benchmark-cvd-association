# preprocessing.py

import os
import subprocess
from pathlib import Path
import sys

# Make sure .../project_root/code is on sys.path
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import PREPROCESSING_DIR, experiments_dir, fraction_dir



def preprocessing_filehandling(file_path, columns_num, fractions, missingness_types):
    """
    Run preprocessing for each combination of fraction and missingness
    on the given file_path with columns_num.

    :param file_path: Path to the original dataset (string)
    :param columns_num: Number of columns (features) to process (int)
    :param fractions: List of fractions (e.g., [0.01, 0.1, 0.3])
    :param missingness_types: List of missingness types (e.g., ["MCAR", "MAR", "MNAR"])
    """

    # Main directory where reprocessed data should be saved
    main_directory = str(experiments_dir(columns_num))
    os.makedirs(main_directory, exist_ok=True)

    # Iterate over each combination of fraction and missingness
    for fraction in fractions:
        for missingness in missingness_types:
            
            # Define subfolder name for this combination
            subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
            full_path = fraction_dir(columns_num, fraction, missingness)
            full_path.mkdir(parents=True, exist_ok=True)
            os.makedirs(full_path, exist_ok=True)

            # Define filenames
            filename_corruption = f'Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}.csv'
            corrupted_data_path = os.path.join(full_path, filename_corruption)

            filename_mask = f'missing_mask_col_{columns_num}.csv'
            missing_mask_path = os.path.join(full_path, filename_mask)

            filename_normalized_data = f'normalized_data_col_{columns_num}.csv'
            normalized_data_path = os.path.join(full_path, filename_normalized_data)

            # Check if the directory already has files
            if os.listdir(full_path):
                print(f"The required input files exist for fraction={fraction}, missingness={missingness}.")
                print("Skipping creation and moving on to imputation.")
            else:
                print(f"Directory is empty for fraction={fraction}, missingness={missingness}.")
                print("Running preprocessing_lable.py to generate corrupted data, missing mask, and normalized data.")
                # Call your external script
                subprocess.run([
                    "python3",
                    str(PREPROCESSING_DIR / "preprocessing_lable.py"),
                    file_path,
                    str(columns_num),
                    str(fraction),
                    missingness,
                    corrupted_data_path,
                    missing_mask_path,
                    normalized_data_path
                ])
