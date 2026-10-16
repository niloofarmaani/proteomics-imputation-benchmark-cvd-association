import os
import subprocess
from pathlib import Path
import sys

# Make sure we can import config_paths from code/
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir, IMPUTATION_DIR


def BPCA(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    nPcs_list,
    maxSteps
):
    """
    Runs BPCA-based imputation for the given parameters.

    Parameters
    ----------
    file_path : str
        Path to the original CSV file (e.g., proteomics.csv).
    columns_num : int
        The column specification (e.g., number of columns or identifier).
    fractions : list of float
        Fractions of missing data to simulate (e.g., [0.01, 0.1, 0.3]).
    missingness_types : list of str
        Mechanisms of missingness (e.g., ["MCAR", "MAR", "MNAR"]).
    nPcs_list : list of int
        Number of principal components to use in BPCA (e.g., [2, 3, 5]).
    maxSteps : int
        Max iteration steps for BPCA (similar to an 'iter' parameter).
    """

    # Define the main directory for experiments (from config_paths)
    main_directory = str(experiments_dir(columns_num))


    # Iterate over each combination of fraction, missingness, and nPcs
    for fraction in fractions:
        for missingness in missingness_types:
            for nPcs in nPcs_list:
                # Create a subfolder based on fraction and missingness
                subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
                full_path = os.path.join(main_directory, subfolder_name)
                os.makedirs(full_path, exist_ok=True)

                # Define relevant file paths
                filename_corruption = (
                    f'Corrupted_data_fraction_{fraction}_missingness_{missingness}'
                    f'_col_{columns_num}.csv'
                )
                corrupted_data_path = os.path.join(full_path, filename_corruption)

                filename_mask = f'missing_mask_col_{columns_num}.csv'
                missing_mask_path = os.path.join(full_path, filename_mask)

                filename_normalized_data = f'normalized_data_col_{columns_num}.csv'
                normalized_data_path = os.path.join(full_path, filename_normalized_data)

                # Define RMSE and imputed output paths using nPcs and maxSteps
                filename_rmse_bpca = (
                    f'RMSE_bpca_col_{columns_num}_npcs_{nPcs}_maxSteps_{maxSteps}.csv'
                )
                rmse_path_bpca = os.path.join(full_path, filename_rmse_bpca)

                filename_imputed_bpca = (
                    f'imputed_bpca_col_{columns_num}_npcs_{nPcs}_maxSteps_{maxSteps}.csv'
                )
                imputed_path_bpca = os.path.join(full_path, filename_imputed_bpca)

                # Check if the directory has the necessary input files
                if len(os.listdir(full_path)) > 0:
                    print(
                        f"Files found for fraction={fraction}, missingness={missingness}, "
                        f"nPcs={nPcs}. Starting BPCA imputation..."
                    )

                    # Run the R script for BPCA imputation
                    # bpca_imputer.R expects 7 args:
                    # 1) corrupted_data_path
                    # 2) normalized_data_path
                    # 3) missing_mask_path
                    # 4) nPcs
                    # 5) maxSteps
                    # 6) rmse_path
                    # 7) imputed_data_path
                    subprocess.run([
                        "Rscript",
                        str(IMPUTATION_DIR / "BPCA_imputer.R"),
                        corrupted_data_path,
                        normalized_data_path,
                        missing_mask_path,
                        str(nPcs),
                        str(maxSteps),
                        rmse_path_bpca,
                        imputed_path_bpca
                    ]
                    )
                else:
                    print(
                        f"No files found in {full_path}. "
                        f"Please generate corrupted data and masks first."
                    )


