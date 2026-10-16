import os
import subprocess
from pathlib import Path
import sys

# Make sure we can import config_paths from code/
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir, IMPUTATION_DIR


def MICE(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    maxit,
    m_list
):
    """
    Runs MICE-based imputation for the given parameters.

    Parameters
    ----------
    file_path : str
        Path to the original CSV file (e.g., proteomics.csv).
    columns_num : int or str
        The column specification (e.g., number of columns or "all").
    fractions : list of float
        Fractions of missing data to simulate (e.g., [0.01, 0.1, 0.3]).
    missingness_types : list of str
        Mechanisms of missingness (e.g., ["MCAR", "MAR", "MNAR"]).
    maxit : int
        Number of MICE iterations per chain.
    m_list : list of int
        Each element represents MICE's 'm' (number of multiple imputations).
    """

    # Main directory for experiments
    main_directory = str(experiments_dir(columns_num))

    # Check if R script exists
    r_script_path = str(IMPUTATION_DIR / "MICE_imputer.R")

    if not os.path.exists(r_script_path):
        print(f"Error: R script not found at {r_script_path}")
        return

    # Iterate over each combination of fraction, missingness, and m_value
    for fraction in fractions:
        for missingness in missingness_types:
            for m_value in m_list:
                # Create the subfolder
                subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
                full_path = os.path.join(main_directory, subfolder_name)
                os.makedirs(full_path, exist_ok=True)

                # File paths
                filename_corruption = (
                    f'Corrupted_data_fraction_{fraction}_missingness_{missingness}'
                    f'_col_{columns_num}.csv'
                )
                corrupted_data_path = os.path.join(full_path, filename_corruption)

                filename_normalized_data = f'normalized_data_col_{columns_num}.csv'
                normalized_data_path = os.path.join(full_path, filename_normalized_data)

                # Define MICE output filenames
                filename_rmse_path_mice = (
                    f'RMSE_mice_col_{columns_num}_m_{m_value}_maxit_{maxit}.csv'
                )
                rmse_path_mice = os.path.join(full_path, filename_rmse_path_mice)

                filename_imputed_data_path = (
                    f'imputed_mice_col_{columns_num}_m_{m_value}_maxit_{maxit}.csv'
                )
                imputed_data_path = os.path.join(full_path, filename_imputed_data_path)

                # Check if the directory has necessary input files
                if os.path.exists(corrupted_data_path) and os.path.exists(normalized_data_path):
                    print(
                        f"Files found for fraction={fraction}, missingness={missingness}, "
                        f"m={m_value} (MICE). Starting imputation..."
                    )

                    # Run the R script for MICE imputation
                    process = subprocess.run(
                        [
                            "Rscript",
                            r_script_path,          
                            corrupted_data_path,
                            normalized_data_path,
                            str(maxit),
                            str(m_value),
                            rmse_path_mice,
                            imputed_data_path
                        ],
                        stdout=subprocess.PIPE,    # Capture standard output
                        stderr=subprocess.PIPE,    # Capture standard error
                        text=True                  # Convert bytes to string
                    )

                    # Print output and errors (for debugging)
                    if process.stdout:
                        print("STDOUT:", process.stdout)
                    if process.stderr:
                        print("STDERR:", process.stderr)

                else:
                    print(f"Required data files not found in {full_path}. Please generate corrupted data first.")