import os
import subprocess
from pathlib import Path
import sys

# Make sure .../project_root/code is on sys.path
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir


def lasso(imputation_methods, columns_num, fractions, missingness_types):
    """
    Reads a log file to find input files for LASSO feature selection based on the specified imputation method.
    Dynamically generates output directories as before.

    Parameters
    ----------
    log_file : str
        Path to the log file containing details of RMSE and file locations.
    imputation_method : str
        Imputation method to filter (e.g., "bpca", "KNN").
    columns_num : str
        Columns specification (e.g., "10").
    fractions : list of float
        Fractions of data to be processed (e.g., [0.01, 0.1]).
    missingness_types : list of str
        Types of missingness to process (e.g., ["MCAR", "MAR"]).
    """
    # Define the main experiment directory
    main_directory = str(experiments_dir(columns_num))
    log_file = os.path.join(main_directory, "best_rmse_log.txt")


    # Read the log file
    with open(log_file, "r") as file:
        lines = file.readlines()

    # Process the log lines and filter for the given imputation methods
    for line in lines:
        for method in imputation_methods:
            if f"Method: {method}," in line:
                # Parse the details from the log line
                parts = line.split(", ")
                file_path = None
                fraction = None
                missingness = None

                for part in parts:
                    if part.startswith("File:"):
                        file_path = part.replace("File: ", "").strip()
                    elif part.startswith("Fraction:"):
                        fraction = float(part.replace("Fraction: ", "").strip())
                    elif part.startswith("Missingness:"):
                        missingness = part.replace("Missingness: ", "").strip()

                # Skip if fraction or missingness do not match the provided lists
                if not file_path or fraction not in fractions or missingness not in missingness_types:
                    continue

                # Construct the imputed file path by replacing "RMSE" with "imputed"
                input_file = file_path.replace("RMSE", "imputed")

                # Dynamically generate the output directory
                subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
                output_directory = os.path.join(main_directory, subfolder_name)
                os.makedirs(output_directory, exist_ok=True)

                # Construct the output file path
                output_file_name = os.path.basename(input_file).replace("imputed", "features", 1)
                output_file = os.path.join(output_directory, output_file_name)

                # Print the processing details
                print(f"Processing input file: {input_file}")
                print(f"Output will be saved to: {output_file}")

                # Run the R script for LASSO feature selection
                subprocess.run([
                    "Rscript",
                    str(Path(__file__).resolve().parent / "Lasso.R"),
                    input_file,
                    output_file
                ]
                )
                print(f"Feature selection completed for {input_file}")

