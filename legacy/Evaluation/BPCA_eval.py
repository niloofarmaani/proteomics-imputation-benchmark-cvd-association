import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Make sure .../project_root/code is on sys.path
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir


def BPCA_eval(
    fractions,
    missingness_types,
    columns_num
):
    """
    Gathers and plots RMSE values for specified missingness types and fractions,
    using CSV files named like:
      RMSE_bpca_col_{columns_num}_npcs_{npcs_value}_maxSteps_{maxSteps_value}.csv

    For each subfolder (fraction_{f}_missingness_{m}), we look for files matching
    that pattern and parse out the 'npcs_value' and 'maxSteps_value' from the filename.
    Then we plot RMSE vs. fractions for each (npcs, maxSteps) combination.
    """

    # Main directory where subfolders (fraction_x_missingness_y) reside
    main_directory = str(experiments_dir(columns_num))


    # Initialize a dictionary to hold RMSE data for each missingness type
    # keyed by (npcs_value, maxSteps_value)
    rmse_data = {missingness: {} for missingness in missingness_types}

    # Regex pattern to match filenames like:
    # RMSE_bpca_col_{columns_num}_npcs_(\d+)_maxSteps_(\d+).csv
    pattern = rf"^RMSE_bpca_col_{columns_num}_npcs_(\d+)_maxSteps_(\d+)\.csv$"

    # Loop over each missingness type and fraction to gather RMSE values
    for missingness in missingness_types:
        for fraction in fractions:
            subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
            full_path = os.path.join(main_directory, subfolder_name)

            # Skip if folder doesn't exist or is empty
            if not os.path.isdir(full_path) or not os.listdir(full_path):
                print(f"No files in {full_path} for fraction={fraction}, missingness={missingness}.")
                continue

            # Look for BPCA RMSE files in this folder
            for filename in os.listdir(full_path):
                match = re.match(pattern, filename)
                if match:
                    npcs_value = int(match.group(1))
                    maxSteps_value = int(match.group(2))

                    # Create a tuple key
                    key = (npcs_value, maxSteps_value)
                    if key not in rmse_data[missingness]:
                        rmse_data[missingness][key] = []

                    rmse_path = os.path.join(full_path, filename)

                    # Read the RMSE value (assumes the first row, first column)
                    try:
                        rmse_value = pd.read_csv(rmse_path).iloc[0, 0]
                        rmse_data[missingness][key].append(rmse_value)
                    except Exception as e:
                        print(f"Error reading {rmse_path}: {e}")
                        rmse_data[missingness][key].append(None)

    # Create as many subplots as there are missingness types
    num_missing_types = len(missingness_types)
    fig, axes = plt.subplots(
        1,
        num_missing_types,
        figsize=(6 * num_missing_types, 5),
        sharey=True
    )

    # If there's only one missingness type, axes might not be a list
    if num_missing_types == 1:
        axes = [axes]

    # Plot data for each missingness type
    for i, missingness in enumerate(missingness_types):
        ax = axes[i]
        # For each (npcs, maxSteps) combo, we have a list of RMSEs in the same order as 'fractions'
        for (npcs_value, maxSteps_value), rmse_values in rmse_data[missingness].items():
            label = f"nPcs={npcs_value}, maxSteps={maxSteps_value}"

            # Ensure fractions and rmse_values align in length
            if len(rmse_values) == len(fractions):
                ax.plot(fractions, rmse_values, marker='o', label=label)
            else:
                # Possibly missing fraction subdirectories or files
                print(f"Warning: mismatch in fraction count vs. RMSE count for {label}")

        ax.set_title(f'{missingness} Missingness')
        ax.set_xlabel('Missing Fraction')
        ax.set_xticks(fractions)
        if i == 0:
            ax.set_ylabel('RMSE')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
