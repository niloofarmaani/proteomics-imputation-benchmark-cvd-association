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


def MICE_eval(
    fractions,
    missingness_types,
    columns_num
):
    """
    Gathers and plots RMSE values for specified missingness types and fractions,
    using CSV files named like:
      RMSE_mice_col_{columns_num}_m_{m_value}_maxit_{maxit_value}.csv

    Parameters
    ----------
    fractions : list of float
        Missing data fractions to evaluate (e.g., [0.01, 0.1, 0.3]).
    missingness_types : list of str
        Missingness mechanisms (e.g., ["MCAR", "MAR", "MNAR"]).
    columns_num : int
        The column specification (e.g. 10).
    """

    # Directory structure: Experiments/{columns_num}/fraction_X_missingness_Y
    main_directory = str(experiments_dir(columns_num))


    # Initialize a dictionary to hold RMSE data for each missingness type
    # We'll key by (m_value, maxit_value) in the filename
    rmse_data = {missingness: {} for missingness in missingness_types}

    # Pattern to match filenames like: RMSE_mice_col_10_m_3_maxit_1.csv
    pattern = rf"^RMSE_mice_col_{columns_num}_m_(\d+)_maxit_(\d+)\.csv$"

    # Loop over each missingness type and fraction, gather RMSE values
    for missingness in missingness_types:
        for fraction in fractions:
            subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
            full_path = os.path.join(main_directory, subfolder_name)

            # If the subfolder doesn't exist or is empty, skip
            if not os.path.exists(full_path) or not os.listdir(full_path):
                print(f"No data found in {full_path} for fraction={fraction}, missingness={missingness}.")
                continue

            # Check each file in the subfolder
            for filename in os.listdir(full_path):
                match = re.match(pattern, filename)
                if match:
                    m_value = int(match.group(1))      # captured from _m_(\d+)
                    maxit_value = int(match.group(2)) # captured from _maxit_(\d+)

                    # Create a unique key for these parameters
                    key = (m_value, maxit_value)
                    if key not in rmse_data[missingness]:
                        # We'll store RMSE in a list, each index matching the fraction order
                        rmse_data[missingness][key] = []

                    rmse_file_path = os.path.join(full_path, filename)

                    # Read the RMSE (assuming first row, first column)
                    try:
                        rmse_value = pd.read_csv(rmse_file_path).iloc[0, 0]
                        # Append the RMSE for this fraction at the end of the list
                        rmse_data[missingness][key].append(rmse_value)
                    except Exception as e:
                        print(f"Error reading {rmse_file_path}: {e}")
                        rmse_data[missingness][key].append(None)

    # Now we have, for each missingness, a dictionary: { (m, maxit): [rmse_f1, rmse_f2, ...] }
    # where the i-th value in that list corresponds to fractions[i].

    num_missing_types = len(missingness_types)
    fig, axes = plt.subplots(
        1,
        num_missing_types,
        figsize=(6 * num_missing_types, 5),
        sharey=True
    )

    # If there's only one missingness type, axes won't be a list
    if num_missing_types == 1:
        axes = [axes]

    # Plot the data for each missingness type
    for i, missingness in enumerate(missingness_types):
        ax = axes[i]
        # For each (m, maxit) combo, we have a list of RMSE values in the same order as 'fractions'
        for (m_value, maxit_value), rmse_values in rmse_data[missingness].items():
            label = f"m={m_value}, maxit={maxit_value}"
            # Make sure fractions and rmse_values align in length
            # We assume the user gave the same fraction list and we appended in that order
            if len(rmse_values) == len(fractions):
                ax.plot(fractions, rmse_values, marker='o', label=label)
            else:
                # If there's a mismatch (some fraction directories missing?), handle gracefully
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
