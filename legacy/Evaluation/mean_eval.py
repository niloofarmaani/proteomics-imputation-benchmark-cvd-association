import os
import re  # Not strictly necessary now, but left if you need further pattern matching
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Make sure .../project_root/code is on sys.path
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir


def mean_eval(fractions, missingness_types, columns_num):
    """
    Gathers and plots RMSE values for Mean Imputation, with CSV files named like:
      RMSE_Mean_col_{columns_num}.csv
    located in subdirectories of the form: fraction_{fraction}_missingness_{missingness}.

    Parameters
    ----------
    fractions : list of float
        Fractions of data missing (e.g. [0.01, 0.1, 0.3]).
    missingness_types : list of str
        Types of missingness (e.g. ["MCAR", "MAR", "MNAR"]).
    columns_num : int
        The column number of interest (e.g. 10).
    """

    # Main directory where subfolders reside
    main_directory = str(experiments_dir(columns_num))


    # Initialize a dictionary to hold RMSE data for each missingness type.
    # Each missingness will map to a list of RMSE values, one for each fraction.
    rmse_data = {missingness: [] for missingness in missingness_types}

    # Loop through each missingness type and fraction to gather RMSE values
    for missingness in missingness_types:
        for fraction in fractions:
            subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
            full_path = os.path.join(main_directory, subfolder_name)

            # The expected file: RMSE_Mean_col_{columns_num}.csv
            rmse_file = os.path.join(full_path, f'RMSE_Mean_col_{columns_num}.csv')

            # Attempt to read the RMSE value if the file exists
            if os.path.isfile(rmse_file):
                try:
                    # Assumes the first row, first column contains the RMSE
                    rmse_value = pd.read_csv(rmse_file).iloc[0, 0]
                    rmse_data[missingness].append(rmse_value)
                except Exception as e:
                    print(f"Error reading {rmse_file}: {e}")
                    rmse_data[missingness].append(None)
            else:
                print(f"No RMSE file found: {rmse_file}")
                rmse_data[missingness].append(None)

    # Create a number of subplots equal to the number of missingness types
    num_missing_types = len(missingness_types)
    fig, axes = plt.subplots(
        1, 
        num_missing_types, 
        figsize=(6 * num_missing_types, 5), 
        sharey=True
    )

    # If there's only one missingness type, axes is not a list—make it one for consistency
    if num_missing_types == 1:
        axes = [axes]

    # Plot the RMSE values for each missingness type
    for i, missingness in enumerate(missingness_types):
        ax = axes[i]
        ax.plot(
            fractions,
            rmse_data[missingness],
            marker='o',
            label='Mean Imputation'
        )
        ax.set_title(f'{missingness} Missingness')
        ax.set_xlabel('Missing Fraction')
        ax.set_xticks(fractions)
        if i == 0:
            ax.set_ylabel('RMSE')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
