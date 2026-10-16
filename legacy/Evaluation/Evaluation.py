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


def compare_imputation_methods(methods, fractions, missingness_types, columns_num, log_file_path):
    """
    Compares multiple imputation methods by finding the lowest RMSE files for each method,
    missingness type, and fraction, logs the file paths, and plots the best RMSE values.
    """

    main_directory = str(experiments_dir(columns_num))

    best_rmse = {method: {miss: {} for miss in missingness_types} for method in methods}
    best_rmse_paths = {method: {miss: {} for miss in missingness_types} for method in methods}

    # Ensure log directory exists
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Open log file for writing
    with open(log_file_path, "w") as log_file:
        for method in methods:
            pattern = rf"^RMSE_{method}_col_{columns_num}.*\.csv$"

            for missingness in missingness_types:
                for fraction in fractions:
                    subfolder_name = f"fraction_{fraction}_missingness_{missingness}"
                    full_path = os.path.join(main_directory, subfolder_name)

                    if not os.path.isdir(full_path) or not os.listdir(full_path):
                        print(f"No files found in {full_path} for method={method}")
                        continue

                    lowest_rmse = float("inf")
                    lowest_file = None

                    for filename in os.listdir(full_path):
                        if re.match(pattern, filename):
                            file_path = os.path.join(full_path, filename)
                            try:
                                rmse_value = pd.read_csv(file_path).iloc[0, 0]
                                if rmse_value < lowest_rmse:
                                    lowest_rmse = rmse_value
                                    lowest_file = file_path
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")

                    if lowest_file is not None:
                        best_rmse[method][missingness][fraction] = lowest_rmse
                        best_rmse_paths[method][missingness][fraction] = lowest_file
                        log_file.write(
                            f"Method: {method}, Missingness: {missingness}, "
                            f"Fraction: {fraction}, File: {lowest_file}, RMSE: {lowest_rmse}\n"
                        )

    # Plotting
    num_missing_types = len(missingness_types)
    fig, axes = plt.subplots(
        1, num_missing_types, figsize=(6 * num_missing_types, 5), sharey=True
    )

    if num_missing_types == 1:
        axes = [axes]

    for i, miss in enumerate(missingness_types):
        ax = axes[i]
        for method in methods:
            rmse_values = []
            for frac in fractions:
                rmse_val = best_rmse[method][miss].get(frac)
                rmse_values.append(rmse_val if rmse_val is not None else float("nan"))

            ax.plot(fractions, rmse_values, marker="o", label=method)

        ax.set_title(f"{miss} Missingness")
        ax.set_xlabel("Missing Fraction")
        ax.set_xticks(fractions)

        if i == 0:
            ax.set_ylabel("RMSE")

        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    # Save plot as PDF in the same directory as the log file
    pdf_path = log_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to: {pdf_path}")

    plt.show()
    plt.close(fig)
