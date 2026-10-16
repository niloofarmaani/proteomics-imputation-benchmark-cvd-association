import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Make sure config_paths.py can be imported
# ---------------------------------------------------------------------
def add_project_root_to_syspath():
    """
    Tries to find config_paths.py by walking up parent directories
    and adds the correct root to sys.path.

    Supports layouts like:
      project_root/config_paths.py
      project_root/code/config_paths.py
    """
    here = Path(__file__).resolve()

    candidates = []
    for i in range(len(here.parents)):
        candidates.append(here.parents[i])
        candidates.append(here.parents[i] / "code")

    for root in candidates:
        if (root / "config_paths.py").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return

    raise ImportError("Could not find config_paths.py. Adjust add_project_root_to_syspath().")


# ---------------------------------------------------------------------
# Robust RMSE reader
# ---------------------------------------------------------------------
def read_first_numeric_rmse(csv_path: str) -> float:
    """
    Reads an RMSE CSV and returns the first numeric value found.
    Works for files like:
      rmse_vae
      0.123
      rmse_vae, other_col
    """
    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        raise ValueError(f"Empty RMSE file: {csv_path}")

    # Try first row, any column
    for v in df.iloc[0].values:
        try:
            x = float(v)
            if pd.notna(x):
                return x
        except Exception:
            pass

    # Fallback: numeric columns
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] > 0:
        return float(num_df.iloc[0, 0])

    raise ValueError(f"No numeric RMSE found in {csv_path}")


# ---------------------------------------------------------------------
# Find best RMSE file for a method in one folder
# ---------------------------------------------------------------------
def find_lowest_rmse_file(
    folder: str,
    method: str,
    columns_num: str,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Finds the lowest RMSE among files matching:
      RMSE_{method}_*_col_{columns_num}*.csv

    Supports both:
      RMSE_VAE_col_all_params_*.csv
      RMSE_VAE_best_col_all.csv
    """
    if not os.path.isdir(folder):
        return None, None

    files = os.listdir(folder)
    if not files:
        return None, None

    method_l = method.lower()
    col_tag_l = f"col_{str(columns_num).lower()}"

    best_val = float("inf")
    best_file = None

    for fn in files:
        fn_l = fn.lower()
        if not fn_l.endswith(".csv"):
            continue
        if not fn_l.startswith(f"rmse_{method_l}"):
            continue
        if col_tag_l not in fn_l:
            continue

        fp = os.path.join(folder, fn)
        try:
            val = read_first_numeric_rmse(fp)
            if val < best_val:
                best_val = val
                best_file = fp
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    if best_file is None:
        return None, None

    return float(best_val), best_file


# ---------------------------------------------------------------------
# Main comparison + plotting
# ---------------------------------------------------------------------
def compare_imputation_methods(
    methods: List[str],
    fractions: List[float],
    missingness_types: List[str],
    columns_num: str,
    log_file_path: str,
):
    """
    For each method / missingness / fraction:
      - finds the lowest RMSE CSV
      - logs the chosen file path
      - plots RMSE vs missing fraction
    """
    add_project_root_to_syspath()
    from config_paths import experiments_dir  # noqa

    main_directory = str(experiments_dir(columns_num))

    # Store results
    best_rmse: Dict[str, Dict[str, Dict[float, float]]] = {
        m: {miss: {} for miss in missingness_types} for m in methods
    }
    best_paths: Dict[str, Dict[str, Dict[float, str]]] = {
        m: {miss: {} for miss in missingness_types} for m in methods
    }

    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"EXPERIMENTS_DIR: {main_directory}\n\n")

        for method in methods:
            for missingness in missingness_types:
                for fraction in fractions:
                    subfolder = f"fraction_{fraction}_missingness_{missingness}"
                    full_path = os.path.join(main_directory, subfolder)

                    rmse_val, rmse_file = find_lowest_rmse_file(
                        full_path, method, columns_num
                    )

                    if rmse_file is None:
                        msg = (
                            f"No RMSE file found | Method={method}, "
                            f"Missingness={missingness}, Fraction={fraction}"
                        )
                        print(msg)
                        log_file.write(msg + "\n")
                        continue

                    best_rmse[method][missingness][fraction] = rmse_val
                    best_paths[method][missingness][fraction] = rmse_file

                    log_file.write(
                        f"Method: {method}, Missingness: {missingness}, "
                        f"Fraction: {fraction}, File: {rmse_file}, RMSE: {rmse_val}\n"
                    )

    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------
    num_missing_types = len(missingness_types)
    fig, axes = plt.subplots(
        1, num_missing_types, figsize=(6 * num_missing_types, 5), sharey=True
    )

    if num_missing_types == 1:
        axes = [axes]

    x_vals = [float(f) for f in fractions]

    for i, miss in enumerate(missingness_types):
        ax = axes[i]
        for method in methods:
            y_vals = []
            for frac in fractions:
                y_vals.append(best_rmse[method][miss].get(frac, float("nan")))

            ax.plot(x_vals, y_vals, marker="o", label=method)

        ax.set_title(f"{miss} Missingness")
        ax.set_xlabel("Missing Fraction")
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(f) for f in fractions])
        if i == 0:
            ax.set_ylabel("RMSE")

        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    pdf_path = log_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to: {pdf_path}")

    plt.show()
    plt.close(fig)
