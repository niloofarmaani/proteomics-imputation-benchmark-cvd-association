import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir


def gan_eval(fractions, missingness_types, columns_num, max_curves=None):
    return _rmse_eval(
        fractions=fractions,
        missingness_types=missingness_types,
        columns_num=columns_num,
        rmse_prefix="RMSE_GAN",
        title_prefix="GAN",
        max_curves=max_curves,
    )


def _rmse_eval(fractions, missingness_types, columns_num, rmse_prefix, title_prefix, max_curves=None):
    main_directory = str(experiments_dir(columns_num))

    all_param_ids = set()
    pattern = re.compile(rf"^{rmse_prefix}_col_{re.escape(str(columns_num))}_params_(.+)\.csv$")

    for missingness in missingness_types:
        for fraction in fractions:
            full_path = os.path.join(main_directory, f"fraction_{fraction}_missingness_{missingness}")
            if not os.path.exists(full_path):
                continue
            for filename in os.listdir(full_path):
                m = pattern.match(filename)
                if m:
                    all_param_ids.add(m.group(1))

    rmse_data = {missingness: {} for missingness in missingness_types}
    for missingness in missingness_types:
        for param_id in all_param_ids:
            rmse_data[missingness][param_id] = [np.nan] * len(fractions)

    for missingness in missingness_types:
        for f_idx, fraction in enumerate(fractions):
            full_path = os.path.join(main_directory, f"fraction_{fraction}_missingness_{missingness}")
            if not os.path.exists(full_path):
                continue

            for param_id in all_param_ids:
                rmse_path = os.path.join(full_path, f"{rmse_prefix}_col_{columns_num}_params_{param_id}.csv")
                if not os.path.exists(rmse_path):
                    continue
                try:
                    rmse_value = pd.read_csv(rmse_path)["rmse_vae"].iloc[0]
                    rmse_data[missingness][param_id][f_idx] = rmse_value
                except (KeyError, IndexError, pd.errors.EmptyDataError):
                    rmse_data[missingness][param_id][f_idx] = np.nan

    num_missing_types = len(missingness_types)
    fig, axes = plt.subplots(1, num_missing_types, figsize=(6 * num_missing_types, 5), sharey=True)
    if num_missing_types == 1:
        axes = [axes]

    for i, missingness in enumerate(missingness_types):
        ax = axes[i]
        items = list(rmse_data[missingness].items())

        if max_curves is not None:
            scored = []
            for pid, vals in items:
                v = np.array(vals, dtype=float)
                score = np.nanmean(v) if np.isfinite(v).any() else np.inf
                scored.append((score, pid, vals))
            scored.sort(key=lambda x: x[0])
            items = [(pid, vals) for _, pid, vals in scored[:max_curves]]

        for param_id, rmse_values in items:
            short_id = param_id[:12] + "..." if len(param_id) > 12 else param_id
            ax.plot(fractions, rmse_values, marker="o", label=f"Params={short_id}")

        ax.set_title(f"{title_prefix} | {missingness} Missingness")
        ax.set_xlabel("Missing Fraction")
        ax.set_xticks(fractions)
        if i == 0:
            ax.set_ylabel("RMSE")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

    return rmse_data
