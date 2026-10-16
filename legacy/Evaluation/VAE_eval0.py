# Save as Evaluation/VAE_eval.py
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


def vae_eval(fractions, missingness_types, columns_num):
    main_directory = str(experiments_dir(columns_num))
    rmse_data = {missingness: {} for missingness in missingness_types}

    for missingness in missingness_types:
        for fraction in fractions:
            subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
            full_path = os.path.join(main_directory, subfolder_name)
            param_grid_ids_in_folder = []
            if not os.path.exists(full_path):
                continue  # Skip if directory doesn’t exist
            for filename in os.listdir(full_path):
                match = re.match(f'RMSE_VAE_col_{columns_num}_params_(.+).csv', filename)
                if match:
                    param_grid_id = match.group(1)
                    if param_grid_id not in rmse_data[missingness]:
                        rmse_data[missingness][param_grid_id] = []
                    param_grid_ids_in_folder.append(param_grid_id)

            for param_grid_id in set(param_grid_ids_in_folder):
                rmse_path_vae = os.path.join(full_path, f'RMSE_VAE_col_{columns_num}_params_{param_grid_id}.csv')
                try:
                    rmse_value = pd.read_csv(rmse_path_vae)['rmse_vae'].iloc[0]
                    rmse_data[missingness][param_grid_id].append(rmse_value)
                except (FileNotFoundError, KeyError, IndexError) as e:
                    rmse_data[missingness][param_grid_id].append(None)

    num_missing_types = len(missingness_types)
    fig, axes = plt.subplots(1, num_missing_types, figsize=(6 * num_missing_types, 5), sharey=True)
    if num_missing_types == 1:
        axes = [axes]

    for i, missingness in enumerate(missingness_types):
        ax = axes[i]
        for param_grid_id, rmse_values in rmse_data[missingness].items():
            short_id = param_grid_id[:10] + "..." if len(param_grid_id) > 10 else param_grid_id
            ax.plot(fractions, rmse_values, marker='o', label=f'Params={short_id}')
        ax.set_title(f'{missingness} Missingness')
        ax.set_xlabel('Missing Fraction')
        ax.set_xticks(fractions)
        if i == 0:
            ax.set_ylabel('RMSE')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()