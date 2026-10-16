import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Make sure .../project_root/code is on sys.path
# (Assumes Evaluation/ is inside .../code/Evaluation/)
# ---------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir  # type: ignore


def _fraction_to_str(f: Any) -> str:
    """
    Folder names are built like f"fraction_{fraction}_missingness_{missingness}".
    If fractions are floats, using :g avoids ugly representations like 0.0100000000002.
    """
    if isinstance(f, float):
        return f"{f:g}"
    return str(f)


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Pick the first column that exists from candidates.
    Raise KeyError if none exist.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns exist. Tried: {candidates}. Found: {list(df.columns)}")


def multi_eval(fractions, missingness_types, columns_num):
    main_directory = str(experiments_dir(columns_num))

    # rmse_maps[(missingness, param_id, "single" or "multi")][fraction] = value
    rmse_maps: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    # Pattern for RMSE files produced by your gapVAE runner
    col_str = str(columns_num)
    pattern = re.compile(rf"^RMSE_gapVAE_col_{re.escape(col_str)}_params_(.+)\.csv$")

    # Scan folders
    for missingness in missingness_types:
        for frac in fractions:
            frac_str = _fraction_to_str(frac)
            subfolder_name = f"fraction_{frac_str}_missingness_{missingness}"
            full_path = os.path.join(main_directory, subfolder_name)

            if not os.path.exists(full_path):
                continue

            # find all RMSE_gapVAE files in this folder
            for filename in os.listdir(full_path):
                m = pattern.match(filename)
                if not m:
                    continue

                param_id = m.group(1)
                rmse_path = os.path.join(full_path, filename)

                try:
                    df = pd.read_csv(rmse_path)

                    # Prefer "full" metrics from the new RMSE file
                    single_col = _pick_col(df, ["rmse_single_full", "rmse_single"])
                    multi_col = _pick_col(df, ["rmse_multi_full", "rmse_multi"])

                    rmse_single = float(df[single_col].iloc[0])
                    rmse_multi = float(df[multi_col].iloc[0])

                except Exception:
                    # If file is corrupted or missing columns, record NaN so plot can still render
                    rmse_single = float("nan")
                    rmse_multi = float("nan")

                key_s = (missingness, param_id, "single")
                key_m = (missingness, param_id, "multi")
                rmse_maps.setdefault(key_s, {})[frac_str] = rmse_single
                rmse_maps.setdefault(key_m, {})[frac_str] = rmse_multi

    # If nothing found, fail loudly with helpful message
    if len(rmse_maps) == 0:
        raise FileNotFoundError(
            f"No RMSE_gapVAE files found under: {main_directory}\n"
            f"Expected filenames like: RMSE_gapVAE_col_{columns_num}_params_*.csv"
        )

    # Prepare plot
    num_missing_types = len(missingness_types)
    fig, axes = plt.subplots(
        1,
        num_missing_types,
        figsize=(6 * num_missing_types, 5),
        sharey=True,
    )
    if num_missing_types == 1:
        axes = [axes]

    frac_strs = [_fraction_to_str(f) for f in fractions]
    x_vals = list(range(len(frac_strs)))  # use integer x so missing values align cleanly

    # Determine all param_ids per missingness
    for i, missingness in enumerate(missingness_types):
        ax = axes[i]

        # Collect param_ids seen for this missingness
        param_ids = sorted(
            {pid for (miss, pid, which) in rmse_maps.keys() if miss == missingness}
        )

        for param_id in param_ids:
            # Build y arrays aligned to fractions
            ys_single = []
            ys_multi = []

            for fs in frac_strs:
                ys_single.append(rmse_maps.get((missingness, param_id, "single"), {}).get(fs, np.nan))
                ys_multi.append(rmse_maps.get((missingness, param_id, "multi"), {}).get(fs, np.nan))

            # Short label like your old eval
            short_id = (param_id[:10] + "...") if len(param_id) > 10 else param_id

            # Plot both lines for the same param_id
            ax.plot(x_vals, ys_single, marker="o", linestyle="-", label=f"Single | {short_id}")
            ax.plot(x_vals, ys_multi, marker="s", linestyle="--", label=f"Multi  | {short_id}")

        ax.set_title(f"{missingness} Missingness")
        ax.set_xlabel("Missing Fraction")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(frac_strs, rotation=0)

        if i == 0:
            ax.set_ylabel("RMSE")

        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
