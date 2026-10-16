import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Make sure .../project_root/code is on sys.path
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir  # type: ignore


def _fraction_to_str(f: Any) -> str:
    # Robust folder naming for floats
    if isinstance(f, float):
        return f"{f:g}"
    return str(f)


def _read_rmse_cols(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Reads single and multi RMSE from a gapVAE RMSE csv.
    Prefers *_full columns, falls back to non-full if needed.
    """
    candidates_single = ["rmse_single_full", "rmse_single"]
    candidates_multi = ["rmse_multi_full", "rmse_multi"]

    single_col = next((c for c in candidates_single if c in df.columns), None)
    multi_col = next((c for c in candidates_multi if c in df.columns), None)

    if single_col is None or multi_col is None:
        raise KeyError(
            f"Expected columns not found. Need one of {candidates_single} and one of {candidates_multi}. "
            f"Found: {list(df.columns)}"
        )

    rmse_single = float(df[single_col].iloc[0])
    rmse_multi = float(df[multi_col].iloc[0])
    return rmse_single, rmse_multi


def _set_tight_ylim(ax, y_values: np.ndarray, pad_frac: float = 0.08) -> None:
    """
    Tighten y-limits to data range with a small padding.
    """
    y = y_values[np.isfinite(y_values)]
    if y.size == 0:
        return

    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if np.isclose(y_min, y_max):
        # If flat line, give a small symmetric window
        delta = 0.05 * (abs(y_min) + 1.0)
        ax.set_ylim(y_min - delta, y_max + delta)
        return

    pad = pad_frac * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)


def multi_eval(
    fractions,
    missingness_types,
    columns_num,
    log_file_path: Optional[str] = None,
):
    """
    - Looks for RMSE_gapVAE files in each fraction/missingness folder.
    - Picks the best file per folder (lowest multi RMSE) if multiple exist.
    - Plots Single vs Multi RMSE (top row) and the gap (bottom row).
    - Writes a log file similar to compare_imputation_methods and saves PDF next to it.
    """
    main_directory = str(experiments_dir(columns_num))

    # Default log location inside the experiment root
    if log_file_path is None:
        eval_dir = Path(main_directory) / "Evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = str(eval_dir / f"multi_eval_gapVAE_col_{columns_num}.log")

    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = log_path.with_suffix(".pdf")

    # Storage: rmse_by_missingness[missingness] = {"single": [...], "multi": [...], "gap": [...], "paths": [...]}
    rmse_by_missingness: Dict[str, Dict[str, List[float]]] = {
        miss: {"single": [], "multi": [], "gap": []} for miss in missingness_types
    }
    chosen_paths: Dict[str, List[str]] = {miss: [] for miss in missingness_types}

    col_str = str(columns_num)
    pattern = re.compile(rf"^RMSE_gapVAE_col_{re.escape(col_str)}_params_(.+)\.csv$")

    with open(log_file_path, "w") as log_file:
        for missingness in missingness_types:
            for frac in fractions:
                frac_str = _fraction_to_str(frac)
                subfolder_name = f"fraction_{frac_str}_missingness_{missingness}"
                full_path = os.path.join(main_directory, subfolder_name)

                if not os.path.isdir(full_path) or not os.listdir(full_path):
                    # keep alignment with fractions even if missing
                    rmse_by_missingness[missingness]["single"].append(float("nan"))
                    rmse_by_missingness[missingness]["multi"].append(float("nan"))
                    rmse_by_missingness[missingness]["gap"].append(float("nan"))
                    chosen_paths[missingness].append("")
                    print(f"No files found in {full_path} for gapVAE")
                    continue

                best_file = None
                best_rmse_multi = float("inf")
                best_rmse_single = float("nan")

                for filename in os.listdir(full_path):
                    if not pattern.match(filename):
                        continue

                    file_path = os.path.join(full_path, filename)
                    try:
                        df = pd.read_csv(file_path)
                        rmse_single, rmse_multi = _read_rmse_cols(df)

                        # pick lowest multi RMSE if multiple RMSE_gapVAE exist
                        if np.isfinite(rmse_multi) and rmse_multi < best_rmse_multi:
                            best_rmse_multi = rmse_multi
                            best_rmse_single = rmse_single
                            best_file = file_path

                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

                if best_file is None:
                    rmse_s = float("nan")
                    rmse_m = float("nan")
                    gap = float("nan")
                    chosen_paths[missingness].append("")
                else:
                    rmse_s = float(best_rmse_single)
                    rmse_m = float(best_rmse_multi)
                    gap = rmse_s - rmse_m
                    chosen_paths[missingness].append(best_file)

                    # Log format aligned with your compare_imputation_methods style
                    log_file.write(
                        f"Method: gapVAE_single, Missingness: {missingness}, "
                        f"Fraction: {frac}, File: {best_file}, RMSE: {rmse_s}\n"
                    )
                    log_file.write(
                        f"Method: gapVAE_multi, Missingness: {missingness}, "
                        f"Fraction: {frac}, File: {best_file}, RMSE: {rmse_m}\n"
                    )

                rmse_by_missingness[missingness]["single"].append(rmse_s)
                rmse_by_missingness[missingness]["multi"].append(rmse_m)
                rmse_by_missingness[missingness]["gap"].append(gap)

    # -----------------------------
    # Plot
    # -----------------------------
    num_missing_types = len(missingness_types)

    fig, axes = plt.subplots(
        2,
        num_missing_types,
        figsize=(5.2 * num_missing_types, 6.0),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    if num_missing_types == 1:
        # axes becomes shape (2,), normalize to (2,1)
        axes = np.array([[axes[0]], [axes[1]]])

    # Use numeric x for plotting
    x = np.array([float(f) for f in fractions], dtype=float)

    for j, missingness in enumerate(missingness_types):
        ax_top = axes[0, j]
        ax_bot = axes[1, j]

        y_single = np.array(rmse_by_missingness[missingness]["single"], dtype=float)
        y_multi = np.array(rmse_by_missingness[missingness]["multi"], dtype=float)
        y_gap = np.array(rmse_by_missingness[missingness]["gap"], dtype=float)

        # Top: RMSE curves
        ax_top.plot(x, y_single, marker="o", linewidth=2.0, label="Single")
        ax_top.plot(x, y_multi, marker="s", linewidth=2.0, linestyle="--", label="Multi")

        # Shade difference to make it pop
        finite_mask = np.isfinite(y_single) & np.isfinite(y_multi) & np.isfinite(x)
        if np.any(finite_mask):
            ax_top.fill_between(
                x[finite_mask],
                y_single[finite_mask],
                y_multi[finite_mask],
                alpha=0.15,
                step=None,
            )

        ax_top.set_title(f"{missingness} Missingness")
        ax_top.grid(True)

        # Tight y-limits to reduce empty space (per subplot)
        _set_tight_ylim(ax_top, np.concatenate([y_single, y_multi]))

        # Bottom: gap curve
        ax_bot.axhline(0.0, linewidth=1.0)
        ax_bot.plot(x, y_gap, marker="d", linewidth=1.8, label="Gap (Single - Multi)")
        ax_bot.grid(True)

        _set_tight_ylim(ax_bot, y_gap)

        # x ticks only on bottom
        ax_bot.set_xlabel("Missing Fraction")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([_fraction_to_str(f) for f in fractions])

        if j == 0:
            ax_top.set_ylabel("RMSE")
            ax_bot.set_ylabel("Gap")

        # Legends: keep top legend only to reduce clutter
        ax_top.legend(loc="best")

    # Save PDF next to log file (same behavior as your compare function)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved log to:  {log_path}")
    print(f"Saved plot to: {pdf_path}")

    plt.show()
    plt.close(fig)
