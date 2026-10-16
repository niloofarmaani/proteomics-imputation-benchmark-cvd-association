import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages

import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir


# ---------------------------------------------------------------------
# Method aliases
# ---------------------------------------------------------------------
METHOD_PATTERNS = {
    "VAE": r"(^|[_\-.])vae([_\-.]|$)",
    "DAE": r"(^|[_\-.])dae([_\-.]|$)",
    "AE": r"(^|[_\-.])ae([_\-.]|$)",
    "GAIN": r"(^|[_\-.])gain([_\-.]|$)",
    "GAN": r"(^|[_\-.])gan([_\-.]|$)",
    "RNN": r"(^|[_\-.])rnn([_\-.]|$)",
    "TRF": r"(^|[_\-.])(trf|transformer)([_\-.]|$)",
    "KNN": r"(^|[_\-.])knn([_\-.]|$)",
    "MEAN": r"(^|[_\-.])mean([_\-.]|$)",
    "MF": r"(^|[_\-.])(mf|missforest)([_\-.]|$)",
    "MISSFOREST": r"(^|[_\-.])(mf|missforest)([_\-.]|$)",
    "ITERATIVE": r"(^|[_\-.])(iterative|iterative_imputer)([_\-.]|$)",
    "SOFTIMPUTE": r"(^|[_\-.])(softimpute|softimpute_py)([_\-.]|$)",
    "MICE": r"(^|[_\-.])mice([_\-.]|$)",
    "BPCA": r"(^|[_\-.])bpca([_\-.]|$)",
}


def _canonical_method_name(method: str) -> str:
    m = str(method).strip()

    aliases = {
        "mf": "MF",
        "missforest": "MF",
        "iterative_imputer": "ITERATIVE",
        "iterative": "ITERATIVE",
        "softimpute": "SOFTIMPUTE",
        "softImpute": "SOFTIMPUTE",
        "softImpute_py": "SOFTIMPUTE",
        "trf": "TRF",
        "transformer": "TRF",
        "mean": "MEAN",
    }

    return aliases.get(m, aliases.get(m.lower(), m.upper()))


def _method_matches_trial_file(method: str, trial_path: Path) -> bool:
    canonical = _canonical_method_name(method)
    pattern = METHOD_PATTERNS.get(canonical)

    if pattern is None:
        pattern = rf"(^|[_\-.]){re.escape(canonical.lower())}([_\-.]|$)"

    stem = trial_path.stem.lower()
    return re.search(pattern, stem, flags=re.IGNORECASE) is not None


def _safe_fraction_token(frac: float) -> str:
    return str(frac).replace(".", "p")


# ---------------------------------------------------------------------
# Parse Optuna trial tables
# ---------------------------------------------------------------------
def _parse_score_rows(raw_value: Any) -> List[Dict[str, Any]]:
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        return raw_value

    if isinstance(raw_value, dict):
        if "score_rows" in raw_value:
            return raw_value["score_rows"]
        return [raw_value]

    try:
        if pd.isna(raw_value):
            return []
    except Exception:
        pass

    s = str(raw_value).strip()

    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return parsed.get("score_rows", [parsed])
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return parsed.get("score_rows", [parsed])
    except Exception:
        pass

    return []


def _extract_params_from_trial_row(row: pd.Series) -> Dict[str, Any]:
    params = {}

    for col in row.index:
        if str(col).startswith("params_"):
            key = str(col).replace("params_", "", 1)
            val = row[col]
            if not pd.isna(val):
                params[key] = val

    method_param_cols = [
        c for c in row.index
        if "method_params" in str(c).lower()
    ]

    for col in method_param_cols:
        raw = row[col]

        if pd.isna(raw):
            continue

        try:
            parsed = json.loads(str(raw))
        except Exception:
            try:
                parsed = ast.literal_eval(str(raw))
            except Exception:
                parsed = None

        if isinstance(parsed, dict):
            params.update(parsed)

    return params


def _find_trial_tables_for_method(
    method: str,
    trial_dir: Path,
    exclude_debug: bool = True,
    study_name_contains: Optional[str] = None,
    use_latest_study_per_method: bool = True,
) -> List[Path]:
    if not trial_dir.exists():
        return []

    all_trials = sorted(trial_dir.glob("*_trials.csv"))

    matched = []
    for p in all_trials:
        name_low = p.name.lower()

        if exclude_debug and "debug" in name_low:
            continue

        if study_name_contains is not None:
            if str(study_name_contains).lower() not in name_low:
                continue

        if _method_matches_trial_file(method, p):
            matched.append(p)

    if use_latest_study_per_method and len(matched) > 1:
        matched = sorted(matched, key=lambda x: x.stat().st_mtime, reverse=True)[:1]

    return matched


def _load_trials_for_method(
    method: str,
    trial_paths: List[Path],
    requested_fractions: List[float],
    requested_missingness_types: List[str],
) -> pd.DataFrame:
    rows = []

    for trial_path in trial_paths:
        try:
            trial_df = pd.read_csv(trial_path)
        except Exception as e:
            print(f"Could not read trial table for {method}: {trial_path} | {e}")
            continue

        if "state" in trial_df.columns:
            trial_df = trial_df[trial_df["state"].astype(str).str.upper() == "COMPLETE"].copy()

        score_cols = [
            c for c in trial_df.columns
            if "score_rows" in str(c).lower()
        ]

        if len(score_cols) == 0:
            print(f"No score_rows column found for {method}: {trial_path}")
            continue

        score_col = score_cols[0]
        study_name = trial_path.name.replace("_trials.csv", "")

        for _, trial_row in trial_df.iterrows():
            trial_number = trial_row["number"] if "number" in trial_row.index else np.nan
            trial_value = trial_row["value"] if "value" in trial_row.index else np.nan
            params = _extract_params_from_trial_row(trial_row)

            score_rows = _parse_score_rows(trial_row[score_col])

            for sr in score_rows:
                try:
                    frac = float(sr.get("fraction"))
                    miss = str(sr.get("missingness")).upper()
                    rmse = float(sr.get("validation_rmse"))
                except Exception:
                    continue

                keep_fraction = any(np.isclose(frac, float(f)) for f in requested_fractions)
                keep_missingness = miss in [m.upper() for m in requested_missingness_types]

                if not (keep_fraction and keep_missingness):
                    continue

                rows.append(
                    {
                        "method": method,
                        "source": "optuna_trial",
                        "study_name": study_name,
                        "trial_table": str(trial_path),
                        "trial_number": trial_number,
                        "trial_value_mean_validation_rmse": trial_value,
                        "fraction": frac,
                        "missingness": miss,
                        "validation_rmse": rmse,
                        "n_validation_cells": sr.get("n_validation_cells", np.nan),
                        "params": json.dumps(params, default=str),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Mean imputation support
# ---------------------------------------------------------------------
def _read_rmse_value_from_csv(path: Path) -> Optional[float]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    preferred_cols = [
        "rmse_vae",
        "rmse_mean",
        "rmse",
        "RMSE",
        "rmse_knn",
        "rmse_mf",
        "rmse_iterative",
        "rmse_softimpute",
    ]

    for col in preferred_cols:
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").dropna()
            if not val.empty:
                return float(val.iloc[0])

    try:
        return float(pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().iloc[0])
    except Exception:
        return None


def _load_mean_as_fixed_method(
    method_label: str,
    main_dir: Path,
    columns_num: str,
    requested_fractions: List[float],
    requested_missingness_types: List[str],
) -> pd.DataFrame:
    rows = []

    for missingness in requested_missingness_types:
        missingness_upper = str(missingness).upper()

        for fraction in requested_fractions:
            folder = main_dir / f"fraction_{fraction}_missingness_{missingness_upper}"

            candidate_paths = [
                folder / f"RMSE_Mean_col_{columns_num}.csv",
                folder / f"RMSE_mean_col_{columns_num}.csv",
                folder / f"RMSE_MEAN_col_{columns_num}.csv",
            ]

            candidate_paths.extend(sorted(folder.glob("RMSE*Mean*.csv")))
            candidate_paths.extend(sorted(folder.glob("RMSE*mean*.csv")))

            rmse_path = None
            rmse_value = None

            for p in candidate_paths:
                if not p.exists():
                    continue

                if "rmse" not in p.name.lower():
                    continue

                value = _read_rmse_value_from_csv(p)
                if value is not None and np.isfinite(value):
                    rmse_path = p
                    rmse_value = value
                    break

            if rmse_value is None:
                print(
                    f"WARNING: Could not find Mean RMSE for fraction={fraction}, "
                    f"missingness={missingness_upper} in {folder}"
                )
                continue

            rows.append(
                {
                    "method": method_label,
                    "source": "fixed_no_optimization",
                    "study_name": "Mean_no_optimization",
                    "trial_table": str(rmse_path),
                    "trial_number": 0,
                    "trial_value_mean_validation_rmse": rmse_value,
                    "fraction": float(fraction),
                    "missingness": missingness_upper,
                    "validation_rmse": float(rmse_value),
                    "n_validation_cells": np.nan,
                    "params": json.dumps(
                        {
                            "note": (
                                "Mean has no tunable hyperparameters. "
                                "Value read from final RMSE file."
                            )
                        }
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def _missingness_color_map(missingness_types: List[str]) -> Dict[str, str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    if not colors:
        colors = [f"C{i}" for i in range(10)]

    return {
        miss: colors[i % len(colors)]
        for i, miss in enumerate(missingness_types)
    }


def _sort_methods_for_fraction(
    plot_df: pd.DataFrame,
    sort_by: str,
) -> List[str]:
    if sort_by == "median":
        return (
            plot_df.groupby("method")["validation_rmse"]
            .median()
            .sort_values()
            .index
            .tolist()
        )

    if sort_by == "mean":
        return (
            plot_df.groupby("method")["validation_rmse"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )

    if sort_by == "min":
        return (
            plot_df.groupby("method")["validation_rmse"]
            .min()
            .sort_values()
            .index
            .tolist()
        )

    raise ValueError("sort_by must be one of: 'min', 'median', 'mean'.")


def _add_title_page_to_pdf(
    pdf: PdfPages,
    *,
    methods_to_compare,
    fractions,
    missingness_types,
    columns_num,
    study_name_contains,
    sort_by,
    zoom_y_range,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    lines = [
        "Bayesian Optimization RMSE Grouped Boxplot Report",
        "",
        f"columns_num: {columns_num}",
        f"methods: {methods_to_compare}",
        f"fractions: {fractions}",
        f"missingness_types: {missingness_types}",
        f"study_name_contains: {study_name_contains}",
        f"sort_by: {sort_by}",
        f"zoom_y_range: {zoom_y_range}",
        "",
        "Notes:",
        "Each page shows one missingness fraction.",
        "Within each method, boxes are shown side by side for each missingness mechanism.",
        "No values are removed from the full-scale panel.",
        "If zoom_y_range is provided, a second zoomed panel is added for readability.",
        "Mean imputation is shown as a fixed single-value method when included.",
    ]

    y = 0.95
    for i, line in enumerate(lines):
        if i == 0:
            ax.text(
                0.05,
                y,
                line,
                fontsize=18,
                fontweight="bold",
                va="top",
                ha="left",
                transform=ax.transAxes,
            )
            y -= 0.055
        else:
            ax.text(
                0.05,
                y,
                line,
                fontsize=10,
                va="top",
                ha="left",
                transform=ax.transAxes,
            )
            y -= 0.030

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_grouped_box_data(
    plot_df: pd.DataFrame,
    missingness_types: List[str],
    sort_by: str,
):
    plot_df = plot_df.copy()
    plot_df["validation_rmse"] = pd.to_numeric(plot_df["validation_rmse"], errors="coerce")
    plot_df = plot_df.dropna(subset=["validation_rmse"])

    missingness_types = [str(m).upper() for m in missingness_types]

    order = _sort_methods_for_fraction(plot_df, sort_by=sort_by)

    order = [
        method for method in order
        if plot_df.loc[plot_df["method"] == method, "validation_rmse"].notna().any()
    ]

    if len(order) == 0:
        raise ValueError("No methods with data to plot.")

    n_missingness = len(missingness_types)
    group_gap = 1.0
    group_width = n_missingness + group_gap

    positions = []
    box_data = []
    box_missingness = []
    box_method = []

    for method_index, method in enumerate(order):
        base = method_index * group_width + 1.0

        for miss_index, missingness in enumerate(missingness_types):
            vals = plot_df.loc[
                (plot_df["method"] == method)
                & (plot_df["missingness"].astype(str).str.upper() == missingness),
                "validation_rmse",
            ].to_numpy(dtype=float)

            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue

            pos = base + miss_index
            positions.append(pos)
            box_data.append(vals)
            box_missingness.append(missingness)
            box_method.append(method)

    if len(box_data) == 0:
        raise ValueError("No grouped boxes could be built.")

    tick_positions = [
        method_index * group_width + 1.0 + (n_missingness - 1) / 2.0
        for method_index in range(len(order))
    ]

    return {
        "order": order,
        "positions": positions,
        "box_data": box_data,
        "box_missingness": box_missingness,
        "box_method": box_method,
        "tick_positions": tick_positions,
        "group_width": group_width,
    }


def _draw_grouped_boxplot_on_axis(
    ax,
    *,
    grouped,
    missingness_types: List[str],
    show_points: bool = False,
    show_legend: bool = True,
    show_xlabels: bool = True,
    ylabel: str = "Validation RMSE",
    title: Optional[str] = None,
) -> None:
    positions = grouped["positions"]
    box_data = grouped["box_data"]
    box_missingness = grouped["box_missingness"]
    box_method = grouped["box_method"]
    order = grouped["order"]
    tick_positions = grouped["tick_positions"]

    missingness_types = [str(m).upper() for m in missingness_types]
    missingness_colors = _missingness_color_map(missingness_types)

    boxprops = dict(linewidth=1.25, color="black")
    medianprops = dict(linewidth=1.50, color="black")
    whiskerprops = dict(linewidth=1.10, color="black")
    capprops = dict(linewidth=1.10, color="black")

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.70,
        patch_artist=True,
        showmeans=False,
        showfliers=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=dict(
            marker="o",
            markersize=3,
            alpha=0.45,
            markeredgewidth=0.0,
        ),
    )

    for patch, missingness in zip(bp["boxes"], box_missingness):
        patch.set_facecolor(missingness_colors[missingness])
        patch.set_alpha(0.45)

    for flier, missingness in zip(bp["fliers"], box_missingness):
        flier.set_markerfacecolor(missingness_colors[missingness])
        flier.set_markeredgecolor(missingness_colors[missingness])
        flier.set_alpha(0.45)

    if show_points:
        rng = np.random.default_rng(42)

        for pos, vals, missingness in zip(positions, box_data, box_missingness):
            x = rng.normal(pos, 0.045, size=len(vals))

            ax.scatter(
                x,
                vals,
                alpha=0.30,
                s=10,
                color=missingness_colors[missingness],
                edgecolor="none",
                zorder=3,
            )

    # Mark single-value fixed methods such as Mean.
    for pos, vals, missingness, method in zip(positions, box_data, box_missingness, box_method):
        if len(vals) == 1:
            ax.scatter(
                [pos],
                [vals[0]],
                marker="D",
                s=36,
                color=missingness_colors[missingness],
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )

    if title is not None:
        ax.set_title(title)

    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.28)

    ax.set_xticks(tick_positions)

    if show_xlabels:
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_xlabel("Imputation method")
    else:
        ax.set_xticklabels([])

    if show_legend:
        legend_handles = []
        for missingness in missingness_types:
            handle = plt.Line2D(
                [0],
                [0],
                color=missingness_colors[missingness],
                lw=8,
                alpha=0.45,
                label=missingness,
            )
            legend_handles.append(handle)

        ax.legend(
            handles=legend_handles,
            title="Missingness",
            loc="best",
            frameon=True,
        )


def _plot_grouped_boxplot_for_fraction(
    plot_df: pd.DataFrame,
    fraction: float,
    missingness_types: List[str],
    out_dir: Path,
    sort_by: str = "min",
    show: bool = True,
    save_pdf: bool = True,
    y_tick_step: Optional[float] = None,
    zoom_y_range: Optional[Tuple[float, float]] = None,
    zoom_y_tick_step: Optional[float] = 0.01,
    show_points: bool = False,
    pdf_pages: Optional[PdfPages] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Create one plot for one missingness fraction.

    If zoom_y_range is provided, the figure has two panels:
        top = full y-axis with all values
        bottom = zoomed low-RMSE range
    """

    if plot_df.empty:
        raise ValueError(f"No data to plot for fraction={fraction}")

    grouped = _build_grouped_box_data(
        plot_df=plot_df,
        missingness_types=missingness_types,
        sort_by=sort_by,
    )

    n_methods = len(grouped["order"])
    n_missingness = len(missingness_types)
    fig_width = max(10, 1.05 * n_methods * max(n_missingness, 2))

    if zoom_y_range is None:
        fig, ax = plt.subplots(figsize=(fig_width, 5.8))

        _draw_grouped_boxplot_on_axis(
            ax,
            grouped=grouped,
            missingness_types=missingness_types,
            show_points=show_points,
            show_legend=True,
            show_xlabels=True,
            ylabel="Validation RMSE",
            title=f"Bayesian optimization validation RMSEs\nMissing fraction = {fraction:g}",
        )

        if y_tick_step is not None and y_tick_step > 0:
            ax.yaxis.set_major_locator(MultipleLocator(float(y_tick_step)))

    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(fig_width, 8.6),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 2.4]},
        )

        ax_full, ax_zoom = axes

        fig.suptitle(
            f"Bayesian optimization validation RMSEs, missing fraction = {fraction:g}\n"
            f"Top: full scale, Bottom: zoomed low-RMSE region",
            y=0.98,
            fontsize=13,
        )

        _draw_grouped_boxplot_on_axis(
            ax_full,
            grouped=grouped,
            missingness_types=missingness_types,
            show_points=False,
            show_legend=True,
            show_xlabels=False,
            ylabel="Full scale",
            title=None,
        )

        _draw_grouped_boxplot_on_axis(
            ax_zoom,
            grouped=grouped,
            missingness_types=missingness_types,
            show_points=show_points,
            show_legend=False,
            show_xlabels=True,
            ylabel="Zoomed RMSE",
            title=None,
        )

        ax_zoom.set_ylim(float(zoom_y_range[0]), float(zoom_y_range[1]))

        if y_tick_step is not None and y_tick_step > 0:
            ax_full.yaxis.set_major_locator(MultipleLocator(float(y_tick_step)))

        if zoom_y_tick_step is not None and zoom_y_tick_step > 0:
            ax_zoom.yaxis.set_major_locator(MultipleLocator(float(zoom_y_tick_step)))

        # Add a subtle note to clarify that values are not removed.
        ax_full.text(
            0.01,
            0.98,
            "All trial values shown",
            transform=ax_full.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )

    plt.tight_layout()

    frac_token = _safe_fraction_token(fraction)
    out_png = out_dir / f"bayesopt_grouped_boxplot_fraction_{frac_token}.png"
    out_pdf = out_dir / f"bayesopt_grouped_boxplot_fraction_{frac_token}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    if save_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    else:
        out_pdf = None

    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return out_png, out_pdf


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------
def compare_imputation_methods(
    methods_to_compare,
    fractions,
    missingness_types,
    columns_num,
    study_name_contains: Optional[str] = None,
    exclude_debug: bool = True,
    use_latest_study_per_method: bool = True,
    sort_by: str = "min",
    show: bool = True,
    save_pdf: bool = True,
    y_tick_step: Optional[float] = None,
    zoom_y_range: Optional[Tuple[float, float]] = None,
    zoom_y_tick_step: Optional[float] = 0.01,
    show_points: bool = False,
    include_fixed_mean: bool = True,
    save_combined_pdf: bool = True,
    combined_pdf_name: str = "bayesopt_grouped_boxplot_report.pdf",
    include_title_page: bool = True,
):
    """
    Makes grouped Bayesian optimization RMSE boxplots.

    One plot is generated per missing fraction. Within each plot, each method has
    side-by-side boxes for each requested missingness type.

    Example layout for missingness_types=["MAR", "MNAR"]:
        KNN-MAR | KNN-MNAR | AE-MAR | AE-MNAR | ...

    No values are removed. If zoom_y_range is provided, each figure contains a
    full-scale panel and a zoomed panel.

    Parameters
    ----------
    methods_to_compare : list[str]
        Example:
            ["KNN", "softImpute", "VAE", "DAE", "AE", "mf", "GAIN", "iterative", "Mean"]

    fractions : list[float]
        One plot is made per fraction.

    missingness_types : list[str]
        Missingness mechanisms shown side by side within each method.

    columns_num : str
        Usually "all".

    study_name_contains : str or None
        Optional filter. For example, use "balanced" to only use balanced Optuna studies.

    exclude_debug : bool
        If True, trial tables with "DEBUG" in the filename are ignored.

    use_latest_study_per_method : bool
        If True and multiple trial tables exist for one method, use only the most recently modified one.

    sort_by : {"min", "median", "mean"}
        How to sort methods on the x-axis, using all requested missingness types for that fraction.

    zoom_y_range : tuple(float, float) or None
        If provided, add a zoomed lower panel with this y-range.
        Example: (0.24, 0.36)

    show_points : bool
        If True, show small colored jittered trial dots for all values.

    include_fixed_mean : bool
        If True, Mean is read from final RMSE files because it has no Optuna trials.

    save_combined_pdf : bool
        If True, save all fraction plots into one multi-page PDF.

    Returns
    -------
    all_trial_values : pd.DataFrame
        Long-format dataframe containing every validation RMSE used for plotting.

    output_paths : dict
        Mapping from fraction to saved figure paths.
        Also contains output_paths["combined_pdf"] when save_combined_pdf=True.
    """

    fractions = [float(f) for f in fractions]
    missingness_types = [str(m).upper() for m in missingness_types]

    main_dir = Path(experiments_dir(columns_num))
    trial_dir = main_dir / "Optimization" / "trial_tables"
    out_dir = main_dir / "Optimization" / "boxplots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_method_dfs = []

    print("\nReading Optuna trial tables from:")
    print(trial_dir)

    for method in methods_to_compare:
        canonical = _canonical_method_name(method)

        if canonical == "MEAN" and include_fixed_mean:
            print("\nMethod: Mean")
            print("  Mean has no Bayesian optimization trials. It will be shown as a fixed single-value method.")

            mean_df = _load_mean_as_fixed_method(
                method_label=method,
                main_dir=main_dir,
                columns_num=columns_num,
                requested_fractions=fractions,
                requested_missingness_types=missingness_types,
            )

            if mean_df.empty:
                print("WARNING: No Mean RMSE values found.")
            else:
                all_method_dfs.append(mean_df)

            continue

        trial_paths = _find_trial_tables_for_method(
            method=method,
            trial_dir=trial_dir,
            exclude_debug=exclude_debug,
            study_name_contains=study_name_contains,
            use_latest_study_per_method=use_latest_study_per_method,
        )

        if len(trial_paths) == 0:
            print(
                f"WARNING: No Optuna trial table found for method='{method}'. "
                f"If this method has no hyperparameters, it will not appear in the boxplot."
            )
            continue

        print(f"\nMethod: {method}")
        for p in trial_paths:
            print(f"  Using trial table: {p}")

        method_df = _load_trials_for_method(
            method=method,
            trial_paths=trial_paths,
            requested_fractions=fractions,
            requested_missingness_types=missingness_types,
        )

        if method_df.empty:
            print(f"WARNING: Trial table found for method='{method}', but no matching fraction/missingness rows.")
            continue

        all_method_dfs.append(method_df)

    if len(all_method_dfs) == 0:
        raise ValueError("No Bayesian optimization trial data found for the requested methods/settings.")

    all_trial_values = pd.concat(all_method_dfs, ignore_index=True)

    values_csv = out_dir / "bayesopt_grouped_boxplot_values.csv"
    all_trial_values.to_csv(values_csv, index=False)
    print(f"\nSaved raw grouped boxplot values to: {values_csv}")

    output_paths = {}

    combined_pdf_path = out_dir / combined_pdf_name if save_combined_pdf else None

    if save_combined_pdf:
        pdf_context = PdfPages(combined_pdf_path)
    else:
        pdf_context = None

    try:
        if pdf_context is not None and include_title_page:
            _add_title_page_to_pdf(
                pdf_context,
                methods_to_compare=methods_to_compare,
                fractions=fractions,
                missingness_types=missingness_types,
                columns_num=columns_num,
                study_name_contains=study_name_contains,
                sort_by=sort_by,
                zoom_y_range=zoom_y_range,
            )

        for fraction in fractions:
            plot_df = all_trial_values[
                np.isclose(all_trial_values["fraction"].astype(float), float(fraction))
            ].copy()

            if plot_df.empty:
                print(f"Skipping plot: no data for fraction={fraction}")
                continue

            png_path, pdf_path = _plot_grouped_boxplot_for_fraction(
                plot_df=plot_df,
                fraction=fraction,
                missingness_types=missingness_types,
                out_dir=out_dir,
                sort_by=sort_by,
                show=show,
                save_pdf=save_pdf,
                y_tick_step=y_tick_step,
                zoom_y_range=zoom_y_range,
                zoom_y_tick_step=zoom_y_tick_step,
                show_points=show_points,
                pdf_pages=pdf_context,
            )

            output_paths[fraction] = {
                "png": str(png_path),
                "pdf": str(pdf_path) if pdf_path is not None else None,
            }

            print(f"Saved grouped plot for fraction={fraction}:")
            print(f"  {png_path}")
            if pdf_path is not None:
                print(f"  {pdf_path}")

    finally:
        if pdf_context is not None:
            pdf_context.close()
            output_paths["combined_pdf"] = str(combined_pdf_path)
            print(f"\nSaved combined grouped PDF report to: {combined_pdf_path}")

    return all_trial_values, output_paths