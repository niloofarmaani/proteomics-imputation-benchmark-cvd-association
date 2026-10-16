# Evaluation/iterative_eval.py

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config_paths


# =============================================================================
# Parsing and naming
# =============================================================================
def _ordered_missingness(values: Sequence[str]) -> List[str]:
    preferred = ["MCAR", "MAR", "MNAR"]
    normalized = [str(value).upper() for value in values]
    present = [value for value in preferred if value in set(normalized)]
    extras = sorted(value for value in set(normalized) if value not in present)
    return present + extras


def _float_token(value: float) -> str:
    token = f"{float(value):.12g}"
    return token.replace("-", "m").replace(".", "p")


def _format_float(value: float) -> str:
    value = float(value)
    return f"{value:.2e}" if abs(value) < 0.001 or abs(value) >= 100 else f"{value:.4g}"


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))

    token = str(value).strip().lower()

    if token in {"1", "true", "t", "yes", "y"}:
        return True

    if token in {"0", "false", "f", "no", "n"}:
        return False

    raise ValueError(f"Could not parse {name} as boolean. Value: {value!r}")


def _parse_optional_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    token = str(value).strip().lower()

    if token in {"none", "all", "null"}:
        return None

    parsed = int(float(token))

    if parsed < 1:
        raise ValueError(f"{name} must be at least 1 or None. Got: {parsed}")

    return parsed


def _nearest_token(value: Optional[int]) -> str:
    return "all" if value is None else str(int(value))


def _bool_token(value: bool) -> str:
    return "1" if bool(value) else "0"


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    required = {
        "max_iter",
        "tol",
        "n_nearest_features",
        "imputation_order",
        "initial_strategy",
        "estimator_name",
        "sample_posterior",
        "skip_complete",
    }
    missing = required - set(params)

    if missing:
        raise KeyError(
            f"IterativeImputer parameter dictionary is missing keys {sorted(missing)}. "
            f"Parameters: {params}"
        )

    normalized = {
        "max_iter": int(params["max_iter"]),
        "tol": float(params["tol"]),
        "n_nearest_features": _parse_optional_int(
            params["n_nearest_features"],
            "n_nearest_features",
        ),
        "imputation_order": str(params["imputation_order"]).lower(),
        "initial_strategy": str(params["initial_strategy"]).lower(),
        "estimator_name": str(params["estimator_name"]),
        "sample_posterior": _parse_bool(
            params["sample_posterior"],
            "sample_posterior",
        ),
        "skip_complete": _parse_bool(
            params["skip_complete"],
            "skip_complete",
        ),
    }

    if normalized["max_iter"] < 1:
        raise ValueError("max_iter must be at least 1.")

    if not np.isfinite(normalized["tol"]) or normalized["tol"] <= 0:
        raise ValueError("tol must be finite and positive.")

    if normalized["estimator_name"] != "BayesianRidge":
        raise ValueError(
            f"Unsupported estimator_name={normalized['estimator_name']!r}."
        )

    return normalized


def _param_id_from_params(params: Dict[str, Any]) -> str:
    params = _normalize_params(params)
    estimator_token = {"BayesianRidge": "br"}[params["estimator_name"]]

    return (
        f"mi{params['max_iter']}"
        f"_tol{_float_token(params['tol'])}"
        f"_nnf{_nearest_token(params['n_nearest_features'])}"
        f"_ord{params['imputation_order']}"
        f"_init{params['initial_strategy']}"
        f"_est{estimator_token}"
        f"_post{_bool_token(params['sample_posterior'])}"
        f"_skip{_bool_token(params['skip_complete'])}"
    )


def _pretty_config_label(params: Dict[str, Any]) -> str:
    params = _normalize_params(params)
    return (
        f"max_iter={params['max_iter']}, "
        f"tol={_format_float(params['tol'])}, "
        f"nnf={_nearest_token(params['n_nearest_features'])}, "
        f"order={params['imputation_order']}, "
        f"init={params['initial_strategy']}, "
        f"est={params['estimator_name']}, "
        f"posterior={int(params['sample_posterior'])}, "
        f"skip={int(params['skip_complete'])}"
    )


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist:\n{path}")

    with path.open("r") as handle:
        return json.load(handle)


def _parse_json_or_literal(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}

    if isinstance(raw, dict):
        return raw

    try:
        if pd.isna(raw):
            return {}
    except Exception:
        pass

    token = str(raw).strip()

    if not token or token.lower() in {"none", "nan", "null"}:
        return {}

    try:
        parsed = json.loads(token)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(token)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    raise ValueError(f"Could not parse parameter dictionary from: {raw!r}")


# =============================================================================
# Load data
# =============================================================================
def _load_best_params(
    base_dir: Path,
    study_name: str,
) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    best_path = (
        base_dir
        / "Optimization"
        / "best_params"
        / f"{study_name}_best_params.json"
    )
    payload = _read_json(best_path)

    if "params" not in payload:
        raise KeyError(f"Best-parameter JSON has no 'params' key:\n{best_path}")

    return _normalize_params(payload["params"]), best_path, payload


def _extract_params_from_row(row: pd.Series) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if "method_params_json" in row.index:
        params.update(_parse_json_or_literal(row["method_params_json"]))

    keys = [
        "max_iter",
        "tol",
        "n_nearest_features",
        "imputation_order",
        "initial_strategy",
        "estimator_name",
        "sample_posterior",
        "skip_complete",
    ]

    for key in keys:
        for column in (f"method_params_{key}", f"params_{key}", key):
            if column not in row.index:
                continue

            value = row[column]

            try:
                is_missing = bool(pd.isna(value))
            except Exception:
                is_missing = False

            if not is_missing:
                params[key] = value
                break

    return _normalize_params(params)


def _load_condition_scores(
    *,
    base_dir: Path,
    study_name: str,
    require_condition_scores: bool,
) -> pd.DataFrame:
    condition_path = (
        base_dir
        / "Optimization"
        / "trial_tables"
        / f"{study_name}_condition_scores.csv"
    )

    if not condition_path.exists():
        raise FileNotFoundError(
            f"Per-condition Optuna score table does not exist:\n{condition_path}\n"
            "Rerun IterativeImputer optimization with the strict seeded optimizer."
        )

    frame = pd.read_csv(condition_path)

    if frame.empty and require_condition_scores:
        raise RuntimeError(f"Condition-score table is empty:\n{condition_path}")

    required_columns = {
        "trial_number",
        "state",
        "fraction",
        "missingness",
        "validation_rmse",
    }
    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise KeyError(
            f"Condition-score table is missing columns {sorted(missing_columns)}.\n"
            f"File: {condition_path}\nColumns: {list(frame.columns)}"
        )

    frame = frame[
        frame["state"].astype(str).str.upper() == "COMPLETE"
    ].copy()

    if frame.empty and require_condition_scores:
        raise RuntimeError(
            f"No COMPLETE rows were found in the condition-score table:\n"
            f"{condition_path}"
        )

    parsed = []

    for _, row in frame.iterrows():
        params = _extract_params_from_row(row)
        parsed.append(
            (
                params,
                _param_id_from_params(params),
                _pretty_config_label(params),
            )
        )

    frame["params_dict"] = [item[0] for item in parsed]
    frame["param_id"] = [item[1] for item in parsed]
    frame["config_label"] = [item[2] for item in parsed]

    for key in [
        "max_iter",
        "tol",
        "n_nearest_features",
        "imputation_order",
        "initial_strategy",
        "estimator_name",
        "sample_posterior",
        "skip_complete",
    ]:
        frame[key] = frame["params_dict"].map(lambda params: params[key])

    frame["n_nearest_features_label"] = frame["n_nearest_features"].map(
        _nearest_token
    )
    frame["trial_number"] = pd.to_numeric(
        frame["trial_number"], errors="raise"
    ).astype(int)
    frame["fraction"] = pd.to_numeric(frame["fraction"], errors="raise")
    frame["validation_rmse"] = pd.to_numeric(
        frame["validation_rmse"], errors="raise"
    )
    frame["missingness"] = frame["missingness"].astype(str).str.upper()

    if not np.isfinite(frame["validation_rmse"]).all():
        raise ValueError(
            "Condition-score table contains non-finite validation RMSE values."
        )

    return frame.sort_values(
        ["missingness", "fraction", "trial_number"]
    ).reset_index(drop=True)


def _read_rmse_value(path: Path) -> float:
    frame = pd.read_csv(path)

    for column in ("rmse_iterative", "rmse", "RMSE"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                value = float(values.iloc[0])
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite RMSE in {path}: {value}")
                return value

    values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()

    if values.empty:
        raise ValueError(f"No numeric RMSE value was found in:\n{path}")

    value = float(values.iloc[0])

    if not np.isfinite(value):
        raise ValueError(f"Non-finite RMSE in {path}: {value}")

    return value


def _load_seed_rmse_values(
    *,
    base_dir: Path,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    best_params: Dict[str, Any],
    strict: bool,
) -> pd.DataFrame:
    rows = []
    missing_files = []
    param_id = _param_id_from_params(best_params)

    for missingness in missingness_types:
        missingness = str(missingness).upper()

        for fraction in fractions:
            fraction = float(fraction)
            folder = base_dir / f"fraction_{fraction}_missingness_{missingness}"

            for seed in seeds:
                seed = int(seed)
                rmse_path = folder / (
                    f"RMSE_iterative_col_{columns_num}_params_{param_id}"
                    f"_seed_{seed}.csv"
                )
                imputed_path = folder / (
                    f"imputed_iterative_col_{columns_num}_params_{param_id}"
                    f"_seed_{seed}.csv"
                )

                if not rmse_path.exists():
                    missing_files.append(str(rmse_path))
                    continue

                if strict and not imputed_path.exists():
                    missing_files.append(str(imputed_path))
                    continue

                rows.append(
                    {
                        "method": "Iterative",
                        "param_id": param_id,
                        **best_params,
                        "n_nearest_features_label": _nearest_token(
                            best_params["n_nearest_features"]
                        ),
                        "missingness": missingness,
                        "fraction": fraction,
                        "seed": seed,
                        "rmse": _read_rmse_value(rmse_path),
                        "rmse_path": str(rmse_path),
                        "imputed_path": str(imputed_path),
                    }
                )

    if missing_files and strict:
        preview = "\n".join(missing_files[:50])
        raise FileNotFoundError(
            "Some required repeated-seed IterativeImputer outputs are missing.\n\n"
            f"First missing files:\n{preview}\n\n"
            "Run the frozen-best final benchmark for every requested seed, "
            "fraction, and missingness mechanism."
        )

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise RuntimeError(
            "No repeated-seed IterativeImputer RMSE files were found."
        )

    return frame.sort_values(
        ["missingness", "fraction", "seed"]
    ).reset_index(drop=True)


# =============================================================================
# Plot helpers
# =============================================================================
def _config_color_map(config_ids: Sequence[str]) -> Dict[str, Any]:
    color_map = plt.get_cmap("tab20")
    return {
        config_id: color_map(index % color_map.N)
        for index, config_id in enumerate(list(config_ids))
    }


def _make_config_legend_table(
    tuning_df: pd.DataFrame,
    best_param_id: str,
) -> pd.DataFrame:
    rows = []

    for param_id, group in tuning_df.groupby("param_id"):
        rows.append(
            {
                "param_id": param_id,
                "config_label": str(group["config_label"].iloc[0]),
                "trial_numbers": sorted(
                    set(int(value) for value in group["trial_number"])
                ),
                "mean_validation_rmse": float(
                    group["validation_rmse"].mean()
                ),
                "is_best": param_id == best_param_id,
            }
        )

    table = pd.DataFrame(rows)

    if table.empty:
        raise ValueError(
            "No IterativeImputer hyperparameter configurations were available."
        )

    return table.sort_values(
        ["is_best", "mean_validation_rmse"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _legend_label(row: pd.Series) -> str:
    trials = row["trial_numbers"]
    trial_text = (
        f"T{trials[0]}"
        if len(trials) == 1
        else "T" + ",".join(str(value) for value in trials)
    )
    prefix = "BEST | " if bool(row["is_best"]) else ""
    return (
        f"{prefix}{trial_text} | {row['config_label']} | "
        f"mean={float(row['mean_validation_rmse']):.4f}"
    )


def _save_figure(
    fig,
    *,
    out_dir: Path,
    stem: str,
    show: bool,
    pdf_pages: Optional[PdfPages] = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


# =============================================================================
# Plots
# =============================================================================
def plot_iterative_hyperparameter_effect(
    tuning_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    study_name: str,
    show: bool,
    show_full_hyperparameter_legend: bool,
    pdf_pages: Optional[PdfPages] = None,
) -> Dict[str, str]:
    missingness_types = _ordered_missingness(
        tuning_df["missingness"].unique()
    )
    best_param_id = _param_id_from_params(best_params)
    config_table = _make_config_legend_table(
        tuning_df,
        best_param_id=best_param_id,
    )
    config_order = config_table["param_id"].tolist()
    colors = _config_color_map(config_order)

    n_configs = len(config_order)
    legend_columns = 1 if n_configs <= 12 else 2
    right_edge = 0.58 if n_configs > 12 else 0.67

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.2 * len(missingness_types) + 6.0, 5.4),
        sharey=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for axis, missingness in zip(axes, missingness_types):
        subset = tuning_df[
            tuning_df["missingness"] == missingness
        ].copy()
        by_config = (
            subset.groupby(["param_id", "fraction"], as_index=False)[
                "validation_rmse"
            ]
            .mean()
            .sort_values(["param_id", "fraction"])
        )
        band = (
            by_config.groupby("fraction")["validation_rmse"]
            .agg(["min", "max", "mean"])
            .reset_index()
            .sort_values("fraction")
        )

        x_values = band["fraction"].to_numpy(dtype=float)
        axis.fill_between(
            x_values,
            band["min"].to_numpy(dtype=float),
            band["max"].to_numpy(dtype=float),
            color="gray",
            alpha=0.12,
        )
        axis.plot(
            x_values,
            band["mean"].to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=1.8,
        )

        for param_id in config_order:
            group = by_config[
                by_config["param_id"] == param_id
            ].sort_values("fraction")

            if group.empty:
                continue

            is_best = param_id == best_param_id
            axis.plot(
                group["fraction"],
                group["validation_rmse"],
                color="tab:red" if is_best else colors[param_id],
                alpha=1.0 if is_best else 0.70,
                linewidth=3.2 if is_best else 1.5,
                marker="o" if is_best else ".",
                markersize=6 if is_best else 4,
            )

        axis.set_title(missingness)
        axis.set_xlabel("Missing fraction")
        axis.grid(True, alpha=0.30)

    axes[0].set_ylabel("Validation RMSE")

    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="mean across configurations",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=8,
            alpha=0.20,
            label="range across configurations",
        ),
    ]

    if show_full_hyperparameter_legend:
        for _, row in config_table.iterrows():
            param_id = row["param_id"]
            is_best = bool(row["is_best"])
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="tab:red" if is_best else colors[param_id],
                    linewidth=3.2 if is_best else 1.8,
                    marker="o" if is_best else ".",
                    markersize=6 if is_best else 4,
                    label=_legend_label(row),
                )
            )

    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(right_edge + 0.02, 0.5),
        fontsize=7,
        ncol=legend_columns,
        title="IterativeImputer configurations",
        title_fontsize=9,
        frameon=True,
    )
    fig.suptitle(
        "IterativeImputer hyperparameter effect under fixed optimization seed\n"
        f"Study: {study_name}",
        y=1.04,
    )
    fig.tight_layout(rect=[0, 0, right_edge, 1])

    return _save_figure(
        fig,
        out_dir=out_dir,
        stem="Iterative_hyperparameter_effect_fixed_seed",
        show=show,
        pdf_pages=pdf_pages,
    )


def plot_iterative_hyperparameter_search_distribution(
    tuning_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    study_name: str,
    show: bool,
    pdf_pages: Optional[PdfPages] = None,
) -> Dict[str, str]:
    """Show the tuned max_iter distribution and all fixed method settings."""
    max_iter_values = sorted(
        pd.to_numeric(tuning_df["max_iter"], errors="raise")
        .astype(int)
        .unique()
    )

    if not max_iter_values:
        raise ValueError("No max_iter values were found in the tuning table.")

    data = [
        tuning_df.loc[
            pd.to_numeric(tuning_df["max_iter"], errors="raise").astype(int)
            == value,
            "validation_rmse",
        ].to_numpy(dtype=float)
        for value in max_iter_values
    ]

    if any(values.size == 0 for values in data):
        raise ValueError("At least one max_iter category has no validation RMSE values.")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    boxplot = axes[0].boxplot(
        data,
        labels=[str(value) for value in max_iter_values],
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    )

    best_max_iter = int(best_params["max_iter"])
    for patch, value in zip(boxplot["boxes"], max_iter_values):
        is_best = int(value) == best_max_iter
        patch.set_facecolor("tab:red" if is_best else "tab:blue")
        patch.set_alpha(0.45 if is_best else 0.25)

    axes[0].set_title("Tuned hyperparameter")
    axes[0].set_xlabel("Maximum imputation rounds, max_iter")
    axes[0].set_ylabel("Validation RMSE")
    axes[0].grid(True, axis="y", alpha=0.30)

    axes[1].axis("off")
    axes[1].text(
        0.04,
        0.94,
        "Frozen method settings",
        fontsize=13,
        fontweight="bold",
        va="top",
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.04,
        0.82,
        "\n".join(
            [
                f"selected max_iter = {best_params['max_iter']}",
                f"tolerance = {_format_float(best_params['tol'])}",
                "nearest features = "
                f"{_nearest_token(best_params['n_nearest_features'])}",
                f"imputation order = {best_params['imputation_order']}",
                f"initial strategy = {best_params['initial_strategy']}",
                f"estimator = {best_params['estimator_name']}",
                f"sample posterior = {best_params['sample_posterior']}",
                f"skip complete = {best_params['skip_complete']}",
            ]
        ),
        fontsize=10.5,
        va="top",
        linespacing=1.45,
        transform=axes[1].transAxes,
    )

    fig.suptitle(
        "IterativeImputer hyperparameter search distribution\n"
        f"Study: {study_name}",
        y=1.03,
    )
    fig.tight_layout()

    return _save_figure(
        fig,
        out_dir=out_dir,
        stem="Iterative_hyperparameter_search_distribution",
        show=show,
        pdf_pages=pdf_pages,
    )

def plot_iterative_seed_ribbon(
    seed_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    show: bool,
    pdf_pages: Optional[PdfPages] = None,
) -> Dict[str, str]:
    missingness_types = _ordered_missingness(seed_df["missingness"].unique())
    best_label = _pretty_config_label(best_params)

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.2 * len(missingness_types), 4.8),
        sharey=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for axis, missingness in zip(axes, missingness_types):
        subset = seed_df[seed_df["missingness"] == missingness].copy()

        for _, group in subset.groupby("seed"):
            group = group.sort_values("fraction")
            axis.plot(
                group["fraction"],
                group["rmse"],
                color="gray",
                alpha=0.35,
                linewidth=1.0,
                marker=".",
            )

        summary = (
            subset.groupby("fraction")["rmse"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("fraction")
        )
        x_values = summary["fraction"].to_numpy(dtype=float)
        means = summary["mean"].to_numpy(dtype=float)
        standard_deviations = summary["std"].fillna(0.0).to_numpy(dtype=float)

        axis.fill_between(
            x_values,
            means - standard_deviations,
            means + standard_deviations,
            color="black",
            alpha=0.15,
            label="mean ± 1 SD",
        )
        axis.plot(
            x_values,
            means,
            color="black",
            linewidth=2.5,
            marker="o",
            label="mean",
        )
        axis.set_title(missingness)
        axis.set_xlabel("Missing fraction")
        axis.grid(True, alpha=0.30)

    axes[0].set_ylabel("Final RMSE")
    handles, labels = axes[-1].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

    fig.suptitle(
        "IterativeImputer seed robustness with frozen best hyperparameters\n"
        f"{best_label}",
        y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    return _save_figure(
        fig,
        out_dir=out_dir,
        stem="Iterative_seed_effect_fixed_best_params_ribbon",
        show=show,
        pdf_pages=pdf_pages,
    )


def plot_iterative_seed_boxplots(
    seed_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    show: bool,
    pdf_pages: Optional[PdfPages] = None,
) -> Dict[str, str]:
    missingness_types = _ordered_missingness(seed_df["missingness"].unique())
    best_label = _pretty_config_label(best_params)

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.2 * len(missingness_types), 4.8),
        sharey=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for axis, missingness in zip(axes, missingness_types):
        subset = seed_df[seed_df["missingness"] == missingness].copy()
        fractions = sorted(subset["fraction"].unique())
        data = [
            subset.loc[
                np.isclose(subset["fraction"], fraction),
                "rmse",
            ].to_numpy(dtype=float)
            for fraction in fractions
        ]

        boxplot = axis.boxplot(
            data,
            labels=[f"{fraction:g}" for fraction in fractions],
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
        )

        for patch in boxplot["boxes"]:
            patch.set_facecolor("tab:blue")
            patch.set_alpha(0.35)

        axis.set_title(missingness)
        axis.set_xlabel("Missing fraction")
        axis.grid(True, axis="y", alpha=0.30)

    axes[0].set_ylabel("Final RMSE")
    fig.suptitle(
        "IterativeImputer seed robustness boxplots with frozen best hyperparameters\n"
        f"{best_label}",
        y=1.04,
    )
    fig.tight_layout()

    return _save_figure(
        fig,
        out_dir=out_dir,
        stem="Iterative_seed_effect_fixed_best_params_boxplots",
        show=show,
        pdf_pages=pdf_pages,
    )


# =============================================================================
# Public evaluator
# =============================================================================
def iterative_eval(
    fractions,
    missingness_types,
    columns_num: str = "all",
    study_name: str = "Iterative_balanced_MAR_MNAR_10_30_seeded_v1",
    seeds: Sequence[int] = (42, 100, 2024, 777, 999),
    output_dir: Optional[str] = None,
    show: bool = True,
    strict: bool = True,
    require_condition_scores: bool = True,
    show_full_hyperparameter_legend: bool = True,
) -> Dict[str, Any]:
    fractions = [float(value) for value in fractions]
    missingness_types = [str(value).upper() for value in missingness_types]
    seeds = [int(value) for value in seeds]

    if not fractions:
        raise ValueError("fractions cannot be empty.")

    if not missingness_types:
        raise ValueError("missingness_types cannot be empty.")

    if not seeds:
        raise ValueError("seeds cannot be empty.")

    base_dir = Path(config_paths.experiments_dir(columns_num))

    if not base_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist:\n{base_dir}")

    out_dir = (
        base_dir / "Evaluation" / "Iterative_seeded_pipeline"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    best_params, best_path, _ = _load_best_params(base_dir, study_name)

    print("\n" + "=" * 80)
    print("IterativeImputer evaluation suite")
    print("=" * 80)
    print(f"Study name: {study_name}")
    print(f"Best-parameter JSON: {best_path}")
    print(f"Best parameters: {best_params}")
    print(f"Fractions: {fractions}")
    print(f"Missingness types: {missingness_types}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    tuning_df = _load_condition_scores(
        base_dir=base_dir,
        study_name=study_name,
        require_condition_scores=require_condition_scores,
    )
    tuning_csv = out_dir / "Iterative_optuna_per_condition_validation_rmse.csv"
    tuning_df.to_csv(tuning_csv, index=False)

    config_table = _make_config_legend_table(
        tuning_df,
        best_param_id=_param_id_from_params(best_params),
    )
    config_csv = out_dir / "Iterative_hyperparameter_config_legend.csv"
    config_table.to_csv(config_csv, index=False)

    seed_df = _load_seed_rmse_values(
        base_dir=base_dir,
        columns_num=columns_num,
        fractions=fractions,
        missingness_types=missingness_types,
        seeds=seeds,
        best_params=best_params,
        strict=strict,
    )
    seed_csv = out_dir / "Iterative_final_rmse_by_seed.csv"
    seed_df.to_csv(seed_csv, index=False)

    seed_summary = (
        seed_df.groupby(["missingness", "fraction"])["rmse"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )
    seed_summary_csv = out_dir / "Iterative_final_rmse_by_seed_summary.csv"
    seed_summary.to_csv(seed_summary_csv, index=False)

    report_path = out_dir / "Iterative_evaluation_suite_report.pdf"
    outputs: Dict[str, Any] = {
        "best_params": best_params,
        "best_params_path": str(best_path),
        "tuning_csv": str(tuning_csv),
        "config_legend_csv": str(config_csv),
        "seed_csv": str(seed_csv),
        "seed_summary_csv": str(seed_summary_csv),
        "combined_report_pdf": str(report_path),
        "plots": {},
    }

    with PdfPages(report_path) as report:
        cover, cover_axis = plt.subplots(figsize=(8.5, 11))
        cover_axis.axis("off")
        cover_axis.text(
            0.05,
            0.95,
            "IterativeImputer evaluation suite",
            fontsize=18,
            fontweight="bold",
            va="top",
            transform=cover_axis.transAxes,
        )
        cover_axis.text(
            0.05,
            0.88,
            "\n".join(
                [
                    f"Study: {study_name}",
                    f"Best parameters: {_pretty_config_label(best_params)}",
                    f"Fractions: {fractions}",
                    f"Missingness mechanisms: {missingness_types}",
                    f"Seeds: {seeds}",
                ]
            ),
            fontsize=10,
            va="top",
            transform=cover_axis.transAxes,
        )
        report.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        outputs["plots"]["hyperparameter_effect"] = (
            plot_iterative_hyperparameter_effect(
                tuning_df,
                best_params=best_params,
                out_dir=out_dir,
                study_name=study_name,
                show=show,
                show_full_hyperparameter_legend=(
                    show_full_hyperparameter_legend
                ),
                pdf_pages=report,
            )
        )
        outputs["plots"]["hyperparameter_search_distribution"] = (
            plot_iterative_hyperparameter_search_distribution(
                tuning_df,
                best_params=best_params,
                out_dir=out_dir,
                study_name=study_name,
                show=show,
                pdf_pages=report,
            )
        )
        outputs["plots"]["seed_ribbon"] = plot_iterative_seed_ribbon(
            seed_df,
            best_params=best_params,
            out_dir=out_dir,
            show=show,
            pdf_pages=report,
        )
        outputs["plots"]["seed_boxplots"] = plot_iterative_seed_boxplots(
            seed_df,
            best_params=best_params,
            out_dir=out_dir,
            show=show,
            pdf_pages=report,
        )

    print("\nSaved IterativeImputer evaluation outputs:")
    print(f"  Per-condition tuning values: {tuning_csv}")
    print(f"  Hyperparameter legend table: {config_csv}")
    print(f"  Repeated-seed values: {seed_csv}")
    print(f"  Repeated-seed summary: {seed_summary_csv}")
    print(f"  Combined PDF report: {report_path}")

    return outputs


ITERATIVE_eval = iterative_eval
