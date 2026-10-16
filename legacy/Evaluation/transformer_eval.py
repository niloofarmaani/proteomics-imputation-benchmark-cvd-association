# Evaluation/transformer_eval.py

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config_paths


# =============================================================================
# General helpers
# =============================================================================
def _ordered_missingness(values: Sequence[str]) -> List[str]:
    preferred = ["MCAR", "MAR", "MNAR"]
    normalized = [str(value).upper() for value in values]
    ordered = [item for item in preferred if item in set(normalized)]
    ordered.extend(sorted(item for item in set(normalized) if item not in ordered))
    return ordered


def _float_token(value: float) -> str:
    token = f"{float(value):.12g}"
    token = token.replace("-", "m")
    token = token.replace(".", "p")
    return token


def _format_float(value: float) -> str:
    value = float(value)
    if abs(value) < 0.001 or abs(value) >= 100:
        return f"{value:.2e}"
    return f"{value:.4g}"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist:\n{path}")

    with path.open("r") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be a dictionary: {path}")

    return payload


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

    text = str(raw).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    raise ValueError(f"Could not parse parameter dictionary: {raw}")


TRANSFORMER_PARAM_KEYS = [
    "batch_size",
    "learning_rate",
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "dropout",
    "pseudo_mask_rate",
    "weight_decay",
    "grad_clip",
    "internal_val_rate",
]


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    missing = set(TRANSFORMER_PARAM_KEYS) - set(params)
    if missing:
        raise KeyError(
            f"Transformer parameters are missing keys: {sorted(missing)}. "
            f"params={params}"
        )

    return {
        "batch_size": int(params["batch_size"]),
        "learning_rate": float(params["learning_rate"]),
        "d_model": int(params["d_model"]),
        "nhead": int(params["nhead"]),
        "num_layers": int(params["num_layers"]),
        "dim_feedforward": int(params["dim_feedforward"]),
        "dropout": float(params["dropout"]),
        "pseudo_mask_rate": float(params["pseudo_mask_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "grad_clip": float(params["grad_clip"]),
        "internal_val_rate": float(params["internal_val_rate"]),
    }


def _param_id(params: Dict[str, Any]) -> str:
    params = _normalize_params(params)
    return (
        f"bs{params['batch_size']}"
        f"_lr{_float_token(params['learning_rate'])}"
        f"_dm{params['d_model']}"
        f"_nh{params['nhead']}"
        f"_nl{params['num_layers']}"
        f"_ff{params['dim_feedforward']}"
        f"_do{_float_token(params['dropout'])}"
        f"_pm{_float_token(params['pseudo_mask_rate'])}"
        f"_wd{_float_token(params['weight_decay'])}"
        f"_gc{_float_token(params['grad_clip'])}"
        f"_iv{_float_token(params['internal_val_rate'])}"
    )


def _pretty_config(params: Dict[str, Any]) -> str:
    params = _normalize_params(params)
    return (
        f"bs={params['batch_size']}, "
        f"lr={_format_float(params['learning_rate'])}, "
        f"d={params['d_model']}, "
        f"heads={params['nhead']}, "
        f"layers={params['num_layers']}, "
        f"ff={params['dim_feedforward']}, "
        f"drop={_format_float(params['dropout'])}, "
        f"pseudo={_format_float(params['pseudo_mask_rate'])}"
    )


def _load_best_params(
    base_dir: Path,
    study_name: str,
) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    path = (
        base_dir
        / "Optimization"
        / "best_params"
        / f"{study_name}_best_params.json"
    )
    payload = _read_json(path)

    if "params" not in payload:
        raise KeyError(f"Best-parameter JSON does not contain 'params': {path}")

    return _normalize_params(payload["params"]), path, payload


def _extract_params_from_row(row: pd.Series) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if "method_params_json" in row.index:
        params.update(_parse_json_or_literal(row["method_params_json"]))

    for key in TRANSFORMER_PARAM_KEYS:
        for column in (f"method_params_{key}", f"params_{key}", key):
            if column in row.index and not pd.isna(row[column]):
                params[key] = row[column]
                break

    return _normalize_params(params)


# =============================================================================
# Optuna condition-score loading
# =============================================================================
def _load_condition_scores(
    *,
    base_dir: Path,
    study_name: str,
    require_condition_scores: bool,
) -> pd.DataFrame:
    path = (
        base_dir
        / "Optimization"
        / "trial_tables"
        / f"{study_name}_condition_scores.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Transformer per-condition Optuna score table does not exist:\n{path}\n"
            "Run Transformer optimization with the strict seeded optimizer first."
        )

    frame = pd.read_csv(path)

    if frame.empty and require_condition_scores:
        raise RuntimeError(f"Condition-score table is empty: {path}")

    required = {
        "trial_number",
        "state",
        "fraction",
        "missingness",
        "validation_rmse",
    }
    missing = required - set(frame.columns)

    if missing:
        raise KeyError(
            f"Condition-score table is missing columns {sorted(missing)}.\n"
            f"File: {path}\nColumns: {list(frame.columns)}"
        )

    frame = frame[
        frame["state"].astype(str).str.upper() == "COMPLETE"
    ].copy()

    if frame.empty and require_condition_scores:
        raise RuntimeError(f"No COMPLETE rows exist in {path}")

    parsed_rows = []
    for _, row in frame.iterrows():
        params = _extract_params_from_row(row)
        parsed_rows.append(
            {
                "params_dict": params,
                "param_id": _param_id(params),
                "config_label": _pretty_config(params),
            }
        )

    parsed_frame = pd.DataFrame(parsed_rows, index=frame.index)
    frame = pd.concat([frame, parsed_frame], axis=1)

    frame["trial_number"] = pd.to_numeric(
        frame["trial_number"], errors="raise"
    ).astype(int)
    frame["fraction"] = pd.to_numeric(frame["fraction"], errors="raise")
    frame["validation_rmse"] = pd.to_numeric(
        frame["validation_rmse"], errors="raise"
    )
    frame["missingness"] = frame["missingness"].astype(str).str.upper()
    frame["d_model"] = frame["params_dict"].map(lambda item: item["d_model"])
    frame["learning_rate"] = frame["params_dict"].map(
        lambda item: item["learning_rate"]
    )

    if not np.isfinite(frame["validation_rmse"].to_numpy(dtype=float)).all():
        raise ValueError("Transformer condition scores contain non-finite RMSE values.")

    return frame.sort_values(
        ["missingness", "fraction", "trial_number"]
    ).reset_index(drop=True)


# =============================================================================
# Repeated-seed final-result loading
# =============================================================================
def _read_rmse(path: Path) -> float:
    frame = pd.read_csv(path)

    for column in ("rmse_transformer", "rmse_trf", "rmse", "RMSE"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                value = float(values.iloc[0])
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite RMSE in {path}")
                return value

    values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No numeric RMSE value exists in {path}")

    value = float(values.iloc[0])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite RMSE in {path}")
    return value


def _load_seed_results(
    *,
    base_dir: Path,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    best_params: Dict[str, Any],
    strict: bool,
) -> pd.DataFrame:
    param_id = _param_id(best_params)
    rows: List[Dict[str, Any]] = []
    missing_paths: List[str] = []

    for missingness_value in missingness_types:
        missingness = str(missingness_value).upper()

        for fraction_value in fractions:
            fraction = float(fraction_value)
            folder = (
                base_dir
                / f"fraction_{fraction}_missingness_{missingness}"
            )

            for seed_value in seeds:
                seed = int(seed_value)
                rmse_path = (
                    folder
                    / (
                        f"RMSE_TRF_col_{columns_num}_params_{param_id}"
                        f"_seed_{seed}.csv"
                    )
                )
                imputed_path = (
                    folder
                    / (
                        f"imputed_TRF_col_{columns_num}_params_{param_id}"
                        f"_seed_{seed}.csv"
                    )
                )

                if not rmse_path.exists():
                    missing_paths.append(str(rmse_path))
                    continue

                if strict and not imputed_path.exists():
                    missing_paths.append(str(imputed_path))
                    continue

                rows.append(
                    {
                        "method": "Transformer",
                        **best_params,
                        "param_id": param_id,
                        "missingness": missingness,
                        "fraction": fraction,
                        "seed": seed,
                        "rmse": _read_rmse(rmse_path),
                        "rmse_path": str(rmse_path),
                        "imputed_path": str(imputed_path),
                    }
                )

    if missing_paths and strict:
        preview = "\n".join(missing_paths[:50])
        raise FileNotFoundError(
            "Some required repeated-seed Transformer outputs are missing.\n\n"
            f"First missing paths:\n{preview}\n\n"
            "Run the frozen-best Transformer benchmark for every requested seed "
            "and condition."
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No repeated-seed Transformer RMSE outputs were found.")

    expected_count = (
        len(fractions) * len(missingness_types) * len(seeds)
    )
    if strict and len(result) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} repeated-seed rows, found {len(result)}."
        )

    return result.sort_values(
        ["missingness", "fraction", "seed"]
    ).reset_index(drop=True)


# =============================================================================
# Plot helpers
# =============================================================================
def _save_figure(
    figure,
    output_dir: Path,
    stem: str,
    show: bool,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(figure)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _config_table(
    tuning_frame: pd.DataFrame,
    best_param_id: str,
) -> pd.DataFrame:
    rows = []

    for param_id, group in tuning_frame.groupby("param_id"):
        rows.append(
            {
                "param_id": param_id,
                "config_label": str(group["config_label"].iloc[0]),
                "trial_numbers": sorted(
                    set(int(value) for value in group["trial_number"])
                ),
                "mean_validation_rmse": float(group["validation_rmse"].mean()),
                "is_best": param_id == best_param_id,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No Transformer configurations exist for plotting.")

    return table.sort_values(
        ["is_best", "mean_validation_rmse"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _legend_label(row: pd.Series) -> str:
    trials = row["trial_numbers"]
    trial_text = "T" + ",".join(str(value) for value in trials)
    prefix = "BEST | " if bool(row["is_best"]) else ""
    return (
        f"{prefix}{trial_text} | {row['config_label']} | "
        f"mean={float(row['mean_validation_rmse']):.4f}"
    )


def plot_transformer_hyperparameter_effect(
    tuning_frame: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    output_dir: Path,
    study_name: str,
    show: bool,
    show_full_legend: bool,
) -> Dict[str, str]:
    missingness_types = _ordered_missingness(
        tuning_frame["missingness"].unique()
    )
    best_param_id = _param_id(best_params)
    table = _config_table(tuning_frame, best_param_id)
    config_order = table["param_id"].tolist()
    color_map = {
        param_id: plt.get_cmap("tab20")(index % 20)
        for index, param_id in enumerate(config_order)
    }

    right_edge = 0.63 if len(config_order) <= 10 else 0.53
    legend_columns = 1 if len(config_order) <= 10 else 2

    figure, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.0 * len(missingness_types) + 6.0, 5.2),
        sharey=True,
    )
    if len(missingness_types) == 1:
        axes = [axes]

    for axis, missingness in zip(axes, missingness_types):
        subset = tuning_frame[
            tuning_frame["missingness"] == missingness
        ].copy()
        grouped = (
            subset.groupby(["param_id", "fraction"], as_index=False)[
                "validation_rmse"
            ]
            .mean()
            .sort_values(["param_id", "fraction"])
        )
        envelope = (
            grouped.groupby("fraction")["validation_rmse"]
            .agg(["min", "max", "mean"])
            .reset_index()
            .sort_values("fraction")
        )
        x_values = envelope["fraction"].to_numpy(dtype=float)
        axis.fill_between(
            x_values,
            envelope["min"].to_numpy(dtype=float),
            envelope["max"].to_numpy(dtype=float),
            alpha=0.12,
            color="gray",
        )
        axis.plot(
            x_values,
            envelope["mean"].to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=1.8,
        )

        for param_id in config_order:
            group = grouped[grouped["param_id"] == param_id].sort_values(
                "fraction"
            )
            if group.empty:
                continue
            is_best = param_id == best_param_id
            axis.plot(
                group["fraction"],
                group["validation_rmse"],
                color="tab:red" if is_best else color_map[param_id],
                linewidth=3.0 if is_best else 1.5,
                alpha=1.0 if is_best else 0.72,
                marker="o" if is_best else ".",
                markersize=6 if is_best else 4,
            )

        axis.set_title(missingness)
        axis.set_xlabel("Missing fraction")
        axis.grid(True, alpha=0.30)

    axes[0].set_ylabel("Validation RMSE")

    handles = [
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.8,
            label="mean across configurations",
        ),
        Line2D(
            [0], [0], color="gray", linewidth=8, alpha=0.20,
            label="range across configurations",
        ),
    ]

    if show_full_legend:
        for _, row in table.iterrows():
            param_id = row["param_id"]
            is_best = bool(row["is_best"])
            handles.append(
                Line2D(
                    [0], [0],
                    color="tab:red" if is_best else color_map[param_id],
                    linewidth=3.0 if is_best else 1.8,
                    marker="o" if is_best else ".",
                    label=_legend_label(row),
                )
            )

    figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(right_edge + 0.02, 0.5),
        fontsize=7,
        ncol=legend_columns,
        title="Transformer configurations",
        title_fontsize=9,
        frameon=True,
    )
    figure.suptitle(
        f"Transformer hyperparameter effect at fixed seed\nStudy: {study_name}",
        y=1.03,
    )
    figure.tight_layout(rect=[0, 0, right_edge, 1])

    return _save_figure(
        figure,
        output_dir,
        "Transformer_hyperparameter_effect_fixed_seed",
        show,
    )


def plot_transformer_search_distribution(
    tuning_frame: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    output_dir: Path,
    study_name: str,
    show: bool,
) -> Dict[str, str]:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

    d_models = sorted(tuning_frame["d_model"].unique())
    d_model_values = [
        tuning_frame.loc[
            tuning_frame["d_model"] == value,
            "validation_rmse",
        ].to_numpy(dtype=float)
        for value in d_models
    ]
    boxes = axes[0].boxplot(
        d_model_values,
        labels=[str(value) for value in d_models],
        patch_artist=True,
        showmeans=True,
    )
    for patch, value in zip(boxes["boxes"], d_models):
        patch.set_facecolor(
            "tab:red" if int(value) == int(best_params["d_model"]) else "tab:blue"
        )
        patch.set_alpha(0.42 if int(value) == int(best_params["d_model"]) else 0.25)
    axes[0].set_title("d_model")
    axes[0].set_xlabel("Token dimension")
    axes[0].set_ylabel("Validation RMSE")
    axes[0].grid(True, axis="y", alpha=0.30)

    axes[1].scatter(
        tuning_frame["learning_rate"],
        tuning_frame["validation_rmse"],
        alpha=0.70,
        s=32,
        edgecolor="black",
        linewidth=0.25,
    )
    axes[1].axvline(
        best_params["learning_rate"],
        color="tab:red",
        linewidth=2,
        label="selected best",
    )
    axes[1].set_xscale("log")
    axes[1].set_title("learning_rate")
    axes[1].set_xlabel("Learning rate")
    axes[1].grid(True, axis="y", alpha=0.30)
    axes[1].legend(fontsize=8)

    fixed_text = (
        f"Fixed: batch={best_params['batch_size']}, heads={best_params['nhead']}, "
        f"layers={best_params['num_layers']}, dropout={best_params['dropout']}, "
        f"pseudo={best_params['pseudo_mask_rate']}"
    )
    figure.suptitle(
        f"Transformer reduced hyperparameter search\n{fixed_text}\nStudy: {study_name}",
        y=1.06,
    )
    figure.tight_layout()

    return _save_figure(
        figure,
        output_dir,
        "Transformer_hyperparameter_search_distribution",
        show,
    )


def plot_transformer_seed_ribbon(
    seed_frame: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    output_dir: Path,
    show: bool,
) -> Dict[str, str]:
    missingness_types = _ordered_missingness(seed_frame["missingness"].unique())
    figure, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.0 * len(missingness_types), 4.8),
        sharey=True,
    )
    if len(missingness_types) == 1:
        axes = [axes]

    for axis, missingness in zip(axes, missingness_types):
        subset = seed_frame[seed_frame["missingness"] == missingness].copy()

        for seed, group in subset.groupby("seed"):
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
        standard_deviations = summary["std"].fillna(0).to_numpy(dtype=float)
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
        figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))

    figure.suptitle(
        "Transformer seed robustness with frozen best parameters\n"
        + _pretty_config(best_params),
        y=1.04,
    )
    figure.tight_layout(rect=[0, 0, 0.88, 1])

    return _save_figure(
        figure,
        output_dir,
        "Transformer_seed_effect_fixed_best_params_ribbon",
        show,
    )


def plot_transformer_seed_boxplots(
    seed_frame: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    output_dir: Path,
    show: bool,
) -> Dict[str, str]:
    missingness_types = _ordered_missingness(seed_frame["missingness"].unique())
    figure, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.0 * len(missingness_types), 4.8),
        sharey=True,
    )
    if len(missingness_types) == 1:
        axes = [axes]

    for axis, missingness in zip(axes, missingness_types):
        subset = seed_frame[seed_frame["missingness"] == missingness].copy()
        fractions = sorted(subset["fraction"].unique())
        data = [
            subset.loc[np.isclose(subset["fraction"], fraction), "rmse"].to_numpy(dtype=float)
            for fraction in fractions
        ]
        boxes = axis.boxplot(
            data,
            labels=[f"{fraction:g}" for fraction in fractions],
            patch_artist=True,
            showmeans=True,
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor("tab:blue")
            patch.set_alpha(0.35)
        axis.set_title(missingness)
        axis.set_xlabel("Missing fraction")
        axis.grid(True, axis="y", alpha=0.30)

    axes[0].set_ylabel("Final RMSE")
    figure.suptitle(
        "Transformer seed robustness boxplots with frozen best parameters\n"
        + _pretty_config(best_params),
        y=1.04,
    )
    figure.tight_layout()

    return _save_figure(
        figure,
        output_dir,
        "Transformer_seed_effect_fixed_best_params_boxplots",
        show,
    )


# =============================================================================
# Public API
# =============================================================================
def transformer_eval(
    fractions,
    missingness_types,
    columns_num: str = "all",
    study_name: str = "Transformer_balanced_MAR_MNAR_10_30_seeded_v1",
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

    main_dir = Path(config_paths.experiments_dir(columns_num))
    canonical_output_dir = (
        main_dir / "Evaluation" / "Transformer_seeded_pipeline"
    )
    plot_output_dir = (
        canonical_output_dir
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    canonical_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    best_params, best_json_path, _ = _load_best_params(main_dir, study_name)
    tuning_frame = _load_condition_scores(
        base_dir=main_dir,
        study_name=study_name,
        require_condition_scores=require_condition_scores,
    )
    seed_frame = _load_seed_results(
        base_dir=main_dir,
        columns_num=columns_num,
        fractions=fractions,
        missingness_types=missingness_types,
        seeds=seeds,
        best_params=best_params,
        strict=strict,
    )

    # Machine-readable outputs always use the canonical location expected by the
    # cross-method seeded comparison evaluator.
    tuning_csv = canonical_output_dir / "Transformer_optuna_per_condition_validation_rmse.csv"
    seed_csv = canonical_output_dir / "Transformer_final_rmse_by_seed.csv"
    seed_summary_csv = canonical_output_dir / "Transformer_final_rmse_by_seed_summary.csv"
    config_csv = canonical_output_dir / "Transformer_hyperparameter_config_legend.csv"

    tuning_frame.to_csv(tuning_csv, index=False)
    seed_frame.to_csv(seed_csv, index=False)
    (
        seed_frame.groupby(["missingness", "fraction"])["rmse"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .to_csv(seed_summary_csv, index=False)
    )
    table = _config_table(tuning_frame, _param_id(best_params))
    table.to_csv(config_csv, index=False)

    outputs: Dict[str, Any] = {
        "best_params": best_params,
        "best_params_path": str(best_json_path),
        "tuning_csv": str(tuning_csv),
        "seed_csv": str(seed_csv),
        "seed_summary_csv": str(seed_summary_csv),
        "config_legend_csv": str(config_csv),
        "plots": {},
    }

    outputs["plots"]["hyperparameter_effect"] = plot_transformer_hyperparameter_effect(
        tuning_frame,
        best_params=best_params,
        output_dir=plot_output_dir,
        study_name=study_name,
        show=show,
        show_full_legend=show_full_hyperparameter_legend,
    )
    outputs["plots"]["hyperparameter_search_distribution"] = plot_transformer_search_distribution(
        tuning_frame,
        best_params=best_params,
        output_dir=plot_output_dir,
        study_name=study_name,
        show=show,
    )
    outputs["plots"]["seed_ribbon"] = plot_transformer_seed_ribbon(
        seed_frame,
        best_params=best_params,
        output_dir=plot_output_dir,
        show=show,
    )
    outputs["plots"]["seed_boxplots"] = plot_transformer_seed_boxplots(
        seed_frame,
        best_params=best_params,
        output_dir=plot_output_dir,
        show=show,
    )

    report_path = plot_output_dir / "Transformer_evaluation_suite_report.pdf"
    with PdfPages(report_path) as report:
        figure, axis = plt.subplots(figsize=(8.5, 11))
        axis.axis("off")
        lines = [
            "Transformer seeded evaluation suite",
            "",
            f"Study: {study_name}",
            f"Best parameters: {_pretty_config(best_params)}",
            "",
            "Individual plot files:",
            *[
                f"- {value['pdf']}"
                for value in outputs["plots"].values()
            ],
            "",
            "Canonical CSV outputs:",
            f"- {tuning_csv}",
            f"- {seed_csv}",
            f"- {seed_summary_csv}",
            f"- {config_csv}",
        ]
        y_position = 0.95
        for index, line in enumerate(lines):
            axis.text(
                0.05,
                y_position,
                line,
                fontsize=16 if index == 0 else 10,
                fontweight="bold" if index == 0 else "normal",
                va="top",
                transform=axis.transAxes,
            )
            y_position -= 0.042 if index == 0 else 0.028
        report.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    outputs["combined_report_pdf"] = str(report_path)

    print("\nTransformer evaluation complete")
    print(f"Canonical seed CSV: {seed_csv}")
    print(f"Plot directory: {plot_output_dir}")

    return outputs


# Compatibility aliases.
trf_eval = transformer_eval
TRF_eval = transformer_eval
