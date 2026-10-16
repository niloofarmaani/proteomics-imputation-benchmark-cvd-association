# Evaluation/GAIN_eval.py

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
# Generic helpers
# =============================================================================
def _ordered_missingness(values: Sequence[str]) -> List[str]:
    preferred = ["MCAR", "MAR", "MNAR"]
    values = [str(value).upper() for value in values]
    present = [value for value in preferred if value in set(values)]
    extras = sorted(value for value in set(values) if value not in present)
    return present + extras


def _float_token(x: float) -> str:
    return (
        f"{float(x):.12g}"
        .replace("-", "m")
        .replace(".", "p")
        .replace("+", "")
    )


def _text_token(x: Any) -> str:
    text = str(x).strip().lower()
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _format_float(x: float) -> str:
    x = float(x)
    if abs(x) < 0.001 or abs(x) >= 100:
        return f"{x:.2e}"
    return f"{x:.4g}"




def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value}")

def _parse_n_neurons(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        parsed = [int(x) for x in value]
    elif isinstance(value, str):
        text = value.strip()
        try:
            literal = ast.literal_eval(text)
            if isinstance(literal, (list, tuple)):
                parsed = [int(x) for x in literal]
            else:
                raise ValueError
        except Exception:
            text = text.replace("[", "").replace("]", "")
            text = text.replace("(", "").replace(")", "")
            text = text.replace("-", ",")
            parsed = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        raise ValueError(f"Could not parse n_neurons from: {value}")

    if not parsed or any(x <= 0 for x in parsed):
        raise ValueError(f"Invalid n_neurons: {parsed}")
    return parsed


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found:\n{path}")
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

    return {}


# =============================================================================
# GAIN parameter helpers
# =============================================================================
_REQUIRED_GAIN_PARAMS = {
    "batch_size",
    "lr",
    "n_neurons",
    "alpha",
    "hint_rate",
    "val_rate",
    "validation_mode",
    "refit_final",
    "mc_samples_validation",
    "mc_samples_final",
    "output_activation",
    "noise_type",
    "noise_scale",
    "dropout_rate",
    "weight_decay",
    "grad_clip",
    "pseudo_rate",
    "pseudo_beta",
    "hint_mode",
}


def _normalize_gain_params(params: Dict[str, Any]) -> Dict[str, Any]:
    missing = _REQUIRED_GAIN_PARAMS - set(params)
    if missing:
        raise KeyError(f"GAIN params missing required keys: {sorted(missing)}")

    noise_scale = params["noise_scale"]
    if isinstance(noise_scale, str) and noise_scale.strip().lower() in {
        "none",
        "null",
        "nan",
        "",
    }:
        noise_scale = None

    return {
        "batch_size": int(params["batch_size"]),
        "lr": float(params["lr"]),
        "n_neurons": _parse_n_neurons(params["n_neurons"]),
        "alpha": float(params["alpha"]),
        "hint_rate": float(params["hint_rate"]),
        "val_rate": float(params["val_rate"]),
        "validation_mode": str(params["validation_mode"]),
        "refit_final": _as_bool(params["refit_final"]),
        "mc_samples_validation": int(params["mc_samples_validation"]),
        "mc_samples_final": int(params["mc_samples_final"]),
        "output_activation": str(params["output_activation"]),
        "noise_type": str(params["noise_type"]),
        "noise_scale": None if noise_scale is None else float(noise_scale),
        "dropout_rate": float(params["dropout_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "grad_clip": float(params["grad_clip"]),
        "pseudo_rate": float(params["pseudo_rate"]),
        "pseudo_beta": float(params["pseudo_beta"]),
        "hint_mode": str(params["hint_mode"]),
    }


def _gain_param_id(
    params: Dict[str, Any],
    *,
    epochs: int,
    patience: int,
) -> str:
    params = _normalize_gain_params(params)
    neurons = "-".join(str(x) for x in params["n_neurons"])

    return (
        f"bs{params['batch_size']}"
        f"_lr{_float_token(params['lr'])}"
        f"_nn{neurons}"
        f"_al{_float_token(params['alpha'])}"
        f"_hr{_float_token(params['hint_rate'])}"
        f"_do{_float_token(params['dropout_rate'])}"
        f"_wd{_float_token(params['weight_decay'])}"
        f"_ps{_float_token(params['pseudo_rate'])}"
        f"_pb{_float_token(params['pseudo_beta'])}"
        f"_hm{_text_token(params['hint_mode'])}"
        f"_mc{params['mc_samples_final']}"
        f"_ep{int(epochs)}"
        f"_pat{int(patience)}"
    )


def _pretty_gain_config_label(params: Dict[str, Any]) -> str:
    params = _normalize_gain_params(params)
    architecture = "-".join(str(x) for x in params["n_neurons"])

    return (
        f"bs={params['batch_size']}, "
        f"lr={_format_float(params['lr'])}, "
        f"nn={architecture}, "
        f"a={_format_float(params['alpha'])}, "
        f"h={_format_float(params['hint_rate'])}, "
        f"do={_format_float(params['dropout_rate'])}, "
        f"wd={_format_float(params['weight_decay'])}, "
        f"ps={_format_float(params['pseudo_rate'])}, "
        f"pb={_format_float(params['pseudo_beta'])}"
    )


def _load_best_params(
    base_dir: Path,
    study_name: str,
) -> Tuple[Dict[str, Any], int, int, Path, Dict[str, Any]]:
    path = base_dir / "Optimization" / "best_params" / f"{study_name}_best_params.json"
    payload = _read_json(path)

    if "params" not in payload:
        raise KeyError(f"Best-params JSON has no 'params' key:\n{path}")
    if "epochs" not in payload or "patience" not in payload:
        raise KeyError(f"Best-params JSON must contain epochs and patience:\n{path}")

    params = _normalize_gain_params(payload["params"])
    return params, int(payload["epochs"]), int(payload["patience"]), path, payload


def _extract_params_from_row(row: pd.Series) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if "method_params_json" in row.index:
        params.update(_parse_json_or_literal(row["method_params_json"]))

    for key in sorted(_REQUIRED_GAIN_PARAMS):
        for column in [f"method_params_{key}", f"params_{key}", key]:
            if column in row.index:
                value = row[column]
                try:
                    is_missing = pd.isna(value)
                except Exception:
                    is_missing = False
                if not is_missing:
                    params[key] = value
                    break

    return _normalize_gain_params(params)


# =============================================================================
# Input-table loading
# =============================================================================
def _load_gain_condition_scores(
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
            f"Per-condition Optuna score table not found:\n{path}\n"
            "Rerun GAIN optimization with the strict seeded run_bayes_opt.py."
        )

    frame = pd.read_csv(path)
    if frame.empty and require_condition_scores:
        raise RuntimeError(f"GAIN condition-score table is empty:\n{path}")

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
            f"GAIN condition-score table is missing {sorted(missing)}.\n"
            f"Columns: {list(frame.columns)}"
        )

    frame = frame[frame["state"].astype(str).str.upper() == "COMPLETE"].copy()
    if frame.empty and require_condition_scores:
        raise RuntimeError(f"No COMPLETE rows found in:\n{path}")

    params_rows = []
    for _, row in frame.iterrows():
        params = _extract_params_from_row(row)
        params_rows.append(
            (
                params,
                _pretty_gain_config_label(params),
                json.dumps(params, sort_keys=True, default=str),
            )
        )

    frame["params_dict"] = [item[0] for item in params_rows]
    frame["config_label"] = [item[1] for item in params_rows]
    frame["config_id"] = [item[2] for item in params_rows]

    for key in [
        "batch_size",
        "lr",
        "alpha",
        "hint_rate",
        "dropout_rate",
        "weight_decay",
        "pseudo_rate",
        "pseudo_beta",
    ]:
        frame[key] = frame["params_dict"].map(lambda value, k=key: value[k])

    frame["n_neurons_label"] = frame["params_dict"].map(
        lambda value: "-".join(str(x) for x in value["n_neurons"])
    )

    frame["trial_number"] = pd.to_numeric(frame["trial_number"], errors="raise").astype(int)
    frame["fraction"] = pd.to_numeric(frame["fraction"], errors="raise")
    frame["missingness"] = frame["missingness"].astype(str).str.upper()
    frame["validation_rmse"] = pd.to_numeric(frame["validation_rmse"], errors="raise")

    return frame.sort_values(
        ["missingness", "fraction", "trial_number"]
    ).reset_index(drop=True)


def _read_rmse(path: Path) -> float:
    frame = pd.read_csv(path)
    for column in ["rmse_gain", "rmse", "RMSE"]:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                return float(values.iloc[0])

    values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
    if not len(values):
        raise ValueError(f"No numeric RMSE value was found in:\n{path}")
    return float(values.iloc[0])


def _load_gain_seed_rmse_values(
    *,
    base_dir: Path,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    best_params: Dict[str, Any],
    epochs: int,
    patience: int,
    strict: bool,
) -> pd.DataFrame:
    parameter_id = _gain_param_id(
        best_params,
        epochs=epochs,
        patience=patience,
    )

    rows = []
    missing_files = []

    for missingness in missingness_types:
        missingness = str(missingness).upper()
        for fraction in fractions:
            fraction = float(fraction)
            folder = base_dir / f"fraction_{fraction}_missingness_{missingness}"

            for seed in seeds:
                seed = int(seed)
                rmse_path = (
                    folder
                    / f"RMSE_GAIN_col_{columns_num}_params_{parameter_id}_seed_{seed}.csv"
                )
                imputed_path = (
                    folder
                    / f"imputed_GAIN_col_{columns_num}_params_{parameter_id}_seed_{seed}.csv"
                )

                if not rmse_path.exists():
                    missing_files.append(str(rmse_path))
                    continue
                if strict and not imputed_path.exists():
                    missing_files.append(str(imputed_path))
                    continue

                rows.append(
                    {
                        "method": "GAIN",
                        "parameter_id": parameter_id,
                        "missingness": missingness,
                        "fraction": fraction,
                        "seed": seed,
                        "rmse": _read_rmse(rmse_path),
                        "rmse_path": str(rmse_path),
                        "imputed_path": str(imputed_path),
                    }
                )

    if missing_files and strict:
        preview = "\n".join(missing_files[:50])
        raise FileNotFoundError(
            "Some required repeated-seed GAIN outputs are missing.\n\n"
            f"First missing files:\n{preview}"
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No repeated-seed GAIN RMSE files were found.")

    return frame.sort_values(
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
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return {"png": str(png), "pdf": str(pdf)}


def _config_color_map(config_ids: Sequence[str]) -> Dict[str, Any]:
    config_ids = list(config_ids)
    color_map = plt.get_cmap("tab20")
    return {
        config_id: color_map(index % color_map.N)
        for index, config_id in enumerate(config_ids)
    }


def _make_config_legend_table(
    tuning_df: pd.DataFrame,
    best_config_id: str,
) -> pd.DataFrame:
    rows = []
    for config_id, group in tuning_df.groupby("config_id"):
        rows.append(
            {
                "config_id": config_id,
                "config_label": str(group["config_label"].iloc[0]),
                "trial_numbers": sorted(
                    set(int(value) for value in group["trial_number"])
                ),
                "mean_validation_rmse": float(group["validation_rmse"].mean()),
                "is_best": config_id == best_config_id,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No GAIN configurations were found for the legend.")

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


# =============================================================================
# Plots
# =============================================================================
def plot_gain_hyperparameter_effect(
    tuning_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    output_dir: Path,
    study_name: str,
    show: bool,
    show_full_hyperparameter_legend: bool,
) -> Dict[str, str]:
    best_config_id = json.dumps(
        _normalize_gain_params(best_params), sort_keys=True, default=str
    )
    config_table = _make_config_legend_table(tuning_df, best_config_id)
    config_order = config_table["config_id"].tolist()
    colors = _config_color_map(config_order)
    missingness_types = _ordered_missingness(tuning_df["missingness"].unique())

    n_configs = len(config_order)
    legend_columns = 1 if n_configs <= 10 else 2
    right_edge = 0.55 if n_configs > 10 else 0.65

    figure, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.2 * len(missingness_types) + 7.0, 5.4),
        sharey=True,
    )
    if len(missingness_types) == 1:
        axes = [axes]

    for axis, missingness in zip(axes, missingness_types):
        subset = tuning_df[tuning_df["missingness"] == missingness].copy()
        by_config = (
            subset.groupby(["config_id", "fraction"], as_index=False)[
                "validation_rmse"
            ]
            .mean()
            .sort_values(["config_id", "fraction"])
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
            alpha=0.12,
        )
        axis.plot(
            x_values,
            band["mean"].to_numpy(dtype=float),
            linestyle="--",
            linewidth=1.8,
        )

        for config_id in config_order:
            group = by_config[by_config["config_id"] == config_id].sort_values(
                "fraction"
            )
            if group.empty:
                continue
            is_best = config_id == best_config_id
            axis.plot(
                group["fraction"],
                group["validation_rmse"],
                linewidth=3.2 if is_best else 1.4,
                marker="o" if is_best else ".",
                markersize=6 if is_best else 4,
                alpha=1.0 if is_best else 0.72,
                color="tab:red" if is_best else colors[config_id],
            )

        axis.set_title(missingness)
        axis.set_xlabel("Missing fraction")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Validation RMSE")

    handles = [
        Line2D([0], [0], linestyle="--", linewidth=1.8, label="mean across configs"),
        Line2D([0], [0], linewidth=8, alpha=0.2, label="range across configs"),
    ]
    if show_full_hyperparameter_legend:
        for _, row in config_table.iterrows():
            config_id = row["config_id"]
            is_best = bool(row["is_best"])
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="tab:red" if is_best else colors[config_id],
                    linewidth=3.2 if is_best else 1.8,
                    marker="o" if is_best else ".",
                    markersize=6 if is_best else 4,
                    label=_legend_label(row),
                )
            )

    figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(right_edge + 0.02, 0.5),
        fontsize=6.8,
        ncol=legend_columns,
        title="GAIN hyperparameter configurations",
        title_fontsize=9,
        frameon=True,
    )
    figure.suptitle(
        f"GAIN hyperparameter effect under fixed optimization seed\nStudy: {study_name}",
        y=1.03,
    )
    figure.tight_layout(rect=[0, 0, right_edge, 1])

    return _save_figure(
        figure,
        output_dir,
        "GAIN_hyperparameter_effect_fixed_seed",
        show,
    )


def plot_gain_hyperparameter_search_distribution(
    tuning_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    output_dir: Path,
    study_name: str,
    show: bool,
) -> Dict[str, str]:
    best_params = _normalize_gain_params(best_params)
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 12.0), sharey=True)
    axes = axes.ravel()

    specifications = [
        ("batch_size", best_params["batch_size"], "category"),
        (
            "n_neurons_label",
            "-".join(str(x) for x in best_params["n_neurons"]),
            "category",
        ),
        ("alpha", best_params["alpha"], "category"),
        ("hint_rate", best_params["hint_rate"], "category"),
        ("dropout_rate", best_params["dropout_rate"], "category"),
        ("pseudo_rate", best_params["pseudo_rate"], "category"),
        ("pseudo_beta", best_params["pseudo_beta"], "category"),
        ("lr", best_params["lr"], "scatter_log"),
        ("weight_decay", best_params["weight_decay"], "scatter_log"),
    ]

    for axis, (parameter, best_value, mode) in zip(axes, specifications):
        if mode == "category":
            values = sorted(
                tuning_df[parameter].dropna().unique(), key=lambda value: str(value)
            )
            data = [
                tuning_df.loc[
                    tuning_df[parameter] == value, "validation_rmse"
                ].to_numpy(dtype=float)
                for value in values
            ]
            boxplot = axis.boxplot(
                data,
                labels=[str(value) for value in values],
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                },
            )
            for patch, value in zip(boxplot["boxes"], values):
                if str(value) == str(best_value):
                    patch.set_facecolor("tab:red")
                    patch.set_alpha(0.45)
                else:
                    patch.set_alpha(0.25)
            axis.tick_params(axis="x", rotation=35)
        else:
            axis.scatter(
                tuning_df[parameter],
                tuning_df["validation_rmse"],
                alpha=0.65,
                s=28,
                edgecolor="black",
                linewidth=0.25,
            )
            axis.axvline(
                float(best_value),
                linewidth=2.0,
                color="tab:red",
                label="selected best",
            )
            axis.set_xscale("log")
            axis.legend(fontsize=8)

        axis.set_title(parameter)
        axis.set_xlabel(parameter)
        axis.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Validation RMSE")
    axes[3].set_ylabel("Validation RMSE")
    axes[6].set_ylabel("Validation RMSE")
    figure.suptitle(
        f"GAIN hyperparameter search distributions\nStudy: {study_name}",
        y=1.01,
    )
    figure.tight_layout()

    return _save_figure(
        figure,
        output_dir,
        "GAIN_hyperparameter_search_distribution",
        show,
    )


def plot_gain_seed_ribbon(
    seed_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    output_dir: Path,
    show: bool,
) -> Dict[str, str]:
    frame = seed_df.copy()
    frame["missingness"] = frame["missingness"].astype(str).str.upper()
    frame["fraction"] = pd.to_numeric(frame["fraction"], errors="raise")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["rmse"] = pd.to_numeric(frame["rmse"], errors="raise")

    missingness_types = _ordered_missingness(frame["missingness"].unique())
    best_label = _pretty_gain_config_label(best_params)
    figure, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.2 * len(missingness_types), 4.8),
        sharey=True,
    )
    if len(missingness_types) == 1:
        axes = [axes]

    for axis, missingness in zip(axes, missingness_types):
        subset = frame[frame["missingness"] == missingness]
        for _, group in subset.groupby("seed"):
            group = group.sort_values("fraction")
            axis.plot(
                group["fraction"],
                group["rmse"],
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
            alpha=0.15,
            label="mean ± 1 SD",
        )
        axis.plot(
            x_values,
            means,
            linewidth=2.5,
            marker="o",
            label="mean",
        )
        axis.set_title(missingness)
        axis.set_xlabel("Missing fraction")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Final RMSE")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
    figure.suptitle(
        "GAIN seed robustness with frozen best hyperparameters\n" + best_label,
        y=1.04,
    )
    figure.tight_layout(rect=[0, 0, 0.88, 1])

    return _save_figure(
        figure,
        output_dir,
        "GAIN_seed_effect_fixed_best_params_ribbon",
        show,
    )


def plot_gain_seed_boxplots(
    seed_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    output_dir: Path,
    show: bool,
) -> Dict[str, str]:
    frame = seed_df.copy()
    frame["missingness"] = frame["missingness"].astype(str).str.upper()
    frame["fraction"] = pd.to_numeric(frame["fraction"], errors="raise")
    frame["rmse"] = pd.to_numeric(frame["rmse"], errors="raise")

    missingness_types = _ordered_missingness(frame["missingness"].unique())
    best_label = _pretty_gain_config_label(best_params)
    figure, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.2 * len(missingness_types), 4.8),
        sharey=True,
    )
    if len(missingness_types) == 1:
        axes = [axes]

    for axis, missingness in zip(axes, missingness_types):
        subset = frame[frame["missingness"] == missingness]
        fractions = sorted(subset["fraction"].unique())
        data = [
            subset.loc[np.isclose(subset["fraction"], fraction), "rmse"].to_numpy(
                dtype=float
            )
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
            patch.set_alpha(0.35)
        axis.set_title(missingness)
        axis.set_xlabel("Missing fraction")
        axis.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Final RMSE")
    figure.suptitle(
        "GAIN seed robustness boxplots with frozen best hyperparameters\n"
        + best_label,
        y=1.04,
    )
    figure.tight_layout()

    return _save_figure(
        figure,
        output_dir,
        "GAIN_seed_effect_fixed_best_params_boxplots",
        show,
    )


# =============================================================================
# Public evaluator
# =============================================================================
def gain_eval(
    fractions,
    missingness_types,
    columns_num: str = "all",
    study_name: str = "GAIN_balanced_MAR_MNAR_10_30_seeded_v1",
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

    base_dir = Path(config_paths.experiments_dir(columns_num))
    if output_dir is None:
        out_dir = base_dir / "Evaluation" / "GAIN_seeded_pipeline"
    else:
        out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_params, epochs, patience, best_params_path, _ = _load_best_params(
        base_dir, study_name
    )

    print("\n" + "=" * 80)
    print("GAIN evaluation suite")
    print("=" * 80)
    print(f"Study name: {study_name}")
    print(f"Best params JSON: {best_params_path}")
    print(f"Best params: {best_params}")
    print(f"Epochs: {epochs}; patience: {patience}")
    print(f"Fractions: {fractions}")
    print(f"Missingness types: {missingness_types}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    tuning_df = _load_gain_condition_scores(
        base_dir=base_dir,
        study_name=study_name,
        require_condition_scores=require_condition_scores,
    )
    tuning_csv = out_dir / "GAIN_optuna_per_condition_validation_rmse.csv"
    tuning_df.to_csv(tuning_csv, index=False)

    best_config_id = json.dumps(best_params, sort_keys=True, default=str)
    config_table = _make_config_legend_table(tuning_df, best_config_id)
    config_table_csv = out_dir / "GAIN_hyperparameter_config_legend.csv"
    config_table.to_csv(config_table_csv, index=False)

    seed_df = _load_gain_seed_rmse_values(
        base_dir=base_dir,
        columns_num=columns_num,
        fractions=fractions,
        missingness_types=missingness_types,
        seeds=seeds,
        best_params=best_params,
        epochs=epochs,
        patience=patience,
        strict=strict,
    )
    seed_csv = out_dir / "GAIN_final_rmse_by_seed.csv"
    seed_df.to_csv(seed_csv, index=False)

    seed_summary = (
        seed_df.groupby(["missingness", "fraction"])["rmse"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    seed_summary_csv = out_dir / "GAIN_final_rmse_by_seed_summary.csv"
    seed_summary.to_csv(seed_summary_csv, index=False)

    outputs: Dict[str, Any] = {
        "best_params": best_params,
        "best_params_path": str(best_params_path),
        "tuning_csv": str(tuning_csv),
        "config_legend_csv": str(config_table_csv),
        "seed_csv": str(seed_csv),
        "seed_summary_csv": str(seed_summary_csv),
        "plots": {},
    }

    outputs["plots"]["hyperparameter_effect"] = plot_gain_hyperparameter_effect(
        tuning_df,
        best_params=best_params,
        output_dir=out_dir,
        study_name=study_name,
        show=show,
        show_full_hyperparameter_legend=show_full_hyperparameter_legend,
    )
    outputs["plots"][
        "hyperparameter_search_distribution"
    ] = plot_gain_hyperparameter_search_distribution(
        tuning_df,
        best_params=best_params,
        output_dir=out_dir,
        study_name=study_name,
        show=show,
    )
    outputs["plots"]["seed_ribbon"] = plot_gain_seed_ribbon(
        seed_df,
        best_params=best_params,
        output_dir=out_dir,
        show=show,
    )
    outputs["plots"]["seed_boxplots"] = plot_gain_seed_boxplots(
        seed_df,
        best_params=best_params,
        output_dir=out_dir,
        show=show,
    )

    report_pdf = out_dir / "GAIN_evaluation_suite_report.pdf"
    with PdfPages(report_pdf) as pdf:
        figure, axis = plt.subplots(figsize=(8.5, 11))
        axis.axis("off")
        lines = [
            "GAIN evaluation suite",
            "",
            f"Study name: {study_name}",
            f"Best params: {best_params}",
            f"Epochs: {epochs}; patience: {patience}",
            "",
            "Generated plots:",
            f"1. {outputs['plots']['hyperparameter_effect']['pdf']}",
            f"2. {outputs['plots']['hyperparameter_search_distribution']['pdf']}",
            f"3. {outputs['plots']['seed_ribbon']['pdf']}",
            f"4. {outputs['plots']['seed_boxplots']['pdf']}",
            "",
            "CSV outputs:",
            f"- {tuning_csv}",
            f"- {config_table_csv}",
            f"- {seed_csv}",
            f"- {seed_summary_csv}",
        ]
        y_position = 0.95
        for index, line in enumerate(lines):
            axis.text(
                0.05,
                y_position,
                line,
                fontsize=16 if index == 0 else 10,
                fontweight="bold" if index == 0 else "normal",
                ha="left",
                va="top",
                transform=axis.transAxes,
            )
            y_position -= 0.04 if index == 0 else 0.028
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    outputs["combined_report_pdf"] = str(report_pdf)

    print("\n" + "=" * 80)
    print("GAIN evaluation complete")
    print("=" * 80)
    for key, value in outputs["plots"].items():
        print(f"{key}: {value['pdf']}")
    print(f"Report: {report_pdf}")
    print("=" * 80)

    return outputs


GAIN_eval = gain_eval
