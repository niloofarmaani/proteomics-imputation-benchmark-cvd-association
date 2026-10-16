# Evaluation/AE_eval.py

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
# Helpers
# =============================================================================
def _ordered_missingness(values: Sequence[str]) -> List[str]:
    preferred = ["MCAR", "MAR", "MNAR"]
    values = [str(v).upper() for v in values]
    present = [m for m in preferred if m in set(values)]
    extras = sorted([m for m in set(values) if m not in present])
    return present + extras


def _float_token(x: float) -> str:
    s = f"{float(x):.12g}"
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def _format_lr(x: float) -> str:
    return f"{float(x):.2e}"


def _parse_n_neurons(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]

    if isinstance(value, str):
        s = value.strip()

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            pass

        s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        s = s.replace("-", ",")
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    raise ValueError(f"Could not parse n_neurons from value: {value}")


def _param_id_from_params(params: Dict[str, Any]) -> str:
    required = {"batch_size", "learning_rate", "n_neurons", "dim_latent"}
    missing = required - set(params.keys())

    if missing:
        raise KeyError(f"AE params missing required keys: {sorted(missing)}. params={params}")

    n_neurons = _parse_n_neurons(params["n_neurons"])
    nn_token = "-".join(str(x) for x in n_neurons)

    return (
        f"bs{int(params['batch_size'])}"
        f"_lr{_float_token(float(params['learning_rate']))}"
        f"_nn{nn_token}"
        f"_dl{int(params['dim_latent'])}"
    )


def _pretty_ae_config_label(params: Dict[str, Any]) -> str:
    n_neurons = _parse_n_neurons(params["n_neurons"])
    nn_token = "-".join(str(x) for x in n_neurons)

    return (
        f"bs={int(params['batch_size'])}, "
        f"lr={_format_lr(float(params['learning_rate']))}, "
        f"nn={nn_token}, "
        f"dl={int(params['dim_latent'])}"
    )


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found:\n{path}")

    with open(path, "r") as f:
        return json.load(f)


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

    s = str(raw).strip()

    if not s or s.lower() in {"none", "nan", "null"}:
        return {}

    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def _load_best_params(base_dir: Path, study_name: str) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    best_params_path = base_dir / "Optimization" / "best_params" / f"{study_name}_best_params.json"
    payload = _read_json(best_params_path)

    params = payload.get("params", {})

    required = {"batch_size", "learning_rate", "n_neurons", "dim_latent"}
    missing = required - set(params.keys())

    if missing:
        raise KeyError(
            f"Best params JSON exists but is missing required AE params: {sorted(missing)}\n"
            f"File: {best_params_path}\n"
            f"params: {params}"
        )

    params = {
        "batch_size": int(params["batch_size"]),
        "learning_rate": float(params["learning_rate"]),
        "n_neurons": _parse_n_neurons(params["n_neurons"]),
        "dim_latent": int(params["dim_latent"]),
        "freebits": float(params.get("freebits", 0.1)),
    }

    return params, best_params_path, payload


def _extract_params_from_row(row: pd.Series) -> Dict[str, Any]:
    params = {}

    if "method_params_json" in row.index:
        params.update(_parse_json_or_literal(row["method_params_json"]))

    for key in ["batch_size", "learning_rate", "n_neurons", "dim_latent", "freebits"]:
        for col in [f"method_params_{key}", f"params_{key}", key]:
            if col in row.index and not pd.isna(row[col]):
                params[key] = row[col]
                break

    required = {"batch_size", "learning_rate", "n_neurons", "dim_latent"}
    missing = required - set(params.keys())

    if missing:
        raise KeyError(
            f"Could not extract AE params from condition-score row. Missing: {sorted(missing)}\n"
            f"Available row columns: {list(row.index)}"
        )

    params["batch_size"] = int(params["batch_size"])
    params["learning_rate"] = float(params["learning_rate"])
    params["n_neurons"] = _parse_n_neurons(params["n_neurons"])
    params["dim_latent"] = int(params["dim_latent"])
    params["freebits"] = float(params.get("freebits", 0.1))

    return params


def _load_ae_condition_scores(
    *,
    base_dir: Path,
    study_name: str,
    require_condition_scores: bool = True,
) -> pd.DataFrame:
    condition_csv = base_dir / "Optimization" / "trial_tables" / f"{study_name}_condition_scores.csv"

    if not condition_csv.exists():
        raise FileNotFoundError(
            f"Per-condition Optuna score table not found:\n{condition_csv}\n\n"
            "Rerun AE optimization with the updated run_bayes_opt.py."
        )

    df = pd.read_csv(condition_csv)

    if df.empty and require_condition_scores:
        raise RuntimeError(f"Condition score table is empty:\n{condition_csv}")

    required = {"trial_number", "state", "fraction", "missingness", "validation_rmse"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Condition score table is missing required columns {sorted(missing)}.\n"
            f"File: {condition_csv}\n"
            f"Columns: {list(df.columns)}"
        )

    df = df[df["state"].astype(str).str.upper() == "COMPLETE"].copy()

    if df.empty and require_condition_scores:
        raise RuntimeError(f"No COMPLETE rows found in condition score table:\n{condition_csv}")

    param_rows = []

    for _, row in df.iterrows():
        params = _extract_params_from_row(row)
        param_id = _param_id_from_params(params)
        pretty_label = _pretty_ae_config_label(params)
        param_rows.append((params, param_id, pretty_label))

    df["params_dict"] = [p for p, _, _ in param_rows]
    df["param_id"] = [pid for _, pid, _ in param_rows]
    df["config_label"] = [lab for _, _, lab in param_rows]

    df["batch_size"] = df["params_dict"].map(lambda d: int(d["batch_size"]))
    df["learning_rate"] = df["params_dict"].map(lambda d: float(d["learning_rate"]))
    df["dim_latent"] = df["params_dict"].map(lambda d: int(d["dim_latent"]))
    df["freebits"] = df["params_dict"].map(lambda d: float(d.get("freebits", 0.1)))
    df["n_neurons_label"] = df["params_dict"].map(
        lambda d: "-".join(str(int(x)) for x in d["n_neurons"])
    )

    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="raise").astype(int)
    df["fraction"] = pd.to_numeric(df["fraction"], errors="raise")
    df["missingness"] = df["missingness"].astype(str).str.upper()
    df["validation_rmse"] = pd.to_numeric(df["validation_rmse"], errors="raise")
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="raise").astype(int)
    df["learning_rate"] = pd.to_numeric(df["learning_rate"], errors="raise")
    df["dim_latent"] = pd.to_numeric(df["dim_latent"], errors="raise").astype(int)

    return df.sort_values(["missingness", "fraction", "trial_number"]).reset_index(drop=True)


def _read_rmse_first_value(rmse_path: Path) -> float:
    df = pd.read_csv(rmse_path)

    preferred_cols = ["rmse_ae", "rmse_vae", "rmse", "RMSE"]

    for col in preferred_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                return float(vals.iloc[0])

    vals = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()

    if len(vals) == 0:
        raise ValueError(f"Could not find numeric RMSE value in {rmse_path}")

    return float(vals.iloc[0])


def _load_ae_seed_rmse_values(
    *,
    base_dir: Path,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    best_params: Dict[str, Any],
    strict: bool = True,
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

                rmse_path = (
                    folder
                    / f"RMSE_AE_col_{columns_num}_params_{param_id}_seed_{seed}.csv"
                )

                imputed_path = (
                    folder
                    / f"imputed_AE_col_{columns_num}_params_{param_id}_seed_{seed}.csv"
                )

                if not rmse_path.exists():
                    missing_files.append(str(rmse_path))
                    continue

                if strict and not imputed_path.exists():
                    missing_files.append(str(imputed_path))
                    continue

                rmse = _read_rmse_first_value(rmse_path)

                rows.append(
                    {
                        "method": "AE",
                        "param_id": param_id,
                        "batch_size": int(best_params["batch_size"]),
                        "learning_rate": float(best_params["learning_rate"]),
                        "n_neurons": "-".join(str(x) for x in best_params["n_neurons"]),
                        "dim_latent": int(best_params["dim_latent"]),
                        "missingness": missingness,
                        "fraction": fraction,
                        "seed": seed,
                        "rmse": rmse,
                        "rmse_path": str(rmse_path),
                        "imputed_path": str(imputed_path),
                    }
                )

    if missing_files and strict:
        preview = "\n".join(missing_files[:40])
        raise FileNotFoundError(
            "Some required repeated-seed final AE benchmark files are missing.\n\n"
            f"First missing files:\n{preview}\n\n"
            "Run the AE repeated final benchmark for all requested seeds, fractions, "
            "and missingness types."
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No AE repeated-seed RMSE files were found.")

    return df.sort_values(["missingness", "fraction", "seed"]).reset_index(drop=True)


def _save_current_figure(fig, out_dir: Path, stem: str, show: bool) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return {"png": str(png), "pdf": str(pdf)}


def _config_color_map(config_ids: Sequence[str]) -> Dict[str, Any]:
    config_ids = list(config_ids)
    cmap = plt.get_cmap("tab20")
    return {cid: cmap(i % cmap.N) for i, cid in enumerate(config_ids)}


def _make_config_legend_table(tuning_df: pd.DataFrame, best_param_id: str) -> pd.DataFrame:
    """
    One row per hyperparameter configuration, sorted by mean validation RMSE.
    """
    rows = []

    for param_id, g in tuning_df.groupby("param_id"):
        config_label = str(g["config_label"].iloc[0])
        trial_numbers = sorted(set(int(x) for x in g["trial_number"].tolist()))
        mean_rmse = float(g["validation_rmse"].mean())

        rows.append(
            {
                "param_id": param_id,
                "config_label": config_label,
                "trial_numbers": trial_numbers,
                "mean_validation_rmse": mean_rmse,
                "is_best": param_id == best_param_id,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("No AE hyperparameter configurations found for legend.")

    out = out.sort_values(["is_best", "mean_validation_rmse"], ascending=[False, True]).reset_index(drop=True)

    return out


def _legend_label_from_config_row(row: pd.Series) -> str:
    trials = row["trial_numbers"]

    if len(trials) == 1:
        trial_text = f"T{trials[0]}"
    else:
        trial_text = "T" + ",".join(str(x) for x in trials)

    prefix = "BEST | " if bool(row["is_best"]) else ""

    return (
        f"{prefix}{trial_text} | "
        f"{row['config_label']} | "
        f"mean={float(row['mean_validation_rmse']):.4f}"
    )


# =============================================================================
# Plotting
# =============================================================================
def plot_ae_hyperparameter_effect(
    tuning_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    study_name: str,
    show: bool = True,
    show_full_hyperparameter_legend: bool = True,
) -> Dict[str, str]:
    """
    Fixed optimization seed, changing AE hyperparameters.

    One panel per missingness type.
    x-axis = missing fraction.
    y-axis = validation RMSE.
    colored lines = AE hyperparameter configurations.
    red line = selected best configuration.
    legend = full hyperparameter values for each configuration.
    """
    df = tuning_df.copy()
    missingness_types = _ordered_missingness(df["missingness"].unique())
    best_param_id = _param_id_from_params(best_params)

    config_table = _make_config_legend_table(df, best_param_id=best_param_id)
    config_order = config_table["param_id"].tolist()
    color_map = _config_color_map(config_order)

    n_configs = len(config_order)
    legend_ncol = 1 if n_configs <= 12 else 2
    right_rect = 0.58 if n_configs > 12 else 0.68

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.2 * len(missingness_types) + 5.5, 5.2),
        sharey=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for ax, missingness in zip(axes, missingness_types):
        sub = df[df["missingness"] == missingness].copy()

        by_config = (
            sub.groupby(["param_id", "fraction"], as_index=False)["validation_rmse"]
            .mean()
            .sort_values(["param_id", "fraction"])
        )

        band = (
            by_config.groupby("fraction")["validation_rmse"]
            .agg(["min", "max", "mean"])
            .reset_index()
            .sort_values("fraction")
        )

        x_band = band["fraction"].to_numpy(dtype=float)

        ax.fill_between(
            x_band,
            band["min"].to_numpy(dtype=float),
            band["max"].to_numpy(dtype=float),
            color="gray",
            alpha=0.12,
            label="range across configs",
        )

        ax.plot(
            x_band,
            band["mean"].to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="mean across configs",
        )

        for param_id in config_order:
            g = by_config[by_config["param_id"] == param_id].sort_values("fraction")

            if g.empty:
                continue

            is_best = param_id == best_param_id

            ax.plot(
                g["fraction"],
                g["validation_rmse"],
                color="tab:red" if is_best else color_map[param_id],
                alpha=1.0 if is_best else 0.70,
                linewidth=3.2 if is_best else 1.5,
                marker="o" if is_best else ".",
                markersize=6 if is_best else 4,
            )

        ax.set_title(missingness)
        ax.set_xlabel("Missing fraction")
        ax.grid(True, alpha=0.30)

    axes[0].set_ylabel("Validation RMSE")

    legend_handles = []

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="mean across configs",
        )
    )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=8,
            alpha=0.20,
            label="range across configs",
        )
    )

    if show_full_hyperparameter_legend:
        for _, row in config_table.iterrows():
            param_id = row["param_id"]
            is_best = bool(row["is_best"])

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="tab:red" if is_best else color_map[param_id],
                    linewidth=3.2 if is_best else 1.8,
                    marker="o" if is_best else ".",
                    markersize=6 if is_best else 4,
                    label=_legend_label_from_config_row(row),
                )
            )

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(right_rect + 0.02, 0.5),
        fontsize=7,
        ncol=legend_ncol,
        title="AE hyperparameter configurations",
        title_fontsize=9,
        frameon=True,
    )

    fig.suptitle(
        f"AE hyperparameter effect under fixed optimization seed\nStudy: {study_name}",
        y=1.04,
    )

    fig.tight_layout(rect=[0, 0, right_rect, 1])

    return _save_current_figure(
        fig,
        out_dir,
        "AE_hyperparameter_effect_fixed_seed",
        show,
    )


def plot_ae_hyperparameter_search_distribution(
    tuning_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    study_name: str,
    show: bool = True,
) -> Dict[str, str]:
    df = tuning_df.copy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True)

    plot_specs = [
        ("batch_size", int(best_params["batch_size"]), "category"),
        ("dim_latent", int(best_params["dim_latent"]), "category"),
        ("n_neurons_label", "-".join(str(x) for x in best_params["n_neurons"]), "category"),
        ("learning_rate", float(best_params["learning_rate"]), "scatter_log"),
    ]

    for ax, (param, best_value, mode) in zip(axes, plot_specs):
        if mode == "category":
            values = sorted(df[param].dropna().unique(), key=lambda x: str(x))

            data = [
                df.loc[df[param] == value, "validation_rmse"].to_numpy(dtype=float)
                for value in values
            ]

            bp = ax.boxplot(
                data,
                labels=[str(v) for v in values],
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                },
            )

            for patch, value in zip(bp["boxes"], values):
                if str(value) == str(best_value):
                    patch.set_facecolor("tab:red")
                    patch.set_alpha(0.45)
                else:
                    patch.set_facecolor("tab:blue")
                    patch.set_alpha(0.25)

            ax.tick_params(axis="x", rotation=35)

        else:
            ax.scatter(
                df[param],
                df["validation_rmse"],
                alpha=0.65,
                s=28,
                color="tab:blue",
                edgecolor="black",
                linewidth=0.25,
            )
            ax.axvline(float(best_value), color="tab:red", linewidth=2.0, label="selected best")
            ax.set_xscale("log")
            ax.legend(fontsize=8)

        ax.set_title(param)
        ax.set_xlabel(param)
        ax.grid(True, axis="y", alpha=0.30)

    axes[0].set_ylabel("Validation RMSE")

    fig.suptitle(
        f"AE hyperparameter search distributions\nStudy: {study_name}",
        y=1.04,
    )

    fig.tight_layout()

    return _save_current_figure(
        fig,
        out_dir,
        "AE_hyperparameter_search_distribution",
        show,
    )


def plot_ae_seed_ribbon(
    seed_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    show: bool = True,
) -> Dict[str, str]:
    df = seed_df.copy()
    df["missingness"] = df["missingness"].astype(str).str.upper()
    df["fraction"] = pd.to_numeric(df["fraction"], errors="raise")
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["rmse"] = pd.to_numeric(df["rmse"], errors="raise")

    missingness_types = _ordered_missingness(df["missingness"].unique())
    best_param_id = _param_id_from_params(best_params)
    best_label = _pretty_ae_config_label(best_params)

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.2 * len(missingness_types), 4.8),
        sharey=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for ax, missingness in zip(axes, missingness_types):
        sub = df[df["missingness"] == missingness].copy()

        for seed, g in sub.groupby("seed"):
            g = g.sort_values("fraction")
            ax.plot(
                g["fraction"],
                g["rmse"],
                color="gray",
                alpha=0.35,
                linewidth=1.0,
                marker=".",
            )

        summary = (
            sub.groupby("fraction")["rmse"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("fraction")
        )

        x = summary["fraction"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        std = summary["std"].fillna(0.0).to_numpy(dtype=float)

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color="black",
            alpha=0.15,
            label="mean ± 1 SD",
        )

        ax.plot(
            x,
            mean,
            color="black",
            linewidth=2.5,
            marker="o",
            label="mean",
        )

        ax.set_title(missingness)
        ax.set_xlabel("Missing fraction")
        ax.grid(True, alpha=0.30)

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
        "AE seed robustness with frozen best hyperparameters\n"
        f"{best_label}",
        y=1.04,
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    return _save_current_figure(
        fig,
        out_dir,
        "AE_seed_effect_fixed_best_params_ribbon",
        show,
    )


def plot_ae_seed_boxplots(
    seed_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    show: bool = True,
) -> Dict[str, str]:
    df = seed_df.copy()
    df["missingness"] = df["missingness"].astype(str).str.upper()
    df["fraction"] = pd.to_numeric(df["fraction"], errors="raise")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="raise")

    missingness_types = _ordered_missingness(df["missingness"].unique())
    best_label = _pretty_ae_config_label(best_params)

    fig, axes = plt.subplots(
        1,
        len(missingness_types),
        figsize=(6.2 * len(missingness_types), 4.8),
        sharey=True,
    )

    if len(missingness_types) == 1:
        axes = [axes]

    for ax, missingness in zip(axes, missingness_types):
        sub = df[df["missingness"] == missingness].copy()
        fractions = sorted(sub["fraction"].unique())

        data = [
            sub.loc[np.isclose(sub["fraction"], frac), "rmse"].to_numpy(dtype=float)
            for frac in fractions
        ]

        bp = ax.boxplot(
            data,
            labels=[f"{frac:g}" for frac in fractions],
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
        )

        for patch in bp["boxes"]:
            patch.set_facecolor("tab:blue")
            patch.set_alpha(0.35)

        ax.set_title(missingness)
        ax.set_xlabel("Missing fraction")
        ax.grid(True, axis="y", alpha=0.30)

    axes[0].set_ylabel("Final RMSE")

    fig.suptitle(
        "AE seed robustness boxplots with frozen best hyperparameters\n"
        f"{best_label}",
        y=1.04,
    )

    fig.tight_layout()

    return _save_current_figure(
        fig,
        out_dir,
        "AE_seed_effect_fixed_best_params_boxplots",
        show,
    )


# =============================================================================
# Public entry point
# =============================================================================
def ae_eval(
    fractions,
    missingness_types,
    columns_num: str = "all",
    study_name: str = "AE_balanced_MAR_MNAR_10_30_seeded_v1",
    seeds: Sequence[int] = (42, 100, 2024, 777, 999),
    output_dir: Optional[str] = None,
    show: bool = True,
    strict: bool = True,
    require_condition_scores: bool = True,
    show_full_hyperparameter_legend: bool = True,
) -> Dict[str, Any]:
    fractions = [float(x) for x in fractions]
    missingness_types = [str(x).upper() for x in missingness_types]
    seeds = [int(x) for x in seeds]

    base_dir = Path(config_paths.experiments_dir(columns_num))

    if output_dir is None:
        out_dir = base_dir / "Evaluation" / "AE_seeded_pipeline"
    else:
        out_dir = Path(output_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    best_params, best_params_path, best_payload = _load_best_params(base_dir, study_name)

    print("\n" + "=" * 80)
    print("AE evaluation suite")
    print("=" * 80)
    print(f"Study name: {study_name}")
    print(f"Best params JSON: {best_params_path}")
    print(f"Best params: {best_params}")
    print(f"Fractions: {fractions}")
    print(f"Missingness types: {missingness_types}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    tuning_df = _load_ae_condition_scores(
        base_dir=base_dir,
        study_name=study_name,
        require_condition_scores=require_condition_scores,
    )

    tuning_csv = out_dir / "AE_optuna_per_condition_validation_rmse.csv"
    tuning_df.to_csv(tuning_csv, index=False)
    print(f"\nSaved Optuna per-condition values: {tuning_csv}")

    config_table = _make_config_legend_table(
        tuning_df,
        best_param_id=_param_id_from_params(best_params),
    )

    config_table_csv = out_dir / "AE_hyperparameter_config_legend.csv"
    config_table.to_csv(config_table_csv, index=False)
    print(f"Saved AE hyperparameter legend table: {config_table_csv}")

    seed_df = _load_ae_seed_rmse_values(
        base_dir=base_dir,
        columns_num=columns_num,
        fractions=fractions,
        missingness_types=missingness_types,
        seeds=seeds,
        best_params=best_params,
        strict=strict,
    )

    seed_csv = out_dir / "AE_final_rmse_by_seed.csv"
    seed_df.to_csv(seed_csv, index=False)
    print(f"Saved final seed RMSE values: {seed_csv}")

    seed_summary = (
        seed_df.groupby(["missingness", "fraction"], as_index=False)["rmse"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    seed_summary_csv = out_dir / "AE_final_rmse_by_seed_summary.csv"
    seed_summary.to_csv(seed_summary_csv, index=False)
    print(f"Saved final seed RMSE summary: {seed_summary_csv}")

    output_paths: Dict[str, Any] = {
        "best_params": best_params,
        "best_params_path": str(best_params_path),
        "tuning_csv": str(tuning_csv),
        "config_legend_csv": str(config_table_csv),
        "seed_csv": str(seed_csv),
        "seed_summary_csv": str(seed_summary_csv),
        "plots": {},
    }

    output_paths["plots"]["hyperparameter_effect"] = plot_ae_hyperparameter_effect(
        tuning_df,
        best_params=best_params,
        out_dir=out_dir,
        study_name=study_name,
        show=show,
        show_full_hyperparameter_legend=show_full_hyperparameter_legend,
    )

    output_paths["plots"]["hyperparameter_search_distribution"] = plot_ae_hyperparameter_search_distribution(
        tuning_df,
        best_params=best_params,
        out_dir=out_dir,
        study_name=study_name,
        show=show,
    )

    output_paths["plots"]["seed_ribbon"] = plot_ae_seed_ribbon(
        seed_df,
        best_params=best_params,
        out_dir=out_dir,
        show=show,
    )

    output_paths["plots"]["seed_boxplots"] = plot_ae_seed_boxplots(
        seed_df,
        best_params=best_params,
        out_dir=out_dir,
        show=show,
    )

    report_pdf = out_dir / "AE_evaluation_suite_report.pdf"

    with PdfPages(report_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        lines = [
            "AE evaluation suite",
            "",
            f"Study name: {study_name}",
            f"Best params: {best_params}",
            "",
            "Generated plots:",
            f"1. {output_paths['plots']['hyperparameter_effect']['pdf']}",
            f"2. {output_paths['plots']['hyperparameter_search_distribution']['pdf']}",
            f"3. {output_paths['plots']['seed_ribbon']['pdf']}",
            f"4. {output_paths['plots']['seed_boxplots']['pdf']}",
            "",
            "CSV outputs:",
            f"- {tuning_csv}",
            f"- {config_table_csv}",
            f"- {seed_csv}",
            f"- {seed_summary_csv}",
        ]

        y = 0.95

        for i, line in enumerate(lines):
            ax.text(
                0.05,
                y,
                line,
                fontsize=16 if i == 0 else 10,
                fontweight="bold" if i == 0 else "normal",
                ha="left",
                va="top",
                transform=ax.transAxes,
            )
            y -= 0.04 if i == 0 else 0.028

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    output_paths["combined_report_pdf"] = str(report_pdf)

    print("\n" + "=" * 80)
    print("AE evaluation complete")
    print("=" * 80)
    print(f"Combined output summary PDF: {report_pdf}")
    print("Publication-quality individual plots:")
    for key, value in output_paths["plots"].items():
        print(f"  {key}: {value['pdf']}")
    print("=" * 80)

    return output_paths


AE_eval = ae_eval