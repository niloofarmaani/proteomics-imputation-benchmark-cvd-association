# Evaluation/softimpute_eval.py

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

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config_paths


def _ordered_missingness(values: Sequence[str]) -> List[str]:
    preferred = ["MCAR", "MAR", "MNAR"]
    values = [str(v).upper() for v in values]
    present = [m for m in preferred if m in set(values)]
    extras = sorted([m for m in set(values) if m not in present])
    return present + extras


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

    required = {"rank_max", "maxit"}
    missing = required - set(params.keys())

    if missing:
        raise KeyError(
            f"Best params JSON exists but is missing required params: {sorted(missing)}\n"
            f"File: {best_params_path}\n"
            f"params: {params}"
        )

    params = {
        "rank_max": int(params["rank_max"]),
        "maxit": int(params["maxit"]),
    }

    return params, best_params_path, payload


def _load_softimpute_condition_scores(
    *,
    base_dir: Path,
    study_name: str,
    require_condition_scores: bool = True,
) -> pd.DataFrame:
    condition_csv = base_dir / "Optimization" / "trial_tables" / f"{study_name}_condition_scores.csv"

    if not condition_csv.exists():
        raise FileNotFoundError(
            f"Per-condition Optuna score table not found:\n{condition_csv}\n"
            "Rerun SoftImpute optimization with the updated run_bayes_opt.py."
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

    param_map = {
        "rank_max": ["params_rank_max", "method_params_rank_max", "rank_max"],
        "maxit": ["params_maxit", "method_params_maxit", "maxit"],
    }

    for param_name, candidates in param_map.items():
        if param_name not in df.columns:
            found = None

            for col in candidates:
                if col in df.columns:
                    found = col
                    break

            if found is not None:
                df[param_name] = df[found]

    if not all(col in df.columns for col in ["rank_max", "maxit"]):
        if "method_params_json" not in df.columns:
            raise KeyError(
                "Could not find SoftImpute parameter columns and method_params_json is missing."
            )

        parsed_params = df["method_params_json"].map(_parse_json_or_literal)

        for param_name in ["rank_max", "maxit"]:
            if param_name not in df.columns:
                df[param_name] = parsed_params.map(lambda d: d.get(param_name, np.nan))

    for col in ["rank_max", "maxit"]:
        if df[col].isna().any():
            bad = df[df[col].isna()].head()
            raise ValueError(
                f"Column {col} has missing values in condition score table.\n"
                f"Bad rows:\n{bad.to_string(index=False)}"
            )

    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="raise").astype(int)
    df["fraction"] = pd.to_numeric(df["fraction"], errors="raise")
    df["missingness"] = df["missingness"].astype(str).str.upper()
    df["validation_rmse"] = pd.to_numeric(df["validation_rmse"], errors="raise")
    df["rank_max"] = pd.to_numeric(df["rank_max"], errors="raise").astype(int)
    df["maxit"] = pd.to_numeric(df["maxit"], errors="raise").astype(int)

    df["config_label"] = df.apply(
        lambda row: f"rank={int(row['rank_max'])}, maxit={int(row['maxit'])}",
        axis=1,
    )

    return df.sort_values(["missingness", "fraction", "trial_number"]).reset_index(drop=True)


def _load_softimpute_seed_rmse_values(
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

    rank_max = int(best_params["rank_max"])
    maxit = int(best_params["maxit"])

    for missingness in missingness_types:
        missingness = str(missingness).upper()

        for fraction in fractions:
            fraction = float(fraction)
            folder = base_dir / f"fraction_{fraction}_missingness_{missingness}"

            for seed in seeds:
                seed = int(seed)

                rmse_path = (
                    folder
                    / f"RMSE_softImpute_col_{columns_num}_rank_{rank_max}_maxit_{maxit}_seed_{seed}.csv"
                )

                imputed_path = (
                    folder
                    / f"imputed_softImpute_col_{columns_num}_rank_{rank_max}_maxit_{maxit}_seed_{seed}.csv"
                )

                if not rmse_path.exists():
                    missing_files.append(str(rmse_path))
                    continue

                if strict and not imputed_path.exists():
                    missing_files.append(str(imputed_path))
                    continue

                try:
                    rmse_df = pd.read_csv(rmse_path)
                    rmse = float(pd.to_numeric(rmse_df.iloc[:, 0], errors="coerce").dropna().iloc[0])
                except Exception as e:
                    raise ValueError(
                        f"Could not read RMSE from:\n{rmse_path}\nError: {e}"
                    )

                rows.append(
                    {
                        "method": "softImpute",
                        "rank_max": rank_max,
                        "maxit": maxit,
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
            "Some required repeated-seed final SoftImpute benchmark files are missing.\n\n"
            f"First missing files:\n{preview}\n\n"
            "Run the SoftImpute repeated final benchmark for all requested seeds, "
            "fractions, and missingness types."
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No SoftImpute repeated-seed RMSE files were found.")

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


def plot_softimpute_hyperparameter_effect(
    tuning_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    study_name: str,
    show: bool = True,
) -> Dict[str, str]:
    df = tuning_df.copy()
    missingness_types = _ordered_missingness(df["missingness"].unique())

    best_label = f"rank={int(best_params['rank_max'])}, maxit={int(best_params['maxit'])}"

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

        by_config = (
            sub.groupby(["config_label", "fraction"], as_index=False)["validation_rmse"]
            .mean()
            .sort_values(["config_label", "fraction"])
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
            alpha=0.15,
            label="range across configs",
        )

        ax.plot(
            x_band,
            band["mean"].to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="mean across configs",
        )

        for config_label, g in by_config.groupby("config_label"):
            g = g.sort_values("fraction")
            is_best = config_label == best_label

            ax.plot(
                g["fraction"],
                g["validation_rmse"],
                color="tab:red" if is_best else "gray",
                alpha=1.0 if is_best else 0.25,
                linewidth=3.0 if is_best else 1.0,
                marker="o" if is_best else ".",
                label="selected best config" if is_best else None,
            )

        ax.set_title(missingness)
        ax.set_xlabel("Missing fraction")
        ax.grid(True, alpha=0.30)

    axes[0].set_ylabel("Validation RMSE")

    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.suptitle(
        f"SoftImpute hyperparameter effect under fixed optimization seed\nStudy: {study_name}",
        y=1.04,
    )

    fig.tight_layout(rect=[0, 0, 0.84, 1])

    return _save_current_figure(
        fig,
        out_dir,
        "SoftImpute_hyperparameter_effect_fixed_seed",
        show,
    )


def plot_softimpute_hyperparameter_search_distribution(
    tuning_df: pd.DataFrame,
    *,
    best_params: Dict[str, Any],
    out_dir: Path,
    study_name: str,
    show: bool = True,
) -> Dict[str, str]:
    df = tuning_df.copy()

    params = [
        ("rank_max", int(best_params["rank_max"])),
        ("maxit", int(best_params["maxit"])),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

    for ax, (param, best_value) in zip(axes, params):
        values = sorted(df[param].dropna().unique())

        data = [
            df.loc[df[param] == value, "validation_rmse"].to_numpy(dtype=float)
            for value in values
        ]

        bp = ax.boxplot(
            data,
            labels=[str(int(v)) for v in values],
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
        )

        for patch, value in zip(bp["boxes"], values):
            if int(value) == int(best_value):
                patch.set_facecolor("tab:red")
                patch.set_alpha(0.45)
            else:
                patch.set_facecolor("tab:blue")
                patch.set_alpha(0.25)

        ax.set_title(param)
        ax.set_xlabel(param)
        ax.grid(True, axis="y", alpha=0.30)

    axes[0].set_ylabel("Validation RMSE")

    fig.suptitle(
        f"SoftImpute hyperparameter search distributions\nStudy: {study_name}",
        y=1.04,
    )

    fig.tight_layout()

    return _save_current_figure(
        fig,
        out_dir,
        "SoftImpute_hyperparameter_search_distribution",
        show,
    )


def plot_softimpute_seed_ribbon(
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
        "SoftImpute seed robustness with frozen best hyperparameters\n"
        f"rank={best_params['rank_max']}, maxit={best_params['maxit']}",
        y=1.04,
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    return _save_current_figure(
        fig,
        out_dir,
        "SoftImpute_seed_effect_fixed_best_params_ribbon",
        show,
    )


def plot_softimpute_seed_boxplots(
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
        "SoftImpute seed robustness boxplots with frozen best hyperparameters\n"
        f"rank={best_params['rank_max']}, maxit={best_params['maxit']}",
        y=1.04,
    )

    fig.tight_layout()

    return _save_current_figure(
        fig,
        out_dir,
        "SoftImpute_seed_effect_fixed_best_params_boxplots",
        show,
    )


def softimpute_eval(
    fractions,
    missingness_types,
    columns_num: str = "all",
    study_name: str = "SoftImpute_balanced_MAR_MNAR_10_30_seeded_v1",
    seeds: Sequence[int] = (42, 100, 2024, 777, 999),
    output_dir: Optional[str] = None,
    show: bool = True,
    strict: bool = True,
    require_condition_scores: bool = True,
) -> Dict[str, Any]:
    fractions = [float(x) for x in fractions]
    missingness_types = [str(x).upper() for x in missingness_types]
    seeds = [int(x) for x in seeds]

    base_dir = Path(config_paths.experiments_dir(columns_num))

    if output_dir is None:
        out_dir = base_dir / "Evaluation" / "SoftImpute_seeded_pipeline"
    else:
        out_dir = Path(output_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    best_params, best_params_path, best_payload = _load_best_params(base_dir, study_name)

    print("\n" + "=" * 80)
    print("SoftImpute evaluation suite")
    print("=" * 80)
    print(f"Study name: {study_name}")
    print(f"Best params JSON: {best_params_path}")
    print(f"Best params: {best_params}")
    print(f"Fractions: {fractions}")
    print(f"Missingness types: {missingness_types}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    tuning_df = _load_softimpute_condition_scores(
        base_dir=base_dir,
        study_name=study_name,
        require_condition_scores=require_condition_scores,
    )

    tuning_csv = out_dir / "SoftImpute_optuna_per_condition_validation_rmse.csv"
    tuning_df.to_csv(tuning_csv, index=False)
    print(f"\nSaved Optuna per-condition values: {tuning_csv}")

    seed_df = _load_softimpute_seed_rmse_values(
        base_dir=base_dir,
        columns_num=columns_num,
        fractions=fractions,
        missingness_types=missingness_types,
        seeds=seeds,
        best_params=best_params,
        strict=strict,
    )

    seed_csv = out_dir / "SoftImpute_final_rmse_by_seed.csv"
    seed_df.to_csv(seed_csv, index=False)
    print(f"Saved final seed RMSE values: {seed_csv}")

    seed_summary = (
        seed_df.groupby(["missingness", "fraction"], as_index=False)["rmse"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    seed_summary_csv = out_dir / "SoftImpute_final_rmse_by_seed_summary.csv"
    seed_summary.to_csv(seed_summary_csv, index=False)
    print(f"Saved final seed RMSE summary: {seed_summary_csv}")

    output_paths: Dict[str, Any] = {
        "best_params": best_params,
        "best_params_path": str(best_params_path),
        "tuning_csv": str(tuning_csv),
        "seed_csv": str(seed_csv),
        "seed_summary_csv": str(seed_summary_csv),
        "plots": {},
    }

    output_paths["plots"]["hyperparameter_effect"] = plot_softimpute_hyperparameter_effect(
        tuning_df,
        best_params=best_params,
        out_dir=out_dir,
        study_name=study_name,
        show=show,
    )

    output_paths["plots"]["hyperparameter_search_distribution"] = plot_softimpute_hyperparameter_search_distribution(
        tuning_df,
        best_params=best_params,
        out_dir=out_dir,
        study_name=study_name,
        show=show,
    )

    output_paths["plots"]["seed_ribbon"] = plot_softimpute_seed_ribbon(
        seed_df,
        best_params=best_params,
        out_dir=out_dir,
        show=show,
    )

    output_paths["plots"]["seed_boxplots"] = plot_softimpute_seed_boxplots(
        seed_df,
        best_params=best_params,
        out_dir=out_dir,
        show=show,
    )

    report_pdf = out_dir / "SoftImpute_evaluation_suite_report.pdf"

    with PdfPages(report_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        lines = [
            "SoftImpute evaluation suite",
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
    print("SoftImpute evaluation complete")
    print("=" * 80)
    print(f"Combined output summary PDF: {report_pdf}")
    print("Publication-quality individual plots:")
    for key, value in output_paths["plots"].items():
        print(f"  {key}: {value['pdf']}")
    print("=" * 80)

    return output_paths