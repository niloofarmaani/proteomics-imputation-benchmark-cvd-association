# Evaluation/knn_eval.py

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

try:
    import optuna
except Exception as e:
    optuna = None
    _OPTUNA_IMPORT_ERROR = e


# =============================================================================
# Helpers
# =============================================================================
def _require_optuna() -> None:
    if optuna is None:
        raise ImportError(
            "Optuna is required to read the KNN optimization database. "
            f"Original import error: {_OPTUNA_IMPORT_ERROR}"
        )


def _safe_fraction_token(frac: float) -> str:
    return str(frac).replace(".", "p")


def _ordered_missingness(values: Sequence[str]) -> List[str]:
    preferred = ["MCAR", "MAR", "MNAR"]
    values = [str(v).upper() for v in values]
    present = [m for m in preferred if m in set(values)]
    extras = sorted([m for m in set(values) if m not in present])
    return present + extras


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def _parse_score_rows(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        if "score_rows" in raw:
            return raw["score_rows"]
        return [raw]

    if isinstance(raw, str):
        s = raw.strip()

        if not s or s.lower() in {"none", "nan", "null"}:
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


def _load_best_k(base_dir: Path, study_name: str) -> Tuple[int, Path, Dict[str, Any]]:
    best_params_path = base_dir / "Optimization" / "best_params" / f"{study_name}_best_params.json"
    payload = _read_json(best_params_path)

    params = payload.get("params", {})

    if "k" not in params:
        raise KeyError(
            f"Best params JSON exists but does not contain params['k'].\n"
            f"File: {best_params_path}\n"
            f"Payload keys: {list(payload.keys())}\n"
            f"params: {params}"
        )

    return int(params["k"]), best_params_path, payload


def _load_knn_optuna_condition_scores(
    *,
    base_dir: Path,
    study_name: str,
    require_condition_scores: bool = True,
) -> pd.DataFrame:
    """
    Load per-condition KNN validation RMSE values from the Optuna DB.

    Requires each completed trial to have user_attrs['score_rows'], which is created
    by the corrected Optimization/run_bayes_opt.py.
    """
    _require_optuna()

    db_path = base_dir / "Optimization" / f"{study_name}.db"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Optuna database not found:\n{db_path}\n"
            "Run optimize_knn.py first, using the same study_name."
        )

    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )

    rows = []
    complete_trials = 0
    trials_without_score_rows = []

    for trial in study.trials:
        if getattr(trial.state, "name", str(trial.state)) != "COMPLETE":
            continue

        complete_trials += 1

        method_params = trial.user_attrs.get("method_params", {})
        k = trial.params.get("k", method_params.get("k", None))

        if k is None:
            raise KeyError(
                f"Completed Optuna trial {trial.number} does not contain parameter k.\n"
                f"trial.params={trial.params}\n"
                f"user_attrs={trial.user_attrs}"
            )

        score_rows = _parse_score_rows(trial.user_attrs.get("score_rows", None))

        if len(score_rows) == 0:
            trials_without_score_rows.append(trial.number)
            continue

        for sr in score_rows:
            try:
                fraction = float(sr["fraction"])
                missingness = str(sr["missingness"]).upper()
                validation_rmse = float(sr["validation_rmse"])
                n_validation_cells = int(sr.get("n_validation_cells", np.nan))
            except Exception as e:
                raise ValueError(
                    f"Could not parse score row from trial {trial.number}:\n"
                    f"{sr}\nError: {e}"
                )

            rows.append(
                {
                    "trial_number": int(trial.number),
                    "k": int(k),
                    "trial_mean_validation_rmse": float(trial.value),
                    "fraction": fraction,
                    "missingness": missingness,
                    "validation_rmse": validation_rmse,
                    "n_validation_cells": n_validation_cells,
                }
            )

    if complete_trials == 0:
        raise RuntimeError(
            f"No COMPLETE trials found in Optuna study '{study_name}'.\n"
            f"Database: {db_path}"
        )

    if len(rows) == 0 and require_condition_scores:
        raise RuntimeError(
            "No per-condition score_rows were found in the KNN Optuna database.\n\n"
            "This means the optimization was run before run_bayes_opt.py was fixed to store:\n"
            "    trial.set_user_attr('score_rows', score_rows)\n\n"
            f"Study: {study_name}\n"
            f"Database: {db_path}\n"
            f"Completed trials: {complete_trials}\n"
            f"Trials missing score_rows: {trials_without_score_rows[:20]}\n\n"
            "Fix Optimization/run_bayes_opt.py, then rerun KNN optimization with a new study name."
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(["missingness", "fraction", "k", "trial_number"]).reset_index(drop=True)

    return df


def _load_knn_seed_rmse_values(
    *,
    base_dir: Path,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    best_k: int,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Load final repeated-seed KNN RMSE files.

    Expected filenames:
        RMSE_KNN_col_all_kn_<best_k>_seed_<seed>.csv
    """
    rows = []
    missing_files = []

    for missingness in missingness_types:
        missingness = str(missingness).upper()

        for fraction in fractions:
            fraction = float(fraction)
            folder = base_dir / f"fraction_{fraction}_missingness_{missingness}"

            for seed in seeds:
                seed = int(seed)
                rmse_path = folder / f"RMSE_KNN_col_{columns_num}_kn_{best_k}_seed_{seed}.csv"

                if not rmse_path.exists():
                    missing_files.append(str(rmse_path))
                    continue

                try:
                    rmse_df = pd.read_csv(rmse_path)
                    rmse = float(pd.to_numeric(rmse_df.iloc[:, 0], errors="coerce").dropna().iloc[0])
                except Exception as e:
                    raise ValueError(
                        f"Could not read RMSE from:\n{rmse_path}\nError: {e}"
                    )

                imputed_path = folder / f"imputed_KNN_col_{columns_num}_kn_{best_k}_seed_{seed}.csv"

                if strict and not imputed_path.exists():
                    missing_files.append(str(imputed_path))
                    continue

                rows.append(
                    {
                        "method": "KNN",
                        "k": int(best_k),
                        "missingness": missingness,
                        "fraction": fraction,
                        "seed": seed,
                        "rmse": rmse,
                        "rmse_path": str(rmse_path),
                        "imputed_path": str(imputed_path),
                    }
                )

    if missing_files and strict:
        preview = "\n".join(missing_files[:30])
        raise FileNotFoundError(
            "Some required repeated-seed final benchmark files are missing.\n"
            "The KNN evaluation will not continue with incomplete data.\n\n"
            f"First missing files:\n{preview}\n\n"
            "Run optimize_knn.py final benchmark loop for all requested seeds, fractions, and missingness types."
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No KNN repeated-seed RMSE files were found.\n"
            "Check the requested seeds and filename convention."
        )

    df = df.sort_values(["missingness", "fraction", "seed"]).reset_index(drop=True)

    return df


def _line_color_map(values: Sequence[int]) -> Dict[int, str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    if not colors:
        colors = [f"C{i}" for i in range(10)]

    return {int(v): colors[i % len(colors)] for i, v in enumerate(sorted(set(values)))}


def _save_current_figure(
    fig,
    out_dir: Path,
    stem: str,
    show: bool,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return {
        "png": str(png),
        "pdf": str(pdf),
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_knn_hyperparameter_effect(
    tuning_df: pd.DataFrame,
    *,
    best_k: int,
    out_dir: Path,
    study_name: str,
    show: bool = True,
) -> Dict[str, str]:
    """
    Plot how K changes validation RMSE under the fixed optimization seed.

    One panel per missingness type.
    x-axis = missing fraction.
    y-axis = validation RMSE.
    lines = k values.
    shaded band = min to max over k at each fraction.
    """
    required = {"k", "fraction", "missingness", "validation_rmse"}

    missing = required - set(tuning_df.columns)

    if missing:
        raise ValueError(f"tuning_df is missing required columns: {sorted(missing)}")

    df = tuning_df.copy()
    df["k"] = pd.to_numeric(df["k"], errors="raise").astype(int)
    df["fraction"] = pd.to_numeric(df["fraction"], errors="raise")
    df["validation_rmse"] = pd.to_numeric(df["validation_rmse"], errors="raise")
    df["missingness"] = df["missingness"].astype(str).str.upper()

    missingness_types = _ordered_missingness(df["missingness"].unique())
    k_values = sorted(df["k"].unique())
    colors = _line_color_map(k_values)

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

        # Average repeated trials with the same k and fraction, if Optuna sampled the same k more than once.
        mean_by_k = (
            sub.groupby(["k", "fraction"], as_index=False)["validation_rmse"]
            .mean()
            .sort_values(["k", "fraction"])
        )

        band = (
            mean_by_k.groupby("fraction")["validation_rmse"]
            .agg(["min", "max", "mean"])
            .reset_index()
            .sort_values("fraction")
        )

        x_band = band["fraction"].to_numpy(dtype=float)
        ax.fill_between(
            x_band,
            band["min"].to_numpy(dtype=float),
            band["max"].to_numpy(dtype=float),
            alpha=0.15,
            color="gray",
            label="range across k",
        )

        ax.plot(
            x_band,
            band["mean"].to_numpy(dtype=float),
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="mean across k",
        )

        for k in k_values:
            g = mean_by_k[mean_by_k["k"] == k].sort_values("fraction")

            if g.empty:
                continue

            linewidth = 3.0 if int(k) == int(best_k) else 1.4
            alpha = 1.0 if int(k) == int(best_k) else 0.65
            marker = "o" if int(k) == int(best_k) else "."

            ax.plot(
                g["fraction"],
                g["validation_rmse"],
                marker=marker,
                linewidth=linewidth,
                alpha=alpha,
                color=colors[int(k)],
                label=f"k={k}" + (" best" if int(k) == int(best_k) else ""),
            )

        ax.set_title(f"{missingness}")
        ax.set_xlabel("Missing fraction")
        ax.grid(True, alpha=0.30)

    axes[0].set_ylabel("Validation RMSE")

    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="KNN hyperparameter",
    )

    fig.suptitle(
        f"KNN hyperparameter effect under fixed optimization seed\nStudy: {study_name}",
        y=1.04,
    )

    fig.tight_layout(rect=[0, 0, 0.84, 1])

    return _save_current_figure(
        fig,
        out_dir,
        "KNN_hyperparameter_effect_fixed_seed",
        show,
    )


def plot_knn_hyperparameter_search_space(
    tuning_df: pd.DataFrame,
    *,
    best_k: int,
    out_dir: Path,
    study_name: str,
    show: bool = True,
) -> Dict[str, str]:
    """
    Overall validation objective behavior by k.

    This plot ignores fraction/missingness and shows the per-condition validation RMSE distribution
    for every k value. It is useful as a compact diagnostic.
    """
    df = tuning_df.copy()
    df["k"] = pd.to_numeric(df["k"], errors="raise").astype(int)
    df["validation_rmse"] = pd.to_numeric(df["validation_rmse"], errors="raise")

    k_values = sorted(df["k"].unique())
    data = [
        df.loc[df["k"] == k, "validation_rmse"].to_numpy(dtype=float)
        for k in k_values
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))

    bp = ax.boxplot(
        data,
        labels=[str(k) for k in k_values],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
    )

    for patch, k in zip(bp["boxes"], k_values):
        if int(k) == int(best_k):
            patch.set_facecolor("tab:red")
            patch.set_alpha(0.45)
        else:
            patch.set_facecolor("tab:blue")
            patch.set_alpha(0.25)

    ax.set_xlabel("KNN number of neighbors, k")
    ax.set_ylabel("Validation RMSE")
    ax.set_title(f"KNN hyperparameter search distribution\nStudy: {study_name}")
    ax.grid(True, axis="y", alpha=0.30)

    return _save_current_figure(
        fig,
        out_dir,
        "KNN_hyperparameter_search_distribution",
        show,
    )


def plot_knn_seed_ribbon(
    seed_df: pd.DataFrame,
    *,
    best_k: int,
    out_dir: Path,
    show: bool = True,
) -> Dict[str, str]:
    """
    Plot how changing preprocessing/random seed affects final RMSE
    while k is fixed to the Optuna-selected best value.

    One panel per missingness type.
    x-axis = missing fraction.
    y-axis = final test RMSE.
    gray lines = individual seeds.
    black line and ribbon = mean ± 1 standard deviation.
    """
    required = {"missingness", "fraction", "seed", "rmse"}

    missing = required - set(seed_df.columns)

    if missing:
        raise ValueError(f"seed_df is missing required columns: {sorted(missing)}")

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

        ax.set_title(f"{missingness}")
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
        f"KNN seed robustness with frozen best hyperparameter k={best_k}",
        y=1.04,
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    return _save_current_figure(
        fig,
        out_dir,
        "KNN_seed_effect_fixed_best_k_ribbon",
        show,
    )


def plot_knn_seed_boxplots(
    seed_df: pd.DataFrame,
    *,
    best_k: int,
    out_dir: Path,
    show: bool = True,
) -> Dict[str, str]:
    """
    Boxplot version of seed robustness.

    One panel per missingness type.
    x-axis = missing fraction.
    box values = repeated seed RMSE values.
    """
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
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )

        for patch in bp["boxes"]:
            patch.set_facecolor("tab:blue")
            patch.set_alpha(0.35)

        ax.set_title(f"{missingness}")
        ax.set_xlabel("Missing fraction")
        ax.grid(True, axis="y", alpha=0.30)

    axes[0].set_ylabel("Final RMSE")

    fig.suptitle(
        f"KNN seed robustness boxplots with frozen best hyperparameter k={best_k}",
        y=1.04,
    )

    fig.tight_layout()

    return _save_current_figure(
        fig,
        out_dir,
        "KNN_seed_effect_fixed_best_k_boxplots",
        show,
    )


# =============================================================================
# Public entry point
# =============================================================================
def knn_eval(
    fractions,
    missingness_types,
    columns_num: str = "all",
    study_name: str = "KNN_balanced_MAR_MNAR_10_30_seeded_v1",
    seeds: Sequence[int] = (42, 100, 2024, 777, 999),
    output_dir: Optional[str] = None,
    show: bool = True,
    strict: bool = True,
    require_condition_scores: bool = True,
) -> Dict[str, Any]:
    """
    KNN evaluation suite for the corrected seeded pipeline.

    Produces two required plot groups:

    1. Hyperparameter effect at fixed optimization seed:
       Uses Optuna per-condition validation RMSE stored in trial.user_attrs['score_rows'].
       Shows how k changes validation RMSE across MAR/MNAR tuning fractions.

    2. Seed robustness at fixed best hyperparameter:
       Uses final benchmark RMSE files generated with the frozen best k across multiple preprocessing seeds.
       Shows how RMSE changes with seed across fractions and missingness types.

    This function is intentionally strict.
    It does not silently skip missing inputs when strict=True.
    """
    fractions = [float(x) for x in fractions]
    missingness_types = [str(x).upper() for x in missingness_types]
    seeds = [int(x) for x in seeds]

    base_dir = Path(config_paths.experiments_dir(columns_num))

    if output_dir is None:
        out_dir = base_dir / "Evaluation" / "KNN_seeded_pipeline"
    else:
        out_dir = Path(output_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    best_k, best_params_path, best_payload = _load_best_k(base_dir, study_name)

    print("\n" + "=" * 80)
    print("KNN evaluation suite")
    print("=" * 80)
    print(f"Study name: {study_name}")
    print(f"Best params JSON: {best_params_path}")
    print(f"Best k: {best_k}")
    print(f"Fractions: {fractions}")
    print(f"Missingness types: {missingness_types}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    tuning_df = _load_knn_optuna_condition_scores(
        base_dir=base_dir,
        study_name=study_name,
        require_condition_scores=require_condition_scores,
    )

    tuning_csv = out_dir / "KNN_optuna_per_condition_validation_rmse.csv"
    tuning_df.to_csv(tuning_csv, index=False)
    print(f"\nSaved Optuna per-condition values: {tuning_csv}")

    seed_df = _load_knn_seed_rmse_values(
        base_dir=base_dir,
        columns_num=columns_num,
        fractions=fractions,
        missingness_types=missingness_types,
        seeds=seeds,
        best_k=best_k,
        strict=strict,
    )

    seed_csv = out_dir / "KNN_final_rmse_by_seed.csv"
    seed_df.to_csv(seed_csv, index=False)
    print(f"Saved final seed RMSE values: {seed_csv}")

    seed_summary = (
        seed_df.groupby(["missingness", "fraction"], as_index=False)["rmse"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    seed_summary_csv = out_dir / "KNN_final_rmse_by_seed_summary.csv"
    seed_summary.to_csv(seed_summary_csv, index=False)
    print(f"Saved final seed RMSE summary: {seed_summary_csv}")

    output_paths: Dict[str, Any] = {
        "best_k": best_k,
        "best_params_path": str(best_params_path),
        "tuning_csv": str(tuning_csv),
        "seed_csv": str(seed_csv),
        "seed_summary_csv": str(seed_summary_csv),
        "plots": {},
    }

    output_paths["plots"]["hyperparameter_effect"] = plot_knn_hyperparameter_effect(
        tuning_df,
        best_k=best_k,
        out_dir=out_dir,
        study_name=study_name,
        show=show,
    )

    output_paths["plots"]["hyperparameter_search_distribution"] = plot_knn_hyperparameter_search_space(
        tuning_df,
        best_k=best_k,
        out_dir=out_dir,
        study_name=study_name,
        show=show,
    )

    output_paths["plots"]["seed_ribbon"] = plot_knn_seed_ribbon(
        seed_df,
        best_k=best_k,
        out_dir=out_dir,
        show=show,
    )

    output_paths["plots"]["seed_boxplots"] = plot_knn_seed_boxplots(
        seed_df,
        best_k=best_k,
        out_dir=out_dir,
        show=show,
    )

    combined_pdf = out_dir / "KNN_evaluation_suite_report.pdf"

    with PdfPages(combined_pdf) as pdf:
        for key in [
            "hyperparameter_effect",
            "hyperparameter_search_distribution",
            "seed_ribbon",
            "seed_boxplots",
        ]:
            pdf_path = Path(output_paths["plots"][key]["pdf"])

            if not pdf_path.exists():
                raise FileNotFoundError(f"Expected plot PDF missing: {pdf_path}")

            # Re-rendering PDFs inside PdfPages directly is messy without extra dependencies.
            # Instead, create a clear report page listing individual output PDFs.
            # The individual PDFs are the publication-quality outputs.
            pass

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        lines = [
            "KNN evaluation suite",
            "",
            f"Study name: {study_name}",
            f"Best k: {best_k}",
            "",
            "Generated outputs:",
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

    output_paths["combined_report_pdf"] = str(combined_pdf)

    print("\n" + "=" * 80)
    print("KNN evaluation complete")
    print("=" * 80)
    print(f"Combined output summary PDF: {combined_pdf}")
    print("Publication-quality individual plots:")
    for key, value in output_paths["plots"].items():
        print(f"  {key}: {value['pdf']}")
    print("=" * 80)

    return output_paths