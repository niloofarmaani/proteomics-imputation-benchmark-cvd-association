# Evaluation/paper_seed_rmse_boxplots.py
#
# Manuscript-ready RMSE boxplots and Bayesian-optimization tables
# for the seeded MyoVasc imputation benchmark.
#
# This module reads only plot-ready result files:
#   - method-specific *_final_rmse_by_seed.csv
#   - Optuna condition score tables
#   - best-parameter JSON files
#
# It does not read raw data, phenotype files, corrupted matrices, masks,
# or imputed matrices.

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config_paths


# =============================================================================
# Method metadata
# =============================================================================
METHOD_CONFIGS = {
    "KNN": {
        "label": "KNN",
        "eval_dir": "KNN_seeded_pipeline",
        "csv_prefix": "KNN",
        "aliases": ["KNN", "knn"],
    },
    "SoftImpute": {
        "label": "SoftImpute",
        "eval_dir": "SoftImpute_seeded_pipeline",
        "csv_prefix": "SoftImpute",
        "aliases": ["softImpute", "SoftImpute", "SOFTIMPUTE", "softimpute"],
    },
    "VAE": {
        "label": "VAE",
        "eval_dir": "VAE_seeded_pipeline",
        "csv_prefix": "VAE",
        "aliases": ["VAE", "vae"],
    },
    "DAE": {
        "label": "DAE",
        "eval_dir": "DAE_seeded_pipeline",
        "csv_prefix": "DAE",
        "aliases": ["DAE", "dae"],
    },
    "AE": {
        "label": "AE",
        "eval_dir": "AE_seeded_pipeline",
        "csv_prefix": "AE",
        "aliases": ["AE", "ae"],
    },
    "GAIN": {
        "label": "GAIN",
        "eval_dir": "GAIN_seeded_pipeline",
        "csv_prefix": "GAIN",
        "aliases": ["GAIN", "gain"],
    },
    "Transformer": {
        "label": "Transformer",
        "eval_dir": "Transformer_seeded_pipeline",
        "csv_prefix": "Transformer",
        "aliases": ["Transformer", "TRANSFORMER", "TRF", "trf"],
    },
    "RNN": {
        "label": "RNN",
        "eval_dir": "RNN_seeded_pipeline",
        "csv_prefix": "RNN",
        "aliases": ["RNN", "rnn"],
    },
    "MissForest": {
        "label": "MissForest",
        "eval_dir": "MissForest_seeded_pipeline",
        "csv_prefix": "MissForest",
        "aliases": ["mf", "MF", "MissForest", "MISSFOREST", "missforest"],
    },
    "Iterative": {
        "label": "Iterative",
        "eval_dir": "Iterative_seeded_pipeline",
        "csv_prefix": "Iterative",
        "aliases": ["iterative", "Iterative", "ITERATIVE", "IterativeImputer"],
    },
    "Mean": {
        "label": "Mean",
        "eval_dir": "Mean_seeded_pipeline",
        "csv_prefix": "Mean",
        "aliases": ["Mean", "mean", "MEAN"],
    },
}


def canonical_method_name(method: str) -> str:
    method = str(method).strip()
    for canonical, cfg in METHOD_CONFIGS.items():
        if method in cfg["aliases"]:
            return canonical
    known = []
    for cfg in METHOD_CONFIGS.values():
        known.extend(cfg["aliases"])
    raise KeyError(f"Unknown method '{method}'. Known aliases include: {sorted(set(known))}")


def label_for_method(method: str) -> str:
    return METHOD_CONFIGS[canonical_method_name(method)]["label"]


def ordered_missingness(values: Sequence[str]) -> List[str]:
    preferred = ["MCAR", "MAR", "MNAR"]
    values = [str(v).upper() for v in values]
    present = [m for m in preferred if m in set(values)]
    extras = sorted([m for m in set(values) if m not in present])
    return present + extras


def safe_fraction_token(frac: float) -> str:
    return str(float(frac)).replace(".", "p")


def format_fraction_percent(frac: float) -> str:
    pct = float(frac) * 100.0
    if pct.is_integer():
        return f"{int(pct)}%"
    return f"{pct:g}%"


def mean_sd_string(mean: float, sd: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def compact_json(obj: Any, max_len: int = 260) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def safe_stem(text: str) -> str:
    out = str(text)
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        ("%", "pct"),
        (",", "_"),
        (";", "_"),
        (":", "_"),
        ("(", "_"),
        (")", "_"),
        ("[", "_"),
        ("]", "_"),
    ]:
        out = out.replace(old, new)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# =============================================================================
# Seed RMSE loading
# =============================================================================
def _read_seed_csv(path: Path, method_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required repeated-seed RMSE CSV does not exist for {method_label}:\n{path}\n\n"
            "Run the method-specific seeded evaluator first."
        )

    df = pd.read_csv(path)
    required = {"missingness", "fraction", "seed", "rmse"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Seed RMSE CSV for {method_label} is missing columns: {sorted(missing)}\n"
            f"File: {path}\nColumns: {list(df.columns)}"
        )

    df = df.copy()
    df["missingness"] = df["missingness"].astype(str).str.upper()
    df["fraction"] = pd.to_numeric(df["fraction"], errors="raise")
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["rmse"] = pd.to_numeric(df["rmse"], errors="raise")
    if not np.isfinite(df["rmse"].to_numpy(dtype=float)).all():
        bad = df.loc[~np.isfinite(df["rmse"].to_numpy(dtype=float))]
        raise ValueError(f"Non-finite RMSE values in {path}:\n{bad.head(20).to_string(index=False)}")

    df["method"] = method_label
    df["source_csv"] = str(path)
    return df


def load_seed_rmse_values(
    *,
    columns_num: str,
    methods_to_compare: Sequence[str],
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    seed_values_csv: Optional[Union[str, Path]] = None,
    csv_overrides: Optional[Dict[str, Union[str, Path]]] = None,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Load repeated-seed RMSE values.

    Preferred source is method-specific seeded evaluator CSVs.
    You can also pass a combined seed_values_csv if you already have one.
    """
    main_dir = Path(config_paths.experiments_dir(columns_num))
    fractions = [float(f) for f in fractions]
    missingness_types = [str(m).upper() for m in missingness_types]
    seeds = [int(s) for s in seeds]
    method_labels = [label_for_method(m) for m in methods_to_compare]

    if seed_values_csv is not None:
        path = Path(seed_values_csv).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Combined seed values CSV not found:\n{path}")
        df = pd.read_csv(path)
        required = {"method", "missingness", "fraction", "seed", "rmse"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Combined seed CSV missing columns {sorted(missing)}:\n{path}")
        df = df.copy()
        df["method"] = df["method"].astype(str).map(label_for_method)
        df["missingness"] = df["missingness"].astype(str).str.upper()
        df["fraction"] = pd.to_numeric(df["fraction"], errors="raise")
        df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
        df["rmse"] = pd.to_numeric(df["rmse"], errors="raise")
        df = df[
            df["method"].isin(method_labels)
            & df["missingness"].isin(missingness_types)
            & df["fraction"].isin(fractions)
            & df["seed"].isin(seeds)
        ].copy()
    else:
        override_map: Dict[str, Path] = {}
        if csv_overrides:
            for method, p in csv_overrides.items():
                override_map[canonical_method_name(method)] = Path(p).expanduser().resolve()

        frames = []
        for method in methods_to_compare:
            canonical = canonical_method_name(method)
            cfg = METHOD_CONFIGS[canonical]
            label = cfg["label"]

            if canonical in override_map:
                csv_path = override_map[canonical]
            else:
                csv_path = main_dir / "Evaluation" / cfg["eval_dir"] / f"{cfg['csv_prefix']}_final_rmse_by_seed.csv"

            frame = _read_seed_csv(csv_path, method_label=label)
            frame = frame[
                frame["fraction"].isin(fractions)
                & frame["missingness"].isin(missingness_types)
                & frame["seed"].isin(seeds)
            ].copy()
            frames.append(frame)

        if not frames:
            raise ValueError("No repeated-seed RMSE values were loaded.")
        df = pd.concat(frames, ignore_index=True)

    expected = {
        (label, m, float(f), int(s))
        for label in method_labels
        for m in missingness_types
        for f in fractions
        for s in seeds
    }
    observed = {
        (str(r["method"]), str(r["missingness"]).upper(), float(r["fraction"]), int(r["seed"]))
        for _, r in df.iterrows()
    }
    missing_rows = sorted(expected - observed)
    if missing_rows and strict:
        preview = "\n".join(str(x) for x in missing_rows[:40])
        raise ValueError(
            "Missing repeated-seed RMSE rows.\n"
            "First missing rows as (method, missingness, fraction, seed):\n"
            f"{preview}"
        )

    duplicated = df.duplicated(subset=["method", "missingness", "fraction", "seed"], keep=False)
    if duplicated.any():
        bad = df.loc[duplicated, ["method", "missingness", "fraction", "seed", "rmse"]]
        raise ValueError(f"Duplicate repeated-seed RMSE rows:\n{bad.head(40).to_string(index=False)}")

    df["method"] = pd.Categorical(df["method"], categories=method_labels, ordered=True)
    df = df.sort_values(["missingness", "fraction", "method", "seed"]).reset_index(drop=True)
    df["method"] = df["method"].astype(str)

    return df


def summarize_seed_rmse(seed_df: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_df
        .groupby(["missingness", "fraction", "method"])["rmse"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )


# =============================================================================
# Bayesian optimization metadata and tables
# =============================================================================
def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found:\n{path}")
    with path.open("r") as f:
        return json.load(f)


def _study_name_map(study_names: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for method, study_name in study_names.items():
        out[canonical_method_name(method)] = str(study_name)
    return out


def _condition_scores_path(main_dir: Path, study_name: str) -> Path:
    return main_dir / "Optimization" / "trial_tables" / f"{study_name}_condition_scores.csv"


def _best_params_path(main_dir: Path, study_name: str) -> Path:
    return main_dir / "Optimization" / "best_params" / f"{study_name}_best_params.json"


def _load_condition_scores(main_dir: Path, study_name: str) -> pd.DataFrame:
    path = _condition_scores_path(main_dir, study_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Condition score table not found:\n{path}\n"
            "Run the seeded Bayesian optimization first."
        )
    df = pd.read_csv(path)
    required = {"trial_number", "state", "fraction", "missingness", "validation_rmse"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Condition score table missing {sorted(missing)}:\n{path}")
    df = df[df["state"].astype(str).str.upper() == "COMPLETE"].copy()
    if df.empty:
        raise ValueError(f"No COMPLETE condition-score rows found:\n{path}")
    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="raise").astype(int)
    df["fraction"] = pd.to_numeric(df["fraction"], errors="raise")
    df["missingness"] = df["missingness"].astype(str).str.upper()
    df["validation_rmse"] = pd.to_numeric(df["validation_rmse"], errors="raise")
    return df


def _try_parse_dict(raw: Any) -> Dict[str, Any]:
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
    if not s or s.lower() in {"nan", "none", "null"}:
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


def _collect_param_values(condition_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Collect parameter values from method_params_json and params_* columns.
    This is strict enough for table generation but independent of method type.
    """
    values: Dict[str, set] = {}

    if "method_params_json" in condition_df.columns:
        for raw in condition_df["method_params_json"]:
            d = _try_parse_dict(raw)
            for key, val in d.items():
                values.setdefault(str(key), set()).add(str(val))

    for col in condition_df.columns:
        if col.startswith("params_"):
            key = col.replace("params_", "", 1)
            vals = condition_df[col].dropna().tolist()
            if vals:
                for val in vals:
                    values.setdefault(str(key), set()).add(str(val))
        elif col.startswith("method_params_"):
            key = col.replace("method_params_", "", 1)
            vals = condition_df[col].dropna().tolist()
            if vals:
                for val in vals:
                    values.setdefault(str(key), set()).add(str(val))

    return {k: sorted(v) for k, v in values.items()}


def _format_value_set(vals: List[str]) -> str:
    if len(vals) == 0:
        return ""
    numeric = pd.to_numeric(pd.Series(vals), errors="coerce")
    if numeric.notna().all() and len(vals) > 6:
        return f"{float(numeric.min()):.3g}–{float(numeric.max()):.3g}"
    if len(vals) <= 8:
        return "{" + ", ".join(vals) + "}"
    return f"{len(vals)} values"


def _search_space_summary(condition_df: pd.DataFrame) -> Tuple[str, str]:
    values = _collect_param_values(condition_df)
    tuned_parts = []
    fixed_parts = []

    for key in sorted(values.keys()):
        vals = values[key]
        formatted = _format_value_set(vals)
        if len(vals) > 1:
            tuned_parts.append(f"{key}: {formatted}")
        else:
            fixed_parts.append(f"{key}: {formatted}")

    return "; ".join(tuned_parts), "; ".join(fixed_parts)


def _best_validation_by_condition(best_payload: Dict[str, Any], missingness: str, fraction: float) -> Optional[float]:
    rows = best_payload.get("score_rows", [])
    for row in rows:
        try:
            if str(row.get("missingness")).upper() == str(missingness).upper() and np.isclose(float(row.get("fraction")), float(fraction)):
                return float(row.get("validation_rmse"))
        except Exception:
            continue
    return None


def _final_rmse_cell(summary_df: pd.DataFrame, method: str, missingness: str, fraction: float) -> str:
    sub = summary_df[
        (summary_df["method"].astype(str) == str(method))
        & (summary_df["missingness"].astype(str).str.upper() == str(missingness).upper())
        & (np.isclose(summary_df["fraction"].astype(float), float(fraction)))
    ].copy()
    if sub.empty:
        return ""
    row = sub.iloc[0]
    return mean_sd_string(float(row["mean"]), float(row["std"]), digits=3)


def make_bayesian_optimization_tables(
    *,
    columns_num: str,
    methods_to_compare: Sequence[str],
    study_names: Dict[str, str],
    seed_summary: pd.DataFrame,
    optimization_seed: int = 42,
    final_seeds: Sequence[int] = (42, 100, 2024, 777, 999),
    tuning_missingness_types: Sequence[str] = ("MAR", "MNAR"),
    tuning_fractions: Sequence[float] = (0.1, 0.3),
    main_missingness: str = "MAR",
    main_fractions: Sequence[float] = (0.1, 0.3, 0.5),
    supplementary_missingness_types: Sequence[str] = ("MCAR", "MNAR"),
    supplementary_fractions: Sequence[float] = (0.1, 0.3, 0.5),
    output_dir: Optional[Union[str, Path]] = None,
    print_latex: bool = True,
    print_markdown: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Create BMC/Springer-ready tables describing:
      1. Bayesian optimization design.
      2. Main-text MAR final RMSE with selected hyperparameters.
      3. Supplementary MCAR/MNAR final RMSE.
      4. Supplementary search-space and fixed-setting details.
    """
    main_dir = Path(config_paths.experiments_dir(columns_num))
    study_map = _study_name_map(study_names)

    if output_dir is None:
        output_dir = main_dir / "paper_figures" / "seed_rmse_boxplots_final" / "tables"
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    design_rows = [
        ("Optimizer", "Optuna Tree-structured Parzen Estimator sampler"),
        ("Optimization seed", str(int(optimization_seed))),
        ("Tuning missingness mechanisms", ", ".join(str(x).upper() for x in tuning_missingness_types)),
        ("Tuning missingness fractions", ", ".join(format_fraction_percent(float(x)) for x in tuning_fractions)),
        ("Validation objective", "Mean validation RMSE across the tuning panel"),
        ("Validation-mask construction", "Observed protein cells were hidden within the optimization split; sample identifiers and labels were excluded from imputation and scoring"),
        ("Final benchmark strategy", "Selected hyperparameters were frozen before final evaluation"),
        ("Final benchmark mechanisms", "MCAR, MAR, and MNAR"),
        ("Final benchmark fractions", ", ".join(format_fraction_percent(float(x)) for x in main_fractions)),
        ("Final seeds", ", ".join(str(int(s)) for s in final_seeds)),
        ("Leakage control", "Final RMSE, CVD prediction metrics, and association-preservation metrics were not used for hyperparameter selection"),
    ]
    design_table = pd.DataFrame(design_rows, columns=["Item", "Specification"])

    main_rows = []
    search_rows = []
    per_condition_rows = []

    for method in methods_to_compare:
        canonical = canonical_method_name(method)
        label = METHOD_CONFIGS[canonical]["label"]
        if canonical not in study_map:
            raise KeyError(f"No study name supplied for method '{method}' / canonical '{canonical}'.")

        study_name = study_map[canonical]
        best_path = _best_params_path(main_dir, study_name)
        cond_path = _condition_scores_path(main_dir, study_name)

        best_payload = _read_json(best_path)
        params = best_payload.get("params", {})
        condition_df = _load_condition_scores(main_dir, study_name)
        tuned_summary, fixed_summary = _search_space_summary(condition_df)

        selected_params = compact_json(params, max_len=360)
        best_mean_val = best_payload.get("best_value_validation_rmse", np.nan)
        n_trials = int(condition_df["trial_number"].nunique())

        search_rows.append(
            {
                "Method": label,
                "Study name": study_name,
                "Completed trials": n_trials,
                "Tuned hyperparameters": tuned_summary,
                "Fixed settings recorded in trial table": fixed_summary,
                "Selected hyperparameters": selected_params,
                "Best mean validation RMSE": best_mean_val,
                "Best trial": best_payload.get("best_trial_number", ""),
                "Condition score table": str(cond_path),
                "Best-parameter JSON": str(best_path),
            }
        )

        row = {
            "Method": label,
            "Completed trials": n_trials,
            "Selected hyperparameters": selected_params,
            "Best mean validation RMSE": best_mean_val,
            f"{main_missingness} 10% validation RMSE": _best_validation_by_condition(best_payload, main_missingness, 0.1),
            f"{main_missingness} 30% validation RMSE": _best_validation_by_condition(best_payload, main_missingness, 0.3),
        }
        for frac in main_fractions:
            row[f"{main_missingness} {format_fraction_percent(float(frac))} final RMSE, mean ± SD"] = _final_rmse_cell(
                seed_summary, label, main_missingness, float(frac)
            )
        main_rows.append(row)

        for missingness in ordered_missingness(["MCAR", "MAR", "MNAR"]):
            for frac in main_fractions:
                per_condition_rows.append(
                    {
                        "Method": label,
                        "Missingness": missingness,
                        "Fraction": format_fraction_percent(float(frac)),
                        "Final RMSE, mean ± SD": _final_rmse_cell(seed_summary, label, missingness, float(frac)),
                    }
                )

    main_mar_table = pd.DataFrame(main_rows)
    search_space_table = pd.DataFrame(search_rows)
    all_condition_long_table = pd.DataFrame(per_condition_rows)

    supp_rows = []
    for method in methods_to_compare:
        label = label_for_method(method)
        row = {"Method": label}
        for missingness in ordered_missingness(supplementary_missingness_types):
            for frac in supplementary_fractions:
                row[f"{missingness} {format_fraction_percent(float(frac))} final RMSE, mean ± SD"] = _final_rmse_cell(
                    seed_summary, label, missingness, float(frac)
                )
        supp_rows.append(row)
    supplementary_rmse_table = pd.DataFrame(supp_rows)

    supplementary_materials_index = pd.DataFrame(
        [
            {
                "Item": "Main Figure",
                "File": "Figure_RMSE_MAR_seed_boxplots.pdf",
                "Content": "Final RMSE across repeated seeds for MAR at 10%, 30%, and 50% missingness using frozen Bayesian-selected hyperparameters.",
            },
            {
                "Item": "Supplementary Figure",
                "File": "Supplementary_RMSE_MCAR_MNAR_seed_boxplots.pdf",
                "Content": "Final RMSE across repeated seeds for MCAR and MNAR at 10%, 30%, and 50% missingness.",
            },
            {
                "Item": "Supplementary Figure",
                "File": "Supplementary_RMSE_all_conditions_seed_boxplots.pdf",
                "Content": "Final RMSE across repeated seeds for all MCAR, MAR, and MNAR benchmark settings.",
            },
            {
                "Item": "Supplementary Table",
                "File": "supplementary_non_mar_rmse_table.tex",
                "Content": "MCAR and MNAR final RMSE mean ± SD by method and missingness fraction.",
            },
            {
                "Item": "Supplementary Table",
                "File": "supplementary_hyperparameter_search_space_table.tex",
                "Content": "Completed trials, tuned hyperparameter ranges, fixed settings, and selected hyperparameters for each method.",
            },
            {
                "Item": "Source data",
                "File": "paper_seed_rmse_values.csv",
                "Content": "Long-format RMSE values by method, missingness type, missingness fraction, and seed.",
            },
            {
                "Item": "Source data",
                "File": "paper_seed_rmse_summary.csv",
                "Content": "Mean, standard deviation, median, minimum, and maximum RMSE across seeds.",
            },
        ]
    )

    tables = {
        "bayesian_optimization_design": design_table,
        "main_mar_bayesian_and_rmse_table": main_mar_table,
        "supplementary_non_mar_rmse_table": supplementary_rmse_table,
        "supplementary_hyperparameter_search_space_table": search_space_table,
        "supplementary_all_conditions_long_rmse_table": all_condition_long_table,
        "supplementary_materials_index": supplementary_materials_index,
    }

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        table.to_latex(output_dir / f"{name}.tex", index=False, escape=True)

    if print_markdown:
        for title, table in [
            ("Table 1. Bayesian optimization design", design_table),
            (f"Main table. Bayesian optimization and final RMSE for {main_missingness}", main_mar_table),
            ("Supplementary table. Final RMSE for non-MAR missingness mechanisms", supplementary_rmse_table),
            ("Supplementary table. Hyperparameter search spaces and selected settings", search_space_table[
                ["Method", "Completed trials", "Tuned hyperparameters", "Fixed settings recorded in trial table", "Selected hyperparameters", "Best mean validation RMSE"]
            ]),
            ("Supplementary materials index", supplementary_materials_index),
        ]:
            print("\n" + "=" * 120)
            print(title)
            print("=" * 120)
            print(table.to_markdown(index=False))

    if print_latex:
        for title, table in [
            ("LaTeX: Bayesian optimization design", design_table),
            (f"LaTeX: Main MAR Bayesian/RMSE table", main_mar_table),
            ("LaTeX: Supplementary non-MAR RMSE table", supplementary_rmse_table),
            ("LaTeX: Supplementary materials index", supplementary_materials_index),
        ]:
            print("\n" + "=" * 120)
            print(title)
            print("=" * 120)
            print(table.to_latex(index=False, escape=True))

    return tables


# =============================================================================
# Plotting
# =============================================================================
def _method_order_for_condition(df: pd.DataFrame, methods_to_compare: Sequence[str], sort_by: str) -> List[str]:
    input_order = [label_for_method(m) for m in methods_to_compare]
    if sort_by == "input":
        return [m for m in input_order if m in set(df["method"])]

    grouped = df.groupby("method")["rmse"]
    if sort_by == "mean":
        return grouped.mean().sort_values().index.tolist()
    if sort_by == "median":
        return grouped.median().sort_values().index.tolist()
    if sort_by == "min":
        return grouped.min().sort_values().index.tolist()
    raise ValueError("sort_by must be one of: input, mean, median, min.")


def _global_order(df: pd.DataFrame, methods_to_compare: Sequence[str], sort_by: str) -> List[str]:
    return _method_order_for_condition(df, methods_to_compare, sort_by)


def _colors(method_order: Sequence[str]) -> Dict[str, Any]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = [f"C{i}" for i in range(10)]
    return {m: cycle[i % len(cycle)] for i, m in enumerate(method_order)}


def _set_y_limits(ax, values: np.ndarray, y_range: Optional[Tuple[float, float]]) -> None:
    if y_range is not None:
        ax.set_ylim(float(y_range[0]), float(y_range[1]))
        return
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    ymin, ymax = float(values.min()), float(values.max())
    pad = max(0.004, (ymax - ymin) * 0.12)
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)


def _plot_axis(
    ax,
    *,
    condition_df: pd.DataFrame,
    method_order: Sequence[str],
    color_map: Dict[str, Any],
    missingness: str,
    fraction: float,
    show_seed_points: bool,
    show_means: bool,
    y_range: Optional[Tuple[float, float]],
    y_tick_step: Optional[float],
    show_condition_title: bool,
    show_ylabel: bool,
    rotate_xticks: float,
    rng_seed: int,
) -> None:
    data, labels = [], []
    for method in method_order:
        vals = condition_df.loc[condition_df["method"] == method, "rmse"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            data.append(vals)
            labels.append(method)

    if not data:
        raise ValueError(f"No data to plot for {missingness}, fraction={fraction}")

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.68,
        showmeans=show_means,
        showfliers=not show_seed_points,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4.2},
        medianprops={"color": "black", "linewidth": 1.15},
        boxprops={"linewidth": 0.9},
        whiskerprops={"linewidth": 0.85},
        capprops={"linewidth": 0.85},
    )

    for patch, method in zip(bp["boxes"], labels):
        patch.set_facecolor(color_map[method])
        patch.set_alpha(0.55)

    if show_seed_points:
        rng = np.random.default_rng(int(rng_seed))
        for i, method in enumerate(labels, start=1):
            vals = condition_df.loc[condition_df["method"] == method, "rmse"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            x = rng.normal(loc=i, scale=0.035, size=len(vals))
            ax.scatter(
                x,
                vals,
                s=14,
                alpha=0.62,
                facecolors="none",
                edgecolors="black",
                linewidths=0.55,
                zorder=3,
            )

    if show_condition_title:
        ax.set_title(f"{missingness}, {format_fraction_percent(float(fraction))} missingness", fontsize=11, fontweight="bold", pad=7)

    if show_ylabel:
        ax.set_ylabel("RMSE", fontsize=11)
    else:
        ax.set_ylabel("")

    ax.grid(True, axis="y", alpha=0.22, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=rotate_xticks, labelsize=9, pad=1)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
        tick.set_rotation_mode("anchor")

    ax.tick_params(axis="y", labelsize=9)
    _set_y_limits(ax, condition_df["rmse"].to_numpy(dtype=float), y_range)

    if y_tick_step is not None:
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(float(y_tick_step)))


def plot_conditions_row(
    *,
    seed_df: pd.DataFrame,
    conditions: Sequence[Tuple[str, float]],
    methods_to_compare: Sequence[str],
    output_dir: Union[str, Path],
    filename: str,
    sort_by: str = "mean",
    shared_method_order: bool = True,
    y_range: Optional[Tuple[float, float]] = None,
    y_tick_step: Optional[float] = None,
    show_seed_points: bool = True,
    show_means: bool = True,
    show_condition_titles: bool = True,
    show_panel_labels: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    rotate_xticks: float = 28,
    save_pdf: bool = True,
    save_png: bool = False,
    show: bool = True,
) -> Dict[str, str]:
    """
    One-row paper figure. No figure-level title/caption is added.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not conditions:
        raise ValueError("conditions cannot be empty.")

    if figsize is None:
        figsize = (6.15 * len(conditions), 5.35)

    fig, axes = plt.subplots(1, len(conditions), figsize=figsize, squeeze=False, sharey=True)
    axes = axes.ravel()

    condition_frames = []
    for missingness, fraction in conditions:
        sub = seed_df[
            (seed_df["missingness"].astype(str).str.upper() == str(missingness).upper())
            & (np.isclose(seed_df["fraction"].astype(float), float(fraction)))
        ].copy()
        if sub.empty:
            raise ValueError(f"No data for {missingness}, fraction={fraction}")
        condition_frames.append(sub)

    if shared_method_order:
        order_df = pd.concat(condition_frames, ignore_index=True)
        method_order = _global_order(order_df, methods_to_compare, sort_by=sort_by)
    else:
        method_order = None

    color_basis = method_order if method_order is not None else [label_for_method(m) for m in methods_to_compare]
    color_map = _colors(color_basis)

    for i, (ax, (missingness, fraction), sub) in enumerate(zip(axes, conditions, condition_frames)):
        local_order = method_order or _method_order_for_condition(sub, methods_to_compare, sort_by=sort_by)
        _plot_axis(
            ax,
            condition_df=sub,
            method_order=local_order,
            color_map=color_map,
            missingness=str(missingness).upper(),
            fraction=float(fraction),
            show_seed_points=show_seed_points,
            show_means=show_means,
            y_range=y_range,
            y_tick_step=y_tick_step,
            show_condition_title=show_condition_titles,
            show_ylabel=False,
            rotate_xticks=rotate_xticks,
            rng_seed=42 + i,
        )

        if i > 0:
            ax.tick_params(axis="y", labelleft=False)

        if show_panel_labels:
            ax.text(
                -0.105,
                1.075,
                chr(ord("A") + i),
                transform=ax.transAxes,
                fontsize=15,
                fontweight="bold",
                va="bottom",
                ha="right",
            )

    fig.supylabel("RMSE", x=0.015, fontsize=11)
    fig.supxlabel("Imputation method", y=0.055, fontsize=11)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.90, wspace=0.035)

    outputs: Dict[str, str] = {}
    if save_pdf:
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs["pdf"] = str(pdf_path)
    if save_png:
        png_path = output_dir / f"{filename}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        outputs["png"] = str(png_path)

    if show:
        plt.show()
    plt.close(fig)
    return outputs


def plot_conditions_grid(
    *,
    seed_df: pd.DataFrame,
    conditions: Sequence[Tuple[str, float]],
    methods_to_compare: Sequence[str],
    output_dir: Union[str, Path],
    filename: str,
    ncols: int = 3,
    sort_by: str = "mean",
    shared_method_order: bool = True,
    y_range: Optional[Tuple[float, float]] = None,
    y_tick_step: Optional[float] = None,
    show_seed_points: bool = True,
    show_means: bool = True,
    show_condition_titles: bool = True,
    show_panel_labels: bool = True,
    rotate_xticks: float = 28,
    save_pdf: bool = True,
    save_png: bool = False,
    show: bool = True,
) -> Dict[str, str]:
    """
    Supplementary multi-panel grid. No figure-level title/caption is added.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(conditions)
    ncols = int(ncols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.05 * nrows), squeeze=False, sharey=True)
    axes_flat = axes.ravel()

    condition_frames = []
    for missingness, fraction in conditions:
        sub = seed_df[
            (seed_df["missingness"].astype(str).str.upper() == str(missingness).upper())
            & (np.isclose(seed_df["fraction"].astype(float), float(fraction)))
        ].copy()
        if sub.empty:
            raise ValueError(f"No data for {missingness}, fraction={fraction}")
        condition_frames.append(sub)

    if shared_method_order:
        order_df = pd.concat(condition_frames, ignore_index=True)
        method_order = _global_order(order_df, methods_to_compare, sort_by=sort_by)
    else:
        method_order = None

    color_basis = method_order if method_order is not None else [label_for_method(m) for m in methods_to_compare]
    color_map = _colors(color_basis)

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue

        missingness, fraction = conditions[i]
        sub = condition_frames[i]
        local_order = method_order or _method_order_for_condition(sub, methods_to_compare, sort_by=sort_by)

        _plot_axis(
            ax,
            condition_df=sub,
            method_order=local_order,
            color_map=color_map,
            missingness=str(missingness).upper(),
            fraction=float(fraction),
            show_seed_points=show_seed_points,
            show_means=show_means,
            y_range=y_range,
            y_tick_step=y_tick_step,
            show_condition_title=show_condition_titles,
            show_ylabel=False,
            rotate_xticks=rotate_xticks,
            rng_seed=99 + i,
        )

        if i % ncols != 0:
            ax.tick_params(axis="y", labelleft=False)

        if show_panel_labels:
            ax.text(
                -0.105,
                1.075,
                chr(ord("A") + i),
                transform=ax.transAxes,
                fontsize=15,
                fontweight="bold",
                va="bottom",
                ha="right",
            )

    fig.supylabel("RMSE", x=0.015, fontsize=11)
    fig.supxlabel("Imputation method", y=0.04, fontsize=11)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.14, top=0.93, wspace=0.04, hspace=0.50)

    outputs: Dict[str, str] = {}
    if save_pdf:
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs["pdf"] = str(pdf_path)
    if save_png:
        png_path = output_dir / f"{filename}.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        outputs["png"] = str(png_path)

    if show:
        plt.show()
    plt.close(fig)
    return outputs


def plot_individual_condition_panels(
    *,
    seed_df: pd.DataFrame,
    conditions: Sequence[Tuple[str, float]],
    methods_to_compare: Sequence[str],
    output_dir: Union[str, Path],
    sort_by: str = "mean",
    y_range: Optional[Tuple[float, float]] = None,
    y_tick_step: Optional[float] = None,
    show_seed_points: bool = True,
    show_means: bool = True,
    show_condition_titles: bool = True,
    rotate_xticks: float = 28,
    save_png: bool = False,
    show: bool = True,
) -> Dict[str, Dict[str, str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    for missingness, fraction in conditions:
        missingness = str(missingness).upper()
        fraction = float(fraction)

        sub = seed_df[
            (seed_df["missingness"].astype(str).str.upper() == missingness)
            & (np.isclose(seed_df["fraction"].astype(float), fraction))
        ].copy()
        if sub.empty:
            raise ValueError(f"No data for {missingness}, fraction={fraction}")

        method_order = _method_order_for_condition(sub, methods_to_compare, sort_by=sort_by)
        color_map = _colors(method_order)

        fig, ax = plt.subplots(figsize=(6.4, 5.15))
        _plot_axis(
            ax,
            condition_df=sub,
            method_order=method_order,
            color_map=color_map,
            missingness=missingness,
            fraction=fraction,
            show_seed_points=show_seed_points,
            show_means=show_means,
            y_range=y_range,
            y_tick_step=y_tick_step,
            show_condition_title=show_condition_titles,
            show_ylabel=True,
            rotate_xticks=rotate_xticks,
            rng_seed=777,
        )
        ax.set_xlabel("Imputation method", fontsize=11)
        fig.subplots_adjust(left=0.135, right=0.985, bottom=0.25, top=0.90)

        stem = f"Panel_RMSE_{missingness}_{safe_fraction_token(fraction)}"
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        panel_paths = {"pdf": str(pdf_path)}

        if save_png:
            png_path = output_dir / f"{stem}.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            panel_paths["png"] = str(png_path)

        if show:
            plt.show()
        plt.close(fig)
        outputs[f"{missingness}_{fraction:g}"] = panel_paths

    return outputs


# =============================================================================
# Public API
# =============================================================================
def make_paper_seed_rmse_boxplots(
    *,
    columns_num: str,
    methods_to_compare: Sequence[str],
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    study_names: Optional[Dict[str, str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    seed_values_csv: Optional[Union[str, Path]] = None,
    csv_overrides: Optional[Dict[str, Union[str, Path]]] = None,
    strict: bool = True,
    sort_by: str = "mean",
    shared_method_order: bool = True,
    y_range: Optional[Tuple[float, float]] = None,
    y_tick_step: Optional[float] = None,
    show_seed_points: bool = True,
    show_means: bool = True,
    show: bool = True,
    save_png: bool = False,
    show_condition_titles: bool = True,
    show_panel_labels: bool = True,
    make_individual_panels: bool = True,
    make_bayes_tables: bool = True,
    print_latex_tables: bool = True,
    print_markdown_tables: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Generate:
      - main MAR RMSE figure
      - supplementary MCAR/MNAR figure
      - supplementary all-condition figure
      - separate panel PDFs
      - Bayesian optimization and supplementary tables

    No figure-level captions are inserted in the PDFs.
    """
    main_dir = Path(config_paths.experiments_dir(columns_num))
    if output_dir is None:
        output_dir = main_dir / "paper_figures" / "seed_rmse_boxplots_final"
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_df = load_seed_rmse_values(
        columns_num=columns_num,
        methods_to_compare=methods_to_compare,
        fractions=fractions,
        missingness_types=missingness_types,
        seeds=seeds,
        seed_values_csv=seed_values_csv,
        csv_overrides=csv_overrides,
        strict=strict,
    )
    seed_summary = summarize_seed_rmse(seed_df)

    values_csv = output_dir / "paper_seed_rmse_values.csv"
    summary_csv = output_dir / "paper_seed_rmse_summary.csv"
    seed_df.to_csv(values_csv, index=False)
    seed_summary.to_csv(summary_csv, index=False)

    outputs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "values_csv": str(values_csv),
        "summary_csv": str(summary_csv),
        "main_mar_figure": None,
        "supplementary_non_mar_figure": None,
        "supplementary_all_conditions_figure": None,
        "individual_panels": None,
        "tables": {},
    }

    main_mar_conditions = [("MAR", 0.1), ("MAR", 0.3), ("MAR", 0.5)]

    outputs["main_mar_figure"] = plot_conditions_row(
        seed_df=seed_df,
        conditions=main_mar_conditions,
        methods_to_compare=methods_to_compare,
        output_dir=output_dir,
        filename="Figure_RMSE_MAR_seed_boxplots",
        sort_by=sort_by,
        shared_method_order=shared_method_order,
        y_range=y_range,
        y_tick_step=y_tick_step,
        show_seed_points=show_seed_points,
        show_means=show_means,
        show_condition_titles=show_condition_titles,
        show_panel_labels=show_panel_labels,
        rotate_xticks=28,
        save_pdf=True,
        save_png=save_png,
        show=show,
    )

    non_mar_conditions = []
    for miss in ["MCAR", "MNAR"]:
        if miss in [m.upper() for m in missingness_types]:
            for frac in [0.1, 0.3, 0.5]:
                if float(frac) in [float(f) for f in fractions]:
                    non_mar_conditions.append((miss, frac))

    if non_mar_conditions:
        outputs["supplementary_non_mar_figure"] = plot_conditions_grid(
            seed_df=seed_df,
            conditions=non_mar_conditions,
            methods_to_compare=methods_to_compare,
            output_dir=output_dir,
            filename="Supplementary_RMSE_MCAR_MNAR_seed_boxplots",
            ncols=3,
            sort_by=sort_by,
            shared_method_order=shared_method_order,
            y_range=y_range,
            y_tick_step=y_tick_step,
            show_seed_points=show_seed_points,
            show_means=show_means,
            show_condition_titles=show_condition_titles,
            show_panel_labels=show_panel_labels,
            rotate_xticks=28,
            save_pdf=True,
            save_png=save_png,
            show=show,
        )

    all_conditions = []
    for miss in ordered_missingness(missingness_types):
        for frac in fractions:
            all_conditions.append((miss, float(frac)))

    outputs["supplementary_all_conditions_figure"] = plot_conditions_grid(
        seed_df=seed_df,
        conditions=all_conditions,
        methods_to_compare=methods_to_compare,
        output_dir=output_dir,
        filename="Supplementary_RMSE_all_conditions_seed_boxplots",
        ncols=len(fractions),
        sort_by=sort_by,
        shared_method_order=shared_method_order,
        y_range=y_range,
        y_tick_step=y_tick_step,
        show_seed_points=show_seed_points,
        show_means=show_means,
        show_condition_titles=show_condition_titles,
        show_panel_labels=show_panel_labels,
        rotate_xticks=28,
        save_pdf=True,
        save_png=save_png,
        show=show,
    )

    if make_individual_panels:
        outputs["individual_panels"] = plot_individual_condition_panels(
            seed_df=seed_df,
            conditions=all_conditions,
            methods_to_compare=methods_to_compare,
            output_dir=output_dir / "individual_panels",
            sort_by=sort_by,
            y_range=y_range,
            y_tick_step=y_tick_step,
            show_seed_points=show_seed_points,
            show_means=show_means,
            show_condition_titles=show_condition_titles,
            rotate_xticks=28,
            save_png=save_png,
            show=show,
        )

    tables: Dict[str, pd.DataFrame] = {}
    if make_bayes_tables:
        if study_names is None:
            raise ValueError("study_names must be supplied when make_bayes_tables=True.")

        tables = make_bayesian_optimization_tables(
            columns_num=columns_num,
            methods_to_compare=methods_to_compare,
            study_names=study_names,
            seed_summary=seed_summary,
            optimization_seed=42,
            final_seeds=seeds,
            tuning_missingness_types=("MAR", "MNAR"),
            tuning_fractions=(0.1, 0.3),
            main_missingness="MAR",
            main_fractions=(0.1, 0.3, 0.5),
            supplementary_missingness_types=("MCAR", "MNAR"),
            supplementary_fractions=(0.1, 0.3, 0.5),
            output_dir=output_dir / "tables",
            print_latex=print_latex_tables,
            print_markdown=print_markdown_tables,
        )
        outputs["tables"] = {name: str(output_dir / "tables" / f"{name}.csv") for name in tables}

    return outputs, seed_df, seed_summary, tables
