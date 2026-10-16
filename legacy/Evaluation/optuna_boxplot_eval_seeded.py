# Evaluation/optuna_boxplot_eval_seeded.py

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, MultipleLocator


# ---------------------------------------------------------------------
# Project imports
# Configure config_paths in the runner before importing this module.
# ---------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config_paths
from config_paths import experiments_dir
from Optimization.common import make_validation_mask, stable_combo_seed


# =====================================================================
# Method registry
# =====================================================================
METHOD_REGISTRY: Dict[str, Dict[str, str]] = {
    "KNN": {"pipeline_prefix": "KNN"},
    "SOFTIMPUTE": {"pipeline_prefix": "SoftImpute"},
    "VAE": {"pipeline_prefix": "VAE"},
    "DAE": {"pipeline_prefix": "DAE"},
    "AE": {"pipeline_prefix": "AE"},
    "MF": {"pipeline_prefix": "MissForest"},
    "GAIN": {"pipeline_prefix": "GAIN"},
    "RNN": {"pipeline_prefix": "RNN"},
    "ITERATIVE": {"pipeline_prefix": "Iterative"},
    "TRF": {"pipeline_prefix": "Transformer"},
    "MEAN": {"pipeline_prefix": "Mean"},
}

METHOD_ALIASES = {
    "knn": "KNN",
    "softimpute": "SOFTIMPUTE",
    "softimpute_py": "SOFTIMPUTE",
    "vae": "VAE",
    "dae": "DAE",
    "ae": "AE",
    "mf": "MF",
    "missforest": "MF",
    "gain": "GAIN",
    "rnn": "RNN",
    "iterative": "ITERATIVE",
    "iterative_imputer": "ITERATIVE",
    "trf": "TRF",
    "transformer": "TRF",
    "mean": "MEAN",
}


# =====================================================================
# General helpers
# =====================================================================
def _canonical_method_name(method: str) -> str:
    raw = str(method).strip()
    canonical = METHOD_ALIASES.get(raw.lower(), raw.upper())

    if canonical not in METHOD_REGISTRY:
        raise ValueError(
            f"Unsupported method '{method}'. Supported canonical methods: "
            f"{sorted(METHOD_REGISTRY)}"
        )

    return canonical


def _safe_fraction_token(fraction: float) -> str:
    return f"{float(fraction):g}".replace(".", "p")


def _ordered_missingness(values: Sequence[str]) -> List[str]:
    preferred = ["MCAR", "MAR", "MNAR"]
    observed = [str(value).upper() for value in values]

    present = [
        value for value in preferred
        if value in set(observed)
    ]

    extras = sorted(
        value for value in set(observed)
        if value not in present
    )

    return present + extras


def _normalise_study_name_mapping(
    study_names: Mapping[str, str],
) -> Dict[str, str]:
    normalised: Dict[str, str] = {}

    for method, study_name in study_names.items():
        canonical = _canonical_method_name(method)

        if canonical == "MEAN":
            raise ValueError(
                "Mean imputation must not have an Optuna study name."
            )

        if not str(study_name).strip():
            raise ValueError(
                f"Study name for method '{method}' is empty."
            )

        if (
            canonical in normalised
            and normalised[canonical] != str(study_name)
        ):
            raise ValueError(
                "Multiple different study names were supplied for "
                f"canonical method {canonical}."
            )

        normalised[canonical] = str(study_name)

    return normalised


def _validate_requested_grid(
    df: pd.DataFrame,
    *,
    method: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    replicate_column: Optional[str],
    expected_replicates: Optional[Sequence[int]] = None,
) -> None:
    expected_pairs = {
        (float(fraction), str(missingness).upper())
        for fraction in fractions
        for missingness in missingness_types
    }

    observed_pairs = {
        (float(row.fraction), str(row.missingness).upper())
        for row in df[
            ["fraction", "missingness"]
        ].itertuples(index=False)
    }

    missing_pairs = sorted(expected_pairs - observed_pairs)

    if missing_pairs:
        raise RuntimeError(
            f"Method '{method}' is missing requested "
            f"fraction/missingness conditions: {missing_pairs}"
        )

    if replicate_column is None:
        return

    if replicate_column not in df.columns:
        raise KeyError(
            f"Method '{method}' data is missing replicate column "
            f"'{replicate_column}'."
        )

    duplicated = df.duplicated(
        subset=["fraction", "missingness", replicate_column],
        keep=False,
    )

    if duplicated.any():
        bad = df.loc[
            duplicated,
            ["fraction", "missingness", replicate_column],
        ].sort_values(
            ["missingness", "fraction", replicate_column]
        )

        raise RuntimeError(
            f"Method '{method}' contains duplicate "
            f"condition/replicate rows:\n{bad.to_string(index=False)}"
        )

    if expected_replicates is not None:
        expected_set = {
            int(value) for value in expected_replicates
        }

        for fraction, missingness in sorted(expected_pairs):
            sub = df[
                np.isclose(
                    df["fraction"].astype(float),
                    float(fraction),
                )
                & (
                    df["missingness"].astype(str).str.upper()
                    == missingness
                )
            ]

            observed_set = {
                int(value)
                for value in pd.to_numeric(
                    sub[replicate_column],
                    errors="raise",
                ).tolist()
            }

            if observed_set != expected_set:
                raise RuntimeError(
                    f"Method '{method}', fraction={fraction}, "
                    f"missingness={missingness}: expected "
                    f"{replicate_column} values "
                    f"{sorted(expected_set)}, observed "
                    f"{sorted(observed_set)}."
                )


def _validate_reference_and_corrupted_frames(
    corrupted: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    corrupted_path: Path,
    reference_path: Path,
) -> None:
    if corrupted.shape[1] < 3 or reference.shape[1] < 3:
        raise ValueError(
            "Both corrupted and reference files must contain "
            "eid, label, and at least one protein column."
        )

    if list(corrupted.columns[:2]) != ["eid", "label"]:
        raise ValueError(
            "Corrupted file first two columns must be "
            f"['eid', 'label']:\n{corrupted_path}"
        )

    if list(reference.columns[:2]) != ["eid", "label"]:
        raise ValueError(
            "Reference file first two columns must be "
            f"['eid', 'label']:\n{reference_path}"
        )

    if corrupted.shape != reference.shape:
        raise ValueError(
            "Corrupted and reference files have different shapes.\n"
            f"Corrupted: {corrupted.shape} | {corrupted_path}\n"
            f"Reference: {reference.shape} | {reference_path}"
        )

    if list(corrupted.columns) != list(reference.columns):
        raise ValueError(
            "Corrupted and reference files have different "
            "columns or column order.\n"
            f"Corrupted: {corrupted_path}\n"
            f"Reference: {reference_path}"
        )

    if not corrupted["eid"].astype(str).equals(
        reference["eid"].astype(str)
    ):
        raise ValueError(
            "eid order differs between corrupted and reference files:\n"
            f"{corrupted_path}\n{reference_path}"
        )

    if not corrupted["label"].astype(str).equals(
        reference["label"].astype(str)
    ):
        raise ValueError(
            "label values differ between corrupted and reference files:\n"
            f"{corrupted_path}\n{reference_path}"
        )


def _mean_impute_matrix(
    matrix_with_nan: np.ndarray,
) -> np.ndarray:
    matrix_with_nan = np.asarray(matrix_with_nan, dtype=float)
    means = np.nanmean(matrix_with_nan, axis=0)

    bad_columns = np.where(~np.isfinite(means))[0]

    if bad_columns.size:
        raise ValueError(
            "Mean imputation cannot proceed because some protein "
            "columns contain no observed values. "
            f"Bad column indices, first 20: "
            f"{bad_columns[:20].tolist()}"
        )

    imputed = matrix_with_nan.copy()
    missing_rows, missing_cols = np.where(~np.isfinite(imputed))
    imputed[missing_rows, missing_cols] = means[missing_cols]

    if not np.isfinite(imputed).all():
        raise FloatingPointError(
            "Mean-imputed matrix contains non-finite values."
        )

    return imputed


def _rmse_on_exact_mask(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> float:
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if (
        reference.shape != prediction.shape
        or reference.shape != mask.shape
    ):
        raise ValueError(
            "reference, prediction, and mask must have identical shapes."
        )

    if int(mask.sum()) == 0:
        raise ValueError(
            "The requested RMSE mask contains zero cells."
        )

    if not np.isfinite(reference[mask]).all():
        raise ValueError(
            "Reference values on the RMSE mask contain "
            "non-finite values."
        )

    if not np.isfinite(prediction[mask]).all():
        raise ValueError(
            "Predicted values on the RMSE mask contain "
            "non-finite values."
        )

    return float(
        np.sqrt(
            np.mean(
                (reference[mask] - prediction[mask]) ** 2
            )
        )
    )


# =====================================================================
# Hyperparameter-stage loading
# =====================================================================
def _load_exact_condition_score_table(
    *,
    main_dir: Path,
    display_method: str,
    canonical_method: str,
    study_name: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    optimization_seed: int,
) -> pd.DataFrame:
    condition_path = (
        main_dir
        / "Optimization"
        / "trial_tables"
        / f"{study_name}_condition_scores.csv"
    )

    best_json_path = (
        main_dir
        / "Optimization"
        / "best_params"
        / f"{study_name}_best_params.json"
    )

    if not condition_path.exists():
        raise FileNotFoundError(
            f"Condition-score table not found for method "
            f"'{display_method}':\n{condition_path}"
        )

    if not best_json_path.exists():
        raise FileNotFoundError(
            f"Best-parameter JSON not found for method "
            f"'{display_method}':\n{best_json_path}"
        )

    with open(best_json_path, "r") as handle:
        best_payload = json.load(handle)

    stored_study_name = str(
        best_payload.get("study_name", "")
    )

    if stored_study_name and stored_study_name != study_name:
        raise RuntimeError(
            f"Best JSON study name mismatch for method "
            f"'{display_method}'. Expected '{study_name}', "
            f"found '{stored_study_name}'."
        )

    if "seed" not in best_payload:
        raise KeyError(
            f"Best JSON for method '{display_method}' does not "
            "contain the optimization seed."
        )

    stored_seed = int(best_payload["seed"])

    if stored_seed != int(optimization_seed):
        raise RuntimeError(
            f"Optimization seed mismatch for method "
            f"'{display_method}'. Requested {optimization_seed}, "
            f"best JSON contains {stored_seed}."
        )

    df = pd.read_csv(condition_path)

    required_columns = {
        "trial_number",
        "state",
        "fraction",
        "missingness",
        "validation_rmse",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            f"Condition-score table for method '{display_method}' "
            f"is missing columns {sorted(missing_columns)}. "
            f"File: {condition_path}"
        )

    df = df[
        df["state"].astype(str).str.upper() == "COMPLETE"
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No COMPLETE Optuna rows found for method "
            f"'{display_method}':\n{condition_path}"
        )

    df["fraction"] = pd.to_numeric(
        df["fraction"], errors="raise"
    )
    df["missingness"] = (
        df["missingness"].astype(str).str.upper()
    )
    df["validation_rmse"] = pd.to_numeric(
        df["validation_rmse"], errors="raise"
    )
    df["trial_number"] = pd.to_numeric(
        df["trial_number"], errors="raise"
    ).astype(int)

    fraction_mask = np.zeros(len(df), dtype=bool)

    for fraction in fractions:
        fraction_mask |= np.isclose(
            df["fraction"].to_numpy(dtype=float),
            float(fraction),
        )

    df = df[
        fraction_mask
        & df["missingness"].isin(
            [str(value).upper() for value in missingness_types]
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"Method '{display_method}' has no Optuna rows "
            "for the requested conditions."
        )

    _validate_requested_grid(
        df,
        method=display_method,
        fractions=fractions,
        missingness_types=missingness_types,
        replicate_column="trial_number",
    )

    df["method"] = display_method
    df["canonical_method"] = canonical_method
    df["study_name"] = study_name
    df["optimization_seed"] = int(optimization_seed)
    df["rmse"] = df["validation_rmse"]
    df["source"] = "optuna_validation_trial"
    df["best_params_json"] = str(best_json_path)
    df["condition_scores_csv"] = str(condition_path)

    return df


def _compute_mean_validation_baseline(
    *,
    main_dir: Path,
    display_method: str,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    optimization_seed: int,
    val_rate: float,
    validation_mode: str,
) -> pd.DataFrame:
    """Used only if Mean is explicitly included in methods_to_compare."""
    rows: List[Dict[str, Any]] = []

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            folder = (
                main_dir
                / f"fraction_{fraction}_missingness_{missingness}"
            )

            corrupted_path = folder / (
                f"Corrupted_data_fraction_{fraction}"
                f"_missingness_{missingness}"
                f"_col_{columns_num}"
                f"_seed_{int(optimization_seed)}.csv"
            )

            reference_path = (
                folder / f"normalized_data_col_{columns_num}.csv"
            )

            if not corrupted_path.exists():
                raise FileNotFoundError(
                    "Seeded corrupted file required for Mean "
                    f"validation baseline is missing:\n{corrupted_path}"
                )

            if not reference_path.exists():
                raise FileNotFoundError(
                    "Reference file required for Mean validation "
                    f"baseline is missing:\n{reference_path}"
                )

            corrupted = pd.read_csv(corrupted_path)
            reference = pd.read_csv(reference_path)

            _validate_reference_and_corrupted_frames(
                corrupted,
                reference,
                corrupted_path=corrupted_path,
                reference_path=reference_path,
            )

            corrupted_values = corrupted.iloc[:, 2:].to_numpy(
                dtype=float
            )
            reference_values = reference.iloc[:, 2:].to_numpy(
                dtype=float
            )

            observed_mask = np.isfinite(corrupted_values)

            combo_seed = stable_combo_seed(
                int(optimization_seed),
                fraction,
                missingness,
            )

            validation_mask = make_validation_mask(
                data=corrupted_values,
                observed_mask=observed_mask,
                val_rate=float(val_rate),
                seed=int(combo_seed),
                mode=str(validation_mode),
            )

            validation_corrupted = corrupted_values.copy()
            validation_corrupted[validation_mask] = np.nan

            imputed = _mean_impute_matrix(validation_corrupted)

            score = _rmse_on_exact_mask(
                reference_values,
                imputed,
                validation_mask,
            )

            rows.append(
                {
                    "method": display_method,
                    "canonical_method": "MEAN",
                    "study_name": "fixed_mean_no_optimization",
                    "trial_number": 0,
                    "fraction": fraction,
                    "missingness": missingness,
                    "validation_rmse": score,
                    "rmse": score,
                    "n_validation_cells": int(validation_mask.sum()),
                    "optimization_seed": int(optimization_seed),
                    "source": "fixed_mean_validation_baseline",
                    "best_params_json": "fixed_no_optimization",
                    "condition_scores_csv": "computed_directly",
                }
            )

    result = pd.DataFrame(rows)

    _validate_requested_grid(
        result,
        method=display_method,
        fractions=fractions,
        missingness_types=missingness_types,
        replicate_column=None,
    )

    return result


def load_hyperparameter_comparison_values(
    *,
    main_dir: Path,
    methods_to_compare: Sequence[str],
    study_names: Mapping[str, str],
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    columns_num: str,
    optimization_seed: int,
    mean_val_rate: float,
    mean_validation_mode: str,
) -> pd.DataFrame:
    exact_studies = _normalise_study_name_mapping(study_names)
    frames: List[pd.DataFrame] = []

    for display_method in methods_to_compare:
        canonical = _canonical_method_name(display_method)

        if canonical == "MEAN":
            frame = _compute_mean_validation_baseline(
                main_dir=main_dir,
                display_method=str(display_method),
                columns_num=columns_num,
                fractions=fractions,
                missingness_types=missingness_types,
                optimization_seed=optimization_seed,
                val_rate=mean_val_rate,
                validation_mode=mean_validation_mode,
            )
        else:
            if canonical not in exact_studies:
                raise KeyError(
                    f"No exact study name was supplied for method "
                    f"'{display_method}' (canonical {canonical})."
                )

            frame = _load_exact_condition_score_table(
                main_dir=main_dir,
                display_method=str(display_method),
                canonical_method=canonical,
                study_name=exact_studies[canonical],
                fractions=fractions,
                missingness_types=missingness_types,
                optimization_seed=optimization_seed,
            )

        frames.append(frame)

    if not frames:
        raise ValueError(
            "No methods were supplied for hyperparameter comparison."
        )

    return pd.concat(frames, ignore_index=True)


# =====================================================================
# Repeated-seed final benchmark loading
# =====================================================================
def _default_seed_results_csv(
    *,
    main_dir: Path,
    canonical_method: str,
) -> Path:
    prefix = METHOD_REGISTRY[
        canonical_method
    ]["pipeline_prefix"]

    return (
        main_dir
        / "Evaluation"
        / f"{prefix}_seeded_pipeline"
        / f"{prefix}_final_rmse_by_seed.csv"
    )


def _load_method_seed_results(
    *,
    main_dir: Path,
    display_method: str,
    canonical_method: str,
    study_name: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    override_path: Optional[Path],
    require_not_older_than_best_json: bool,
) -> pd.DataFrame:
    seed_csv = (
        Path(override_path).expanduser().resolve()
        if override_path is not None
        else _default_seed_results_csv(
            main_dir=main_dir,
            canonical_method=canonical_method,
        )
    )

    best_json_path = (
        main_dir
        / "Optimization"
        / "best_params"
        / f"{study_name}_best_params.json"
    )

    if not seed_csv.exists():
        raise FileNotFoundError(
            f"Repeated-seed RMSE CSV not found for method "
            f"'{display_method}':\n{seed_csv}\n"
            "Run the method-specific seeded evaluator first."
        )

    if not best_json_path.exists():
        raise FileNotFoundError(
            f"Best-parameter JSON not found for method "
            f"'{display_method}':\n{best_json_path}"
        )

    if require_not_older_than_best_json:
        if seed_csv.stat().st_mtime < best_json_path.stat().st_mtime:
            raise RuntimeError(
                f"Repeated-seed CSV for method '{display_method}' "
                "is older than the best-parameter JSON. "
                "Re-run the method-specific final benchmark "
                "and evaluator.\n"
                f"CSV: {seed_csv}\nBest JSON: {best_json_path}"
            )

    df = pd.read_csv(seed_csv)

    required_columns = {
        "fraction",
        "missingness",
        "seed",
        "rmse",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            f"Repeated-seed CSV for method '{display_method}' "
            f"is missing columns {sorted(missing_columns)}. "
            f"File: {seed_csv}"
        )

    df["fraction"] = pd.to_numeric(
        df["fraction"], errors="raise"
    )
    df["missingness"] = (
        df["missingness"].astype(str).str.upper()
    )
    df["seed"] = pd.to_numeric(
        df["seed"], errors="raise"
    ).astype(int)
    df["rmse"] = pd.to_numeric(
        df["rmse"], errors="raise"
    )

    fraction_mask = np.zeros(len(df), dtype=bool)

    for fraction in fractions:
        fraction_mask |= np.isclose(
            df["fraction"].to_numpy(dtype=float),
            float(fraction),
        )

    df = df[
        fraction_mask
        & df["missingness"].isin(
            [str(value).upper() for value in missingness_types]
        )
        & df["seed"].isin(
            [int(value) for value in seeds]
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"Repeated-seed CSV for method '{display_method}' "
            "contains no requested rows."
        )

    _validate_requested_grid(
        df,
        method=display_method,
        fractions=fractions,
        missingness_types=missingness_types,
        replicate_column="seed",
        expected_replicates=seeds,
    )

    df["method"] = display_method
    df["canonical_method"] = canonical_method
    df["study_name"] = study_name
    df["source"] = "frozen_best_repeated_seed"
    df["seed_results_csv"] = str(seed_csv)
    df["best_params_json"] = str(best_json_path)

    return df


def _compute_mean_seed_results(
    *,
    main_dir: Path,
    display_method: str,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Used only if Mean is explicitly included in methods_to_compare."""
    rows: List[Dict[str, Any]] = []

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            folder = (
                main_dir
                / f"fraction_{fraction}_missingness_{missingness}"
            )

            reference_path = (
                folder / f"normalized_data_col_{columns_num}.csv"
            )

            if not reference_path.exists():
                raise FileNotFoundError(
                    "Reference file required for Mean seed "
                    f"benchmark is missing:\n{reference_path}"
                )

            reference = pd.read_csv(reference_path)

            for seed in seeds:
                seed = int(seed)

                corrupted_path = folder / (
                    f"Corrupted_data_fraction_{fraction}"
                    f"_missingness_{missingness}"
                    f"_col_{columns_num}_seed_{seed}.csv"
                )

                if not corrupted_path.exists():
                    raise FileNotFoundError(
                        "Seeded corrupted file required for Mean "
                        f"benchmark is missing:\n{corrupted_path}"
                    )

                corrupted = pd.read_csv(corrupted_path)

                _validate_reference_and_corrupted_frames(
                    corrupted,
                    reference,
                    corrupted_path=corrupted_path,
                    reference_path=reference_path,
                )

                corrupted_values = corrupted.iloc[:, 2:].to_numpy(
                    dtype=float
                )
                reference_values = reference.iloc[:, 2:].to_numpy(
                    dtype=float
                )

                missing_mask = ~np.isfinite(corrupted_values)

                if int(missing_mask.sum()) == 0:
                    raise ValueError(
                        "Mean benchmark found no missing cells: "
                        f"{corrupted_path}"
                    )

                imputed = _mean_impute_matrix(corrupted_values)

                score = _rmse_on_exact_mask(
                    reference_values,
                    imputed,
                    missing_mask,
                )

                rows.append(
                    {
                        "method": display_method,
                        "canonical_method": "MEAN",
                        "study_name": "fixed_mean_no_optimization",
                        "fraction": fraction,
                        "missingness": missingness,
                        "seed": seed,
                        "rmse": score,
                        "source": "fixed_mean_repeated_seed",
                        "seed_results_csv": "computed_directly",
                        "best_params_json": "fixed_no_optimization",
                        "corrupted_path": str(corrupted_path),
                        "reference_path": str(reference_path),
                    }
                )

    result = pd.DataFrame(rows)

    _validate_requested_grid(
        result,
        method=display_method,
        fractions=fractions,
        missingness_types=missingness_types,
        replicate_column="seed",
        expected_replicates=seeds,
    )

    return result


def load_seed_robustness_values(
    *,
    main_dir: Path,
    methods_to_compare: Sequence[str],
    study_names: Mapping[str, str],
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    columns_num: str,
    seed_result_csv_overrides: Optional[Mapping[str, str]],
    require_not_older_than_best_json: bool,
) -> pd.DataFrame:
    exact_studies = _normalise_study_name_mapping(study_names)
    override_map: Dict[str, Path] = {}

    if seed_result_csv_overrides is not None:
        for method, path in seed_result_csv_overrides.items():
            canonical = _canonical_method_name(method)
            override_map[canonical] = Path(path)

    frames: List[pd.DataFrame] = []

    for display_method in methods_to_compare:
        canonical = _canonical_method_name(display_method)

        if canonical == "MEAN":
            frame = _compute_mean_seed_results(
                main_dir=main_dir,
                display_method=str(display_method),
                columns_num=columns_num,
                fractions=fractions,
                missingness_types=missingness_types,
                seeds=seeds,
            )
        else:
            if canonical not in exact_studies:
                raise KeyError(
                    f"No exact study name was supplied for method "
                    f"'{display_method}' (canonical {canonical})."
                )

            frame = _load_method_seed_results(
                main_dir=main_dir,
                display_method=str(display_method),
                canonical_method=canonical,
                study_name=exact_studies[canonical],
                fractions=fractions,
                missingness_types=missingness_types,
                seeds=seeds,
                override_path=override_map.get(canonical),
                require_not_older_than_best_json=(
                    require_not_older_than_best_json
                ),
            )

        frames.append(frame)

    if not frames:
        raise ValueError(
            "No methods were supplied for repeated-seed comparison."
        )

    return pd.concat(frames, ignore_index=True)


# =====================================================================
# Shared plotting helpers
# =====================================================================
def _missingness_color_map(
    missingness_types: Sequence[str],
) -> Dict[str, Any]:
    missingness_types = _ordered_missingness(missingness_types)

    colors = (
        plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    )

    if len(colors) < len(missingness_types):
        colors = [
            f"C{index}"
            for index in range(max(10, len(missingness_types)))
        ]

    return {
        missingness: colors[index]
        for index, missingness in enumerate(missingness_types)
    }


def _resolve_method_order(
    plot_df: pd.DataFrame,
    *,
    methods_to_compare: Sequence[str],
    sort_by: str,
) -> List[str]:
    input_order = [
        str(method) for method in methods_to_compare
    ]

    present_methods = set(
        plot_df["method"].astype(str).unique()
    )

    input_order = [
        method for method in input_order
        if method in present_methods
    ]

    sort_by = str(sort_by).lower()

    if sort_by == "input":
        return input_order

    aggregators = {
        "min": "min",
        "median": "median",
        "mean": "mean",
    }

    if sort_by not in aggregators:
        raise ValueError(
            "sort_by must be one of: input, min, median, mean."
        )

    summary = (
        plot_df.groupby("method")["rmse"]
        .agg(aggregators[sort_by])
    )

    ordered = summary.sort_values().index.astype(str).tolist()

    return [
        method for method in ordered
        if method in present_methods
    ]


def _build_grouped_box_data(
    *,
    plot_df: pd.DataFrame,
    methods_to_compare: Sequence[str],
    missingness_types: Sequence[str],
    sort_by: str,
) -> Dict[str, Any]:
    df = plot_df.copy()

    df["rmse"] = pd.to_numeric(
        df["rmse"], errors="raise"
    )
    df["missingness"] = (
        df["missingness"].astype(str).str.upper()
    )

    missingness_types = _ordered_missingness(missingness_types)

    order = _resolve_method_order(
        df,
        methods_to_compare=methods_to_compare,
        sort_by=sort_by,
    )

    if not order:
        raise ValueError(
            "No methods with plot data were found."
        )

    group_gap = 1.0
    group_width = len(missingness_types) + group_gap

    positions: List[float] = []
    box_data: List[np.ndarray] = []
    box_missingness: List[str] = []
    box_methods: List[str] = []

    for method_index, method in enumerate(order):
        base = method_index * group_width + 1.0

        for missingness_index, missingness in enumerate(
            missingness_types
        ):
            values = df.loc[
                (df["method"].astype(str) == method)
                & (df["missingness"] == missingness),
                "rmse",
            ].to_numpy(dtype=float)

            values = values[np.isfinite(values)]

            if values.size == 0:
                raise RuntimeError(
                    f"No values found for method={method}, "
                    f"missingness={missingness}."
                )

            positions.append(base + missingness_index)
            box_data.append(values)
            box_missingness.append(missingness)
            box_methods.append(method)

    tick_positions = [
        method_index * group_width
        + 1.0
        + (len(missingness_types) - 1) / 2.0
        for method_index in range(len(order))
    ]

    return {
        "order": order,
        "positions": positions,
        "box_data": box_data,
        "box_missingness": box_missingness,
        "box_methods": box_methods,
        "tick_positions": tick_positions,
        "group_width": group_width,
    }


def _draw_grouped_boxplot_on_axis(
    ax: Any,
    *,
    grouped: Mapping[str, Any],
    missingness_types: Sequence[str],
    show_points: bool,
    point_seed_column: Optional[str],
    show_legend: bool,
    show_xlabels: bool,
    ylabel: str,
    title: Optional[str],
) -> None:
    """Original final-seed boxplot drawing behavior."""
    missingness_types = _ordered_missingness(missingness_types)
    color_map = _missingness_color_map(missingness_types)

    bp = ax.boxplot(
        grouped["box_data"],
        positions=grouped["positions"],
        widths=0.70,
        patch_artist=True,
        showmeans=False,
        showfliers=True,
        boxprops={
            "linewidth": 1.2,
            "color": "black",
        },
        medianprops={
            "linewidth": 1.5,
            "color": "black",
        },
        whiskerprops={
            "linewidth": 1.0,
            "color": "black",
        },
        capprops={
            "linewidth": 1.0,
            "color": "black",
        },
        flierprops={
            "marker": "o",
            "markersize": 3,
            "alpha": 0.45,
            "markeredgewidth": 0.0,
        },
    )

    for patch, missingness in zip(
        bp["boxes"], grouped["box_missingness"]
    ):
        patch.set_facecolor(color_map[missingness])
        patch.set_alpha(0.50)

    for flier, missingness in zip(
        bp["fliers"], grouped["box_missingness"]
    ):
        flier.set_markerfacecolor(color_map[missingness])
        flier.set_markeredgecolor(color_map[missingness])

    if show_points:
        rng = np.random.default_rng(42)

        for position, values, missingness in zip(
            grouped["positions"],
            grouped["box_data"],
            grouped["box_missingness"],
        ):
            jittered_x = rng.normal(
                position, 0.045, size=len(values)
            )

            ax.scatter(
                jittered_x,
                values,
                s=16,
                alpha=0.55,
                color=color_map[missingness],
                edgecolor="none",
                zorder=3,
            )

    ax.set_xticks(grouped["tick_positions"])

    if show_xlabels:
        ax.set_xticklabels(
            grouped["order"],
            rotation=25,
            ha="right",
        )
        ax.set_xlabel("Imputation method")
    else:
        ax.set_xticklabels([])

    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.28)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color_map[missingness],
                edgecolor="none",
                alpha=0.50,
                label=missingness,
            )
            for missingness in missingness_types
        ]

        ax.legend(
            handles=handles,
            title="Missingness",
            loc="upper right",
        )


# =====================================================================
# Publication-sized S1 plots
# =====================================================================
def _plot_hyperparameter_publication_fraction(
    *,
    plot_df,
    methods_to_compare,
    fraction,
    missingness_types,
    out_dir,
    filename_prefix,
    zoom_y_range,
    full_y_tick_step,
    show_points,
    show,
    save_pdf,
    pdf_pages,
):
    """
    Create one S1 figure per missingness fraction.

    Output width: 131 mm.
    With zoom enabled: full range above, expanded view below.
    Both panels use the same data and the same point jitter.
    """
    width_mm = 131.0
    zoom_tick_step = 0.02

    preferred = [
        "ITERATIVE",
        "MF",
        "SOFTIMPUTE",
        "KNN",
        "AE",
        "DAE",
        "GAIN",
        "VAE",
        "TRF",
        "MEAN",
        "RNN",
    ]

    priority = {
        method: index
        for index, method in enumerate(preferred)
    }

    methods = sorted(
        methods_to_compare,
        key=lambda method: priority.get(
            _canonical_method_name(method),
            len(priority),
        ),
    )

    mechanisms = _ordered_missingness(missingness_types)

    if not np.isfinite(
        pd.to_numeric(plot_df["rmse"], errors="raise")
    ).all():
        raise ValueError(
            "Non-finite validation RMSE values were found "
            "in the S1 data."
        )

    grouped = _build_grouped_box_data(
        plot_df=plot_df,
        methods_to_compare=methods,
        missingness_types=mechanisms,
        sort_by="input",
    )

    labels = [
        METHOD_REGISTRY[
            _canonical_method_name(method)
        ]["pipeline_prefix"]
        for method in grouped["order"]
    ]

    colors = {
        "MCAR": "#009E73",
        "MAR": "#0072B2",
        "MNAR": "#D55E00",
    }

    for index, mechanism in enumerate(mechanisms):
        if mechanism not in colors:
            colors[mechanism] = plt.get_cmap("tab10")(index % 10)

    has_zoom = zoom_y_range is not None

    if has_zoom:
        low, high = map(float, zoom_y_range)

        if (
            not np.isfinite([low, high]).all()
            or low >= high
        ):
            raise ValueError(
                "The zoom limits must be finite and increasing."
            )

    # These settings apply only to S1.
    # Final-seed plots retain their existing styling.
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
    }

    with plt.rc_context(style):
        fig = plt.figure(
            figsize=(
                width_mm / 25.4,
                5.8 if has_zoom else 3.8,
            ),
            constrained_layout=True,
        )

        grid = fig.add_gridspec(
            3 if has_zoom else 2,
            1,
            height_ratios=(
                [1.3, 2.7, 0.25]
                if has_zoom
                else [3.0, 0.25]
            ),
        )

        ax_full = fig.add_subplot(grid[0, 0])
        axes = [ax_full]

        if has_zoom:
            axes.append(
                fig.add_subplot(
                    grid[1, 0],
                    sharex=ax_full,
                )
            )

        # Identical jitter in the full and expanded views.
        rng = np.random.default_rng(42)

        point_positions = [
            position
            + rng.uniform(-0.13, 0.13, size=len(values))
            for position, values in zip(
                grouped["positions"],
                grouped["box_data"],
            )
        ]

        for axis_index, ax in enumerate(axes):
            boxes = ax.boxplot(
                grouped["box_data"],
                positions=grouped["positions"],
                widths=0.70,
                patch_artist=True,
                showmeans=False,
                showfliers=not show_points,
                boxprops={
                    "linewidth": 0.7,
                    "color": "0.25",
                },
                medianprops={
                    "linewidth": 1.1,
                    "color": "black",
                },
                whiskerprops={
                    "linewidth": 0.7,
                    "color": "0.3",
                },
                capprops={
                    "linewidth": 0.7,
                    "color": "0.3",
                },
                flierprops={
                    "marker": "o",
                    "markersize": 2.5,
                    "alpha": 0.65,
                },
            )

            for box, mechanism in zip(
                boxes["boxes"],
                grouped["box_missingness"],
            ):
                box.set_facecolor(colors[mechanism])
                box.set_alpha(0.35)

            for flier, mechanism in zip(
                boxes["fliers"],
                grouped["box_missingness"],
            ):
                flier.set_markerfacecolor(colors[mechanism])
                flier.set_markeredgecolor(colors[mechanism])

            if show_points:
                for x, values, mechanism in zip(
                    point_positions,
                    grouped["box_data"],
                    grouped["box_missingness"],
                ):
                    ax.scatter(
                        x,
                        values,
                        s=7,
                        color=colors[mechanism],
                        alpha=0.65,
                        edgecolors="none",
                        zorder=3,
                    )

            ax.set_xticks(grouped["tick_positions"])
            ax.set_xticklabels(
                labels,
                rotation=45,
                ha="right",
                fontsize=9,
            )

            ax.tick_params(
                axis="x",
                labelbottom=axis_index == len(axes) - 1,
            )
            ax.tick_params(axis="y", labelsize=9)

            ax.set_ylabel("Validation RMSE")
            ax.grid(
                axis="y",
                color="0.90",
                linewidth=0.6,
            )
            ax.set_axisbelow(True)

            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        ax_full.set_title(
            f"{fraction:.0%} missingness: full range",
            loc="left",
            pad=7,
        )

        if (
            full_y_tick_step is not None
            and full_y_tick_step > 0
        ):
            ax_full.yaxis.set_major_locator(
                MultipleLocator(float(full_y_tick_step))
            )
        else:
            ax_full.yaxis.set_major_locator(
                MaxNLocator(nbins=4)
            )

        if has_zoom:
            ax_zoom = axes[1]
            ax_zoom.set_ylim(low, high)

            ax_zoom.set_title(
                "Expanded view",
                loc="left",
                fontsize=10,
                pad=7,
            )

            ax_zoom.yaxis.set_major_locator(
                MultipleLocator(zoom_tick_step)
            )

            ax_full.axhspan(
                low,
                high,
                color="0.4",
                alpha=0.08,
                zorder=0,
            )

        axes[-1].set_xlabel("Imputation method")

        legend_ax = fig.add_subplot(grid[-1, 0])
        legend_ax.axis("off")

        handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=colors[mechanism],
                alpha=0.55,
                edgecolor="none",
                label=mechanism,
            )
            for mechanism in mechanisms
        ]

        legend_ax.legend(
            handles=handles,
            loc="center",
            ncol=len(mechanisms),
            frameon=False,
            fontsize=9,
        )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        token = _safe_fraction_token(fraction)

        output_png = (
            out_dir
            / f"{filename_prefix}_fraction_{token}.png"
        )

        output_pdf = (
            out_dir
            / f"{filename_prefix}_fraction_{token}.pdf"
            if save_pdf
            else None
        )

        fig.canvas.draw()

        # Preserve the physical width of 131 mm.
        # Do not use bbox_inches="tight" for these S1 figures.
        fig.savefig(
            output_png,
            dpi=300,
            facecolor="white",
        )

        if output_pdf is not None:
            fig.savefig(
                output_pdf,
                facecolor="white",
            )

        if pdf_pages is not None:
            pdf_pages.savefig(
                fig,
                facecolor="white",
            )

        if show:
            plt.show()

        plt.close(fig)

    return output_png, output_pdf


# =====================================================================
# Plot dispatcher and original final-seed layout
# =====================================================================
def _plot_fraction_distribution(
    *,
    plot_df: pd.DataFrame,
    methods_to_compare: Sequence[str],
    fraction: float,
    missingness_types: Sequence[str],
    out_dir: Path,
    filename_prefix: str,
    figure_title: str,
    sort_by: str,
    zoom_y_range: Optional[Tuple[float, float]],
    full_y_tick_step: Optional[float],
    zoom_y_tick_step: Optional[float],
    show_points: bool,
    show: bool,
    save_pdf: bool,
    pdf_pages: Optional[PdfPages],
) -> Tuple[Path, Optional[Path]]:
    # Use the new publication layout only for S1.
    if filename_prefix == "hyperparameter_grouped_boxplot":
        return _plot_hyperparameter_publication_fraction(
            plot_df=plot_df,
            methods_to_compare=methods_to_compare,
            fraction=fraction,
            missingness_types=missingness_types,
            out_dir=out_dir,
            filename_prefix=filename_prefix,
            zoom_y_range=zoom_y_range,
            full_y_tick_step=full_y_tick_step,
            show_points=show_points,
            show=show,
            save_pdf=save_pdf,
            pdf_pages=pdf_pages,
        )

    # Existing final-seed plotting behavior.
    grouped = _build_grouped_box_data(
        plot_df=plot_df,
        methods_to_compare=methods_to_compare,
        missingness_types=missingness_types,
        sort_by=sort_by,
    )

    n_methods = len(grouped["order"])
    n_missingness = len(
        _ordered_missingness(missingness_types)
    )

    fig_width = max(
        10.0,
        1.05 * n_methods * max(n_missingness, 2),
    )

    if zoom_y_range is None:
        fig, ax = plt.subplots(
            figsize=(fig_width, 5.8)
        )

        _draw_grouped_boxplot_on_axis(
            ax,
            grouped=grouped,
            missingness_types=missingness_types,
            show_points=show_points,
            point_seed_column=None,
            show_legend=True,
            show_xlabels=True,
            ylabel="RMSE",
            title=figure_title,
        )

        if (
            full_y_tick_step is not None
            and full_y_tick_step > 0
        ):
            ax.yaxis.set_major_locator(
                MultipleLocator(float(full_y_tick_step))
            )

    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(fig_width, 8.6),
            sharex=True,
            gridspec_kw={
                "height_ratios": [1.0, 2.4],
            },
        )

        ax_full, ax_zoom = axes

        fig.suptitle(
            f"{figure_title}\n"
            "Top: full scale, Bottom: zoomed region",
            y=0.985,
            fontsize=13,
        )

        _draw_grouped_boxplot_on_axis(
            ax_full,
            grouped=grouped,
            missingness_types=missingness_types,
            show_points=False,
            point_seed_column=None,
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
            point_seed_column=None,
            show_legend=False,
            show_xlabels=True,
            ylabel="Zoomed RMSE",
            title=None,
        )

        ax_zoom.set_ylim(
            float(zoom_y_range[0]),
            float(zoom_y_range[1]),
        )

        if (
            full_y_tick_step is not None
            and full_y_tick_step > 0
        ):
            ax_full.yaxis.set_major_locator(
                MultipleLocator(float(full_y_tick_step))
            )

        if (
            zoom_y_tick_step is not None
            and zoom_y_tick_step > 0
        ):
            ax_zoom.yaxis.set_major_locator(
                MultipleLocator(float(zoom_y_tick_step))
            )

        ax_full.text(
            0.01,
            0.98,
            "All values shown",
            transform=ax_full.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )

    plt.tight_layout()

    fraction_token = _safe_fraction_token(fraction)

    output_png = (
        out_dir
        / f"{filename_prefix}_fraction_{fraction_token}.png"
    )

    output_pdf = (
        out_dir
        / f"{filename_prefix}_fraction_{fraction_token}.pdf"
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    if save_pdf:
        fig.savefig(
            output_pdf,
            bbox_inches="tight",
        )
    else:
        output_pdf = None

    if pdf_pages is not None:
        pdf_pages.savefig(
            fig,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    plt.close(fig)

    return output_png, output_pdf


# =====================================================================
# Combined-report title page
# =====================================================================
def _add_report_title_page(
    pdf: PdfPages,
    *,
    methods_to_compare: Sequence[str],
    study_names: Mapping[str, str],
    optimization_seed: int,
    hyperparameter_fractions: Sequence[float],
    hyperparameter_missingness_types: Sequence[str],
    final_fractions: Sequence[float],
    final_missingness_types: Sequence[str],
    seeds: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    lines = [
        "Imputation RMSE comparison report",
        "",
        "Panel family 1:",
        (
            "  Hyperparameter optimization trials at "
            f"fixed seed {optimization_seed}"
        ),
        f"  Fractions: {list(hyperparameter_fractions)}",
        (
            "  Missingness: "
            f"{list(hyperparameter_missingness_types)}"
        ),
        "",
        "Panel family 2:",
        (
            "  Final RMSE with frozen best hyperparameters "
            "across preprocessing/model seeds"
        ),
        f"  Fractions: {list(final_fractions)}",
        f"  Missingness: {list(final_missingness_types)}",
        f"  Seeds: {list(seeds)}",
        "",
        f"Methods: {list(methods_to_compare)}",
        "",
        "Exact Optuna studies:",
    ]

    for method, study_name in study_names.items():
        lines.append(f"  {method}: {study_name}")

    lines.extend(
        [
            "",
            "Notes:",
            "  No values are removed from full-scale panels.",
            (
                "  Repeated-seed boxes contain one final RMSE "
                "per requested seed."
            ),
        ]
    )

    if any(
        _canonical_method_name(method) == "MEAN"
        for method in methods_to_compare
    ):
        lines.extend(
            [
                (
                    "  Mean validation RMSE uses the same "
                    "validation-mask logic."
                ),
                (
                    "  Mean final RMSE is calculated directly "
                    "from each seeded corrupted matrix."
                ),
            ]
        )

    y = 0.96

    for index, line in enumerate(lines):
        is_title = index == 0

        ax.text(
            0.05,
            y,
            line,
            fontsize=18 if is_title else 10,
            fontweight=(
                "bold"
                if is_title or line.endswith(":")
                else "normal"
            ),
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

        y -= 0.050 if is_title else 0.027

    pdf.savefig(
        fig,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# Public comparison function
# =====================================================================
def compare_imputation_methods_seeded(
    *,
    methods_to_compare: Sequence[str],
    study_names: Mapping[str, str],
    columns_num: str,
    optimization_seed: int,
    hyperparameter_fractions: Sequence[float],
    hyperparameter_missingness_types: Sequence[str],
    final_fractions: Sequence[float],
    final_missingness_types: Sequence[str],
    seeds: Sequence[int],
    sort_by: str = "median",
    hyperparameter_zoom_y_range: Optional[
        Tuple[float, float]
    ] = None,
    seed_zoom_y_range: Optional[
        Tuple[float, float]
    ] = None,
    full_y_tick_step: Optional[float] = None,
    zoom_y_tick_step: Optional[float] = 0.01,
    hyperparameter_show_points: bool = False,
    seed_show_points: bool = True,
    mean_val_rate: float = 0.05,
    mean_validation_mode: str = "stratified_quantile",
    seed_result_csv_overrides: Optional[
        Mapping[str, str]
    ] = None,
    require_seed_csv_not_older_than_best_json: bool = True,
    output_dir: Optional[str] = None,
    show: bool = True,
    save_pdf: bool = True,
    save_combined_pdf: bool = True,
    combined_pdf_name: str = (
        "imputation_hyperparameter_and_seed_comparison.pdf"
    ),
    include_title_page: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Compare imputation methods in two complementary ways.

    1. Hyperparameter optimization at one fixed seed.
       Each box contains per-trial validation RMSE values.
       S1 uses a publication-sized figure for each fraction.

    2. Final reconstruction after hyperparameter freezing.
       Each box contains one final RMSE per requested seed.
       The existing final-seed plotting layout is retained.

    Exact study names are required. Studies are never selected
    using a loose substring or file modification time.

    Returns
    -------
    hyperparameter_values : pandas.DataFrame
    seed_values : pandas.DataFrame
    outputs : dict
        Paths to exported tables and figures.
    """
    if not methods_to_compare:
        raise ValueError(
            "methods_to_compare cannot be empty."
        )

    if not hyperparameter_fractions:
        raise ValueError(
            "hyperparameter_fractions cannot be empty."
        )

    if not hyperparameter_missingness_types:
        raise ValueError(
            "hyperparameter_missingness_types cannot be empty."
        )

    if not final_fractions:
        raise ValueError(
            "final_fractions cannot be empty."
        )

    if not final_missingness_types:
        raise ValueError(
            "final_missingness_types cannot be empty."
        )

    if not seeds:
        raise ValueError(
            "seeds cannot be empty."
        )

    hyperparameter_fractions = [
        float(value)
        for value in hyperparameter_fractions
    ]

    hyperparameter_missingness_types = _ordered_missingness(
        hyperparameter_missingness_types
    )

    final_fractions = [
        float(value)
        for value in final_fractions
    ]

    final_missingness_types = _ordered_missingness(
        final_missingness_types
    )

    seeds = [int(value) for value in seeds]

    main_dir = Path(experiments_dir(columns_num))

    if not main_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist:\n{main_dir}"
        )

    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            main_dir
            / "Evaluation"
            / "seeded_method_comparison"
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\nLoading fixed-seed hyperparameter optimization values..."
    )

    hyperparameter_values = load_hyperparameter_comparison_values(
        main_dir=main_dir,
        methods_to_compare=methods_to_compare,
        study_names=study_names,
        fractions=hyperparameter_fractions,
        missingness_types=hyperparameter_missingness_types,
        columns_num=columns_num,
        optimization_seed=int(optimization_seed),
        mean_val_rate=float(mean_val_rate),
        mean_validation_mode=str(mean_validation_mode),
    )

    print("Loading frozen-best repeated-seed values...")

    seed_values = load_seed_robustness_values(
        main_dir=main_dir,
        methods_to_compare=methods_to_compare,
        study_names=study_names,
        fractions=final_fractions,
        missingness_types=final_missingness_types,
        seeds=seeds,
        columns_num=columns_num,
        seed_result_csv_overrides=seed_result_csv_overrides,
        require_not_older_than_best_json=(
            require_seed_csv_not_older_than_best_json
        ),
    )

    hyperparameter_csv = (
        out_dir
        / "all_methods_hyperparameter_validation_rmse.csv"
    )

    seed_csv = (
        out_dir
        / "all_methods_final_rmse_by_seed.csv"
    )

    seed_summary_csv = (
        out_dir
        / "all_methods_final_rmse_by_seed_summary.csv"
    )

    hyperparameter_values.to_csv(
        hyperparameter_csv,
        index=False,
    )

    seed_values.to_csv(
        seed_csv,
        index=False,
    )

    seed_summary = (
        seed_values.groupby(
            ["method", "missingness", "fraction"],
            as_index=False,
        )["rmse"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    seed_summary["coefficient_of_variation"] = (
        seed_summary["std"] / seed_summary["mean"]
    )

    seed_summary.to_csv(
        seed_summary_csv,
        index=False,
    )

    print(
        f"Saved hyperparameter values: {hyperparameter_csv}"
    )
    print(
        f"Saved repeated-seed values: {seed_csv}"
    )
    print(
        f"Saved repeated-seed summary: {seed_summary_csv}"
    )

    outputs: Dict[str, Any] = {
        "hyperparameter_csv": str(hyperparameter_csv),
        "seed_csv": str(seed_csv),
        "seed_summary_csv": str(seed_summary_csv),
        "hyperparameter_plots": {},
        "seed_robustness_plots": {},
    }

    combined_pdf_path = out_dir / combined_pdf_name

    pdf_context = (
        PdfPages(combined_pdf_path)
        if save_combined_pdf
        else None
    )

    try:
        if pdf_context is not None and include_title_page:
            _add_report_title_page(
                pdf_context,
                methods_to_compare=methods_to_compare,
                study_names=study_names,
                optimization_seed=int(optimization_seed),
                hyperparameter_fractions=hyperparameter_fractions,
                hyperparameter_missingness_types=(
                    hyperparameter_missingness_types
                ),
                final_fractions=final_fractions,
                final_missingness_types=final_missingness_types,
                seeds=seeds,
            )

        # -------------------------------------------------------------
        # S1: one publication-sized figure for each tuning fraction.
        # -------------------------------------------------------------
        for fraction in hyperparameter_fractions:
            fraction_df = hyperparameter_values[
                np.isclose(
                    hyperparameter_values["fraction"].astype(float),
                    float(fraction),
                )
            ].copy()

            output_png, output_pdf = _plot_fraction_distribution(
                plot_df=fraction_df,
                methods_to_compare=methods_to_compare,
                fraction=fraction,
                missingness_types=hyperparameter_missingness_types,
                out_dir=out_dir,
                filename_prefix="hyperparameter_grouped_boxplot",
                figure_title=(
                    "Hyperparameter optimization validation RMSE, "
                    f"fixed seed = {int(optimization_seed)}, "
                    f"missing fraction = {fraction:g}"
                ),
                sort_by=sort_by,
                zoom_y_range=hyperparameter_zoom_y_range,
                full_y_tick_step=full_y_tick_step,
                zoom_y_tick_step=zoom_y_tick_step,
                show_points=hyperparameter_show_points,
                show=show,
                save_pdf=save_pdf,
                pdf_pages=pdf_context,
            )

            outputs["hyperparameter_plots"][fraction] = {
                "png": str(output_png),
                "pdf": (
                    str(output_pdf)
                    if output_pdf is not None
                    else None
                ),
            }

        # -------------------------------------------------------------
        # Final reconstruction: existing final-seed plot layout.
        # -------------------------------------------------------------
        for fraction in final_fractions:
            fraction_df = seed_values[
                np.isclose(
                    seed_values["fraction"].astype(float),
                    float(fraction),
                )
            ].copy()

            output_png, output_pdf = _plot_fraction_distribution(
                plot_df=fraction_df,
                methods_to_compare=methods_to_compare,
                fraction=fraction,
                missingness_types=final_missingness_types,
                out_dir=out_dir,
                filename_prefix="seed_robustness_grouped_boxplot",
                figure_title=(
                    "Final RMSE with frozen best hyperparameters "
                    f"across seeds {seeds}, "
                    f"missing fraction = {fraction:g}"
                ),
                sort_by=sort_by,
                zoom_y_range=seed_zoom_y_range,
                full_y_tick_step=full_y_tick_step,
                zoom_y_tick_step=zoom_y_tick_step,
                show_points=seed_show_points,
                show=show,
                save_pdf=save_pdf,
                pdf_pages=pdf_context,
            )

            outputs["seed_robustness_plots"][fraction] = {
                "png": str(output_png),
                "pdf": (
                    str(output_pdf)
                    if output_pdf is not None
                    else None
                ),
            }

    finally:
        if pdf_context is not None:
            pdf_context.close()

    if save_combined_pdf:
        outputs["combined_pdf"] = str(combined_pdf_path)
        print(
            f"Saved combined report: {combined_pdf_path}"
        )

    return hyperparameter_values, seed_values, outputs


# ---------------------------------------------------------------------
# Public name used by your existing runner.
# Keep this line at module level, outside all functions.
# ---------------------------------------------------------------------
compare_imputation_methods = compare_imputation_methods_seeded