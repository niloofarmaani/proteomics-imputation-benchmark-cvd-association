# imputation/iterative_imputer.py

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


ALLOWED_IMPUTATION_ORDERS = {
    "ascending",
    "descending",
    "roman",
    "arabic",
    "random",
}
ALLOWED_INITIAL_STRATEGIES = {
    "mean",
    "median",
    "most_frequent",
    "constant",
}
SUPPORTED_ESTIMATORS = {"BayesianRidge"}


def _parse_bool(value: object, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))

    token = str(value).strip().lower()

    if token in {"1", "true", "t", "yes", "y"}:
        return True

    if token in {"0", "false", "f", "no", "n"}:
        return False

    raise ValueError(
        f"{name} must be a boolean token such as true/false or 1/0. "
        f"Got: {value!r}"
    )


def _parse_optional_int(value: object, name: str) -> Optional[int]:
    if value is None:
        return None

    token = str(value).strip().lower()

    if token in {"none", "all", "null"}:
        return None

    parsed = int(float(token))

    if parsed < 1:
        raise ValueError(f"{name} must be at least 1 or None. Got: {parsed}")

    return parsed


def _load_and_validate_inputs(
    corrupted_data_path: Path,
    normalized_data_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not corrupted_data_path.exists():
        raise FileNotFoundError(
            f"Corrupted data file does not exist:\n{corrupted_data_path}"
        )

    if not normalized_data_path.exists():
        raise FileNotFoundError(
            f"Normalized/reference data file does not exist:\n"
            f"{normalized_data_path}"
        )

    corrupted_df = pd.read_csv(corrupted_data_path)
    reference_df = pd.read_csv(normalized_data_path)

    if corrupted_df.shape[1] < 3:
        raise ValueError(
            "Corrupted data must contain eid, label, and at least one protein "
            f"column. Observed shape: {corrupted_df.shape}"
        )

    if reference_df.shape[1] < 3:
        raise ValueError(
            "Reference data must contain eid, label, and at least one protein "
            f"column. Observed shape: {reference_df.shape}"
        )

    expected_metadata = ["eid", "label"]

    if list(corrupted_df.columns[:2]) != expected_metadata:
        raise ValueError(
            "Corrupted data must have ['eid', 'label'] as its first two columns. "
            f"Observed: {list(corrupted_df.columns[:2])}"
        )

    if list(reference_df.columns[:2]) != expected_metadata:
        raise ValueError(
            "Reference data must have ['eid', 'label'] as its first two columns. "
            f"Observed: {list(reference_df.columns[:2])}"
        )

    if corrupted_df.shape != reference_df.shape:
        raise ValueError(
            "Corrupted and reference data have different shapes.\n"
            f"Corrupted shape: {corrupted_df.shape}\n"
            f"Reference shape: {reference_df.shape}"
        )

    if list(corrupted_df.columns) != list(reference_df.columns):
        raise ValueError(
            "Corrupted and reference data have different columns or column order."
        )

    if not corrupted_df["eid"].astype(str).equals(
        reference_df["eid"].astype(str)
    ):
        raise ValueError(
            "eid values or row order differ between corrupted and reference data."
        )

    if not corrupted_df["label"].astype(str).equals(
        reference_df["label"].astype(str)
    ):
        raise ValueError(
            "label values or row order differ between corrupted and reference data."
        )

    return corrupted_df, reference_df


def _make_estimator(estimator_name: str):
    estimator_name = str(estimator_name)

    if estimator_name == "BayesianRidge":
        return BayesianRidge()

    raise ValueError(
        f"Unsupported estimator_name={estimator_name!r}. "
        f"Supported estimators: {sorted(SUPPORTED_ESTIMATORS)}"
    )


def iterative_imputer(
    *,
    corrupted_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    max_iter: int,
    tol: float,
    n_nearest_features: Optional[int],
    imputation_order: str,
    initial_strategy: str,
    estimator_name: str,
    sample_posterior: bool,
    skip_complete: bool,
    seed: int,
    rmse_path: Path,
    imputed_data_path: Path,
    verbose: int = 0,
) -> tuple[pd.DataFrame, float]:
    """Run one strict, seeded scikit-learn IterativeImputer experiment."""
    max_iter = int(max_iter)
    tol = float(tol)
    seed = int(seed)
    imputation_order = str(imputation_order).lower()
    initial_strategy = str(initial_strategy).lower()
    estimator_name = str(estimator_name)
    sample_posterior = bool(sample_posterior)
    skip_complete = bool(skip_complete)
    verbose = int(verbose)

    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1. Got: {max_iter}")

    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"tol must be finite and positive. Got: {tol}")

    if imputation_order not in ALLOWED_IMPUTATION_ORDERS:
        raise ValueError(
            f"Unsupported imputation_order={imputation_order!r}. Allowed values: "
            f"{sorted(ALLOWED_IMPUTATION_ORDERS)}"
        )

    if initial_strategy not in ALLOWED_INITIAL_STRATEGIES:
        raise ValueError(
            f"Unsupported initial_strategy={initial_strategy!r}. Allowed values: "
            f"{sorted(ALLOWED_INITIAL_STRATEGIES)}"
        )

    estimator = _make_estimator(estimator_name)

    metadata_df = corrupted_df.iloc[:, :2].copy()
    feature_columns = list(corrupted_df.columns[2:])

    try:
        corrupted_numeric_df = corrupted_df[feature_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
        reference_numeric_df = reference_df[feature_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
    except Exception as exc:
        raise ValueError(
            "Protein feature columns must be numeric. "
            f"Original conversion error: {exc}"
        ) from exc

    X_corrupted = corrupted_numeric_df.to_numpy(dtype=float)
    X_reference = reference_numeric_df.to_numpy(dtype=float)

    if np.isinf(X_corrupted).any():
        raise ValueError("Corrupted protein matrix contains infinite values.")

    if not np.isfinite(X_reference).all():
        bad_count = int((~np.isfinite(X_reference)).sum())
        raise ValueError(
            "Reference protein matrix must be complete and finite. "
            f"Found {bad_count} non-finite cells."
        )

    missing_mask = np.isnan(X_corrupted)
    n_missing_cells = int(missing_mask.sum())

    if n_missing_cells == 0:
        raise ValueError(
            "The corrupted protein matrix contains no missing cells. "
            "IterativeImputer cannot be evaluated without missing targets."
        )

    observed_per_feature = np.sum(~missing_mask, axis=0)
    all_missing_features = np.where(observed_per_feature == 0)[0]

    if len(all_missing_features) > 0:
        bad_names = [feature_columns[index] for index in all_missing_features[:20]]
        raise ValueError(
            "IterativeImputer cannot fit protein columns that are entirely missing. "
            f"Affected columns, first 20: {bad_names}"
        )

    n_features = X_corrupted.shape[1]

    if n_nearest_features is not None:
        n_nearest_features = int(n_nearest_features)

        if n_nearest_features < 1:
            raise ValueError(
                "n_nearest_features must be at least 1 or None. "
                f"Got: {n_nearest_features}"
            )

        if n_nearest_features >= n_features:
            raise ValueError(
                "n_nearest_features must be smaller than the number of protein "
                f"features ({n_features}). Got: {n_nearest_features}"
            )

    imputer = IterativeImputer(
        estimator=estimator,
        missing_values=np.nan,
        sample_posterior=sample_posterior,
        max_iter=max_iter,
        tol=tol,
        n_nearest_features=n_nearest_features,
        initial_strategy=initial_strategy,
        imputation_order=imputation_order,
        skip_complete=skip_complete,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=verbose,
        random_state=seed,
        add_indicator=False,
    )

    print("\n" + "=" * 80)
    print("IterativeImputer run")
    print("=" * 80)
    print(f"matrix shape: {X_corrupted.shape}")
    print(f"missing protein cells: {n_missing_cells}")
    print(f"estimator: {estimator_name}")
    print(f"max_iter: {max_iter}")
    print(f"tol: {tol}")
    print(f"n_nearest_features: {n_nearest_features}")
    print(f"imputation_order: {imputation_order}")
    print(f"initial_strategy: {initial_strategy}")
    print(f"sample_posterior: {sample_posterior}")
    print(f"skip_complete: {skip_complete}")
    print(f"seed: {seed}")
    print("=" * 80)

    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        transformed = imputer.fit_transform(X_corrupted)
    fit_seconds = float(time.perf_counter() - fit_start)

    warning_messages = [str(record.message) for record in caught_warnings]
    convergence_warning_count = sum(
        1
        for record in caught_warnings
        if issubclass(record.category, ConvergenceWarning)
    )

    for record in caught_warnings:
        print(
            f"WARNING [{record.category.__name__}]: {record.message}",
            flush=True,
        )

    if transformed.shape != X_corrupted.shape:
        raise RuntimeError(
            "IterativeImputer changed the protein matrix shape. "
            f"Input: {X_corrupted.shape}, output: {transformed.shape}"
        )

    if not np.isfinite(transformed).all():
        bad_count = int((~np.isfinite(transformed)).sum())
        raise FloatingPointError(
            f"IterativeImputer output contains {bad_count} non-finite cells."
        )

    # Preserve every observed value exactly and replace only missing cells.
    X_final = X_corrupted.copy()
    X_final[missing_mask] = transformed[missing_mask]

    reference_missing = X_reference[missing_mask]
    imputed_missing = X_final[missing_mask]

    rmse = float(
        np.sqrt(np.mean((reference_missing - imputed_missing) ** 2))
    )

    if not np.isfinite(rmse):
        raise FloatingPointError(
            f"Computed IterativeImputer RMSE is non-finite: {rmse}"
        )

    n_iter_actual = int(getattr(imputer, "n_iter_", max_iter))
    converged_without_warning = convergence_warning_count == 0

    imputed_numeric_df = pd.DataFrame(
        X_final,
        columns=feature_columns,
        index=corrupted_df.index,
    )
    imputed_df = pd.concat(
        [
            metadata_df.reset_index(drop=True),
            imputed_numeric_df.reset_index(drop=True),
        ],
        axis=1,
    )

    if list(imputed_df.columns) != list(corrupted_df.columns):
        raise RuntimeError(
            "Internal error: imputed output columns do not match input columns."
        )

    rmse_path = Path(rmse_path).expanduser().resolve()
    imputed_data_path = Path(imputed_data_path).expanduser().resolve()
    rmse_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_data_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "rmse_iterative": [rmse],
            "max_iter": [max_iter],
            "n_iter_actual": [n_iter_actual],
            "tol": [tol],
            "n_nearest_features": [
                "all" if n_nearest_features is None else n_nearest_features
            ],
            "imputation_order": [imputation_order],
            "initial_strategy": [initial_strategy],
            "estimator_name": [estimator_name],
            "sample_posterior": [sample_posterior],
            "skip_complete": [skip_complete],
            "seed": [seed],
            "n_missing_cells": [n_missing_cells],
            "converged_without_warning": [converged_without_warning],
            "warning_count": [len(warning_messages)],
            "convergence_warning_count": [convergence_warning_count],
            "warnings_json": [json.dumps(warning_messages)],
            "fit_seconds": [fit_seconds],
            "sklearn_version": [sklearn.__version__],
        }
    ).to_csv(rmse_path, index=False)
    imputed_df.to_csv(imputed_data_path, index=False)

    print(f"Actual completed rounds: {n_iter_actual}")
    print(f"Fit time in seconds: {fit_seconds:.3f}")
    print(f"Converged without warning: {converged_without_warning}")
    print(f"IterativeImputer RMSE: {rmse:.8f}")
    print(f"RMSE saved to: {rmse_path}")
    print(f"Imputed data saved to: {imputed_data_path}")

    return imputed_df, rmse


def main() -> None:
    if len(sys.argv) != 15:
        raise ValueError(
            "Expected exactly 14 command-line arguments:\n"
            "1 corrupted_data_path\n"
            "2 normalized_data_path\n"
            "3 max_iter\n"
            "4 tol\n"
            "5 n_nearest_features (integer or 'all')\n"
            "6 imputation_order\n"
            "7 initial_strategy\n"
            "8 estimator_name\n"
            "9 sample_posterior\n"
            "10 skip_complete\n"
            "11 seed\n"
            "12 rmse_path\n"
            "13 imputed_data_path\n"
            "14 verbose level\n"
            f"Received {len(sys.argv) - 1} arguments."
        )

    corrupted_path = Path(sys.argv[1]).expanduser().resolve()
    normalized_path = Path(sys.argv[2]).expanduser().resolve()
    corrupted_frame, reference_frame = _load_and_validate_inputs(
        corrupted_data_path=corrupted_path,
        normalized_data_path=normalized_path,
    )

    iterative_imputer(
        corrupted_df=corrupted_frame,
        reference_df=reference_frame,
        max_iter=int(sys.argv[3]),
        tol=float(sys.argv[4]),
        n_nearest_features=_parse_optional_int(
            sys.argv[5],
            "n_nearest_features",
        ),
        imputation_order=str(sys.argv[6]),
        initial_strategy=str(sys.argv[7]),
        estimator_name=str(sys.argv[8]),
        sample_posterior=_parse_bool(
            sys.argv[9],
            "sample_posterior",
        ),
        skip_complete=_parse_bool(
            sys.argv[10],
            "skip_complete",
        ),
        seed=int(sys.argv[11]),
        rmse_path=Path(sys.argv[12]),
        imputed_data_path=Path(sys.argv[13]),
        verbose=int(sys.argv[14]),
    )


if __name__ == "__main__":
    main()
