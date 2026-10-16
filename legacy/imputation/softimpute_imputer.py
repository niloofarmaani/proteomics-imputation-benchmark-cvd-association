# imputation/softimpute_imputer.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


def _check_required_columns(df: pd.DataFrame, file_label: str) -> None:
    if df.shape[1] < 3:
        raise ValueError(
            f"{file_label} must contain at least eid, label, and one protein column. "
            f"Observed shape: {df.shape}"
        )

    expected_first_two = ["eid", "label"]
    observed_first_two = list(df.columns[:2])

    if observed_first_two != expected_first_two:
        raise ValueError(
            f"{file_label} must have first two columns ['eid', 'label'].\n"
            f"Observed first two columns: {observed_first_two}"
        )


def _validate_input_frames(df_with_na: pd.DataFrame, original_df: pd.DataFrame) -> None:
    _check_required_columns(df_with_na, "Corrupted data")
    _check_required_columns(original_df, "Normalized/reference data")

    if df_with_na.shape != original_df.shape:
        raise ValueError(
            "Corrupted data and normalized/reference data have different shapes.\n"
            f"Corrupted shape: {df_with_na.shape}\n"
            f"Reference shape: {original_df.shape}"
        )

    if list(df_with_na.columns) != list(original_df.columns):
        raise ValueError(
            "Corrupted data and normalized/reference data have different columns or column order."
        )

    if not df_with_na["eid"].astype(str).equals(original_df["eid"].astype(str)):
        raise ValueError("eid column does not match between corrupted and normalized/reference data.")

    if not df_with_na["label"].astype(str).equals(original_df["label"].astype(str)):
        raise ValueError("label column does not match between corrupted and normalized/reference data.")


def _read_inputs(corrupted_data_path: str, normalized_data_path: str):
    corrupted_data_path = Path(corrupted_data_path).expanduser().resolve()
    normalized_data_path = Path(normalized_data_path).expanduser().resolve()

    if not corrupted_data_path.exists():
        raise FileNotFoundError(f"Corrupted data file not found: {corrupted_data_path}")

    if not normalized_data_path.exists():
        raise FileNotFoundError(f"Normalized/reference data file not found: {normalized_data_path}")

    df_with_na = pd.read_csv(corrupted_data_path)
    original_df = pd.read_csv(normalized_data_path)

    _validate_input_frames(df_with_na, original_df)

    return df_with_na, original_df


def _validate_numeric_matrix(X: np.ndarray, label: str) -> None:
    if X.ndim != 2:
        raise ValueError(f"{label} must be a 2D matrix. Observed ndim={X.ndim}")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{label} has an empty dimension. Shape={X.shape}")


def softimpute_imputer(
    df_with_na: pd.DataFrame,
    original_df: pd.DataFrame,
    rank_max: int,
    maxit: int,
    seed: int,
    rmse_path: str,
    imputed_data_path: str,
):
    """
    Strict seeded low-rank SoftImpute-style imputation.

    This implementation:
      - protects eid and label
      - imputes only protein columns
      - evaluates only cells missing in the corrupted protein matrix
      - fails if required inputs are inconsistent
      - fails if no missing protein cells are present
      - uses deterministic ARPACK initialization through v0
    """

    rank_max = int(rank_max)
    maxit = int(maxit)
    seed = int(seed)

    if rank_max < 1:
        raise ValueError(f"rank_max must be >= 1. Got {rank_max}")

    if maxit < 1:
        raise ValueError(f"maxit must be >= 1. Got {maxit}")

    rmse_path = Path(rmse_path).expanduser().resolve()
    imputed_data_path = Path(imputed_data_path).expanduser().resolve()

    rmse_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_data_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n================ SoftImpute-style imputation ================")
    print(f"rank_max: {rank_max}")
    print(f"maxit: {maxit}")
    print(f"seed: {seed}")
    print("=============================================================\n")

    metadata_cols = ["eid", "label"]
    feature_cols = list(df_with_na.columns[2:])

    if len(feature_cols) == 0:
        raise ValueError("No protein feature columns found after eid and label.")

    X = df_with_na[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X_orig = original_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    _validate_numeric_matrix(X, "Corrupted protein matrix")
    _validate_numeric_matrix(X_orig, "Reference protein matrix")

    if X.shape != X_orig.shape:
        raise ValueError(
            f"Corrupted and reference protein matrices have different shapes: {X.shape} vs {X_orig.shape}"
        )

    print("Protein matrix shape:", X.shape)

    nan_mask = np.isnan(X)
    n_missing = int(nan_mask.sum())

    print("Total missing protein cells to impute/score:", n_missing)

    if n_missing == 0:
        raise ValueError(
            "No missing protein cells found in corrupted data. "
            "SoftImpute evaluation requires artificially missing entries."
        )

    reference_values_at_missing = X_orig[nan_mask]

    if not np.isfinite(reference_values_at_missing).all():
        bad = int((~np.isfinite(reference_values_at_missing)).sum())
        raise ValueError(
            f"Reference values at missing positions contain {bad} non-finite values. "
            "This indicates a preprocessing or evaluation-mask mismatch."
        )

    min_dim = min(X.shape)

    if rank_max >= min_dim:
        raise ValueError(
            f"rank_max={rank_max} must be smaller than min(matrix shape)={min_dim}. "
            "Use a smaller rank in the SoftImpute search space."
        )

    col_means = np.nanmean(X, axis=0)

    if not np.isfinite(col_means).all():
        bad_cols = np.where(~np.isfinite(col_means))[0]
        raise ValueError(
            f"Some protein columns are entirely missing in the corrupted matrix. "
            f"Bad column indices, first 20: {bad_cols[:20].tolist()}"
        )

    X_filled = X.copy()

    for j in range(X.shape[1]):
        X_filled[nan_mask[:, j], j] = col_means[j]

    if not np.isfinite(X_filled).all():
        raise ValueError("Initial column-mean-filled matrix contains non-finite values.")

    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=min_dim)

    for iteration in range(maxit):
        try:
            U, s, Vt = svds(X_filled, k=rank_max, v0=v0)
        except Exception as e:
            raise RuntimeError(
                f"scipy.sparse.linalg.svds failed at iteration {iteration + 1}. "
                f"rank_max={rank_max}, matrix_shape={X_filled.shape}. Error: {e}"
            )

        idx = np.argsort(s)[::-1]
        U = U[:, idx]
        s = s[idx]
        Vt = Vt[idx, :]

        low_rank = U @ np.diag(s) @ Vt

        if not np.isfinite(low_rank).all():
            raise FloatingPointError(
                f"Low-rank reconstruction contains non-finite values at iteration {iteration + 1}."
            )

        X_filled[nan_mask] = low_rank[nan_mask]

        if not np.isfinite(X_filled).all():
            raise FloatingPointError(
                f"Filled matrix contains non-finite values after iteration {iteration + 1}."
            )

    imputed_values_at_missing = X_filled[nan_mask]

    rmse = float(
        np.sqrt(
            np.mean(
                (reference_values_at_missing - imputed_values_at_missing) ** 2
            )
        )
    )

    if not np.isfinite(rmse):
        raise FloatingPointError(f"Computed RMSE is not finite: {rmse}")

    print(f"SoftImpute RMSE: {rmse}")

    pd.DataFrame({"rmse_softimpute": [rmse]}).to_csv(rmse_path, index=False)

    imputed_numeric_df = pd.DataFrame(X_filled, columns=feature_cols)

    imputed_df = pd.concat(
        [
            df_with_na[metadata_cols].reset_index(drop=True),
            imputed_numeric_df.reset_index(drop=True),
        ],
        axis=1,
    )

    if list(imputed_df.columns[:2]) != metadata_cols:
        raise RuntimeError("Internal error: imputed output does not preserve eid and label as first two columns.")

    imputed_df.to_csv(imputed_data_path, index=False)

    print(f"RMSE saved to: {rmse_path}")
    print(f"Imputed data saved to: {imputed_data_path}")


def main():
    if len(sys.argv) != 8:
        raise ValueError(
            "Expected exactly 7 arguments:\n"
            "1 corrupted_data_path\n"
            "2 normalized_data_path\n"
            "3 rank_max\n"
            "4 maxit\n"
            "5 seed\n"
            "6 rmse_path\n"
            "7 imputed_data_path\n"
            f"Received {len(sys.argv) - 1} arguments."
        )

    corrupted_data_path = sys.argv[1]
    normalized_data_path = sys.argv[2]
    rank_max = int(sys.argv[3])
    maxit = int(sys.argv[4])
    seed = int(sys.argv[5])
    rmse_path = sys.argv[6]
    imputed_data_path = sys.argv[7]

    df_with_na, original_df = _read_inputs(
        corrupted_data_path=corrupted_data_path,
        normalized_data_path=normalized_data_path,
    )

    softimpute_imputer(
        df_with_na=df_with_na,
        original_df=original_df,
        rank_max=rank_max,
        maxit=maxit,
        seed=seed,
        rmse_path=rmse_path,
        imputed_data_path=imputed_data_path,
    )


if __name__ == "__main__":
    main()