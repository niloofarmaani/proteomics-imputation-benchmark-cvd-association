# imputation/iterative.py

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import IMPUTATION_DIR, experiments_dir


def _float_token(value: float) -> str:
    token = f"{float(value):.12g}"
    return token.replace("-", "m").replace(".", "p")


def _bool_token(value: bool) -> str:
    return "1" if bool(value) else "0"


def _nearest_token(value: Optional[int]) -> str:
    return "all" if value is None else str(int(value))


def iterative_param_id(
    *,
    max_iter: int,
    tol: float,
    n_nearest_features: Optional[int],
    imputation_order: str,
    initial_strategy: str,
    estimator_name: str,
    sample_posterior: bool,
    skip_complete: bool,
) -> str:
    estimator_token = {"BayesianRidge": "br"}.get(str(estimator_name))

    if estimator_token is None:
        raise ValueError(
            f"Unsupported estimator_name for IterativeImputer: {estimator_name!r}"
        )

    return (
        f"mi{int(max_iter)}"
        f"_tol{_float_token(float(tol))}"
        f"_nnf{_nearest_token(n_nearest_features)}"
        f"_ord{str(imputation_order).lower()}"
        f"_init{str(initial_strategy).lower()}"
        f"_est{estimator_token}"
        f"_post{_bool_token(sample_posterior)}"
        f"_skip{_bool_token(skip_complete)}"
    )


def ITERATIVE_IMPUTER(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    max_iter,
    tol,
    n_nearest_features,
    imputation_order,
    initial_strategy,
    estimator_name,
    sample_posterior,
    skip_complete,
    seed=42,
    show_output=True,
    print_command=True,
    print_rmse_after_run=True,
):
    """Strict seeded wrapper for scikit-learn IterativeImputer."""
    raw_data_path = Path(file_path).expanduser().resolve()

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Input data file does not exist:\n{raw_data_path}")

    main_directory = Path(experiments_dir(columns_num))

    if not main_directory.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist:\n{main_directory}\n"
            "Run preprocessing before IterativeImputer."
        )

    backend_script = Path(IMPUTATION_DIR) / "iterative_imputer.py"

    if not backend_script.exists():
        raise FileNotFoundError(
            f"IterativeImputer backend script does not exist:\n{backend_script}"
        )

    max_iter = int(max_iter)
    tol = float(tol)
    seed = int(seed)
    n_nearest_features = (
        None if n_nearest_features is None else int(n_nearest_features)
    )
    imputation_order = str(imputation_order).lower()
    initial_strategy = str(initial_strategy).lower()
    estimator_name = str(estimator_name)
    sample_posterior = bool(sample_posterior)
    skip_complete = bool(skip_complete)

    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1. Got: {max_iter}")

    if tol <= 0:
        raise ValueError(f"tol must be positive. Got: {tol}")

    param_id = iterative_param_id(
        max_iter=max_iter,
        tol=tol,
        n_nearest_features=n_nearest_features,
        imputation_order=imputation_order,
        initial_strategy=initial_strategy,
        estimator_name=estimator_name,
        sample_posterior=sample_posterior,
        skip_complete=skip_complete,
    )

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()
            full_path = main_directory / (
                f"fraction_{fraction}_missingness_{missingness}"
            )

            if not full_path.exists():
                raise FileNotFoundError(
                    f"Required fraction/missingness directory does not exist:\n{full_path}"
                )

            corrupted_data_path = full_path / (
                f"Corrupted_data_fraction_{fraction}_missingness_{missingness}"
                f"_col_{columns_num}_seed_{seed}.csv"
            )
            normalized_data_path = (
                full_path / f"normalized_data_col_{columns_num}.csv"
            )

            if not corrupted_data_path.exists():
                raise FileNotFoundError(
                    "IterativeImputer could not find the exact seeded corrupted file:\n"
                    f"{corrupted_data_path}\n"
                    "Run preprocessing for this exact fraction, mechanism, and seed."
                )

            if not normalized_data_path.exists():
                raise FileNotFoundError(
                    f"Normalized/reference data file does not exist:\n{normalized_data_path}"
                )

            rmse_path = full_path / (
                f"RMSE_iterative_col_{columns_num}_params_{param_id}_seed_{seed}.csv"
            )
            imputed_path = full_path / (
                f"imputed_iterative_col_{columns_num}_params_{param_id}_seed_{seed}.csv"
            )

            command = [
                sys.executable,
                str(backend_script),
                str(corrupted_data_path),
                str(normalized_data_path),
                str(max_iter),
                str(tol),
                _nearest_token(n_nearest_features),
                imputation_order,
                initial_strategy,
                estimator_name,
                str(sample_posterior),
                str(skip_complete),
                str(seed),
                str(rmse_path),
                str(imputed_path),
                str(1 if show_output else 0),
            ]

            print("\n" + "=" * 80)
            print("Running IterativeImputer")
            print("=" * 80)
            print(f"fraction: {fraction}")
            print(f"missingness: {missingness}")
            print(f"seed: {seed}")
            print(f"parameter ID: {param_id}")

            if print_command:
                print("Command:")
                print(" ".join(command), flush=True)

            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=not show_output,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"IterativeImputer subprocess failed with return code "
                    f"{exc.returncode}."
                )

                if exc.stdout:
                    print("\nCaptured STDOUT:")
                    print(exc.stdout)

                if exc.stderr:
                    print("\nCaptured STDERR:")
                    print(exc.stderr)

                raise

            if not show_output:
                if result.stdout:
                    print("\nCaptured STDOUT:")
                    print(result.stdout)

                if result.stderr:
                    print("\nCaptured STDERR:")
                    print(result.stderr)

            if not rmse_path.exists():
                raise FileNotFoundError(
                    "IterativeImputer subprocess completed but the RMSE file was not "
                    f"created:\n{rmse_path}"
                )

            if not imputed_path.exists():
                raise FileNotFoundError(
                    "IterativeImputer subprocess completed but the imputed file was not "
                    f"created:\n{imputed_path}"
                )

            if print_rmse_after_run:
                try:
                    rmse_df = pd.read_csv(rmse_path)
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not read generated RMSE file:\n{rmse_path}\n"
                        f"Error: {exc}"
                    ) from exc

                if rmse_df.empty:
                    raise ValueError(f"Generated RMSE file is empty:\n{rmse_path}")

                print("\nGenerated IterativeImputer RMSE record:")
                print(rmse_df.to_string(index=False), flush=True)

    print("\n--- IterativeImputer pipeline complete ---")


iterative_imputer = ITERATIVE_IMPUTER
