# optimize_iterative_seeded_pipeline.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import config_paths


# =============================================================================
# Project configuration
# =============================================================================
# These are explicit runner inputs. Change them here when running a different
# project or dataset. The imputer itself contains no project-specific paths.
config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

from Evaluation.iterative_eval import iterative_eval
from imputation.iterative import iterative_param_id
from Optimization.Iterative_bayes_opt import run_iterative_bayes_opt
from Optimization.run_final_with_best import run_final_with_best


DATA_FILE = (
    "/scratch/biobank/niloo/mayoVasc/"
    "cureASciD_01_data_olinkTarget96_20241030.csv"
)
COLUMNS_NUM = "all"
STUDY_NAME = "Iterative_balanced_MAR_MNAR_10_30_seeded_v1"

OPTIMIZATION_SEED = 42
FINAL_SEEDS = [42, 100, 2024, 777, 999]

TUNING_FRACTIONS = [0.1, 0.3]
TUNING_MISSINGNESS_TYPES = ["MAR", "MNAR"]

FINAL_FRACTIONS = [0.01, 0.1, 0.3, 0.5]
FINAL_MISSINGNESS_TYPES = ["MCAR", "MAR", "MNAR"]

# The current IterativeImputer search space contains four max_iter choices.
# Repeated categorical proposals are allowed and are retained in the Optuna
# trial table. n_trials is interpreted by run_bayes_opt as the requested total
# number of completed trials, not as the number to append on every rerun.
N_TRIALS = 12
VAL_RATE = 0.05
VALIDATION_MODE = "stratified_quantile"

RUN_OPTIMIZATION = True
RUN_FINAL_BENCHMARK = True
RUN_EVALUATION = True
RUN_MAINFIG_EVALUATION = True


# =============================================================================
# Strict helpers
# =============================================================================
def experiment_dir() -> Path:
    return Path(config_paths.experiments_dir(COLUMNS_NUM))


def best_json_path() -> Path:
    return (
        experiment_dir()
        / "Optimization"
        / "best_params"
        / f"{STUDY_NAME}_best_params.json"
    )


def assert_data_file_exists() -> None:
    path = Path(DATA_FILE).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DATA_FILE does not exist:\n{path}")


def assert_seeded_preprocessing_files_exist(
    *,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    context_name: str,
) -> None:
    missing_paths = []
    base_dir = experiment_dir()

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist:\n{base_dir}\n"
            "Run preprocessing before this pipeline."
        )

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()
            folder = base_dir / (
                f"fraction_{fraction}_missingness_{missingness}"
            )
            reference_path = folder / f"normalized_data_col_{COLUMNS_NUM}.csv"

            if not reference_path.exists():
                missing_paths.append(str(reference_path))

            for seed in seeds:
                corrupted_path = folder / (
                    f"Corrupted_data_fraction_{fraction}_missingness_"
                    f"{missingness}_col_{COLUMNS_NUM}_seed_{int(seed)}.csv"
                )
                if not corrupted_path.exists():
                    missing_paths.append(str(corrupted_path))

    if missing_paths:
        preview = "\n".join(missing_paths[:50])
        raise FileNotFoundError(
            f"Missing required seeded preprocessing files for {context_name}.\n\n"
            f"First missing paths:\n{preview}\n\n"
            "The pipeline will not fall back to unseeded or alternative files."
        )


def _load_best_params(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Best-parameter JSON does not exist:\n{path}\n"
            "Run optimization first or verify STUDY_NAME."
        )

    with path.open("r") as handle:
        payload = json.load(handle)

    if payload.get("method") not in {"Iterative", "ITERATIVE", "IterativeImputer"}:
        raise ValueError(
            f"Best-parameter JSON method is not Iterative: {payload.get('method')!r}"
        )

    if "params" not in payload or not isinstance(payload["params"], dict):
        raise KeyError(f"Best-parameter JSON has no valid 'params' dictionary:\n{path}")

    required = {
        "max_iter",
        "tol",
        "n_nearest_features",
        "imputation_order",
        "initial_strategy",
        "estimator_name",
        "sample_posterior",
        "skip_complete",
    }
    missing = required - set(payload["params"])

    if missing:
        raise KeyError(
            f"Best IterativeImputer parameters are missing keys {sorted(missing)}.\n"
            f"File: {path}\nParameters: {payload['params']}"
        )

    return payload["params"]


def assert_condition_score_table_exists() -> None:
    path = (
        experiment_dir()
        / "Optimization"
        / "trial_tables"
        / f"{STUDY_NAME}_condition_scores.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Per-condition Optuna score table was not created:\n{path}"
        )


def assert_final_outputs_exist(
    *,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    seeds: Sequence[int],
    best_params: Dict[str, Any],
) -> None:
    param_id = iterative_param_id(
        max_iter=int(best_params["max_iter"]),
        tol=float(best_params["tol"]),
        n_nearest_features=best_params["n_nearest_features"],
        imputation_order=str(best_params["imputation_order"]),
        initial_strategy=str(best_params["initial_strategy"]),
        estimator_name=str(best_params["estimator_name"]),
        sample_posterior=bool(best_params["sample_posterior"]),
        skip_complete=bool(best_params["skip_complete"]),
    )

    missing_paths = []

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()
            folder = experiment_dir() / (
                f"fraction_{fraction}_missingness_{missingness}"
            )

            for seed in seeds:
                seed = int(seed)
                rmse_path = folder / (
                    f"RMSE_iterative_col_{COLUMNS_NUM}_params_{param_id}"
                    f"_seed_{seed}.csv"
                )
                imputed_path = folder / (
                    f"imputed_iterative_col_{COLUMNS_NUM}_params_{param_id}"
                    f"_seed_{seed}.csv"
                )

                if not rmse_path.exists():
                    missing_paths.append(str(rmse_path))
                if not imputed_path.exists():
                    missing_paths.append(str(imputed_path))

    if missing_paths:
        preview = "\n".join(missing_paths[:50])
        raise FileNotFoundError(
            "The repeated final benchmark did not create every required "
            f"IterativeImputer output.\n\nFirst missing paths:\n{preview}"
        )


# =============================================================================
# Pipeline
# =============================================================================
def main() -> None:
    print("\n" + "=" * 100)
    print("ITERATIVEIMPUTER SEEDED PIPELINE")
    print("=" * 100)
    print(f"Experiment directory: {experiment_dir()}")
    print(f"Study name: {STUDY_NAME}")
    print(f"Optimization seed: {OPTIMIZATION_SEED}")
    print(f"Final seeds: {FINAL_SEEDS}")
    print(f"Tuning panel: {TUNING_MISSINGNESS_TYPES} x {TUNING_FRACTIONS}")
    print(f"Final panel: {FINAL_MISSINGNESS_TYPES} x {FINAL_FRACTIONS}")
    print("=" * 100)

    assert_data_file_exists()

    print("\nPhase 0: validating optimization inputs")
    assert_seeded_preprocessing_files_exist(
        fractions=TUNING_FRACTIONS,
        missingness_types=TUNING_MISSINGNESS_TYPES,
        seeds=[OPTIMIZATION_SEED],
        context_name="IterativeImputer optimization",
    )

    if RUN_OPTIMIZATION:
        print("\nPhase 1: Bayesian optimization")
        best_payload, study = run_iterative_bayes_opt(
            file_path=DATA_FILE,
            columns_num=COLUMNS_NUM,
            tuning_fractions=TUNING_FRACTIONS,
            tuning_missingness_types=TUNING_MISSINGNESS_TYPES,
            n_trials=N_TRIALS,
            seed=OPTIMIZATION_SEED,
            val_rate=VAL_RATE,
            validation_mode=VALIDATION_MODE,
            study_name=STUDY_NAME,
            cleanup_tmp=True,
            n_jobs=1,
            verbose_method=True,
        )
        print("\nBest optimizer payload:")
        print(best_payload)
    else:
        print("\nPhase 1 skipped because RUN_OPTIMIZATION=False")

    assert_condition_score_table_exists()
    json_path = best_json_path()
    best_params = _load_best_params(json_path)

    print("\nFrozen best IterativeImputer parameters:")
    print(best_params)
    print(f"Best-parameter JSON: {json_path}")

    if RUN_FINAL_BENCHMARK:
        print("\nPhase 2: validating repeated-seed benchmark inputs")
        assert_seeded_preprocessing_files_exist(
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            context_name="IterativeImputer repeated final benchmark",
        )

        print("\nPhase 3: repeated final benchmark with frozen parameters")
        for current_seed in FINAL_SEEDS:
            print("\n" + "-" * 100)
            print(f"Running frozen IterativeImputer for seed={current_seed}")
            print("-" * 100)
            run_final_with_best(
                best_json_path=str(json_path),
                file_path=DATA_FILE,
                columns_num=COLUMNS_NUM,
                fractions=FINAL_FRACTIONS,
                missingness_types=FINAL_MISSINGNESS_TYPES,
                seed=int(current_seed),
                verbose_method=True,
            )

        print("\nPhase 4: validating all repeated final outputs")
        assert_final_outputs_exist(
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            best_params=best_params,
        )
        print("All expected IterativeImputer outputs exist.")
    else:
        print("\nPhases 2 through 4 skipped because RUN_FINAL_BENCHMARK=False")

    if RUN_EVALUATION:
        print("\nPhase 5: generating full evaluation report")
        outputs = iterative_eval(
            columns_num=COLUMNS_NUM,
            study_name=STUDY_NAME,
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            show=True,
            strict=True,
            require_condition_scores=True,
            show_full_hyperparameter_legend=True,
        )
        print(outputs)

        if RUN_MAINFIG_EVALUATION:
            print("\nPhase 6: generating main-figure report without 1% missingness")
            main_output_dir = (
                experiment_dir()
                / "Evaluation"
                / "Iterative_seeded_pipeline_mainfig"
            )
            main_outputs = iterative_eval(
                columns_num=COLUMNS_NUM,
                study_name=STUDY_NAME,
                fractions=[0.1, 0.3, 0.5],
                missingness_types=FINAL_MISSINGNESS_TYPES,
                seeds=FINAL_SEEDS,
                output_dir=str(main_output_dir),
                show=True,
                strict=True,
                require_condition_scores=True,
                show_full_hyperparameter_legend=True,
            )
            print(main_outputs)
    else:
        print("\nEvaluation skipped because RUN_EVALUATION=False")

    print("\n" + "=" * 100)
    print("IterativeImputer seeded pipeline completed")
    print("=" * 100)


if __name__ == "__main__":
    main()
