# optimize_softimpute_seeded_pipeline.py

from __future__ import annotations

import json
from pathlib import Path

import config_paths


# =============================================================================
# Set paths before project imports
# =============================================================================
config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")


from Optimization.SoftImpute_bayes_opt import run_softimpute_bayes_opt
from Optimization.run_final_with_best import run_final_with_best
from Evaluation.softimpute_eval import softimpute_eval


# =============================================================================
# Main configuration
# =============================================================================
DATA_FILE = "/scratch/biobank/niloo/mayoVasc/cureASciD_01_data_olinkTarget96_20241030.csv"

COLUMNS_NUM = "all"

# Use a new study name because older Optuna studies do not contain per-condition score_rows.
STUDY_NAME = "SoftImpute_balanced_MAR_MNAR_10_30_seeded_v1"

OPTIMIZATION_SEED = 42

FINAL_SEEDS = [
    42,
    100,
    2024,
    777,
    999,
]

TUNING_FRACTIONS = [
    0.1,
    0.3,
]

TUNING_MISSINGNESS_TYPES = [
    "MAR",
    "MNAR",
]

FINAL_FRACTIONS = [
    0.01,
    0.1,
    0.3,
    0.5,
]

FINAL_MISSINGNESS_TYPES = [
    "MCAR",
    "MAR",
    "MNAR",
]

N_TRIALS = 15

VAL_RATE = 0.05
VALIDATION_MODE = "stratified_quantile"

RUN_OPTIMIZATION = True
RUN_FINAL_BENCHMARK = True
RUN_EVALUATION = True

# Set this to True to also generate the cleaner main-figure version without 1 percent.
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


def load_best_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"\nBest-params JSON not found:\n{path}\n\n"
            "Run Bayesian optimization first, or check STUDY_NAME."
        )

    with open(path, "r") as f:
        payload = json.load(f)

    if "params" not in payload:
        raise KeyError(
            f"Best-params JSON does not contain key 'params'.\nFile: {path}"
        )

    params = payload["params"]

    required = {"rank_max", "maxit"}
    missing = required - set(params.keys())

    if missing:
        raise KeyError(
            f"SoftImpute best params are missing required keys: {sorted(missing)}\n"
            f"File: {path}\n"
            f"params: {params}"
        )

    return {
        "rank_max": int(params["rank_max"]),
        "maxit": int(params["maxit"]),
    }


def assert_data_file_exists() -> None:
    data_path = Path(DATA_FILE)

    if not data_path.exists():
        raise FileNotFoundError(
            f"\nDATA_FILE does not exist:\n{data_path}"
        )


def assert_seeded_preprocessing_files_exist(
    *,
    fractions,
    missingness_types,
    seeds,
    context_name: str,
) -> None:
    """
    Strictly verify that corrected seeded preprocessing already created
    the exact files required by this pipeline.
    """
    base_dir = experiment_dir()
    missing_files = []

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            folder = base_dir / f"fraction_{fraction}_missingness_{missingness}"

            normalized_path = folder / f"normalized_data_col_{COLUMNS_NUM}.csv"

            if not normalized_path.exists():
                missing_files.append(str(normalized_path))

            for seed in seeds:
                seed = int(seed)

                corrupted_path = (
                    folder
                    / f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{COLUMNS_NUM}_seed_{seed}.csv"
                )

                if not corrupted_path.exists():
                    missing_files.append(str(corrupted_path))

    if missing_files:
        preview = "\n".join(missing_files[:50])

        raise FileNotFoundError(
            f"\nMissing required seeded preprocessing files for: {context_name}\n\n"
            f"First missing files:\n{preview}\n\n"
            "Run corrected preprocessing first for these exact fractions, "
            "missingness types, and seeds."
        )


def assert_softimpute_final_outputs_exist(
    *,
    fractions,
    missingness_types,
    seeds,
    best_params: dict,
) -> None:
    """
    Verify that final SoftImpute outputs exist for all requested settings.
    """
    base_dir = experiment_dir()
    missing_files = []

    rank_max = int(best_params["rank_max"])
    maxit = int(best_params["maxit"])

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            folder = base_dir / f"fraction_{fraction}_missingness_{missingness}"

            for seed in seeds:
                seed = int(seed)

                rmse_path = (
                    folder
                    / f"RMSE_softImpute_col_{COLUMNS_NUM}_rank_{rank_max}_maxit_{maxit}_seed_{seed}.csv"
                )

                imputed_path = (
                    folder
                    / f"imputed_softImpute_col_{COLUMNS_NUM}_rank_{rank_max}_maxit_{maxit}_seed_{seed}.csv"
                )

                if not rmse_path.exists():
                    missing_files.append(str(rmse_path))

                if not imputed_path.exists():
                    missing_files.append(str(imputed_path))

    if missing_files:
        preview = "\n".join(missing_files[:50])

        raise FileNotFoundError(
            "\nFinal SoftImpute benchmark did not create all expected outputs.\n\n"
            f"First missing files:\n{preview}\n\n"
            "Check run_final_with_best, Optimization/common.py, and imputation/Softimpute.py."
        )


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    print("\n" + "=" * 100)
    print("SOFTIMPUTE SEEDED PIPELINE")
    print("=" * 100)
    print(f"Experiment directory: {experiment_dir()}")
    print(f"Study name: {STUDY_NAME}")
    print(f"Optimization seed: {OPTIMIZATION_SEED}")
    print(f"Final seeds: {FINAL_SEEDS}")
    print(f"Tuning fractions: {TUNING_FRACTIONS}")
    print(f"Tuning missingness types: {TUNING_MISSINGNESS_TYPES}")
    print(f"Final fractions: {FINAL_FRACTIONS}")
    print(f"Final missingness types: {FINAL_MISSINGNESS_TYPES}")
    print("=" * 100)

    assert_data_file_exists()

    # -------------------------------------------------------------------------
    # Phase 0: strict check for tuning inputs
    # -------------------------------------------------------------------------
    print("\n--- Phase 0: Checking seeded preprocessing files for Optuna tuning ---")

    assert_seeded_preprocessing_files_exist(
        fractions=TUNING_FRACTIONS,
        missingness_types=TUNING_MISSINGNESS_TYPES,
        seeds=[OPTIMIZATION_SEED],
        context_name="SoftImpute Optuna tuning",
    )

    # -------------------------------------------------------------------------
    # Phase 1: Bayesian optimization
    # -------------------------------------------------------------------------
    if RUN_OPTIMIZATION:
        print("\n--- Phase 1: Optuna tuning for SoftImpute ---")

        best_payload, study = run_softimpute_bayes_opt(
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
            verbose_method=True,
        )

        print("\nBest optimizer payload:")
        print(best_payload)

    else:
        print("\n--- Phase 1 skipped: RUN_OPTIMIZATION=False ---")

    # -------------------------------------------------------------------------
    # Phase 2: final benchmark with frozen best hyperparameters over seeds
    # -------------------------------------------------------------------------
    bj_path = best_json_path()
    params = load_best_params(bj_path)

    print("\nBest SoftImpute parameters:")
    print(params)
    print("\nBest JSON path:")
    print(bj_path)

    if RUN_FINAL_BENCHMARK:
        print("\n--- Phase 2: Checking seeded preprocessing files for final benchmark ---")

        assert_seeded_preprocessing_files_exist(
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            context_name="SoftImpute repeated final benchmark",
        )

        print("\n--- Phase 3: Final benchmark with frozen best SoftImpute hyperparameters ---")

        for current_seed in FINAL_SEEDS:
            current_seed = int(current_seed)

            print("\n" + "-" * 100)
            print(f"Running final SoftImpute benchmark for seed={current_seed}")
            print("-" * 100)

            run_final_with_best(
                best_json_path=str(bj_path),
                file_path=DATA_FILE,
                columns_num=COLUMNS_NUM,
                fractions=FINAL_FRACTIONS,
                missingness_types=FINAL_MISSINGNESS_TYPES,
                seed=current_seed,
                verbose_method=True,
            )

        print("\n--- Phase 4: Checking final SoftImpute outputs ---")

        assert_softimpute_final_outputs_exist(
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            best_params=params,
        )

        print("All expected final SoftImpute outputs exist.")

    else:
        print("\n--- Phase 2/3 skipped: RUN_FINAL_BENCHMARK=False ---")

    # -------------------------------------------------------------------------
    # Phase 5: evaluation plots
    # -------------------------------------------------------------------------
    if RUN_EVALUATION:
        print("\n--- Phase 5: SoftImpute evaluation plots, all fractions ---")

        outputs = softimpute_eval(
            columns_num=COLUMNS_NUM,
            study_name=STUDY_NAME,
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            show=True,
            strict=True,
            require_condition_scores=True,
        )

        print("\nSoftImpute evaluator outputs:")
        print(outputs)

        if RUN_MAINFIG_EVALUATION:
            print("\n--- Phase 6: SoftImpute evaluation plots, main-figure fractions ---")

            mainfig_output_dir = (
                experiment_dir()
                / "Evaluation"
                / "SoftImpute_seeded_pipeline_mainfig"
            )

            outputs_mainfig = softimpute_eval(
                columns_num=COLUMNS_NUM,
                study_name=STUDY_NAME,
                fractions=[0.1, 0.3, 0.5],
                missingness_types=FINAL_MISSINGNESS_TYPES,
                seeds=FINAL_SEEDS,
                output_dir=str(mainfig_output_dir),
                show=True,
                strict=True,
                require_condition_scores=True,
            )

            print("\nSoftImpute main-figure evaluator outputs:")
            print(outputs_mainfig)

    else:
        print("\n--- Phase 5 skipped: RUN_EVALUATION=False ---")

    print("\n" + "=" * 100)
    print("SoftImpute seeded pipeline completed.")
    print("=" * 100)


if __name__ == "__main__":
    main()