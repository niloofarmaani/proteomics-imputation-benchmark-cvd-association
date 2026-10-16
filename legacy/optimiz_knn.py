# optimize_knn_seeded_pipeline.py

from pathlib import Path
import config_paths


# =============================================================================
# Configuration
# =============================================================================
config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

from Optimization.run_bayes_opt import run_bayes_opt
from Optimization.run_final_with_best import run_final_with_best


DATA_FILE = "/scratch/biobank/niloo/mayoVasc/cureASciD_01_data_olinkTarget96_20241030.csv"

COLUMNS_NUM = "all"

# IMPORTANT:
# Use a new study name because old studies do not contain per-condition score_rows.
STUDY_NAME = "KNN_balanced_MAR_MNAR_10_30_seeded_v1"

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

N_TRIALS = 20

VAL_RATE = 0.05
VALIDATION_MODE = "stratified_quantile"

RUN_OPTIMIZATION = True
RUN_FINAL_BENCHMARK = True


# =============================================================================
# Strict checks
# =============================================================================
def experiment_dir() -> Path:
    return Path(config_paths.experiments_dir(COLUMNS_NUM))


def assert_seeded_preprocessing_files_exist(
    *,
    fractions,
    missingness_types,
    seeds,
    context_name: str,
) -> None:
    """
    Strictly verify that preprocessing generated the exact seeded corrupted files
    required by the imputation code.

    This intentionally fails if files are missing.
    """
    base_dir = experiment_dir()
    missing_files = []

    for fraction in fractions:
        for missingness in missingness_types:
            folder = base_dir / f"fraction_{fraction}_missingness_{missingness}"

            normalized_path = folder / f"normalized_data_col_{COLUMNS_NUM}.csv"

            if not normalized_path.exists():
                missing_files.append(str(normalized_path))

            for seed in seeds:
                corrupted_path = (
                    folder
                    / f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{COLUMNS_NUM}_seed_{seed}.csv"
                )

                if not corrupted_path.exists():
                    missing_files.append(str(corrupted_path))

    if missing_files:
        preview = "\n".join(missing_files[:40])

        raise FileNotFoundError(
            f"\nMissing required seeded preprocessing files for: {context_name}\n\n"
            f"First missing files:\n{preview}\n\n"
            "Run your corrected preprocessing first for these exact fractions, "
            "missingness types, and seeds. The runner will not continue with "
            "missing or unseeded placeholder inputs."
        )


def assert_best_json_exists(best_json_path: Path) -> None:
    if not best_json_path.exists():
        raise FileNotFoundError(
            f"\nBest-params JSON not found:\n{best_json_path}\n\n"
            "Run Phase 1 optimization first, or check STUDY_NAME."
        )


# =============================================================================
# Main
# =============================================================================
def main():
    print("\n" + "=" * 100)
    print("KNN SEEDED PIPELINE")
    print("=" * 100)
    print(f"Experiment directory: {experiment_dir()}")
    print(f"Study name: {STUDY_NAME}")
    print(f"Optimization seed: {OPTIMIZATION_SEED}")
    print(f"Final seeds: {FINAL_SEEDS}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Phase 0: strict input checks for optimization seed
    # -------------------------------------------------------------------------
    print("\n--- Phase 0: Checking seeded preprocessing files for Optuna tuning ---")

    assert_seeded_preprocessing_files_exist(
        fractions=TUNING_FRACTIONS,
        missingness_types=TUNING_MISSINGNESS_TYPES,
        seeds=[OPTIMIZATION_SEED],
        context_name="KNN Optuna tuning",
    )

    # -------------------------------------------------------------------------
    # Phase 1: Bayesian optimization
    # -------------------------------------------------------------------------
    if RUN_OPTIMIZATION:
        print("\n--- Phase 1: Optuna tuning for KNN ---")

        best_payload, study = run_bayes_opt(
            method="KNN",
            file_path=DATA_FILE,
            columns_num=COLUMNS_NUM,
            tuning_fractions=TUNING_FRACTIONS,
            tuning_missingness_types=TUNING_MISSINGNESS_TYPES,
            n_trials=N_TRIALS,
            seed=OPTIMIZATION_SEED,
            val_rate=VAL_RATE,
            validation_mode=VALIDATION_MODE,
            study_name=STUDY_NAME,
            verbose_method=True,
        )

        print("\nBest payload:")
        print(best_payload)

    else:
        print("\n--- Phase 1 skipped: RUN_OPTIMIZATION=False ---")

    # -------------------------------------------------------------------------
    # Phase 2: final benchmark with frozen best hyperparameter over seeds
    # -------------------------------------------------------------------------
    best_json_path = (
        experiment_dir()
        / "Optimization"
        / "best_params"
        / f"{STUDY_NAME}_best_params.json"
    )

    assert_best_json_exists(best_json_path)

    if RUN_FINAL_BENCHMARK:
        print("\n--- Phase 2: Checking seeded preprocessing files for final benchmark ---")

        assert_seeded_preprocessing_files_exist(
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            context_name="KNN repeated final benchmark",
        )

        print("\n--- Phase 3: Final benchmark with frozen best KNN hyperparameter ---")
        print(f"Using best JSON:\n{best_json_path}")

        for current_seed in FINAL_SEEDS:
            print("\n" + "-" * 100)
            print(f"Running final KNN benchmark for seed={current_seed}")
            print("-" * 100)

            run_final_with_best(
                best_json_path=str(best_json_path),
                file_path=DATA_FILE,
                columns_num=COLUMNS_NUM,
                fractions=FINAL_FRACTIONS,
                missingness_types=FINAL_MISSINGNESS_TYPES,
                seed=current_seed,
                verbose_method=True,
            )

    else:
        print("\n--- Phase 2/3 skipped: RUN_FINAL_BENCHMARK=False ---")

    print("\n" + "=" * 100)
    print("KNN seeded pipeline completed.")
    print("=" * 100)
    print("\nNext step:")
    print(
        "Run Evaluation.knn_eval.knn_eval(...) using the same STUDY_NAME and FINAL_SEEDS."
    )


if __name__ == "__main__":
    main()