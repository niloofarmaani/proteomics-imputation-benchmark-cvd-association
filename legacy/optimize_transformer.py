# optimize_transformer_seeded_pipeline.py

from __future__ import annotations

import json
from pathlib import Path

import config_paths


# =============================================================================
# Project configuration
# =============================================================================
config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

from Optimization.Transformer_bayes_opt import run_transformer_bayes_opt
from Optimization.run_final_with_best import run_final_with_best
from Evaluation.transformer_eval import transformer_eval
import Optimization.method_spaces as method_spaces


DATA_FILE = (
    "/scratch/biobank/niloo/mayoVasc/"
    "cureASciD_01_data_olinkTarget96_20241030.csv"
)
COLUMNS_NUM = "all"
STUDY_NAME = "Transformer_balanced_MAR_MNAR_10_30_seeded_v2"

OPTIMIZATION_SEED = 42
FINAL_SEEDS = [42, 100, 2024, 777, 999]

TUNING_FRACTIONS = [0.1, 0.3]
TUNING_MISSINGNESS_TYPES = ["MAR", "MNAR"]

FINAL_FRACTIONS = [0.01, 0.1, 0.3, 0.5]
FINAL_MISSINGNESS_TYPES = ["MCAR", "MAR", "MNAR"]

# The reduced search optimizes only learning_rate and d_model.
# All other Transformer settings are explicit fixed settings in method_spaces.py.
N_TRIALS = 6
EPOCHS = 100
PATIENCE = 12
WARMUP_EPOCHS = 0  # Accepted by the shared optimizer but not used by Transformer.
VAL_RATE = 0.05
VALIDATION_MODE = "stratified_quantile"

RUN_OPTIMIZATION = True
RUN_FINAL_BENCHMARK = True
RUN_EVALUATION = True
RUN_MAINFIG_EVALUATION = True


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
    path = Path(DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(f"DATA_FILE does not exist:\n{path}")


def assert_seeded_preprocessing_files_exist(
    *,
    fractions,
    missingness_types,
    seeds,
    context_name: str,
) -> None:
    missing_paths = []
    base_dir = experiment_dir()

    for fraction_value in fractions:
        fraction = float(fraction_value)

        for missingness_value in missingness_types:
            missingness = str(missingness_value).upper()
            folder = (
                base_dir
                / f"fraction_{fraction}_missingness_{missingness}"
            )
            reference_path = folder / f"normalized_data_col_{COLUMNS_NUM}.csv"

            if not reference_path.exists():
                missing_paths.append(str(reference_path))

            for seed_value in seeds:
                seed = int(seed_value)
                corrupted_path = (
                    folder
                    / (
                        f"Corrupted_data_fraction_{fraction}_missingness_{missingness}"
                        f"_col_{COLUMNS_NUM}_seed_{seed}.csv"
                    )
                )
                if not corrupted_path.exists():
                    missing_paths.append(str(corrupted_path))

    if missing_paths:
        preview = "\n".join(missing_paths[:50])
        raise FileNotFoundError(
            f"Missing required seeded preprocessing files for {context_name}.\n\n"
            f"First missing paths:\n{preview}\n\n"
            "Run corrected preprocessing for every requested condition and seed."
        )


def load_best_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Best-parameter JSON does not exist:\n{path}"
        )

    with path.open("r") as handle:
        payload = json.load(handle)

    if "params" not in payload or not isinstance(payload["params"], dict):
        raise KeyError(f"Best-parameter JSON lacks a valid params dictionary: {path}")

    required = {
        "batch_size",
        "learning_rate",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "pseudo_mask_rate",
        "weight_decay",
        "grad_clip",
        "internal_val_rate",
    }
    missing = required - set(payload["params"])

    if missing:
        raise KeyError(
            f"Transformer best parameters are missing keys: {sorted(missing)}\n"
            f"params={payload['params']}"
        )

    return payload["params"]


def _float_token(value: float) -> str:
    token = f"{float(value):.12g}"
    return token.replace("-", "m").replace(".", "p")


def transformer_param_id(params: dict) -> str:
    return (
        f"bs{int(params['batch_size'])}"
        f"_lr{_float_token(float(params['learning_rate']))}"
        f"_dm{int(params['d_model'])}"
        f"_nh{int(params['nhead'])}"
        f"_nl{int(params['num_layers'])}"
        f"_ff{int(params['dim_feedforward'])}"
        f"_do{_float_token(float(params['dropout']))}"
        f"_pm{_float_token(float(params['pseudo_mask_rate']))}"
        f"_wd{_float_token(float(params['weight_decay']))}"
        f"_gc{_float_token(float(params['grad_clip']))}"
        f"_iv{_float_token(float(params['internal_val_rate']))}"
    )


def assert_final_outputs_exist(*, best_params: dict) -> None:
    param_id = transformer_param_id(best_params)
    missing_paths = []
    base_dir = experiment_dir()

    for fraction in FINAL_FRACTIONS:
        for missingness in FINAL_MISSINGNESS_TYPES:
            folder = (
                base_dir
                / f"fraction_{float(fraction)}_missingness_{str(missingness).upper()}"
            )

            for seed in FINAL_SEEDS:
                rmse_path = (
                    folder
                    / (
                        f"RMSE_TRF_col_{COLUMNS_NUM}_params_{param_id}"
                        f"_seed_{int(seed)}.csv"
                    )
                )
                imputed_path = (
                    folder
                    / (
                        f"imputed_TRF_col_{COLUMNS_NUM}_params_{param_id}"
                        f"_seed_{int(seed)}.csv"
                    )
                )

                if not rmse_path.exists():
                    missing_paths.append(str(rmse_path))
                if not imputed_path.exists():
                    missing_paths.append(str(imputed_path))

    if missing_paths:
        preview = "\n".join(missing_paths[:50])
        raise FileNotFoundError(
            "The frozen-best Transformer benchmark did not create every expected output.\n\n"
            f"First missing paths:\n{preview}"
        )


def assert_transformer_search_space_is_current() -> None:
    """Fail before any expensive training if the wrong method_spaces.py is active."""
    expected_fixed = {
        "batch_size": 32,
        "nhead": 4,
        "num_layers": 1,
        "dropout": 0.10,
        "pseudo_mask_rate": 0.20,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "internal_val_rate": 0.05,
    }

    module_path = Path(method_spaces.__file__).resolve()
    expected_path = (Path(__file__).resolve().parent / "Optimization" / "method_spaces.py").resolve()

    print(f"Active method_spaces.py: {module_path}")

    if module_path != expected_path:
        raise RuntimeError(
            "Python imported method_spaces.py from the wrong location.\n"
            f"Expected: {expected_path}\n"
            f"Loaded:   {module_path}\n"
            "Run this script from the project root and remove stale duplicate modules."
        )

    active_fixed = getattr(method_spaces, "TRANSFORMER_FIXED_SETTINGS", None)

    if active_fixed != expected_fixed:
        raise RuntimeError(
            "The active Optimization/method_spaces.py does not contain the corrected "
            "reduced Transformer settings.\n"
            f"Expected: {expected_fixed}\n"
            f"Observed: {active_fixed}"
        )

    import optuna

    fixed_trial = optuna.trial.FixedTrial(
        {
            "d_model": 16,
            "learning_rate": 3e-4,
        }
    )
    params = method_spaces.suggest_params("Transformer", fixed_trial)

    expected_keys = {
        "batch_size",
        "learning_rate",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "pseudo_mask_rate",
        "weight_decay",
        "grad_clip",
        "internal_val_rate",
    }

    missing = expected_keys - set(params)
    unexpected = set(params) - expected_keys

    if missing or unexpected:
        raise RuntimeError(
            "Transformer search-space preflight failed.\n"
            f"Missing keys: {sorted(missing)}\n"
            f"Unexpected keys: {sorted(unexpected)}\n"
            f"Returned params: {params}"
        )

    if params["d_model"] not in (16, 32):
        raise RuntimeError(f"Unexpected d_model from reduced search: {params['d_model']}")

    if params["dim_feedforward"] != 2 * params["d_model"]:
        raise RuntimeError(
            "dim_feedforward must equal 2 * d_model in the reduced search."
        )

    print("Transformer search-space preflight passed.")
    print(f"Preflight params: {params}")


def main() -> None:
    assert_data_file_exists()
    assert_transformer_search_space_is_current()

    print("\n" + "=" * 100)
    print("TRANSFORMER SEEDED PIPELINE WITH REDUCED SEARCH")
    print("=" * 100)
    print(f"Study name: {STUDY_NAME}")
    print(f"Optimization trials: {N_TRIALS}")
    print(f"Maximum epochs: {EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print("Only learning_rate and d_model are optimized.")
    print("=" * 100)

    assert_seeded_preprocessing_files_exist(
        fractions=TUNING_FRACTIONS,
        missingness_types=TUNING_MISSINGNESS_TYPES,
        seeds=[OPTIMIZATION_SEED],
        context_name="Transformer Bayesian optimization",
    )

    if RUN_OPTIMIZATION:
        best_payload, _ = run_transformer_bayes_opt(
            file_path=DATA_FILE,
            columns_num=COLUMNS_NUM,
            tuning_fractions=TUNING_FRACTIONS,
            tuning_missingness_types=TUNING_MISSINGNESS_TYPES,
            n_trials=N_TRIALS,
            epochs=EPOCHS,
            patience=PATIENCE,
            warmup_epochs=WARMUP_EPOCHS,
            seed=OPTIMIZATION_SEED,
            val_rate=VAL_RATE,
            validation_mode=VALIDATION_MODE,
            study_name=STUDY_NAME,
            cleanup_tmp=True,
            n_jobs=1,
            verbose_method=True,
        )
        print("\nBest Transformer optimization payload:")
        print(best_payload)

    best_path = best_json_path()
    best_params = load_best_params(best_path)

    if RUN_FINAL_BENCHMARK:
        assert_seeded_preprocessing_files_exist(
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            context_name="Transformer repeated final benchmark",
        )

        for current_seed in FINAL_SEEDS:
            print("\n" + "-" * 100)
            print(f"Running frozen-best Transformer benchmark for seed={current_seed}")
            print("-" * 100)

            run_final_with_best(
                best_json_path=str(best_path),
                file_path=DATA_FILE,
                columns_num=COLUMNS_NUM,
                fractions=FINAL_FRACTIONS,
                missingness_types=FINAL_MISSINGNESS_TYPES,
                epochs=EPOCHS,
                patience=PATIENCE,
                warmup_epochs=WARMUP_EPOCHS,
                seed=int(current_seed),
                verbose_method=True,
            )

        assert_final_outputs_exist(best_params=best_params)

    if RUN_EVALUATION:
        outputs = transformer_eval(
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
        print("\nTransformer evaluation outputs:")
        print(outputs)

        if RUN_MAINFIG_EVALUATION:
            mainfig_dir = (
                experiment_dir()
                / "Evaluation"
                / "Transformer_seeded_pipeline_mainfig"
            )
            mainfig_outputs = transformer_eval(
                columns_num=COLUMNS_NUM,
                study_name=STUDY_NAME,
                fractions=[0.1, 0.3, 0.5],
                missingness_types=FINAL_MISSINGNESS_TYPES,
                seeds=FINAL_SEEDS,
                output_dir=str(mainfig_dir),
                show=True,
                strict=True,
                require_condition_scores=True,
                show_full_hyperparameter_legend=True,
            )
            print("\nTransformer main-figure outputs:")
            print(mainfig_outputs)

    print("\nTransformer seeded pipeline completed.")


if __name__ == "__main__":
    main()
