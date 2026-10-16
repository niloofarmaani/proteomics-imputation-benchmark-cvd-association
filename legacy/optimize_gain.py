# optimize_gain_seeded_pipeline.py

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import config_paths


# =============================================================================
# Set project paths before project imports
# =============================================================================
config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

from Optimization.GAIN_bayes_opt import run_gain_bayes_opt
from Optimization.run_final_with_best import run_final_with_best
from Evaluation.GAIN_eval import gain_eval


# =============================================================================
# Configuration
# =============================================================================
DATA_FILE = "/scratch/biobank/niloo/mayoVasc/cureASciD_01_data_olinkTarget96_20241030.csv"
COLUMNS_NUM = "all"
STUDY_NAME = "GAIN_balanced_MAR_MNAR_10_30_seeded_v1"

OPTIMIZATION_SEED = 42
FINAL_SEEDS = [42, 100, 2024, 777, 999]

TUNING_FRACTIONS = [0.1, 0.3]
TUNING_MISSINGNESS_TYPES = ["MAR", "MNAR"]

FINAL_FRACTIONS = [0.01, 0.1, 0.3, 0.5]
FINAL_MISSINGNESS_TYPES = ["MCAR", "MAR", "MNAR"]

N_TRIALS = 15
EPOCHS = 300
PATIENCE = 50
WARMUP_EPOCHS = 100
VAL_RATE = 0.05
VALIDATION_MODE = "stratified_quantile"

RUN_OPTIMIZATION = True
RUN_FINAL_BENCHMARK = True
RUN_EVALUATION = True
RUN_MAINFIG_EVALUATION = True


# =============================================================================
# Parameter and filename helpers
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


def _float_token(x: float) -> str:
    return (
        f"{float(x):.12g}"
        .replace("-", "m")
        .replace(".", "p")
        .replace("+", "")
    )


def _text_token(value: Any) -> str:
    text = str(value).strip().lower()
    output = []
    for char in text:
        output.append(char if char.isalnum() else "_")
    token = "".join(output)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _parse_n_neurons(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        parsed = [int(x) for x in value]
    elif isinstance(value, str):
        text = value.strip()
        try:
            literal = ast.literal_eval(text)
            if isinstance(literal, (list, tuple)):
                parsed = [int(x) for x in literal]
            else:
                raise ValueError
        except Exception:
            text = text.replace("[", "").replace("]", "")
            text = text.replace("(", "").replace(")", "")
            text = text.replace("-", ",")
            parsed = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        raise ValueError(f"Could not parse n_neurons from: {value}")

    if not parsed or any(x <= 0 for x in parsed):
        raise ValueError(f"Invalid n_neurons: {parsed}")
    return parsed


_REQUIRED_PARAMS = {
    "batch_size",
    "lr",
    "n_neurons",
    "alpha",
    "hint_rate",
    "val_rate",
    "validation_mode",
    "refit_final",
    "mc_samples_validation",
    "mc_samples_final",
    "output_activation",
    "noise_type",
    "noise_scale",
    "dropout_rate",
    "weight_decay",
    "grad_clip",
    "pseudo_rate",
    "pseudo_beta",
    "hint_mode",
}


def load_best_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Best-params JSON not found:\n{path}\n"
            "Run GAIN optimization first or check STUDY_NAME."
        )

    with path.open("r") as handle:
        payload = json.load(handle)

    if "params" not in payload:
        raise KeyError(f"Best-params JSON contains no 'params' key:\n{path}")

    missing = _REQUIRED_PARAMS - set(payload["params"])
    if missing:
        raise KeyError(
            f"GAIN best params are missing {sorted(missing)}.\n"
            f"File: {path}\nparams: {payload['params']}"
        )

    if "epochs" not in payload or "patience" not in payload:
        raise KeyError(f"Best-params JSON must contain epochs and patience:\n{path}")

    return payload


def gain_parameter_id(payload: Dict[str, Any]) -> str:
    params = payload["params"]
    neurons = "-".join(str(x) for x in _parse_n_neurons(params["n_neurons"]))
    return (
        f"bs{int(params['batch_size'])}"
        f"_lr{_float_token(float(params['lr']))}"
        f"_nn{neurons}"
        f"_al{_float_token(float(params['alpha']))}"
        f"_hr{_float_token(float(params['hint_rate']))}"
        f"_do{_float_token(float(params['dropout_rate']))}"
        f"_wd{_float_token(float(params['weight_decay']))}"
        f"_ps{_float_token(float(params['pseudo_rate']))}"
        f"_pb{_float_token(float(params['pseudo_beta']))}"
        f"_hm{_text_token(params['hint_mode'])}"
        f"_mc{int(params['mc_samples_final'])}"
        f"_ep{int(payload['epochs'])}"
        f"_pat{int(payload['patience'])}"
    )


# =============================================================================
# Strict input and output checks
# =============================================================================
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
    missing_files = []
    base_dir = experiment_dir()

    for fraction in fractions:
        fraction = float(fraction)
        for missingness in missingness_types:
            missingness = str(missingness).upper()
            folder = base_dir / f"fraction_{fraction}_missingness_{missingness}"
            reference_path = folder / f"normalized_data_col_{COLUMNS_NUM}.csv"
            if not reference_path.exists():
                missing_files.append(str(reference_path))

            for seed in seeds:
                corrupted_path = (
                    folder
                    / f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{COLUMNS_NUM}_seed_{int(seed)}.csv"
                )
                if not corrupted_path.exists():
                    missing_files.append(str(corrupted_path))

    if missing_files:
        preview = "\n".join(missing_files[:50])
        raise FileNotFoundError(
            f"Missing seeded preprocessing files for {context_name}.\n\n"
            f"First missing files:\n{preview}"
        )


def assert_gain_final_outputs_exist(
    *,
    fractions,
    missingness_types,
    seeds,
    payload: Dict[str, Any],
) -> None:
    parameter_id = gain_parameter_id(payload)
    missing_files = []
    base_dir = experiment_dir()

    for fraction in fractions:
        fraction = float(fraction)
        for missingness in missingness_types:
            missingness = str(missingness).upper()
            folder = base_dir / f"fraction_{fraction}_missingness_{missingness}"

            for seed in seeds:
                seed = int(seed)
                rmse_path = (
                    folder
                    / f"RMSE_GAIN_col_{COLUMNS_NUM}_params_{parameter_id}_seed_{seed}.csv"
                )
                imputed_path = (
                    folder
                    / f"imputed_GAIN_col_{COLUMNS_NUM}_params_{parameter_id}_seed_{seed}.csv"
                )
                if not rmse_path.exists():
                    missing_files.append(str(rmse_path))
                if not imputed_path.exists():
                    missing_files.append(str(imputed_path))

    if missing_files:
        preview = "\n".join(missing_files[:50])
        raise FileNotFoundError(
            "Final GAIN benchmark did not create all expected outputs.\n\n"
            f"First missing files:\n{preview}"
        )


# =============================================================================
# Pipeline
# =============================================================================
def main() -> None:
    print("\n" + "=" * 100)
    print("GAIN SEEDED PIPELINE")
    print("=" * 100)
    print(f"Experiment directory: {experiment_dir()}")
    print(f"Study name: {STUDY_NAME}")
    print(f"Optimization seed: {OPTIMIZATION_SEED}")
    print(f"Final seeds: {FINAL_SEEDS}")
    print("=" * 100)

    assert_data_file_exists()
    assert_seeded_preprocessing_files_exist(
        fractions=TUNING_FRACTIONS,
        missingness_types=TUNING_MISSINGNESS_TYPES,
        seeds=[OPTIMIZATION_SEED],
        context_name="GAIN Optuna tuning",
    )

    if RUN_OPTIMIZATION:
        print("\n--- Phase 1: Optuna tuning for GAIN ---")
        best_payload, _ = run_gain_bayes_opt(
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
            verbose_method=True,
        )
        print("\nBest optimizer payload:")
        print(best_payload)
    else:
        print("\n--- Phase 1 skipped ---")

    json_path = best_json_path()
    payload = load_best_payload(json_path)
    print("\nBest GAIN parameters:")
    print(payload["params"])
    print(f"Best JSON: {json_path}")

    if RUN_FINAL_BENCHMARK:
        print("\n--- Phase 2: Checking repeated-seed inputs ---")
        assert_seeded_preprocessing_files_exist(
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            context_name="GAIN repeated final benchmark",
        )

        print("\n--- Phase 3: Repeated final GAIN benchmark ---")
        for current_seed in FINAL_SEEDS:
            print("\n" + "-" * 100)
            print(f"Running final GAIN benchmark for seed={current_seed}")
            print("-" * 100)
            run_final_with_best(
                best_json_path=str(json_path),
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

        print("\n--- Phase 4: Checking final outputs ---")
        assert_gain_final_outputs_exist(
            fractions=FINAL_FRACTIONS,
            missingness_types=FINAL_MISSINGNESS_TYPES,
            seeds=FINAL_SEEDS,
            payload=payload,
        )
        print("All expected final GAIN outputs exist.")
    else:
        print("\n--- Final benchmark skipped ---")

    if RUN_EVALUATION:
        print("\n--- Phase 5: GAIN evaluation, all fractions ---")
        outputs = gain_eval(
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
            print("\n--- Phase 6: GAIN evaluation, main-figure fractions ---")
            output_dir = experiment_dir() / "Evaluation" / "GAIN_seeded_pipeline_mainfig"
            main_outputs = gain_eval(
                columns_num=COLUMNS_NUM,
                study_name=STUDY_NAME,
                fractions=[0.1, 0.3, 0.5],
                missingness_types=FINAL_MISSINGNESS_TYPES,
                seeds=FINAL_SEEDS,
                output_dir=str(output_dir),
                show=True,
                strict=True,
                require_condition_scores=True,
                show_full_hyperparameter_legend=True,
            )
            print(main_outputs)
    else:
        print("\n--- Evaluation skipped ---")

    print("\n" + "=" * 100)
    print("GAIN seeded pipeline completed.")
    print("=" * 100)


if __name__ == "__main__":
    main()
