# Optimization/run_bayes_opt.py

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import optuna
except Exception as exc:
    optuna = None
    _OPTUNA_IMPORT_ERROR = exc

from .common import (
    call_existing_method,
    cleanup_torch,
    get_current_experiments_name,
    get_experiment_dir,
    prepare_validation_experiment,
    sanitize_name,
    score_trial_outputs,
    temporary_experiments_name,
    write_json,
)
from .method_spaces import suggest_params


PIPELINE_SCHEMA_VERSION = "seeded_per_condition_v2"


def _default_study_name(
    method: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
) -> str:
    fraction_token = "-".join(str(x).replace(".", "p") for x in fractions)
    missingness_token = "-".join(str(x).upper() for x in missingness_types)
    return sanitize_name(
        f"{method}_bayes_f{fraction_token}_{missingness_token}"
    )


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None

    if isinstance(obj, np.ndarray):
        return [_json_safe(value) for value in obj.tolist()]

    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None

    return obj


def _normalized_panel(
    fractions: Sequence[float],
    missingness_types: Sequence[str],
) -> Tuple[List[float], List[str]]:
    normalized_fractions = [float(value) for value in fractions]
    normalized_missingness = [str(value).upper() for value in missingness_types]

    if not normalized_fractions:
        raise ValueError("tuning_fractions cannot be empty.")

    if not normalized_missingness:
        raise ValueError("tuning_missingness_types cannot be empty.")

    if len(set(normalized_fractions)) != len(normalized_fractions):
        raise ValueError(
            f"tuning_fractions contains duplicates: {normalized_fractions}"
        )

    if len(set(normalized_missingness)) != len(normalized_missingness):
        raise ValueError(
            "tuning_missingness_types contains duplicates: "
            f"{normalized_missingness}"
        )

    if any(not (0.0 < fraction < 1.0) for fraction in normalized_fractions):
        raise ValueError(
            "Every tuning fraction must lie strictly between 0 and 1. "
            f"Got: {normalized_fractions}"
        )

    return normalized_fractions, normalized_missingness


def _validate_score_rows(
    *,
    score_rows: Any,
    tuning_fractions: Sequence[float],
    tuning_missingness_types: Sequence[str],
    trial_number: int,
) -> List[Dict[str, Any]]:
    if not isinstance(score_rows, list):
        raise TypeError(
            f"Trial {trial_number}: score_rows must be a list, "
            f"got {type(score_rows)}."
        )

    expected_pairs = {
        (float(fraction), str(missingness).upper())
        for fraction in tuning_fractions
        for missingness in tuning_missingness_types
    }

    if len(score_rows) != len(expected_pairs):
        raise ValueError(
            f"Trial {trial_number}: expected {len(expected_pairs)} score rows, "
            f"but received {len(score_rows)}.\nscore_rows={score_rows}"
        )

    observed_pairs = set()
    clean_rows: List[Dict[str, Any]] = []

    for raw_row in score_rows:
        if not isinstance(raw_row, dict):
            raise TypeError(
                f"Trial {trial_number}: each score row must be a dictionary. "
                f"Got: {raw_row}"
            )

        required = {"fraction", "missingness", "validation_rmse"}
        missing = required - set(raw_row)

        if missing:
            raise KeyError(
                f"Trial {trial_number}: score row is missing keys "
                f"{sorted(missing)}. Row: {raw_row}"
            )

        fraction = float(raw_row["fraction"])
        missingness = str(raw_row["missingness"]).upper()
        validation_rmse = float(raw_row["validation_rmse"])

        if not np.isfinite(validation_rmse):
            raise ValueError(
                f"Trial {trial_number}: validation RMSE is non-finite. "
                f"Row: {raw_row}"
            )

        pair = (fraction, missingness)

        if pair in observed_pairs:
            raise ValueError(
                f"Trial {trial_number}: duplicate condition score for {pair}."
            )

        observed_pairs.add(pair)

        clean_row = dict(raw_row)
        clean_row["fraction"] = fraction
        clean_row["missingness"] = missingness
        clean_row["validation_rmse"] = validation_rmse

        if "n_validation_cells" in clean_row:
            if clean_row["n_validation_cells"] is None:
                raise ValueError(
                    f"Trial {trial_number}: n_validation_cells is None for {pair}."
                )
            clean_row["n_validation_cells"] = int(
                clean_row["n_validation_cells"]
            )
            if clean_row["n_validation_cells"] <= 0:
                raise ValueError(
                    f"Trial {trial_number}: n_validation_cells must be positive "
                    f"for {pair}."
                )

        clean_rows.append(_json_safe(clean_row))

    if observed_pairs != expected_pairs:
        raise ValueError(
            f"Trial {trial_number}: condition scores do not match the tuning panel.\n"
            f"Expected: {sorted(expected_pairs)}\n"
            f"Observed: {sorted(observed_pairs)}"
        )

    clean_rows.sort(
        key=lambda row: (str(row["missingness"]), float(row["fraction"]))
    )
    return clean_rows


def _trial_duration_seconds(trial: Any) -> Optional[float]:
    if trial.datetime_start is None or trial.datetime_complete is None:
        return None

    return float(
        (trial.datetime_complete - trial.datetime_start).total_seconds()
    )


def _count_complete_trials(study: Any) -> int:
    return sum(
        1
        for trial in study.trials
        if getattr(trial.state, "name", str(trial.state)) == "COMPLETE"
    )


def _completed_trials_missing_score_rows(study: Any) -> List[int]:
    missing_trial_numbers: List[int] = []

    for trial in study.trials:
        state_name = getattr(trial.state, "name", str(trial.state))

        if state_name != "COMPLETE":
            continue

        score_rows = trial.user_attrs.get("score_rows")

        if not isinstance(score_rows, list) or not score_rows:
            missing_trial_numbers.append(int(trial.number))

    return missing_trial_numbers


def _expected_study_metadata(
    *,
    method: str,
    base_experiments_name: str,
    columns_num: str,
    tuning_fractions: Sequence[float],
    tuning_missingness_types: Sequence[str],
    seed: int,
    val_rate: float,
    validation_mode: str,
) -> Dict[str, Any]:
    return {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "method": str(method).upper(),
        "base_experiments_name": str(base_experiments_name),
        "columns_num": str(columns_num),
        "tuning_fractions": [float(x) for x in tuning_fractions],
        "tuning_missingness_types": [
            str(x).upper() for x in tuning_missingness_types
        ],
        "seed": int(seed),
        "val_rate": float(val_rate),
        "validation_mode": str(validation_mode),
        "sampler": "TPESampler",
    }


def _assert_or_set_study_metadata(
    study: Any,
    expected_metadata: Dict[str, Any],
) -> None:
    existing_trial_count = len(study.trials)
    existing_metadata = dict(study.user_attrs)

    if existing_trial_count == 0:
        for key, value in expected_metadata.items():
            study.set_user_attr(key, _json_safe(value))
        return

    missing_keys = [
        key for key in expected_metadata if key not in existing_metadata
    ]

    if missing_keys:
        raise RuntimeError(
            "The existing Optuna study has trials but lacks required study "
            "metadata. It may have been created by an older pipeline.\n"
            f"Missing metadata keys: {missing_keys}\n"
            "Use a new study_name or delete the old database."
        )

    mismatches = []

    for key, expected_value in expected_metadata.items():
        observed_value = existing_metadata[key]

        if _json_safe(observed_value) != _json_safe(expected_value):
            mismatches.append(
                f"{key}: expected={expected_value!r}, observed={observed_value!r}"
            )

    if mismatches:
        raise RuntimeError(
            "The existing Optuna study metadata does not match this run.\n"
            + "\n".join(mismatches)
            + "\nUse a new study_name or delete the incompatible database."
        )


def _write_trial_tables(
    *,
    study: Any,
    opt_dir: Path,
    study_name: str,
) -> Tuple[Path, Path]:
    trial_table_dir = opt_dir / "trial_tables"
    trial_table_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: List[Dict[str, Any]] = []
    condition_rows: List[Dict[str, Any]] = []

    for trial in study.trials:
        state_name = getattr(trial.state, "name", str(trial.state))
        method_params = trial.user_attrs.get("method_params", {})
        score_rows = trial.user_attrs.get("score_rows", [])

        trial_row: Dict[str, Any] = {
            "number": int(trial.number),
            "state": state_name,
            "value": trial.value if trial.value is not None else np.nan,
            "mean_validation_rmse": trial.user_attrs.get(
                "mean_validation_rmse", np.nan
            ),
            "seed": trial.user_attrs.get("seed", np.nan),
            "datetime_start": trial.datetime_start,
            "datetime_complete": trial.datetime_complete,
            "duration_seconds": _trial_duration_seconds(trial),
            "params_json": json.dumps(
                _json_safe(trial.params), sort_keys=True
            ),
            "method_params_json": json.dumps(
                _json_safe(method_params), sort_keys=True
            ),
            "score_rows_json": json.dumps(
                _json_safe(score_rows), sort_keys=True
            ),
        }

        for key, value in trial.params.items():
            trial_row[f"params_{key}"] = _json_safe(value)

        if isinstance(method_params, dict):
            for key, value in method_params.items():
                output_key = f"method_params_{key}"
                if isinstance(value, (dict, list, tuple)):
                    trial_row[output_key] = json.dumps(
                        _json_safe(value), sort_keys=True
                    )
                else:
                    trial_row[output_key] = _json_safe(value)

        trial_rows.append(trial_row)

        if isinstance(score_rows, list):
            for score_row in score_rows:
                if not isinstance(score_row, dict):
                    raise TypeError(
                        f"Trial {trial.number} contains a non-dictionary score row: "
                        f"{score_row}"
                    )

                condition_row: Dict[str, Any] = {
                    "trial_number": int(trial.number),
                    "state": state_name,
                    "trial_value": (
                        trial.value if trial.value is not None else np.nan
                    ),
                    "seed": trial.user_attrs.get("seed", np.nan),
                    "method_params_json": json.dumps(
                        _json_safe(method_params), sort_keys=True
                    ),
                }

                for key, value in trial.params.items():
                    condition_row[f"params_{key}"] = _json_safe(value)

                if isinstance(method_params, dict):
                    for key, value in method_params.items():
                        output_key = f"method_params_{key}"
                        if isinstance(value, (dict, list, tuple)):
                            condition_row[output_key] = json.dumps(
                                _json_safe(value), sort_keys=True
                            )
                        else:
                            condition_row[output_key] = _json_safe(value)

                for key, value in score_row.items():
                    condition_row[key] = _json_safe(value)

                condition_rows.append(condition_row)

    trials_csv_path = trial_table_dir / f"{study_name}_trials.csv"
    condition_csv_path = (
        trial_table_dir / f"{study_name}_condition_scores.csv"
    )

    pd.DataFrame(trial_rows).to_csv(trials_csv_path, index=False)

    if not condition_rows:
        raise RuntimeError(
            "No per-condition score rows were available when writing trial tables."
        )

    pd.DataFrame(condition_rows).to_csv(condition_csv_path, index=False)

    return trials_csv_path, condition_csv_path


def run_bayes_opt(
    *,
    method: str,
    file_path: str,
    columns_num: str,
    tuning_fractions: Sequence[float],
    tuning_missingness_types: Sequence[str],
    n_trials: int = 40,
    epochs: int = 300,
    patience: int = 50,
    warmup_epochs: int = 100,
    seed: int = 42,
    val_rate: float = 0.05,
    validation_mode: str = "stratified_quantile",
    study_name: Optional[str] = None,
    base_experiments_name: Optional[str] = None,
    tmp_prefix: str = "OPTUNA_TMP",
    cleanup_tmp: bool = True,
    timeout: Optional[int] = None,
    n_jobs: int = 1,
    verbose_method: bool = False,
) -> Tuple[Dict[str, Any], Any]:
    if optuna is None:
        raise ImportError(
            "Optuna is not installed or could not be imported. "
            f"Original error: {_OPTUNA_IMPORT_ERROR}"
        )

    if int(n_trials) < 1:
        raise ValueError(f"n_trials must be at least 1. Got {n_trials}")

    if int(n_jobs) != 1:
        raise ValueError(
            "The seeded benchmark requires n_jobs=1. Parallel Optuna trials "
            "can change suggestion order and compete for shared compute resources."
        )

    if int(epochs) < 1:
        raise ValueError(f"epochs must be at least 1. Got {epochs}")

    if int(patience) < 1:
        raise ValueError(f"patience must be at least 1. Got {patience}")

    if int(warmup_epochs) < 0:
        raise ValueError(
            f"warmup_epochs must be non-negative. Got {warmup_epochs}"
        )

    if not (0.0 < float(val_rate) < 1.0):
        raise ValueError(f"val_rate must lie in (0, 1). Got {val_rate}")

    method = str(method)
    tuning_fractions, tuning_missingness_types = _normalized_panel(
        tuning_fractions,
        tuning_missingness_types,
    )

    data_path = Path(file_path).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Input data file does not exist:\n{data_path}")

    base_name = base_experiments_name or get_current_experiments_name()

    if not str(base_name):
        raise ValueError("The base experiments name cannot be empty.")

    study_name = sanitize_name(
        study_name
        or _default_study_name(
            method,
            tuning_fractions,
            tuning_missingness_types,
        )
    )

    base_columns_dir = get_experiment_dir(base_name, columns_num)

    if not base_columns_dir.exists():
        raise FileNotFoundError(
            f"Base experiment directory does not exist:\n{base_columns_dir}\n"
            "Run preprocessing before Bayesian optimization."
        )

    opt_dir = base_columns_dir / "Optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)
    storage_path = opt_dir / f"{study_name}.db"

    sampler = optuna.samplers.TPESampler(seed=int(seed))

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=sampler,
    )

    expected_metadata = _expected_study_metadata(
        method=method,
        base_experiments_name=base_name,
        columns_num=columns_num,
        tuning_fractions=tuning_fractions,
        tuning_missingness_types=tuning_missingness_types,
        seed=seed,
        val_rate=val_rate,
        validation_mode=validation_mode,
    )
    _assert_or_set_study_metadata(study, expected_metadata)

    old_trials_without_scores = _completed_trials_missing_score_rows(study)

    if old_trials_without_scores:
        raise RuntimeError(
            "This study contains completed trials without per-condition "
            "score_rows. It was created by an older optimizer pipeline.\n"
            f"Study: {study_name}\nDatabase: {storage_path}\n"
            "Affected trial numbers: "
            f"{old_trials_without_scores[:20]}\n"
            "Use a new study_name or delete the old database."
        )

    print("\n" + "=" * 80)
    print("Bayesian optimization started")
    print("=" * 80)
    print(f"Method: {method}")
    print(f"Study name: {study_name}")
    print(f"Storage path: {storage_path}")
    print(f"Base experiments name: {base_name}")
    print(f"Columns: {columns_num}")
    print(f"Tuning fractions: {tuning_fractions}")
    print(f"Tuning missingness types: {tuning_missingness_types}")
    print(f"Optimization seed: {seed}")
    print(f"Validation rate: {val_rate}")
    print(f"Validation mode: {validation_mode}")
    print("=" * 80)

    def objective(trial: Any) -> float:
        params = _json_safe(suggest_params(method, trial))

        if not isinstance(params, dict) or not params:
            raise ValueError(
                f"suggest_params returned an invalid parameter dictionary: {params}"
            )

        trial_seed = int(seed)
        trial.set_user_attr("method_params", params)
        trial.set_user_attr("seed", trial_seed)

        tmp_name = sanitize_name(
            f"{base_name}__{tmp_prefix}__{study_name}__trial_{trial.number}"
        )
        prepared = None

        try:
            prepared = prepare_validation_experiment(
                columns_num=columns_num,
                fractions=tuning_fractions,
                missingness_types=tuning_missingness_types,
                base_experiments_name=base_name,
                tmp_experiments_name=tmp_name,
                val_rate=val_rate,
                validation_mode=validation_mode,
                seed=trial_seed,
            )

            with temporary_experiments_name(tmp_name):
                call_existing_method(
                    method=method,
                    params=params,
                    file_path=str(data_path),
                    columns_num=columns_num,
                    fractions=tuning_fractions,
                    missingness_types=tuning_missingness_types,
                    epochs=int(epochs),
                    patience=int(patience),
                    warmup_epochs=int(warmup_epochs),
                    seed=trial_seed,
                    verbose=bool(verbose_method),
                )

            mean_score, score_rows = score_trial_outputs(
                prepared=prepared,
                columns_num=columns_num,
                fractions=tuning_fractions,
                missingness_types=tuning_missingness_types,
            )

            mean_score = float(mean_score)

            if not np.isfinite(mean_score):
                raise ValueError(
                    f"Trial {trial.number}: mean validation RMSE is non-finite: "
                    f"{mean_score}"
                )

            clean_score_rows = _validate_score_rows(
                score_rows=score_rows,
                tuning_fractions=tuning_fractions,
                tuning_missingness_types=tuning_missingness_types,
                trial_number=int(trial.number),
            )

            recomputed_mean = float(
                np.mean(
                    [
                        float(row["validation_rmse"])
                        for row in clean_score_rows
                    ]
                )
            )

            if not np.isclose(
                mean_score,
                recomputed_mean,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Trial {trial.number}: mean score {mean_score} does not "
                    f"equal the mean of condition scores {recomputed_mean}."
                )

            trial.set_user_attr("score_rows", clean_score_rows)
            trial.set_user_attr("mean_validation_rmse", mean_score)
            trial.set_user_attr(
                "tuning_fractions", _json_safe(tuning_fractions)
            )
            trial.set_user_attr(
                "tuning_missingness_types",
                _json_safe(tuning_missingness_types),
            )

            return mean_score

        except Exception as exc:
            print("\n" + "!" * 80)
            print(f"Trial {trial.number} failed")
            print(f"Method: {method}")
            print(f"Study: {study_name}")
            print(f"Parameters: {params}")
            print(f"Error: {exc!r}")
            print("!" * 80 + "\n")
            raise

        finally:
            cleanup_torch()

            if cleanup_tmp and prepared is not None:
                temporary_experiment_dir = prepared.tmp_columns_dir.parent
                if temporary_experiment_dir.exists():
                    shutil.rmtree(temporary_experiment_dir)

    completed_before = _count_complete_trials(study)
    remaining_trials = int(n_trials) - completed_before

    print("\nOptuna trial accounting")
    print("=" * 80)
    print(f"Requested total complete trials: {n_trials}")
    print(f"Already complete trials: {completed_before}")
    print(f"Remaining trials to run: {remaining_trials}")
    print("=" * 80)

    if remaining_trials < 0:
        raise RuntimeError(
            f"Study already has {completed_before} complete trials, which exceeds "
            f"the requested total n_trials={n_trials}. Use a new study_name, "
            "increase n_trials, or remove the old database."
        )

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=timeout,
            n_jobs=1,
        )
    else:
        print("No new trials are required.")

    completed_after = _count_complete_trials(study)

    if completed_after != int(n_trials):
        raise RuntimeError(
            f"Optimization ended with {completed_after} complete trials; "
            f"the requested total was {n_trials}."
        )

    best_trial = study.best_trial
    best_score_rows = best_trial.user_attrs.get("score_rows")

    if not isinstance(best_score_rows, list) or not best_score_rows:
        raise RuntimeError(
            "The best trial does not contain per-condition score_rows. "
            "Use a new study name and rerun with this optimizer."
        )

    trials_csv_path, condition_scores_csv_path = _write_trial_tables(
        study=study,
        opt_dir=opt_dir,
        study_name=study_name,
    )

    best_dir = opt_dir / "best_params"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_json_path = best_dir / f"{study_name}_best_params.json"

    best_payload: Dict[str, Any] = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "method": method,
        "study_name": study_name,
        "best_value_validation_rmse": float(study.best_value),
        "best_trial_number": int(best_trial.number),
        "params": _json_safe(
            best_trial.user_attrs.get("method_params", {})
        ),
        "optuna_params": _json_safe(best_trial.params),
        "score_rows": _json_safe(best_score_rows),
        "tuning_fractions": _json_safe(tuning_fractions),
        "tuning_missingness_types": _json_safe(
            tuning_missingness_types
        ),
        "epochs": int(epochs),
        "patience": int(patience),
        "warmup_epochs": int(warmup_epochs),
        "seed": int(seed),
        "val_rate": float(val_rate),
        "validation_mode": str(validation_mode),
        "base_experiments_name": str(base_name),
        "columns_num": str(columns_num),
        "storage_path": str(storage_path),
        "trials_csv_path": str(trials_csv_path),
        "condition_scores_csv_path": str(condition_scores_csv_path),
        "n_complete_trials": int(completed_after),
    }

    write_json(best_json_path, best_payload)

    print("\n" + "=" * 80)
    print("Bayesian optimization complete")
    print("=" * 80)
    print(f"Method: {method}")
    print(f"Study name: {study_name}")
    print(f"Best trial: {best_trial.number}")
    print(f"Best validation RMSE: {study.best_value}")
    print(f"Best params JSON: {best_json_path}")
    print(f"Trial table: {trials_csv_path}")
    print(f"Condition score table: {condition_scores_csv_path}")
    print("=" * 80)

    return best_payload, study


if __name__ == "__main__":
    pass
