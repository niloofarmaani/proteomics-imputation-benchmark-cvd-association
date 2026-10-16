# Optimization/method_spaces.py

from __future__ import annotations

from typing import Any, Dict, List


# =============================================================================
# Explicit fixed method settings
# =============================================================================
# These are method-level execution settings, not Optuna suggestions. Keeping them
# here makes them visible, serializable in method_params, and identical between
# optimization and the frozen final benchmark.
GAIN_FIXED_SETTINGS: Dict[str, Any] = {
    "val_rate": 0.05,
    "validation_mode": "stratified_quantile",
    "refit_final": True,
    "mc_samples_validation": 10,
    "mc_samples_final": 30,
    "output_activation": "auto",
    "noise_type": "auto",
    "noise_scale": None,
    "grad_clip": 1.0,
    "hint_mode": "bernoulli",
}


ITERATIVE_FIXED_SETTINGS: Dict[str, Any] = {
    "tol": 1e-3,
    "n_nearest_features": None,
    "imputation_order": "ascending",
    "initial_strategy": "mean",
    "estimator_name": "BayesianRidge",
    "sample_posterior": False,
    "skip_complete": False,
}


TRANSFORMER_FIXED_SETTINGS: Dict[str, Any] = {
    "batch_size": 32,
    "nhead": 4,
    "num_layers": 1,
    "dropout": 0.10,
    "pseudo_mask_rate": 0.20,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "internal_val_rate": 0.05,
}


def _arch_choice(trial, name: str, choices: Dict[str, List[int]]) -> List[int]:
    if not choices:
        raise ValueError(f"Architecture choice dictionary for '{name}' is empty.")

    key = trial.suggest_categorical(name, list(choices.keys()))

    if key not in choices:
        raise KeyError(
            f"Optuna returned architecture key '{key}', which is not defined for '{name}'."
        )

    return list(choices[key])


def suggest_params(method: str, trial) -> Dict[str, Any]:
    """
    Return the complete method parameter dictionary for one Optuna trial.

    Every parameter consumed by Optimization.common.call_existing_method must be
    present in the returned dictionary. Fixed execution settings are explicit and
    are stored alongside tuned parameters in Optuna user attributes and the final
    best-parameter JSON.
    """
    m = str(method).upper()

    ae_archs = {
        "32-16": [32, 16],
        "50-30-20": [50, 30, 20],
        "64-32": [64, 32],
        "96-48": [96, 48],
        "128-64": [128, 64],
        "128-64-32": [128, 64, 32],
    }

    if m == "AE":
        return {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "n_neurons": _arch_choice(trial, "architecture", ae_archs),
            "dim_latent": trial.suggest_categorical(
                "dim_latent", [8, 16, 32, 64, 96, 128]
            ),
            "freebits": 0.1,
        }

    if m == "DAE":
        return {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "n_neurons": _arch_choice(trial, "architecture", ae_archs),
            "dim_latent": trial.suggest_categorical(
                "dim_latent", [8, 16, 32, 64, 96, 128]
            ),
            "freebits": 0.1,
            "noise_std": trial.suggest_float("noise_std", 0.0, 0.30),
            "feature_dropout": trial.suggest_float(
                "feature_dropout", 0.0, 0.30
            ),
        }

    if m == "VAE":
        return {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "n_neurons": _arch_choice(trial, "architecture", ae_archs),
            "dim_latent": trial.suggest_categorical(
                "dim_latent", [8, 16, 32, 64, 96, 128]
            ),
            "freebits": trial.suggest_categorical(
                "freebits", [0.01, 0.05, 0.1, 0.2, 0.5]
            ),
        }

    if m == "GAIN":
        gain_archs = {
            "64-64": [64, 64],
            "96-96": [96, 96],
            "128-128": [128, 128],
            "192-96": [192, 96],
            "256-128": [256, 128],
        }

        params: Dict[str, Any] = {
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 64, 128]
            ),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "n_neurons": _arch_choice(trial, "architecture", gain_archs),
            "alpha": trial.suggest_categorical(
                "alpha", [1.0, 5.0, 10.0, 50.0, 100.0]
            ),
            "hint_rate": trial.suggest_categorical(
                "hint_rate", [0.7, 0.8, 0.9]
            ),
            "dropout_rate": trial.suggest_categorical(
                "dropout_rate", [0.0, 0.05, 0.1]
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-6, 1e-4, log=True
            ),
            "pseudo_rate": trial.suggest_categorical(
                "pseudo_rate", [0.0, 0.02, 0.05, 0.10]
            ),
            "pseudo_beta": trial.suggest_categorical(
                "pseudo_beta", [0.0, 1.0, 5.0, 10.0]
            ),
        }
        params.update(GAIN_FIXED_SETTINGS)
        return params

    if m == "GAN":
        gen_archs = {
            "64-64": [64, 64],
            "128-64": [128, 64],
            "256-128": [256, 128],
        }
        disc_archs = {
            "64-64": [64, 64],
            "128-64": [128, 64],
            "256-128": [256, 128],
        }
        return {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "freebits": trial.suggest_categorical(
                "freebits", [0.01, 0.05, 0.1]
            ),
            "gen_hidden": _arch_choice(trial, "gen_hidden", gen_archs),
            "disc_hidden": _arch_choice(trial, "disc_hidden", disc_archs),
            "hint_rate": trial.suggest_categorical(
                "hint_rate", [0.7, 0.8, 0.9]
            ),
            "alpha": trial.suggest_categorical(
                "alpha", [1.0, 5.0, 10.0, 50.0]
            ),
        }

    if m == "RNN":
        return {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [32, 64, 128, 256]
            ),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "pseudo_mask_rate": trial.suggest_categorical(
                "pseudo_mask_rate", [0.05, 0.10, 0.20, 0.30]
            ),
        }

    if m in ("TRANSFORMER", "TRF"):
        # Reduced Transformer search. Only d_model and learning_rate are tuned.
        # All other settings are explicit fixed settings and are included in
        # method_params, the Optuna trial tables, and the best-parameter JSON.
        d_model = trial.suggest_categorical(
            "d_model",
            [16, 32],
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-4,
            1e-3,
            log=True,
        )

        fixed = dict(TRANSFORMER_FIXED_SETTINGS)

        if int(d_model) % int(fixed["nhead"]) != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by "
                f"nhead={fixed['nhead']}."
            )

        return {
            **fixed,
            "learning_rate": float(learning_rate),
            "d_model": int(d_model),
            "dim_feedforward": int(2 * int(d_model)),
        }

    if m == "KNN":
        return {
            "k": trial.suggest_categorical("k", [3, 5, 7, 10, 15, 20, 30])
        }

    if m in ("MISSFOREST", "MF"):
        return {
            "niter": trial.suggest_categorical("niter", [3, 5, 10]),
            "ntree": trial.suggest_categorical("ntree", [50, 100, 200]),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 20]),
            "sample_fraction": trial.suggest_categorical(
                "sample_fraction", [0.5, 0.7, 1.0]
            ),
        }

    if m in ("SOFTIMPUTE", "SOFTIMPUTE_PY"):
        return {
            "rank_max": trial.suggest_categorical(
                "rank_max", [5, 10, 20, 40, 80]
            ),
            "maxit": trial.suggest_categorical("maxit", [50, 100, 200]),
        }

    if m in ("ITERATIVE", "ITERATIVE_IMPUTER"):
        params: Dict[str, Any] = {
            "max_iter": trial.suggest_categorical(
                "max_iter", [10, 25, 50, 100]
            )
        }
        params.update(ITERATIVE_FIXED_SETTINGS)
        return params

    if m == "MICE":
        return {
            "m": trial.suggest_categorical("m", [1, 3, 5]),
            "maxit": trial.suggest_categorical("maxit", [10, 25, 50]),
        }

    if m == "BPCA":
        return {
            "nPcs": trial.suggest_categorical("nPcs", [2, 5, 10, 20, 40]),
            "maxSteps": trial.suggest_categorical(
                "maxSteps", [50, 100, 200]
            ),
        }

    raise ValueError(f"No search space is defined for method={method}")
