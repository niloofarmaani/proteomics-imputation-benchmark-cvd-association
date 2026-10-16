from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common import call_existing_method, get_current_experiments_name, get_experiment_dir, read_json


def run_final_with_best(
    *,
    best_json_path: str,
    file_path: str,
    columns_num: str,
    fractions: Sequence[float],
    missingness_types: Sequence[str],
    epochs: int | None = None,
    patience: int | None = None,
    warmup_epochs: int | None = None,
    seed: int | None = None,
    verbose_method: bool = True,
):
    """
    Runs the real benchmark using frozen hyperparameters from an Optuna JSON.
    This writes normal RMSE_*.csv and imputed_*.csv outputs in your current
    configured experiment directory.
    """
    best = read_json(Path(best_json_path))
    method = best["method"]
    params = best["params"]

    epochs = int(epochs if epochs is not None else best.get("epochs", 300))
    patience = int(patience if patience is not None else best.get("patience", 50))
    warmup_epochs = int(warmup_epochs if warmup_epochs is not None else best.get("warmup_epochs", 100))
    seed = int(seed if seed is not None else best.get("seed", 42))

    call_existing_method(
        method=method,
        params=params,
        file_path=file_path,
        columns_num=columns_num,
        fractions=[float(x) for x in fractions],
        missingness_types=[str(x) for x in missingness_types],
        epochs=epochs,
        patience=patience,
        warmup_epochs=warmup_epochs,
        seed=seed,
        verbose=verbose_method,
    )

    return best


def main():
    parser = argparse.ArgumentParser(description="Run final frozen imputer from an Optuna best-params JSON.")
    parser.add_argument("--best-json", required=True)
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--columns-num", default="all")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.1, 0.3, 0.5])
    parser.add_argument("--missingness", nargs="+", default=["MCAR", "MAR", "MNAR"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_final_with_best(
        best_json_path=args.best_json,
        file_path=args.file_path,
        columns_num=args.columns_num,
        fractions=args.fractions,
        missingness_types=args.missingness,
        epochs=args.epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        verbose_method=not args.quiet,
    )


if __name__ == "__main__":
    main()
