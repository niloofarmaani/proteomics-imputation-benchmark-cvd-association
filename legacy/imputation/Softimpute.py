# imputation/Softimpute.py

from __future__ import annotations

import subprocess
from pathlib import Path
import sys

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir, IMPUTATION_DIR


def softImpute(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    rank_max_list,
    maxit,
    seed=42,
    show_output=True,
    print_command=True,
    print_rmse_after_run=True,
):
    """
    Strict seeded SoftImpute wrapper.

    Required input:
        Corrupted_data_fraction_<fraction>_missingness_<missingness>_col_<columns_num>_seed_<seed>.csv

    Output:
        RMSE_softImpute_col_<columns_num>_rank_<rank>_maxit_<maxit>_seed_<seed>.csv
        imputed_softImpute_col_<columns_num>_rank_<rank>_maxit_<maxit>_seed_<seed>.csv

    This function intentionally fails if the exact seeded corrupted file is missing.
    """

    maxit = int(maxit)
    seed = int(seed)

    main_directory = Path(experiments_dir(columns_num))

    if not main_directory.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist:\n{main_directory}\n"
            "Run preprocessing before SoftImpute."
        )

    imputer_script = Path(IMPUTATION_DIR) / "softimpute_imputer.py"

    if not imputer_script.exists():
        raise FileNotFoundError(
            f"SoftImpute backend script not found:\n{imputer_script}"
        )

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            full_path = main_directory / f"fraction_{fraction}_missingness_{missingness}"

            if not full_path.exists():
                raise FileNotFoundError(
                    f"Required fraction/missingness directory does not exist:\n{full_path}"
                )

            corrupted_data_path = (
                full_path
                / f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}_seed_{seed}.csv"
            )

            if not corrupted_data_path.exists():
                raise FileNotFoundError(
                    f"\n[CRITICAL] SoftImpute could not find the exact seeded corrupted file:\n"
                    f"{corrupted_data_path}\n\n"
                    "Run corrected preprocessing for this exact fraction, missingness type, and seed."
                )

            normalized_data_path = full_path / f"normalized_data_col_{columns_num}.csv"

            if not normalized_data_path.exists():
                raise FileNotFoundError(
                    f"\n[CRITICAL] SoftImpute could not find normalized data:\n"
                    f"{normalized_data_path}"
                )

            for rank_max in rank_max_list:
                rank_max = int(rank_max)

                rmse_path = (
                    full_path
                    / f"RMSE_softImpute_col_{columns_num}_rank_{rank_max}_maxit_{maxit}_seed_{seed}.csv"
                )

                imputed_data_path = (
                    full_path
                    / f"imputed_softImpute_col_{columns_num}_rank_{rank_max}_maxit_{maxit}_seed_{seed}.csv"
                )

                print(
                    f"\nRunning SoftImpute "
                    f"(fraction={fraction}, missingness={missingness}, "
                    f"rank={rank_max}, maxit={maxit}, seed={seed})"
                )

                cmd = [
                    sys.executable,
                    str(imputer_script),
                    str(corrupted_data_path),
                    str(normalized_data_path),
                    str(rank_max),
                    str(maxit),
                    str(seed),
                    str(rmse_path),
                    str(imputed_data_path),
                ]

                if print_command:
                    print("Executing:", " ".join(cmd), flush=True)

                try:
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=not show_output,
                        text=True,
                    )

                    if show_output:
                        print("SoftImpute subprocess finished successfully.", flush=True)
                    else:
                        if result.stdout:
                            print("\nCaptured SoftImpute STDOUT:")
                            print(result.stdout)

                        if result.stderr:
                            print("\nCaptured SoftImpute STDERR:")
                            print(result.stderr)

                except subprocess.CalledProcessError as e:
                    print("\nERROR: SoftImpute subprocess failed")
                    print("Command:")
                    print(" ".join(e.cmd))

                    stdout = getattr(e, "stdout", None)
                    stderr = getattr(e, "stderr", None)

                    if stdout:
                        print("\nCaptured STDOUT:")
                        print(stdout)

                    if stderr:
                        print("\nCaptured STDERR:")
                        print(stderr)

                    raise

                if not rmse_path.exists():
                    raise FileNotFoundError(
                        f"SoftImpute subprocess completed but RMSE file was not created:\n{rmse_path}"
                    )

                if not imputed_data_path.exists():
                    raise FileNotFoundError(
                        f"SoftImpute subprocess completed but imputed file was not created:\n{imputed_data_path}"
                    )

                if print_rmse_after_run:
                    try:
                        rmse_df = pd.read_csv(rmse_path)
                        print("\nGenerated SoftImpute RMSE CSV:")
                        print(rmse_df.to_string(index=False), flush=True)
                    except Exception as e:
                        raise RuntimeError(
                            f"SoftImpute RMSE file was created but could not be read:\n"
                            f"{rmse_path}\nError: {e}"
                        )

    print("\n--- SoftImpute Pipeline Complete ---")