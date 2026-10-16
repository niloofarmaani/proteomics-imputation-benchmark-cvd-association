# imputation/missforest.py

from __future__ import annotations

import subprocess
from pathlib import Path
import sys

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import IMPUTATION_DIR, fraction_dir


def _sf_token(sample_fraction) -> str:
    """
    Keep sample_fraction readable and stable in filenames.
    """
    return str(sample_fraction)


def missforest(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    niter,
    tree_list,
    max_depth,
    sample_fraction,
    seed=42,
    show_r_output=True,
    print_command=True,
    print_rmse_after_run=True,
):
    """
    Strict seeded MissForest/missRanger wrapper.

    Required input:
        Corrupted_data_fraction_<fraction>_missingness_<missingness>_col_<columns_num>_seed_<seed>.csv

    Output:
        RMSE_mf_col_<columns_num>_nt_<tree>_niter_<niter>_md_<max_depth>_sf_<sample_fraction>_seed_<seed>.csv
        imputed_mf_col_<columns_num>_nt_<tree>_niter_<niter>_md_<max_depth>_sf_<sample_fraction>_seed_<seed>.csv

    This function intentionally fails if the exact seeded corrupted file is missing.
    """

    niter = int(niter)
    max_depth = int(max_depth)
    seed = int(seed)

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            full_path = Path(fraction_dir(columns_num, fraction, missingness))
            full_path.mkdir(parents=True, exist_ok=True)

            corrupted_data_path = (
                full_path
                / f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}_seed_{seed}.csv"
            )

            if not corrupted_data_path.exists():
                raise FileNotFoundError(
                    f"\n[CRITICAL] MissForest could not find the exact seeded corrupted file:\n"
                    f"{corrupted_data_path}\n\n"
                    "Run corrected preprocessing for this exact fraction, missingness type, and seed."
                )

            normalized_data_path = full_path / f"normalized_data_col_{columns_num}.csv"

            if not normalized_data_path.exists():
                raise FileNotFoundError(
                    f"\n[CRITICAL] MissForest could not find normalized data:\n"
                    f"{normalized_data_path}"
                )

            for tree in tree_list:
                tree = int(tree)
                sf = _sf_token(sample_fraction)

                rmse_path_mf = (
                    full_path
                    / f"RMSE_mf_col_{columns_num}_nt_{tree}_niter_{niter}_md_{max_depth}_sf_{sf}_seed_{seed}.csv"
                )

                imputed_path_mf = (
                    full_path
                    / f"imputed_mf_col_{columns_num}_nt_{tree}_niter_{niter}_md_{max_depth}_sf_{sf}_seed_{seed}.csv"
                )

                print(
                    f"\nRunning MissForest/missRanger "
                    f"(ntree={tree}, niter={niter}, max_depth={max_depth}, "
                    f"sample_fraction={sample_fraction}, seed={seed})..."
                )

                cmd = [
                    "Rscript",
                    str(IMPUTATION_DIR / "missforest_imputer.R"),
                    str(corrupted_data_path),
                    str(normalized_data_path),
                    str(niter),
                    str(tree),
                    str(max_depth),
                    str(sample_fraction),
                    str(seed),
                    str(rmse_path_mf),
                    str(imputed_path_mf),
                ]

                if print_command:
                    print("Executing:", " ".join(cmd), flush=True)

                try:
                    capture = not show_r_output

                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=capture,
                        text=True,
                    )

                    if show_r_output:
                        print("MissForest R script finished successfully.", flush=True)
                    else:
                        if result.stdout:
                            print("\nCaptured R STDOUT:")
                            print(result.stdout)
                        if result.stderr:
                            print("\nCaptured R STDERR:")
                            print(result.stderr)

                except subprocess.CalledProcessError as e:
                    print(f"\nERROR: MissForest R script failed with return code {e.returncode}")

                    stdout = getattr(e, "stdout", None)
                    stderr = getattr(e, "stderr", None)

                    if stdout:
                        print("\nCaptured R STDOUT:")
                        print(stdout)

                    if stderr:
                        print("\nCaptured R STDERR:")
                        print(stderr)

                    raise

                if not rmse_path_mf.exists():
                    raise FileNotFoundError(
                        f"MissForest R script completed but RMSE file was not created:\n{rmse_path_mf}"
                    )

                if not imputed_path_mf.exists():
                    raise FileNotFoundError(
                        f"MissForest R script completed but imputed file was not created:\n{imputed_path_mf}"
                    )

                if print_rmse_after_run:
                    try:
                        rmse_df = pd.read_csv(rmse_path_mf)
                        print("\nGenerated RMSE CSV:")
                        print(rmse_df.to_string(index=False), flush=True)
                    except Exception as e:
                        raise RuntimeError(
                            f"MissForest RMSE file was created but could not be read:\n"
                            f"{rmse_path_mf}\nError: {e}"
                        )

    print("\n--- MissForest Imputation Pipeline Complete ---")