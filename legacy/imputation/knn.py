import os
import subprocess
from pathlib import Path
import sys
import pandas as pd

# Make sure we can import config_paths from code/
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir, IMPUTATION_DIR, fraction_dir

def knn(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    kn_list,
    show_r_output=True,
    print_command=True,
    print_rmse_after_run=True,
    seed=42
):
    """
    Handles files for KNN imputation experiments using strict seed matching.
    """
    for fraction in fractions:
        for missingness in missingness_types:
            full_path = Path(fraction_dir(columns_num, fraction, missingness))
            
            # --- STRICT LOGIC: Require Seeded file only ---
            filename_seeded = f'Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}_seed_{seed}.csv'
            corrupted_data_path = full_path / filename_seeded
            
            if not corrupted_data_path.exists():
                raise FileNotFoundError(
                    f"\n[CRITICAL] KNN could not find the exact seed file:\n{corrupted_data_path}\n"
                    "Ensure your preprocessing script generated files for this exact seed."
                )
            
            # Ground truth file does not need a seed
            filename_normalized_data = f'normalized_data_col_{columns_num}.csv'
            normalized_data_path = full_path / filename_normalized_data

            # Ensure ground truth exists
            if not normalized_data_path.exists():
                raise FileNotFoundError(f"Error: Missing normalized data in {full_path}.")

            # Run for each K
            for kn in kn_list:
                filename_rmse = f'RMSE_KNN_col_{columns_num}_kn_{kn}_seed_{seed}.csv'
                rmse_path_knn = full_path / filename_rmse

                filename_imputed = f'imputed_KNN_col_{columns_num}_kn_{kn}_seed_{seed}.csv'
                imputed_path_knn = full_path / filename_imputed

                print(f"\nRunning R script for KNN Imputation (k={kn}, seed={seed})...")

                cmd = [
                    "Rscript",
                    str(IMPUTATION_DIR / "knn_imputer.R"),
                    str(corrupted_data_path),
                    str(normalized_data_path),
                    str(kn),
                    str(rmse_path_knn),
                    str(imputed_path_knn)
                ]

                if print_command:
                    print("Executing:", " ".join(cmd), flush=True)

                try:
                    capture = not show_r_output
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=capture,
                        text=True
                    )
                    if show_r_output:
                        print("R Script finished successfully.", flush=True)

                except subprocess.CalledProcessError as e:
                    print(f"\nERROR: KNN R script failed with return code {e.returncode}")
                    if not show_r_output:
                        stdout = getattr(e, "stdout", None)
                        stderr = getattr(e, "stderr", None)
                        if stdout:
                            print("\nCaptured R STDOUT:", flush=True)
                            print(stdout, flush=True)
                        if stderr:
                            print("\nCaptured R STDERR:", flush=True)
                            print(stderr, flush=True)
                    raise

                # Verify outputs
                if not rmse_path_knn.exists():
                    raise FileNotFoundError(f"KNN R script completed but RMSE file was not created: {rmse_path_knn}")

                if not imputed_path_knn.exists():
                    raise FileNotFoundError(f"KNN R script completed but imputed file was not created: {imputed_path_knn}")

                if print_rmse_after_run:
                    try:
                        rmse_df = pd.read_csv(rmse_path_knn)
                        print("\nGenerated RMSE CSV:")
                        print(rmse_df.to_string(index=False), flush=True)
                    except Exception as e:
                        print(f"\nCould not read RMSE CSV for printing: {e}", flush=True)

    print("\n--- KNN Imputation Pipeline Complete ---")