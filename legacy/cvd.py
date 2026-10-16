import config_paths
from pathlib import Path

config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

from Evaluation.cvd_baseline_tools import (
    run_cvd_baseline_from_best_rmse_log,
    plot_cvd_results_rocauc_panels,
    plot_cvd_all_key_metrics,
)

columns_num = "all"
main_dir = Path(config_paths.experiments_dir(columns_num))

log_path = main_dir / "best_rmse_log.txt"

results_csv = run_cvd_baseline_from_best_rmse_log(
    log_file_path=log_path,
    imputed_root=main_dir,
    out_csv_name="cvd_baseline_elasticnet_repeated_summary.csv",
    n_splits=5,
    n_repeats=5,
    inner_cv=5,
    random_state=42,
    debug_print_labels=True,
    allow_drop_missing_labels=False,
    verbose=True,
)

print("Saved downstream results to:")
print(results_csv)

plot_paths_roc = plot_cvd_results_rocauc_panels(
    results_csv=results_csv,
    err_mode="ci95",
    show=True,
)

print("ROC-AUC plots:")
print(plot_paths_roc)

plot_paths_all = plot_cvd_all_key_metrics(
    results_csv=results_csv,
    err_mode="ci95",
    show=True,
)

print("All metric plots:")
print(plot_paths_all)