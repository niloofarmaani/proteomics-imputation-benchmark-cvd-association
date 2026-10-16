import config_paths
from pathlib import Path

config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

from Optimization.MissForest_bayes_opt import run_missforest_bayes_opt
from Optimization.run_final_with_best import run_final_with_best
from Evaluation.missforest_eval import missforest_eval


DATA_FILE = "/scratch/biobank/niloo/mayoVasc/cureASciD_01_data_olinkTarget96_20241030.csv"

columns_num = "all"

# -----------------------------
# 1. Balanced tuning setup
# -----------------------------
study_name = "MissForest_balanced_MAR_MNAR_10_30"

tuning_fractions = [0.1, 0.3]
tuning_missingness_types = ["MAR", "MNAR"]

# -----------------------------
# 2. Final benchmark setup
# -----------------------------
final_fractions = [0.01, 0.1, 0.3, 0.5]
final_missingness_types = ["MCAR", "MAR", "MNAR"]

# -----------------------------
# 3. Run Bayesian optimization
# -----------------------------
best, study = run_missforest_bayes_opt(
    file_path=DATA_FILE,
    columns_num=columns_num,
    tuning_fractions=tuning_fractions,
    tuning_missingness_types=tuning_missingness_types,
    n_trials=15,
    seed=42,
    val_rate=0.05,
    validation_mode="stratified_quantile",
    study_name=study_name,
    cleanup_tmp=True,
    verbose_method=False,
)

print("\nBest optimizer result:")
print(best)

# -----------------------------
# 4. Run final frozen MissForest benchmark
# -----------------------------
best_json_path = (
    Path(config_paths.experiments_dir(columns_num))
    / "Optimization"
    / "best_params"
    / f"{study_name}_best_params.json"
)

print("\nRunning final benchmark using:")
print(best_json_path)

run_final_with_best(
    best_json_path=str(best_json_path),
    file_path=DATA_FILE,
    columns_num=columns_num,
    fractions=final_fractions,
    missingness_types=final_missingness_types,
    seed=42,
    verbose_method=True,
)

# -----------------------------
# 5. Plot with your normal evaluator
# -----------------------------
missforest_eval(
    fractions=final_fractions,
    missingness_types=final_missingness_types,
    columns_num=columns_num,
)