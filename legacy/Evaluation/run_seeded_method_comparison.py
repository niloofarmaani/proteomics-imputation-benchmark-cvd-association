import config_paths


config_paths.set_biobank_root("/home/maani/niloo/imputationproject")
config_paths.set_experiments_name("Experiments-mayovasc-proteomic-ccs")

from Evaluation.optuna_boxplot_eval import compare_imputation_methods


methods_to_compare = [
    "KNN",
    "softImpute",
    "VAE",
    "DAE",
    "AE",
    "mf",
    "GAIN",
    "RNN",
    "iterative",
    "Mean",
]

# Exact study names are required. Mean has no study and is omitted from this mapping.
study_names = {
    "KNN": "KNN_balanced_MAR_MNAR_10_30_seeded_v1",
    "softImpute": "SoftImpute_balanced_MAR_MNAR_10_30_seeded_v1",
    "VAE": "VAE_balanced_MAR_MNAR_10_30_seeded_v1",
    "DAE": "DAE_balanced_MAR_MNAR_10_30_seeded_v1",
    "AE": "AE_balanced_MAR_MNAR_10_30_seeded_v1",
    "mf": "MissForest_balanced_MAR_MNAR_10_30_seeded_v1",
    "GAIN": "GAIN_balanced_MAR_MNAR_10_30_seeded_v1",
    "RNN": "RNN_balanced_MAR_MNAR_10_30_seeded_v1",
    "iterative": "Iterative_balanced_MAR_MNAR_10_30_seeded_v1",
}

hyperparameter_values, seed_values, output_paths = compare_imputation_methods(
    methods_to_compare=methods_to_compare,
    study_names=study_names,
    columns_num="all",
    optimization_seed=42,

    # Hyperparameter-stage comparison, fixed seed.
    hyperparameter_fractions=[0.1, 0.3],
    hyperparameter_missingness_types=["MAR", "MNAR"],

    # Final frozen-best comparison across seeds.
    final_fractions=[0.1, 0.3, 0.5],
    final_missingness_types=["MCAR", "MAR", "MNAR"],
    seeds=[42, 100, 2024, 777, 999],

    # Use "input" to keep the exact method order above, or median/min/mean to sort.
    sort_by="median",

    # Full-scale and zoomed panels. No values are deleted.
    hyperparameter_zoom_y_range=(0.24, 0.38),
    seed_zoom_y_range=(0.24, 0.38),
    full_y_tick_step=None,
    zoom_y_tick_step=0.01,

    # Five seed values are easier to inspect when shown as dots over each box.
    hyperparameter_show_points=False,
    seed_show_points=True,

    # Must match the Optuna validation experiment settings.
    mean_val_rate=0.05,
    mean_validation_mode="stratified_quantile",

    show=True,
    save_pdf=True,
    save_combined_pdf=True,
    combined_pdf_name="imputation_hyperparameter_and_seed_comparison.pdf",
    include_title_page=True,
)

print("\nOutput paths:")
print(output_paths)
