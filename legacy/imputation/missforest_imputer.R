# imputation/missforest_imputer.R

suppressPackageStartupMessages({
  library(Metrics)
  library(missRanger)
})

mr_imputer <- function(
  df_with_na,
  original_df,
  iter,
  tree,
  max_depth,
  sample_fraction,
  seed,
  rmse_path,
  imputed_data_path
) {

  cat("\n================ MissForest / missRanger imputation ================\n")
  cat("iter:", iter, "\n")
  cat("num.trees:", tree, "\n")
  cat("max.depth:", max_depth, "\n")
  cat("sample.fraction:", sample_fraction, "\n")
  cat("seed:", seed, "\n")
  cat("====================================================================\n\n")

  if (ncol(df_with_na) < 3) {
    stop("Input corrupted data must contain at least eid, label, and one protein column.")
  }

  if (ncol(original_df) != ncol(df_with_na)) {
    stop("Corrupted data and normalized data have different numbers of columns.")
  }

  if (!all(colnames(original_df) == colnames(df_with_na))) {
    stop("Corrupted data and normalized data column names are not identical.")
  }

  # First two columns are metadata: eid and label.
  eid_label_df <- df_with_na[, 1:2, drop = FALSE]
  df_with_na_numeric <- df_with_na[, -c(1, 2), drop = FALSE]
  original_df_numeric <- original_df[, -c(1, 2), drop = FALSE]

  # Ensure numeric protein matrix.
  df_with_na_numeric <- as.data.frame(lapply(df_with_na_numeric, function(x) as.numeric(x)))
  original_df_numeric <- as.data.frame(lapply(original_df_numeric, function(x) as.numeric(x)))

  missing_indices <- which(is.na(as.matrix(df_with_na_numeric)))

  if (length(missing_indices) == 0) {
    stop("No missing numeric protein entries found in corrupted data. Nothing to impute/evaluate.")
  }

  set.seed(as.integer(seed))

  cat("Running missRanger imputation...\n")

  imputed_mf_numeric <- missRanger(
    data            = df_with_na_numeric,
    num.trees       = as.integer(tree),
    max.depth       = as.integer(max_depth),
    sample.fraction = as.numeric(sample_fraction),
    maxiter         = as.integer(iter),
    seed            = as.integer(seed),
    verbose         = 1
  )

  cat("Calculating RMSE for MissForest imputation...\n")

  original_vals <- as.numeric(as.matrix(original_df_numeric))
  imputed_vals <- as.numeric(as.matrix(imputed_mf_numeric))

  rmse_mf <- rmse(
    original_vals[missing_indices],
    imputed_vals[missing_indices]
  )

  cat("RMSE for MissForest Imputation:", rmse_mf, "\n")

  imputed_mf <- cbind(eid_label_df, imputed_mf_numeric)

  write.csv(data.frame(rmse_mf = rmse_mf), file = rmse_path, row.names = FALSE)
  write.csv(imputed_mf, file = imputed_data_path, row.names = FALSE)

  cat("RMSE saved to:", rmse_path, "\n")
  cat("Imputed data saved to:", imputed_data_path, "\n")
}


args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 9) {
  stop(
    paste0(
      "Expected 9 arguments:\n",
      "1 corrupted_data_path\n",
      "2 normalized_data_path\n",
      "3 iter\n",
      "4 ntree\n",
      "5 max_depth\n",
      "6 sample_fraction\n",
      "7 seed\n",
      "8 rmse_path\n",
      "9 imputed_data_path\n",
      "Received ", length(args), " arguments."
    )
  )
}

corrupted_data_path  <- args[1]
normalized_data_path <- args[2]
iter                 <- as.integer(args[3])
ntree                <- as.integer(args[4])
max_depth            <- as.integer(args[5])
sample_fraction      <- as.numeric(args[6])
seed                 <- as.integer(args[7])
rmse_path            <- args[8]
imputed_data_path    <- args[9]

if (!file.exists(corrupted_data_path)) {
  stop(paste("Corrupted data file does not exist:", corrupted_data_path))
}

if (!file.exists(normalized_data_path)) {
  stop(paste("Normalized data file does not exist:", normalized_data_path))
}

df_with_na  <- read.csv(corrupted_data_path, check.names = FALSE)
original_df <- read.csv(normalized_data_path, check.names = FALSE)

cat("Corrupted data:", corrupted_data_path, "\n")
cat("Normalized data:", normalized_data_path, "\n")
cat("Imputed data will be saved to:", imputed_data_path, "\n")

mr_imputer(
  df_with_na        = df_with_na,
  original_df       = original_df,
  iter              = iter,
  tree              = ntree,
  max_depth         = max_depth,
  sample_fraction   = sample_fraction,
  seed              = seed,
  rmse_path         = rmse_path,
  imputed_data_path = imputed_data_path
)