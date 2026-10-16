# Load necessary libraries
library(Metrics)
library(impute)
# Removed missRanger/missForest/VIM as they aren't used in this specific KNN script

# Define the function to perform KNN and calculate RMSE
KNN_imputer <- function(df_with_na, original_df, kn, rmse_path, imputed_data_path) {

  # 1) Separate the first two columns (assumed to be 'eid' and 'label')
  eid_label_df <- df_with_na[, 1:2]
  df_with_na_numeric <- df_with_na[, -c(1,2)] 

  # Do the same for the original data to keep shapes consistent
  original_df_numeric <- original_df[, -c(1,2)]
  
  # 2) Perform KNN Imputation on the numeric portion
  # Convert to matrix since impute.knn() expects a matrix
  imputed_knn_numeric <- impute.knn(
    as.matrix(df_with_na_numeric),
    k = kn,
    rowmax = 0.9  # adjust rowmax if needed
  )$data

  # 3) Calculate RMSE dynamically
  # Since the original_df is complete (no native missing values), 
  # the simulated missing entries are exactly where df_with_na_numeric is NA.
  missing_indices <- which(is.na(df_with_na_numeric))

  # Flatten both numeric matrices into vectors to compare them
  original_vals <- as.numeric(as.matrix(original_df_numeric))
  imputed_vals <- as.numeric(as.matrix(imputed_knn_numeric))

  # Calculate RMSE strictly on the cells that were imputed
  rmse_knn <- rmse(
    original_vals[missing_indices],
    imputed_vals[missing_indices]
  )

  # Print and save RMSE results
  cat("RMSE for KNN Imputation (k =", kn, "):", rmse_knn, "\n")
  write.csv(
    data.frame(rmse_knn = rmse_knn),
    file = rmse_path,
    row.names = FALSE
  )

  # 4) Reattach the first two columns (eid, label) to the imputed numeric columns
  imputed_knn <- cbind(eid_label_df, imputed_knn_numeric)

  # 5) Save the final DataFrame (with eid/label restored) to a CSV
  write.csv(imputed_knn, file = imputed_data_path, row.names = FALSE)
  cat("Imputed data saved to:", imputed_data_path, "\n")
}

# --- Main execution ---

# Gather command line arguments
args <- commandArgs(trailingOnly = TRUE)
corrupted_data_path <- args[1]
normalized_data_path <- args[2]
kn <- args[3]
rmse_path <- args[4]
imputed_data_path <- args[5]

cat("\nStarting KNN Imputation...\n")

# Read in the CSVs
df_with_na <- read.csv(corrupted_data_path)
original_df <- read.csv(normalized_data_path)

# Call the function to perform imputation and calculate RMSE
KNN_imputer(
  df_with_na,
  original_df,
  as.integer(kn),
  rmse_path,
  imputed_data_path
)