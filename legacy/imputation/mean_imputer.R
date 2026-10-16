# Load necessary libraries
suppressPackageStartupMessages(library(Metrics))

# Define the function to perform Mean imputation and calculate RMSE
mean_imputer <- function(df_with_na, original_df, rmse_path, imputed_data_path) {
  
  # 1. Separate first two columns (assumed eid, label)
  non_imputed_cols <- df_with_na[, 1:2]  
  df_impute <- df_with_na[, -c(1,2)]     
  
  original_df_numeric <- original_df[, -c(1,2)]
  
  # --- DYNAMIC MASK CALCULATION ---
  # Find exactly where the NA values are BEFORE we fill them with means
  missing_indices <- which(is.na(df_impute))
  
  # 2. Perform mean imputation column by column
  for (col in seq_along(df_impute)) {
    col_mean <- mean(df_impute[[col]], na.rm = TRUE)  # Calculate mean of each column, ignoring NA
    df_impute[[col]][is.na(df_impute[[col]])] <- col_mean
  }
  
  # 3. Calculate RMSE for mean imputation dynamically
  cat("Calculating RMSE for Mean Imputation...\n")
  rmse_mean <- rmse(
    as.numeric(as.matrix(original_df_numeric))[missing_indices],  
    as.numeric(as.matrix(df_impute))[missing_indices]
  )
  
  # Print RMSE results
  cat("RMSE for Mean Imputation: ", rmse_mean, "\n")
  
  # Save RMSE value to CSV
  write.csv(data.frame(rmse_mean = rmse_mean), file = rmse_path, row.names = FALSE)
  
  # 4. Reattach the first two columns back to the imputed dataset
  df_final <- cbind(non_imputed_cols, df_impute)
  
  # 5. Save the mean-imputed data to CSV
  write.csv(df_final, file = imputed_data_path, row.names = FALSE)
  print(paste("Mean-imputed data saved to", imputed_data_path))
}

# ------------------------------------
# Main Execution via Command Args
# ------------------------------------
args <- commandArgs(trailingOnly = TRUE)

# We expect 4 arguments now:
corrupted_data_path  <- args[1]
normalized_data_path <- args[2]
rmse_path            <- args[3]
imputed_data_path    <- args[4]

# Read data
df_with_na  <- read.csv(corrupted_data_path)
original_df <- read.csv(normalized_data_path)

# Call the function to perform mean imputation and calculate RMSE
mean_imputer(df_with_na, original_df, rmse_path, imputed_data_path)