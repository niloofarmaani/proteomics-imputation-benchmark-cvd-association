import sys
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error


def softimpute_imputer(
    df_with_na,
    original_df,
    missing_mask,
    rank_max,
    maxit,
    rmse_path,
    imputed_data_path
):
    print("Starting softImpute-style imputation (Python + RMSE)...")

    # --------------------------------------------------
    # 1) Feature matrix only (drop eid / label)
    # --------------------------------------------------
    feature_cols = [c for c in df_with_na.columns if c not in ("eid", "label")]

    X = df_with_na[feature_cols].to_numpy(dtype=float)
    X_orig = original_df[feature_cols].to_numpy(dtype=float)

    print("Matrix shape:", X.shape)
    print("Initial NA count:", np.isnan(X).sum())

    # --------------------------------------------------
    # 2) Missing mask (same semantics as IterativeImputer)
    # --------------------------------------------------
    mask = missing_mask[feature_cols].to_numpy().astype(bool)
    valid_indices = np.where(~mask)

    print("Eval positions (RMSE):", len(valid_indices[0]))

    # --------------------------------------------------
    # 3) Initialize missing values with column means
    # --------------------------------------------------
    obs = ~np.isnan(X)

    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)

    X_filled = X.copy()
    for j in range(X.shape[1]):
        X_filled[~obs[:, j], j] = col_means[j]

    # --------------------------------------------------
    # 4) Iterative low-rank approximation (softImpute-style)
    # --------------------------------------------------
    rank_use = min(rank_max, X.shape[0] - 1, X.shape[1] - 1)
    if rank_use < 1:
        raise ValueError("rank_max too small for matrix dimensions")

    for it in range(maxit):
        print(f"Iteration {it + 1}")

        U, s, Vt = svds(X_filled, k=rank_use)
        idx = np.argsort(s)[::-1]
        U, s, Vt = U[:, idx], s[idx], Vt[idx, :]

        M = U @ np.diag(s) @ Vt
        X_filled[~obs] = M[~obs]

    print("Imputation finished.")
    print("Final NA count:", np.isnan(X_filled).sum())

    # --------------------------------------------------
    # 5) Compute RMSE (IDENTICAL LOGIC to IterativeImputer)
    # --------------------------------------------------
    print("Calculating RMSE...")
    rmse = mean_squared_error(
        X_orig[valid_indices],
        X_filled[valid_indices],
        squared=False,
    )

    print(f"RMSE: {rmse}")

    pd.DataFrame({"best_rmse": [rmse]}).to_csv(rmse_path, index=False)

    # --------------------------------------------------
    # 6) Save imputed data
    # --------------------------------------------------
    imputed_df = pd.DataFrame(X_filled, columns=feature_cols)
    imputed_df.to_csv(imputed_data_path, index=False)

    print(f"RMSE saved to {rmse_path}")
    print(f"Imputed data saved to {imputed_data_path}")


if __name__ == "__main__":
    # --------------------------------------------------
    # CLI interface (UNCHANGED)
    # --------------------------------------------------
    corrupted_data_path = sys.argv[1]
    normalized_data_path = sys.argv[2]
    missing_mask_path = sys.argv[3]
    rank_max = int(sys.argv[4])
    maxit = int(sys.argv[5])
    rmse_path = sys.argv[6]
    imputed_data_path = sys.argv[7]

    df_with_na = pd.read_csv(corrupted_data_path)
    original_df = pd.read_csv(normalized_data_path)
    missing_mask = pd.read_csv(missing_mask_path)

    softimpute_imputer(
        df_with_na,
        original_df,
        missing_mask,
        rank_max,
        maxit,
        rmse_path,
        imputed_data_path,
    )
