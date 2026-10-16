import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple
from itertools import product
import copy
import sys
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Repro / device
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Constants (avoid device issues)
# -----------------------------
LOG_2PI = float(np.log(2.0 * np.pi))
LOG2 = float(np.log(2.0))
leaky_relu_default = nn.LeakyReLU(0.1)

# -----------------------------
# VAE (unchanged architecture)
# -----------------------------
class VAE(nn.Module):
    def __init__(self, n_features: int, n_neurons: List[int], activation=leaky_relu_default,
                 last_decoder_activation=None, dim_latent: int = 10):
        super().__init__()
        self.n_features, self.n_neurons = n_features, list(n_neurons)
        self.layers = [n_features, *self.n_neurons]
        self.dim_latent = dim_latent

        def build_layer(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Dropout(0.1), nn.BatchNorm1d(out_feat), activation]

        self.encoder = []
        for i in range(len(self.layers) - 1):
            in_feat, out_feat = self.layers[i:i + 2]
            self.encoder.extend(build_layer(in_feat=in_feat, out_feat=out_feat))
        self.encoder.append(nn.Linear(out_feat, dim_latent * 2))
        self.encoder = nn.Sequential(*self.encoder)

        self.layers_decoder = self.layers[::-1]
        self.decoder = build_layer(in_feat=self.dim_latent, out_feat=self.layers_decoder[0])
        i = -1
        for i in range(len(self.layers_decoder) - 2):
            in_feat, out_feat = self.layers_decoder[i:i + 2]
            self.decoder.extend(build_layer(in_feat=in_feat, out_feat=out_feat))
        in_feat, out_feat = self.layers_decoder[i + 1:i + 3]
        self.decoder.append(nn.Linear(in_feat, out_feat * 2))
        self.decoder = nn.Sequential(*self.decoder)

    def encode(self, x):
        z_params = self.encoder(x)
        z_mu = z_params[:, :self.dim_latent]
        z_logvar = z_params[:, self.dim_latent:]
        return z_mu, z_logvar

    def decode(self, z):
        x_params = self.decoder(z)
        x_mu = x_params[:, :self.n_features]
        x_logvar = x_params[:, self.n_features:]
        return x_mu, x_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.decode(z)
        return x_mu, x_logvar, z_mu, z_logvar

# -----------------------------
# Loss (same logic, constants safer)
# -----------------------------
def compute_kld(z_mu, z_logvar):
    return 0.5 * (z_mu**2 + torch.exp(z_logvar) - 1 - z_logvar)

def gaussian_log_prob(z, mu, logvar):
    # same formula as yours
    return -0.5 * (LOG_2PI + logvar + (z - mu)**2 / torch.exp(logvar))

def loss_fct(pred, y, mask, epoch, warmup_epochs=50, reduction='sum',
             results: List = None, freebits=0.05):
    x_mu, x_logvar, z_mu, z_logvar = pred
    batch = y

    masked_log_prob = gaussian_log_prob(batch[mask], x_mu[mask], x_logvar[mask])
    l_rec = -torch.sum(masked_log_prob)

    l_reg = torch.sum((F.relu(compute_kld(z_mu, z_logvar) - freebits * LOG2) + freebits * LOG2), 1)

    kl_weight = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0

    if results is not None:
        results.append((l_rec.item(), torch.mean(l_reg).item()))

    return (l_rec / y.shape[0]) + (kl_weight * torch.mean(l_reg))

# -----------------------------
# RMSE helpers (KEEP SAME RMSE FORMULA)
# -----------------------------
def rmse_on_mask(original_values: np.ndarray, imputed_values: np.ndarray, mask_values: np.ndarray) -> float:
    """
    Matches your grid-search RMSE:
      masked_original = original[mask]
      masked_final    = imputed[mask]
      rmse = sqrt(mean((masked_original - masked_final)^2))
    """
    masked_original = original_values[mask_values]
    masked_final = imputed_values[mask_values]
    return float(np.sqrt(np.mean((masked_original - masked_final) ** 2)))

def rmse_r_style(original_values: np.ndarray, imputed_values: np.ndarray, missing_mask_values: np.ndarray) -> float:
    """
    Matches your final RMSE block as intended (flatten-safe).
    You compute RMSE on indices where data was NOT missing: where(~mask).
    """
    mask_subset = missing_mask_values
    # make it robust regardless of mask dimensionality
    mask_flat = mask_subset.flatten() if mask_subset.ndim > 1 else mask_subset
    valid_indices = np.where(~mask_flat)

    original_vals = original_values.flatten()
    imputed_vals = imputed_values.flatten()
    return float(np.sqrt(np.mean((original_vals[valid_indices] - imputed_vals[valid_indices]) ** 2)))

# -----------------------------
# Train/val/test split
# -----------------------------
def split_rows(n: int, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx

# -----------------------------
# Core training routine (early stop on VAL metric)
# -----------------------------
def train_with_early_stopping(
    vae: VAE,
    train_loader: DataLoader,
    val_data_tensor: torch.Tensor,
    val_missing_mask: np.ndarray,
    val_original_values: np.ndarray,
    freebits: float,
    learning_rate: float,
    warmup_epochs: int,
    epochs: int,
    patience: int,
    device: torch.device,
) -> Dict:
    vae.to(device)

    optimizer = optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate / 20,
        max_lr=learning_rate,
        step_size_up=100,
        mode='triangular',
        cycle_momentum=False
    )

    best_metric = float("inf")  # we minimize val RMSE on missing entries
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0.0
        batches = 0

        for data_batch, obs_mask_batch in train_loader:
            data_batch = data_batch.to(device)
            obs_mask_batch = obs_mask_batch.to(device)

            optimizer.zero_grad()
            x_mu, x_logvar, z_mu, z_logvar = vae(data_batch)

            batch_results = []
            loss = loss_fct(
                (x_mu, x_logvar, z_mu, z_logvar),
                data_batch,
                obs_mask_batch,
                epoch,
                warmup_epochs=warmup_epochs,
                freebits=freebits,
                results=batch_results
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            batches += 1

        avg_train_loss = epoch_loss / max(1, batches)

        # ----- validation RMSE on missing entries -----
        vae.eval()
        with torch.no_grad():
            x_mu_val, _, _, _ = vae(val_data_tensor.to(device))
            val_recon = x_mu_val.detach().cpu().numpy()

        # IMPORTANT: RMSE computation stays exactly sqrt(mean((a-b)^2)) on masked entries
        val_rmse_missing = rmse_on_mask(
            original_values=val_original_values,
            imputed_values=val_recon,
            mask_values=val_missing_mask
        )

        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_rmse_missing": val_rmse_missing})

        # early stopping on validation RMSE (imputation-focused)
        if val_rmse_missing < best_metric:
            best_metric = val_rmse_missing
            best_state = copy.deepcopy(vae.state_dict())  # CRITICAL FIX
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return {
        "best_val_rmse_missing": best_metric,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "history": history
    }

# -----------------------------
# Full imputer with logical eval
# -----------------------------
def VAE_imputer2(
    df_with_na: pd.DataFrame,
    original_df: pd.DataFrame,
    missing_mask: pd.DataFrame,
    param_grid: Dict,
    rmse_path: str,
    imputed_data_path: str,
    epochs: int = 500,
    patience: int = 100,
    warmup_epochs: int = 50,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
):
    set_seed(seed)
    device = get_device()

    # 1) separate first two columns (eid, label)
    eid_label_df = df_with_na.iloc[:, :2]
    df_with_na_numeric = df_with_na.iloc[:, 2:]
    original_df_numeric = original_df.iloc[:, 2:]
    missing_mask_numeric = missing_mask.iloc[:, 2:]

    # 2) split rows
    n = len(df_with_na_numeric)
    train_idx, val_idx, test_idx = split_rows(n, seed=seed, train_frac=train_frac, val_frac=val_frac)

    # 3) preprocess: fill NaNs (compute fill value from TRAIN ONLY to avoid leakage)
    data = df_with_na_numeric.to_numpy()
    nan_mask = np.isnan(data)  # True where missing in corrupted data

    train_data = data[train_idx]
    percentile_1 = np.nanpercentile(train_data, 1)
    data_imputed = np.where(nan_mask, percentile_1, data)
    data_tensor = torch.tensor(data_imputed, dtype=torch.float32)          # CPU tensor
    obs_mask_tensor = torch.tensor(~nan_mask, dtype=torch.bool)            # True where observed

    # masks/ground truth arrays for RMSE (numpy)
    original_values = original_df_numeric.to_numpy()
    missing_mask_values = missing_mask_numeric.to_numpy().astype(bool)

    # sanity warning (common silent failure in papers)
    # training loss uses NaNs from df_with_na; RMSE uses missing_mask file
    file_mask_matches_nans = np.array_equal(missing_mask_values, nan_mask)
    if not file_mask_matches_nans:
        warnings.warn(
            "missing_mask file does NOT exactly match NaNs in corrupted data. "
            "Training will mask by NaNs; RMSE will use provided missing_mask."
        )

    # 4) grid search (choose best on VALIDATION missing-RMSE)
    results = []
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    for idx_combo, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        print(f"\n[{idx_combo+1}/{len(param_combinations)}] Training with params: {param_dict}")

        # build model for THIS combo
        vae = VAE(
            n_features=df_with_na_numeric.shape[1],
            n_neurons=param_dict['n_neurons'],
            dim_latent=param_dict['dim_latent']
        )

        # DataLoader for train
        train_dataset = TensorDataset(data_tensor[train_idx], obs_mask_tensor[train_idx])
        # drop_last=True reduces BatchNorm edge cases
        train_loader = DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True, drop_last=True)

        # Validation tensors (full-batch forward is usually ok; adjust if huge)
        val_data_tensor = data_tensor[val_idx]
        val_original = original_values[val_idx]
        val_missing_mask = missing_mask_values[val_idx]

        train_out = train_with_early_stopping(
            vae=vae,
            train_loader=train_loader,
            val_data_tensor=val_data_tensor,
            val_missing_mask=val_missing_mask,
            val_original_values=val_original,
            freebits=param_dict['freebits'],
            learning_rate=param_dict['learning_rate'],
            warmup_epochs=warmup_epochs,
            epochs=epochs,
            patience=patience,
            device=device
        )

        results.append({
            "params": param_dict,
            "best_val_rmse_missing": train_out["best_val_rmse_missing"],
            "best_epoch": train_out["best_epoch"],
            "history": train_out["history"],
            "best_state": train_out["best_state"]
        })

        print(f"Done. Best VAL RMSE (missing entries): {train_out['best_val_rmse_missing']:.4f} at epoch {train_out['best_epoch']}")

    best = min(results, key=lambda r: r["best_val_rmse_missing"])
    best_params = best["params"]
    print(f"\nBest params by VAL missing-RMSE: {best_params}")
    print(f"Best VAL missing-RMSE: {best['best_val_rmse_missing']:.4f}")

    # 5) evaluate on TEST using the best saved state (no tuning on test)
    best_model = VAE(
        n_features=df_with_na_numeric.shape[1],
        n_neurons=best_params['n_neurons'],
        dim_latent=best_params['dim_latent']
    ).to(device)
    best_model.load_state_dict(best["best_state"])
    best_model.eval()

    with torch.no_grad():
        x_mu_test, _, _, _ = best_model(data_tensor[test_idx].to(device))
        test_recon = x_mu_test.detach().cpu().numpy()

    test_original = original_values[test_idx]
    test_missing_mask = missing_mask_values[test_idx]

    # KEEP your RMSE formula; compute explicitly what it means:
    test_rmse_missing = rmse_on_mask(test_original, test_recon, test_missing_mask)   # imputation RMSE
    test_rmse_observed_rstyle = rmse_r_style(test_original, test_recon, test_missing_mask)  # your final-style rmse block

    print(f"\nTEST RMSE on missing entries (imputation): {test_rmse_missing:.4f}")
    print(f"TEST RMSE on observed entries (your R-style final block): {test_rmse_observed_rstyle:.4f}")

    # 6) (optional, but typical) retrain on FULL data using best hyperparams, then impute and save
    # For “deployment” imputation you normally use all rows.
    final_model = VAE(
        n_features=df_with_na_numeric.shape[1],
        n_neurons=best_params['n_neurons'],
        dim_latent=best_params['dim_latent']
    ).to(device)

    full_dataset = TensorDataset(data_tensor, obs_mask_tensor)
    full_loader = DataLoader(full_dataset, batch_size=best_params['batch_size'], shuffle=True, drop_last=True)

    # train for best_epoch epochs (paper-consistent)
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
    scheduler = CyclicLR(
        optimizer,
        base_lr=best_params['learning_rate'] / 20,
        max_lr=best_params['learning_rate'],
        step_size_up=100,
        mode='triangular',
        cycle_momentum=False
    )

    for epoch in range(max(1, best["best_epoch"])):
        final_model.train()
        for xb, mb in full_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            optimizer.zero_grad()
            pred = final_model(xb)
            loss = loss_fct(pred, xb, mb, epoch, warmup_epochs=warmup_epochs, freebits=best_params['freebits'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

    final_model.eval()
    with torch.no_grad():
        x_mu_all, _, _, _ = final_model(data_tensor.to(device))
        reconstructed_all = x_mu_all.detach().cpu().numpy()

    df_imputed_numeric = pd.DataFrame(reconstructed_all, columns=df_with_na_numeric.columns)
    imputed_vae = pd.concat([eid_label_df, df_imputed_numeric], axis=1)

    # 7) Save RMSE outputs (clear names)
    # (still your RMSE formula; now just clearly labeled)
    out_rmse = pd.DataFrame({
        "test_rmse_missing_imputation": [test_rmse_missing],
        "test_rmse_observed_rstyle": [test_rmse_observed_rstyle],
        "best_val_rmse_missing": [best["best_val_rmse_missing"]],
        "seed": [seed],
    })
    out_rmse.to_csv(rmse_path, index=False)
    print(f"\nRMSE summary saved to {rmse_path}")

    imputed_vae.to_csv(imputed_data_path, index=False)
    print(f"Imputed data saved to {imputed_data_path}")

    return imputed_vae, out_rmse, results, best

# -----------------------------
# Main
# -----------------------------
args = sys.argv[1:]
if len(args) != 6:
    print("Usage: python script.py corrupted_data_path normalized_data_path missing_mask_path param_grid rmse_path imputed_data_path")
    sys.exit(1)

corrupted_data_path = args[0]
normalized_data_path = args[1]
missing_mask_path = args[2]
param_grid_str = args[3]
rmse_path = args[4]
imputed_data_path = args[5]

# Parse param_grid from string
batch_sizes, lrs, neurons, dim_latents, freebits = param_grid_str.split('|')
param_grid = {
    'batch_size': [int(x) for x in batch_sizes.split(',')],
    'learning_rate': [float(x) for x in lrs.split(',')],
    'n_neurons': [[int(n) for n in x.split('-')] for x in neurons.split(',')],
    'dim_latent': [int(x) for x in dim_latents.split(',')],
    'freebits': [float(x) for x in freebits.split(',')]
}

df_with_na = pd.read_csv(corrupted_data_path)
original_df = pd.read_csv(normalized_data_path)
missing_mask = pd.read_csv(missing_mask_path)

VAE_imputer2(
    df_with_na=df_with_na,
    original_df=original_df,
    missing_mask=missing_mask,
    param_grid=param_grid,
    rmse_path=rmse_path,
    imputed_data_path=imputed_data_path,
    epochs=500,
    patience=100,
    warmup_epochs=50,
    seed=42,
    train_frac=0.8,
    val_frac=0.1
)
