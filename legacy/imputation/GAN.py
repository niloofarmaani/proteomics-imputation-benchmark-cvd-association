import os
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

import sys as _sys

_HERE = Path(__file__).resolve() if "__file__" in globals() else Path.cwd().resolve()
_candidate_roots = []
for p in [
    _HERE.parents[1] if len(_HERE.parents) > 1 else None,
    _HERE.parents[2] if len(_HERE.parents) > 2 else None,
]:
    if p is not None:
        _candidate_roots.append(p)
        _candidate_roots.append(p / "code")

for _root in _candidate_roots:
    try:
        if (_root / "config_paths.py").exists():
            if str(_root) not in _sys.path:
                _sys.path.insert(0, str(_root))
            break
    except Exception:
        pass

try:
    from config_paths import experiments_dir  # type: ignore
except Exception as e:
    experiments_dir = None  # type: ignore
    _IMPORT_ERROR = e


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_2PI = float(np.log(2.0 * np.pi))
LOG2 = float(np.log(2.0))


def compute_kld(z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (z_mu**2 + torch.exp(z_logvar) - 1 - z_logvar)


def gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (LOG_2PI + logvar + (z - mu) ** 2 / torch.exp(logvar))


def loss_fct(pred, y, mask, epoch, warmup_epochs, reduction="sum", results=None, freebits=0.05):
    x_mu, x_logvar, z_mu, z_logvar = pred
    masked_log_prob = gaussian_log_prob(y[mask], x_mu[mask], x_logvar[mask])
    l_rec = -torch.sum(masked_log_prob)
    l_reg = torch.sum((F.relu(compute_kld(z_mu, z_logvar) - freebits * LOG2) + freebits * LOG2), 1)
    kl_weight = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
    return (l_rec / y.shape[0]) + (kl_weight * torch.mean(l_reg))


def rmse_like_baseline(original_values: np.ndarray, final_values: np.ndarray, missing_mask_values: np.ndarray) -> float:
    masked_original = original_values[missing_mask_values]
    masked_final = final_values[missing_mask_values]
    return float(np.sqrt(np.mean((masked_original - masked_final) ** 2)))


def resolve_missing_mask_numeric(missing_mask: pd.DataFrame, df_with_na: pd.DataFrame, df_with_na_numeric: pd.DataFrame) -> np.ndarray:
    if missing_mask.shape[1] == df_with_na_numeric.shape[1]:
        mm = missing_mask
    elif missing_mask.shape[1] == df_with_na.shape[1]:
        mm = missing_mask.iloc[:, 2:]
    else:
        mm = missing_mask.iloc[:, -df_with_na_numeric.shape[1] :]
    return mm.to_numpy().astype(bool)


def fill_only_nans_like_baseline(df_with_na_numeric: pd.DataFrame, df_imputed_numeric: pd.DataFrame, nan_mask: np.ndarray) -> pd.DataFrame:
    mask_df = pd.DataFrame(nan_mask, index=df_with_na_numeric.index, columns=df_with_na_numeric.columns)
    df_final = df_with_na_numeric.copy()
    df_final = pd.DataFrame(
        np.where(mask_df, df_imputed_numeric, df_final),
        index=df_with_na_numeric.index,
        columns=df_with_na_numeric.columns,
    )
    return df_final


def write_rmse_csv(rmse_path: str, rmse_value: float) -> None:
    pd.DataFrame({"rmse_vae": [rmse_value]}).to_csv(rmse_path, index=False)


def split_rows(n: int, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def hint_matrix(m: torch.Tensor, hint_rate: float) -> torch.Tensor:
    b = (torch.rand_like(m) < hint_rate).float()
    return b * m + 0.5 * (1.0 - b)


class Generator(nn.Module):
    def __init__(self, n_features: int, hidden: List[int]):
        super().__init__()
        dims = [n_features * 2] + hidden + [n_features * 2]  # mu + logvar
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self.n_features = n_features

    def forward(self, x_tilde: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([x_tilde, m], dim=1))
        mu = out[:, : self.n_features]
        logvar = out[:, self.n_features :]
        return mu, logvar


class Discriminator(nn.Module):
    def __init__(self, n_features: int, hidden: List[int]):
        super().__init__()
        dims = [n_features * 2] + hidden + [n_features]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([x_hat, h], dim=1)))


def GAN_imputer_papergrade(
    df_with_na: pd.DataFrame,
    original_df: pd.DataFrame,
    missing_mask: pd.DataFrame,
    batch_size: int,
    learning_rate: float,
    freebits: float,
    gen_hidden: List[int],
    disc_hidden: List[int],
    hint_rate: float,
    alpha: float,
    rmse_path: str,
    imputed_data_path: str,
    epochs: int,
    patience: int,
    warmup_epochs: int,
    seed: int = 42,
):
    set_seed(seed)
    device = get_device()

    eid_label_df = df_with_na.iloc[:, :2]
    df_with_na_numeric = df_with_na.iloc[:, 2:]
    original_df_numeric = original_df.iloc[:, 2:]
    original_values = original_df_numeric.to_numpy()

    data = df_with_na_numeric.to_numpy()
    nan_mask = np.isnan(data)
    missing_mask_values = resolve_missing_mask_numeric(missing_mask, df_with_na, df_with_na_numeric)

    percentile_1 = np.nanpercentile(data, 1)
    x_np = np.where(nan_mask, percentile_1, data).astype(np.float32)

    m_np = (~nan_mask).astype(np.float32)  # 1 observed, 0 missing

    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    m_tensor = torch.tensor(m_np, dtype=torch.float32)

    n = len(df_with_na_numeric)
    train_idx, val_idx, _ = split_rows(n, seed=seed)

    train_ds = TensorDataset(x_tensor[train_idx], m_tensor[train_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    G = Generator(n_features=x_np.shape[1], hidden=gen_hidden).to(device)
    D = Discriminator(n_features=x_np.shape[1], hidden=disc_hidden).to(device)

    opt_g = optim.Adam(G.parameters(), lr=learning_rate)
    opt_d = optim.Adam(D.parameters(), lr=learning_rate)

    best_val = float("inf")
    best_epoch = 1
    patience_counter = 0

    for epoch in range(epochs):
        G.train()
        D.train()

        for xb, mb in train_loader:
            xb = xb.to(device)
            mb = mb.to(device)

            z = torch.rand_like(xb)
            x_tilde = mb * xb + (1.0 - mb) * z

            # Train D
            with torch.no_grad():
                g_mu, g_logvar = G(x_tilde, mb)
                x_hat = mb * xb + (1.0 - mb) * g_mu

            h = hint_matrix(mb, hint_rate)
            d_prob = D(x_hat, h)

            d_loss = -torch.mean(mb * torch.log(d_prob + 1e-8) + (1.0 - mb) * torch.log(1.0 - d_prob + 1e-8))

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Train G
            g_mu, g_logvar = G(x_tilde, mb)
            x_hat = mb * xb + (1.0 - mb) * g_mu
            h = hint_matrix(mb, hint_rate)
            d_prob = D(x_hat, h)

            g_adv = -torch.mean((1.0 - mb) * torch.log(d_prob + 1e-8))

            obs_mask_bool = mb.bool()
            z_mu = torch.zeros((xb.shape[0], 1), device=device)
            z_logvar = torch.zeros_like(z_mu)
            g_rec = loss_fct((g_mu, g_logvar, z_mu, z_logvar), xb, obs_mask_bool, epoch, warmup_epochs, freebits=freebits)

            g_loss = g_adv + alpha * g_rec

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        # validation RMSE (deterministic z=0 for stability)
        G.eval()
        with torch.no_grad():
            xv = x_tensor[val_idx].to(device)
            mv = m_tensor[val_idx].to(device)
            z = torch.zeros_like(xv)
            x_tilde_v = mv * xv + (1.0 - mv) * z
            v_mu, _ = G(x_tilde_v, mv)
            val_recon = v_mu.detach().cpu().numpy()

        val_rmse = rmse_like_baseline(original_values[val_idx], val_recon, missing_mask_values[val_idx])

        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # retrain on full data for best_epoch
    set_seed(seed)
    G = Generator(n_features=x_np.shape[1], hidden=gen_hidden).to(device)
    D = Discriminator(n_features=x_np.shape[1], hidden=disc_hidden).to(device)
    opt_g = optim.Adam(G.parameters(), lr=learning_rate)
    opt_d = optim.Adam(D.parameters(), lr=learning_rate)

    full_ds = TensorDataset(x_tensor, m_tensor)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(max(1, best_epoch)):
        G.train()
        D.train()
        for xb, mb in full_loader:
            xb = xb.to(device)
            mb = mb.to(device)

            z = torch.rand_like(xb)
            x_tilde = mb * xb + (1.0 - mb) * z

            with torch.no_grad():
                g_mu, g_logvar = G(x_tilde, mb)
                x_hat = mb * xb + (1.0 - mb) * g_mu

            h = hint_matrix(mb, hint_rate)
            d_prob = D(x_hat, h)
            d_loss = -torch.mean(mb * torch.log(d_prob + 1e-8) + (1.0 - mb) * torch.log(1.0 - d_prob + 1e-8))

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            g_mu, g_logvar = G(x_tilde, mb)
            x_hat = mb * xb + (1.0 - mb) * g_mu
            h = hint_matrix(mb, hint_rate)
            d_prob = D(x_hat, h)
            g_adv = -torch.mean((1.0 - mb) * torch.log(d_prob + 1e-8))

            obs_mask_bool = mb.bool()
            z_mu = torch.zeros((xb.shape[0], 1), device=device)
            z_logvar = torch.zeros_like(z_mu)
            g_rec = loss_fct((g_mu, g_logvar, z_mu, z_logvar), xb, obs_mask_bool, epoch, warmup_epochs, freebits=freebits)

            g_loss = g_adv + alpha * g_rec

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

    # final imputation (deterministic z=0)
    G.eval()
    with torch.no_grad():
        x_all = x_tensor.to(device)
        m_all = m_tensor.to(device)
        z = torch.zeros_like(x_all)
        x_tilde_all = m_all * x_all + (1.0 - m_all) * z
        mu_all, _ = G(x_tilde_all, m_all)
        reconstructed = mu_all.detach().cpu().numpy()

    df_imputed_numeric = pd.DataFrame(reconstructed, columns=df_with_na_numeric.columns, index=df_with_na_numeric.index)
    df_final = fill_only_nans_like_baseline(df_with_na_numeric, df_imputed_numeric, nan_mask)

    imputed_df = pd.concat([eid_label_df, df_final], axis=1)

    rmse_value = rmse_like_baseline(original_values, df_final.to_numpy(), missing_mask_values)
    print(f"RMSE for GAN Imputation: {rmse_value:.4f}")

    write_rmse_csv(rmse_path, rmse_value)
    imputed_df.to_csv(imputed_data_path, index=False)

    return imputed_df, rmse_value


def gan(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    batch_size,
    learning_rate,
    freebits,
    gen_hidden=[128, 128],
    disc_hidden=[128, 128],
    hint_rate=0.9,
    alpha=10.0,
    epochs=200,
    patience=50,
    warmup_epochs=100,
    seed: int = 42,
):
    if experiments_dir is None:
        raise ImportError(f"Could not import experiments_dir from config_paths. Original error: {_IMPORT_ERROR}")

    main_directory = str(experiments_dir(columns_num))

    for fraction in fractions:
        for missingness in missingness_types:
            full_path = os.path.join(main_directory, f"fraction_{fraction}_missingness_{missingness}")
            os.makedirs(full_path, exist_ok=True)

            corrupted_data_path = os.path.join(
                full_path, f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}.csv"
            )
            missing_mask_path = os.path.join(full_path, f"missing_mask_col_{columns_num}.csv")
            normalized_data_path = os.path.join(full_path, f"normalized_data_col_{columns_num}.csv")

            gh = "-".join(map(str, gen_hidden))
            dh = "-".join(map(str, disc_hidden))
            param_id = f"bs{batch_size}_lr{learning_rate}_fb{freebits}_gh{gh}_dh{dh}_hr{hint_rate}_a{alpha}"
            rmse_path = os.path.join(full_path, f"RMSE_GAN_col_{columns_num}_params_{param_id}.csv")
            imputed_path = os.path.join(full_path, f"imputed_GAN_col_{columns_num}_params_{param_id}.csv")

            if len(os.listdir(full_path)) > 0:
                if not (os.path.exists(corrupted_data_path) and os.path.exists(normalized_data_path) and os.path.exists(missing_mask_path)):
                    continue

                df_with_na = pd.read_csv(corrupted_data_path)
                original_df = pd.read_csv(normalized_data_path)
                missing_mask_df = pd.read_csv(missing_mask_path)

                GAN_imputer_papergrade(
                    df_with_na=df_with_na,
                    original_df=original_df,
                    missing_mask=missing_mask_df,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    freebits=freebits,
                    gen_hidden=gen_hidden,
                    disc_hidden=disc_hidden,
                    hint_rate=hint_rate,
                    alpha=alpha,
                    rmse_path=rmse_path,
                    imputed_data_path=imputed_path,
                    epochs=epochs,
                    patience=patience,
                    warmup_epochs=warmup_epochs,
                    seed=seed,
                )
            else:
                pass
