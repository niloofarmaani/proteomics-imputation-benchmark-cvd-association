import os
import copy
import json
import random
import warnings
from itertools import product
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Ensure we can import experiments_dir from config_paths
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Repro and device
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
LOG_2PI = float(np.log(2.0 * np.pi))
LOG2 = float(np.log(2.0))
leaky_relu_default = nn.LeakyReLU(0.1)


# ---------------------------------------------------------------------
# Mask-aware VAE
# Encoder input dimension = 2*n_features (x + mask)
# Decoder output dimension = 2*n_features (mu + logvar)
# ---------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(
        self,
        n_input_features: int,     # = 2*n_features
        n_output_features: int,    # = n_features
        n_neurons: List[int],
        activation=leaky_relu_default,
        dim_latent: int = 16,
        dropout_p: float = 0.0,
        use_batchnorm: bool = False,
        x_logvar_min: float = -6.0,
        x_logvar_max: float = 6.0,
    ):
        super().__init__()
        self.n_input_features = int(n_input_features)
        self.n_output_features = int(n_output_features)
        self.n_neurons = list(n_neurons)
        self.dim_latent = int(dim_latent)

        self.dropout_p = float(dropout_p)
        self.use_batchnorm = bool(use_batchnorm)
        self.x_logvar_min = float(x_logvar_min)
        self.x_logvar_max = float(x_logvar_max)

        def block(in_f: int, out_f: int) -> List[nn.Module]:
            layers: List[nn.Module] = [nn.Linear(in_f, out_f)]
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(out_f))
            layers.append(activation)
            return layers

        # Encoder
        enc_layers: List[nn.Module] = []
        cur = self.n_input_features
        for h in self.n_neurons:
            enc_layers += block(cur, h)
            cur = h
        enc_layers.append(nn.Linear(cur, self.dim_latent * 2))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        cur = self.dim_latent
        for h in reversed(self.n_neurons):
            dec_layers += block(cur, h)
            cur = h
        dec_layers.append(nn.Linear(cur, self.n_output_features * 2))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_params = self.encoder(x)
        z_mu = z_params[:, : self.dim_latent]
        z_logvar = z_params[:, self.dim_latent :]
        return z_mu, z_logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_params = self.decoder(z)
        x_mu = x_params[:, : self.n_output_features]
        x_logvar = x_params[:, self.n_output_features :]
        x_logvar = torch.clamp(x_logvar, min=self.x_logvar_min, max=self.x_logvar_max)
        return x_mu, x_logvar

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.decode(z)
        return x_mu, x_logvar, z_mu, z_logvar


# ---------------------------------------------------------------------
# Loss
# recon_loss:
#   - "mse" is aligned to RMSE
#   - "gaussian_nll" uses learned log-variance (often worse for RMSE)
# beta:
#   - scales KL term (beta=0 makes it a denoising autoencoder)
# ---------------------------------------------------------------------
def compute_kld(z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (z_mu**2 + torch.exp(z_logvar) - 1 - z_logvar)


def gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (LOG_2PI + logvar + (z - mu) ** 2 / torch.exp(logvar))


def loss_fct(
    pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y_true: torch.Tensor,
    obs_mask_true: torch.Tensor,
    epoch: int,
    warmup_epochs: int,
    freebits: float,
    recon_loss: str = "mse",
    beta: float = 0.0,
) -> torch.Tensor:
    x_mu, x_logvar, z_mu, z_logvar = pred

    recon_loss = str(recon_loss).lower()
    if recon_loss == "mse":
        diff = (y_true[obs_mask_true] - x_mu[obs_mask_true])
        l_rec = torch.sum(diff * diff)
    elif recon_loss == "gaussian_nll":
        masked_log_prob = gaussian_log_prob(y_true[obs_mask_true], x_mu[obs_mask_true], x_logvar[obs_mask_true])
        l_rec = -torch.sum(masked_log_prob)
    else:
        raise ValueError(f"Unknown recon_loss: {recon_loss}")

    # KL with freebits
    kld = compute_kld(z_mu, z_logvar)  # [batch, latent]
    l_reg = torch.sum((F.relu(kld - freebits * LOG2) + freebits * LOG2), dim=1)

    kl_weight = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0

    return (l_rec / y_true.shape[0]) + (float(beta) * kl_weight * torch.mean(l_reg))


def rmse_on_mask(original_values: np.ndarray, imputed_values: np.ndarray, mask_values: np.ndarray) -> float:
    masked_original = original_values[mask_values]
    masked_final = imputed_values[mask_values]
    if masked_original.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean((masked_original - masked_final) ** 2)))


# ---------------------------------------------------------------------
# Training (early stop on RMSE over ALL missing entries, stable even for 0.01)
# corrupt_prob: randomly hide observed values in input, but still compute loss on them
# ---------------------------------------------------------------------
def train_with_early_stopping(
    model: VAE,
    train_loader: DataLoader,
    full_data_tensor: torch.Tensor,          # [N, F], filled
    full_obs_mask_tensor: torch.Tensor,      # [N, F] bool
    full_missing_mask_np: np.ndarray,        # [N, F] bool, where we evaluate RMSE
    full_original_np: np.ndarray,            # [N, F] ground truth
    device: torch.device,
    epochs: int,
    patience: int,
    warmup_epochs: int,
    learning_rate: float,
    weight_decay: float,
    freebits: float,
    recon_loss: str,
    beta: float,
    scheduler_type: str = "none",
    cyclic_base_lr: float = 1e-4,
    cyclic_max_lr: float = 1e-3,
    cyclic_step_size_up: int = 50,
    grad_clip_norm: Optional[float] = None,
    nan_fill_value: float = 0.0,
    corrupt_prob: float = 0.2,
) -> Dict[str, Any]:
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    scheduler = None
    scheduler_type = str(scheduler_type).lower()
    if scheduler_type == "cyclic":
        scheduler = CyclicLR(
            optimizer,
            base_lr=float(cyclic_base_lr),
            max_lr=float(cyclic_max_lr),
            step_size_up=int(cyclic_step_size_up),
            mode="triangular",
            cycle_momentum=False,
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    best_rmse = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0

    full_data_tensor = full_data_tensor.to(device)
    full_obs_mask_tensor = full_obs_mask_tensor.to(device)

    for epoch in range(int(epochs)):
        model.train()
        for xb, mb in train_loader:
            xb = xb.to(device)  # [B, F]
            mb = mb.to(device)  # [B, F] bool

            # Create corrupted input (denoising)
            x_in = xb.clone()
            m_in = mb.clone()

            if corrupt_prob > 0:
                drop = (torch.rand_like(x_in) < float(corrupt_prob)) & m_in
                x_in[drop] = float(nan_fill_value)
                m_in[drop] = False

            # Encoder input: [x_in, m_in]
            enc_in = torch.cat([x_in, m_in.float()], dim=1)

            optimizer.zero_grad()
            pred = model(enc_in)

            loss = loss_fct(
                pred=pred,
                y_true=xb,
                obs_mask_true=mb,
                epoch=epoch,
                warmup_epochs=int(warmup_epochs),
                freebits=float(freebits),
                recon_loss=recon_loss,
                beta=float(beta),
            )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # Evaluate RMSE on ALL missing entries (stable for small missing fractions)
        model.eval()
        with torch.no_grad():
            enc_full = torch.cat([full_data_tensor, full_obs_mask_tensor.float()], dim=1)
            x_mu_all, _, _, _ = model(enc_full)
            recon_all = x_mu_all.detach().cpu().numpy()

        rmse_missing = rmse_on_mask(full_original_np, recon_all, full_missing_mask_np)

        if rmse_missing < best_rmse:
            best_rmse = rmse_missing
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    return {
        "best_rmse_missing": float(best_rmse),
        "best_epoch": int(best_epoch),
        "best_state": best_state,
    }


# ---------------------------------------------------------------------
# Hyperparameter search + saving outputs
# ---------------------------------------------------------------------
def VAE_imputer(
    df_with_na: pd.DataFrame,
    original_df: pd.DataFrame,
    missing_mask: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    rmse_path: str,
    imputed_data_path: str,
    trials_csv_path: Optional[str] = None,
    best_json_path: Optional[str] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, float]:
    set_seed(seed)
    device = get_device()

    # Split eid/label and numeric
    eid_label_df = df_with_na.iloc[:, :2]
    df_with_na_numeric = df_with_na.iloc[:, 2:]
    original_df_numeric = original_df.iloc[:, 2:]

    # Align missing mask to numeric
    if missing_mask.shape[1] == df_with_na_numeric.shape[1]:
        missing_mask_numeric = missing_mask
    elif missing_mask.shape[1] == df_with_na.shape[1]:
        missing_mask_numeric = missing_mask.iloc[:, 2:]
    else:
        missing_mask_numeric = missing_mask.iloc[:, -df_with_na_numeric.shape[1] :]

    X = df_with_na_numeric.to_numpy()
    nan_mask = np.isnan(X)
    X_org = original_df_numeric.to_numpy()
    M = missing_mask_numeric.to_numpy().astype(bool)

    if M.shape == nan_mask.shape and not np.array_equal(M, nan_mask):
        warnings.warn("missing_mask does not exactly match NaNs in corrupted data. Training uses NaNs, RMSE uses missing_mask file.")

    required = {"batch_size", "learning_rate", "n_neurons", "dim_latent"}
    missing = required - set(param_grid.keys())
    if missing:
        raise ValueError(f"param_grid missing required keys: {sorted(missing)}")

    # Defaults for optional params
    def get_param(d: Dict[str, Any], k: str, default: Any) -> Any:
        return d[k] if k in d else default

    names = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    records: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for vals in combos:
        p = dict(zip(names, vals))

        # Defaults
        epochs = int(get_param(p, "epochs", 200))
        patience = int(get_param(p, "patience", 50))
        warmup_epochs = int(get_param(p, "warmup_epochs", 0))

        recon_loss = str(get_param(p, "recon_loss", "mse"))
        beta = float(get_param(p, "beta", 0.0))
        freebits = float(get_param(p, "freebits", 0.0))

        dropout_p = float(get_param(p, "dropout_p", 0.0))
        use_batchnorm = bool(get_param(p, "use_batchnorm", False))

        weight_decay = float(get_param(p, "weight_decay", 0.0))
        grad_clip_norm = get_param(p, "grad_clip_norm", None)

        scheduler_type = str(get_param(p, "scheduler_type", "none"))
        cyclic_base_lr = float(get_param(p, "cyclic_base_lr", 1e-4))
        cyclic_max_lr = float(get_param(p, "cyclic_max_lr", 1e-3))
        cyclic_step_size_up = int(get_param(p, "cyclic_step_size_up", 50))

        nan_fill_value = float(get_param(p, "nan_fill_value", 0.0))
        corrupt_prob = float(get_param(p, "corrupt_prob", 0.2))

        # Fill NaNs for input tensor
        X_filled = np.where(nan_mask, nan_fill_value, X)

        data_tensor = torch.tensor(X_filled, dtype=torch.float32)
        obs_mask_tensor = torch.tensor(~nan_mask, dtype=torch.bool)

        train_ds = TensorDataset(data_tensor, obs_mask_tensor)
        train_loader = DataLoader(train_ds, batch_size=int(p["batch_size"]), shuffle=True)

        model = VAE(
            n_input_features=2 * X_filled.shape[1],
            n_output_features=X_filled.shape[1],
            n_neurons=list(p["n_neurons"]),
            dim_latent=int(p["dim_latent"]),
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
            x_logvar_min=float(get_param(p, "x_logvar_min", -6.0)),
            x_logvar_max=float(get_param(p, "x_logvar_max", 6.0)),
        )

        print("Training with params:", p)

        out = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            full_data_tensor=data_tensor,
            full_obs_mask_tensor=obs_mask_tensor,
            full_missing_mask_np=M,
            full_original_np=X_org,
            device=device,
            epochs=epochs,
            patience=patience,
            warmup_epochs=warmup_epochs,
            learning_rate=float(p["learning_rate"]),
            weight_decay=weight_decay,
            freebits=freebits,
            recon_loss=recon_loss,
            beta=beta,
            scheduler_type=scheduler_type,
            cyclic_base_lr=cyclic_base_lr,
            cyclic_max_lr=cyclic_max_lr,
            cyclic_step_size_up=cyclic_step_size_up,
            grad_clip_norm=grad_clip_norm,
            nan_fill_value=nan_fill_value,
            corrupt_prob=corrupt_prob,
        )

        rec = {**p}
        rec["best_rmse_missing"] = out["best_rmse_missing"]
        rec["best_epoch"] = out["best_epoch"]
        records.append(rec)

        print(f"Best RMSE(missing) for this run: {out['best_rmse_missing']:.6f} at epoch {out['best_epoch']}")

        if best is None or out["best_rmse_missing"] < best["best_rmse_missing"]:
            best = {
                "params": p,
                "best_rmse_missing": out["best_rmse_missing"],
                "best_epoch": out["best_epoch"],
                "best_state": out["best_state"],
            }

    if best is None:
        raise RuntimeError("No trials ran. Check param_grid.")

    # Save trials
    if trials_csv_path is not None:
        os.makedirs(os.path.dirname(trials_csv_path), exist_ok=True)
        pd.DataFrame(records).sort_values("best_rmse_missing", ascending=True).to_csv(trials_csv_path, index=False)

    # Save best
    if best_json_path is not None:
        os.makedirs(os.path.dirname(best_json_path), exist_ok=True)
        best_to_save = {k: v for k, v in best.items() if k != "best_state"}
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(best_to_save, f, indent=2)

    best_params = best["params"]
    print("\nSelected best params:", best_params)

    # Rebuild best model and load best_state
    nan_fill_value = float(best_params.get("nan_fill_value", 0.0))
    X_filled = np.where(nan_mask, nan_fill_value, X)

    data_tensor = torch.tensor(X_filled, dtype=torch.float32)
    obs_mask_tensor = torch.tensor(~nan_mask, dtype=torch.bool)

    best_model = VAE(
        n_input_features=2 * X_filled.shape[1],
        n_output_features=X_filled.shape[1],
        n_neurons=list(best_params["n_neurons"]),
        dim_latent=int(best_params["dim_latent"]),
        dropout_p=float(best_params.get("dropout_p", 0.0)),
        use_batchnorm=bool(best_params.get("use_batchnorm", False)),
        x_logvar_min=float(best_params.get("x_logvar_min", -6.0)),
        x_logvar_max=float(best_params.get("x_logvar_max", 6.0)),
    ).to(device)
    best_model.load_state_dict(best["best_state"])

    best_model.eval()
    with torch.no_grad():
        enc_full = torch.cat([data_tensor.to(device), obs_mask_tensor.to(device).float()], dim=1)
        x_mu_all, _, _, _ = best_model(enc_full)
        recon_all = x_mu_all.detach().cpu().numpy()

    # Fill only NaNs
    X_final = np.where(nan_mask, recon_all, X_filled)
    df_final = pd.DataFrame(X_final, columns=df_with_na_numeric.columns, index=df_with_na_numeric.index)
    imputed_vae = pd.concat([eid_label_df, df_final], axis=1)

    rmse = float(np.sqrt(np.mean((X_org[M] - X_final[M]) ** 2)))
    print(f"RMSE for VAE Imputation: {rmse:.6f}")

    os.makedirs(os.path.dirname(rmse_path), exist_ok=True)
    pd.DataFrame({"rmse_vae": [rmse]}).to_csv(rmse_path, index=False)

    os.makedirs(os.path.dirname(imputed_data_path), exist_ok=True)
    imputed_vae.to_csv(imputed_data_path, index=False)

    return imputed_vae, rmse


# ---------------------------------------------------------------------
# Driver function
# ---------------------------------------------------------------------
def vae(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    batch_size,
    learning_rate,
    n_neurons,
    dim_latent,
    freebits,
    epochs: int = 200,
    patience: int = 50,
    warmup_epochs: int = 0,
    param_grid: Optional[Dict[str, List[Any]]] = None,
):
    if experiments_dir is None:
        raise ImportError(f"Could not import experiments_dir from config_paths. Original error: {_IMPORT_ERROR}")

    main_directory = str(experiments_dir(columns_num))

    if param_grid is None:
        # Single run defaults (no HPO)
        param_grid = {
            "batch_size": [int(batch_size)],
            "learning_rate": [float(learning_rate)],
            "n_neurons": [list(n_neurons)],
            "dim_latent": [int(dim_latent)],
            "freebits": [float(freebits)],
            "epochs": [int(epochs)],
            "patience": [int(patience)],
            "warmup_epochs": [int(warmup_epochs)],
            "recon_loss": ["mse"],
            "beta": [0.0],
            "nan_fill_value": [0.0],
            "corrupt_prob": [0.2],
            "scheduler_type": ["none"],
        }

    is_hpo = any(isinstance(v, list) and len(v) > 1 for v in param_grid.values())

    for fraction in fractions:
        for missingness in missingness_types:
            subfolder_name = f"fraction_{fraction}_missingness_{missingness}"
            full_path = os.path.join(main_directory, subfolder_name)
            os.makedirs(full_path, exist_ok=True)

            corrupted_data_path = os.path.join(
                full_path,
                f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}.csv",
            )
            missing_mask_path = os.path.join(full_path, f"missing_mask_col_{columns_num}.csv")
            normalized_data_path = os.path.join(full_path, f"normalized_data_col_{columns_num}.csv")

            required_files_exist = all(os.path.exists(p) for p in [corrupted_data_path, missing_mask_path, normalized_data_path])
            if not required_files_exist:
                continue

            df_with_na = pd.read_csv(corrupted_data_path)
            original_df = pd.read_csv(normalized_data_path)
            missing_mask_df = pd.read_csv(missing_mask_path)

            if not is_hpo:
                param_id = f"bs{batch_size}_lr{learning_rate}_nn{'-'.join(map(str, n_neurons))}_dl{dim_latent}_fb{freebits}"
                rmse_path = os.path.join(full_path, f"RMSE_VAE_col_{columns_num}_params_{param_id}.csv")
                imputed_path = os.path.join(full_path, f"imputed_VAE_col_{columns_num}_params_{param_id}.csv")
                trials_path = None
                best_path = None
            else:
                rmse_path = os.path.join(full_path, f"RMSE_VAE_best_col_{columns_num}.csv")
                imputed_path = os.path.join(full_path, f"imputed_VAE_best_col_{columns_num}.csv")
                trials_path = os.path.join(full_path, f"VAE_HPO_trials_col_{columns_num}.csv")
                best_path = os.path.join(full_path, f"VAE_HPO_best_col_{columns_num}.json")

            VAE_imputer(
                df_with_na=df_with_na,
                original_df=original_df,
                missing_mask=missing_mask_df,
                param_grid=param_grid,
                rmse_path=rmse_path,
                imputed_data_path=imputed_path,
                trials_csv_path=trials_path,
                best_json_path=best_path,
                seed=42,
            )
