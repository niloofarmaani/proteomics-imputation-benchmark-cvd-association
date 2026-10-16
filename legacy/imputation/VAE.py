# imputation/VAE.py

from __future__ import annotations

import ast
import random
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir


LOG_2PI = float(np.log(2.0 * np.pi))
LOG2 = float(np.log(2.0))


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Filename helpers
# =============================================================================
def _float_token(x: float) -> str:
    s = f"{float(x):.12g}"
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def _parse_n_neurons(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]

    if isinstance(value, str):
        s = value.strip()

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            pass

        s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        s = s.replace("-", ",")
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    raise ValueError(f"Could not parse n_neurons from value: {value}")


def _param_id(
    batch_size: int,
    learning_rate: float,
    n_neurons: Sequence[int],
    dim_latent: int,
    freebits: float,
) -> str:
    nn_token = "-".join(str(int(x)) for x in n_neurons)

    return (
        f"bs{int(batch_size)}"
        f"_lr{_float_token(float(learning_rate))}"
        f"_nn{nn_token}"
        f"_dl{int(dim_latent)}"
        f"_fb{_float_token(float(freebits))}"
    )


# =============================================================================
# Strict input validation
# =============================================================================
def _validate_input_frames(df_with_na: pd.DataFrame, original_df: pd.DataFrame) -> None:
    if df_with_na.shape[1] < 3:
        raise ValueError(
            f"Corrupted data must contain eid, label, and at least one protein column. "
            f"Observed shape: {df_with_na.shape}"
        )

    if original_df.shape[1] < 3:
        raise ValueError(
            f"Reference data must contain eid, label, and at least one protein column. "
            f"Observed shape: {original_df.shape}"
        )

    if list(df_with_na.columns[:2]) != ["eid", "label"]:
        raise ValueError(
            f"Corrupted data first two columns must be ['eid', 'label']. "
            f"Observed: {list(df_with_na.columns[:2])}"
        )

    if list(original_df.columns[:2]) != ["eid", "label"]:
        raise ValueError(
            f"Reference data first two columns must be ['eid', 'label']. "
            f"Observed: {list(original_df.columns[:2])}"
        )

    if df_with_na.shape != original_df.shape:
        raise ValueError(
            "Corrupted and reference data have different shapes.\n"
            f"Corrupted shape: {df_with_na.shape}\n"
            f"Reference shape: {original_df.shape}"
        )

    if list(df_with_na.columns) != list(original_df.columns):
        raise ValueError("Corrupted and reference data columns differ or are reordered.")

    if not df_with_na["eid"].astype(str).equals(original_df["eid"].astype(str)):
        raise ValueError("eid column does not match between corrupted and reference data.")

    if not df_with_na["label"].astype(str).equals(original_df["label"].astype(str)):
        raise ValueError("label column does not match between corrupted and reference data.")


# =============================================================================
# Mask and RMSE helpers
# =============================================================================
def make_cell_validation_mask(
    data: np.ndarray,
    observed_mask: np.ndarray,
    val_rate: float,
    seed: int,
) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    observed_mask = np.asarray(observed_mask, dtype=bool)

    if data.shape != observed_mask.shape:
        raise ValueError(
            f"data and observed_mask must have same shape. "
            f"data={data.shape}, observed_mask={observed_mask.shape}"
        )

    if not (0.0 < float(val_rate) < 1.0):
        raise ValueError(f"val_rate must be in (0, 1). Got {val_rate}")

    rng = np.random.default_rng(int(seed))
    val_mask = np.zeros_like(observed_mask, dtype=bool)

    for j in range(data.shape[1]):
        rows = np.where(observed_mask[:, j] & np.isfinite(data[:, j]))[0]

        if len(rows) == 0:
            continue

        n_val = int(np.ceil(len(rows) * float(val_rate)))
        n_val = min(max(n_val, 1), len(rows))

        chosen = rng.choice(rows, size=n_val, replace=False)
        val_mask[chosen, j] = True

    if val_mask.sum() == 0:
        raise ValueError("Internal VAE validation mask has zero cells.")

    return val_mask


def column_means_from_mask(data: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)

    means = np.empty(data.shape[1], dtype=float)
    bad_cols = []

    for j in range(data.shape[1]):
        vals = data[train_mask[:, j], j]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            bad_cols.append(j)
            means[j] = np.nan
        else:
            means[j] = float(np.mean(vals))

    if bad_cols:
        raise ValueError(
            "Some protein columns have no train-observed values for VAE fill initialization. "
            f"Bad column indices, first 20: {bad_cols[:20]}"
        )

    return means


def fill_missing_with_column_means(
    data: np.ndarray,
    fill_means: np.ndarray,
    mask_to_fill: np.ndarray,
) -> np.ndarray:
    out = np.asarray(data, dtype=float).copy()
    mask_to_fill = np.asarray(mask_to_fill, dtype=bool)

    if out.shape != mask_to_fill.shape:
        raise ValueError("data and mask_to_fill have different shapes.")

    for j in range(out.shape[1]):
        out[mask_to_fill[:, j], j] = fill_means[j]

    if not np.isfinite(out).all():
        raise ValueError("VAE input matrix contains non-finite values after column-mean filling.")

    return out.astype(np.float32)


def rmse_on_mask(original: np.ndarray, imputed: np.ndarray, mask: np.ndarray) -> float:
    original = np.asarray(original, dtype=float)
    imputed = np.asarray(imputed, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if original.shape != imputed.shape or original.shape != mask.shape:
        raise ValueError(
            f"original, imputed, and mask must have same shape. "
            f"original={original.shape}, imputed={imputed.shape}, mask={mask.shape}"
        )

    valid = mask & np.isfinite(original) & np.isfinite(imputed)

    if valid.sum() == 0:
        raise ValueError("No finite cells available for RMSE calculation.")

    return float(np.sqrt(np.mean((original[valid] - imputed[valid]) ** 2)))


def fill_only_missing_cells(
    observed_df: pd.DataFrame,
    predicted_values: np.ndarray,
    nan_mask: np.ndarray,
) -> pd.DataFrame:
    final_values = observed_df.to_numpy(dtype=float).copy()
    final_values[nan_mask] = predicted_values[nan_mask]
    return pd.DataFrame(final_values, index=observed_df.index, columns=observed_df.columns)


def write_rmse_csv(
    rmse_path: str,
    test_rmse_missing: float,
    internal_best_val_rmse: float,
    seed: int,
    best_epoch: int,
    n_missing_cells: int,
    n_internal_val_cells: int,
) -> None:
    rmse_path = Path(rmse_path)
    rmse_path.parent.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(
        {
            "rmse_vae": [float(test_rmse_missing)],
            "internal_best_val_rmse": [float(internal_best_val_rmse)],
            "seed": [int(seed)],
            "best_epoch": [int(best_epoch)],
            "n_missing_cells": [int(n_missing_cells)],
            "n_internal_val_cells": [int(n_internal_val_cells)],
        }
    )

    out.to_csv(rmse_path, index=False)


# =============================================================================
# VAE model
# =============================================================================
class VAEModel(nn.Module):
    def __init__(self, n_features: int, n_neurons: List[int], dim_latent: int = 16):
        super().__init__()

        if n_features <= 0:
            raise ValueError(f"n_features must be positive. Got {n_features}")

        if dim_latent <= 0:
            raise ValueError(f"dim_latent must be positive. Got {dim_latent}")

        if len(n_neurons) == 0:
            raise ValueError("n_neurons cannot be empty.")

        for h in n_neurons:
            if int(h) <= 0:
                raise ValueError(f"All hidden dimensions must be positive. Got {n_neurons}")

        activation = nn.LeakyReLU(0.1)

        def block(in_features: int, out_features: int):
            return [
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                activation,
                nn.Dropout(0.1),
            ]

        dims = [n_features] + [int(x) for x in n_neurons]

        encoder_layers = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            encoder_layers.extend(block(in_dim, out_dim))

        encoder_layers.append(nn.Linear(dims[-1], int(dim_latent) * 2))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        rev_hidden = list(reversed([int(x) for x in n_neurons]))

        decoder_input = int(dim_latent)

        if len(rev_hidden) > 0:
            decoder_layers.extend(block(decoder_input, rev_hidden[0]))

            for in_dim, out_dim in zip(rev_hidden[:-1], rev_hidden[1:]):
                decoder_layers.extend(block(in_dim, out_dim))

            decoder_layers.append(nn.Linear(rev_hidden[-1], n_features * 2))
        else:
            decoder_layers.append(nn.Linear(decoder_input, n_features * 2))

        self.decoder = nn.Sequential(*decoder_layers)
        self.n_features = int(n_features)
        self.dim_latent = int(dim_latent)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_params = self.encoder(x)
        z_mu = z_params[:, : self.dim_latent]
        z_logvar = z_params[:, self.dim_latent :]
        z_logvar = torch.clamp(z_logvar, min=-8.0, max=8.0)
        return z_mu, z_logvar

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_params = self.decoder(z)
        x_mu = x_params[:, : self.n_features]
        x_logvar = x_params[:, self.n_features :]
        x_logvar = torch.clamp(x_logvar, min=-8.0, max=8.0)
        return x_mu, x_logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.decode(z)
        return x_mu, x_logvar, z_mu, z_logvar


def gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if target.shape != mu.shape or target.shape != logvar.shape or target.shape != mask.shape:
        raise ValueError("target, mu, logvar, and mask must have the same shape.")

    if mask.sum().item() == 0:
        raise ValueError("A training batch has zero train-observed cells.")

    nll = 0.5 * (LOG_2PI + logvar + (target - mu) ** 2 / torch.exp(logvar))
    return torch.mean(nll[mask])


def kl_freebits(z_mu: torch.Tensor, z_logvar: torch.Tensor, freebits: float) -> torch.Tensor:
    freebits = float(freebits)

    if freebits < 0:
        raise ValueError(f"freebits must be >= 0. Got {freebits}")

    kld = 0.5 * (z_mu.pow(2) + torch.exp(z_logvar) - 1.0 - z_logvar)

    if freebits > 0:
        threshold = freebits * LOG2
        kld = torch.maximum(kld, torch.full_like(kld, threshold))

    return torch.mean(torch.sum(kld, dim=1))


def vae_loss(
    pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    train_mask: torch.Tensor,
    epoch: int,
    warmup_epochs: int,
    freebits: float,
) -> torch.Tensor:
    x_mu, x_logvar, z_mu, z_logvar = pred

    rec = gaussian_nll(
        target=target,
        mu=x_mu,
        logvar=x_logvar,
        mask=train_mask,
    )

    kl = kl_freebits(z_mu, z_logvar, freebits=freebits)

    if warmup_epochs > 0:
        kl_weight = min(1.0, float(epoch + 1) / float(warmup_epochs))
    else:
        kl_weight = 1.0

    return rec + kl_weight * kl


def _make_loader(
    x_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    dataset = TensorDataset(x_tensor, mask_tensor)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        drop_last=False,
    )


# =============================================================================
# VAE imputer
# =============================================================================
def VAE_imputer(
    df_with_na: pd.DataFrame,
    original_df: pd.DataFrame,
    batch_size: int,
    learning_rate: float,
    n_neurons: List[int],
    dim_latent: int,
    freebits: float,
    rmse_path: str,
    imputed_data_path: str,
    epochs: int,
    patience: int,
    warmup_epochs: int,
    seed: int = 42,
    internal_val_rate: float = 0.05,
):
    """
    Strict VAE imputer.

    Internal early stopping uses a validation mask drawn only from observed cells.
    Final benchmark missing cells are not used for training or early stopping.
    """
    set_seed(seed)
    device = get_device()

    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    dim_latent = int(dim_latent)
    freebits = float(freebits)
    epochs = int(epochs)
    patience = int(patience)
    warmup_epochs = int(warmup_epochs)
    seed = int(seed)

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1. Got {batch_size}")

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive. Got {learning_rate}")

    if freebits < 0:
        raise ValueError(f"freebits must be >= 0. Got {freebits}")

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1. Got {epochs}")

    if patience < 1:
        raise ValueError(f"patience must be >= 1. Got {patience}")

    _validate_input_frames(df_with_na, original_df)

    metadata_df = df_with_na.iloc[:, :2].copy()
    corrupted_numeric_df = df_with_na.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    reference_numeric_df = original_df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

    data = corrupted_numeric_df.to_numpy(dtype=float)
    reference = reference_numeric_df.to_numpy(dtype=float)

    nan_mask = np.isnan(data)
    observed_mask = np.isfinite(data)

    n_missing = int(nan_mask.sum())

    if n_missing == 0:
        raise ValueError("No missing protein cells found in corrupted input. VAE cannot evaluate imputation RMSE.")

    if not np.isfinite(reference[nan_mask]).all():
        bad = int((~np.isfinite(reference[nan_mask])).sum())
        raise ValueError(
            f"Reference values at missing positions contain {bad} non-finite values. "
            "This indicates preprocessing or mask mismatch."
        )

    internal_val_mask = make_cell_validation_mask(
        data=data,
        observed_mask=observed_mask,
        val_rate=internal_val_rate,
        seed=seed + 53_111,
    )

    train_mask = observed_mask & ~internal_val_mask

    if train_mask.sum() == 0:
        raise ValueError("VAE train mask has zero cells.")

    train_fill_means = column_means_from_mask(data, train_mask)

    input_mask_to_fill = nan_mask | internal_val_mask
    data_train_input = fill_missing_with_column_means(
        data=data,
        fill_means=train_fill_means,
        mask_to_fill=input_mask_to_fill,
    )

    x_tensor = torch.tensor(data_train_input, dtype=torch.float32)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)

    loader = _make_loader(
        x_tensor=x_tensor,
        mask_tensor=train_mask_tensor,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )

    model = VAEModel(
        n_features=corrupted_numeric_df.shape[1],
        n_neurons=[int(x) for x in n_neurons],
        dim_latent=dim_latent,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\n" + "=" * 80)
    print("VAE training")
    print("=" * 80)
    print(f"device: {device}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"n_neurons: {n_neurons}")
    print(f"dim_latent: {dim_latent}")
    print(f"freebits: {freebits}")
    print(f"epochs: {epochs}")
    print(f"patience: {patience}")
    print(f"warmup_epochs: {warmup_epochs}")
    print(f"seed: {seed}")
    print(f"n_missing_cells: {n_missing}")
    print(f"n_internal_val_cells: {int(internal_val_mask.sum())}")
    print("=" * 80)

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for xb, mb in loader:
            xb = xb.to(device)
            mb = mb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = vae_loss(
                pred=pred,
                target=xb,
                train_mask=mb,
                epoch=epoch,
                warmup_epochs=warmup_epochs,
                freebits=freebits,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(float(loss.item()))

        model.eval()

        with torch.no_grad():
            z_mu, _ = model.encode(x_tensor.to(device))
            x_mu, _ = model.decode(z_mu)
            pred_all = x_mu.cpu().numpy()

        val_rmse = rmse_on_mask(reference, pred_all, internal_val_mask)
        avg_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *new best*"
        else:
            patience_counter += 1
            marker = ""

        if epoch == 0 or (epoch + 1) % 10 == 0 or marker:
            print(
                f"Epoch {epoch + 1:03d}/{epochs:03d} | "
                f"train_loss={avg_loss:.6f} | internal_val_rmse={val_rmse:.6f}{marker}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_state is None:
        raise RuntimeError("VAE training did not produce a best state.")

    if best_epoch < 1:
        raise RuntimeError("VAE best_epoch is invalid.")

    print(f"Best internal validation RMSE: {best_val:.6f} at epoch {best_epoch}")

    # -------------------------------------------------------------------------
    # Final refit on all observed cells for best_epoch
    # -------------------------------------------------------------------------
    full_train_mask = observed_mask

    if full_train_mask.sum() == 0:
        raise ValueError("VAE full train mask has zero cells.")

    full_fill_means = column_means_from_mask(data, full_train_mask)

    data_full_input = fill_missing_with_column_means(
        data=data,
        fill_means=full_fill_means,
        mask_to_fill=nan_mask,
    )

    x_full_tensor = torch.tensor(data_full_input, dtype=torch.float32)
    full_mask_tensor = torch.tensor(full_train_mask, dtype=torch.bool)

    full_loader = _make_loader(
        x_tensor=x_full_tensor,
        mask_tensor=full_mask_tensor,
        batch_size=batch_size,
        seed=seed + 57_001,
        shuffle=True,
    )

    final_model = VAEModel(
        n_features=corrupted_numeric_df.shape[1],
        n_neurons=[int(x) for x in n_neurons],
        dim_latent=dim_latent,
    ).to(device)

    optimizer = optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(best_epoch):
        final_model.train()
        losses = []

        for xb, mb in full_loader:
            xb = xb.to(device)
            mb = mb.to(device)

            optimizer.zero_grad()
            pred = final_model(xb)
            loss = vae_loss(
                pred=pred,
                target=xb,
                train_mask=mb,
                epoch=epoch,
                warmup_epochs=warmup_epochs,
                freebits=freebits,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(float(loss.item()))

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == best_epoch:
            print(
                f"Final refit epoch {epoch + 1:03d}/{best_epoch:03d} | "
                f"train_loss={float(np.mean(losses)):.6f}"
            )

    final_model.eval()

    with torch.no_grad():
        z_mu, _ = final_model.encode(x_full_tensor.to(device))
        x_mu, _ = final_model.decode(z_mu)
        pred_full = x_mu.cpu().numpy()

    final_numeric_df = fill_only_missing_cells(
        observed_df=corrupted_numeric_df,
        predicted_values=pred_full,
        nan_mask=nan_mask,
    )

    final_values = final_numeric_df.to_numpy(dtype=float)
    test_rmse = rmse_on_mask(reference, final_values, nan_mask)

    if not np.isfinite(test_rmse):
        raise FloatingPointError(f"VAE final RMSE is non-finite: {test_rmse}")

    imputed_df = pd.concat(
        [
            metadata_df.reset_index(drop=True),
            final_numeric_df.reset_index(drop=True),
        ],
        axis=1,
    )

    if list(imputed_df.columns[:2]) != ["eid", "label"]:
        raise RuntimeError("Internal error: VAE output does not preserve eid and label.")

    write_rmse_csv(
        rmse_path=rmse_path,
        test_rmse_missing=test_rmse,
        internal_best_val_rmse=best_val,
        seed=seed,
        best_epoch=best_epoch,
        n_missing_cells=n_missing,
        n_internal_val_cells=int(internal_val_mask.sum()),
    )

    imputed_data_path = Path(imputed_data_path)
    imputed_data_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_df.to_csv(imputed_data_path, index=False)

    print("\n" + "=" * 80)
    print(f"VAE final RMSE on missing cells: {test_rmse:.6f}")
    print(f"RMSE saved to: {rmse_path}")
    print(f"Imputed data saved to: {imputed_data_path}")
    print("=" * 80)

    return imputed_df, test_rmse


# =============================================================================
# File-handling driver
# =============================================================================
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
    epochs=300,
    patience=50,
    warmup_epochs=100,
    seed: int = 42,
):
    """
    Strict seeded VAE wrapper.

    Required input:
        Corrupted_data_fraction_<fraction>_missingness_<missingness>_col_<columns_num>_seed_<seed>.csv

    Output:
        RMSE_VAE_col_<columns_num>_params_<param_id>_seed_<seed>.csv
        imputed_VAE_col_<columns_num>_params_<param_id>_seed_<seed>.csv
    """
    main_directory = Path(experiments_dir(columns_num))

    if not main_directory.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist:\n{main_directory}\n"
            "Run preprocessing before VAE."
        )

    seed = int(seed)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    dim_latent = int(dim_latent)
    freebits = float(freebits)
    n_neurons = _parse_n_neurons(n_neurons)

    param_id = _param_id(
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_neurons=n_neurons,
        dim_latent=dim_latent,
        freebits=freebits,
    )

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            print(
                f"\n>>> Running VAE | missingness={missingness} | "
                f"fraction={fraction} | seed={seed}"
            )

            full_path = main_directory / f"fraction_{fraction}_missingness_{missingness}"

            if not full_path.exists():
                raise FileNotFoundError(
                    f"Required fraction/missingness directory does not exist:\n{full_path}"
                )

            corrupted_data_path = (
                full_path
                / f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}_seed_{seed}.csv"
            )

            normalized_data_path = full_path / f"normalized_data_col_{columns_num}.csv"

            if not corrupted_data_path.exists():
                raise FileNotFoundError(
                    f"\n[CRITICAL] VAE could not find the exact seeded corrupted file:\n"
                    f"{corrupted_data_path}\n\n"
                    "Run corrected preprocessing for this exact fraction, missingness type, and seed."
                )

            if not normalized_data_path.exists():
                raise FileNotFoundError(
                    f"\n[CRITICAL] VAE could not find normalized/reference data:\n"
                    f"{normalized_data_path}"
                )

            rmse_path = (
                full_path
                / f"RMSE_VAE_col_{columns_num}_params_{param_id}_seed_{seed}.csv"
            )

            imputed_path = (
                full_path
                / f"imputed_VAE_col_{columns_num}_params_{param_id}_seed_{seed}.csv"
            )

            df_with_na = pd.read_csv(corrupted_data_path)
            original_df = pd.read_csv(normalized_data_path)

            VAE_imputer(
                df_with_na=df_with_na,
                original_df=original_df,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_neurons=n_neurons,
                dim_latent=dim_latent,
                freebits=freebits,
                rmse_path=str(rmse_path),
                imputed_data_path=str(imputed_path),
                epochs=int(epochs),
                patience=int(patience),
                warmup_epochs=int(warmup_epochs),
                seed=seed,
                internal_val_rate=0.05,
            )

    print("\n--- VAE imputation pipeline complete ---")