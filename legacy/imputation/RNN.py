# imputation/RNN.py

from __future__ import annotations

import ast
import random
import sys
from pathlib import Path
from typing import Any, List, Sequence

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


def _param_id(
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    pseudo_mask_rate: float,
) -> str:
    return (
        f"bs{int(batch_size)}"
        f"_lr{_float_token(float(learning_rate))}"
        f"_hs{int(hidden_size)}"
        f"_nl{int(num_layers)}"
        f"_dr{_float_token(float(dropout))}"
        f"_pm{_float_token(float(pseudo_mask_rate))}"
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
        raise ValueError("Internal RNN validation mask has zero cells.")

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
            "Some protein columns have no train-observed values for RNN fill initialization. "
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
        raise ValueError("RNN input matrix contains non-finite values after column-mean filling.")

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
            "rmse_rnn": [float(test_rmse_missing)],
            "internal_best_val_rmse": [float(internal_best_val_rmse)],
            "seed": [int(seed)],
            "best_epoch": [int(best_epoch)],
            "n_missing_cells": [int(n_missing_cells)],
            "n_internal_val_cells": [int(n_internal_val_cells)],
        }
    )

    out.to_csv(rmse_path, index=False)


# =============================================================================
# RNN model
# =============================================================================
class TabularBiGRUImputer(nn.Module):
    """
    Bidirectional GRU over the fixed protein-column order.

    Each protein position receives two channels:
      1. current filled protein value
      2. observed indicator

    The model predicts one reconstructed value per protein position.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        if n_features < 1:
            raise ValueError(f"n_features must be positive. Got {n_features}")

        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive. Got {hidden_size}")

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1. Got {num_layers}")

        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1). Got {dropout}")

        recurrent_dropout = float(dropout) if int(num_layers) > 1 else 0.0

        self.gru = nn.GRU(
            input_size=2,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.output_head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(2 * int(hidden_size), int(hidden_size)),
            nn.LeakyReLU(0.1),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape != observed_mask.shape:
            raise ValueError(
                f"values and observed_mask must have the same shape. "
                f"values={values.shape}, observed_mask={observed_mask.shape}"
            )

        sequence_input = torch.stack(
            [values, observed_mask.to(dtype=values.dtype)],
            dim=-1,
        )

        hidden, _ = self.gru(sequence_input)
        return self.output_head(hidden).squeeze(-1)


def sample_pseudo_mask(
    observed_train_mask: torch.Tensor,
    pseudo_mask_rate: float,
) -> torch.Tensor:
    pseudo_mask_rate = float(pseudo_mask_rate)

    if not (0.0 < pseudo_mask_rate < 1.0):
        raise ValueError(
            f"pseudo_mask_rate must be in (0, 1). Got {pseudo_mask_rate}"
        )

    available = torch.nonzero(observed_train_mask, as_tuple=False)

    if available.numel() == 0:
        raise ValueError("Training batch contains no observed cells for pseudo-masking.")

    pseudo_mask = (
        torch.rand(observed_train_mask.shape, device=observed_train_mask.device)
        < pseudo_mask_rate
    ) & observed_train_mask

    if pseudo_mask.sum().item() == 0:
        chosen = available[
            torch.randint(
                low=0,
                high=available.shape[0],
                size=(1,),
                device=observed_train_mask.device,
            ).item()
        ]
        pseudo_mask[chosen[0], chosen[1]] = True

    return pseudo_mask


def _make_loader(
    target_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    dataset = TensorDataset(target_tensor, mask_tensor)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        drop_last=False,
    )


def _run_training_epoch(
    *,
    model: TabularBiGRUImputer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    fill_means_tensor: torch.Tensor,
    pseudo_mask_rate: float,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    for target_batch, observed_train_mask in loader:
        target_batch = target_batch.to(device)
        observed_train_mask = observed_train_mask.to(device)

        pseudo_mask = sample_pseudo_mask(
            observed_train_mask=observed_train_mask,
            pseudo_mask_rate=pseudo_mask_rate,
        )

        input_observed_mask = observed_train_mask & ~pseudo_mask
        fill_values = fill_means_tensor.unsqueeze(0).expand_as(target_batch)
        input_values = torch.where(input_observed_mask, target_batch, fill_values)

        optimizer.zero_grad()
        predicted = model(input_values, input_observed_mask)
        loss = F.mse_loss(predicted[pseudo_mask], target_batch[pseudo_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.item()))

    if len(losses) == 0:
        raise RuntimeError("RNN training epoch produced no batches.")

    return float(np.mean(losses))


def _predict_with_mask(
    *,
    model: TabularBiGRUImputer,
    target_tensor: torch.Tensor,
    input_observed_mask: torch.Tensor,
    fill_means_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        target_tensor = target_tensor.to(device)
        input_observed_mask = input_observed_mask.to(device)
        fill_values = fill_means_tensor.to(device).unsqueeze(0).expand_as(target_tensor)
        input_values = torch.where(input_observed_mask, target_tensor, fill_values)
        predicted = model(input_values, input_observed_mask)

    return predicted.cpu().numpy()


# =============================================================================
# RNN imputer
# =============================================================================
def RNN_imputer(
    df_with_na: pd.DataFrame,
    original_df: pd.DataFrame,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    pseudo_mask_rate: float,
    rmse_path: str,
    imputed_data_path: str,
    epochs: int,
    patience: int,
    seed: int = 42,
    internal_val_rate: float = 0.05,
):
    """
    Strict recurrent imputer for tabular proteomics.

    Training uses random pseudo-masking of observed protein cells. Internal early
    stopping uses a separate validation mask drawn only from observed cells. Final
    benchmark missing cells are never used for training or early stopping.
    """
    set_seed(seed)
    device = get_device()

    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    dropout = float(dropout)
    pseudo_mask_rate = float(pseudo_mask_rate)
    epochs = int(epochs)
    patience = int(patience)
    seed = int(seed)

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1. Got {batch_size}")

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive. Got {learning_rate}")

    if hidden_size < 1:
        raise ValueError(f"hidden_size must be positive. Got {hidden_size}")

    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1. Got {num_layers}")

    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1). Got {dropout}")

    if not (0.0 < pseudo_mask_rate < 1.0):
        raise ValueError(
            f"pseudo_mask_rate must be in (0, 1). Got {pseudo_mask_rate}"
        )

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
        raise ValueError(
            "No missing protein cells found in corrupted input. "
            "RNN cannot evaluate imputation RMSE."
        )

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
        seed=seed + 61_013,
    )

    train_mask = observed_mask & ~internal_val_mask

    if train_mask.sum() == 0:
        raise ValueError("RNN train mask has zero cells.")

    train_fill_means = column_means_from_mask(data, train_mask)

    target_filled = fill_missing_with_column_means(
        data=data,
        fill_means=train_fill_means,
        mask_to_fill=nan_mask,
    )

    target_tensor = torch.tensor(target_filled, dtype=torch.float32)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)
    fill_means_tensor = torch.tensor(train_fill_means, dtype=torch.float32, device=device)

    loader = _make_loader(
        target_tensor=target_tensor,
        mask_tensor=train_mask_tensor,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )

    model = TabularBiGRUImputer(
        n_features=corrupted_numeric_df.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\n" + "=" * 80)
    print("RNN training")
    print("=" * 80)
    print(f"device: {device}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"hidden_size: {hidden_size}")
    print(f"num_layers: {num_layers}")
    print(f"dropout: {dropout}")
    print(f"pseudo_mask_rate: {pseudo_mask_rate}")
    print(f"epochs: {epochs}")
    print(f"patience: {patience}")
    print(f"seed: {seed}")
    print(f"n_missing_cells: {n_missing}")
    print(f"n_internal_val_cells: {int(internal_val_mask.sum())}")
    print("=" * 80)

    validation_input_mask = torch.tensor(train_mask, dtype=torch.bool)

    for epoch in range(epochs):
        train_mse = _run_training_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            fill_means_tensor=fill_means_tensor,
            pseudo_mask_rate=pseudo_mask_rate,
            device=device,
        )

        pred_all = _predict_with_mask(
            model=model,
            target_tensor=target_tensor,
            input_observed_mask=validation_input_mask,
            fill_means_tensor=fill_means_tensor,
            device=device,
        )

        val_rmse = rmse_on_mask(reference, pred_all, internal_val_mask)

        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
            marker = " *new best*"
        else:
            patience_counter += 1
            marker = ""

        if epoch == 0 or (epoch + 1) % 10 == 0 or marker:
            print(
                f"Epoch {epoch + 1:03d}/{epochs:03d} | "
                f"pseudo_mask_train_mse={train_mse:.6f} | "
                f"internal_val_rmse={val_rmse:.6f}{marker}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_state is None or best_epoch < 1:
        raise RuntimeError("RNN training did not produce a valid best state.")

    print(f"Best internal validation RMSE: {best_val:.6f} at epoch {best_epoch}")

    # -------------------------------------------------------------------------
    # Final refit on all observed cells for best_epoch
    # -------------------------------------------------------------------------
    set_seed(seed + 67_001)

    full_train_mask = observed_mask
    full_fill_means = column_means_from_mask(data, full_train_mask)

    full_target_filled = fill_missing_with_column_means(
        data=data,
        fill_means=full_fill_means,
        mask_to_fill=nan_mask,
    )

    full_target_tensor = torch.tensor(full_target_filled, dtype=torch.float32)
    full_mask_tensor = torch.tensor(full_train_mask, dtype=torch.bool)
    full_fill_means_tensor = torch.tensor(
        full_fill_means,
        dtype=torch.float32,
        device=device,
    )

    full_loader = _make_loader(
        target_tensor=full_target_tensor,
        mask_tensor=full_mask_tensor,
        batch_size=batch_size,
        seed=seed + 67_001,
        shuffle=True,
    )

    final_model = TabularBiGRUImputer(
        n_features=corrupted_numeric_df.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    final_optimizer = optim.AdamW(
        final_model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    for epoch in range(best_epoch):
        train_mse = _run_training_epoch(
            model=final_model,
            loader=full_loader,
            optimizer=final_optimizer,
            fill_means_tensor=full_fill_means_tensor,
            pseudo_mask_rate=pseudo_mask_rate,
            device=device,
        )

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == best_epoch:
            print(
                f"Final refit epoch {epoch + 1:03d}/{best_epoch:03d} | "
                f"pseudo_mask_train_mse={train_mse:.6f}"
            )

    pred_full = _predict_with_mask(
        model=final_model,
        target_tensor=full_target_tensor,
        input_observed_mask=full_mask_tensor,
        fill_means_tensor=full_fill_means_tensor,
        device=device,
    )

    final_numeric_df = fill_only_missing_cells(
        observed_df=corrupted_numeric_df,
        predicted_values=pred_full,
        nan_mask=nan_mask,
    )

    final_values = final_numeric_df.to_numpy(dtype=float)
    test_rmse = rmse_on_mask(reference, final_values, nan_mask)

    if not np.isfinite(test_rmse):
        raise FloatingPointError(f"RNN final RMSE is non-finite: {test_rmse}")

    imputed_df = pd.concat(
        [
            metadata_df.reset_index(drop=True),
            final_numeric_df.reset_index(drop=True),
        ],
        axis=1,
    )

    if list(imputed_df.columns[:2]) != ["eid", "label"]:
        raise RuntimeError("Internal error: RNN output does not preserve eid and label.")

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
    print(f"RNN final RMSE on missing cells: {test_rmse:.6f}")
    print(f"RMSE saved to: {rmse_path}")
    print(f"Imputed data saved to: {imputed_data_path}")
    print("=" * 80)

    return imputed_df, test_rmse


# =============================================================================
# File-handling driver
# =============================================================================
def rnn(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    batch_size,
    learning_rate,
    hidden_size,
    num_layers,
    dropout,
    pseudo_mask_rate,
    epochs=300,
    patience=50,
    seed: int = 42,
):
    """
    Strict seeded RNN wrapper.

    Protein columns are processed in their fixed dataset column order by a
    bidirectional GRU. The observed-mask channel is supplied explicitly.

    Required input:
        Corrupted_data_fraction_<fraction>_missingness_<missingness>_col_<columns_num>_seed_<seed>.csv

    Output:
        RMSE_RNN_col_<columns_num>_params_<param_id>_seed_<seed>.csv
        imputed_RNN_col_<columns_num>_params_<param_id>_seed_<seed>.csv
    """
    main_directory = Path(experiments_dir(columns_num))

    if not main_directory.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist:\n{main_directory}\n"
            "Run preprocessing before RNN."
        )

    seed = int(seed)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    dropout = float(dropout)
    pseudo_mask_rate = float(pseudo_mask_rate)

    param_id = _param_id(
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pseudo_mask_rate=pseudo_mask_rate,
    )

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()

            print(
                f"\n>>> Running RNN | missingness={missingness} | "
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
                    f"\n[CRITICAL] RNN could not find the exact seeded corrupted file:\n"
                    f"{corrupted_data_path}\n\n"
                    "Run corrected preprocessing for this exact fraction, missingness type, and seed."
                )

            if not normalized_data_path.exists():
                raise FileNotFoundError(
                    f"\n[CRITICAL] RNN could not find normalized/reference data:\n"
                    f"{normalized_data_path}"
                )

            rmse_path = (
                full_path
                / f"RMSE_RNN_col_{columns_num}_params_{param_id}_seed_{seed}.csv"
            )

            imputed_path = (
                full_path
                / f"imputed_RNN_col_{columns_num}_params_{param_id}_seed_{seed}.csv"
            )

            df_with_na = pd.read_csv(corrupted_data_path)
            original_df = pd.read_csv(normalized_data_path)

            RNN_imputer(
                df_with_na=df_with_na,
                original_df=original_df,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                pseudo_mask_rate=pseudo_mask_rate,
                rmse_path=str(rmse_path),
                imputed_data_path=str(imputed_path),
                epochs=int(epochs),
                patience=int(patience),
                seed=seed,
                internal_val_rate=0.05,
            )

    print("\n--- RNN imputation pipeline complete ---")
