# imputation/Transformer.py

from __future__ import annotations

import ast
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
# Parameter and filename helpers
# =============================================================================
def _float_token(value: float) -> str:
    token = f"{float(value):.12g}"
    token = token.replace("-", "m")
    token = token.replace(".", "p")
    return token


def transformer_param_id(params: Dict[str, Any]) -> str:
    required = {
        "batch_size",
        "learning_rate",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "pseudo_mask_rate",
        "weight_decay",
        "grad_clip",
        "internal_val_rate",
    }
    missing = required - set(params)

    if missing:
        raise KeyError(
            f"Transformer parameters are missing required keys: {sorted(missing)}. "
            f"params={params}"
        )

    return (
        f"bs{int(params['batch_size'])}"
        f"_lr{_float_token(float(params['learning_rate']))}"
        f"_dm{int(params['d_model'])}"
        f"_nh{int(params['nhead'])}"
        f"_nl{int(params['num_layers'])}"
        f"_ff{int(params['dim_feedforward'])}"
        f"_do{_float_token(float(params['dropout']))}"
        f"_pm{_float_token(float(params['pseudo_mask_rate']))}"
        f"_wd{_float_token(float(params['weight_decay']))}"
        f"_gc{_float_token(float(params['grad_clip']))}"
        f"_iv{_float_token(float(params['internal_val_rate']))}"
    )


# =============================================================================
# Strict input validation
# =============================================================================
def _numeric_frame(df: pd.DataFrame, columns: Sequence[str], label: str) -> pd.DataFrame:
    try:
        out = df.loc[:, columns].apply(lambda series: pd.to_numeric(series, errors="raise"))
    except Exception as exc:
        raise ValueError(f"{label} contains a non-numeric protein value: {exc}") from exc

    return out


def _validate_input_frames(
    corrupted_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> None:
    if corrupted_df.shape[1] < 3:
        raise ValueError(
            "Corrupted data must contain eid, label, and at least one protein column. "
            f"Observed shape: {corrupted_df.shape}"
        )

    if reference_df.shape[1] < 3:
        raise ValueError(
            "Reference data must contain eid, label, and at least one protein column. "
            f"Observed shape: {reference_df.shape}"
        )

    expected_metadata = ["eid", "label"]

    if list(corrupted_df.columns[:2]) != expected_metadata:
        raise ValueError(
            f"Corrupted data first two columns must be {expected_metadata}. "
            f"Observed: {list(corrupted_df.columns[:2])}"
        )

    if list(reference_df.columns[:2]) != expected_metadata:
        raise ValueError(
            f"Reference data first two columns must be {expected_metadata}. "
            f"Observed: {list(reference_df.columns[:2])}"
        )

    if corrupted_df.shape != reference_df.shape:
        raise ValueError(
            "Corrupted and reference data have different shapes.\n"
            f"Corrupted shape: {corrupted_df.shape}\n"
            f"Reference shape: {reference_df.shape}"
        )

    if list(corrupted_df.columns) != list(reference_df.columns):
        raise ValueError(
            "Corrupted and reference data columns differ or are reordered."
        )

    if not corrupted_df["eid"].astype(str).equals(reference_df["eid"].astype(str)):
        raise ValueError(
            "eid column does not match between corrupted and reference data."
        )

    if not corrupted_df["label"].astype(str).equals(reference_df["label"].astype(str)):
        raise ValueError(
            "label column does not match between corrupted and reference data."
        )


# =============================================================================
# Mask, fill, and metric helpers
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
            "data and observed_mask must have the same shape. "
            f"data={data.shape}, observed_mask={observed_mask.shape}"
        )

    if not (0.0 < float(val_rate) < 1.0):
        raise ValueError(f"val_rate must be in (0, 1). Got {val_rate}")

    rng = np.random.default_rng(int(seed))
    validation_mask = np.zeros_like(observed_mask, dtype=bool)

    for column_index in range(data.shape[1]):
        candidate_rows = np.where(
            observed_mask[:, column_index]
            & np.isfinite(data[:, column_index])
        )[0]

        # At least one observed value must remain for fitting/filling.
        if candidate_rows.size < 2:
            continue

        n_validation = int(np.ceil(candidate_rows.size * float(val_rate)))
        n_validation = min(max(n_validation, 1), candidate_rows.size - 1)

        chosen = rng.choice(candidate_rows, size=n_validation, replace=False)
        validation_mask[chosen, column_index] = True

    if int(validation_mask.sum()) == 0:
        raise ValueError(
            "Internal Transformer validation mask contains zero cells."
        )

    return validation_mask


def column_means_from_mask(
    data: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)

    if data.shape != train_mask.shape:
        raise ValueError("data and train_mask have different shapes.")

    means = np.empty(data.shape[1], dtype=float)
    bad_columns: List[int] = []

    for column_index in range(data.shape[1]):
        values = data[train_mask[:, column_index], column_index]
        values = values[np.isfinite(values)]

        if values.size == 0:
            bad_columns.append(column_index)
            means[column_index] = np.nan
        else:
            means[column_index] = float(np.mean(values))

    if bad_columns:
        raise ValueError(
            "Some protein columns contain no train-observed values for Transformer "
            "fill initialization. Bad column indices, first 20: "
            f"{bad_columns[:20]}"
        )

    return means


def fill_masked_cells(
    data: np.ndarray,
    fill_values: np.ndarray,
    mask_to_fill: np.ndarray,
) -> np.ndarray:
    output = np.asarray(data, dtype=float).copy()
    mask_to_fill = np.asarray(mask_to_fill, dtype=bool)

    if output.shape != mask_to_fill.shape:
        raise ValueError("data and mask_to_fill have different shapes.")

    if fill_values.shape != (output.shape[1],):
        raise ValueError(
            "fill_values must have one value per protein column. "
            f"Expected {(output.shape[1],)}, got {fill_values.shape}."
        )

    for column_index in range(output.shape[1]):
        output[mask_to_fill[:, column_index], column_index] = fill_values[column_index]

    if not np.isfinite(output).all():
        raise ValueError(
            "Transformer input contains non-finite values after column-mean filling."
        )

    return output.astype(np.float32)


def rmse_on_mask(
    reference: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> float:
    reference = np.asarray(reference, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if reference.shape != predictions.shape or reference.shape != mask.shape:
        raise ValueError(
            "reference, predictions, and mask must have the same shape. "
            f"reference={reference.shape}, predictions={predictions.shape}, "
            f"mask={mask.shape}"
        )

    valid = mask & np.isfinite(reference) & np.isfinite(predictions)
    n_valid = int(valid.sum())

    if n_valid == 0:
        raise ValueError("No finite cells are available for RMSE calculation.")

    return float(
        np.sqrt(np.mean((reference[valid] - predictions[valid]) ** 2))
    )


def preserve_observed_and_fill_missing(
    corrupted_numeric_df: pd.DataFrame,
    predictions: np.ndarray,
    missing_mask: np.ndarray,
) -> pd.DataFrame:
    observed_values = corrupted_numeric_df.to_numpy(dtype=float)

    if predictions.shape != observed_values.shape:
        raise ValueError(
            "Transformer predictions have the wrong shape. "
            f"predictions={predictions.shape}, expected={observed_values.shape}"
        )

    output = observed_values.copy()
    output[missing_mask] = predictions[missing_mask]

    observed_mask = ~missing_mask

    if not np.array_equal(
        output[observed_mask],
        observed_values[observed_mask],
        equal_nan=True,
    ):
        raise RuntimeError(
            "Transformer output changed values that were observed in the corrupted input."
        )

    if not np.isfinite(output).all():
        raise FloatingPointError(
            "Transformer final imputed protein matrix contains non-finite values."
        )

    return pd.DataFrame(
        output,
        index=corrupted_numeric_df.index,
        columns=corrupted_numeric_df.columns,
    )


def write_rmse_csv(
    rmse_path: Path,
    *,
    final_rmse: float,
    internal_best_val_rmse: float,
    seed: int,
    best_epoch: int,
    n_missing_cells: int,
    n_internal_val_cells: int,
    params: Dict[str, Any],
) -> None:
    rmse_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "rmse_transformer": [float(final_rmse)],
        "internal_best_val_rmse": [float(internal_best_val_rmse)],
        "seed": [int(seed)],
        "best_epoch": [int(best_epoch)],
        "n_missing_cells": [int(n_missing_cells)],
        "n_internal_val_cells": [int(n_internal_val_cells)],
    }

    for key, value in params.items():
        payload[key] = [value]

    pd.DataFrame(payload).to_csv(rmse_path, index=False)


# =============================================================================
# Feature-tokenized Transformer
# =============================================================================
class FeatureTokenTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()

        n_features = int(n_features)
        d_model = int(d_model)
        nhead = int(nhead)
        num_layers = int(num_layers)
        dim_feedforward = int(dim_feedforward)
        dropout = float(dropout)

        if n_features < 1:
            raise ValueError(f"n_features must be positive. Got {n_features}")

        if d_model < 1:
            raise ValueError(f"d_model must be positive. Got {d_model}")

        if nhead < 1:
            raise ValueError(f"nhead must be positive. Got {nhead}")

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by nhead={nhead}."
            )

        if num_layers < 1:
            raise ValueError(f"num_layers must be positive. Got {num_layers}")

        if dim_feedforward < d_model:
            raise ValueError(
                "dim_feedforward must be at least d_model. "
                f"Got dim_feedforward={dim_feedforward}, d_model={d_model}."
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1). Got {dropout}")

        self.n_features = n_features
        self.d_model = d_model

        # Each protein has its own scalar-to-token projection and positional identity.
        self.feature_scale = nn.Parameter(
            torch.randn(n_features, d_model) * 0.02
        )
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.missing_embedding = nn.Parameter(torch.randn(d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_weight = nn.Parameter(
            torch.randn(n_features, d_model) * 0.02
        )
        self.output_bias = nn.Parameter(torch.zeros(n_features))

    def forward(
        self,
        values: torch.Tensor,
        observed_input_mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(
                f"values must be 2D [batch, features]. Got {values.shape}"
            )

        if observed_input_mask.shape != values.shape:
            raise ValueError(
                "observed_input_mask and values must have the same shape."
            )

        if values.shape[1] != self.n_features:
            raise ValueError(
                f"Model expects {self.n_features} features, got {values.shape[1]}."
            )

        observed = observed_input_mask.unsqueeze(-1)
        observed_tokens = values.unsqueeze(-1) * self.feature_scale.unsqueeze(0)
        missing_tokens = self.missing_embedding.view(1, 1, -1).expand(
            values.shape[0],
            self.n_features,
            self.d_model,
        )

        tokens = torch.where(observed, observed_tokens, missing_tokens)
        tokens = tokens + self.feature_bias.unsqueeze(0)

        encoded = self.encoder(tokens)

        predictions = (
            encoded * self.output_weight.unsqueeze(0)
        ).sum(dim=-1) + self.output_bias.unsqueeze(0)

        return predictions


# =============================================================================
# Training helpers
# =============================================================================
def _make_loader(
    values_tensor: torch.Tensor,
    observed_mask_tensor: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    if values_tensor.shape != observed_mask_tensor.shape:
        raise ValueError(
            "values_tensor and observed_mask_tensor have different shapes."
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    dataset = TensorDataset(values_tensor, observed_mask_tensor)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        drop_last=False,
        num_workers=0,
    )


def sample_pseudo_mask(
    observed_mask: torch.Tensor,
    pseudo_mask_rate: float,
) -> torch.Tensor:
    pseudo_mask_rate = float(pseudo_mask_rate)

    if not (0.0 < pseudo_mask_rate < 1.0):
        raise ValueError(
            f"pseudo_mask_rate must be in (0, 1). Got {pseudo_mask_rate}"
        )

    if int(observed_mask.sum().item()) == 0:
        raise ValueError("Training batch contains no observed cells.")

    pseudo_mask = (
        torch.rand_like(observed_mask, dtype=torch.float32)
        < pseudo_mask_rate
    ) & observed_mask

    # A valid pseudo-masking objective requires at least one target cell.
    if int(pseudo_mask.sum().item()) == 0:
        candidates = torch.nonzero(observed_mask, as_tuple=False)
        selected_index = torch.randint(
            low=0,
            high=candidates.shape[0],
            size=(1,),
            device=candidates.device,
        ).item()
        selected = candidates[selected_index]
        pseudo_mask[selected[0], selected[1]] = True

    return pseudo_mask


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if predictions.shape != targets.shape or predictions.shape != target_mask.shape:
        raise ValueError(
            "predictions, targets, and target_mask must have the same shape."
        )

    n_targets = int(target_mask.sum().item())

    if n_targets == 0:
        raise ValueError("Masked MSE received a target mask with zero cells.")

    return F.mse_loss(
        predictions[target_mask],
        targets[target_mask],
        reduction="mean",
    )


def _train_one_epoch(
    *,
    model: FeatureTokenTransformer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    pseudo_mask_rate: float,
    grad_clip: float,
) -> float:
    model.train()
    losses: List[float] = []

    for values_batch, observed_batch in loader:
        values_batch = values_batch.to(device)
        observed_batch = observed_batch.to(device)

        pseudo_mask = sample_pseudo_mask(
            observed_batch,
            pseudo_mask_rate=pseudo_mask_rate,
        )
        input_mask = observed_batch & ~pseudo_mask

        optimizer.zero_grad(set_to_none=True)
        predictions = model(values_batch, input_mask)
        loss = masked_mse_loss(
            predictions,
            values_batch,
            pseudo_mask,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Transformer training loss is non-finite: {loss.item()}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(grad_clip),
        )
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        raise RuntimeError("Transformer training loader produced no batches.")

    return float(np.mean(losses))


def _predict(
    *,
    model: FeatureTokenTransformer,
    values_tensor: torch.Tensor,
    input_mask_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = _make_loader(
        values_tensor,
        input_mask_tensor,
        batch_size=batch_size,
        seed=0,
        shuffle=False,
    )

    model.eval()
    outputs: List[np.ndarray] = []

    with torch.no_grad():
        for values_batch, mask_batch in loader:
            predictions = model(
                values_batch.to(device),
                mask_batch.to(device),
            )
            outputs.append(predictions.cpu().numpy())

    if not outputs:
        raise RuntimeError("Transformer prediction loader produced no batches.")

    prediction_matrix = np.concatenate(outputs, axis=0)

    if not np.isfinite(prediction_matrix).all():
        raise FloatingPointError(
            "Transformer predictions contain non-finite values."
        )

    return prediction_matrix


# =============================================================================
# Strict Transformer imputer
# =============================================================================
def Transformer_imputer(
    *,
    corrupted_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    batch_size: int,
    learning_rate: float,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    pseudo_mask_rate: float,
    weight_decay: float,
    grad_clip: float,
    internal_val_rate: float,
    rmse_path: Path,
    imputed_data_path: Path,
    epochs: int,
    patience: int,
    seed: int,
) -> Tuple[pd.DataFrame, float]:
    set_seed(seed)
    device = get_device()

    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    d_model = int(d_model)
    nhead = int(nhead)
    num_layers = int(num_layers)
    dim_feedforward = int(dim_feedforward)
    dropout = float(dropout)
    pseudo_mask_rate = float(pseudo_mask_rate)
    weight_decay = float(weight_decay)
    grad_clip = float(grad_clip)
    internal_val_rate = float(internal_val_rate)
    epochs = int(epochs)
    patience = int(patience)
    seed = int(seed)

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive. Got {batch_size}")

    if learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be positive. Got {learning_rate}"
        )

    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative. Got {weight_decay}")

    if grad_clip <= 0:
        raise ValueError(f"grad_clip must be positive. Got {grad_clip}")

    if epochs < 1:
        raise ValueError(f"epochs must be positive. Got {epochs}")

    if patience < 1:
        raise ValueError(f"patience must be positive. Got {patience}")

    _validate_input_frames(corrupted_df, reference_df)

    metadata_df = corrupted_df.iloc[:, :2].copy()
    protein_columns = list(corrupted_df.columns[2:])

    corrupted_numeric_df = _numeric_frame(
        corrupted_df,
        protein_columns,
        "Corrupted data",
    )
    reference_numeric_df = _numeric_frame(
        reference_df,
        protein_columns,
        "Reference data",
    )

    data = corrupted_numeric_df.to_numpy(dtype=float)
    reference = reference_numeric_df.to_numpy(dtype=float)

    if not np.isfinite(reference).all():
        bad_count = int((~np.isfinite(reference)).sum())
        raise ValueError(
            f"Reference protein matrix contains {bad_count} non-finite values. "
            "A complete benchmark reference is required."
        )

    invalid_corrupted = ~(np.isfinite(data) | np.isnan(data))
    if invalid_corrupted.any():
        raise ValueError(
            "Corrupted protein matrix contains infinite values."
        )

    missing_mask = np.isnan(data)
    observed_mask = np.isfinite(data)
    n_missing_cells = int(missing_mask.sum())

    if n_missing_cells == 0:
        raise ValueError(
            "No missing protein cells were found in the corrupted input. "
            "Transformer imputation requires missing targets."
        )

    internal_validation_mask = make_cell_validation_mask(
        data=data,
        observed_mask=observed_mask,
        val_rate=internal_val_rate,
        seed=seed + 73_109,
    )
    training_observed_mask = observed_mask & ~internal_validation_mask

    if int(training_observed_mask.sum()) == 0:
        raise ValueError("Transformer training mask contains zero cells.")

    training_means = column_means_from_mask(
        data,
        training_observed_mask,
    )
    training_input = fill_masked_cells(
        data,
        training_means,
        missing_mask | internal_validation_mask,
    )

    values_tensor = torch.tensor(training_input, dtype=torch.float32)
    training_mask_tensor = torch.tensor(
        training_observed_mask,
        dtype=torch.bool,
    )
    validation_input_mask_tensor = torch.tensor(
        observed_mask & ~internal_validation_mask,
        dtype=torch.bool,
    )

    training_loader = _make_loader(
        values_tensor,
        training_mask_tensor,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )

    model = FeatureTokenTransformer(
        n_features=data.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_validation_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\n" + "=" * 88)
    print("Transformer training")
    print("=" * 88)
    print(f"device: {device}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"d_model: {d_model}")
    print(f"nhead: {nhead}")
    print(f"num_layers: {num_layers}")
    print(f"dim_feedforward: {dim_feedforward}")
    print(f"dropout: {dropout}")
    print(f"pseudo_mask_rate: {pseudo_mask_rate}")
    print(f"weight_decay: {weight_decay}")
    print(f"grad_clip: {grad_clip}")
    print(f"internal_val_rate: {internal_val_rate}")
    print(f"epochs: {epochs}")
    print(f"patience: {patience}")
    print(f"seed: {seed}")
    print(f"n_missing_cells: {n_missing_cells}")
    print(f"n_internal_val_cells: {int(internal_validation_mask.sum())}")
    print("=" * 88)

    for epoch_index in range(epochs):
        train_loss = _train_one_epoch(
            model=model,
            loader=training_loader,
            optimizer=optimizer,
            device=device,
            pseudo_mask_rate=pseudo_mask_rate,
            grad_clip=grad_clip,
        )

        validation_predictions = _predict(
            model=model,
            values_tensor=values_tensor,
            input_mask_tensor=validation_input_mask_tensor,
            device=device,
            batch_size=batch_size,
        )
        validation_rmse = rmse_on_mask(
            reference,
            validation_predictions,
            internal_validation_mask,
        )

        improved = validation_rmse < best_validation_rmse

        if improved:
            best_validation_rmse = validation_rmse
            best_epoch = epoch_index + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if (
            epoch_index == 0
            or improved
            or (epoch_index + 1) % 10 == 0
        ):
            marker = " *new best*" if improved else ""
            print(
                f"Epoch {epoch_index + 1:03d}/{epochs:03d} | "
                f"train_mse={train_loss:.6f} | "
                f"internal_val_rmse={validation_rmse:.6f}{marker}"
            )

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch_index + 1}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_epoch < 1 or not np.isfinite(best_validation_rmse):
        raise RuntimeError(
            "Transformer training did not produce a valid best epoch."
        )

    print(
        f"Best internal validation RMSE: {best_validation_rmse:.6f} "
        f"at epoch {best_epoch}."
    )

    # -------------------------------------------------------------------------
    # Final refit on every observed protein cell for the selected epoch count.
    # -------------------------------------------------------------------------
    set_seed(seed + 79_003)

    final_means = column_means_from_mask(data, observed_mask)
    final_input = fill_masked_cells(
        data,
        final_means,
        missing_mask,
    )

    final_values_tensor = torch.tensor(final_input, dtype=torch.float32)
    final_observed_mask_tensor = torch.tensor(
        observed_mask,
        dtype=torch.bool,
    )

    final_loader = _make_loader(
        final_values_tensor,
        final_observed_mask_tensor,
        batch_size=batch_size,
        seed=seed + 79_003,
        shuffle=True,
    )

    final_model = FeatureTokenTransformer(
        n_features=data.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    final_optimizer = optim.AdamW(
        final_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for epoch_index in range(best_epoch):
        final_train_loss = _train_one_epoch(
            model=final_model,
            loader=final_loader,
            optimizer=final_optimizer,
            device=device,
            pseudo_mask_rate=pseudo_mask_rate,
            grad_clip=grad_clip,
        )

        if (
            epoch_index == 0
            or (epoch_index + 1) % 10 == 0
            or epoch_index + 1 == best_epoch
        ):
            print(
                f"Final refit epoch {epoch_index + 1:03d}/{best_epoch:03d} | "
                f"train_mse={final_train_loss:.6f}"
            )

    final_predictions = _predict(
        model=final_model,
        values_tensor=final_values_tensor,
        input_mask_tensor=final_observed_mask_tensor,
        device=device,
        batch_size=batch_size,
    )

    final_numeric_df = preserve_observed_and_fill_missing(
        corrupted_numeric_df,
        final_predictions,
        missing_mask,
    )
    final_values = final_numeric_df.to_numpy(dtype=float)
    final_rmse = rmse_on_mask(reference, final_values, missing_mask)

    imputed_df = pd.concat(
        [
            metadata_df.reset_index(drop=True),
            final_numeric_df.reset_index(drop=True),
        ],
        axis=1,
    )

    if list(imputed_df.columns[:2]) != ["eid", "label"]:
        raise RuntimeError(
            "Internal error: Transformer output does not preserve eid and label."
        )

    params = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "pseudo_mask_rate": pseudo_mask_rate,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "internal_val_rate": internal_val_rate,
    }

    write_rmse_csv(
        rmse_path,
        final_rmse=final_rmse,
        internal_best_val_rmse=best_validation_rmse,
        seed=seed,
        best_epoch=best_epoch,
        n_missing_cells=n_missing_cells,
        n_internal_val_cells=int(internal_validation_mask.sum()),
        params=params,
    )

    imputed_data_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_df.to_csv(imputed_data_path, index=False)

    print("\n" + "=" * 88)
    print(f"Transformer final RMSE on missing cells: {final_rmse:.6f}")
    print(f"RMSE saved to: {rmse_path}")
    print(f"Imputed data saved to: {imputed_data_path}")
    print("=" * 88)

    return imputed_df, final_rmse


# =============================================================================
# File-handling driver used by the shared optimization pipeline
# =============================================================================
def transformer(
    file_path,
    columns_num,
    fractions,
    missingness_types,
    batch_size,
    learning_rate,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    dropout,
    pseudo_mask_rate,
    weight_decay,
    grad_clip,
    internal_val_rate,
    epochs=100,
    patience=12,
    seed: int = 42,
):
    """
    Strict seeded Transformer wrapper.

    Required input:
        Corrupted_data_fraction_<fraction>_missingness_<missingness>_col_<columns_num>_seed_<seed>.csv

    Outputs:
        RMSE_TRF_col_<columns_num>_params_<param_id>_seed_<seed>.csv
        imputed_TRF_col_<columns_num>_params_<param_id>_seed_<seed>.csv
    """
    main_directory = Path(experiments_dir(columns_num))

    if not main_directory.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist:\n{main_directory}\n"
            "Run preprocessing before the Transformer."
        )

    params = {
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "d_model": int(d_model),
        "nhead": int(nhead),
        "num_layers": int(num_layers),
        "dim_feedforward": int(dim_feedforward),
        "dropout": float(dropout),
        "pseudo_mask_rate": float(pseudo_mask_rate),
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
        "internal_val_rate": float(internal_val_rate),
    }
    param_id = transformer_param_id(params)
    seed = int(seed)

    for fraction_value in fractions:
        fraction = float(fraction_value)

        for missingness_value in missingness_types:
            missingness = str(missingness_value).upper()
            folder = (
                main_directory
                / f"fraction_{fraction}_missingness_{missingness}"
            )

            if not folder.exists():
                raise FileNotFoundError(
                    f"Required fraction/missingness directory does not exist:\n{folder}"
                )

            corrupted_path = (
                folder
                / (
                    f"Corrupted_data_fraction_{fraction}_missingness_{missingness}"
                    f"_col_{columns_num}_seed_{seed}.csv"
                )
            )
            reference_path = folder / f"normalized_data_col_{columns_num}.csv"

            if not corrupted_path.exists():
                raise FileNotFoundError(
                    "Transformer could not find the exact seeded corrupted file:\n"
                    f"{corrupted_path}\n"
                    "Run corrected preprocessing for this exact condition and seed."
                )

            if not reference_path.exists():
                raise FileNotFoundError(
                    f"Transformer reference file does not exist:\n{reference_path}"
                )

            rmse_path = (
                folder
                / (
                    f"RMSE_TRF_col_{columns_num}_params_{param_id}"
                    f"_seed_{seed}.csv"
                )
            )
            imputed_path = (
                folder
                / (
                    f"imputed_TRF_col_{columns_num}_params_{param_id}"
                    f"_seed_{seed}.csv"
                )
            )

            print(
                f"\n>>> Running Transformer | missingness={missingness} | "
                f"fraction={fraction} | seed={seed}"
            )

            corrupted_df = pd.read_csv(corrupted_path)
            reference_df = pd.read_csv(reference_path)

            Transformer_imputer(
                corrupted_df=corrupted_df,
                reference_df=reference_df,
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate"],
                d_model=params["d_model"],
                nhead=params["nhead"],
                num_layers=params["num_layers"],
                dim_feedforward=params["dim_feedforward"],
                dropout=params["dropout"],
                pseudo_mask_rate=params["pseudo_mask_rate"],
                weight_decay=params["weight_decay"],
                grad_clip=params["grad_clip"],
                internal_val_rate=params["internal_val_rate"],
                rmse_path=rmse_path,
                imputed_data_path=imputed_path,
                epochs=int(epochs),
                patience=int(patience),
                seed=seed,
            )

    print("\n--- Transformer imputation pipeline complete ---")
