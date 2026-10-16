# imputation/GAIN.py

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# Reproducibility and filename helpers
# =============================================================================
def set_all_seeds(seed: int) -> None:
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


def stable_combo_seed(seed: int, fraction: float, missingness: str, stage: str) -> int:
    key = (
        f"{int(seed)}|{float(fraction):.12g}|"
        f"{str(missingness).upper()}|{str(stage)}"
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % 1_000_000_000
    return int(seed) + offset


def _float_token(x: float) -> str:
    s = f"{float(x):.12g}"
    return s.replace("-", "m").replace(".", "p").replace("+", "")


def _text_token(x: Any) -> str:
    text = str(x).strip().lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "_")
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _parse_n_neurons(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        result = [int(x) for x in value]
    elif isinstance(value, str):
        import ast

        s = value.strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                result = [int(x) for x in parsed]
            else:
                raise ValueError
        except Exception:
            s = s.replace("[", "").replace("]", "")
            s = s.replace("(", "").replace(")", "")
            s = s.replace("-", ",")
            result = [int(x.strip()) for x in s.split(",") if x.strip()]
    else:
        raise ValueError(f"Could not parse n_neurons from value: {value}")

    if not result or any(x <= 0 for x in result):
        raise ValueError(f"n_neurons must contain positive integers. Got: {result}")

    return result


def gain_param_id(
    params: Dict[str, Any],
    *,
    epochs: int,
    patience: int,
) -> str:
    required = {
        "batch_size",
        "lr",
        "n_neurons",
        "alpha",
        "hint_rate",
        "dropout_rate",
        "weight_decay",
        "pseudo_rate",
        "pseudo_beta",
        "hint_mode",
        "mc_samples_final",
    }
    missing = required - set(params)
    if missing:
        raise KeyError(f"GAIN params missing required keys: {sorted(missing)}")

    neurons = _parse_n_neurons(params["n_neurons"])
    nn_token = "-".join(str(x) for x in neurons)

    return (
        f"bs{int(params['batch_size'])}"
        f"_lr{_float_token(float(params['lr']))}"
        f"_nn{nn_token}"
        f"_al{_float_token(float(params['alpha']))}"
        f"_hr{_float_token(float(params['hint_rate']))}"
        f"_do{_float_token(float(params['dropout_rate']))}"
        f"_wd{_float_token(float(params['weight_decay']))}"
        f"_ps{_float_token(float(params['pseudo_rate']))}"
        f"_pb{_float_token(float(params['pseudo_beta']))}"
        f"_hm{_text_token(params['hint_mode'])}"
        f"_mc{int(params['mc_samples_final'])}"
        f"_ep{int(epochs)}"
        f"_pat{int(patience)}"
    )


# =============================================================================
# Strict data validation and metrics
# =============================================================================
def _validate_input_frames(
    df_with_na: pd.DataFrame,
    original_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for label, df in [
        ("Corrupted data", df_with_na),
        ("Reference data", original_df),
    ]:
        if df.shape[1] < 3:
            raise ValueError(
                f"{label} must contain eid, label, and at least one protein column. "
                f"Observed shape: {df.shape}"
            )
        if list(df.columns[:2]) != ["eid", "label"]:
            raise ValueError(
                f"{label} first two columns must be ['eid', 'label']. "
                f"Observed: {list(df.columns[:2])}"
            )

    if df_with_na.shape != original_df.shape:
        raise ValueError(
            "Corrupted and reference data have different shapes.\n"
            f"Corrupted: {df_with_na.shape}\nReference: {original_df.shape}"
        )

    if list(df_with_na.columns) != list(original_df.columns):
        raise ValueError("Corrupted and reference columns differ or are reordered.")

    if not df_with_na["eid"].astype(str).equals(original_df["eid"].astype(str)):
        raise ValueError("eid values or row order differ between corrupted and reference data.")

    if not df_with_na["label"].astype(str).equals(original_df["label"].astype(str)):
        raise ValueError("label values or row order differ between corrupted and reference data.")

    corrupted_numeric = df_with_na.iloc[:, 2:].apply(pd.to_numeric, errors="raise")
    reference_numeric = original_df.iloc[:, 2:].apply(pd.to_numeric, errors="raise")

    reference_values = reference_numeric.to_numpy(dtype=np.float32)
    if not np.isfinite(reference_values).all():
        bad = int((~np.isfinite(reference_values)).sum())
        raise ValueError(
            f"Reference protein matrix contains {bad} non-finite values. "
            "The benchmark reference must be complete."
        )

    return corrupted_numeric, reference_numeric


def rmse_on_mask(
    original: np.ndarray,
    completed: np.ndarray,
    eval_mask: np.ndarray,
) -> float:
    original = np.asarray(original, dtype=float)
    completed = np.asarray(completed, dtype=float)
    eval_mask = np.asarray(eval_mask, dtype=bool)

    if original.shape != completed.shape or original.shape != eval_mask.shape:
        raise ValueError(
            "original, completed, and eval_mask must have identical shapes. "
            f"Got {original.shape}, {completed.shape}, {eval_mask.shape}."
        )

    valid = eval_mask & np.isfinite(original) & np.isfinite(completed)
    if int(valid.sum()) == 0:
        raise ValueError("No finite cells are available for GAIN RMSE evaluation.")

    return float(np.sqrt(np.mean((original[valid] - completed[valid]) ** 2)))


def fill_only_missing_cells(
    corrupted_numeric: pd.DataFrame,
    generated_values: np.ndarray,
    missing_mask: np.ndarray,
) -> pd.DataFrame:
    observed = corrupted_numeric.to_numpy(dtype=float)
    generated_values = np.asarray(generated_values, dtype=float)
    missing_mask = np.asarray(missing_mask, dtype=bool)

    if observed.shape != generated_values.shape or observed.shape != missing_mask.shape:
        raise ValueError("Observed values, generated values, and missing mask differ in shape.")

    completed = observed.copy()
    completed[missing_mask] = generated_values[missing_mask]

    if not np.isfinite(completed).all():
        bad = int((~np.isfinite(completed)).sum())
        raise FloatingPointError(f"GAIN completed matrix contains {bad} non-finite values.")

    return pd.DataFrame(
        completed,
        index=corrupted_numeric.index,
        columns=corrupted_numeric.columns,
    )


def _csv_safe_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if value is None:
        return "none"
    return value


def write_rmse_csv(
    *,
    rmse_path: Path,
    test_rmse: float,
    internal_best_val_rmse: float,
    best_epoch: int,
    seed: int,
    n_missing_cells: int,
    n_internal_val_cells: int,
    max_observed_difference: float,
    config: Dict[str, Any],
) -> None:
    rmse_path.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, List[Any]] = {
        "rmse_gain": [float(test_rmse)],
        "internal_best_val_rmse": [float(internal_best_val_rmse)],
        "best_epoch": [int(best_epoch)],
        "seed": [int(seed)],
        "n_missing_cells": [int(n_missing_cells)],
        "n_internal_val_cells": [int(n_internal_val_cells)],
        "max_observed_difference": [float(max_observed_difference)],
    }

    for key, value in config.items():
        row[str(key)] = [_csv_safe_value(value)]

    pd.DataFrame(row).to_csv(rmse_path, index=False)


# =============================================================================
# Validation mask, output scale, and noise
# =============================================================================
def make_validation_mask(
    *,
    data: np.ndarray,
    observed_mask: np.ndarray,
    val_rate: float,
    seed: int,
    mode: str,
    n_quantiles: int = 5,
) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    observed_mask = np.asarray(observed_mask, dtype=bool)

    if data.shape != observed_mask.shape:
        raise ValueError("data and observed_mask have different shapes.")
    if not (0.0 < float(val_rate) < 1.0):
        raise ValueError(f"val_rate must be in (0, 1). Got {val_rate}")

    rng = np.random.default_rng(int(seed))
    val_mask = np.zeros_like(observed_mask, dtype=bool)

    if mode == "random":
        coordinates = np.argwhere(observed_mask & np.isfinite(data))
        if len(coordinates) == 0:
            raise ValueError("No observed cells are available for GAIN validation masking.")

        rng.shuffle(coordinates)
        n_val = int(np.ceil(len(coordinates) * float(val_rate)))
        selected = coordinates[:n_val]
        val_mask[selected[:, 0], selected[:, 1]] = True

    elif mode == "stratified_quantile":
        for j in range(data.shape[1]):
            rows = np.where(observed_mask[:, j] & np.isfinite(data[:, j]))[0]
            if len(rows) == 0:
                continue

            values = data[rows, j]
            q = min(int(n_quantiles), int(len(np.unique(values))))

            if q <= 1:
                labels = np.zeros(len(rows), dtype=int)
            else:
                labels = np.asarray(
                    pd.qcut(values, q=q, labels=False, duplicates="drop")
                )

            for label in np.unique(labels):
                label_rows = rows[labels == label]
                if len(label_rows) == 0:
                    continue

                n_val = min(
                    len(label_rows),
                    max(1, int(np.ceil(len(label_rows) * float(val_rate)))),
                )
                selected = rng.choice(label_rows, size=n_val, replace=False)
                val_mask[selected, j] = True
    else:
        raise ValueError("validation_mode must be 'random' or 'stratified_quantile'.")

    if int(val_mask.sum()) == 0:
        raise ValueError("GAIN internal validation mask has zero cells.")

    return val_mask


def infer_output_activation_and_noise(
    *,
    data: np.ndarray,
    observed_mask: np.ndarray,
    output_activation: str,
    noise_type: str,
    noise_scale: Optional[float],
) -> Tuple[Optional[str], str, float]:
    values = data[observed_mask & np.isfinite(data)]
    if values.size == 0:
        raise ValueError("No finite observed protein values were found.")

    data_min = float(values.min())
    data_max = float(values.max())
    is_unit_interval = data_min >= -1e-6 and data_max <= 1.0 + 1e-6

    output_activation = str(output_activation).lower()
    if output_activation == "auto":
        resolved_activation: Optional[str] = "sigmoid" if is_unit_interval else None
    elif output_activation == "sigmoid":
        resolved_activation = "sigmoid"
    elif output_activation in {"none", "linear"}:
        resolved_activation = None
    else:
        raise ValueError(
            "output_activation must be one of: 'auto', 'sigmoid', 'none', 'linear'."
        )

    noise_type = str(noise_type).lower()
    if noise_type == "auto":
        resolved_noise = "uniform" if is_unit_interval else "normal"
    elif noise_type in {"uniform", "normal"}:
        resolved_noise = noise_type
    else:
        raise ValueError("noise_type must be one of: 'auto', 'uniform', 'normal'.")

    if noise_scale is None:
        resolved_scale = 1.0
    else:
        resolved_scale = float(noise_scale)

    if not np.isfinite(resolved_scale) or resolved_scale <= 0:
        raise ValueError(f"noise_scale must be finite and positive. Got {resolved_scale}")

    return resolved_activation, resolved_noise, resolved_scale


def sample_training_noise(
    reference_tensor: torch.Tensor,
    *,
    noise_type: str,
    noise_scale: float,
) -> torch.Tensor:
    if noise_type == "uniform":
        return torch.rand_like(reference_tensor) * float(noise_scale)
    if noise_type == "normal":
        return torch.randn_like(reference_tensor) * float(noise_scale)
    raise ValueError(f"Unsupported noise_type: {noise_type}")


def _numpy_noise(
    rng: np.random.Generator,
    shape: Tuple[int, int],
    *,
    noise_type: str,
    noise_scale: float,
) -> np.ndarray:
    if noise_type == "uniform":
        return rng.uniform(0.0, float(noise_scale), size=shape).astype(np.float32)
    if noise_type == "normal":
        return rng.normal(0.0, float(noise_scale), size=shape).astype(np.float32)
    raise ValueError(f"Unsupported noise_type: {noise_type}")


# =============================================================================
# GAIN architecture
# =============================================================================
class Generator(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_neurons: Sequence[int],
        dropout_rate: float,
        output_activation: Optional[str],
    ):
        super().__init__()

        layers: List[nn.Module] = []
        in_features = int(dim) * 2

        for width in n_neurons:
            width = int(width)
            layers.extend(
                [
                    nn.Linear(in_features, width),
                    nn.LayerNorm(width),
                    nn.ReLU(),
                    nn.Dropout(float(dropout_rate)),
                ]
            )
            in_features = width

        layers.append(nn.Linear(in_features, int(dim)))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([x, mask], dim=1))


class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_neurons: Sequence[int],
        dropout_rate: float,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        in_features = int(dim) * 2

        for width in n_neurons:
            width = int(width)
            layers.extend(
                [
                    nn.Linear(in_features, width),
                    nn.LayerNorm(width),
                    nn.ReLU(),
                    nn.Dropout(float(dropout_rate)),
                ]
            )
            in_features = width

        layers.append(nn.Linear(in_features, int(dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, completed: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([completed, hint], dim=1))


def sample_hint(
    mask: torch.Tensor,
    *,
    hint_rate: float,
    hint_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hint_rate = float(hint_rate)
    if not (0.0 < hint_rate < 1.0):
        raise ValueError(f"hint_rate must be in (0, 1). Got {hint_rate}")

    hint_mode = str(hint_mode).lower()

    if hint_mode == "bernoulli":
        revealed = (torch.rand_like(mask) < hint_rate).float()
    elif hint_mode == "paper":
        revealed = torch.ones_like(mask)
        rows = torch.arange(mask.size(0), device=mask.device)
        columns = torch.randint(0, mask.size(1), (mask.size(0),), device=mask.device)
        revealed[rows, columns] = 0.0
    else:
        raise ValueError("hint_mode must be 'bernoulli' or 'paper'.")

    hint = revealed * mask + 0.5 * (1.0 - revealed)
    return revealed, hint


# =============================================================================
# Prediction and training
# =============================================================================
def gain_predict_mean(
    *,
    generator_model: Generator,
    x_input: np.ndarray,
    mask_input: np.ndarray,
    device: torch.device,
    noise_type: str,
    noise_scale: float,
    mc_samples: int,
    seed: int,
) -> np.ndarray:
    mc_samples = int(mc_samples)
    if mc_samples < 1:
        raise ValueError(f"mc_samples must be >= 1. Got {mc_samples}")

    x_input = np.asarray(x_input, dtype=np.float32)
    mask_input = np.asarray(mask_input, dtype=bool)
    if x_input.shape != mask_input.shape:
        raise ValueError("x_input and mask_input have different shapes.")

    generator_model.eval()
    x_tensor = torch.tensor(x_input, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask_input.astype(np.float32), device=device)

    rng = np.random.default_rng(int(seed))
    prediction_sum = np.zeros_like(x_input, dtype=np.float64)

    with torch.no_grad():
        for _ in range(mc_samples):
            noise = _numpy_noise(
                rng,
                x_input.shape,
                noise_type=noise_type,
                noise_scale=noise_scale,
            )
            noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)
            generator_input = x_tensor * mask_tensor + noise_tensor * (1.0 - mask_tensor)
            generated = generator_model(generator_input, mask_tensor)
            completed = x_tensor * mask_tensor + generated * (1.0 - mask_tensor)
            prediction_sum += completed.cpu().numpy().astype(np.float64)

    prediction = (prediction_sum / float(mc_samples)).astype(np.float32)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("GAIN prediction contains non-finite values.")

    return prediction


def train_gain_once(
    *,
    data_for_training: np.ndarray,
    base_mask: np.ndarray,
    reference_values: np.ndarray,
    validation_mask: Optional[np.ndarray],
    n_neurons: Sequence[int],
    batch_size: int,
    learning_rate: float,
    alpha: float,
    hint_rate: float,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    output_activation: Optional[str],
    noise_type: str,
    noise_scale: float,
    dropout_rate: float,
    weight_decay: float,
    grad_clip: float,
    pseudo_rate: float,
    pseudo_beta: float,
    hint_mode: str,
    mc_samples_validation: int,
    use_early_stopping: bool,
    verbose: bool,
) -> Tuple[Generator, float, int, Dict[str, Any]]:
    set_all_seeds(seed)

    data_for_training = np.asarray(data_for_training, dtype=np.float32)
    base_mask = np.asarray(base_mask, dtype=bool)
    reference_values = np.asarray(reference_values, dtype=np.float32)

    if data_for_training.shape != base_mask.shape:
        raise ValueError("data_for_training and base_mask differ in shape.")
    if data_for_training.shape != reference_values.shape:
        raise ValueError("data_for_training and reference_values differ in shape.")
    if not np.isfinite(data_for_training).all():
        raise ValueError("data_for_training contains non-finite values.")
    if int(base_mask.sum()) == 0:
        raise ValueError("base_mask has zero observed training cells.")

    batch_size = int(batch_size)
    max_epochs = int(max_epochs)
    patience = int(patience)
    learning_rate = float(learning_rate)
    alpha = float(alpha)
    pseudo_rate = float(pseudo_rate)
    pseudo_beta = float(pseudo_beta)
    dropout_rate = float(dropout_rate)
    weight_decay = float(weight_decay)
    grad_clip = float(grad_clip)

    if batch_size < 2:
        raise ValueError("GAIN batch_size must be at least 2.")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must both be positive.")
    if learning_rate <= 0 or alpha < 0 or pseudo_beta < 0:
        raise ValueError("learning_rate must be positive; alpha and pseudo_beta non-negative.")
    if not (0.0 <= pseudo_rate < 1.0):
        raise ValueError(f"pseudo_rate must be in [0, 1). Got {pseudo_rate}")
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1). Got {dropout_rate}")
    if weight_decay < 0 or grad_clip < 0:
        raise ValueError("weight_decay and grad_clip must be non-negative.")

    dimension = data_for_training.shape[1]
    generator_model = Generator(
        dim=dimension,
        n_neurons=n_neurons,
        dropout_rate=dropout_rate,
        output_activation=output_activation,
    ).to(device)
    discriminator_model = Discriminator(
        dim=dimension,
        n_neurons=n_neurons,
        dropout_rate=dropout_rate,
    ).to(device)

    generator_optimizer = optim.AdamW(
        generator_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    discriminator_optimizer = optim.AdamW(
        discriminator_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    x_tensor = torch.tensor(data_for_training, dtype=torch.float32)
    mask_tensor = torch.tensor(base_mask.astype(np.float32), dtype=torch.float32)
    dataset = TensorDataset(x_tensor, mask_tensor)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(seed))
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=False,
    )

    best_validation_rmse = float("inf")
    best_epoch = 0
    best_generator_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0

    history: Dict[str, Any] = {
        "epoch": [],
        "discriminator_loss": [],
        "generator_loss": [],
        "generator_adversarial": [],
        "generator_mse_visible": [],
        "generator_mse_pseudo": [],
        "validation_rmse": [],
    }

    fixed_validation_prediction_seed = int(seed) + 91_001

    for epoch in range(max_epochs):
        generator_model.train()
        discriminator_model.train()

        d_total = 0.0
        g_total = 0.0
        adv_total = 0.0
        visible_total = 0.0
        pseudo_total = 0.0
        n_batches = 0

        for x_batch, mask_batch in train_loader:
            if x_batch.size(0) < 2:
                continue

            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)

            if pseudo_rate > 0:
                pseudo_hidden = (
                    (torch.rand_like(mask_batch) < pseudo_rate).float() * mask_batch
                )
            else:
                pseudo_hidden = torch.zeros_like(mask_batch)

            input_mask = mask_batch * (1.0 - pseudo_hidden)
            noise = sample_training_noise(
                x_batch,
                noise_type=noise_type,
                noise_scale=noise_scale,
            )
            noisy_input = x_batch * input_mask + noise * (1.0 - input_mask)
            revealed_hint, hint = sample_hint(
                input_mask,
                hint_rate=hint_rate,
                hint_mode=hint_mode,
            )

            # Discriminator update
            with torch.no_grad():
                generated_detached = generator_model(noisy_input, input_mask)
                completed_detached = (
                    x_batch * input_mask
                    + generated_detached * (1.0 - input_mask)
                )

            discriminator_logits = discriminator_model(completed_detached, hint)
            d_bce = F.binary_cross_entropy_with_logits(
                discriminator_logits,
                input_mask,
                reduction="none",
            )
            d_weight = 1.0 - revealed_hint
            d_loss = torch.sum(d_bce * d_weight) / torch.clamp(
                torch.sum(d_weight), min=1.0
            )

            discriminator_optimizer.zero_grad()
            d_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    discriminator_model.parameters(), max_norm=grad_clip
                )
            discriminator_optimizer.step()

            # Generator update
            for parameter in discriminator_model.parameters():
                parameter.requires_grad_(False)

            generated = generator_model(noisy_input, input_mask)
            completed = x_batch * input_mask + generated * (1.0 - input_mask)
            discriminator_logits_for_generator = discriminator_model(completed, hint)

            adversarial_bce = F.binary_cross_entropy_with_logits(
                discriminator_logits_for_generator,
                torch.ones_like(discriminator_logits_for_generator),
                reduction="none",
            )
            adversarial_weight = (1.0 - revealed_hint) * (1.0 - input_mask)
            adversarial_loss = torch.sum(
                adversarial_bce * adversarial_weight
            ) / torch.clamp(torch.sum(adversarial_weight), min=1.0)

            visible_mse = torch.sum(
                ((x_batch - generated) ** 2) * input_mask
            ) / torch.clamp(torch.sum(input_mask), min=1.0)

            pseudo_mse = torch.sum(
                ((x_batch - generated) ** 2) * pseudo_hidden
            ) / torch.clamp(torch.sum(pseudo_hidden), min=1.0)

            generator_loss = (
                adversarial_loss + alpha * visible_mse + pseudo_beta * pseudo_mse
            )

            generator_optimizer.zero_grad()
            generator_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    generator_model.parameters(), max_norm=grad_clip
                )
            generator_optimizer.step()

            for parameter in discriminator_model.parameters():
                parameter.requires_grad_(True)

            d_total += float(d_loss.item())
            g_total += float(generator_loss.item())
            adv_total += float(adversarial_loss.item())
            visible_total += float(visible_mse.item())
            pseudo_total += float(pseudo_mse.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("GAIN completed an epoch without any valid training batches.")

        d_mean = d_total / n_batches
        g_mean = g_total / n_batches
        adv_mean = adv_total / n_batches
        visible_mean = visible_total / n_batches
        pseudo_mean = pseudo_total / n_batches

        if use_early_stopping:
            if validation_mask is None or int(np.asarray(validation_mask).sum()) == 0:
                raise ValueError("Early stopping requested without a non-empty validation mask.")

            validation_prediction = gain_predict_mean(
                generator_model=generator_model,
                x_input=data_for_training,
                mask_input=base_mask,
                device=device,
                noise_type=noise_type,
                noise_scale=noise_scale,
                mc_samples=mc_samples_validation,
                seed=fixed_validation_prediction_seed,
            )
            validation_rmse = rmse_on_mask(
                reference_values,
                validation_prediction,
                validation_mask,
            )
        else:
            validation_rmse = float("nan")

        history["epoch"].append(epoch + 1)
        history["discriminator_loss"].append(d_mean)
        history["generator_loss"].append(g_mean)
        history["generator_adversarial"].append(adv_mean)
        history["generator_mse_visible"].append(visible_mean)
        history["generator_mse_pseudo"].append(pseudo_mean)
        history["validation_rmse"].append(validation_rmse)

        if use_early_stopping:
            is_best = np.isfinite(validation_rmse) and validation_rmse < best_validation_rmse
            if is_best:
                best_validation_rmse = float(validation_rmse)
                best_epoch = epoch + 1
                best_generator_state = copy.deepcopy(generator_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or is_best):
                marker = " *new best*" if is_best else ""
                print(
                    f"Epoch {epoch + 1:04d}/{max_epochs:04d} | "
                    f"D={d_mean:.5f} | G={g_mean:.5f} | "
                    f"adv={adv_mean:.5f} | visible={visible_mean:.5f} | "
                    f"pseudo={pseudo_mean:.5f} | val_rmse={validation_rmse:.5f}{marker}"
                )

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}.")
                break
        elif verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == max_epochs):
            print(
                f"Refit epoch {epoch + 1:04d}/{max_epochs:04d} | "
                f"D={d_mean:.5f} | G={g_mean:.5f} | "
                f"adv={adv_mean:.5f} | visible={visible_mean:.5f} | "
                f"pseudo={pseudo_mean:.5f}"
            )

    if use_early_stopping:
        if best_generator_state is None or best_epoch < 1:
            raise RuntimeError("GAIN did not obtain a finite internal validation result.")
        generator_model.load_state_dict(best_generator_state)
    else:
        best_epoch = max_epochs
        best_validation_rmse = float("nan")

    return generator_model, best_validation_rmse, best_epoch, history


# =============================================================================
# Main imputer
# =============================================================================
def GAIN_imputer(
    *,
    df_with_na: pd.DataFrame,
    original_df: pd.DataFrame,
    n_neurons: Sequence[int],
    batch_size: int,
    lr: float,
    alpha: float,
    hint_rate: float,
    epochs: int,
    patience: int,
    seed: int,
    rmse_path: Path,
    imputed_path: Path,
    val_rate: float,
    validation_mode: str,
    refit_final: bool,
    mc_samples_validation: int,
    mc_samples_final: int,
    output_activation: str,
    noise_type: str,
    noise_scale: Optional[float],
    dropout_rate: float,
    weight_decay: float,
    grad_clip: float,
    pseudo_rate: float,
    pseudo_beta: float,
    hint_mode: str,
    fraction: float,
    missingness: str,
    verbose: bool,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    device = get_device()

    n_neurons = _parse_n_neurons(n_neurons)
    batch_size = int(batch_size)
    lr = float(lr)
    alpha = float(alpha)
    hint_rate = float(hint_rate)
    epochs = int(epochs)
    patience = int(patience)
    seed = int(seed)
    val_rate = float(val_rate)
    mc_samples_validation = int(mc_samples_validation)
    mc_samples_final = int(mc_samples_final)
    dropout_rate = float(dropout_rate)
    weight_decay = float(weight_decay)
    grad_clip = float(grad_clip)
    pseudo_rate = float(pseudo_rate)
    pseudo_beta = float(pseudo_beta)
    fraction = float(fraction)
    missingness = str(missingness).upper()

    if not (0.0 <= pseudo_rate < 1.0):
        raise ValueError(f"pseudo_rate must be in [0, 1). Got {pseudo_rate}")
    if pseudo_beta < 0:
        raise ValueError(f"pseudo_beta must be non-negative. Got {pseudo_beta}")
    if mc_samples_validation < 1 or mc_samples_final < 1:
        raise ValueError("mc_samples_validation and mc_samples_final must be positive.")

    corrupted_numeric, reference_numeric = _validate_input_frames(df_with_na, original_df)
    metadata = df_with_na.iloc[:, :2].copy()

    data = corrupted_numeric.to_numpy(dtype=np.float32)
    reference_values = reference_numeric.to_numpy(dtype=np.float32)
    missing_mask = np.isnan(data)
    observed_mask = ~missing_mask

    n_missing_cells = int(missing_mask.sum())
    if n_missing_cells == 0:
        raise ValueError("No missing protein cells were found in the corrupted GAIN input.")

    max_observed_difference = float(
        np.max(np.abs(data[observed_mask] - reference_values[observed_mask]))
    )
    if max_observed_difference > 1e-6:
        raise ValueError(
            "Observed entries in corrupted and reference data do not match. "
            f"Maximum absolute difference: {max_observed_difference}"
        )

    internal_validation_seed = stable_combo_seed(
        seed, fraction, missingness, "gain_internal_validation"
    )
    validation_mask = make_validation_mask(
        data=data,
        observed_mask=observed_mask,
        val_rate=val_rate,
        seed=internal_validation_seed,
        mode=validation_mode,
    )
    validation_mask &= observed_mask
    training_mask = observed_mask & ~validation_mask

    if int(training_mask.sum()) == 0:
        raise ValueError("GAIN training mask has zero cells.")

    training_values = np.where(training_mask, data, 0.0).astype(np.float32)

    (
        resolved_output_activation,
        resolved_noise_type,
        resolved_noise_scale,
    ) = infer_output_activation_and_noise(
        data=data,
        observed_mask=observed_mask,
        output_activation=output_activation,
        noise_type=noise_type,
        noise_scale=noise_scale,
    )

    config: Dict[str, Any] = {
        "batch_size": batch_size,
        "lr": lr,
        "alpha": alpha,
        "hint_rate": hint_rate,
        "n_neurons": n_neurons,
        "epochs": epochs,
        "patience": patience,
        "val_rate": val_rate,
        "validation_mode": validation_mode,
        "refit_final": bool(refit_final),
        "mc_samples_validation": mc_samples_validation,
        "mc_samples_final": mc_samples_final,
        "output_activation_requested": output_activation,
        "output_activation_resolved": (
            resolved_output_activation if resolved_output_activation is not None else "none"
        ),
        "noise_type_requested": noise_type,
        "noise_type_resolved": resolved_noise_type,
        "noise_scale": resolved_noise_scale,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "pseudo_rate": pseudo_rate,
        "pseudo_beta": pseudo_beta,
        "hint_mode": hint_mode,
        "fraction": fraction,
        "missingness": missingness,
    }

    print("\n" + "=" * 80)
    print("GAIN training configuration")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print(f"device: {device}")
    print(f"missing cells: {n_missing_cells}")
    print(f"internal validation cells: {int(validation_mask.sum())}")
    print("=" * 80)

    validation_training_seed = stable_combo_seed(
        seed, fraction, missingness, "gain_validation_training"
    )
    generator_validation, best_validation_rmse, best_epoch, history = train_gain_once(
        data_for_training=training_values,
        base_mask=training_mask,
        reference_values=reference_values,
        validation_mask=validation_mask,
        n_neurons=n_neurons,
        batch_size=batch_size,
        learning_rate=lr,
        alpha=alpha,
        hint_rate=hint_rate,
        max_epochs=epochs,
        patience=patience,
        seed=validation_training_seed,
        device=device,
        output_activation=resolved_output_activation,
        noise_type=resolved_noise_type,
        noise_scale=resolved_noise_scale,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        pseudo_rate=pseudo_rate,
        pseudo_beta=pseudo_beta,
        hint_mode=hint_mode,
        mc_samples_validation=mc_samples_validation,
        use_early_stopping=True,
        verbose=verbose,
    )

    all_observed_values = np.where(observed_mask, data, 0.0).astype(np.float32)

    if refit_final:
        refit_seed = stable_combo_seed(seed, fraction, missingness, "gain_final_refit")
        generator_final, _, _, refit_history = train_gain_once(
            data_for_training=all_observed_values,
            base_mask=observed_mask,
            reference_values=reference_values,
            validation_mask=None,
            n_neurons=n_neurons,
            batch_size=batch_size,
            learning_rate=lr,
            alpha=alpha,
            hint_rate=hint_rate,
            max_epochs=best_epoch,
            patience=patience,
            seed=refit_seed,
            device=device,
            output_activation=resolved_output_activation,
            noise_type=resolved_noise_type,
            noise_scale=resolved_noise_scale,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            pseudo_rate=pseudo_rate,
            pseudo_beta=pseudo_beta,
            hint_mode=hint_mode,
            mc_samples_validation=mc_samples_validation,
            use_early_stopping=False,
            verbose=verbose,
        )
    else:
        generator_final = generator_validation
        refit_history = None

    prediction_seed = stable_combo_seed(
        seed, fraction, missingness, "gain_final_prediction"
    )
    generated_values = gain_predict_mean(
        generator_model=generator_final,
        x_input=all_observed_values,
        mask_input=observed_mask,
        device=device,
        noise_type=resolved_noise_type,
        noise_scale=resolved_noise_scale,
        mc_samples=mc_samples_final,
        seed=prediction_seed,
    )

    completed_numeric = fill_only_missing_cells(
        corrupted_numeric,
        generated_values,
        missing_mask,
    )
    completed_values = completed_numeric.to_numpy(dtype=np.float32)
    test_rmse = rmse_on_mask(reference_values, completed_values, missing_mask)

    imputed_df = pd.concat(
        [metadata.reset_index(drop=True), completed_numeric.reset_index(drop=True)],
        axis=1,
    )
    if list(imputed_df.columns[:2]) != ["eid", "label"]:
        raise RuntimeError("GAIN output does not preserve eid and label as the first columns.")

    rmse_path = Path(rmse_path)
    imputed_path = Path(imputed_path)
    rmse_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_path.parent.mkdir(parents=True, exist_ok=True)

    write_rmse_csv(
        rmse_path=rmse_path,
        test_rmse=test_rmse,
        internal_best_val_rmse=best_validation_rmse,
        best_epoch=best_epoch,
        seed=seed,
        n_missing_cells=n_missing_cells,
        n_internal_val_cells=int(validation_mask.sum()),
        max_observed_difference=max_observed_difference,
        config=config,
    )
    imputed_df.to_csv(imputed_path, index=False)

    print("\n" + "=" * 80)
    print(f"GAIN final RMSE on benchmark-missing cells: {test_rmse:.6f}")
    print(f"Best internal validation RMSE: {best_validation_rmse:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"RMSE file: {rmse_path}")
    print(f"Imputed file: {imputed_path}")
    print("=" * 80)

    return {
        "test_rmse": test_rmse,
        "internal_best_val_rmse": best_validation_rmse,
        "best_epoch": best_epoch,
        "config": config,
        "history": history,
        "refit_history": refit_history,
        "rmse_path": str(rmse_path),
        "imputed_path": str(imputed_path),
    }


# =============================================================================
# Strict file-handling wrapper
# =============================================================================
def gain(
    *,
    columns_num,
    fractions,
    missingness_types,
    n_neurons,
    batch_size,
    lr,
    alpha,
    hint_rate,
    epochs,
    patience,
    seed,
    val_rate,
    validation_mode,
    refit_final,
    mc_samples_validation,
    mc_samples_final,
    output_activation,
    noise_type,
    noise_scale,
    dropout_rate,
    weight_decay,
    grad_clip,
    pseudo_rate,
    pseudo_beta,
    hint_mode,
    verbose,
):
    main_directory = Path(experiments_dir(columns_num))
    if not main_directory.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist:\n{main_directory}\n"
            "Run preprocessing before GAIN."
        )

    n_neurons = _parse_n_neurons(n_neurons)
    seed = int(seed)

    params_for_id = {
        "batch_size": batch_size,
        "lr": lr,
        "n_neurons": n_neurons,
        "alpha": alpha,
        "hint_rate": hint_rate,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "pseudo_rate": pseudo_rate,
        "pseudo_beta": pseudo_beta,
        "hint_mode": hint_mode,
        "mc_samples_final": mc_samples_final,
    }
    parameter_id = gain_param_id(
        params_for_id,
        epochs=int(epochs),
        patience=int(patience),
    )

    results: List[Dict[str, Any]] = []

    for fraction in fractions:
        fraction = float(fraction)

        for missingness in missingness_types:
            missingness = str(missingness).upper()
            folder = main_directory / f"fraction_{fraction}_missingness_{missingness}"

            if not folder.exists():
                raise FileNotFoundError(
                    f"Required fraction/missingness directory does not exist:\n{folder}"
                )

            corrupted_path = (
                folder
                / f"Corrupted_data_fraction_{fraction}_missingness_{missingness}_col_{columns_num}_seed_{seed}.csv"
            )
            reference_path = folder / f"normalized_data_col_{columns_num}.csv"

            if not corrupted_path.exists():
                raise FileNotFoundError(
                    f"GAIN could not find the exact seeded corrupted file:\n{corrupted_path}"
                )
            if not reference_path.exists():
                raise FileNotFoundError(
                    f"GAIN could not find the reference file:\n{reference_path}"
                )

            rmse_path = (
                folder
                / f"RMSE_GAIN_col_{columns_num}_params_{parameter_id}_seed_{seed}.csv"
            )
            imputed_path = (
                folder
                / f"imputed_GAIN_col_{columns_num}_params_{parameter_id}_seed_{seed}.csv"
            )

            print("\n" + "#" * 90)
            print(
                f"Running GAIN | missingness={missingness} | "
                f"fraction={fraction} | seed={seed}"
            )
            print("#" * 90)

            result = GAIN_imputer(
                df_with_na=pd.read_csv(corrupted_path),
                original_df=pd.read_csv(reference_path),
                n_neurons=n_neurons,
                batch_size=int(batch_size),
                lr=float(lr),
                alpha=float(alpha),
                hint_rate=float(hint_rate),
                epochs=int(epochs),
                patience=int(patience),
                seed=seed,
                rmse_path=rmse_path,
                imputed_path=imputed_path,
                val_rate=float(val_rate),
                validation_mode=str(validation_mode),
                refit_final=bool(refit_final),
                mc_samples_validation=int(mc_samples_validation),
                mc_samples_final=int(mc_samples_final),
                output_activation=str(output_activation),
                noise_type=str(noise_type),
                noise_scale=None if noise_scale is None else float(noise_scale),
                dropout_rate=float(dropout_rate),
                weight_decay=float(weight_decay),
                grad_clip=float(grad_clip),
                pseudo_rate=float(pseudo_rate),
                pseudo_beta=float(pseudo_beta),
                hint_mode=str(hint_mode),
                fraction=fraction,
                missingness=missingness,
                verbose=bool(verbose),
            )
            results.append(result)

    return results
