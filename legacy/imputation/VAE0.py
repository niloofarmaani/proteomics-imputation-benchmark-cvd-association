import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset
from typing import List
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

# Make sure we can import config_paths from code/
CODE_DIR = Path(__file__).resolve().parents[1]  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config_paths import experiments_dir


# Constants for VAE
PI = torch.tensor(torch.pi)
log_of_2 = torch.log(torch.tensor(2.))
leaky_relu_default = nn.LeakyReLU(0.1)

# VAE Class Definition
class VAE(nn.Module):
    def __init__(self, n_features: int, n_neurons: List[int], activation=leaky_relu_default, last_decoder_activation=None, dim_latent: int = 10):
        super().__init__()
        self.n_features, self.n_neurons = n_features, list(n_neurons)
        self.layers = [n_features, *self.n_neurons]
        self.dim_latent = dim_latent

        def build_layer(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat),
                    nn.Dropout(0.1),
                    nn.BatchNorm1d(out_feat),
                    activation]

        self.encoder = []
        for i in range(len(self.layers) - 1):
            in_feat, out_feat = self.layers[i:i + 2]
            self.encoder.extend(build_layer(in_feat=in_feat, out_feat=out_feat))
        self.encoder.append(nn.Linear(out_feat, dim_latent * 2))
        self.encoder = nn.Sequential(*self.encoder)

        self.layers_decoder = self.layers[::-1]
        self.decoder = build_layer(in_feat=self.dim_latent, out_feat=out_feat)
        i = -1
        for i in range(len(self.layers_decoder) - 2):
            in_feat, out_feat = self.layers_decoder[i:i + 2]
            self.decoder.extend(build_layer(in_feat=in_feat, out_feat=out_feat))
        in_feat, out_feat = self.layers_decoder[i + 1:i + 3]
        self.decoder.append(nn.Linear(in_feat, out_feat * 2))
        if last_decoder_activation is not None:
            self.decoder.append(last_decoder_activation)
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

# Loss Function
def compute_kld(z_mu, z_logvar):
    return 0.5 * (z_mu**2 + torch.exp(z_logvar) - 1 - z_logvar)

def gaussian_log_prob(z, mu, logvar):
    return -0.5 * (torch.log(2. * PI) + logvar + (z - mu)**2 / torch.exp(logvar))

def loss_fct(pred, y, mask, epoch, warmup_epochs, reduction='sum', results: List = None, freebits=0.05):
    x_mu, x_logvar, z_mu, z_logvar = pred
    batch = y
    masked_log_prob = gaussian_log_prob(batch[mask], x_mu[mask], x_logvar[mask])
    l_rec = -torch.sum(masked_log_prob)
    l_reg = torch.sum((F.relu(compute_kld(z_mu, z_logvar) - freebits * log_of_2) + freebits * log_of_2), 1)
    kl_weight = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
    if results is not None:
        results.append((l_rec.item(), torch.mean(l_reg).item()))
    return (l_rec / y.shape[0]) + (kl_weight * torch.mean(l_reg))

# Updated VAE Imputer Function
def VAE_imputer(df_with_na, original_df, missing_mask, batch_size, learning_rate, n_neurons, dim_latent, freebits,
                rmse_path, imputed_data_path, epochs, patience, warmup_epochs):
    eid_label_df = df_with_na.iloc[:, :2]
    df_with_na_numeric = df_with_na.iloc[:, 2:]
    original_df_numeric = original_df.iloc[:, 2:]
    missing_mask_numeric = missing_mask.iloc[:, :]

    original_data = original_df_numeric.values
    data = df_with_na_numeric.values
    missing_mask = missing_mask_numeric.values.astype(bool)
    mask = np.isnan(data)

    percentile_1 = np.nanpercentile(data, 1)
    data_imputed = np.where(np.isnan(data), percentile_1, data)
    data_normalized = data_imputed

    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    mask_tensor = torch.tensor(~mask, dtype=torch.bool)

    vae = VAE(n_features=df_with_na_numeric.shape[1], n_neurons=n_neurons, dim_latent=dim_latent)

    dataset = TensorDataset(data_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.002, step_size_up=50, mode='triangular', cycle_momentum=False)
    best_loss = float('inf')
    patience_counter = 0

    results = []
    for epoch in range(epochs):
        epoch_loss = 0
        batches = 0
        
        for data_batch, mask_batch in dataloader:
            optimizer.zero_grad()
            x_mu, x_logvar, z_mu, z_logvar = vae(data_batch)
            loss = loss_fct((x_mu, x_logvar, z_mu, z_logvar), data_batch, mask_batch, epoch, 
                            warmup_epochs=warmup_epochs, freebits=freebits, results=results)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_loss = epoch_loss / batches
        if (epoch + 1) % 20 == 0:
            l_rec, l_reg = results[-1]
            avg_rec_loss = l_rec / data_batch.shape[0]
            avg_kl = l_reg
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Rec Loss: {avg_rec_loss:.4f}, KL: {avg_kl:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    with torch.no_grad():
        x_mu, x_logvar, _, _ = vae(data_tensor)
        reconstructed = x_mu.numpy()

    df_imputed = pd.DataFrame(reconstructed, columns=df_with_na_numeric.columns)
    df_final = df_with_na_numeric.copy()
    mask_df = pd.DataFrame(mask, index=df_with_na_numeric.index, columns=df_with_na_numeric.columns)
    df_final = pd.DataFrame(np.where(mask_df, df_imputed, df_final), 
                            index=df_with_na_numeric.index, columns=df_with_na_numeric.columns)

    imputed_vae = pd.concat([eid_label_df, df_final], axis=1)

    masked_original = original_data[missing_mask]
    masked_final = df_final.values[missing_mask]
    rmse_vae = np.sqrt(np.mean((masked_original - masked_final) ** 2))

    print(f"RMSE for VAE Imputation: {rmse_vae:.4f}")
    pd.DataFrame({'rmse_vae': [rmse_vae]}).to_csv(rmse_path, index=False)
    imputed_vae.to_csv(imputed_data_path, index=False)

    return imputed_vae, rmse_vae

# File Handling Function for VAE
def vae(file_path, columns_num, fractions, missingness_types, batch_size, learning_rate, n_neurons, dim_latent, freebits,
        epochs=200, patience=50, warmup_epochs=100):
    """
    Preprocesses and handles files for VAE imputation experiments with fixed parameters.

    Parameters
    ----------
    file_path : str
        Path to the original CSV dataset.
    columns_num : str
        Columns specification (e.g., "all").
    fractions : list of float
        Fractions of data to be corrupted (e.g., [0.01, 0.05]).
    missingness_types : list of str
        Types of missingness to introduce (e.g., ["MCAR", "MAR"]).
    batch_size : int
        Batch size for training.
    learning_rate : float
        Learning rate for the optimizer.
    n_neurons : list of int
        List of neuron counts for VAE layers (e.g., [50, 30, 20]).
    dim_latent : int
        Latent dimension size.
    freebits : float
        Freebits parameter for KL divergence.
    epochs : int, optional
        Number of training epochs (default: 200).
    patience : int, optional
        Patience for early stopping (default: 50).
    warmup_epochs : int, optional
        Warmup epochs for KL divergence (default: 100).
    """

    main_directory = str(experiments_dir(columns_num))

    for fraction in fractions:
        for missingness in missingness_types:
            subfolder_name = f'fraction_{fraction}_missingness_{missingness}'
            full_path = os.path.join(main_directory, subfolder_name)
            os.makedirs(full_path, exist_ok=True)

            filename_corruption = (
                f'Corrupted_data_fraction_{fraction}_missingness_{missingness}'
                f'_col_{columns_num}.csv'
            )
            corrupted_data_path = os.path.join(full_path, filename_corruption)

            filename_mask = f'missing_mask_col_{columns_num}.csv'
            missing_mask_path = os.path.join(full_path, filename_mask)

            filename_normalized_data = f'normalized_data_col_{columns_num}.csv'
            normalized_data_path = os.path.join(full_path, filename_normalized_data)

            param_id = f"bs{batch_size}_lr{learning_rate}_nn{'-'.join(map(str, n_neurons))}_dl{dim_latent}_fb{freebits}"
            filename_rmse_path_vae = f'RMSE_VAE_col_{columns_num}_params_{param_id}.csv'
            rmse_path_vae = os.path.join(full_path, filename_rmse_path_vae)

            filename_imputed_path_vae = f'imputed_VAE_col_{columns_num}_params_{param_id}.csv'
            imputed_path_vae = os.path.join(full_path, filename_imputed_path_vae)

            if len(os.listdir(full_path)) > 0:
                df_with_na = pd.read_csv(corrupted_data_path)
                original_df = pd.read_csv(normalized_data_path)
                missing_mask = pd.read_csv(missing_mask_path)

                imputed_vae, rmse_vae = VAE_imputer(
                    df_with_na, original_df, missing_mask, 
                    batch_size=batch_size, 
                    learning_rate=learning_rate, 
                    n_neurons=n_neurons, 
                    dim_latent=dim_latent, 
                    freebits=freebits, 
                    rmse_path=rmse_path_vae, 
                    imputed_data_path=imputed_path_vae,
                    epochs=epochs,
                    patience=patience,
                    warmup_epochs=warmup_epochs
                )
            else:
                pass  # Silently skip if no files

