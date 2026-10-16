import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict
from itertools import product
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore")

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

# Loss Function
def compute_kld(z_mu, z_logvar):
    return 0.5 * (z_mu**2 + torch.exp(z_logvar) - 1 - z_logvar)

def gaussian_log_prob(z, mu, logvar):
    return -0.5 * (torch.log(2. * PI) + logvar + (z - mu)**2 / torch.exp(logvar))

def loss_fct(pred, y, mask, epoch, warmup_epochs=50, reduction='sum', results: List = None, freebits=0.05):
    x_mu, x_logvar, z_mu, z_logvar = pred
    batch = y
    masked_log_prob = gaussian_log_prob(batch[mask], x_mu[mask], x_logvar[mask])
    l_rec = -torch.sum(masked_log_prob)
    l_reg = torch.sum((F.relu(compute_kld(z_mu, z_logvar) - freebits * log_of_2) + freebits * log_of_2), 1)
    kl_weight = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
    if results is not None:
        results.append((l_rec.item(), torch.mean(l_reg).item()))
    return (l_rec / y.shape[0]) + (kl_weight * torch.mean(l_reg))

# VAE Imputer Function
def VAE_imputer(df_with_na, original_df, missing_mask, param_grid, rmse_path, imputed_data_path, 
                epochs=500, patience=100, warmup_epochs=50):
    # 1) Separate the first two columns (eid, label)
    eid_label_df = df_with_na.iloc[:, :2]
    df_with_na_numeric = df_with_na.iloc[:, 2:]
    original_df_numeric = original_df.iloc[:, 2:]
    missing_mask_numeric = missing_mask.iloc[:, 2:]

    # 2) Preprocess numeric data
    data = df_with_na_numeric.values
    percentile_1 = np.nanpercentile(data, 1)
    data_imputed = np.where(np.isnan(data), percentile_1, data)
    data_normalized = data_imputed
    mask = np.isnan(data)

    # 3) Convert to tensors
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    mask_tensor = torch.tensor(~mask, dtype=torch.bool)

    # 4) Grid Search Training
    results = []
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    for idx, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        print(f"\nTraining with Parameters {idx+1}/{len(param_combinations)}: {param_dict}")

        vae = VAE(n_features=df_with_na_numeric.shape[1], 
                  n_neurons=param_dict['n_neurons'], 
                  dim_latent=param_dict['dim_latent'])
        
        dataset = TensorDataset(data_tensor, mask_tensor)
        dataloader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=True)

        optimizer = optim.AdamW(vae.parameters(), lr=param_dict['learning_rate'], weight_decay=1e-5)
        scheduler = CyclicLR(optimizer, base_lr=param_dict['learning_rate']/20, 
                            max_lr=param_dict['learning_rate'], 
                            step_size_up=100, mode='triangular', cycle_momentum=False)
        
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_rec_loss = 0
            epoch_kl = 0
            batches = 0
            epoch_results = []
            
            for data_batch, mask_batch in dataloader:
                optimizer.zero_grad()
                x_mu, x_logvar, z_mu, z_logvar = vae(data_batch)
                loss = loss_fct((x_mu, x_logvar, z_mu, z_logvar), data_batch, mask_batch, epoch, 
                               warmup_epochs=warmup_epochs, freebits=param_dict['freebits'], results=epoch_results)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                l_rec, l_reg = epoch_results[-1]
                epoch_rec_loss += l_rec / data_batch.shape[0]
                epoch_kl += l_reg
                batches += 1
            
            avg_loss = epoch_loss / batches
            avg_rec_loss = epoch_rec_loss / batches
            avg_kl = epoch_kl / batches
            loss_history.append((avg_loss, avg_rec_loss, avg_kl))
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Rec Loss: {avg_rec_loss:.4f}, KL: {avg_kl:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = vae.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best Loss: {best_loss:.4f}")
                    break

        vae.load_state_dict(best_model_state)
        with torch.no_grad():
            x_mu, _, _, _ = vae(data_tensor)
            reconstructed = x_mu.numpy()
        df_imputed_numeric = pd.DataFrame(reconstructed, columns=df_with_na_numeric.columns)
        
        masked_original = original_df_numeric.values[missing_mask_numeric]
        masked_final = df_imputed_numeric.values[missing_mask_numeric]
        rmse = np.sqrt(np.mean((masked_original - masked_final) ** 2))

        result = {
            'params': param_dict,
            'loss_history': loss_history,
            'best_loss': best_loss,
            'rmse': rmse,
            'best_model_state': best_model_state
        }
        results.append(result)
        print(f"Completed: Best Loss = {best_loss:.4f}, RMSE = {rmse:.4f}")

    best_rmse_result = min(results, key=lambda x: x['rmse'])

    # 5) Reattach eid and label, calculate final RMSE
    vae.load_state_dict(best_rmse_result['best_model_state'])
    with torch.no_grad():
        x_mu, _, _, _ = vae(data_tensor)
        reconstructed = x_mu.numpy()
    df_imputed_numeric = pd.DataFrame(reconstructed, columns=df_with_na_numeric.columns)
    imputed_vae = pd.concat([eid_label_df, df_imputed_numeric], axis=1)

    # 6) Prepare mask for RMSE calculation (consistent with R)
    mask_subset = missing_mask_numeric.to_numpy()
    valid_indices = np.where(~mask_subset)  # Where data was not originally missing

    original_vals = original_df_numeric.values.flatten()
    imputed_vals = df_imputed_numeric.values.flatten()
    rmse_vae = np.sqrt(np.mean((original_vals[valid_indices] - imputed_vals[valid_indices]) ** 2))

    # Print and save RMSE
    print(f"RMSE for VAE Imputation: {rmse_vae:.4f}")
    pd.DataFrame({'rmse_vae': [rmse_vae]}).to_csv(rmse_path, index=False)

    # 7) Save imputed data
    imputed_vae.to_csv(imputed_data_path, index=False)
    print(f"Imputed data saved to {imputed_data_path}")

    return imputed_vae, rmse_vae, results, best_rmse_result

# Main Execution
args = sys.argv[1:]
if len(args) != 6:
    print("Usage: python script.py corrupted_data_path normalized_data_path missing_mask_path param_grid rmse_path imputed_data_path")
    sys.exit(1)

corrupted_data_path = args[0]
normalized_data_path = args[1]
missing_mask_path = args[2]
param_grid_str = args[3]  # Expecting a string like "32,64|0.001,0.0005|100-50-30,50-30-20|134,269|0.05,0.1"
rmse_path = args[4]
imputed_data_path = args[5]

print(f"Imputed data will be saved to: {imputed_data_path}")

# Parse param_grid from string
batch_sizes, lrs, neurons, dim_latents, freebits = param_grid_str.split('|')
param_grid = {
    'batch_size': [int(x) for x in batch_sizes.split(',')],
    'learning_rate': [float(x) for x in lrs.split(',')],
    'n_neurons': [[int(n) for n in x.split('-')] for x in neurons.split(',')],
    'dim_latent': [int(x) for x in dim_latents.split(',')],
    'freebits': [float(x) for x in freebits.split(',')]
}

# Read CSVs
df_with_na = pd.read_csv(corrupted_data_path)
original_df = pd.read_csv(normalized_data_path)
missing_mask = pd.read_csv(missing_mask_path)

# Call the function
imputed_vae, rmse_vae, results, best_rmse_result = VAE_imputer(
    df_with_na, original_df, missing_mask, param_grid, rmse_path, imputed_data_path
)
