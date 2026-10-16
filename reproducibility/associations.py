"""Adjusted logistic association scans and the paper's preservation metrics."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


def covariate_design(metadata, columns):
    """Build once for all reference/imputed scans in an outcome cohort."""
    cov = metadata[list(columns)].copy()
    for c in cov:
        if pd.api.types.is_numeric_dtype(cov[c]):
            v = cov[c].replace([np.inf, -np.inf], np.nan)
            if v.notna().sum() == 0:
                raise ValueError(f"Covariate {c} is entirely missing.")
            cov[c] = v.fillna(v.median())
        else:
            mode = cov[c].mode(dropna=True)
            if mode.empty:
                raise ValueError(f"Covariate {c} is entirely missing.")
            cov[c] = cov[c].fillna(mode.iloc[0])
    cov = pd.get_dummies(cov, drop_first=True).astype(float)
    cov = cov.loc[:, cov.nunique() > 1]
    design = sm.add_constant(cov, has_constant="add")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The covariate design is rank deficient.")
    return design


def scan(x, metadata, outcome, design):
    if not x.index.equals(metadata.index) or not x.index.equals(design.index):
        raise ValueError("Association inputs must have the same sample order.")
    y = metadata[outcome].astype(float)
    if y.isna().any() or set(y.unique()) != {0, 1} or len(y) < 50:
        raise ValueError("Association scan needs >=50 patients and an observed binary 0/1 outcome.")
    rows = []
    for protein in x:
        v = x[protein].astype(float)
        result = {"protein": protein, "n": len(v), "beta": np.nan, "se": np.nan, "z": np.nan, "p": np.nan, "status": "ok"}
        try:
            if not np.isfinite(v).all() or v.std(ddof=1) <= 1e-12:
                raise ValueError("Nonfinite or constant protein.")
            d = design.copy()
            d["protein_value"] = (v - v.mean()) / v.std(ddof=1)
            if np.linalg.matrix_rank(d) != d.shape[1]:
                raise ValueError("Rank-deficient protein model.")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = sm.GLM(y, d, family=sm.families.Binomial()).fit(maxiter=100)
            if not fitted.converged or any(issubclass(w.category, PerfectSeparationWarning) for w in caught):
                raise ValueError("Separation or nonconvergence.")
            beta, se, p = (float(v["protein_value"]) for v in [fitted.params, fitted.bse, fitted.pvalues])
            if not np.isfinite([beta, se, p]).all() or se <= 0:
                raise ValueError("Invalid coefficient inference.")
            result.update(beta=beta, se=se, z=beta / se, p=p)
        except Exception as exc:
            result["status"] = type(exc).__name__ + ": " + str(exc)[:150]
        rows.append(result)
    out = pd.DataFrame(rows).set_index("protein")
    valid = out.p.notna()
    out["q"] = np.nan
    if valid.any():
        out.loc[valid, "q"] = multipletests(out.loc[valid, "p"], method="fdr_bh")[1]
    return out


def preservation(reference, imputed, top_k=(20, 50), threshold=0.05):
    joint = reference.join(imputed, how="inner", lsuffix="_ref", rsuffix="_imp")
    joint = joint.replace([np.inf, -np.inf], np.nan).dropna(subset=["beta_ref", "beta_imp", "z_ref", "z_imp"])
    def corr(a, b):
        return float(spearmanr(a, b).statistic) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else np.nan
    result = {"n_evaluable_proteins": len(joint),
              "n_failed_ref": int(reference.p.isna().sum()), "n_failed_imp": int(imputed.p.isna().sum()),
              "z_spearman": corr(joint.z_ref, joint.z_imp),
              "beta_spearman": corr(joint.beta_ref, joint.beta_imp),
              "beta_rmse": float(np.sqrt(np.mean((joint.beta_ref - joint.beta_imp) ** 2))) if len(joint) else np.nan}
    r_sig = set(reference.index[reference.q < threshold])
    i_sig = set(imputed.index[imputed.q < threshold])
    result.update(n_ref_fdr=len(r_sig), n_imp_fdr=len(i_sig),
                  fdr_jaccard=len(r_sig & i_sig) / len(r_sig | i_sig) if r_sig | i_sig else np.nan,
                  fdr_retention=len(r_sig & i_sig) / len(r_sig) if r_sig else np.nan)
    for k in top_k:
        r = set(reference.z.dropna().abs().sort_values(ascending=False, kind="stable").head(k).index)
        i = set(imputed.z.dropna().abs().sort_values(ascending=False, kind="stable").head(k).index)
        # A top-50 score is undefined for a 20-feature demonstration, not divided by 50.
        result[f"top{k}_overlap"] = len(r & i) / k if len(r) == k and len(i) == k else np.nan
        members = joint.loc[joint.index.isin(r)]
        result[f"sign_concordance_top{k}"] = float((np.sign(members.beta_ref) == np.sign(members.beta_imp)).mean()) if len(members) and len(r) == k else np.nan
    return result
