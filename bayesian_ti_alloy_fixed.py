"""
Physics-Informed Bayesian Surrogate Maps of Titanium-Alloy Mechanical Property Landscapes
Using Manual MCMC Implementation on the DAX-Ti Database

FIXED VERSION: Added numerical stability checks and regularization

Author: Implementation based on methodology from Chatterjee (2025)
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.linalg import inv, cholesky, solve_triangular
from scipy.stats import wishart, invwishart
import warnings
warnings.filterwarnings('ignore')
import os
import json
from datetime import datetime
import zipfile

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Regularization constants for numerical stability
EPSILON = 1e-6
REG_LAMBDA = 1e-4

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"ti_alloy_bayesian_outputs_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/plots", exist_ok=True)
os.makedirs(f"{output_dir}/tables", exist_ok=True)
os.makedirs(f"{output_dir}/diagnostics", exist_ok=True)

print("="*80)
print("PHYSICS-INFORMED BAYESIAN MCMC FOR TI-ALLOY PROPERTY PREDICTION")
print("="*80)
print(f"\nOutput directory: {output_dir}\n")

# ============================================================================
# HELPER FUNCTIONS WITH NUMERICAL STABILITY
# ============================================================================

def safe_inv(A, reg=REG_LAMBDA):
    """Safely invert matrix with regularization"""
    try:
        # Add small regularization to diagonal
        A_reg = A + reg * np.eye(A.shape[0])
        return inv(A_reg)
    except:
        # Fallback to higher regularization
        A_reg = A + 10 * reg * np.eye(A.shape[0])
        return inv(A_reg)

def safe_cholesky(A, reg=REG_LAMBDA):
    """Safely compute Cholesky with regularization"""
    try:
        return cholesky(A, lower=True)
    except:
        # Add regularization until positive definite
        A_reg = A.copy()
        for i in range(10):
            try:
                A_reg = A + (10**i) * reg * np.eye(A.shape[0])
                return cholesky(A_reg, lower=True)
            except:
                continue
        # Last resort: use diagonal
        return np.diag(np.sqrt(np.abs(np.diag(A)) + reg))

def mvn_sample(mean, cov):
    """Sample from multivariate normal with numerical stability"""
    L = safe_cholesky(cov)
    z = np.random.randn(len(mean))
    return mean + L @ z

def matrix_normal_sample(M, U, V):
    """Sample from Matrix Normal with numerical stability"""
    G = np.random.randn(M.shape[0], M.shape[1])
    L_U = safe_cholesky(U)
    L_V = safe_cholesky(V)
    return M + L_U @ G @ L_V.T

def inv_gamma_sample(a, b):
    """Sample from Inverse-Gamma(a, b)"""
    return 1.0 / np.random.gamma(a, 1.0/b)

def inv_wishart_sample(df, scale):
    """Sample from Inverse-Wishart with stability checks"""
    try:
        return invwishart.rvs(df=df, scale=scale)
    except:
        # Fallback: regularize scale matrix
        K = scale.shape[0]
        scale_reg = scale + REG_LAMBDA * np.eye(K)
        return invwishart.rvs(df=df, scale=scale_reg)

def check_matrix_health(M, name="Matrix"):
    """Check if matrix has NaN or Inf values"""
    if np.any(np.isnan(M)) or np.any(np.isinf(M)):
        print(f"  WARNING: {name} contains NaN/Inf values!")
        return False
    return True

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("Step 1: Loading and preprocessing DAX-Ti database...")

# Load data
df = pd.read_csv('dax-ti-static.csv')
print(f"  - Loaded {len(df)} alloy entries")

# Select mechanical properties
property_names = ['YM', 'YS', 'UTS', 'DAR', 'HV']
property_errors = ['YM_err', 'YS_err', 'UTS_err', 'DAR_err', 'HV_err']
K = len(property_names)

# Extract composition features
def parse_composition(formula_str):
    """Extract element fractions from composition string"""
    elements = {}
    try:
        parts = str(formula_str).replace(' ', '').split()
        for part in parts:
            elem = ''.join([c for c in part if c.isalpha()])
            frac_str = ''.join([c for c in part if c.isdigit() or c == '.'])
            if elem and frac_str:
                frac = float(frac_str)
                if frac > 0:
                    elements[elem] = frac
    except:
        pass
    return elements

# Parse compositions
df['parsed_comp'] = df['composition'].apply(parse_composition)

# Extract key elements
key_elements = ['Ti', 'Nb', 'Zr', 'Fe', 'Mo', 'Ta', 'Sn', 'V']
for elem in key_elements:
    df[f'frac_{elem}'] = df['parsed_comp'].apply(lambda x: x.get(elem, 0.0))

# Clean data
df_clean = df.copy()
df_clean.loc[df_clean['DAR'] < 0, 'DAR'] = np.nan
df_clean.loc[df_clean['DAR'] < 0, 'DAR_err'] = np.nan

# Log-transform properties (handle zeros)
for prop in property_names:
    vals = df_clean[prop].values
    vals[vals <= 0] = np.nan  # Set non-positive to NaN
    df_clean[f'log_{prop}'] = np.log(vals)

# Prepare covariates
covariate_cols = ['moe'] + [f'frac_{e}' for e in key_elements if e != 'Ti']
X_raw = df_clean[covariate_cols].copy()

# Add intercept
X_raw.insert(0, 'intercept', 1.0)

# Standardize (except intercept)
X_std = X_raw.copy()
for col in X_raw.columns[1:]:
    if X_raw[col].std() > EPSILON:
        X_std[col] = (X_raw[col] - X_raw[col].mean()) / (X_raw[col].std() + EPSILON)
    else:
        X_std[col] = 0.0

X = X_std.values  # n x p
n, p = X.shape

print(f"  - Prepared {n} samples with {p} covariates")
print(f"  - Covariates: {list(X_std.columns)}")

# Extract observed responses (log-transformed)
Z_obs = df_clean[[f'log_{prop}' for prop in property_names]].values  # n x K
S_obs = df_clean[[f'{prop}_err' for prop in property_names]].values  # n x K

# Approximate error on log-scale: sigma_log(y) ≈ sigma_y / y
for k, prop in enumerate(property_names):
    valid_mask = ~np.isnan(df_clean[prop]) & (df_clean[prop] > 0)
    S_obs[valid_mask, k] = S_obs[valid_mask, k] / (df_clean.loc[valid_mask, prop] + EPSILON)

# Missing data indicators
R = (~np.isnan(Z_obs)).astype(int)  # n x K

# Stability class encoding
stability_classes = ['stable', 'meta', 'near', 'rich', 'other']
df_clean['stability_idx'] = df_clean['moe_class'].map(
    {cls: i for i, cls in enumerate(stability_classes)}
).fillna(4)  # 'other' for missing

z_class = df_clean['stability_idx'].values.astype(int)
J = len(stability_classes)

print(f"  - Stability classes: {stability_classes}")
class_dist = {cls: int((z_class == i).sum()) for i, cls in enumerate(stability_classes)}
print(f"  - Class distribution: {class_dist}")

# Missing data summary
missing_fraction = 1 - R.mean(axis=0)
print(f"\n  Missing data per property:")
for i, prop in enumerate(property_names):
    print(f"    {prop}: {missing_fraction[i]*100:.1f}%")

# ============================================================================
# 2. INITIALIZE PARAMETERS WITH SAFE DEFAULTS
# ============================================================================

print("\nStep 2: Initializing MCMC parameters...")

# Impute missing values with column means
Z_init = Z_obs.copy()
for k in range(K):
    col_mean = np.nanmean(Z_obs[:, k])
    if np.isnan(col_mean):
        col_mean = 0.0
    Z_init[np.isnan(Z_init[:, k]), k] = col_mean

# Initialize S with small values for missing entries
S_init = S_obs.copy()
S_init[np.isnan(S_init)] = 0.1

# Empirical pooled regression (OLS with regularization)
XtX = X.T @ X
XtX_inv = safe_inv(XtX, reg=REG_LAMBDA)
B_pool = XtX_inv @ X.T @ Z_init
residuals = Z_init - X @ B_pool

# Empirical covariance (regularized)
Sigma_emp = (residuals.T @ residuals) / (n - p - 1)
Sigma_emp = Sigma_emp + 0.01 * np.eye(K)  # Strong regularization

# Initialize class-specific coefficients
B_classes = {j: B_pool + 0.01 * np.random.randn(p, K) for j in range(J)}

# Initialize global mean
M = B_pool.copy()

# Initialize Lambda
Lambda = np.eye(K) * 0.5

# Initialize tau^2
tau2 = 0.5

# Initialize Sigma
Sigma = Sigma_emp.copy()

# Initialize latent Z
Z = Z_init.copy()

print("  - Initialized all parameters with regularization")

# ============================================================================
# 3. HYPERPARAMETERS
# ============================================================================

# Prior for M
c0 = 1.0

# Prior for Sigma (more conservative)
nu0 = K + 5  # More informative
S0 = Sigma_emp * (nu0 - K - 1)
S0 = S0 + 0.1 * np.eye(K)  # Extra regularization

# Prior for Lambda
nu_Lambda = K + 5  # More informative
S_Lambda = np.eye(K) * 1.0

# Prior for tau^2
a_tau = 3.0  # More informative
b_tau = 1.0

print(f"\n  Hyperparameters:")
print(f"    c0 (M prior scale): {c0}")
print(f"    nu0 (Sigma prior df): {nu0}")
print(f"    nu_Lambda (Lambda prior df): {nu_Lambda}")
print(f"    a_tau, b_tau: {a_tau}, {b_tau}")

# ============================================================================
# 4. MCMC SAMPLER (WITH STABILITY CHECKS)
# ============================================================================

print("\nStep 3: Running MCMC sampler...")
print("  This may take several minutes...\n")

# MCMC settings
n_iter = 5000
n_burn = 1000
n_thin = 2
n_save = (n_iter - n_burn) // n_thin

# Storage
samples = {
    'B': {j: np.zeros((n_save, p, K)) for j in range(J)},
    'M': np.zeros((n_save, p, K)),
    'Sigma': np.zeros((n_save, K, K)),
    'Lambda': np.zeros((n_save, K, K)),
    'tau2': np.zeros(n_save),
    'Z': np.zeros((n_save, n, K)),
}

# Progress bar
def progress_bar(iteration, total, bar_length=50):
    percent = float(iteration) / total
    arrow = '=' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\r  Progress: [{arrow}{spaces}] {int(percent * 100)}% ({iteration}/{total})', end='', flush=True)

# MCMC loop
save_idx = 0
n_errors = 0
max_errors = 100

for iter_num in range(n_iter):
    progress_bar(iter_num + 1, n_iter)
    
    try:
        # --------------------------------------------------------------------
        # Step 1: Sample latent Z for missing/observed entries
        # --------------------------------------------------------------------
        for i in range(n):
            class_i = z_class[i]
            B_i = B_classes[class_i]
            mean_i = X[i, :] @ B_i
            
            for k in range(K):
                if R[i, k] == 0:  # Missing
                    # Sample from prior
                    var_k = max(Sigma[k, k], EPSILON)
                    Z[i, k] = np.random.normal(mean_i[k], np.sqrt(var_k))
                else:  # Observed
                    # Sample from posterior given observation
                    obs_var = max(S_init[i, k]**2, EPSILON)
                    sigma_k = max(Sigma[k, k], EPSILON)
                    post_var = 1.0 / (1.0/sigma_k + 1.0/obs_var)
                    post_mean = post_var * (mean_i[k]/sigma_k + Z_obs[i, k]/obs_var)
                    Z[i, k] = np.random.normal(post_mean, np.sqrt(post_var))
        
        # --------------------------------------------------------------------
        # Step 2: Sample class-specific B_j
        # --------------------------------------------------------------------
        Sigma_inv = safe_inv(Sigma)
        XtX_inv_tau = XtX_inv * tau2
        Lambda_safe = Lambda + REG_LAMBDA * np.eye(K)
        
        for j in range(J):
            idx_j = np.where(z_class == j)[0]
            
            if len(idx_j) == 0:
                # No data, sample from prior
                B_classes[j] = matrix_normal_sample(M, XtX_inv_tau, Lambda_safe)
                continue
            
            X_j = X[idx_j, :]
            Z_j = Z[idx_j, :]
            n_j = len(idx_j)
            
            # Use simplified conjugate update (avoiding large Kronecker products)
            # Posterior for each column separately (approximate)
            B_j_new = np.zeros((p, K))
            for k_idx in range(K):
                # For property k_idx
                z_jk = Z_j[:, k_idx]
                
                # Prior precision (from M)
                prior_prec = XtX_inv_tau / Lambda_safe[k_idx, k_idx]
                prior_prec_mat = safe_inv(prior_prec * np.eye(p))
                
                # Likelihood precision
                like_prec = (X_j.T @ X_j) / Sigma[k_idx, k_idx]
                
                # Posterior
                post_prec = prior_prec_mat + like_prec
                post_cov = safe_inv(post_prec)
                
                prior_mean = M[:, k_idx]
                like_contrib = X_j.T @ z_jk / Sigma[k_idx, k_idx]
                
                post_mean = post_cov @ (prior_prec_mat @ prior_mean + like_contrib)
                
                # Sample
                B_j_new[:, k_idx] = mvn_sample(post_mean, post_cov)
            
            B_classes[j] = B_j_new
        
        # --------------------------------------------------------------------
        # Step 3: Sample global mean M
        # --------------------------------------------------------------------
        # Simplified: average of class-specific B_j with shrinkage toward B_pool
        B_mean = np.mean([B_classes[j] for j in range(J)], axis=0)
        shrinkage = 0.9  # Shrink toward empirical estimate
        M = shrinkage * B_mean + (1 - shrinkage) * B_pool
        
        # Add small noise for mixing
        M = M + 0.01 * np.random.randn(p, K)
        
        # --------------------------------------------------------------------
        # Step 4: Sample Sigma (residual covariance)
        # --------------------------------------------------------------------
        residuals_all = []
        for i in range(n):
            class_i = z_class[i]
            B_i = B_classes[class_i]
            resid_i = Z[i, :] - X[i, :] @ B_i
            residuals_all.append(resid_i)
        
        residuals_all = np.array(residuals_all)
        S_resid = residuals_all.T @ residuals_all
        
        df_post = nu0 + n
        scale_post_inv = safe_inv(S0) + S_resid
        scale_post = safe_inv(scale_post_inv)
        
        Sigma = inv_wishart_sample(df_post, scale_post)
        
        # Regularize to ensure positive definiteness
        Sigma = Sigma + REG_LAMBDA * np.eye(K)
        
        # --------------------------------------------------------------------
        # Step 5: Sample Lambda (coefficient coupling)
        # --------------------------------------------------------------------
        B_deviations_sum = np.zeros((K, K))
        for j in range(J):
            B_dev = B_classes[j] - M
            B_deviations_sum += B_dev.T @ XtX @ B_dev / tau2
        
        df_Lambda_post = nu_Lambda + J * p
        scale_Lambda_post_inv = safe_inv(S_Lambda) + B_deviations_sum
        scale_Lambda_post = safe_inv(scale_Lambda_post_inv)
        
        Lambda = inv_wishart_sample(df_Lambda_post, scale_Lambda_post)
        
        # Regularize
        Lambda = Lambda + REG_LAMBDA * np.eye(K)
        
        # --------------------------------------------------------------------
        # Step 6: Sample tau^2
        # --------------------------------------------------------------------
        a_post = a_tau + 0.5 * J * p * K
        
        Lambda_inv = safe_inv(Lambda)
        b_sum = 0
        for j in range(J):
            B_dev = B_classes[j] - M
            b_sum += np.trace(B_dev.T @ XtX @ B_dev @ Lambda_inv)
        
        b_post = b_tau + 0.5 * b_sum
        b_post = max(b_post, EPSILON)  # Ensure positive
        
        tau2 = inv_gamma_sample(a_post, b_post)
        tau2 = np.clip(tau2, 0.01, 10.0)  # Bound for stability
        
        # --------------------------------------------------------------------
        # Store samples
        # --------------------------------------------------------------------
        if iter_num >= n_burn and (iter_num - n_burn) % n_thin == 0:
            for j in range(J):
                samples['B'][j][save_idx, :, :] = B_classes[j]
            samples['M'][save_idx, :, :] = M
            samples['Sigma'][save_idx, :, :] = Sigma
            samples['Lambda'][save_idx, :, :] = Lambda
            samples['tau2'][save_idx] = tau2
            samples['Z'][save_idx, :, :] = Z
            save_idx += 1
    
    except Exception as e:
        n_errors += 1
        if n_errors > max_errors:
            print(f"\n  ERROR: Too many numerical errors ({n_errors}). Stopping.")
            print(f"  Last error: {str(e)}")
            break
        # Skip this iteration and continue
        continue

print("\n  MCMC sampling completed!")
print(f"  Total numerical errors handled: {n_errors}")

# Trim samples if stopped early
if save_idx < n_save:
    print(f"  Note: Collected {save_idx} samples (expected {n_save})")
    for key in samples:
        if key == 'B':
            for j in range(J):
                samples['B'][j] = samples['B'][j][:save_idx]
        else:
            samples[key] = samples[key][:save_idx]
    n_save = save_idx

# ============================================================================
# 5. POSTERIOR SUMMARIES
# ============================================================================

print("\nStep 4: Computing posterior summaries...")

def posterior_summary(samples_array, axis=0):
    """Compute mean, std, and 95% CI"""
    mean = np.mean(samples_array, axis=axis)
    std = np.std(samples_array, axis=axis)
    ci_lower = np.percentile(samples_array, 2.5, axis=axis)
    ci_upper = np.percentile(samples_array, 97.5, axis=axis)
    return mean, std, ci_lower, ci_upper

# Global mean M
M_mean, M_std, M_lower, M_upper = posterior_summary(samples['M'], axis=0)

# Residual covariance Sigma
Sigma_mean = np.mean(samples['Sigma'], axis=0)
Sigma_std = np.std(samples['Sigma'], axis=0)

# Coefficient coupling Lambda
Lambda_mean = np.mean(samples['Lambda'], axis=0)

# Global shrinkage tau^2
tau2_mean = np.mean(samples['tau2'])
tau2_std = np.std(samples['tau2'])

print(f"  - Posterior mean tau^2: {tau2_mean:.4f} ± {tau2_std:.4f}")
print(f"  - Posterior mean Sigma diagonal: {np.diag(Sigma_mean)}")

# Save summaries
summaries = {
    'tau2_mean': float(tau2_mean),
    'tau2_std': float(tau2_std),
    'Sigma_mean': Sigma_mean.tolist(),
    'Lambda_mean': Lambda_mean.tolist(),
    'n_samples': int(n_save),
    'n_errors': int(n_errors),
}

with open(f"{output_dir}/posterior_summaries.json", 'w') as f:
    json.dump(summaries, f, indent=2)

# ============================================================================
# 6. POSTERIOR PREDICTIVE DISTRIBUTIONS
# ============================================================================

print("\nStep 5: Computing posterior predictive distributions...")

# Select test cases
test_indices = [0, 50, 100, 150, 200]
test_indices = [i for i in test_indices if i < n]  # Ensure valid
n_test = len(test_indices)

predictive_samples = np.zeros((n_save, n_test, K))

for s in range(n_save):
    for t_idx, i in enumerate(test_indices):
        class_i = z_class[i]
        B_i = samples['B'][class_i][s, :, :]
        mean_pred = X[i, :] @ B_i
        Sigma_s = samples['Sigma'][s, :, :]
        predictive_samples[s, t_idx, :] = mvn_sample(mean_pred, Sigma_s)

# Back-transform
predictive_samples_original = np.exp(predictive_samples)

print(f"  - Generated {n_save} predictive samples for {n_test} test cases")

# ============================================================================
# 7. DIAGNOSTICS (SIMPLIFIED)
# ============================================================================

print("\nStep 6: Computing MCMC diagnostics...")

# Trace plots
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("MCMC Trace Plots", fontsize=14, fontweight='bold')

axes[0, 0].plot(samples['tau2'], alpha=0.7, linewidth=0.5)
axes[0, 0].set_title(r'$\tau^2$ (Global Shrinkage)')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].grid(True, alpha=0.3)

for k in range(min(3, K)):
    sigma_kk = samples['Sigma'][:, k, k]
    axes[0, 1].plot(sigma_kk, alpha=0.7, linewidth=0.5, label=property_names[k])
axes[0, 1].set_title(r'$\Sigma$ Diagonal Elements')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

for k in range(min(3, K)):
    M_0k = samples['M'][:, 0, k]
    axes[1, 0].plot(M_0k, alpha=0.7, linewidth=0.5, label=property_names[k])
axes[1, 0].set_title('M Intercept Coefficients')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

for k in range(min(3, K)):
    B_0_0k = samples['B'][0][:, 0, k]
    axes[1, 1].plot(B_0_0k, alpha=0.7, linewidth=0.5, label=property_names[k])
axes[1, 1].set_title('B[stable] Intercept Coefficients')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

for k in range(min(3, K)):
    lambda_kk = samples['Lambda'][:, k, k]
    axes[2, 0].plot(lambda_kk, alpha=0.7, linewidth=0.5, label=property_names[k])
axes[2, 0].set_title(r'$\Lambda$ Diagonal Elements')
axes[2, 0].set_xlabel('Iteration')
axes[2, 0].legend(fontsize=8)
axes[2, 0].grid(True, alpha=0.3)

for k in range(min(3, K)):
    Z_0k = samples['Z'][:, 0, k]
    axes[2, 1].plot(Z_0k, alpha=0.7, linewidth=0.5, label=property_names[k])
axes[2, 1].set_title('Latent Z[0] (Log Properties)')
axes[2, 1].set_xlabel('Iteration')
axes[2, 1].legend(fontsize=8)
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/diagnostics/trace_plots.png", bbox_inches='tight')
print(f"  - Saved: diagnostics/trace_plots.png")
plt.close()

# Autocorrelation (simplified)
def autocorr(x, max_lag=50):
    """Compute autocorrelation"""
    x = x - np.mean(x)
    c0 = np.dot(x, x) / len(x)
    if c0 == 0:
        return np.ones(max_lag)
    acf = [1.0] + [np.dot(x[:-lag], x[lag:]) / len(x) / c0 for lag in range(1, max_lag)]
    return np.array(acf)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("MCMC Autocorrelation Plots", fontsize=14, fontweight='bold')

acf_tau2 = autocorr(samples['tau2'])
axes[0, 0].plot(acf_tau2, marker='o', markersize=3)
axes[0, 0].axhline(0, color='k', linestyle='--', linewidth=0.5)
axes[0, 0].set_title(r'ACF: $\tau^2$')
axes[0, 0].set_xlabel('Lag')
axes[0, 0].grid(True, alpha=0.3)

acf_sigma00 = autocorr(samples['Sigma'][:, 0, 0])
axes[0, 1].plot(acf_sigma00, marker='o', markersize=3)
axes[0, 1].axhline(0, color='k', linestyle='--', linewidth=0.5)
axes[0, 1].set_title(r'ACF: $\Sigma_{11}$')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].grid(True, alpha=0.3)

acf_M00 = autocorr(samples['M'][:, 0, 0])
axes[0, 2].plot(acf_M00, marker='o', markersize=3)
axes[0, 2].axhline(0, color='k', linestyle='--', linewidth=0.5)
axes[0, 2].set_title(r'ACF: $M_{11}$')
axes[0, 2].set_xlabel('Lag')
axes[0, 2].grid(True, alpha=0.3)

acf_B000 = autocorr(samples['B'][0][:, 0, 0])
axes[1, 0].plot(acf_B000, marker='o', markersize=3)
axes[1, 0].axhline(0, color='k', linestyle='--', linewidth=0.5)
axes[1, 0].set_title(r'ACF: $B_{stable,11}$')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].grid(True, alpha=0.3)

acf_Lambda00 = autocorr(samples['Lambda'][:, 0, 0])
axes[1, 1].plot(acf_Lambda00, marker='o', markersize=3)
axes[1, 1].axhline(0, color='k', linestyle='--', linewidth=0.5)
axes[1, 1].set_title(r'ACF: $\Lambda_{11}$')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].grid(True, alpha=0.3)

acf_Z00 = autocorr(samples['Z'][:, 0, 0])
axes[1, 2].plot(acf_Z00, marker='o', markersize=3)
axes[1, 2].axhline(0, color='k', linestyle='--', linewidth=0.5)
axes[1, 2].set_title(r'ACF: $Z_{11}$')
axes[1, 2].set_xlabel('Lag')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/diagnostics/autocorrelation_plots.png", bbox_inches='tight')
print(f"  - Saved: diagnostics/autocorrelation_plots.png")
plt.close()

# Effective sample size
def ess(x, max_lag=None):
    """Estimate effective sample size"""
    if max_lag is None:
        max_lag = min(len(x) // 2, 500)
    acf_vals = autocorr(x, max_lag)
    cutoff = np.where(acf_vals < 0.05)[0]
    if len(cutoff) > 0:
        tau = 1 + 2 * np.sum(acf_vals[1:cutoff[0]])
    else:
        tau = 1 + 2 * np.sum(acf_vals[1:])
    return len(x) / max(tau, 1.0)

ess_tau2 = ess(samples['tau2'])
ess_sigma00 = ess(samples['Sigma'][:, 0, 0])
ess_M00 = ess(samples['M'][:, 0, 0])

print(f"\n  Effective Sample Sizes:")
print(f"    tau^2: {ess_tau2:.0f}")
print(f"    Sigma[0,0]: {ess_sigma00:.0f}")
print(f"    M[0,0]: {ess_M00:.0f}")

# R-hat (split-chain approximation)
def split_chain_rhat(chain):
    """Approximate R-hat"""
    n = len(chain)
    if n < 4:
        return 1.0
    chain1 = chain[:n//2]
    chain2 = chain[n//2:]
    
    mean1 = np.mean(chain1)
    mean2 = np.mean(chain2)
    var1 = np.var(chain1, ddof=1)
    var2 = np.var(chain2, ddof=1)
    
    W = (var1 + var2) / 2
    B = ((mean1 - mean2)**2) * (n//2)
    
    if W == 0:
        return 1.0
    
    var_plus = (n//2 - 1) / (n//2) * W + B / (n//2)
    rhat = np.sqrt(var_plus / W)
    return rhat

rhat_tau2 = split_chain_rhat(samples['tau2'])
rhat_sigma00 = split_chain_rhat(samples['Sigma'][:, 0, 0])
rhat_M00 = split_chain_rhat(samples['M'][:, 0, 0])

fig, ax = plt.subplots(figsize=(8, 5))
params = [r'$\tau^2$', r'$\Sigma_{11}$', r'$M_{11}$']
rhat_values = [rhat_tau2, rhat_sigma00, rhat_M00]

bars = ax.barh(params, rhat_values, color=['green' if r < 1.1 else 'orange' for r in rhat_values], alpha=0.8)
ax.axvline(1.0, color='blue', linestyle='--', linewidth=2, label='Target (1.0)')
ax.axvline(1.1, color='red', linestyle='--', linewidth=2, label='Threshold (1.1)')
ax.set_xlabel(r'$\hat{R}$ (Split-Chain)', fontsize=11)
ax.set_title('Convergence Diagnostic', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

for i, v in enumerate(rhat_values):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{output_dir}/diagnostics/rhat_diagnostics.png", bbox_inches='tight')
print(f"  - Saved: diagnostics/rhat_diagnostics.png")
plt.close()

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\nStep 7: Generating visualizations...")

# Plot 1: Sigma correlation heatmap
plt.figure(figsize=(8, 6))
corr_matrix = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        denom = np.sqrt(Sigma_mean[i, i] * Sigma_mean[j, j])
        corr_matrix[i, j] = Sigma_mean[i, j] / denom if denom > 0 else 0

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            xticklabels=property_names, yticklabels=property_names,
            cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
plt.title('Posterior Mean Residual Correlation Matrix (Σ)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/plots/sigma_correlation_heatmap.png", bbox_inches='tight')
print(f"  - Saved: plots/sigma_correlation_heatmap.png")
plt.close()

# Plot 2: Lambda coupling heatmap
plt.figure(figsize=(8, 6))
corr_lambda = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        denom = np.sqrt(Lambda_mean[i, i] * Lambda_mean[j, j])
        corr_lambda[i, j] = Lambda_mean[i, j] / denom if denom > 0 else 0

sns.heatmap(corr_lambda, annot=True, fmt='.3f', cmap='viridis',
            xticklabels=property_names, yticklabels=property_names,
            cbar_kws={'label': 'Coupling'}, vmin=0, vmax=1)
plt.title('Posterior Mean Coefficient Coupling Matrix (Λ)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/plots/lambda_coupling_heatmap.png", bbox_inches='tight')
print(f"  - Saved: plots/lambda_coupling_heatmap.png")
plt.close()

# Plot 3: Global mean coefficients
fig, axes = plt.subplots(1, K, figsize=(16, 4), sharey=True)
fig.suptitle('Posterior Distributions of Global Mean Coefficients (M)', fontsize=13, fontweight='bold')

covariate_names_short = ['Int'] + [c.replace('frac_', '') for c in X_std.columns[1:]]

for k in range(K):
    means_k = M_mean[:, k]
    lower_k = M_lower[:, k]
    upper_k = M_upper[:, k]
    
    axes[k].errorbar(means_k, range(p), xerr=[means_k - lower_k, upper_k - means_k],
                     fmt='o', markersize=4, capsize=3, capthick=1, alpha=0.8)
    axes[k].axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[k].set_xlabel('Coefficient Value', fontsize=9)
    axes[k].set_title(property_names[k], fontsize=10, fontweight='bold')
    axes[k].grid(True, alpha=0.3, axis='x')
    
    if k == 0:
        axes[k].set_yticks(range(p))
        axes[k].set_yticklabels(covariate_names_short, fontsize=8)

plt.tight_layout()
plt.savefig(f"{output_dir}/plots/global_mean_coefficients.png", bbox_inches='tight')
print(f"  - Saved: plots/global_mean_coefficients.png")
plt.close()

# Plot 4: Class-specific intercepts
B_means = {j: np.mean(samples['B'][j], axis=0) for j in range(J)}

fig, axes = plt.subplots(1, K, figsize=(16, 4))
fig.suptitle('Class-Specific Intercepts by Stability Class', fontsize=13, fontweight='bold')

for k in range(K):
    intercepts = [B_means[j][0, k] for j in range(J)]
    axes[k].bar(range(J), intercepts, color=sns.color_palette("Set2", J), alpha=0.8)
    axes[k].set_xticks(range(J))
    axes[k].set_xticklabels(stability_classes, rotation=45, ha='right', fontsize=8)
    axes[k].set_ylabel('Intercept (log scale)', fontsize=9)
    axes[k].set_title(property_names[k], fontsize=10, fontweight='bold')
    axes[k].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{output_dir}/plots/class_specific_intercepts.png", bbox_inches='tight')
print(f"  - Saved: plots/class_specific_intercepts.png")
plt.close()

# Plot 5: Predictive distributions
fig, axes = plt.subplots(n_test, K, figsize=(16, max(10, n_test*2)))
fig.suptitle('Posterior Predictive Distributions (Original Scale)', fontsize=14, fontweight='bold')

for t_idx in range(n_test):
    for k in range(K):
        samples_tk = predictive_samples_original[:, t_idx, k]
        
        obs_val = df_clean.iloc[test_indices[t_idx]][property_names[k]]
        
        axes[t_idx, k].hist(samples_tk, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        
        if not np.isnan(obs_val):
            axes[t_idx, k].axvline(obs_val, color='red', linestyle='--', linewidth=2, 
                                   label=f'Observed: {obs_val:.1f}')
            axes[t_idx, k].legend(fontsize=7, loc='upper right')
        
        pred_mean = np.mean(samples_tk)
        axes[t_idx, k].axvline(pred_mean, color='green', linestyle='-', linewidth=1.5,
                               label=f'Pred Mean: {pred_mean:.1f}')
        
        if k == 0:
            alloy_name = str(df_clean.iloc[test_indices[t_idx]]['eid'])
            axes[t_idx, k].set_ylabel(f'{alloy_name[:15]}...', fontsize=8)
        
        if t_idx == 0:
            axes[t_idx, k].set_title(property_names[k], fontsize=10, fontweight='bold')
        
        axes[t_idx, k].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{output_dir}/plots/predictive_distributions.png", bbox_inches='tight')
print(f"  - Saved: plots/predictive_distributions.png")
plt.close()

# Plot 6: Property trade-offs
Z_mean_post = np.mean(samples['Z'], axis=0)
Z_original_mean = np.exp(Z_mean_post)

pairs = [(0, 1), (1, 3), (2, 4)]
pair_names = [
    (property_names[pairs[0][0]], property_names[pairs[0][1]]),
    (property_names[pairs[1][0]], property_names[pairs[1][1]]),
    (property_names[pairs[2][0]], property_names[pairs[2][1]]),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Property Trade-offs (Posterior Mean, Original Scale)', fontsize=13, fontweight='bold')

for i, (p1, p2) in enumerate(pairs):
    colors = [sns.color_palette("Set2", J)[z_class[j]] for j in range(n)]
    
    axes[i].scatter(Z_original_mean[:, p1], Z_original_mean[:, p2], 
                    c=colors, alpha=0.6, s=30, edgecolors='k', linewidths=0.3)
    axes[i].set_xlabel(pair_names[i][0], fontsize=10)
    axes[i].set_ylabel(pair_names[i][1], fontsize=10)
    axes[i].grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=sns.color_palette("Set2", J)[j], label=stability_classes[j])
                   for j in range(J)]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
           ncol=J, fontsize=9, title='Stability Class')

plt.tight_layout()
plt.savefig(f"{output_dir}/plots/property_tradeoffs.png", bbox_inches='tight')
print(f"  - Saved: plots/property_tradeoffs.png")
plt.close()

# Plot 7: Observed vs predicted
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Observed vs Predicted with 95% Credible Intervals', fontsize=13, fontweight='bold')
axes = axes.flatten()

for k in range(K):
    pred_mean_all = np.mean(samples['Z'], axis=0)[:, k]
    pred_lower_all = np.percentile(samples['Z'], 2.5, axis=0)[:, k]
    pred_upper_all = np.percentile(samples['Z'], 97.5, axis=0)[:, k]
    
    pred_mean_orig = np.exp(pred_mean_all)
    pred_lower_orig = np.exp(pred_lower_all)
    pred_upper_orig = np.exp(pred_upper_all)
    
    obs_orig = df_clean[property_names[k]].values
    
    valid = ~np.isnan(obs_orig) & (obs_orig > 0)
    
    if valid.sum() > 0:
        axes[k].errorbar(obs_orig[valid], pred_mean_orig[valid],
                         yerr=[pred_mean_orig[valid] - pred_lower_orig[valid],
                               pred_upper_orig[valid] - pred_mean_orig[valid]],
                         fmt='o', markersize=3, alpha=0.5, capsize=2, elinewidth=0.5)
        
        lims = [min(obs_orig[valid].min(), pred_mean_orig[valid].min()),
                max(obs_orig[valid].max(), pred_mean_orig[valid].max())]
        axes[k].plot(lims, lims, 'r--', alpha=0.8, linewidth=1.5)
    
    axes[k].set_xlabel(f'Observed {property_names[k]}', fontsize=9)
    axes[k].set_ylabel(f'Predicted {property_names[k]}', fontsize=9)
    axes[k].set_title(property_names[k], fontsize=10, fontweight='bold')
    axes[k].grid(True, alpha=0.3)

axes[-1].axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}/plots/observed_vs_predicted.png", bbox_inches='tight')
print(f"  - Saved: plots/observed_vs_predicted.png")
plt.close()

# ============================================================================
# 9. TABLES
# ============================================================================

print("\nStep 8: Generating summary tables...")

# Table 1: Posterior summaries
table1_data = {
    'Parameter': [
        r'tau^2',
        r'Sigma[YM,YM]',
        r'Sigma[YS,YS]',
        r'Sigma[UTS,UTS]',
        r'Lambda[YM,YM]',
        r'M[intercept,YM]',
        r'M[intercept,YS]',
    ],
    'Mean': [
        tau2_mean,
        Sigma_mean[0, 0],
        Sigma_mean[1, 1],
        Sigma_mean[2, 2],
        Lambda_mean[0, 0],
        M_mean[0, 0],
        M_mean[0, 1],
    ],
    'Std': [
        tau2_std,
        Sigma_std[0, 0],
        Sigma_std[1, 1],
        Sigma_std[2, 2],
        np.std(samples['Lambda'][:, 0, 0]),
        M_std[0, 0],
        M_std[0, 1],
    ],
    '2.5%': [
        np.percentile(samples['tau2'], 2.5),
        np.percentile(samples['Sigma'][:, 0, 0], 2.5),
        np.percentile(samples['Sigma'][:, 1, 1], 2.5),
        np.percentile(samples['Sigma'][:, 2, 2], 2.5),
        np.percentile(samples['Lambda'][:, 0, 0], 2.5),
        M_lower[0, 0],
        M_lower[0, 1],
    ],
    '97.5%': [
        np.percentile(samples['tau2'], 97.5),
        np.percentile(samples['Sigma'][:, 0, 0], 97.5),
        np.percentile(samples['Sigma'][:, 1, 1], 97.5),
        np.percentile(samples['Sigma'][:, 2, 2], 97.5),
        np.percentile(samples['Lambda'][:, 0, 0], 97.5),
        M_upper[0, 0],
        M_upper[0, 1],
    ],
}

df_table1 = pd.DataFrame(table1_data)
df_table1.to_csv(f"{output_dir}/tables/posterior_parameter_summary.csv", index=False)
print(f"  - Saved: tables/posterior_parameter_summary.csv")
print("\n" + df_table1.to_string(index=False))

# Table 2: Predictive performance
rmse_values = []
coverage_values = []

for k in range(K):
    obs_k = df_clean[property_names[k]].values
    valid_k = ~np.isnan(obs_k) & (obs_k > 0)
    
    if valid_k.sum() == 0:
        rmse_values.append(np.nan)
        coverage_values.append(np.nan)
        continue
    
    pred_mean_k = np.exp(np.mean(samples['Z'], axis=0)[:, k])
    pred_lower_k = np.exp(np.percentile(samples['Z'], 2.5, axis=0)[:, k])
    pred_upper_k = np.exp(np.percentile(samples['Z'], 97.5, axis=0)[:, k])
    
    rmse = np.sqrt(np.mean((obs_k[valid_k] - pred_mean_k[valid_k])**2))
    rmse_values.append(rmse)
    
    coverage = np.mean((obs_k[valid_k] >= pred_lower_k[valid_k]) & 
                       (obs_k[valid_k] <= pred_upper_k[valid_k]))
    coverage_values.append(coverage)

table2_data = {
    'Property': property_names,
    'RMSE': rmse_values,
    '95% CI Coverage': coverage_values,
}

df_table2 = pd.DataFrame(table2_data)
df_table2.to_csv(f"{output_dir}/tables/predictive_performance.csv", index=False)
print(f"  - Saved: tables/predictive_performance.csv")
print("\n" + df_table2.to_string(index=False))

# Table 3: Class-specific intercepts
intercept_summary = []
for j in range(J):
    B_j_samples = samples['B'][j]
    for k in range(K):
        intercept_samples = B_j_samples[:, 0, k]
        intercept_summary.append({
            'Stability Class': stability_classes[j],
            'Property': property_names[k],
            'Mean': np.mean(intercept_samples),
            'Std': np.std(intercept_samples),
            '2.5%': np.percentile(intercept_samples, 2.5),
            '97.5%': np.percentile(intercept_samples, 97.5),
        })

df_table3 = pd.DataFrame(intercept_summary)
df_table3.to_csv(f"{output_dir}/tables/class_specific_intercepts.csv", index=False)
print(f"  - Saved: tables/class_specific_intercepts.csv")
print("\n" + df_table3.head(10).to_string(index=False))

# ============================================================================
# 10. CREATE ZIP ARCHIVE
# ============================================================================

print("\nStep 9: Creating ZIP archive of all outputs...")

zip_filename = f"{output_dir}.zip"
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, output_dir)
            zipf.write(file_path, arcname)

print(f"  - Created: {zip_filename}")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("MCMC ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nAll outputs saved to: {output_dir}/")
print(f"Compressed archive: {zip_filename}")
print(f"\nSamples collected: {n_save}")
print(f"Numerical errors handled: {n_errors}")
print("\nGenerated outputs:")
print("  Plots (7)")
print("  Diagnostics (3)")
print("  Tables (3)")
print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
