# Physics-Informed Bayesian MCMC for Ti-Alloy Property Prediction

## Overview

This implementation provides a complete manual MCMC sampler for the physics-informed Bayesian surrogate model described in:

**"Physics-Informed Bayesian Surrogate Maps of Titanium-Alloy Mechanical Property Landscapes Using the DAX-Ti Database"**

The code implements hierarchical multivariate regression with:
- ✅ Manual Metropolis-within-Gibbs MCMC (no external MCMC packages)
- ✅ Physics-informed priors based on Mo-equivalent stability classes
- ✅ Missing data imputation for partially observed properties
- ✅ Measurement error incorporation
- ✅ Comprehensive diagnostics and visualizations

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy
```

**Python version:** 3.7+

## Input Data

Place `dax-ti-static.csv` in the same directory as the script.

The dataset should contain columns:
- Mechanical properties: `YM`, `YS`, `UTS`, `DAR`, `HV`
- Property errors: `YM_err`, `YS_err`, `UTS_err`, `DAR_err`, `HV_err`
- Composition: `composition`, `moe` (Mo-equivalent), `moe_class`
- Other features: parsed from composition string

## Usage

### Running the Analysis

```bash
python bayesian_ti_alloy_mcmc.py
```

The script will:
1. Load and preprocess the DAX-Ti database
2. Run 5000 MCMC iterations (1000 burn-in, thinning=2)
3. Generate 10+ publication-quality plots
4. Create 3 summary tables
5. Compute MCMC diagnostics
6. Package everything into a ZIP archive

### Expected Runtime

- **5000 iterations:** ~10-15 minutes on modern CPU
- **Progress bar** displays real-time iteration count

## Outputs

All outputs are saved to timestamped directory: `ti_alloy_bayesian_outputs_YYYYMMDD_HHMMSS/`

### 📊 Plots (10 total)

**Main Results:**
1. `sigma_correlation_heatmap.png` - Posterior residual correlation matrix (Σ)
2. `lambda_coupling_heatmap.png` - Coefficient coupling matrix (Λ)
3. `global_mean_coefficients.png` - Forest plot of global mean coefficients
4. `class_specific_intercepts.png` - Intercepts by stability class
5. `predictive_distributions.png` - Posterior predictive distributions for test cases
6. `property_tradeoffs.png` - Pairwise scatter plots (YM vs YS, etc.)
7. `observed_vs_predicted.png` - Calibration plots with 95% CI

**Diagnostics:**
8. `trace_plots.png` - MCMC traces for key parameters
9. `autocorrelation_plots.png` - ACF plots for convergence assessment
10. `rhat_diagnostics.png` - Gelman-Rubin R-hat convergence diagnostic

### 📋 Tables (3 CSV files)

1. `posterior_parameter_summary.csv` - Means, SDs, and 95% CIs for:
   - τ² (global shrinkage)
   - Σ (residual covariance diagonal)
   - Λ (coefficient coupling diagonal)
   - M (global mean intercepts)

2. `predictive_performance.csv` - Per-property metrics:
   - RMSE (root mean squared error)
   - 95% CI coverage

3. `class_specific_intercepts.csv` - Intercepts for each:
   - Stability class (stable, meta, near, rich, other)
   - Property (YM, YS, UTS, DAR, HV)

### 📦 Additional Files

- `posterior_summaries.json` - Machine-readable posterior summaries
- `ti_alloy_bayesian_outputs_YYYYMMDD_HHMMSS.zip` - Complete archive

## Methodology Details

### Model Structure

**Observation Model (with measurement error):**
```
z_ik^obs | z_ik, s_ik² ~ N(z_ik, s_ik²)
```

**Latent Regression:**
```
z_i | x_i, class_i, {B_j}, Σ ~ N_K(B_class_i^T x_i, Σ)
```

**Hierarchical Priors:**
```
vec(B_j) | M, Ω ~ N_pK(vec(M), Ω)
Ω = τ² (X^T X)^(-1) ⊗ Λ
vec(M) | Σ ~ N_pK(vec(B_pool), c0 (X^T X)^(-1) ⊗ Σ)
Σ^(-1) ~ Wishart_K(ν0, S0^(-1))
Λ^(-1) ~ Wishart_K(ν_Λ, S_Λ^(-1))
τ² ~ InvGamma(a_τ, b_τ)
```

### MCMC Sampler Steps

The manual Gibbs sampler iterates:

1. **Sample latent Z** for missing/observed entries
2. **Sample class-specific B_j** from matrix-normal conditionals
3. **Sample global mean M** from matrix-normal conditional
4. **Sample residual covariance Σ** from inverse-Wishart
5. **Sample coefficient coupling Λ** from inverse-Wishart
6. **Sample global shrinkage τ²** from inverse-gamma

### Key Features

- **No external MCMC packages:** Pure NumPy/SciPy implementation
- **Missing data handling:** Gibbs sampling for NA entries
- **Physics-informed:** Mo-equivalent stability classes structure priors
- **Measurement errors:** Incorporated directly in likelihood
- **Multivariate:** Joint modeling of 5 properties

## Interpreting Results

### Convergence Diagnostics

**Good convergence indicators:**
- Trace plots show "hairy caterpillar" (no trends/drift)
- ACF decays to near-zero within ~50 lags
- R-hat < 1.1 for all parameters

**If convergence issues:**
- Increase `n_iter` (e.g., 10000)
- Increase `n_burn` (e.g., 2000)
- Check for numerical issues in data preprocessing

### Model Assessment

**Sigma correlation matrix:**
- Shows residual correlations between properties
- Strong positive: properties move together after accounting for covariates
- Example: YS and UTS typically strongly correlated

**Lambda coupling matrix:**
- Shows how regression coefficients couple across properties
- High coupling → covariate effects similar across properties

**Predictive performance:**
- RMSE: Lower is better (compare to naive baseline)
- 95% CI Coverage: Should be close to 0.95 (calibration check)

### Design Insights

**Class-specific intercepts:**
- Compare baseline property levels across stability classes
- Metastable β typically: low YM, moderate strength
- Stable β: higher YM, variable strength

**Property trade-offs:**
- YM vs YS: Often negative correlation (low modulus, high strength desirable)
- YS vs DAR: Strength-ductility trade-off
- Use posterior predictive samples for multi-objective optimization

## Customization

### Adjusting MCMC Settings

In the script, modify:
```python
n_iter = 10000  # Total iterations
n_burn = 2000   # Burn-in
n_thin = 5      # Thinning
```

### Changing Priors

Hyperparameters are set in Section 4:
```python
c0 = 1.0         # M prior scale (unit-information)
nu0 = K + 2      # Sigma prior df (minimally informative)
a_tau = 2.0      # tau^2 prior shape
b_tau = 1.0      # tau^2 prior rate
```

### Adding Covariates

Modify `covariate_cols` in Section 1:
```python
covariate_cols = ['moe', 'frac_Nb', 'frac_Zr', 'frac_Fe', ...]
```

## Troubleshooting

**Issue:** `LinAlgError: Matrix is singular`
- **Fix:** Check for constant columns in covariates
- **Fix:** Increase regularization: `XtX_inv = inv(X.T @ X + 1e-4 * np.eye(p))`

**Issue:** Very slow convergence
- **Fix:** Standardize covariates more carefully
- **Fix:** Increase thinning to reduce autocorrelation

**Issue:** Coverage far from 0.95
- **Fix:** Check measurement error estimates (`S_obs`)
- **Fix:** Adjust prior hyperparameters (ν0, S0)

## Citation

If you use this code, please cite:

```
Chatterjee, D. (2025). Physics-Informed Bayesian Surrogate Maps of 
Titanium-Alloy Mechanical Property Landscapes Using the DAX-Ti Database.
Department of Statistics, Visva-Bharati University.
```

And the original DAX-Ti database:

```
Salvador, C. A. F., et al. (2022). A compilation of experimental data 
on the mechanical properties of multicomponent Ti-based alloys. 
Scientific Data, 9(1), 188.
```

## License

This implementation is provided for research and educational purposes.

## Contact

For questions or issues:
- Check trace plots and diagnostics first
- Review methodology in paper Section 3
- Ensure data preprocessing completed correctly

---

**Note:** This is a research implementation. For production use, consider:
- Stan/PyMC for more robust MCMC
- Variational inference for faster approximation
- Cross-validation for hyperparameter tuning
