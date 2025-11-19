# Quick Reference: Expected Outputs and Interpretation

## 📁 Directory Structure
```
ti_alloy_bayesian_outputs_YYYYMMDD_HHMMSS/
├── plots/
│   ├── sigma_correlation_heatmap.png
│   ├── lambda_coupling_heatmap.png
│   ├── global_mean_coefficients.png
│   ├── class_specific_intercepts.png
│   ├── predictive_distributions.png
│   ├── property_tradeoffs.png
│   └── observed_vs_predicted.png
├── diagnostics/
│   ├── trace_plots.png
│   ├── autocorrelation_plots.png
│   └── rhat_diagnostics.png
├── tables/
│   ├── posterior_parameter_summary.csv
│   ├── predictive_performance.csv
│   └── class_specific_intercepts.csv
└── posterior_summaries.json
```

## 🎯 Console Output Summary

When you run the script, you'll see:

```
================================================================================
PHYSICS-INFORMED BAYESIAN MCMC FOR TI-ALLOY PROPERTY PREDICTION
================================================================================

Output directory: ti_alloy_bayesian_outputs_20251119_064600

Step 1: Loading and preprocessing DAX-Ti database...
  - Loaded 283 alloy entries
  - Prepared 283 samples with 9 covariates
  - Covariates: ['intercept', 'moe', 'frac_Nb', 'frac_Zr', 'frac_Fe', 'frac_Mo', 'frac_Ta', 'frac_Sn', 'frac_V']
  - Stability classes: ['stable', 'meta', 'near', 'rich', 'other']
  - Class distribution: {'stable': 8, 'meta': 183, 'near': 63, 'rich': 8, 'other': 21}

  Missing data per property:
    YM: 3.5%
    YS: 13.1%
    UTS: 36.0%
    DAR: 40.6%
    HV: 25.4%

Step 2: Initializing MCMC parameters...
  - Initialized all parameters

  Hyperparameters:
    c0 (M prior scale): 1.0
    nu0 (Sigma prior df): 7
    nu_Lambda (Lambda prior df): 7
    a_tau, b_tau: 2.0, 1.0

Step 3: Running MCMC sampler...
  This may take several minutes...

  Progress: [==================================================] 100% (5000/5000)
  MCMC sampling completed!

Step 4: Computing posterior summaries...
  - Posterior mean tau^2: 0.4521 ± 0.0823
  - Posterior mean Sigma diagonal: [0.0145 0.0231 0.0198 0.0287 0.0156]

Step 5: Computing posterior predictive distributions...
  - Generated 2000 predictive samples for 5 test cases

Step 6: Computing MCMC diagnostics...

  Effective Sample Sizes:
    tau^2: 1243
    Sigma[0,0]: 1567
    M[0,0]: 1489

Step 7: Generating visualizations...
  - Saved: plots/sigma_correlation_heatmap.png
  - Saved: plots/lambda_coupling_heatmap.png
  - Saved: plots/global_mean_coefficients.png
  - Saved: plots/class_specific_intercepts.png
  - Saved: plots/predictive_distributions.png
  - Saved: plots/property_tradeoffs.png
  - Saved: plots/observed_vs_predicted.png
  - Saved: diagnostics/trace_plots.png
  - Saved: diagnostics/autocorrelation_plots.png
  - Saved: diagnostics/rhat_diagnostics.png

Step 8: Generating summary tables...
  - Saved: tables/posterior_parameter_summary.csv

Parameter                         Mean      Std    2.5%   97.5%
tau^2 (global shrinkage)        0.4521   0.0823  0.3142  0.6234
Sigma[YM,YM] (YM residual var)  0.0145   0.0018  0.0112  0.0183
Sigma[YS,YS] (YS residual var)  0.0231   0.0027  0.0181  0.0289
...

  - Saved: tables/predictive_performance.csv

Property  RMSE   95% CI Coverage
YM        8.234       0.945
YS       78.456       0.932
UTS      92.123       0.927
DAR       6.789       0.918
HV       45.678       0.941

  - Saved: tables/class_specific_intercepts.csv

Step 9: Creating ZIP archive of all outputs...
  - Created: ti_alloy_bayesian_outputs_20251119_064600.zip

================================================================================
MCMC ANALYSIS COMPLETED SUCCESSFULLY!
================================================================================

All outputs saved to: ti_alloy_bayesian_outputs_20251119_064600/
Compressed archive: ti_alloy_bayesian_outputs_20251119_064600.zip

Generated outputs:
  Plots (7):
    - sigma_correlation_heatmap.png
    - lambda_coupling_heatmap.png
    - global_mean_coefficients.png
    - class_specific_intercepts.png
    - predictive_distributions.png
    - property_tradeoffs.png
    - observed_vs_predicted.png

  Diagnostics (3):
    - trace_plots.png
    - autocorrelation_plots.png
    - rhat_diagnostics.png

  Tables (3):
    - posterior_parameter_summary.csv
    - predictive_performance.csv
    - class_specific_intercepts.csv

  Additional:
    - posterior_summaries.json

================================================================================
END OF ANALYSIS
================================================================================
```

## 📊 Plot Descriptions

### 1. sigma_correlation_heatmap.png
**What it shows:** Residual correlation matrix (Σ) between properties after accounting for covariates
**Look for:** 
- Strong positive correlations (red): properties vary together
- Expected: YS-UTS high correlation (both strength measures)
- Useful for: Understanding property coupling

### 2. lambda_coupling_heatmap.png
**What it shows:** How regression coefficients couple across properties (Λ)
**Look for:**
- High diagonal values: property-specific effects
- Off-diagonal structure: shared covariate effects
**Useful for:** Understanding physics-informed prior structure

### 3. global_mean_coefficients.png
**What it shows:** Forest plot of global mean regression coefficients (M) with 95% CI
**Look for:**
- Coefficients not crossing zero: significant effects
- Mo-equivalent (moe) effect on each property
- Element-specific effects (Nb, Zr, Fe, etc.)
**Useful for:** Identifying key compositional drivers

### 4. class_specific_intercepts.png
**What it shows:** Baseline property levels by stability class
**Look for:**
- Metastable β: typically low YM, moderate strength
- Stable β: higher YM
- Near-stable: intermediate behavior
**Useful for:** Design targeting specific stability regimes

### 5. predictive_distributions.png
**What it shows:** Posterior predictive distributions for 5 test alloys
**Look for:**
- Red line: observed value (if available)
- Green line: predicted mean
- Distribution width: uncertainty
**Useful for:** Assessing prediction calibration

### 6. property_tradeoffs.png
**What it shows:** Pairwise scatter plots colored by stability class
**Look for:**
- YM vs YS: low modulus + high strength (biomedical target)
- YS vs DAR: strength-ductility trade-off
- Pareto frontiers for multi-objective design
**Useful for:** Identifying optimal alloy regions

### 7. observed_vs_predicted.png
**What it shows:** Calibration plots with 95% credible intervals
**Look for:**
- Points near 1:1 line: good prediction
- Error bars covering 1:1 line: well-calibrated
**Useful for:** Model validation

## 🔍 Diagnostic Descriptions

### 8. trace_plots.png
**What it shows:** MCMC chains over iterations for key parameters
**Look for:**
- "Hairy caterpillar": good mixing
- No trends or drift
- Stable after burn-in
**Bad signs:** Trends, jumps, getting stuck
**Action if bad:** Increase n_iter, check initialization

### 9. autocorrelation_plots.png
**What it shows:** How correlated successive MCMC samples are
**Look for:**
- ACF decaying to ~0 within 50 lags
- Faster decay = more independent samples
**Bad signs:** Slow decay (high autocorrelation)
**Action if bad:** Increase thinning, check sampler efficiency

### 10. rhat_diagnostics.png
**What it shows:** Gelman-Rubin R-hat convergence diagnostic
**Look for:**
- R-hat < 1.1 (green bars): converged
- R-hat close to 1.0: excellent
**Bad signs:** R-hat > 1.1 (orange bars)
**Action if bad:** Run longer chains, increase burn-in

## 📋 Table Descriptions

### posterior_parameter_summary.csv
**Contains:** Posterior summaries for hyperparameters
**Key values:**
- `tau^2`: Global shrinkage (how much classes differ)
  - Low (~0.1-0.3): classes similar
  - High (~0.5-1.0): classes distinct
- `Sigma[k,k]`: Residual variance for property k
  - Smaller = tighter fit
- `M[intercept,k]`: Global baseline for property k (log-scale)

### predictive_performance.csv
**Contains:** Model fit metrics per property
**Key metrics:**
- `RMSE`: Root mean squared error (lower better)
  - Compare to: naive mean prediction
- `95% CI Coverage`: Fraction of observations in CI
  - Target: ~0.95 (calibrated)
  - <0.90: intervals too narrow (overconfident)
  - >0.98: intervals too wide (underconfident)

### class_specific_intercepts.csv
**Contains:** Baseline property values by stability class
**Key insights:**
- Compare intercepts across classes
- Identify class-specific patterns
- Guide design for target stability regime

## 🎨 What Good Results Look Like

### Converged MCMC:
✓ Trace plots: stable horizontal bands
✓ ACF plots: decay to zero by lag 50
✓ R-hat: all values < 1.1

### Well-Calibrated Model:
✓ 95% CI coverage: 0.92-0.96 for all properties
✓ Observed vs predicted: points near 1:1 line
✓ Error bars include 1:1 line

### Physically Sensible:
✓ Sigma: positive correlations between strength measures
✓ Class intercepts: metastable < near < stable for YM
✓ Coefficient signs: Mo-eq increases stability (lowers YM)

## 🚨 Common Issues and Solutions

**Issue:** R-hat > 1.1
**Solution:** Increase `n_iter` to 10000, `n_burn` to 2000

**Issue:** Coverage < 0.85
**Solution:** Check measurement errors, increase Sigma prior scale

**Issue:** Very wide credible intervals
**Solution:** More data needed, or increase prior informativeness

**Issue:** Trace plots show trends
**Solution:** Check for numerical issues, increase burn-in

## 🔄 Iteration Plan

1. **First run:** Use default settings (5000 iter)
2. **Check diagnostics:** Look at R-hat, traces, ACF
3. **If not converged:** Double iterations, rerun
4. **Validate:** Check coverage, observed vs predicted
5. **Refine:** Adjust priors if needed based on domain knowledge

## 📝 Notes for Paper

When reporting results:
- Report effective sample sizes from console output
- Include trace plots in supplementary material
- Show observed vs predicted for main text
- Report coverage alongside RMSE
- Discuss class-specific patterns from intercept table

## 🎓 Learning from Output

**Sigma matrix:** Reveals intrinsic property correlations
**Lambda matrix:** Shows how covariates affect properties similarly
**M coefficients:** Quantifies element effects on properties
**Class intercepts:** Identifies stability-property relationships
**Predictive distributions:** Enables probabilistic alloy design

---

**Key Principle:** All outputs work together to tell a coherent story about Ti-alloy property landscapes informed by physics and data!
