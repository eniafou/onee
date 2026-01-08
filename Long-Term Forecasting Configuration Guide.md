# Long-Term Forecasting Configuration Guide

## Table of Contents
1. [Introduction](#introduction)
2. [LTF Overview: Model Types](#ltf-overview-model-types)
3. [Gaussian Process Model](#gaussian-process-model)
4. [Intensity Forecast Wrapper](#intensity-forecast-wrapper)
5. [Prior Strategies](#prior-strategies)
6. [Feature Configuration](#feature-configuration)
7. [Running the Forecast](#running-the-forecast)

---

## Introduction

Long-Term Forecasting (LTF) provides **5-year ahead** electricity consumption predictions using **Gaussian Process (GP) regression** combined with deterministic trend priors. The system supports multiple prior strategies and can operate on SRM or CD.

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Time Horizon** | 5 years ahead (annual predictions) |
| **Methodology** | Gaussian Process + Trend Priors |
| **Validation** | Expanding Window Cross-Validation (Walk-Forward) |
| **Model Types** | GaussianProcessForecastModel, IntensityForecastWrapper |
| **Output** | Annual predictions with confidence intervals |

---

## LTF Overview: Model Types

The LTF system uses **two model types**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL COMPARISON                              │
├─────────────────┬───────────────────────────────────────────────┤
│ GaussianProcess │ Direct Annual Prediction                      │
│ ForecastModel   │ • GP regression on annual consumption         │
│                 │ • Combines trend prior + GP residuals         │
│                 │ • Confidence intervals from GP uncertainty    │
│                 │ • Used for SRM data                │
├─────────────────┼───────────────────────────────────────────────┤
│ Intensity       │ Normalized Prediction (CD-specific)           │
│ ForecastWrapper │ • Normalizes consumption by driver variable   │
│                 │ • Predicts consumption per unit (intensity)   │
│                 │ • Scales back using forecasted driver values  │
│                 │ • Used for CD data     │
└─────────────────┴───────────────────────────────────────────────┘
```

### Model Selection Logic

```
1. Run expanding window CV grid search across all configurations
   ↓
2. R² Threshold Filter (default: 0.6)
   ↓
3. Among passing models: Select lowest MAPE
   ↓
4. If no model passes: Fallback to highest R²
   ↓
5. Train selected model on full history
   ↓
6. Forecast horizon years with confidence intervals
```

---

## Gaussian Process Model

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│              GAUSSIAN PROCESS WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Compute Annual Consumption
───────────────────────────────────
For each year:
    Annual_t = Sum of 12 monthly values

STEP 2: Optional Log Transform
───────────────────────────────
If use_log_transform = true:
    y_transformed = log(Annual_t)

STEP 3: Fit Deterministic Trend Prior
──────────────────────────────────────
Prior strategies (e.g., LinearGrowthPrior):
    • Fit trend line on historical data
    • Can enforce minimum growth rate
    • Anchors to recent observations

STEP 4: Compute Residuals
──────────────────────────
residuals = y_transformed - prior.predict(years)

STEP 5: Build Feature Matrix
─────────────────────────────
X features can include:
    • Transformed target values (level, lchg)
    • Lagged values (lag, lag_lchg)
    • Economic features (PIB, GDP sectors)
    • Client counts (for SRM)

STEP 6: Train Gaussian Process on Residuals
────────────────────────────────────────────
GP.fit(X, residuals)
    • Kernel: RBF, Matern, etc.
    • Hyperparameters optimized via MLE

STEP 7: Forecast Horizon Years
───────────────────────────────
For each future year:
    1. Build feature vector X_future
    2. prior_t = prior.predict(year_t, X_future)
    3. residual_t, std_t = GP.predict(X_future)
    4. y_pred_t = prior_t + residual_t
    5. If log transformed: y_pred_t = exp(y_pred_t)
    6. Confidence interval: y_pred_t ± (1.96 * std_t)

STEP 8: Cross-Validation (Expanding Window)
────────────────────────────────────────────
For each test year (starting from year 3):
    • Train on all PRIOR years only (respects temporal order)
    • Predict test year
    • Compute metrics (R², MAPE, etc.)
    • Prevents future data leakage
```

### Configuration Parameters

#### 1. Kernel Selection (`kernel_key`)
```yaml
- model_type: GaussianProcessForecastModel
  kernel_key: ["rbf_white"]
```

Available kernels:
- `rbf_white`: RBF + White noise (smooth, medium-term patterns)
- `matern_white`: Matern(0.5) + White noise (rough, short-term patterns)
- `matern_short`: Short length scale Matern (very local patterns)
- `rbf_dot_white`: RBF + DotProduct + White (linear + smooth)
- `matern_smooth`: Matern(2.5) + White (very smooth)
- `rbf_long`: Long length scale RBF (long-term trends)
- `rational_quadratic`: Rational quadratic kernel

#### 2. GP Hyperparameters
```yaml
n_restarts_optimizer: [10]     # Optimization restarts for kernel params
normalize_y: [true, false]     # Normalize target before fitting
alpha: [1e-10]                 # Noise regularization term
```

#### 3. Transform Options
```yaml
use_log_transform: [true, false]  # Log transform annual values
```

**When to use log transform:**
- `true`: For exponential growth patterns (typical for SRM)
- `false`: For linear/intensity-based patterns (typical for CD)

#### 4. Outlier Detection
```yaml
remove_outliers: [False]
outlier_threshold: [2.5]  # Standard deviations from median
```

---

## Intensity Forecast Wrapper

### How It Works

The Intensity Forecast Wrapper **normalizes consumption by a driver variable** (e.g., number of contracts) before prediction.

```
┌─────────────────────────────────────────────────────────────────┐
│           INTENSITY FORECAST WORKFLOW                            │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Normalize Consumption
──────────────────────────────
intensity_t = consumption_t / normalization_variable_t

Example (CD):
    intensity_t = kwh_t / total_active_contrats_t
    (Consumption per contract)

STEP 2: Fit GP Model on Intensity
──────────────────────────────────
• Same GP workflow as above
• Target = intensity (not absolute consumption)
• Prior strategies apply to intensity trend

STEP 3: Forecast Intensity
───────────────────────────
For each future year:
    intensity_pred_t = GP.predict(...)
    
STEP 4: Scale Back to Consumption
──────────────────────────────────
For each future year:
    consumption_pred_t = intensity_pred_t × normalization_variable_t
    
    (Uses forecasted normalization values from df_features)
```

### Configuration Parameters

#### 1. Normalization Column
```yaml
- model_type: IntensityForecastWrapper
  normalization_col: [total_active_contrats]
```

Specifies the column to normalize by. Must be present in `df_features` with forecasted values.

#### 2. All GP Parameters Apply
All parameters from GaussianProcessForecastModel apply to the internal GP model.

---

## Prior Strategies

Priors provide the **deterministic trend component** of the forecast. The GP models deviations from this trend.

### Available Prior Types

#### 1. LinearGrowthPrior
```yaml
prior_config:
  - type: LinearGrowthPrior
    min_annual_growth: 0.02  # Minimum 2% annual growth
    anchor_window: 3         # Anchor to last 3 years
```

**Equation:** `y = m*t + b`

**Parameters:**
- `min_annual_growth`: Enforces minimum slope (growth floor). `null` = no constraint
- `anchor_window`: Number of recent years to anchor intercept. `null` = use all years

**Best for:** Steady linear growth patterns

---

#### 2. PowerGrowthPrior
```yaml
prior_config:
  - type: PowerGrowthPrior
    power: 0.5              # 1.0=linear, 0.5=sqrt, 0.1≈log
    anchor_window: 3
    min_annual_growth: 0.02
```

**Equation:** `y = a * t^power + b`

**Parameters:**
- `power`: Controls concavity (0 < p ≤ 1)
  - `1.0`: Linear (no slowing)
  - `0.5`: Square root (moderate slowing) - **recommended**
  - `0.1`: Near-logarithmic (fast slowing)
- `anchor_window`: Recent years to anchor
- `min_annual_growth`: Growth floor

**Best for:** Slowing growth patterns (maturing markets)

---

#### 3. FlatPrior
```yaml
prior_config:
  - type: FlatPrior
    method: mean           # Options: mean, median, last_value
    anchor_window: 3
```

**Equation:** `y = constant`

**Parameters:**
- `method`: How to compute baseline
  - `mean`: Average of anchor window
  - `median`: Median of anchor window
  - `last_value`: Most recent value
- `anchor_window`: Number of recent years

**Best for:** Stable/saturated markets with no trend

---

#### 4. NeutralPrior
```yaml
prior_config:
  - type: NeutralPrior
```

**Equation:** `y = 0` (always)

No parameters. Lets the GP model the entire signal without prior assumptions.

**Best for:** When you want pure GP behavior

---

#### 5. AugmentedConsensusPrior
```yaml
prior_config:
  - type: AugmentedConsensusPrior
    power: 0.5
    anchor_window: 1
    min_annual_growth: null
    exog_col_idx: 0          # Index of driver in X_exog
    driver_weight: 0.0       # 0.0=pure history, 1.0=pure driver
    memory_decay: 0.2        # Weight recent years more
    use_dynamic_weights: False
```

**Extends PowerGrowthPrior** by adjusting slope based on exogenous driver trajectory.

**Parameters:**
- `power`, `anchor_window`, `min_annual_growth`: Same as PowerGrowthPrior
- `exog_col_idx`: Which feature column to use as driver
- `driver_weight`: Blend between historical slope and driver-implied slope (0-1)
- `memory_decay`: Exponential decay for weighting recent observations (0 < decay ≤ 1)
  - `1.0`: Equal weight to all years
  - `< 1.0`: Higher weight to recent years
- `use_dynamic_weights`: Experimental feature

**Best for:** Growth driven by specific economic indicators

---

## Feature Configuration

### Transform Options

```yaml
features:
  transforms:
    - [level]              # Raw values
    - [level, lag]         # Raw + lagged values
    - [lchg]               # Log-changes (growth rates)
    - [lchg, lag_lchg]     # Log-changes + lagged log-changes
```

**Transform types:**
- `level`: Raw feature values
- `lag`: Lagged feature values (controlled by `lags` parameter)
- `lchg`: Log-change: `log(x_t / x_{t-1})`
- `lag_lchg`: Lagged log-changes

### Lag Configuration

```yaml
features:
  lags:
    - [1, 2]  # Use t-1 and t-2 lags
    - [1]     # Use only t-1 lag
```

Creates lagged versions of features for autoregressive patterns.

### Feature Blocks

```yaml
features:
  feature_block:
    - []                    # No economic features
    - [pib_mdh]            # GDP only
    - [gdp_primaire, gdp_secondaire, gdp_tertiaire]  # Sector GDPs
```

Economic/exogenous features to include. The system tries all combinations with transforms.

### Additional Features

```yaml
features:
  use_pf: [true, false]        # Puissance facturée (CD only)
  use_clients: [true, false]   # Client counts (SRM only)
  training_window: [null]      # null=all years, N=last N years
```

---

## Running the Forecast

### Command Line Interface

#### For SRM (Regional):
```bash
python run_ltf_srm.py
```

#### For CD (Contracts/Distributeurs):
```bash
python run_ltf_cd.py
```

### Output Structure

```
outputs/outputs_horizon/        # SRM outputs
outputs/outputs_horizon_cd/     # CD outputs
├── {region}_forecast_plot.png  # Visualization with confidence bands
├── ltf_srm_results.csv         # Main results file
└── ... (per-region outputs)
```

### Results CSV Columns

- `Region`: Entity name
- `Train_Start`, `Train_End`: Training period
- `Level`: Aggregation level
- `Year`: Forecast year
- `Predicted_Annual`: Predicted consumption
- `Actual_Annual`: Actual value (for CV years)
- `Percent_Error`: Percentage error
- Model configuration parameters (flattened)