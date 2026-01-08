# Short-Term Forecasting Configuration Guide

## Table of Contents
1. [Introduction](#introduction)
2. [STF Overview: The Four Strategies](#stf-overview-the-four-strategies)
3. [Strategy 1: PC-Based Growth Modeling](#strategy-1-pc-based-growth-modeling)
4. [Strategy 2: Direct Annual Growth](#strategy-2-direct-annual-growth)
5. [Strategy 3: Hybrid Approach](#strategy-3-hybrid-approach)
6. [Strategy 4: Advanced Growth Rate Model](#strategy-4-advanced-growth-rate-model)
7. [Strategy 5: Ensemble Integration](#strategy-5-ensemble-integration)
8. [Feature Engineering for STF](#feature-engineering-for-stf)
9.  [Model Hyperparameters](#model-hyperparameters)
10. [Loss Configuration](#loss-configuration)
11. [Evaluation Methodology](#evaluation-methodology)

---

## Introduction

Short-Term Forecasting (STF) provides **1-year ahead** electricity consumption predictions using **Functional Principal Component Analysis (FPCA)** combined with Ridge Regression. The system evaluates **four complementary strategies** plus an **ensemble** to capture different aspects of consumption patterns.

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Time Horizon** | 1 year ahead (12 monthly predictions) |
| **Methodology** | FPCA + Ridge Regression |
| **Validation** | Leave-One-Out Cross-Validation (LOOCV) |
| **Strategies** | 4 core + 1 ensemble |
| **Output** | Monthly predictions with confidence metrics |

---

## STF Overview: The Four Strategies

The STF system uses **four complementary strategies** that capture different aspects of consumption dynamics:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY COMPARISON                           │
├─────────────────┬───────────────────────────────────────────────┤
│ STRATEGY 1      │ PC-Based Monthly Prediction                   │
│                 │ • Predicts monthly values directly via PCA    │
│                 │ • Ridge regression on PC scores               │
│                 │ • Reconstructs curve from predicted PCs       │
├─────────────────┼───────────────────────────────────────────────┤
│ STRATEGY 2      │ Annual Prediction + Distribution              │
│                 │ • Predicts total annual consumption           │
│                 │ • Ridge regression on annual values           │
│                 │ • Distributes using weighted monthly pattern  │
├─────────────────┼───────────────────────────────────────────────┤
│ STRATEGY 3      │ Hybrid Pattern + Annual Prediction            │
│                 │ • Predicts PC-based shape + historical pattern│
│                 │ • Blends patterns with pc_weight              │
│                 │ • Scales to separately predicted annual total │
├─────────────────┼───────────────────────────────────────────────┤
│ STRATEGY 4      │ Mean-Reverting Growth Rate Model              │
│                 │ • Predicts annual growth rate                 │
│                 │ • Ridge regression on growth with transforms  │
│                 │ • Distributes using weighted monthly pattern  │
├─────────────────┼───────────────────────────────────────────────┤
│ STRATEGY 5      │ Ensemble (Meta-Strategy)                      │
│                 │ • Combines predictions from all strategies    │
│                 │ • Weighted by cross-validation performance    │
│                 │ • Applies bias correction via pinball loss    │
└─────────────────┴───────────────────────────────────────────────┘
```

### Strategy Selection Logic

The system evaluates all strategies and selects the best performer based on:

```
1. R² Threshold Filter (default: 0.6)
   ↓
2. Among passing models: Select lowest MAPE
   ↓
3. If no model passes: Fallback to highest R²
   ↓
4. Ensemble always considered as candidate
```

The system compares all strategies plus ensemble and selects the best 
performer based on cross-validated metrics.

---

## Strategy 1: PC-Based Monthly Prediction

### How It Works

Strategy 1 **predicts monthly consumption directly** by decomposing historical monthly curves into principal components, then using Ridge regression to predict the next year's PC scores.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 1 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Build Monthly Consumption Matrix
─────────────────────────────────────────
    Rows = Years, Columns = 12 Months
    ┌─────────────────────────────────────┐
    │  Jan  Feb  Mar ... Oct  Nov  Dec    │
    ├─────────────────────────────────────┤
2013│  100  105  110 ... 140  145  150    │
2014│  110  115  120 ... 150  155  160    │
2015│  120  125  130 ... 160  165  170    │
...    ...  ...  ... ... ...  ...  ...    │
2022│  180  185  190 ... 220  225  230    │
    └─────────────────────────────────────┘

STEP 2: Apply Weighted PCA on Training Years
─────────────────────────────────────────────────
• Fit PCA on the matrix (optionally with temporal weighting via pca_lambda)
• Extract top n_pcs components (e.g., 3)
• Each PC captures a pattern:  
  - PC1 = overall consumption level
  - PC2 = seasonal variation
  - PC3 = secondary patterns
• Recent years can receive higher weight via pca_lambda parameter

STEP 3: Compute PC Scores for Each Year
────────────────────────────────────────────
For each training year:
    PC_scores_t = [PC1_score_t, PC2_score_t, ..., PCn_score_t]

STEP 4: Build Features for Ridge Regression
────────────────────────────────────────────────
Target: PC_scores_t (each component predicted separately)
Features:
    • Lagged PC scores: PC_scores_{t-1}, PC_scores_{t-2}, ...
    • Optional: Economic features (GDP, sectoral GDP)
    • Optional: Monthly temperature values (12 features)
    • Optional: Monthly client counts (12 features)
    • Optional: Puissance facturée

STEP 5: Train Ridge Regression Models
──────────────────────────────────────
For each PC component:
    Model_i: PC_i_t = Ridge(PC_scores_{t-1, t-2, ...}, features)
    Regularization: alpha parameter

STEP 6: Predict Next Year's PC Scores
──────────────────────────────────────
1. Gather lagged PC scores from most recent years
2. Add external features for target year
3. Predict: PC_scores_{T+1} = [Model_1.predict(...), Model_2.predict(...), ...]

STEP 7: Reconstruct Monthly Curve
──────────────────────────────────
Monthly_curve_{T+1} = PCA.inverse_transform(PC_scores_{T+1})
    → This gives 12 monthly values directly

STEP 8: Cross-Validation (LOOCV)
─────────────────────────────────
For each test year:
    • Refit PCA on training years only (ultra-strict, no leakage)
    • Train Ridge models on training years
    • Predict test year
    • Compute metrics
```

### Configuration Parameters

#### 1. Number of PCs (`n_pcs`)
```yaml
model:
  n_pcs: 3  # Number of principal components to retain
```

Controls how many PC patterns to use. More components = more detail but risk of overfitting.

- Low (1-2): Only major patterns
- Medium (3): Typical choice  
- High (5+): Captures fine details, may overfit

#### 2. Lags (`lags_options`)
```yaml
model:
  lags_options: [1, 2]  # How many years back to look for PC scores
```

Determines lag depth for PC scores in the regression.

```python
# lag=1: Uses only previous year's PC scores as features
X = [PC1_{t-1}, PC2_{t-1}, PC3_{t-1}]

# lag=2: Uses previous two years
X = [PC1_{t-1}, PC2_{t-1}, PC3_{t-1}, PC1_{t-2}, PC2_{t-2}, PC3_{t-2}]
```

#### 3. Ridge Alpha (`alphas`)
```yaml
model:
  alphas: [0.01, 0.1, 1.0, 10.0]  # Regularization strength
```

Ridge regularization parameter. Higher = more smoothing, less overfitting.

#### 4. PCA Lambda (`pca_lambdas`)
```yaml
model:
  pca_lambdas: [0.3, 0.7, 1.0]  # Weighting parameter for recent years
```

Controls temporal weighting when fitting PCA. Higher lambda = more weight to recent years.

- `1.0`: Equal weight to all years
- `0.3-0.7`: Progressively higher weight to recent years

**Formula**: `weight_year = lambda^(distance_from_most_recent)`

#### 5. Feature Configuration

```yaml
features:
  use_monthly_temp_options: [true, false]     # 12 monthly temperature features
  use_monthly_clients_options: [true, false]  # 12 monthly client counts
  use_pf_options: [true, false]               # Puissance facturée (CD only)
  
  feature_blocks:                             # Economic features
    none: []
    gdp_only: [pib_mdh]
    sectoral: [gdp_primaire, gdp_secondaire, gdp_tertiaire]
```

---

## Strategy 2: Annual Prediction with Mean-Curve Distribution

### How It Works

Strategy 2 **predicts the total annual consumption** using Ridge regression on past annual values and external features, then distributes this total across months using a weighted average of historical monthly patterns.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 2 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Compute Annual Consumption
───────────────────────────────────
For each year:
    Annual_t = Sum of 12 monthly values

Example:
    2013 Total: 10,000 kWh
    2014 Total: 10,500 kWh
    2015 Total: 11,000 kWh

STEP 2: Build Feature Matrix for Annual Prediction
────────────────────────────────────────────────────
Target: Annual_t
Features:
    • Lagged annual values: Annual_{t-1}, Annual_{t-2}, ...
    • Optional: Economic features (GDP, sectoral GDP)
    • Optional: Annual client count
    • Optional: Puissance facturée

STEP 3: Train Ridge Regression
───────────────────────────────
Model: Annual_t = Ridge(Annual_{t-1, t-2, ...}, features)
    Regularization: alpha parameter

STEP 4: Predict Next Year's Annual Total
─────────────────────────────────────────
1. Gather lagged annual values from recent years
2. Add external features for target year
3. Predict: Annual_{T+1} = Model.predict(...)

STEP 5: Compute Monthly Distribution Pattern
─────────────────────────────────────────────────
Use weighted average of historical monthly patterns:
    • For each training year: normalize to fractions (month_m / annual_total)
    • Weight recent years higher (using pca_lambda parameter)
    • Compute weighted average pattern: [frac_Jan, frac_Feb, ..., frac_Dec]

Optional client-based weighting:
    If use_monthly_clients and client_pattern_weight:
        • Compute R² between client counts and consumption
        • Blend: pattern = w*client_pattern + (1-w)*historical_pattern
          where w = R²^client_pattern_weight

STEP 6: Distribute Annual Total to Monthly Values
──────────────────────────────────────────────────
For each month m:
    Predicted_monthly_m = Annual_{T+1} × frac_m

STEP 7: Cross-Validation (LOOCV)
─────────────────────────────────
For each test year:
    • Train Ridge on other years only
    • Compute distribution pattern from training years only
    • Predict test year
    • Compute metrics
```

### Configuration Parameters

#### 1. Lags (`lags_options`)
```yaml
model:
  lags_options: [1, 2]  # How many years back for annual values
```

Number of lagged annual consumption values to use as features.

#### 2. Ridge Alpha (`alphas`)
```yaml
model:
  alphas: [0.01, 0.1, 1.0, 10.0]
```

Regularization parameter for Ridge regression.

#### 3. Feature Blocks
```yaml
features:
  feature_blocks:
    none: []                          # Only lagged annual values
    gdp_only: [pib_mdh]              # Add total GDP
    sectoral: [gdp_primaire, gdp_secondaire, gdp_tertiaire]  # Sectoral GDP
    gdp_sectoral: [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire]  # All
```

Economic features to include in the regression. The system tries all combinations.

#### 4. Client Pattern Weight (`client_pattern_weights`)
```yaml
model:
  client_pattern_weights: [0.3, 0.5, 0.8]  # Only used if use_monthly_clients=true
```

Controls blending between historical pattern and client-based pattern when distributing monthly values.

**Formula**:
```python
R² = correlation²(client_counts, consumption)
w = R² ^ client_pattern_weight
final_pattern = w * client_pattern + (1-w) * historical_pattern
```

- Higher value = trust client correlation more
- Only applies when `use_monthly_clients_options: [true]`

#### 5. Feature Configuration
```yaml
features:
  use_monthly_clients_options: [true, false]  # Client counts for pattern
  use_pf_options: [true, false]               # Puissance facturée (CD only)
```

---

## Strategy 3: Hybrid Pattern Reconstruction + Annual Prediction

### How It Works

Strategy 3 **combines two approaches**: it predicts the shape of next year's monthly curve using PC reconstruction (like Strategy 1) and blends it with a weighted average of past yearly patterns. Separately, it predicts the annual total using Ridge regression, then scales the blended monthly pattern to match this total.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 3 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Predict Next Year's PC Scores
──────────────────────────────────────
• Fit weighted PCA on training years (recent years weighted higher via pca_lambda)
• Extract PC scores for all training years
• Train Ridge regression: PC_scores_t = f(PC_scores_{t-1, t-2, ...}, features)
• Predict: PC_scores_{T+1}

STEP 2: Reconstruct Monthly Pattern from PCs
─────────────────────────────────────────────
PC_pattern_{T+1} = PCA.inverse_transform(PC_scores_{T+1})
    → 12 monthly values (shape only, not scaled)

STEP 3: Compute Weighted Historical Pattern
────────────────────────────────────────────
• For each training year: compute normalized monthly pattern (month / annual_total)
• Weight recent years more heavily (using pca_lambda)
• Weighted_avg_pattern = weighted average of normalized patterns

STEP 4: Blend the Two Patterns
───────────────────────────────
Normalize PC_pattern: PC_pattern_norm = PC_pattern / sum(PC_pattern)

Blended_pattern = (pc_weight * PC_pattern_norm) + 
                  ((1 - pc_weight) * Weighted_avg_pattern)

where pc_weight ∈ [0, 1] (hyperparameter)

STEP 5: Predict Annual Total
─────────────────────────────
• Train separate Ridge regression: Annual_t = f(Annual_{t-1, t-2, ...}, features)
• Predict: Annual_{T+1}

STEP 6: Scale Blended Pattern to Annual Total
──────────────────────────────────────────────
For each month m:
    Predicted_monthly_m = Annual_{T+1} × Blended_pattern[m]

STEP 7: Optional Client-Based Adjustment
─────────────────────────────────────────
If use_monthly_clients and client_pattern_weight:
    • Compute R² between client counts and consumption
    • Create client_pattern from client count evolution
    • w = R² ^ client_pattern_weight
    • Final_pattern = w*client_pattern + (1-w)*Blended_pattern

STEP 8: Cross-Validation (LOOCV)
─────────────────────────────────
For each test year:
    • Refit PCA on training years only
    • Train PC Ridge models
    • Train annual Ridge model
    • Predict and blend patterns
    • Compute metrics
```

### Configuration Parameters

Strategy 3 combines parameters from Strategy 1 (for PC prediction) and Strategy 2 (for annual prediction).

#### 1. Number of PCs (`n_pcs`)
```yaml
model:
  n_pcs: 3
```

Number of principal components for shape prediction.

#### 2. Lags (`lags_options`)
```yaml
model:
  lags_options: [1, 2]
```

Lag depth for both PC scores and annual values.

#### 3. PC Weight (`pc_weights`)
```yaml
model:
  pc_weights: [0.5, 0.7]  # Blend between PC pattern and historical pattern
```

Controls blending of PC-reconstructed pattern vs. weighted historical pattern:
```python
Blended_pattern = pc_weight * PC_pattern_norm + (1 - pc_weight) * Historical_pattern
```

- `pc_weight = 1.0`: Use only PC reconstruction
- `pc_weight = 0.0`: Use only historical average
- `pc_weight = 0.5`: Equal blend

#### 4. PCA Lambda (`pca_lambdas`)
```yaml
model:
  pca_lambdas: [0.3, 0.7, 1.0]
```

Weights for temporal weighting in PCA and pattern averaging. Higher = more weight to recent years.

#### 5. Ridge Alpha (`alphas`)
```yaml
model:
  alphas: [0.01, 0.1, 1.0, 10.0]
```

Regularization for both PC prediction and annual prediction.

#### 6. Client Pattern Weight (`client_pattern_weights`)
```yaml
model:
  client_pattern_weights: [0.3, 0.5, 0.8]  # Only if use_monthly_clients=true
```

Controls additional blending with client-based pattern:
```python
R² = correlation²(client_counts, consumption)
w = R² ^ client_pattern_weight
Final_pattern = w * client_pattern + (1-w) * Blended_pattern
```

#### 7. Feature Configuration
```yaml
features:
  feature_blocks:
    none: []
    gdp_only: [pib_mdh]
    # etc.
  
  use_monthly_temp_options: [true, false]      # For PC prediction
  use_monthly_clients_options: [true, false]   # For PC prediction + pattern blending
  use_pf_options: [true, false]                # CD only
```

---

## Strategy 4: Mean-Reverting Growth Rate Model

### How It Works

Strategy 4 **models the annual growth rate** using a mean-reverting framework: growth rates tend to return toward a long-term average over time, influenced by past growth and external features. The predicted annual total is then distributed monthly using a weighted historical pattern.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 4 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Compute Annual Growth Rates
────────────────────────────────────
For each year t:
    Growth_t = (Annual_t - Annual_{t-1}) / Annual_{t-1}

Example:
    2013: 10,000 kWh → 2014: 10,500 kWh → Growth_2014 = 0.05 (5%)

STEP 2: Build Growth Rate Features
───────────────────────────────────
Target: Growth_t
Features:
    • Lagged growth: Growth_{t-1}, Growth_{t-2}, ...
    • Economic features with transforms:
        - level: Raw value (e.g., GDP_t)
        - lchg: Log-change → log(GDP_t / GDP_{t-1})
        - lag_lchg: Lagged log-change → log(GDP_{t-1} / GDP_{t-2})
    • Lagged economic features (via growth_feature_lags)
    • Optional: Annual client count change
    • Optional: Puissance facturée

STEP 3: Fit Mean-Reverting Growth Model
────────────────────────────────────────
Uses MeanRevertingGrowthModel (wraps a scikit-learn regressor):
    • Fits: Growth_t = f(Growth_{t-1, t-2, ...}, economic_features, ...)
    • Default regressor: Ridge (linear)
    • Mean-reversion: growth tends toward long-term average

Optional training_window parameter limits lookback:
    • training_window=10: Use only last 10 years for training
    • training_window=null: Use all available history

STEP 4: Predict Next Year's Growth Rate
────────────────────────────────────────
1. Gather lagged growth rates
2. Compute economic feature transforms (lchg, lag_lchg, etc.)
3. Predict: Growth_{T+1} = Model.predict(...)

STEP 5: Apply Growth to Get Annual Total
─────────────────────────────────────────
Annual_{T+1} = Annual_T × (1 + Growth_{T+1})

STEP 6: Compute Monthly Distribution Pattern
─────────────────────────────────────────────
• For each training year: normalize to fractions (month_m / annual_total)
• Weight recent years more (implicit in training window)
• Compute weighted average pattern

Optional client-based adjustment:
    If use_monthly_clients and client_pattern_weight:
        • Blend with client-based pattern (same as Strategy 2)

STEP 7: Distribute Annual Total to Monthly Values
──────────────────────────────────────────────────
For each month m:
    Predicted_monthly_m = Annual_{T+1} × frac_m

STEP 8: Cross-Validation (LOOCV)
─────────────────────────────────
For each test year:
    • Train growth model on other years (within training window)
    • Predict growth rate
    • Apply to previous year's total
    • Distribute using pattern from training years
    • Compute metrics
```

### Comparison with Strategy 2

| Aspect | Strategy 2 | Strategy 4 |
|--------|-----------|------------|
| **Target** | Annual level | Annual growth rate |
| **Model** | Ridge on levels | Ridge on growth rates |
| **Assumptions** | Direct level prediction | Mean-reverting growth |
| **Features** | Lagged levels + economics | Lagged growth + economics (with transforms) |
| **Distribution** | Same (weighted pattern) | Same (weighted pattern) |

### Configuration Parameters

#### 1. Feature Blocks
```yaml
features:
  feature_blocks:
    none: []                    # Only autoregressive (past growth)
    gdp_only: [pib_mdh]
    sectoral: [gdp_primaire, gdp_secondaire, gdp_tertiaire]
    gdp_sectoral: [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire]
```

Economic features to include in growth rate model.

#### 2. Growth Feature Transforms (`growth_feature_transforms`)
```yaml
features:
  growth_feature_transforms:
    - [lag_lchg]              # Lagged log-change only
    - [lchg, lag_lchg]        # Current + lagged log-change
    - [level, lchg]           # Level + change
```

How to transform economic features:

- `level`: Raw value (GDP_t)
- `lchg`: Log-change from previous year → log(GDP_t / GDP_{t-1})
- `lag_lchg`: Lagged log-change → log(GDP_{t-1} / GDP_{t-2})

**Typical for growth modeling**: Use `lag_lchg` to avoid lookahead bias (predict Growth_t using GDP_{t-1} change).

#### 3. Growth Feature Lags (`growth_feature_lags`)
```yaml
features:
  growth_feature_lags:
    - [1]        # Only t-1
    - [1, 2]     # t-1 and t-2
```

How many lags of economic features to include.

Example with GDP, transform=lag_lchg, lags=[1,2]:
```python
Features = [
    log(GDP_{t-1} / GDP_{t-2}),  # Recent GDP growth
    log(GDP_{t-2} / GDP_{t-3})   # Earlier GDP growth
]
```

#### 4. Training Windows (`training_windows`)
```yaml
model:
  training_windows: [4, 7, 10, null]  # Years of history to use
```

Limits lookback for training the growth model:

- `training_window=10`: Use only last 10 years
- `training_window=null`: Use all available history

Allows model to adapt to recent dynamics vs. long-term patterns.

#### 5. Client Pattern Weight (`client_pattern_weights`)
```yaml
model:
  client_pattern_weights: [0.3, 0.5, 0.8]  # Only if use_monthly_clients=true
```

Controls blending with client-based pattern when distributing monthly (same as Strategy 2).

#### 6. Feature Configuration
```yaml
features:
  use_monthly_clients_options: [true, false]  # Client counts for pattern
  use_pf_options: [true, false]               # Puissance facturée (CD only)
```

---

## Strategy 5: Ensemble Integration

### How It Works

Strategy 5 **combines predictions from multiple strategies**, weighting them by their cross-validated performance.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 5 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Run All Base Strategies
────────────────────────────────
For each strategy (1-4):
    • Run full LOOCV
    • Store cross-validation metrics (MAPE, R²)
    • Store results

STEP 2: Compute Strategy Weights
─────────────────────────────────
Based on cross-validated performance using pinball loss:

tau = under_estimation_penalty / (1 + under_estimation_penalty)

For each prediction year (LOOCV style):
    1. Use past years to compute weights
    2. Weight_i = based on past MAPE/R² performance
    3. Fit bias correction lambda via pinball loss on past years
    4. Combine: Ensemble = Σ(weight_i × Strategy_i) × (1 + lambda)

STEP 3: Evaluate Ensemble
──────────────────────────
Compute metrics for ensemble:
    • R²_ensemble
    • MAPE_ensemble
    • RMSE_ensemble
    • MAE_ensemble

STEP 4: Compare Against Base Strategies
────────────────────────────────────────
Ensemble competes with individual strategies
    → Selected if it has best R² and MAPE
```

### Configuration

Ensemble uses the configuration from all base strategies. Key consideration:

```yaml
loss:
  favor_overestimation: true          # Use pinball loss for ensemble
  under_estimation_penalty: 2         # tau = 2/(1+2) = 0.67
```

The `under_estimation_penalty` controls the asymmetric bias:
- Higher value = penalize underestimation more
- Used to compute tau for pinball loss
- Ensemble applies learned bias correction factor

---

## Model Hyperparameters

### Grid Search Strategy

STF uses **exhaustive grid search** over all hyperparameter combinations:

```python
for strategy in [1, 2, 3, 4]:
    for feature_block in feature_blocks:
        for lag in lags_options:
            for alpha in alphas:
                # ... other params specific to strategy
                train_and_evaluate()
```

**Total Configurations Example**:
```yaml
features:
  feature_blocks: {none: [], gdp: [pib_mdh]}     # 2 options
  use_monthly_clients_options: [true, false]     # 2 options

model:
  lags_options: [1, 2]                           # 2 options
  alphas: [0.1, 1.0]                             # 2 options
  pc_weights: [0.5, 0.8]                         # 2 options (Strategy 3)

# Strategy 1: 2 × 2 × 2 × 2 = 16 configurations
# Strategy 2: 2 × 2 × 2 × 2 = 16 configurations  
# Strategy 3: 2 × 2 × 2 × 2 × 2 = 32 configurations
# Strategy 4: 2 × 2 × 2 = 8 configurations
# Total: 72 model evaluations
```

---

### Hyperparameter Tuning Guidelines

#### Starting Configuration (Conservative)

```yaml
model:
  n_pcs: 3
  lags_options: [1, 2]
  alphas: [0.1, 1.0, 10.0]
  pc_weights: [0.5, 0.8]
  pca_lambdas: [0.3]
  training_windows: [null, 3]
  client_pattern_weights: [0.5]
  r2_threshold: 0.6
```

**Why This Works**:
- Moderate number of PCs
- Short memory (1-2 lags)
- Wide alpha range
- Few training window options
- Single client weight (if Strategy 3 not critical)

**Expected Runtime**: ~5-10 minutes per entity

#### Expanded Configuration (Thorough)

```yaml
model:
  n_pcs: 3
  lags_options: [1, 2, 3]
  alphas: [0.01, 0.1, 1.0, 10.0]
  pc_weights: [0.3, 0.5, 0.7, 0.9]
  pca_lambdas: [0.2, 0.3, 0.4]
  training_windows: [null, 3, 5, 7]
  client_pattern_weights: [0.3, 0.5, 0.8]
  r2_threshold: 0.6
```

**Why Use This**:
- When you need the absolute best model
- For critical forecasts (high stakes)
- When runtime is not a constraint

**Expected Runtime**: ~20-40 minutes per entity

#### Aggressive Configuration (Maximum Flexibility)

```yaml
model:
  n_pcs: 4  # More PCs
  lags_options: [1, 2, 3, 4]  # Longer memory
  alphas: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # Wide regularization range
  pc_weights: [0.2, 0.4, 0.6, 0.8, 1.0]  # Full spectrum
  pca_lambdas: [0.1, 0.2, 0.3, 0.5, 0.7]  # From smooth to rough
  training_windows: [null, 2, 3, 5, 7, 10]  # Many windows
  client_pattern_weights: [0.2, 0.4, 0.6, 0.8]
  r2_threshold: 0.5  # More lenient
```

**When to Use**:
- Research/experimentation phase
- Complex, non-stationary consumption
- When overfitting risk is low (lots of data)

**Warning**: Can take 1-2 hours per entity!

### Hyperparameter Impact Summary

| Parameter | Low Value | High Value | Sweet Spot |
|-----------|-----------|------------|------------|
| `n_pcs` | Captures only main trend | May overfit to noise | 3-4 |
| `lags_options` | Short memory | Long memory | 1-2 |
| `alphas` | Overfits training data | Underfits (too smooth) | 0.1-10.0 |
| `pc_weights` | Trusts historical average | Trusts PCs completely | 0.5-0.8 |
| `pca_lambdas` | Rough PCs (fits noise) | Very smooth PCs | 0.2-0.4 |
| `training_windows` | Recent data only | All history | 3-5 or null |
| `client_pattern_weights` | Ignores client signal | Fully client-driven | 0.3-0.7 |

---

## Loss Configuration

### Asymmetric Loss Function

The system supports **asymmetric penalties** to favor overestimation:

```yaml
loss:
  favor_overestimation: true
  under_estimation_penalty: 1.5
```

### How It Works

**Standard Loss** (favor_overestimation = false):
```python
MAE = mean(|actual - predicted|)
```

**Asymmetric Loss** (favor_overestimation = true):
```python
for each prediction:
    error = actual - predicted
    
    if error > 0:  # Underestimation
        weighted_error = error * under_estimation_penalty
    else:  # Overestimation
        weighted_error = error * 1.0
    
    total_loss += |weighted_error|
```

### Example

**Scenario**: Actual = 1000 kWh, Penalty = 1.5

| Prediction | Error | Standard Loss | Asymmetric Loss | Ratio |
|------------|-------|---------------|-----------------|-------|
| 1050 kWh   | -50   | 50            | 50              | 1.0x  |
| 950 kWh    | +50   | 50            | 75              | 1.5x  |
| 900 kWh    | +100  | 100           | 150             | 1.5x  |

**Effect**: Model is "punished" more for underestimating, leading to upward bias in predictions.

### Configuration Guidance

#### Neutral (No Bias)
```yaml
loss:
  favor_overestimation: false
  under_estimation_penalty: 1.0  # Ignored when false
```

#### Mild Overestimation Bias
```yaml
loss:
  favor_overestimation: true
  under_estimation_penalty: 1.2  # 20% higher cost for underestimation
```

#### Moderate Overestimation Bias (Recommended)
```yaml
loss:
  favor_overestimation: true
  under_estimation_penalty: 1.5  # 50% higher cost
```

#### Strong Overestimation Bias
```yaml
loss:
  favor_overestimation: true
  under_estimation_penalty: 2.0  # 2x cost for underestimation
```

#### Aggressive Overestimation Bias
```yaml
loss:
  favor_overestimation: true
  under_estimation_penalty: 3.0  # 3x cost (use with caution)
```

### When to Use Asymmetric Loss

✅ **Use Cases**:
- Capacity planning (better to have excess than shortage)
- Electricity supply (blackouts very costly)
- Budget planning with safety margins
- Infrastructure investment (underbuilding is expensive)

❌ **Avoid When**:
- Accurate point estimates are critical
- Overestimation has high costs (e.g., resource waste)
- Symmetric loss is business requirement
- Evaluating model accuracy objectively

---

## Evaluation Methodology

### Leave-One-Out Cross-Validation (LOOCV)

STF uses **strict LOOCV** to prevent overfitting and ensure robust evaluation.

#### The Process

```
┌─────────────────────────────────────────────────────────────────┐
│              LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)              │
└─────────────────────────────────────────────────────────────────┘

Evaluation Period: 2021-2023 (3 years)

FOLD 1: Hold out 2021
─────────────────────────────────────────────
Train: [2013, 2014, 2015, ..., 2020, 2022, 2023]
Test:  [2021]
    ↓
Predict 2021, compute metrics

FOLD 2: Hold out 2022
─────────────────────────────────────────────
Train: [2013, 2014, 2015, ..., 2020, 2021, 2023]
Test:  [2022]
    ↓
Predict 2022, compute metrics

FOLD 3: Hold out 2023
─────────────────────────────────────────────
Train: [2013, 2014, 2015, ..., 2020, 2021, 2022]
Test:  [2023]
    ↓
Predict 2023, compute metrics

AGGREGATE METRICS
─────────────────────────────────────────────
R² = aggregate R² across all folds
MAPE = mean of (|pred - actual| / actual) across all folds
MAE = mean of |pred - actual| across all folds
RMSE = sqrt(mean((pred - actual)²)) across all folds
```

#### Why LOOCV?

✅ **Advantages**:
- **No future data leakage**: Each fold strictly separates train/test
- **Maximum data usage**: Every year is used for both training and testing
- **Robust estimates**: Multiple test points reduce variance
- **Fair comparison**: All models evaluated on same folds

⚠️ **Considerations**:
- **Computationally expensive**: N folds for N years
- **May be pessimistic**: Each model misses one training year

### Metrics Explained

#### R² (Coefficient of Determination)
```
R² = 1 - (SS_residual / SS_total)
   = 1 - (Σ(y - ŷ)² / Σ(y - ȳ)²)
```

**Interpretation**:
- **1.0**: Perfect predictions
- **0.6-0.9**: Good to excellent (typical for STF)
- **0.3-0.6**: Moderate (may need improvement)
- **<0.3**: Poor (model not capturing patterns)
- **Negative**: Model worse than mean baseline

**Threshold**:
```yaml
model:
  r2_threshold: 0.6  # Minimum to be considered "good"
```

#### MAPE (Mean Absolute Percentage Error)
```
MAPE = (1/n) × Σ(|actual - predicted| / actual) × 100%
```

**Interpretation**:
- **<5%**: Excellent
- **5-10%**: Good
- **10-20%**: Moderate
- **>20%**: Poor

**Example**:
```
Actual: 1000 kWh
Predicted: 950 kWh
APE = |1000 - 950| / 1000 = 0.05 = 5%
```

#### MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|actual - predicted|
```

**Interpretation**: Average magnitude of errors in consumption units (kWh).

**Example**:
```
MAE = 50 kWh means predictions are off by 50 kWh on average
```

#### RMSE (Root Mean Squared Error)
```
RMSE = sqrt((1/n) × Σ(actual - predicted)²)
```

**Interpretation**: Similar to MAE but penalizes larger errors more heavily.

**Comparison**:
- If RMSE >> MAE: Predictions have large outliers
- If RMSE ≈ MAE: Errors are relatively uniform

---

## Practical Configuration Examples

### Example 1: Conservative CD Configuration

**Context**: Contract-level forecasting with reliable data, moderate volatility.

```yaml
# configs/stf_cd_conservative.yaml

project:
  project_root: .
  exp_name: exp_conservative

data:
  variable: consommation_kwh
  unit: Kwh
  regions: null
  run_levels: [1]
  db_path: data/all_data.db

evaluation:
  eval_years_start: 2021
  eval_years_end: 2023
  train_start_year: 2013
  training_end: null

features:
  feature_blocks:
    none: []
    gdp_only: [pib_mdh]
  
  growth_feature_transforms:
    - [lag_lchg]
  
  growth_feature_lags:
    - [1]
  
  use_monthly_clients_options: [false, true]
  use_pf_options: [true]

model:
  n_pcs: 3
  lags_options: [1, 2]
  alphas: [0.1, 1.0, 10.0]
  pc_weights: [0.5, 0.8]
  pca_lambdas: [0.3]
  training_windows: [null, 3]
  client_pattern_weights: [0.5]
  r2_threshold: 0.6

loss:
  favor_overestimation: true
  under_estimation_penalty: 1.5
  penalties_by_level: null
```

**Why This Configuration**:
- Limited feature exploration (2 blocks)
- Short memory (lags 1-2)
- Moderate alpha range
- Single smoothing parameter
- Two training windows
- Asymmetric loss for safe capacity planning

**Expected Performance**: R² > 0.7, MAPE < 10%

---

### Example 2: Aggressive SRM Configuration

**Context**: Regional forecasting with complex patterns, need best possible accuracy.

```yaml
# configs/stf_srm_aggressive.yaml

project:
  project_root: .
  exp_name: exp_aggressive

data:
  variable: consommation_kwh
  unit: Kwh
  regions:
    Casablanca-Settat: 0
    Fès-Meknès: 1
    Marrakech-Safi: 0
    Rabat-Salé-Kénitra: 0
  run_levels: [0, 1]
  db_path: data/all_data.db

evaluation:
  eval_years_start: 2021
  eval_years_end: 2023
  train_start_year: 2007  # More history for SRM
  training_end: null

features:
  feature_blocks:
    none: []
    gdp_only: [pib_mdh]
    sectoral_only: [gdp_primaire, gdp_secondaire, gdp_tertiaire]
    gdp_sectoral: [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire]
  
  growth_feature_transforms:
    - [lag_lchg]
    - [lchg, lag_lchg]
  
  growth_feature_lags:
    - [1]
    - [1, 2]
  
  use_monthly_clients_options: [true, false]
  use_pf_options: [false]

model:
  n_pcs: 3
  lags_options: [2, 3]
  alphas: [0.1, 1.0, 10.0]
  pc_weights: [0.2, 0.5, 0.8]
  pca_lambdas: [0.3, 0.7, 1.0]
  training_windows: [4, 7, 10]
  client_pattern_weights: [0.3, 0.5, 0.8]
  r2_threshold: 0.6

loss:
  favor_overestimation: true
  under_estimation_penalty: 1.5
  penalties_by_level: null
```

**Why This Configuration**:
- Full feature exploration (4 blocks × 2 transforms × 2 lags = 16 combinations)
- Longer memory (lags 2-3)
- Wide PC weight range
- Multiple smoothing parameters
- Several training windows
- Multiple client weights (Strategy 3 exploration)

**Expected Performance**: R² > 0.75, MAPE < 8%
**Expected Runtime**: 30-60 minutes per region

---

### Example 3: Research/Experimentation Configuration

**Context**: Exploring all possibilities, no time constraints.

```yaml
# configs/stf_research.yaml

project:
  project_root: .
  exp_name: exp_research

data:
  variable: consommation_kwh
  unit: Kwh
  regions: null
  run_levels: [1]
  db_path: data/all_data.db

evaluation:
  eval_years_start: 2020
  eval_years_end: 2023  # 4-year evaluation
  train_start_year: 2010
  training_end: null

features:
  feature_blocks:
    none: []
    gdp_only: [pib_mdh]
    sectoral_only: [gdp_primaire, gdp_secondaire, gdp_tertiaire]
    gdp_sectoral: [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire]
  
  growth_feature_transforms:
    - [level]
    - [lchg]
    - [lag_lchg]
    - [lchg, lag_lchg]
  
  growth_feature_lags:
    - [1]
    - [1, 2]
    - [1, 2, 3]
  
  use_monthly_clients_options: [true, false]
  use_pf_options: [true, false]

model:
  n_pcs: 4  # More PCs for research
  lags_options: [1, 2, 3, 4]
  alphas: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  pc_weights: [0.2, 0.4, 0.6, 0.8, 1.0]
  pca_lambdas: [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
  training_windows: [null, 2, 3, 5, 7, 10]
  client_pattern_weights: [0.1, 0.3, 0.5, 0.7, 0.9]
  r2_threshold: 0.5  # More lenient for exploration

loss:
  favor_overestimation: false  # Neutral for research
  under_estimation_penalty: 1.0
  penalties_by_level: null
```

**Why This Configuration**:
- **Exhaustive exploration**: Trying every reasonable combination
- **Multiple feature representations**: Level, changes, lagged changes
- **Wide hyperparameter ranges**: From very conservative to very aggressive
- **Long evaluation period**: 4 years of LOOCV
- **Neutral loss**: No bias for objective evaluation

**Expected Performance**: Will find the absolute best model
**Expected Runtime**: 2-4 hours per entity
**Use Case**: One-time analysis, research, benchmarking

---

## Troubleshooting & Best Practices

### Common Issues

#### Issue 1: All Strategies Fail R² Threshold

**Symptom**:
```
Warning: No model passes R² threshold (0.6).# Short-Term Forecasting Configuration Guide

## Table of Contents
1. [Introduction](#introduction)
2. [STF Overview: The Four Strategies](#stf-overview-the-four-strategies)
3. [Strategy 1: PC-Based Growth Modeling](#strategy-1-pc-based-growth-modeling)
4. [Strategy 2: Direct Annual Growth](#strategy-2-direct-annual-growth)
5. [Strategy 3: Hybrid Approach](#strategy-3-hybrid-approach)
6. [Strategy 4: Advanced Growth Rate Model](#strategy-4-advanced-growth-rate-model)
7. [Strategy 5: Ensemble Integration](#strategy-5-ensemble-integration)
8. [Configuration Deep Dive](#configuration-deep-dive)
9. [Feature Engineering for STF](#feature-engineering-for-stf)
10. [Model Hyperparameters](#model-hyperparameters)
11. [Loss Configuration](#loss-configuration)
12. [Evaluation Methodology](#evaluation-methodology)
13. [Practical Configuration Examples](#practical-configuration-examples)
14. [Troubleshooting & Best Practices](#troubleshooting--best-practices)

---

## Introduction

Short-Term Forecasting (STF) provides **1-year ahead** electricity consumption predictions using **Functional Principal Component Analysis (FPCA)** combined with Ridge Regression. The system evaluates **four complementary strategies** plus an **ensemble** to capture different aspects of consumption patterns.

### When to Use STF

✅ **Use STF When**:
- You need monthly granularity for next year
- You have 5+ years of historical data
- You want to capture seasonal patterns
- Budget planning or short-term capacity needs

❌ **Don't Use STF When**:
- You need multi-year strategic forecasts (use LTF)
- Historical data is sparse (<3 years)
- You only care about annual totals (consider LTF)

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Time Horizon** | 1 year ahead (12 monthly predictions) |
| **Methodology** | FPCA + Ridge Regression |
| **Validation** | Leave-One-Out Cross-Validation (LOOCV) |
| **Strategies** | 4 core + 1 ensemble |
| **Output** | Monthly predictions with confidence metrics |

---

## STF Overview: The Four Strategies

The STF system uses **four complementary strategies** that capture different aspects of consumption dynamics:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY COMPARISON                           │
├─────────────────┬───────────────────────────────────────────────┤
│ STRATEGY 1      │ PC-Based Growth Modeling                      │
│                 │ • Captures dominant consumption patterns      │
│                 │ • Uses Principal Components as features       │
│                 │ • Best for: Stable patterns                   │
├─────────────────┼───────────────────────────────────────────────┤
│ STRATEGY 2      │ Direct Annual Growth                          │
│                 │ • Models year-over-year changes directly      │
│                 │ • Predicts annual growth rates                │
│                 │ • Best for: Trending consumption              │
├─────────────────┼───────────────────────────────────────────────┤
│ STRATEGY 3      │ Hybrid Approach                               │
│                 │ • Combines patterns + client behavior         │
│                 │ • Weights historical vs. client influence     │
│                 │ • Best for: Client-driven variability         │
├─────────────────┼───────────────────────────────────────────────┤
│ STRATEGY 4      │ Advanced Growth Rate Model                    │
│                 │ • Uses Gaussian Process for growth           │
│                 │ • Incorporates economic features              │
│                 │ • Best for: Complex, non-linear trends        │
├─────────────────┼───────────────────────────────────────────────┤
│ STRATEGY 5      │ Ensemble (Meta-Strategy)                      │
│                 │ • Combines predictions from all strategies    │
│                 │ • Weighted by cross-validation performance    │
│                 │ • Best for: Robust, uncertainty-aware forecasts│
└─────────────────┴───────────────────────────────────────────────┘
```

### Strategy Selection Logic

The system evaluates all strategies and selects the best performer based on:

```
1. R² Threshold Filter (default: 0.6)
   ↓
2. Among passing models: Select lowest MAPE
   ↓
3. If no model passes: Fallback to highest R²
   ↓
4. Ensemble always considered as candidate
```

---

## Strategy 1: PC-Based Growth Modeling

### The Core Idea

Strategy 1 **decomposes historical consumption into Principal Components** (PCs) and uses them to predict future patterns.

**Intuition**: "Can we explain future consumption using the dominant patterns from the past?"

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 1 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Build Consumption Matrix
─────────────────────────────────
    Monthly Consumption (Columns = Months)
    ┌─────────────────────────────────────┐
    │       Jan  Feb  Mar ... Nov  Dec    │
    ├─────────────────────────────────────┤
2013│  100  105  110 ... 140  150         │
2014│  110  115  120 ... 150  160         │
2015│  120  125  130 ... 160  170         │
...    ...  ...  ... ... ...  ...         │
2022│  180  185  190 ... 220  230         │
    └─────────────────────────────────────┘

STEP 2: Compute Principal Components
─────────────────────────────────────────
Apply PCA on the matrix
    ↓
Extract top N components (e.g., n_pcs=3)
    ↓
Each PC captures a dominant pattern:
    • PC1: Overall trend (80% variance)
    • PC2: Seasonal oscillation (15% variance)
    • PC3: Winter peak (5% variance)

STEP 3: Compute Annual Growth Rates
────────────────────────────────────────
For each year:
    Growth = (Total_year_t - Total_year_{t-1}) / Total_year_{t-1}

STEP 4: Build Regression Model
───────────────────────────────────────
Target: Annual Growth Rate
Features:
    • PC scores from previous year (lagged)
    • Optional: Economic features (GDP, sectoral growth)
    • Optional: Lag of growth rate itself

Model: Ridge Regression
    ↓
Fit: Growth_t = f(PC1_{t-1}, PC2_{t-1}, ..., GDP_t, ...)

STEP 5: Predict Next Year
──────────────────────────────────────
1. Compute PCs for most recent year
2. Use regression model to predict growth rate
3. Apply predicted growth to last year's total
4. Distribute annual total across months using:
   - Historical seasonal pattern (weighted average)
   - PC reconstruction

STEP 6: Cross-Validation (LOOCV)
─────────────────────────────────────
For each year in evaluation period:
    • Train on all other years
    • Predict held-out year
    • Compute metrics
    ↓
Select best hyperparameter combination
```

### Key Hyperparameters

#### 1. Number of PCs (`n_pcs`)
```yaml
model:
  n_pcs: 3  # Number of principal components to retain
```

**What It Does**: Controls how many dominant patterns to capture.

**Intuition**:
- **Low (1-2)**: Captures only major trends, ignores nuances
- **Medium (3-5)**: Balances pattern capture and noise filtering
- **High (6+)**: Risks overfitting to noise

**Recommendation**:
```yaml
# Conservative (stable consumption)
n_pcs: 2

# Balanced (typical use case)
n_pcs: 3

# Complex patterns (volatile consumption)
n_pcs: 5
```

#### 2. Lags (`lags_options`)
```yaml
model:
  lags_options: [1, 2, 3]  # How many years back to look
```

**What It Does**: Determines how far back to include lagged PCs.

**Example**:
```python
# lag=1: Uses only previous year's PCs
X = [PC1_{t-1}, PC2_{t-1}, PC3_{t-1}]

# lag=2: Uses previous two years
X = [PC1_{t-1}, PC2_{t-1}, PC3_{t-1}, 
     PC1_{t-2}, PC2_{t-2}, PC3_{t-2}]
```

**Recommendation**:
```yaml
# Short memory (volatile patterns)
lags_options: [1]

# Medium memory (typical)
lags_options: [1, 2]

# Long memory (persistent trends)
lags_options: [1, 2, 3]
```

#### 3. Ridge Alpha (`alphas`)
```yaml
model:
  alphas: [0.01, 0.1, 1.0, 10.0]  # Regularization strength
```

**What It Does**: Controls regularization to prevent overfitting.

**Intuition**:
- **Low (0.01)**: Minimal regularization, fits training data closely
- **Medium (0.1-1.0)**: Balanced generalization
- **High (10.0+)**: Strong regularization, smoother predictions

**Recommendation**:
```yaml
# Small datasets (<10 years)
alphas: [1.0, 10.0, 100.0]

# Medium datasets (10-20 years)
alphas: [0.1, 1.0, 10.0]

# Large datasets (>20 years)
alphas: [0.01, 0.1, 1.0]
```

#### 4. PC Weights (`pc_weights`)
```yaml
model:
  pc_weights: [0.5, 0.8]  # Weight of PCs in pattern reconstruction
```

**What It Does**: Balances PC-based pattern vs. simple historical average.

**Formula**:
```python
monthly_pattern = (pc_weight * PC_reconstruction) + 
                  ((1 - pc_weight) * historical_average)
```

**Recommendation**:
```yaml
# Trust PCs more (stable patterns)
pc_weights: [0.7, 0.9]

# Balanced (typical)
pc_weights: [0.5, 0.7]

# Trust historical average more (noisy PCs)
pc_weights: [0.3, 0.5]
```

#### 5. PCA Lambda (`pca_lambdas`)
```yaml
model:
  pca_lambdas: [0.2, 0.3, 0.4]  # Smoothing parameter for PCA
```

**What It Does**: Applies smoothing penalty in functional PCA.

**Intuition**: Higher lambda = smoother principal components.

**Recommendation**:
```yaml
# Minimal smoothing (trust data)
pca_lambdas: [0.1, 0.2]

# Moderate smoothing (typical)
pca_lambdas: [0.2, 0.3, 0.4]

# Heavy smoothing (noisy data)
pca_lambdas: [0.5, 0.7, 1.0]
```

#### 6. Training Window (`training_windows`)
```yaml
model:
  training_windows: [1, 2, 3, 4]  # Years of training data to use
```

**What It Does**: Limits how far back to use training data.

**Options**:
- `null`: Use all available history
- `1, 2, 3...`: Use only last N years

**Example**:
```yaml
# Rolling 3-year window (adapts quickly)
training_windows: [3]

# Try multiple windows
training_windows: [null, 3, 5, 7]

# Use all history
training_windows: [null]
```

### When Strategy 1 Excels

✅ **Best For**:
- Stable consumption patterns with clear seasonality
- Data with low volatility
- When historical patterns are predictive of future

❌ **Struggles With**:
- Structural breaks (new contracts, policy changes)
- Highly volatile consumption
- Sparse historical data (<5 years)

---

## Strategy 2: Direct Annual Growth

### The Core Idea

Strategy 2 **directly models year-over-year growth rates** using economic and client features, then applies predicted growth to the previous year's consumption.

**Intuition**: "Can we predict how much consumption will grow based on economic indicators?"

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 2 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Compute Annual Growth Rates
────────────────────────────────────────
For each year t:
    Growth_t = (Total_t - Total_{t-1}) / Total_{t-1}

Example:
    2013 Total: 10,000 kWh
    2014 Total: 10,500 kWh
    Growth_2014 = (10,500 - 10,000) / 10,000 = 0.05 (5%)

STEP 2: Build Feature Matrix
─────────────────────────────────────
Features (for predicting Growth_t):
    • Lag of growth rate: Growth_{t-1}
    • Economic indicators: GDP_t, Sectoral_GDP_t
    • Client features: New_clients_t, Active_clients_t
    • Lagged features: GDP_{t-1}, Clients_{t-1}

Example Matrix:
    ┌──────────────────────────────────────────────┐
    │ Year │ Growth_{t-1} │ GDP_t │ Clients_t │ ... │
    ├──────────────────────────────────────────────┤
    │ 2014 │    0.03      │ 1020  │   500     │ ... │
    │ 2015 │    0.05      │ 1050  │   520     │ ... │
    │ ...  │    ...       │  ...  │   ...     │ ... │
    └──────────────────────────────────────────────┘

STEP 3: Fit Ridge Regression
─────────────────────────────────────
Target: Growth_t
Model: Ridge(alpha=...)
    ↓
Fit: Growth_t = β₀ + β₁*Growth_{t-1} + β₂*GDP_t + β₃*Clients_t + ...

STEP 4: Predict Next Year's Growth
───────────────────────────────────────
1. Gather features for year T+1:
   - Growth_T (from data)
   - GDP_{T+1} (from projections)
   - Clients_{T+1} (from projections)
   
2. Predict: Growth_{T+1} = Model.predict(features_{T+1})

STEP 5: Apply Growth to Previous Year
──────────────────────────────────────────
Annual_Total_{T+1} = Annual_Total_T × (1 + Growth_{T+1})

STEP 6: Distribute to Monthly Values
─────────────────────────────────────────
Use historical monthly pattern (averaged over last N years):
    Monthly_frac_m = Avg(Month_m_consumption / Annual_Total)
    
For each month m:
    Predicted_m = Annual_Total_{T+1} × Monthly_frac_m

STEP 7: Cross-Validation (LOOCV)
─────────────────────────────────────
For each year in evaluation period:
    • Train on all other years
    • Predict held-out year's growth
    • Compute metrics
```

### Key Configuration

#### Feature Blocks
```yaml
features:
  feature_blocks:
    none: []  # No economic features
    gdp_only: [pib_mdh]  # Only GDP
    sectoral_only: [gdp_primaire, gdp_secondaire, gdp_tertiaire]
    gdp_sectoral: [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire]
```

**What Each Block Does**:

1. **`none`**: Pure autoregressive (only past growth predicts future growth)
   ```python
   Growth_t = f(Growth_{t-1}, Growth_{t-2})
   ```

2. **`gdp_only`**: GDP as main driver
   ```python
   Growth_t = f(Growth_{t-1}, GDP_t, GDP_{t-1})
   ```

3. **`sectoral_only`**: Sectoral composition matters more than total GDP
   ```python
   Growth_t = f(Growth_{t-1}, GDP_primary, GDP_secondary, GDP_tertiary)
   ```

4. **`gdp_sectoral`**: Full economic context
   ```python
   Growth_t = f(Growth_{t-1}, GDP_t, GDP_primary, GDP_secondary, GDP_tertiary)
   ```

#### Growth Feature Transforms
```yaml
features:
  growth_feature_transforms:
    - [lag_lchg]  # Lagged log-change
    # - [level]   # Level of feature
    # - [lchg]    # Log-change of feature
```

**Options**:
- **`level`**: Raw feature value (e.g., GDP in MDH)
- **`lchg`**: Log-change → `log(GDP_t / GDP_{t-1})`
- **`lag_lchg`**: Lagged log-change → `log(GDP_{t-1} / GDP_{t-2})`

**Recommendation**:
```yaml
# For growth modeling, use changes not levels
growth_feature_transforms:
  - [lag_lchg]
  - [lchg, lag_lchg]  # Current + lagged changes
```

#### Growth Feature Lags
```yaml
features:
  growth_feature_lags:
    - [1]  # Use t-1 features
    # - [1, 2]  # Use t-1 and t-2 features
```

**Example**:
```yaml
growth_feature_lags:
  - [1]     # GDP_{t-1} only
  - [1, 2]  # GDP_{t-1} and GDP_{t-2}
```

### When Strategy 2 Excels

✅ **Best For**:
- Consumption strongly correlated with economic growth
- When GDP/sectoral data is reliable
- Trending consumption (not cyclical)

❌ **Struggles With**:
- Economic shocks not captured in GDP
- Weak correlation between economy and consumption
- High-frequency (monthly) volatility

---

## Strategy 3: Hybrid Approach

### The Core Idea

Strategy 3 **combines PC-based patterns with client behavior**, weighting them based on how much client dynamics explain consumption.

**Intuition**: "How much of consumption change is due to patterns vs. client count changes?"

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 3 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Decompose Consumption into Components
──────────────────────────────────────────────────
Component A: Pattern-Based Prediction (from Strategy 1)
    • Use PCA to capture consumption patterns
    • Predict based on historical shapes

Component B: Client-Based Prediction
    • Track client count evolution
    • Predict consumption proportional to clients

STEP 2: Compute Client Pattern Weight
──────────────────────────────────────────
How much does client count explain consumption?

R² = correlation²(Client_count, Consumption)

Example:
    If R²_clients = 0.64 → Clients explain 64% of variance
    If R²_clients = 0.25 → Clients explain 25% of variance

STEP 3: Weight Components
──────────────────────────────────────
Client_weight = (R²_clients)^α
    where α = client_pattern_weight (hyperparameter)

Final Prediction:
    Pred = (Client_weight × Client_component) + 
           ((1 - Client_weight) × Pattern_component)

STEP 4: Example Calculation
────────────────────────────────────────
Assume:
    • R²_clients = 0.64
    • α = 0.5 (square root weighting)
    • Client_weight = √0.64 = 0.8
    
Final:
    Pred = 0.8 × Client_prediction + 0.2 × Pattern_prediction

If clients grew 10% but patterns suggest 5% growth:
    Pred = 0.8 × 1.10 + 0.2 × 1.05 = 0.88 + 0.21 = 1.09
    → 9% growth
```

### Key Hyperparameter

#### Client Pattern Weights (`client_pattern_weights`)
```yaml
model:
  client_pattern_weights: [0.3, 0.5, 0.8]  # Power for R² transformation
```

**What It Does**: Controls how aggressively to weight client dynamics.

**Formula**:
```python
Client_weight = (R²_clients) ** client_pattern_weight
```

**Examples**:

| R²_clients | α=0.3 | α=0.5 | α=0.8 | Interpretation |
|-----------|-------|-------|-------|----------------|
| 0.81      | 0.95  | 0.90  | 0.84  | Strong client signal |
| 0.64      | 0.91  | 0.80  | 0.68  | Moderate client signal |
| 0.36      | 0.83  | 0.60  | 0.42  | Weak client signal |
| 0.09      | 0.72  | 0.30  | 0.13  | Very weak client signal |

**Intuition**:
- **Low α (0.3)**: Conservative, doesn't trust client signal much
- **Medium α (0.5)**: Balanced, square root weighting
- **High α (0.8)**: Aggressive, heavily trusts client correlation

**Recommendation**:
```yaml
# When clients are a strong driver (CD context)
client_pattern_weights: [0.5, 0.7, 0.9]

# When clients are moderate driver
client_pattern_weights: [0.3, 0.5, 0.7]

# When clients are weak driver (SRM context)
client_pattern_weights: [0.1, 0.3, 0.5]
```

### Configuration Requirements

For Strategy 3 to activate:
```yaml
features:
  use_monthly_clients_options: [true]  # Must be enabled
```

**CD Context**:
```yaml
data:
  run_levels: [1]  # Must forecast at contract/activity level

features:
  use_monthly_clients_options: [true]
```

**SRM Context**:
```yaml
features:
  use_monthly_clients_options: [true]
```

### When Strategy 3 Excels

✅ **Best For**:
- Consumption driven by client count (new contracts, churn)
- CD context with volatile client base
- When client data is reliable and granular

❌ **Struggles With**:
- Weak correlation between clients and consumption
- When consumption per client varies wildly
- Missing or unreliable client data

---

## Strategy 4: Advanced Growth Rate Model

### The Core Idea

Strategy 4 uses a **Gaussian Process Regressor** to model year-over-year growth rates, capturing non-linear relationships between features and growth.

**Intuition**: "Can we model complex, non-linear growth dynamics using a flexible probabilistic model?"

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 4 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Compute Annual Growth Rates
────────────────────────────────────────
Growth_t = (Total_t - Total_{t-1}) / Total_{t-1}

STEP 2: Build Feature Matrix (Exogenous Variables)
───────────────────────────────────────────────────────
Features from economic indicators, clients, etc.
    X = [GDP_growth_t, Sectoral_growth_t, Client_change_t, ...]

STEP 3: Fit Gaussian Process Regressor
───────────────────────────────────────────
Model: GP with specified kernel
    • RBF kernel: Smooth, local similarity
    • Matérn kernel: Flexible smoothness
    • WhiteKernel: Observation noise

Hyperparameters:
    • Length scale: How far influence extends
    • Noise level: Observation uncertainty

STEP 4: Predict Next Year's Growth
───────────────────────────────────────
1. Prepare features for year T+1
2. GP.predict(X_{T+1}) → Growth_{T+1}, σ_{T+1}
   (Returns mean and uncertainty)

STEP 5: Apply Growth to Generate Forecast
──────────────────────────────────────────────
Annual_Total_{T+1} = Annual_Total_T × (1 + Growth_{T+1})

STEP 6: Distribute to Monthly Values
─────────────────────────────────────────
Use historical monthly pattern

STEP 7: Cross-Validation
─────────────────────────────
For each year:
    • Train GP on other years
    • Predict held-out growth
    • Compute metrics
```

### Comparison with Strategy 2

| Aspect | Strategy 2 (Ridge) | Strategy 4 (GP) |
|--------|-------------------|-----------------|
| **Model** | Linear | Non-linear |
| **Assumptions** | Linear relationships | Flexible, smooth relationships |
| **Uncertainty** | No | Yes (predictive variance) |
| **Overfitting Risk** | Lower (regularized) | Higher (more flexible) |
| **Best For** | Clear linear trends | Complex, non-linear dynamics |

### Why Use Strategy 4?

✅ **Advantages**:
- Captures non-linear relationships
- Provides uncertainty estimates
- Adaptive to data (kernel optimization)
- No parametric assumptions

⚠️ **Considerations**:
- More computationally expensive
- Can overfit with sparse data
- Requires careful kernel selection
- Hyperparameter tuning is critical

### Configuration (Brief)

Strategy 4 shares configuration with LTF models. Key settings:

```yaml
# These control the internal GP for growth modeling
# (Full details in LTF Configuration Guide)

# Kernel selection (from KERNEL_REGISTRY)
kernel_options:
  - "rbf_white"       # Smooth trends
  - "matern_white"    # Flexible smoothness

# Feature configuration
features:
  feature_blocks:
    - [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire]
  
  growth_feature_transforms:
    - [lag_lchg]  # Lagged log-changes
```

**Note**: See [Long-Term Forecasting Configuration Guide](ltf_configuration.md) for detailed GP configuration.

### When Strategy 4 Excels

✅ **Best For**:
- Non-linear growth patterns
- When uncertainty quantification is valuable
- Complex feature interactions
- Sufficient data for GP (10+ years)

❌ **Struggles With**:
- Sparse data (<8 years)
- Simple linear trends (overkill)
- High-dimensional feature spaces
- Computational constraints

---

## Strategy 5: Ensemble Integration

### The Core Idea

Strategy 5 **combines predictions from multiple strategies**, weighting them by their cross-validated performance.

**Intuition**: "Why choose one strategy when we can intelligently combine them all?"

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY 5 WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Run All Base Strategies
────────────────────────────────────
For each strategy (1-4):
    • Run full LOOCV
    • Get predictions and metrics
    • Store results

STEP 2: Compute Strategy Weights
─────────────────────────────────────
Based on cross-validated performance:

Option A: Inverse MAPE Weighting
    weight_i = (1 / MAPE_i) / Σ(1 / MAPE_j)
    
    → Lower MAPE gets higher weight

Option B: Softmax of Negative MAPE
    weight_i = exp(-MAPE_i) / Σ exp(-MAPE_j)
    
    → Exponentially favor low MAPE

Option C: R² Weighting
    weight_i = R²_i / Σ R²_j
    
    → Higher R² gets higher weight

STEP 3: Combine Predictions
────────────────────────────────────
For each month m:
    Ensemble_pred_m = Σ (weight_i × Strategy_i_pred_m)

STEP 4: Evaluate Ensemble
──────────────────────────────────
Compute metrics for ensemble:
    • R²_ensemble
    • MAPE_ensemble
    • MAE_ensemble
    • RMSE_ensemble

STEP 5: Compare Against Base Strategies
────────────────────────────────────────────
Ensemble competes with individual strategies
    → Selected if it has best R² and MAPE
```

### Ensemble Weighting Example

**Scenario**: Four strategies with following LOOCV results:

| Strategy | R² | MAPE | Inverse MAPE Weight | Softmax Weight |
|----------|-----|------|-------------------|---------------|
| Strategy 1 | 0.82 | 0.08 | 0.40 (12.5/31.25) | 0.47 |
| Strategy 2 | 0.75 | 0.12 | 0.27 (8.33/31.25) | 0.24 |
| Strategy 3 | 0.78 | 0.10 | 0.32 (10.0/31.25) | 0.29 |
| Strategy 4 | 0.70 | 0.20 | 0.16 (5.0/31.25) | 0.08 |

**Final Prediction**:
```python
Ensemble = 0.40 × Strategy1 + 0.27 × Strategy2 + 
           0.32 × Strategy3 + 0.16 × Strategy4
```

### When Ensemble Excels

✅ **Best For**:
- Hedging against single-strategy failure
- When no single strategy dominates
- Reducing prediction variance
- Robustness to structural changes

❌ **Might Not Help When**:
- One strategy vastly outperforms others
- Strategies are highly correlated
- All strategies fail similarly
- Interpretability is critical

### Ensemble vs. Single Strategy

**Ensemble Advantages**:
- More robust to outliers
- Leverages strengths of multiple approaches
- Often lower variance than individual strategies
- "Wisdom of crowds" effect

**Single Strategy Advantages**:
- Simpler to interpret
- Faster computation (no need to run all)
- Clearer feature importance
- When one approach clearly dominates

---

## Configuration Deep Dive

### Complete Configuration Structure

Here's a fully annotated configuration file:

```yaml
# ═══════════════════════════════════════════════════════════════════════
# ONEE FPCA Forecast Configuration - CD (Contrats/Distributeurs)
# ═══════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────
# PROJECT
# ───────────────────────────────────────────────────────────────────────
project:
  project_root: .          # Root directory (resolved to absolute path)
  exp_name: exp            # Experiment name for output organization

# ───────────────────────────────────────────────────────────────────────
# DATA
# ───────────────────────────────────────────────────────────────────────
data:
  variable: consommation_kwh  # Target variable to forecast
  unit: Kwh                   # Unit for reporting
  
  # For CD: null (no regions)
  # For SRM: list of region names
  regions: null
  
  # Run levels:
  # CD: [1] = Contract/Activity level
  # SRM: [0] = Total only, [1] = Activities + Total
  run_levels: [1]
  
  # Path to SQLite database (relative to project_root)
  db_path: data/all_data.db

# ───────────────────────────────────────────────────────────────────────
# EVALUATION
# ───────────────────────────────────────────────────────────────────────
evaluation:
  eval_years_start: 2021      # Start of evaluation period
  eval_years_end: 2023        # End of evaluation period
  train_start_year: 2013      # Earliest year to use for training
  training_end: null          # Latest year to use (null = use all)

# ───────────────────────────────────────────────────────────────────────
# FEATURES
# ───────────────────────────────────────────────────────────────────────
features:
  # Named feature blocks - system tries each combination
  feature_blocks:
    none: []  # No economic features (pure historical patterns)
    gdp_only: [pib_mdh]  # Only total GDP
    sectoral_only: [gdp_primaire, gdp_secondaire, gdp_tertiaire]
    gdp_sectoral: [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire]
  
  # How to transform economic features for growth modeling
  # Options: 'level', 'lchg', 'lag_lchg'
  growth_feature_transforms:
    - [lag_lchg]  # Lagged log-change (recommended for growth)
  
  # How many lags to include for economic features
  growth_feature_lags:
    - [1]    # Include t-1 features
    - [1, 2] # Include t-1 and t-2 features
  
  # Whether to use monthly client count as feature (Strategy 3)
  # CD: Often true (client dynamics matter)
  # SRM: Can be true or false (less client volatility)
  use_monthly_clients_options: [false, true]
  
  # Whether to use Puissance Facturée (Billed Power)
  # CD: Often true (billed power is a key driver)
  # SRM: Typically false (not applicable)
  use_pf_options: [true, false]

# ───────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ───────────────────────────────────────────────────────────────────────
model:
  # Principal Component Analysis
  n_pcs: 3  # Number of PCs to retain (2-5 typical)
  
  # Lag structure for PC-based features
  lags_options: [1, 2]  # How many years of lagged PCs
  
  # Ridge regression regularization
  alphas: [0.01, 0.1, 1.0, 10.0]  # Try multiple alphas
  
  # Strategy 1: Weight of PCs in monthly pattern reconstruction
  pc_weights: [0.5, 0.8]  # 0.0 = Historical avg, 1.0 = Pure PC
  
  # FPCA smoothing parameter
  pca_lambdas: [0.2, 0.3, 0.4]  # Higher = smoother PCs
  
  # Training window (years of history to use)
  training_windows: [1, 2, 3, 4, null]  # null = all available
  
  # Strategy 3: Client pattern weighting (α parameter)
  client_pattern_weights: [0.3, 0.5, 0.8]
  
  # Model selection threshold
  r2_threshold: 0.6  # Minimum R² to be considered "good"

# ───────────────────────────────────────────────────────────────────────
# LOSS CONFIGURATION
# ───────────────────────────────────────────────────────────────────────
loss:
  # Whether to favor overestimation (safer for capacity planning)
  favor_overestimation: true
  
  # Penalty multiplier for underestimation
  # E.g., 1.5 means underestimating costs 1.5x more than overestimating
  under_estimation_penalty: 1.5
  
  # Level-specific penalties (optional, typically null)
  penalties_by_level: null
```

---

## Feature Configuration

### Feature Blocks

```yaml
features:
  feature_blocks:
    none: []
    gdp_only: [pib_mdh]
    sectoral: [gdp_primaire, gdp_secondaire, gdp_tertiaire]
    gdp_sectoral: [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire]
```

System tries all blocks, selects best performing.

### Growth Feature Transforms (Strategy 4)

```yaml
features:
  growth_feature_transforms:
    - [level]        # Raw value: GDP_t
    - [lchg]         # Log-change: log(GDP_t / GDP_{t-1})
    - [lag_lchg]     # Lagged: log(GDP_{t-1} / GDP_{t-2})
```

Use `lag_lchg` to avoid lookahead bias when predicting Growth_t.

### Growth Feature Lags (Strategy 4)

```yaml
features:
  growth_feature_lags:
    - [1]      # GDP_{t-1}
    - [1, 2]   # GDP_{t-1}, GDP_{t-2}
```

### Monthly Client Counts

```yaml
features:
  use_monthly_clients_options: [true, false]
```

Adds 12 monthly client count features. Required for Strategy 3 client weighting.

### Puissance Facturée (CD only)

```yaml
features:
  use_pf_options: [true, false]
```

Adds billed power features. Not applicable for SRM.

---

## Hyperparameters

### Grid Search

System tries all combinations:
```
configurations = product(feature_blocks, transforms, lags, alphas, ...)
```

Configuration count = product of all list lengths.

### Common Configurations

**Minimal (fast)**:
```yaml
model:
  n_pcs: 3
  lags_options: [1, 2]
  alphas: [0.1, 1.0, 10.0]
  pc_weights: [0.5, 0.8]
  pca_lambdas: [0.3]
  training_windows: [null]
  client_pattern_weights: [0.5]
```

**Standard**:
```yaml
model:
  n_pcs: 3
  lags_options: [1, 2]
  alphas: [0.01, 0.1, 1.0, 10.0]
  pc_weights: [0.5, 0.8]
  pca_lambdas: [0.3, 0.7]
  training_windows: [null, 3, 7]
  client_pattern_weights: [0.3, 0.5, 0.8]
```

**Exhaustive**:
```yaml
model:
  n_pcs: 4
  lags_options: [1, 2, 3]
  alphas: [0.01, 0.1, 1.0, 10.0, 100.0]
  pc_weights: [0.3, 0.5, 0.7, 0.9]
  pca_lambdas: [0.2, 0.3, 0.5, 0.7]
  training_windows: [null, 3, 5, 7, 10]
  client_pattern_weights: [0.3, 0.5, 0.8]
```

### Parameter Effects

| Parameter | Effect |
|-----------|--------|
| `n_pcs` | Number of PCA components (3-4 typical) |
| `lags_options` | Years of history for features |
| `alphas` | Ridge regularization strength |
| `pc_weights` | Weight of PC reconstruction vs historical average (Strategy 3) |
| `pca_lambdas` | Temporal weighting: higher = more weight to recent years |
| `training_windows` | Years of training data (null = all) |
| `client_pattern_weights` | Power transform for R² in client weighting (Strategy 2, 3, 4) |

---

## Loss Configuration

### Asymmetric Loss (Ensemble only)

```yaml
loss:
  favor_overestimation: true
  under_estimation_penalty: 2.0
```

Computes tau for pinball loss:
```python
tau = under_estimation_penalty / (1 + under_estimation_penalty)
# Example: 2.0 → tau = 0.67
```

Ensemble learns bias correction factor λ by minimizing pinball loss on past years, then applies `(1 + λ)` scaling.

**Configuration**:
```yaml
# Symmetric loss
loss:
  favor_overestimation: false

# Mild upward bias
loss:
  favor_overestimation: true
  under_estimation_penalty: 1.5  # tau = 0.6

# Strong upward bias  
loss:
  favor_overestimation: true
  under_estimation_penalty: 3.0  # tau = 0.75
```

---

## Evaluation Methodology

### Leave-One-Out Cross-Validation (LOOCV)

STF uses **strict LOOCV** to prevent overfitting and ensure robust evaluation.

#### The Process

```
┌─────────────────────────────────────────────────────────────────┐
│              LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)              │
└─────────────────────────────────────────────────────────────────┘

Evaluation Period: 2021-2023 (3 years)

FOLD 1: Hold out 2021
─────────────────────────────────────────────
Train: [2013, 2014, 2015, ..., 2020, 2022, 2023]
Test:  [2021]
    ↓
Predict 2021, compute metrics

FOLD 2: Hold out 2022
─────────────────────────────────────────────
Train: [2013, 2014, 2015, ..., 2020, 2021, 2023]
Test:  [2022]
    ↓
Predict 2022, compute metrics

FOLD 3: Hold out 2023
─────────────────────────────────────────────
Train: [2013, 2014, 2015, ..., 2020, 2021, 2022]
Test:  [2023]
    ↓
Predict 2023, compute metrics

AGGREGATE METRICS
─────────────────────────────────────────────
R² = aggregate R² across all folds
MAPE = mean of (|pred - actual| / actual) across all folds
MAE = mean of |pred - actual| across all folds
RMSE = sqrt(mean((pred - actual)²)) across all folds
```

#### Why LOOCV?

✅ **Advantages**:
- **No future data leakage**: Each fold strictly separates train/test
- **Maximum data usage**: Every year is used for both training and testing
- **Robust estimates**: Multiple test points reduce variance
- **Fair comparison**: All models evaluated on same folds

⚠️ **Considerations**:
- **Computationally expensive**: N folds for N years
- **May be pessimistic**: Each model misses one training year

### Metrics Explained

#### R² (Coefficient of Determination)
```
R² = 1 - (SS_residual / SS_total)
   = 1 - (Σ(y - ŷ)² / Σ(y - ȳ)²)
```

**Interpretation**:
- **1.0**: Perfect predictions
- **0.6-0.9**: Good to excellent (typical for STF)
- **0.3-0.6**: Moderate (may need improvement)
- **<0.3**: Poor (model not capturing patterns)
- **Negative**: Model worse than mean baseline

**Threshold**:
```yaml
model:
  r2_threshold: 0.6  # Minimum to be considered "good"
```

#### MAPE (Mean Absolute Percentage Error)
```
MAPE = (1/n) × Σ(|actual - predicted| / actual) × 100%
```

**Interpretation**:
- **<5%**: Excellent
- **5-10%**: Good
- **10-20%**: Moderate
- **>20%**: Poor

**Example**:
```
Actual: 1000 kWh
Predicted: 950 kWh
APE = |1000 - 950| / 1000 = 0.05 = 5%
```

#### MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|actual - predicted|
```

**Interpretation**: Average magnitude of errors in consumption units (kWh).

**Example**:
```
MAE = 50 kWh means predictions are off by 50 kWh on average
```

#### RMSE (Root Mean Squared Error)
```
RMSE = sqrt((1/n) × Σ(actual - predicted)²)
```

**Interpretation**: Similar to MAE but penalizes larger errors more heavily.

**Comparison**:
- If RMSE >> MAE: Predictions have large outliers
- If RMSE ≈ MAE: Errors are relatively uniform

---