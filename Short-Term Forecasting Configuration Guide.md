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