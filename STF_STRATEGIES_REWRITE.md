# Short-Term Forecasting Strategies - Technical Reference

This document describes the four short-term forecasting strategies and their configuration parameters.

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

## Complete Configuration Example

```yaml
# ═══════════════════════════════════════════════════════════════
# SHORT-TERM FORECASTING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

data:
  variable: consommation_kwh
  db_path: data/all_data.db

evaluation:
  eval_years_start: 2021
  eval_years_end: 2023

# ───────────────────────────────────────────────────────────────
# FEATURES
# ───────────────────────────────────────────────────────────────
features:
  # Economic feature blocks (tried in all strategies)
  feature_blocks:
    none: []
    gdp_only: [pib_mdh]
    sectoral: [gdp_primaire, gdp_secondaire, gdp_tertiaire]
    gdp_sectoral: [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire]
  
  # Strategy 4: Growth rate feature transforms
  growth_feature_transforms:
    - [lag_lchg]
    - [lchg, lag_lchg]
  
  # Strategy 4: Growth rate feature lags
  growth_feature_lags:
    - [1]
    - [1, 2]
  
  # Strategy 1, 3: Monthly features
  use_monthly_temp_options: [true, false]
  use_monthly_clients_options: [true, false]
  
  # CD only: Puissance facturée
  use_pf_options: [true, false]

# ───────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ───────────────────────────────────────────────────────────────
model:
  # Strategy 1, 3: PCA configuration
  n_pcs: 3
  pca_lambdas: [0.3, 0.7, 1.0]
  
  # Strategy 1, 2, 3: Lags
  lags_options: [1, 2]
  
  # All strategies: Ridge regularization
  alphas: [0.01, 0.1, 1.0, 10.0]
  
  # Strategy 3: PC weighting
  pc_weights: [0.5, 0.7]
  
  # Strategy 2, 3, 4: Client pattern weighting
  client_pattern_weights: [0.3, 0.5, 0.8]
  
  # Strategy 4: Training window
  training_windows: [4, 7, 10, null]
  
  # Model selection threshold
  r2_threshold: 0.6

# ───────────────────────────────────────────────────────────────
# LOSS CONFIGURATION
# ───────────────────────────────────────────────────────────────
loss:
  favor_overestimation: true
  under_estimation_penalty: 2  # For ensemble pinball loss
```
