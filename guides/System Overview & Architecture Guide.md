# System Overview & Architecture Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Two Forecasting Paradigms](#two-forecasting-paradigms)
4. [Business Contexts: CD vs SRM](#business-contexts-cd-vs-srm)
5. [Understanding Run Levels](#understanding-run-levels)
6. [Data Flow & Pipeline Architecture](#data-flow--pipeline-architecture)
7. [Model Selection Philosophy](#model-selection-philosophy)
8. [Configuration Files Overview](#configuration-files-overview)
9. [Getting Started Checklist](#getting-started-checklist)

---

## Introduction

This forecasting system provides electricity consumption predictions for ONEE (Office National de l'Électricité et de l'Eau potable) across two main business contexts:

- **CD**: Contract-level forecasting
- **SRM**: Regional forecasting

The system supports two forecasting horizons:
- **Short-Term Forecasting (STF)**: 1-year ahead predictions using FPCA-based strategies
- **Long-Term Forecasting (LTF)**: 5-year horizon predictions using Gaussian Process models

---

## Project Structure

```
onee/
├── configs/                    # Configuration files
│   ├── stf_cd.yaml            # Short-term config for CD
│   ├── stf_srm.yaml           # Short-term config for SRM
│   ├── ltf_cd.yaml            # Long-term config for CD
│   └── ltf_srm.yaml           # Long-term config for SRM
│
├── src/onee/
│   ├── growth_rate_model.py   # LTF models (GP, Priors)
│   ├── short_term_forecast_strategies.py  # STF strategies (FPCA)
│   ├── data/
│   │   └── names.py           # Column name aliases
│   └── utils.py               # Feature engineering utilities
│
├── data/
│   └── all_data.db            # SQLite database with all data
│
└── outputs/                    # Forecast results
    ├── outputs_horizon/        # LTF results (SRM)
    └── outputs_horizon_cd/     # LTF results (CD)
```

---

## Two Forecasting Paradigms

### Short-Term Forecasting (STF)
**Time Horizon**: 1 year ahead  
**Methodology**: Functional Principal Component Analysis (FPCA) + Ridge Regression  
**Best For**: Detailed monthly predictions with recent pattern recognition

**Key Characteristics**:
- ✅ Captures seasonal patterns and monthly variations
- ✅ Leverages recent historical behavior
- ✅ Four complementary strategies + ensemble
- ✅ Strict leave-one-out cross-validation
- ⚠️ Requires sufficient historical data (typically 5+ years)

### Long-Term Forecasting (LTF)
**Time Horizon**: 5 years ahead  
**Methodology**: Gaussian Process Regression with Trend Priors  
**Best For**: Strategic planning with growth trend modeling

**Key Characteristics**:
- ✅ Models long-term growth trends
- ✅ Incorporates economic drivers (GDP, sectoral growth)
- ✅ Uncertainty quantification
- ✅ Flexible trend assumptions (linear, power, consensus)
- ⚠️ Don't provide monthly granularity

---

## Business Contexts: CD vs SRM

### CD 

**Key Features**:
- Contract lifecycle tracking (new, 2-year-old, 3-year-old, mature)
- Puissance Facturée (Billed Power) as a driver

**Data Characteristics**:
```yaml
data:
  regions: null  # CD doesn't use regions
  run_levels: [1]  # 1 = Contract/Activity level
```

---

### SRM

**Key Features**:
- Regional GDP and sectoral growth integration
- Client count evolution tracking
- Regional activity aggregations
- Geographic granularity

**Data Characteristics**:
```yaml
data:
  regions:
    - Casablanca-Settat
    - Béni Mellal-Khénifra
    - Drâa-Tafilalet
    - Fès-Meknès
    - Laâyoune-Sakia El Hamra
    - Marrakech-Safi
    - Oriental
    - Rabat-Salé-Kénitra
    - Tanger-Tétouan-Al Hoceïma
    - Souss-Massa
  run_levels: [1]  # 1 = Regional SRM level
```

---

## Understanding Run Levels

Run levels control the **granularity of aggregation** in the forecasting process.

### For SRM (Regional Context)
```yaml
run_levels:
  0: Total SRM (regional aggregate only)
  1: Both Individual Activities AND Total SRM
```

**Example Configuration**:
```yaml
# configs/stf_srm.yaml
data:
  regions:
    Casablanca-Settat: 0      # Only forecast total regional aggregate
    Fès-Meknès: 1             # Forecast activities + total
    Marrakech-Safi: 0         # Only total
```

### For CD (Contract Context)
```yaml
run_levels:
  1: Contract/Activity level  # Individual contract forecasts
  2: Regional aggregation     # Contracts aggregated by region
```

**Example Configuration**:
```yaml
# configs/ltf_cd.yaml
data:
  run_levels: [2]  # Forecast at regional aggregation level
```

---

## Data Flow & Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Consumption  │  │  Economic    │  │  Contract    │         │
│  │   History    │  │  Indicators  │  │   Metadata   │         │
│  │  (Monthly)   │  │  (GDP, etc)  │  │  (Clients)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│          │                 │                 │                  │
│          └─────────────────┴─────────────────┘                  │
│                            │                                     │
│                    ┌───────▼────────┐                           │
│                    │  all_data.db   │                           │
│                    │   (SQLite)     │                           │
│                    └───────┬────────┘                           │
└────────────────────────────┼──────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Feature         │ │  Monthly        │ │  Annual         │
│ Engineering     │ │  Matrix         │ │  Aggregation    │
│                 │ │  Construction   │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                             │
                    ┌────────▼─────────┐
                    │ Forecasting Path │
                    │    Selection     │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
   ┌──────────────────────┐     ┌──────────────────────┐
   │  Short-Term Path     │     │  Long-Term Path      │
   │  (STF)               │     │  (LTF)               │
   │                      │     │                      │
   │ ┌────────────────┐  │     │ ┌────────────────┐  │
   │ │ Strategy 1     │  │     │ │ GP Model       │  │
   │ │ (PC-Based)     │  │     │ │ + Trend Prior  │  │
   │ └────────────────┘  │     │ └────────────────┘  │
   │                      │     │                      │
   │ ┌────────────────┐  │     │ ┌────────────────┐  │
   │ │ Strategy 2     │  │     │ │ Intensity      │  │
   │ │ (Direct Annual)│  │     │ │ Wrapper (CD)   │  │
   │ └────────────────┘  │     │ └────────────────┘  │
   │                      │     │                      │
   │ ┌────────────────┐  │     │                      │
   │ │ Strategy 3     │  │     │                      │
   │ │ (Hybrid)       │  │     │                      │
   │ └────────────────┘  │     │                      │
   │                      │     │                      │
   │ ┌────────────────┐  │     │                      │
   │ │ Strategy 4     │  │     │                      │
   │ │ (Growth Rate)  │  │     │                      │
   │ └────────────────┘  │     │                      │
   │                      │     │                      │
   │ ┌────────────────┐  │     │                      │
   │ │ Strategy 5     │  │     │                      │
   │ │ (Ensemble)     │  │     │                      │
   │ └────────────────┘  │     │                      │
   └──────────┬───────────┘     └──────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Model Selection │
                    │  (Best R²)       │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Final Forecast │
                    │   + Metrics      │
                    └──────────────────┘
```

### Pipeline Stages Explained

#### 1. **Data Ingestion**
- Reads from SQLite database (`all_data.db`)
- Filters by region (SRM) or contract (CD)
- Applies run level filtering

#### 2. **Feature Engineering**
- **Temporal Features**: Lags, log-changes, growth rates
- **Economic Features**: GDP, sectoral indicators
- **Client Features**: Active contracts, new clients, client patterns
- **Power Features**: Puissance Facturée (for CD)

#### 3. **Forecasting Path Selection**
- **STF Path**: Uses `short_term_forecast_strategies.py`
  - Evaluates 4 core strategies
  - Runs strict LOOCV
  - Selects best performer
  
- **LTF Path**: Uses `growth_rate_model.py`
  - Configures GP with trend prior
  - Fits on historical growth rates
  - Projects 5 years ahead

#### 4. **Model Selection**
- **Primary Criterion**: R² threshold (default: 0.6)
- **Fallback**: Highest R² if no model passes threshold
- **Tie-Breaking**: Favors lower MAPE

#### 5. **Output Generation**
- Monthly predictions (STF) or annual predictions (LTF)
- Performance metrics (R², MAPE, MAE, RMSE)
- Model configuration details
- Confidence intervals (for LTF)

---

## Model Selection Philosophy

### The Selection Process

```
┌─────────────────────────────────────────────────────────────┐
│                    HYPERPARAMETER GRID                       │
│                                                              │
│  For each combination of:                                   │
│  • Feature blocks (GDP, sectoral, none)                     │
│  • Lags (1, 2, 3...)                                        │
│  • Model parameters (alphas, PC weights, etc.)              │
│  • Training windows                                         │
│                                                              │
│  ───────────────────────────────────────────────────────    │
│                                                              │
│  Generate a unique model configuration                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              STRICT CROSS-VALIDATION (LOOCV)                 │
│                                                              │
│  For each year in evaluation period:                        │
│    • Train on all other years                               │
│    • Predict held-out year                                  │
│    • Calculate metrics                                      │
│                                                              │
│  Aggregate metrics across all folds                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   SELECTION CRITERIA                         │
│                                                              │
│  1. Filter: R² >= threshold (default 0.6)                   │
│  2. If multiple pass: Select lowest MAPE                    │
│  3. If none pass: Fallback to highest R²                    │
│                                                              │
│  Special: Ensemble combines top strategies                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     BEST MODEL SELECTED                      │
│                                                              │
│  Returns:                                                    │
│  • Model configuration                                       │
│  • Cross-validated predictions                              │
│  • Performance metrics                                      │
│  • Feature importance (where applicable)                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **No Future Data Leakage**: Strict temporal separation in cross-validation
2. **Robust Evaluation**: Leave-one-out ensures every year is tested
3. **Transparent Selection**: Clear hierarchy of selection criteria
4. **Flexibility**: R² threshold can be adjusted per use case
5. **Safety Net**: Always returns a model (fallback to best available)

### Asymmetric Loss (Favor Overestimation)

The system can be configured to **penalize underestimation** more heavily:

```yaml
loss:
  favor_overestimation: true
  under_estimation_penalty: 1.5  # Underestimates cost 1.5x more
```

**Why This Matters**:
- Electricity supply shortfalls are costlier than slight oversupply
- Better to plan for higher capacity than risk blackouts
- Configurable per business context

---

## Configuration Files Overview

### Configuration Structure (All Files)

```yaml
# ═══════════════════════════════════════════════════════════
# 1. PROJECT SETTINGS
# ═══════════════════════════════════════════════════════════
project:
  project_root: .
  exp_name: exp
  output_base_dir: outputs/  # LTF only

# ═══════════════════════════════════════════════════════════
# 2. DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════
data:
  target_variable: consommation_kwh  # or just 'variable' for STF
  unit: Kwh
  regions: [...]  # List for SRM, null for CD
  run_levels: [1, 2]
  db_path: data/all_data.db

# ═══════════════════════════════════════════════════════════
# 3. TEMPORAL SETTINGS
# ═══════════════════════════════════════════════════════════
# STF: Evaluation period
evaluation:
  eval_years_start: 2021
  eval_years_end: 2023
  train_start_year: 2013

# LTF: Forecast horizon
temporal:
  horizon: 5
  forecast_runs:
    - [2013, 2018]

# ═══════════════════════════════════════════════════════════
# 4. FEATURES
# ═══════════════════════════════════════════════════════════
features:
  # Feature blocks, transforms, lags...
  # See detailed guides for specifics

# ═══════════════════════════════════════════════════════════
# 5. MODELS
# ═══════════════════════════════════════════════════════════
# STF: Hyperparameter grids for strategies
model:
  n_pcs: 3
  lags_options: [1, 2]
  alphas: [0.01, 0.1, 1.0]
  # ...

# LTF: Model specifications
models:
  r2_threshold: 0.6
  models:
    - model_type: GaussianProcessForecastModel
      # Model-specific parameters...

# ═══════════════════════════════════════════════════════════
# 6. LOSS CONFIGURATION
# ═══════════════════════════════════════════════════════════
loss:
  favor_overestimation: true
  under_estimation_penalty: 1.5
```

### Configuration Files Quick Reference

| File | Purpose | Forecast Horizon | Business Context |
|------|---------|------------------|------------------|
| `stf_cd.yaml` | Short-term CD forecasting | 1 year | CD |
| `stf_srm.yaml` | Short-term SRM forecasting | 1 year | SRM |
| `ltf_cd.yaml` | Long-term CD forecasting | 5 years | CD |
| `ltf_srm.yaml` | Long-term SRM forecasting | 5 years | SRM |

### When to Modify Which Configuration

**Scenario 1: Changing Evaluation Period (STF)**
```yaml
# File: stf_cd.yaml or stf_srm.yaml
evaluation:
  eval_years_start: 2022  # New start year
  eval_years_end: 2024    # New end year
```

**Scenario 2: Adding New Region (SRM)**
```yaml
# File: stf_srm.yaml or ltf_srm.yaml
data:
  regions:
    - Casablanca-Settat
    - New-Region-Name  # Add here
```

**Scenario 3: Adjusting Model Stringency**
```yaml
# File: Any
model:  # (STF)
  r2_threshold: 0.7  # More strict (was 0.6)

models:  # (LTF)
  r2_threshold: 0.7  # More strict
```

**Scenario 4: Increasing Forecast Horizon (LTF)**
```yaml
# File: ltf_cd.yaml or ltf_srm.yaml
temporal:
  horizon: 10  # Extend to 10 years (was 5)
```

---

## Getting Started Checklist

### For Short-Term Forecasting (STF)

- [ ] **1. Identify Business Context**
  - CD (contracts) → Use `stf_cd.yaml`
  - SRM (regions) → Use `stf_srm.yaml`

- [ ] **2. Configure Data Settings**
  - [ ] Set regions (SRM only)
  - [ ] Define run levels
  - [ ] Verify database path

- [ ] **3. Define Evaluation Period**
  - [ ] Set `eval_years_start` and `eval_years_end`
  - [ ] Ensure sufficient training history (5+ years recommended)

- [ ] **4. Select Feature Blocks**
  - [ ] Choose economic indicators (GDP, sectoral)
  - [ ] Enable/disable client features
  - [ ] Enable/disable Puissance Facturée (CD)

- [ ] **5. Configure Model Hyperparameters**
  - [ ] Set PCA components (`n_pcs`)
  - [ ] Define lag options
  - [ ] Configure Ridge regression alphas
  - [ ] Set training windows

- [ ] **6. Configure Loss Function**
  - [ ] Enable/disable overestimation bias
  - [ ] Set underestimation penalty

- [ ] **7. Run Forecasting**
  ```bash
  python run_stf_cd.py    # For CD
  python run_stf_srm.py   # For SRM
  ```

### For Long-Term Forecasting (LTF)

- [ ] **1. Identify Business Context**
  - CD (contracts) → Use `ltf_cd.yaml`
  - SRM (regions) → Use `ltf_srm.yaml`

- [ ] **2. Configure Data Settings**
  - [ ] Set regions (SRM only)
  - [ ] Define run levels
  - [ ] Verify database path

- [ ] **3. Define Forecast Horizon**
  - [ ] Set horizon years (default: 5)
  - [ ] Configure forecast run periods

- [ ] **4. Select Features**
  - [ ] Choose transforms (level, lchg, lag_lchg)
  - [ ] Define lag structure
  - [ ] Select feature blocks

- [ ] **5. Configure GP Model**
  - [ ] Select kernel from registry
  - [ ] Choose trend prior strategy
  - [ ] Configure prior parameters
  - [ ] Enable/disable outlier detection

- [ ] **6. Special: CD Intensity Wrapper**
  - [ ] Set normalization column (e.g., `total_active_contrats`)
  - [ ] Configure internal GP parameters

- [ ] **7. Run Forecasting**
  ```bash
  python run_ltf_cd.py    # For CD
  python run_ltf_srm.py   # For SRM
  ```

---

## Common Pitfalls & Troubleshooting

### Issue 1: "No model passes R² threshold"
**Symptom**: System falls back to highest R² model  
**Causes**:
- Insufficient training data
- Threshold too high for data quality
- Volatile consumption patterns

**Solutions**:
```yaml
model:  # or models:
  r2_threshold: 0.5  # Lower threshold
```

### Issue 2: "Predictions are too conservative/aggressive"
**Symptom**: Systematic under/over-estimation  
**Solutions**:
```yaml
loss:
  favor_overestimation: true  # Or false
  under_estimation_penalty: 2.0  # Adjust (1.0-3.0)
```

### Issue 3: "Long-term forecasts show unrealistic growth"
**Symptom**: Exponential explosion in LTF predictions  
**Causes**:
- Wrong trend prior
- No growth constraint

**Solutions**:
```yaml
# Switch from NeutralPrior to constrained prior
prior_config:
  - type: LinearGrowthPrior
    min_annual_growth: 0.02  # Floor at 2% growth
    anchor_window: 3
```

### Issue 4: "STF strategies all perform similarly"
**Symptom**: Ensemble doesn't add value  
**Causes**:
- Feature blocks too similar
- Insufficient hyperparameter diversity

**Solutions**:
```yaml
features:
  feature_blocks:
    none: []
    gdp_only: [pib_mdh]
    sectoral_only: [gdp_primaire, gdp_secondaire, gdp_tertiaire]
    full: [pib_mdh, gdp_primaire, gdp_secondaire, gdp_tertiaire, custom_feature]
```

---

## Next Steps

- **Short-Term Forecasting**: See [STF Configuration Guide](stf_configuration.md) for detailed strategy explanations
- **Long-Term Forecasting**: See [LTF Configuration Guide](ltf_configuration.md) for GP model and prior details
- **Advanced Topics**: See [Advanced Topics Guide](advanced_topics.md) for adjustments and edge cases

---

## Glossary

| Term | Definition |
|------|------------|
| **CD** | Clients directs |
| **SRM** | Société Régionale Multiservices |
| **STF** | Short-Term Forecasting (1 year horizon) |
| **LTF** | Long-Term Forecasting (5 year horizon) |
| **FPCA** | Functional Principal Component Analysis |
| **GP** | Gaussian Process |
| **LOOCV** | Leave-One-Out Cross-Validation |
| **Puissance Facturée** | Billed power (kW) - Key driver for CD |
| **Run Level** | Aggregation granularity (activity, region, total) |
| **Trend Prior** | Deterministic baseline for GP models |
| **R² Threshold** | Minimum explained variance for model selection |

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintained By**: Data Science Team