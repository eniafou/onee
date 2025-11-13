"""
Standalone Growth Rate Prediction Model - Mean-Reverting with Scipy
====================================================================
Proper implementation: Δlog(y_{t+1}) = (1-ρ)μ + ρ*Δlog(y_t) + z_t'γ + ε
"""

import warnings
warnings.filterwarnings('ignore')

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Iterable, List


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[0]

# Database paths
DB_REGIONAL_PATH = PROJECT_ROOT / 'data/ONEE_Regional_COMPLETE_2007_2023.db'
DB_DIST_PATH = PROJECT_ROOT / 'data/ONEE_Distributeurs_consumption.db'

# Analysis settings
TARGET_REGION = "Casablanca-Settat"
TARGET_ACTIVITY = "total" #"Menages"
VARIABLE = "consommation_kwh"
TRAINING_END = 2018
TRAIN_WINDOW = None
# Model settings
INCLUDE_AR = True          # Include autoregressive component ρ*Δlog(y_t)
INCLUDE_EXOG = True        # Include exogenous features z_t'γ
FEATURE_TRANSFORMS = ("lag_lchg",)  #("level", "lag", "lchg", "lag_lchg")
EXOG_LAGS = (1,)
L2_PENALTY = 10.0           # L2 regularization strength (like Ridge alpha)
RHO_BOUNDS = (0.0, 1)   # Bounds for ρ (stationarity constraint)


# ============================================================================
# 1. DATA LOADING (Same as before)
# ============================================================================

def load_regional_data(region: str, variable: str = "consommation_kwh") -> pd.DataFrame:
    """Load regional consumption data from database."""
    
    print(f"Loading data for region: {region}")
    print(f"Variable: {variable}")
    
    db = sqlite3.connect(DB_REGIONAL_PATH)
    if variable == "consommation_kwh":
        query_bt = f"""
            SELECT
                Year as annee,
                Month as mois,
                Activity as activite,
                MWh * 1000 as consommation_kwh
            FROM monthly_data_bt
            WHERE Region = '{region}'
            ORDER BY Year, Month, Activity
        """
        
        query_mt = f"""
            SELECT
                Year as annee,
                Month as mois,
                Activity as activite,
                MWh * 1000 as consommation_kwh
            FROM monthly_data_mt
            WHERE Region = '{region}'
            ORDER BY Year, Month, Activity
        """
    else:
        print("Queries for the specified variable not found.")
        return None
    
    df_bt = pd.read_sql_query(query_bt, db)
    df_mt = pd.read_sql_query(query_mt, db)
    
    df_mt['activite'] = df_mt['activite'].replace("Administratif", "Administratif_mt")
    df = pd.concat([df_bt, df_mt], ignore_index=True)
    
    db.close()
    
    print(f"✓ Loaded {len(df)} monthly records")
    print(f"  Years: {df['annee'].min()} - {df['annee'].max()}")
    print(f"  Activities: {sorted(df['activite'].unique())}")
    
    return df


def load_features(region: str) -> pd.DataFrame:
    """Load exogenous features (GDP, temperature, etc.)."""
    
    db = sqlite3.connect(DB_REGIONAL_PATH)
    
    query = f"""
        SELECT
            Year as annee,
            AVG(GDP_Millions_DH) as pib_mdh,
            AVG(GDP_Primaire) as gdp_primaire,
            AVG(GDP_Secondaire) as gdp_secondaire,
            AVG(GDP_Tertiaire) as gdp_tertiaire,
            AVG(temp) as temperature_annuelle
        FROM regional_features
        WHERE Region = '{region}'
        GROUP BY Year
    """
    
    df = pd.read_sql_query(query, db)
    db.close()
    
    print(f"✓ Loaded features for {len(df)} years ({df['annee'].min()} - {df['annee'].max()})")
    
    return df

def prepare_activity_data(
    df_regional: pd.DataFrame, 
    activity: str,
    df_features: pd.DataFrame,
    *,
    transforms: Iterable[str] = ("level",),   # any of {"level","lag","lchg","lag_lchg"}
    lags: Iterable[int] = (1,),               # used for "lag" and/or "lag_lchg"
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare data for a specific activity and add feature transforms per user choice.

    Parameters
    ----------
    transforms : any subset of {"level", "lag", "lchg", "lag_lchg"}
        - "level": keep original feature levels
        - "lag": add lagged *levels* with specified `lags`
        - "lchg": add Δln feature: ln(x_t) - ln(x_{t-1})
        - "lag_lchg": add *lagged* Δln features with specified `lags`
    lags : iterable of positive ints
        Lags to compute when "lag" and/or "lag_lchg" is included.
    """

    transforms = set(transforms)
    allowed = {"level", "lag", "lchg", "lag_lchg"}
    unknown = transforms - allowed
    if unknown:
        raise ValueError(f"Unknown transform(s): {unknown}. Allowed: {allowed}")
    if any(l <= 0 for l in lags):
        raise ValueError("All lags must be positive integers.")

    # -------- Build base activity/annual dataset --------
    if activity == "total":
        df_activity = (
            df_regional.groupby(['annee', 'mois'])
            .agg({VARIABLE: 'sum'})
            .reset_index()
        )
    else:
        df_activity = df_regional[df_regional['activite'] == activity].copy()

    if df_activity.empty:
        raise ValueError(f"No data found for activity: {activity}")

    df_annual = (
        df_activity.groupby('annee')
        .agg({VARIABLE: 'sum'})
        .reset_index()
    )

    df_annual = df_annual.merge(df_features, on='annee', how='left')
    df_annual = df_annual.sort_values('annee').reset_index(drop=True)

    feature_cols = [c for c in df_annual.columns if c not in ['annee', VARIABLE]]

    print(f"\n✓ Prepared data for activity: {activity}")
    print(f"  Annual records: {len(df_annual)}")
    print(f"  Year range: {df_annual['annee'].min()} - {df_annual['annee'].max()}")
    print(f"  Consumption range: {df_annual[VARIABLE].min():.0f} - {df_annual[VARIABLE].max():.0f} kWh")

    # -------- Handle missing in base features (levels) before deriving --------
    missing_counts = df_annual[feature_cols].isnull().sum()
    if missing_counts.any():
        print(f"\n⚠️  Missing values detected in features:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"    {col}: {count} missing")

        print("  → Interpolating missing values...")
        for col in feature_cols:
            df_annual[col] = df_annual[col].interpolate(method='linear', limit_direction='both')
            if df_annual[col].isnull().any():
                df_annual[col] = df_annual[col].fillna(df_annual[col].mean())
        print("  ✓ Missing values handled")

    # -------- Build the final feature matrix according to transforms --------
    keep = ['annee', VARIABLE]
    out = df_annual[keep].copy()
    final_features: List[str] = []

    # Precompute lchg if needed (for either "lchg" or "lag_lchg")
    lchg_needed = ("lchg" in transforms) or ("lag_lchg" in transforms)
    lchg_df = pd.DataFrame(index=df_annual.index)

    if lchg_needed:
        for col in feature_cols:
            s = df_annual[col].astype(float)
            prev = s.shift(1)
            valid = (s > 0) & (prev > 0)
            lchg = pd.Series(np.where(valid, np.log(s) - np.log(prev), np.nan), index=s.index)
            # Interpolate within series; if still NaN (e.g., first row), fill with 0 to avoid loss of rows
            lchg = lchg.interpolate(method='linear', limit_direction='both').fillna(0.0)
            new_col = f"{col}_lchg"
            lchg_df[new_col] = lchg

    # 1) Levels
    if "level" in transforms:
        out = out.join(df_annual[feature_cols])  # keep original names
        final_features.extend(feature_cols)

    # 2) Lagged levels
    if "lag" in transforms:
        for L in lags:
            lagged = df_annual[feature_cols].shift(L)
            lagged_cols = [f"{c}_lag{L}" for c in feature_cols]
            lagged.columns = lagged_cols
            out = out.join(lagged)
            final_features.extend(lagged_cols)

    # 3) Δln (log-change) of levels
    if "lchg" in transforms:
        out = out.join(lchg_df)
        final_features.extend(list(lchg_df.columns))

    # 4) Lagged Δln (log-change)
    if "lag_lchg" in transforms:
        if lchg_df.empty:
            # Shouldn't happen due to lchg_needed, but just in case
            raise RuntimeError("Internal: lchg_df was not computed but 'lag_lchg' requested.")
        for L in lags:
            lagged_lchg = lchg_df.shift(L)
            lagged_lchg_cols = [f"{c}_lag{L}" for c in lchg_df.columns]  # e.g., foo_lchg_lag2
            lagged_lchg.columns = lagged_lchg_cols
            out = out.join(lagged_lchg)
            final_features.extend(lagged_lchg_cols)

    # Optional: if you prefer not to carry NaNs from lagging, you can uncomment:
    # out = out.interpolate(method='linear', limit_direction='both')

    trans_list = ", ".join(sorted(transforms)) or "none"
    lag_info = f"(lags={tuple(lags)})" if ("lag" in transforms or "lag_lchg" in transforms) else ""
    print("  ✓ Feature transforms added:", trans_list, lag_info)

    return df_activity, out, final_features
# ============================================================================
# 2. MEAN-REVERTING GROWTH RATE MODEL
# ============================================================================

class MeanRevertingGrowthModel:
    """
    Mean-reverting growth rate model with constrained optimization.
    
    Model: Δlog(y_{t+1}) = (1-ρ)μ + ρ*Δlog(y_t) + z_t'γ + ε
    
    Parameters:
        - μ: long-run mean growth rate
        - ρ: mean-reversion speed (constrained to [0, 1))
        - γ: coefficients for exogenous features
    """
    
    def __init__(self, 
                 include_ar: bool = True,
                 include_exog: bool = True,
                 l2_penalty: float = 1.0,
                 rho_bounds: Tuple[float, float] = (0.0, 0.99)):
        
        self.include_ar = include_ar
        self.include_exog = include_exog
        self.l2_penalty = l2_penalty
        self.rho_bounds = rho_bounds
        
        # Fitted parameters
        self.mu = None
        self.rho = None
        self.gamma = None
        self.scaler = None
        
        # For prediction
        self.last_growth_rate = None
        self.last_y = None
        
    def _objective(self, params: np.ndarray, growth_rates: np.ndarray, 
                   X_scaled: Optional[np.ndarray]) -> float:
        """
        Objective function: MSE + L2 penalty.
        
        params structure depends on configuration:
        - AR only: [μ, ρ]
        - Exog only: [μ, γ_1, ..., γ_k]
        - Both: [μ, ρ, γ_1, ..., γ_k]
        - Neither: [μ]
        """
        
        idx = 0
        mu = params[idx]
        idx += 1
        
        if self.include_ar:
            rho = params[idx]
            idx += 1
        else:
            rho = 0.0
        
        if self.include_exog and X_scaled is not None:
            gamma = params[idx:]
        else:
            gamma = np.array([])
        
        # Build predictions
        n = len(growth_rates) - 1
        y_pred = np.zeros(n)
        
        for i in range(n):
            # Δlog(y_{t+1}) = (1-ρ)μ + ρ*Δlog(y_t) + z_t'γ
            y_pred[i] = (1 - rho) * mu + rho * growth_rates[i]
            
            if self.include_exog and X_scaled is not None:
                y_pred[i] += np.dot(X_scaled[i], gamma)
        
        y_true = growth_rates[1:]
        
        # MSE
        mse = np.mean((y_true - y_pred) ** 2)
        
        # L2 penalty (exclude μ and ρ from regularization, only penalize γ)
        if self.include_exog and len(gamma) > 0:
            l2_term = self.l2_penalty * np.sum(gamma ** 2)
        else:
            l2_term = 0.0
        
        return mse + l2_term
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """
        Fit the mean-reverting growth model.
        
        Args:
            y: Annual consumption levels (shape: T,)
            X: Exogenous features (shape: T, k) - optional
        """
        y = np.asarray(y, dtype=float).flatten()
        
        # if len(y) < 3:
        #     raise ValueError("Need at least 3 years of data")
        
        # Compute growth rates
        growth_rates = np.diff(np.log(y))
        
        # Handle exogenous features
        X_scaled = None
        if self.include_exog and X is not None:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            # Align X with growth rates (skip first year)
            X_aligned = X[1:]
            
            # Handle NaN
            if np.isnan(X_aligned).any():
                col_means = np.nanmean(X_aligned, axis=0)
                for i in range(X_aligned.shape[1]):
                    X_aligned[np.isnan(X_aligned[:, i]), i] = col_means[i]
            
            # Standardize
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_aligned)
        
        # Initialize parameters
        mu_init = np.mean(growth_rates)
        
        params_init = [mu_init]
        bounds = [(None, None)]  # μ unbounded
        
        if self.include_ar:
            params_init.append(0.5)  # ρ initial guess
            bounds.append(self.rho_bounds)
        
        if self.include_exog and X_scaled is not None:
            k = X_scaled.shape[1]
            params_init.extend([0.0] * k)  # γ initial guess
            bounds.extend([(None, None)] * k)  # γ unbounded
        
        params_init = np.array(params_init)
        
        # Optimize
        result = minimize(
            fun=self._objective,
            x0=params_init,
            args=(growth_rates, X_scaled),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"⚠️  Optimization warning: {result.message}")
        
        # Extract fitted parameters
        idx = 0
        self.mu = result.x[idx]
        idx += 1
        
        if self.include_ar:
            self.rho = result.x[idx]
            idx += 1
        else:
            self.rho = 0.0
        
        if self.include_exog and X_scaled is not None:
            self.gamma = result.x[idx:]
        else:
            self.gamma = np.array([])
        
        # Store for prediction
        self.last_growth_rate = growth_rates[-1]
        self.last_y = y[-1]
        
        return self
    
    def predict(self, X_next: Optional[np.ndarray] = None) -> float:
        """
        Predict next year's consumption.
        
        Args:
            X_next: Exogenous features for next period
        
        Returns:
            Predicted consumption level
        """
        # Δlog(y_{t+1}) = (1-ρ)μ + ρ*Δlog(y_t) + z_t'γ
        predicted_growth = (1 - self.rho) * self.mu
        
        if self.include_ar:
            predicted_growth += self.rho * self.last_growth_rate
        
        if self.include_exog and X_next is not None and self.scaler is not None:
            X_next = np.asarray(X_next, dtype=float).flatten().reshape(1, -1)
            
            # Handle NaN
            if np.isnan(X_next).any():
                X_next = np.nan_to_num(X_next, nan=0.0)
            
            X_next_scaled = self.scaler.transform(X_next)
            predicted_growth += np.dot(X_next_scaled[0], self.gamma)
        
        # Convert to level
        y_pred = self.last_y * np.exp(predicted_growth)
        
        return float(y_pred)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = {
            'mu': self.mu,
            'rho': self.rho,
            'gamma': self.gamma,
            'include_ar': self.include_ar,
            'include_exog': self.include_exog,
            'l2_penalty': self.l2_penalty,
        }
        
        if self.include_ar:
            params['half_life'] = -np.log(2) / np.log(self.rho) if self.rho > 0 else np.inf
        
        return params


# ============================================================================
# 3. EVALUATION
# ============================================================================

def time_series_cv_evaluate(df_annual: pd.DataFrame,
                            feature_cols: list,
                            training_end: int = None,
                            train_window: int = None,
                            include_ar: bool = True,
                            include_exog: bool = True,
                            l2_penalty: float = 1.0,
                            rho_bounds: Tuple[float, float] = (0.0, 0.99)) -> Dict:
    """Evaluate model using time-series cross-validation."""
    
    results = {
        'predictions': [],
        'actuals': [],
        'years': [],
        'errors': [],
        'growth_rates_actual': [],
        'growth_rates_predicted': [],
        'params_history': [],
    }
    
    # Determine split
    if training_end is not None:
        split_idx = df_annual[df_annual['annee'] <= training_end].index[-1] + 1
    else:
        split_idx = len(df_annual) - 3
    # if split_idx < 3:
    #     raise ValueError("Not enough training data")
    
    print(f"\n{'='*70}")
    print(f"TIME-SERIES CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Model configuration:")
    print(f"  Include AR (ρ):        {include_ar}")
    print(f"  Include Exogenous (γ): {include_exog}")
    print(f"  L2 penalty:            {l2_penalty}")
    print(f"  ρ bounds:              {rho_bounds}")
    print(f"\nTraining years: {df_annual['annee'].iloc[0]} - {df_annual['annee'].iloc[split_idx-1]}")
    print(f"Test years: {df_annual['annee'].iloc[split_idx]} - {df_annual['annee'].iloc[-1]}")
    print(f"{'='*70}\n")
    
    # Iteratively predict each test year
    for i in range(len(df_annual) - split_idx):
        train_end_idx = split_idx + i
        if train_window is not None:
            start_idx = max(0, train_end_idx - train_window)
        else:
            start_idx = 0
        
        # Training data
        train_df = df_annual.iloc[start_idx:train_end_idx]
        y_train = train_df[VARIABLE].values
        X_train = train_df[feature_cols].values if (include_exog and feature_cols) else None
        
        # Test data
        test_year = df_annual.iloc[train_end_idx]['annee']
        y_actual = df_annual.iloc[train_end_idx][VARIABLE]
        X_test = df_annual.iloc[train_end_idx][feature_cols].values if (include_exog and feature_cols) else None
        
        # Fit model
        model = MeanRevertingGrowthModel(
            include_ar=include_ar,
            include_exog=include_exog,
            l2_penalty=l2_penalty,
            rho_bounds=rho_bounds
        )
        model.fit(y_train, X_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate growth rates
        growth_actual = (y_actual / y_train[-1] - 1) * 100
        growth_pred = (y_pred / y_train[-1] - 1) * 100
        
        # Store results
        results['predictions'].append(y_pred)
        results['actuals'].append(y_actual)
        results['years'].append(test_year)
        results['errors'].append(y_pred - y_actual)
        results['growth_rates_actual'].append(growth_actual)
        results['growth_rates_predicted'].append(growth_pred)
        results['params_history'].append(model.get_params())
        
        print(f"Year {int(test_year)}:")
        print(f"  Actual:    {y_actual:>15,.0f} kWh  (growth: {growth_actual:>6.2f}%)")
        print(f"  Predicted: {y_pred:>15,.0f} kWh  (growth: {growth_pred:>6.2f}%)")
        print(f"  Error:     {y_pred-y_actual:>15,.0f} kWh  ({(y_pred/y_actual-1)*100:>6.2f}%)")
        if include_ar:
            print(f"  μ={model.mu:.6f}, ρ={model.rho:.4f}")
        else:
            print(f"  μ={model.mu:.6f}")
        print()
    
    return results


def calculate_metrics(results: Dict) -> Dict:
    """Calculate evaluation metrics."""
    actuals = np.array(results['actuals'])
    predictions = np.array(results['predictions'])
    
    metrics = {
        'MAE': mean_absolute_error(actuals, predictions),
        'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
        'MAPE': np.mean(np.abs((actuals - predictions) / actuals)) * 100,
        'R2': r2_score(actuals, predictions),
        'Mean_Error': np.mean(results['errors']),
        'Std_Error': np.std(results['errors']),
        'Mean_Abs_Error_Pct': np.mean(np.abs(results['errors']) / actuals) * 100,
    }
    
    return metrics


# ============================================================================
# 4. VISUALIZATION
# ============================================================================

def plot_results(df_annual: pd.DataFrame, results: Dict, activity: str, region: str):
    """Create comprehensive visualization."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.4)
    
    # Plot 1: Full time series with predictions
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_annual['annee'], df_annual[VARIABLE] / 1e6, 
             'o-', label='Actual', linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(results['years'], np.array(results['predictions']) / 1e6, 
             's--', label='Predicted', linewidth=2, markersize=10, color='#A23B72')
    
    if results['years']:
        ax1.axvline(x=results['years'][0]-0.5, color='gray', 
                   linestyle='--', alpha=0.5, linewidth=2, label='Train/Test Split')
    
    ax1.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Consumption (GWh)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Annual Consumption: {activity} - {region}', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Prediction errors
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['#06A77D' if e >= 0 else '#D62246' for e in results['errors']]
    ax2.bar(results['years'], np.array(results['errors']) / 1e6, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error (GWh)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Errors', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Plot 3: Actual vs Predicted scatter
    ax3 = fig.add_subplot(gs[1, 1])
    actuals_gwh = np.array(results['actuals']) / 1e6
    preds_gwh = np.array(results['predictions']) / 1e6
    
    ax3.scatter(actuals_gwh, preds_gwh, s=150, alpha=0.7, color='#F18F01', edgecolors='black', linewidth=1.5)
    
    min_val = min(actuals_gwh.min(), preds_gwh.min())
    max_val = max(actuals_gwh.max(), preds_gwh.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
    
    for i, year in enumerate(results['years']):
        ax3.annotate(f'{int(year)}', (actuals_gwh[i], preds_gwh[i]), 
                    fontsize=9, ha='right', va='bottom')
    
    ax3.set_xlabel('Actual (GWh)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted (GWh)', fontsize=12, fontweight='bold')
    ax3.set_title('Actual vs Predicted', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Growth rates comparison
    ax4 = fig.add_subplot(gs[2, 0])
    x = np.arange(len(results['years']))
    width = 0.35
    
    ax4.bar(x - width/2, results['growth_rates_actual'], width, 
           label='Actual', alpha=0.8, color='#2E86AB', edgecolor='black')
    ax4.bar(x + width/2, results['growth_rates_predicted'], width, 
           label='Predicted', alpha=0.8, color='#A23B72', edgecolor='black')
    
    ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Growth Rates: Actual vs Predicted', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([int(y) for y in results['years']])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Plot 5: Historical growth rates
    ax5 = fig.add_subplot(gs[2, 1])
    all_growth = np.diff(np.log(df_annual[VARIABLE].values)) * 100
    years_growth = df_annual['annee'].values[1:]
    
    ax5.plot(years_growth, all_growth, 'o-', linewidth=2, markersize=7, color='#06A77D')
    ax5.axhline(y=np.mean(all_growth), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(all_growth):.2f}%')
    ax5.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Historical Growth Rates (Log Difference)', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    plt.savefig(f'growth_model_{activity}_{region.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Plot saved as 'growth_model_{activity}_{region.replace(' ', '_')}.png'")
    plt.show()


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("MEAN-REVERTING GROWTH RATE MODEL - REAL DATA ANALYSIS")
    print("="*70)
    print(f"Region: {TARGET_REGION}")
    print(f"Activity: {TARGET_ACTIVITY}")
    print(f"Variable: {VARIABLE}")
    print(f"Training end: {TRAINING_END}")
    print(f"Train window: {TRAIN_WINDOW}")
    print(f"\nModel Configuration:")
    print(f"  Include AR:        {INCLUDE_AR}")
    print(f"  Include Exogenous: {INCLUDE_EXOG}")
    print(f"  L2 Penalty:        {L2_PENALTY}")
    print(f"  ρ Bounds:          {RHO_BOUNDS}")
    print("="*70 + "\n")
    
    # Step 1: Load data
    print("STEP 1: LOADING DATA")
    print("-" * 70)
    df_regional = load_regional_data(TARGET_REGION, VARIABLE)
    df_features = load_features(TARGET_REGION)
    
    # Step 2: Prepare activity data
    print("\nSTEP 2: PREPARING ACTIVITY DATA")
    print("-" * 70)
    df_activity, df_annual, feature_cols = prepare_activity_data(df_regional, TARGET_ACTIVITY, df_features, transforms = FEATURE_TRANSFORMS, lags = EXOG_LAGS)
    
    # Define features to use
    feature_cols = feature_cols if INCLUDE_EXOG else []
    if feature_cols:
        print(f"\nUsing features: {feature_cols}")
    else:
        print(f"\nNo exogenous features used")
    
    # Step 3: Evaluate model
    print("\nSTEP 3: MODEL EVALUATION")
    print("-" * 70)
    
    results = time_series_cv_evaluate(
        df_annual=df_annual,
        feature_cols=feature_cols,
        training_end=TRAINING_END,
        train_window=TRAIN_WINDOW,
        include_ar=INCLUDE_AR,
        include_exog=INCLUDE_EXOG,
        l2_penalty=L2_PENALTY,
        rho_bounds=RHO_BOUNDS
    )
    
    # Step 4: Calculate metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    metrics = calculate_metrics(results)
    for metric_name, value in metrics.items():
        if 'MAPE' in metric_name or 'Pct' in metric_name:
            print(f"{metric_name:25s}: {value:>10.2f}%")
        elif 'MAE' in metric_name or 'RMSE' in metric_name or 'Error' in metric_name:
            print(f"{metric_name:25s}: {value:>10,.0f} kWh")
        else:
            print(f"{metric_name:25s}: {value:>10.4f}")
    
    # Step 5: Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_results(df_annual, results, TARGET_ACTIVITY, TARGET_REGION)
    
    # Step 6: Final forecast
    print("\n" + "="*70)
    print("FINAL FORECAST")
    print("="*70)
    
    y_all = df_annual[VARIABLE].values
    X_all = df_annual[feature_cols].values if (INCLUDE_EXOG and feature_cols) else None
    
    final_model = MeanRevertingGrowthModel(
        include_ar=INCLUDE_AR,
        include_exog=INCLUDE_EXOG,
        l2_penalty=L2_PENALTY,
        rho_bounds=RHO_BOUNDS
    )
    final_model.fit(y_all, X_all)
    
    # Use last row features for next year
    X_next = df_annual[feature_cols].iloc[-1].values if (INCLUDE_EXOG and feature_cols) else None
    next_year = int(df_annual['annee'].iloc[-1] + 1)
    next_year_pred = final_model.predict(X_next)
    
    last_actual = df_annual[VARIABLE].iloc[-1]
    implied_growth = (next_year_pred / last_actual - 1) * 100
    
    print(f"\nForecast for year {next_year}:")
    print(f"  Predicted consumption: {next_year_pred:>15,.0f} kWh ({next_year_pred/1e6:.2f} GWh)")
    print(f"  Last actual ({int(df_annual['annee'].iloc[-1])}):     {last_actual:>15,.0f} kWh ({last_actual/1e6:.2f} GWh)")
    print(f"  Implied growth rate:   {implied_growth:>15.2f}%")
    
    # Model parameters
    print(f"\n" + "="*70)
    print("MODEL PARAMETERS")
    print("="*70)
    params = final_model.get_params()
    print(f"Long-run mean (μ):     {params['mu']:>10.6f}  ({params['mu']*100:.4f}% annual growth)")
    if INCLUDE_AR:
        print(f"Mean-reversion (ρ):    {params['rho']:>10.6f}")
        if 'half_life' in params and np.isfinite(params['half_life']):
            print(f"Half-life:             {params['half_life']:>10.2f} years")
    if INCLUDE_EXOG and len(params['gamma']) > 0:
        print(f"\nExogenous coefficients (γ):")
        for i, (feat, coef) in enumerate(zip(feature_cols, params['gamma'])):
            print(f"  {feat:20s}: {coef:>10.6f}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()