import warnings
warnings.filterwarnings('ignore')

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Iterable, List, Mapping, Union
import inspect
from onee.utils import add_annual_client_feature, add_yearly_feature
from onee.data.names import Aliases
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from scipy.interpolate import lagrange, CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel as C, Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
import os
import re

# ═══════════════════════════════════════════════════════════════════════
# KERNEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════
# Available kernels for GaussianProcessForecastModel
# Keys can be used in config files to select a kernel
# Note: These are FACTORY FUNCTIONS that return fresh kernel instances
# to avoid mutation issues when GP optimizes kernel hyperparameters
KERNEL_REGISTRY = {
    "rbf_white": lambda: C(1.0, (1e-2, 1e3))
    * RBF(length_scale=3.0, length_scale_bounds=(2.0, 6.0))
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 0.5)),
    "matern_white": lambda: C(1.0, (1e-2, 1e3))
    * Matern(length_scale=3.0, length_scale_bounds=(1.0, 10.0), nu=0.5)
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 0.5)),
    "matern_short": lambda: C(1.0)
    * Matern(length_scale=0.5, length_scale_bounds=(0.1, 1.5), nu=0.5)
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 0.5)),
    "rbf_dot_white": lambda: C(1.0, (1e-2, 1e3))
    * RBF(length_scale=3.0, length_scale_bounds=(1.0, 10.0))
    + DotProduct(sigma_0=1.0)
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 0.5)),
    "matern_smooth": lambda: C(1.0, (1e-2, 1e3))
    * Matern(length_scale=5.0, length_scale_bounds=(2.0, 15.0), nu=2.5)
    + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-6, 0.2)),
    "rbf_long": lambda: C(1.0, (1e-2, 1e3))
    * RBF(length_scale=5.0, length_scale_bounds=(3.0, 10.0))
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 0.5)),
    "rational_quadratic": lambda: C(1.0) * RationalQuadratic(alpha=0.1, length_scale=1.0, length_scale_bounds=(0.1, 5.0)) + 
    WhiteKernel(noise_level=0.1)
}

def get_kernel(kernel_key: Optional[str]):
    """
    Get a NEW kernel instance from the registry by key.
    Returns None if kernel_key is None (will use default kernel).
    Raises ValueError if kernel_key is not found.
    
    Note: Each call returns a fresh kernel instance to avoid mutation
    issues when GaussianProcessRegressor optimizes kernel hyperparameters.
    """
    if kernel_key is None:
        return None
    if kernel_key not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel_key: '{kernel_key}'. Available: {list(KERNEL_REGISTRY.keys())}")
    # Call the factory function to get a fresh kernel instance
    return KERNEL_REGISTRY[kernel_key]()


def build_growth_rate_features(
    years: Iterable[int],
    feature_block: Iterable[str],
    df_features: pd.DataFrame,
    *,
    clients_lookup: Optional[Mapping[int, np.ndarray]] = None,
    use_clients: bool = False,
    df_monthly: pd.DataFrame = None,
    use_pf: bool = False,
    transforms: Iterable[str] = ("lag_lchg",),
    lags: Iterable[int] = (1,),
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Assemble feature matrix for the growth-rate model aligned with `years`.
    Returns `None` when no contextual features are available.
    """
    years_list = [int(y) for y in years]
    if not years_list:
        return None

    feature_cols = list(feature_block) if feature_block else []
    if not feature_cols and not (use_clients and clients_lookup) and not use_pf:
        return None

    df = df_features.copy()

    if Aliases.ANNEE not in df.columns:
        raise ValueError(f"df_features must contain an '{Aliases.ANNEE}' column.")

    cols = [Aliases.ANNEE] + feature_cols if feature_cols else [Aliases.ANNEE]
    df = (
        df[cols]
        .drop_duplicates(subset=[Aliases.ANNEE])
        .sort_values(Aliases.ANNEE)
        .set_index(Aliases.ANNEE)
    )

    if feature_cols:
        base_levels = df[feature_cols].astype(float)

        # Handle missing values similarly to prepare_activity_data
        for col in feature_cols:
            series = base_levels[col]
            if series.isnull().any():
                series_interp = series.interpolate(
                    method="linear", limit_direction="both"
                )
                if series_interp.isnull().any():
                    series_interp = series_interp.fillna(series.mean())
                base_levels[col] = series_interp
    else:
        base_levels = pd.DataFrame(index=df.index)

    transforms = tuple(transforms) if transforms else ()
    lags = tuple(lags) if lags else ()

    transformed_frames = []
    column_order: list[str] = []

    if "level" in transforms and not base_levels.empty:
        transformed_frames.append(base_levels)
        column_order.extend(base_levels.columns.tolist())

    lchg_df = None
    if ("lchg" in transforms or "lag_lchg" in transforms) and not base_levels.empty:
        lchg_df = pd.DataFrame(index=base_levels.index)
        for col in base_levels.columns:
            s = base_levels[col]
            prev = s.shift(1)
            valid = (s > 0) & (prev > 0)
            lchg = pd.Series(
                np.where(valid, np.log(s) - np.log(prev), np.nan), index=s.index
            )
            lchg = lchg.interpolate(method="linear", limit_direction="both").fillna(0.0)
            lchg_df[f"{col}_lchg"] = lchg

    if "lag" in transforms and not base_levels.empty:
        for lag in lags:
            lagged = base_levels.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in base_levels.columns]
            transformed_frames.append(lagged)
            column_order.extend(lagged.columns.tolist())

    if "lchg" in transforms and lchg_df is not None:
        transformed_frames.append(lchg_df)
        column_order.extend(lchg_df.columns.tolist())

    if "lag_lchg" in transforms and lchg_df is not None:
        for lag in lags:
            lagged_lchg = lchg_df.shift(lag)
            lagged_lchg.columns = [f"{col}_lag{lag}" for col in lchg_df.columns]
            transformed_frames.append(lagged_lchg)
            column_order.extend(lagged_lchg.columns.tolist())

    if transformed_frames:
        feature_df = pd.concat(transformed_frames, axis=1)
        feature_df = feature_df.reindex(columns=column_order)
    else:
        feature_df = pd.DataFrame(index=df.index)

    # Ensure numeric array aligned to requested years
    if feature_df.shape[1] == 0:
        feature_array = np.zeros((len(years_list), 0), dtype=float)
    else:
        feature_df = feature_df.astype(float)
        col_means = np.nanmean(feature_df.values, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)

        rows = []
        zero_row = np.zeros(feature_df.shape[1], dtype=float)
        for year in years_list:
            if year in feature_df.index:
                row = feature_df.loc[year].to_numpy(dtype=float)
                if np.isnan(row).any():
                    row = np.where(np.isnan(row), col_means, row)
            else:
                row = zero_row
            rows.append(row)
        feature_array = np.vstack(rows)

    if use_clients and clients_lookup:
        feature_array = add_annual_client_feature(
            feature_array, years_list, clients_lookup
        )
    
    if use_pf:
        feature_array = add_yearly_feature(
            feature_array, years_list, df_monthly, feature=Aliases.PUISSANCE_FACTUREE, agg_method="sum"
        )

    return feature_array if feature_array.size else None


class BaseTrendPrior(ABC):
    """
    Abstract strategy for the Deterministic Trend (Prior).
    """
    @abstractmethod
    def fit(self, years: np.ndarray, y_transformed: np.ndarray, X_exog: Optional[np.ndarray] = None):
        """
        Fit the trend line to the history.
        """
        pass

    @abstractmethod
    def predict(self, years: Union[np.ndarray, float, int], X_exog: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        """
        Predict the trend values. Accepts either a vector of years or a single year.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """
        Return fitted parameters (slope, intercept, etc.) for inspection.
        """
        pass


# ==========================================
# 2. LINEAR GROWTH PRIOR (The "Growth Floor")
# ==========================================
class LinearGrowthPrior(BaseTrendPrior):
    """
    Models a Linear Trend with an optional forced minimum slope (Growth Floor) 
    and an anchor to recent data.
    
    Equation: y = m*t + b
    Where 'm' is at least 'min_annual_growth' (if specified).
    
    Parameters
    ----------
    min_annual_growth : float or None
        Minimum slope constraint. If None, no constraint is applied and the 
        natural slope from linear regression is used.
    anchor_window : int or None
        Number of recent observations to anchor the intercept to.
        If None, uses all observations.
    """
    def __init__(self, min_annual_growth: Optional[float] = 0.02, anchor_window: Optional[int] = 3):
        self.min_annual_growth = min_annual_growth
        self.anchor_window = anchor_window
        
        # Fitted params
        self.forced_slope = None
        self.forced_intercept = None
        self.original_raw_slope = None

    def fit(self, years: np.ndarray, y_transformed: np.ndarray, X_exog: Optional[np.ndarray] = None):
        # 1. Fit Raw Linear Regression to get the "Natural" Slope
        # We use raw years (e.g. 2020) so the slope represents "Change per Year"
        X_years = np.array(years).reshape(-1, 1)
        
        lr = LinearRegression()
        lr.fit(X_years, y_transformed)
        
        self.original_raw_slope = float(lr.coef_[0])
        
        # 2. Apply the "Floor" (only if min_annual_growth is specified)
        # If natural slope is -0.05 but min is 0.01, we force 0.01.
        if self.min_annual_growth is not None:
            self.forced_slope = max(self.original_raw_slope, self.min_annual_growth)
        else:
            # No constraint - use natural slope
            self.forced_slope = self.original_raw_slope
        
        # 3. Anchor the Intercept to RECENT data
        # We pivot the new line so it passes through the average of the last N years.
        if self.anchor_window is not None:
            valid_window = min(self.anchor_window, len(y_transformed))
        else:
            valid_window = len(y_transformed)  # Use all data
        
        recent_y_mean = np.mean(y_transformed[-valid_window:])
        recent_x_mean = np.mean(X_years[-valid_window:])
        
        # b = y - mx
        self.forced_intercept = recent_y_mean - (self.forced_slope * recent_x_mean)

    def predict(self, years: Union[np.ndarray, float, int], X_exog: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        if self.forced_slope is None:
            raise RuntimeError("LinearGrowthPrior must be fitted before prediction.")

        # Handle Scalar Input
        is_scalar = np.isscalar(years)
        if is_scalar:
            years_arr = np.array([years])
        else:
            years_arr = np.array(years)

        # Calculation: y = mx + b
        result = (self.forced_slope * years_arr) + self.forced_intercept

        if is_scalar:
            return float(result[0])
        return result

    def get_params(self) -> Dict:
        return {
            "forced_slope": self.forced_slope,
            "forced_intercept": self.forced_intercept,
            "original_raw_slope": self.original_raw_slope,
        }


class PowerGrowthPrior(BaseTrendPrior):
    """
    Models a trend using a Power function: y = a * (t_relative)^p + b
    
    Parameters
    ----------
    power : float (0.0 < p <= 1.0)
        Controls the concavity.
        - 1.0 = Linear (No slowing down)
        - 0.5 = Square Root (Moderate slowing) - RECOMMENDED START
        - 0.1 = Very similar to Logarithmic (Fast slowing)
    min_annual_growth : float or None
        Minimum slope constraint. If None, no constraint is applied and the 
        natural slope from regression is used.
    anchor_window : int or None
        Number of recent observations to anchor the intercept to.
        If None, uses all observations.
    """
    def __init__(self, power: float = 0.5, anchor_window: Optional[int] = 3, min_annual_growth: Optional[float] = 0.02):
        self.power = power
        self.anchor_window = anchor_window
        self.min_annual_growth = min_annual_growth
        
        self.slope_a = None
        self.intercept_b = None
        self.start_year_ref = None
        self.original_raw_slope = None

    def _get_relative_time(self, years):
        # t=1, 2, 3...
        return (years - self.start_year_ref) + 1.0

    def fit(self, years: np.ndarray, y_transformed: np.ndarray, X_exog: Optional[np.ndarray] = None):
        self.start_year_ref = np.min(years)
        t_relative = self._get_relative_time(years)
        
        # Transform Time: X = t^p
        X_power = np.power(t_relative, self.power).reshape(-1, 1)
        
        # Fit Linear Regression on (t^p, y)
        lr = LinearRegression()
        lr.fit(X_power, y_transformed)
        
        self.original_raw_slope = float(lr.coef_[0])
        
        # Apply the "Floor" (only if min_annual_growth is specified)
        if self.min_annual_growth is not None:
            self.slope_a = max(self.original_raw_slope, self.min_annual_growth)
        else:
            # No constraint - use natural slope
            self.slope_a = self.original_raw_slope

        # Anchor Intercept
        if self.anchor_window is not None:
            valid_window = min(self.anchor_window, len(y_transformed))
        else:
            valid_window = len(y_transformed)  # Use all data
        recent_y_avg = np.mean(y_transformed[-valid_window:])
        recent_x_avg = np.mean(X_power[-valid_window:])
        
        self.intercept_b = recent_y_avg - (self.slope_a * recent_x_avg)

    def predict(self, years: Union[np.ndarray, float, int], X_exog: Optional[np.ndarray] = None):
        if self.slope_a is None:
            raise RuntimeError("PowerGrowthPrior must be fitted.")
            
        years_arr = np.array(years) if not np.isscalar(years) else np.array([years])
        
        t_relative = self._get_relative_time(years_arr)
        X_power = np.power(t_relative, self.power)
        
        result = (self.slope_a * X_power) + self.intercept_b
        
        if np.isscalar(years):
            return float(result[0])
        return result

    def get_params(self) -> Dict:
        return {
            "power": self.power,
            "slope_a": self.slope_a,
            "intercept_b": self.intercept_b,
            "original_raw_slope": self.original_raw_slope
        }

class FlatPrior(BaseTrendPrior):
    """
    A strictly neutral prior.
    It predicts a horizontal line at the mean (or median) of the history.
    
    Equation: y = constant
    
    Parameters
    ----------
    method : str
        How to compute the baseline value. Options: 'mean', 'median', 'last_value'
    anchor_window : int
        Number of recent observations to use for computing the baseline.
        If None, uses all observations.
    """
    def __init__(self, method="mean", anchor_window: int = None):
        # method can be 'mean', 'median', or 'last_value'
        self.method = method
        self.anchor_window = anchor_window
        self.baseline_value = None

    def fit(self, years: np.ndarray, y_transformed: np.ndarray, X_exog=None):
        # Apply anchor window if specified
        if self.anchor_window is not None:
            valid_window = min(self.anchor_window, len(y_transformed))
            y_windowed = y_transformed[-valid_window:]
        else:
            y_windowed = y_transformed
        
        if self.method == "mean":
            self.baseline_value = np.mean(y_windowed)
        elif self.method == "median":
            self.baseline_value = np.median(y_windowed)
        elif self.method == "last_value":
            # Very useful for 'Random Walk' assumptions
            self.baseline_value = y_transformed[-1]
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict(self, years: Union[np.ndarray, float, int], X_exog=None):
        if self.baseline_value is None:
            raise RuntimeError("FlatPrior must be fitted.")
            
        # Return the constant value for every requested year
        if np.isscalar(years):
            return float(self.baseline_value)
            
        return np.full_like(years, self.baseline_value, dtype=float)

    def get_params(self) -> Dict:
        return {
            "method": self.method,
            "anchor_window": self.anchor_window,
            "baseline_value": self.baseline_value
        }


class NeutralPrior(BaseTrendPrior):
    """
    A completely neutral prior that outputs 0 for all predictions.
    
    This leaves everything to the Gaussian Process - the GP will model
    the entire signal without any deterministic trend component.
    
    Use this when you want the GP to have full control over the predictions
    without any prior assumptions about trend direction or magnitude.
    
    Equation: y = 0 (always)
    """
    def __init__(self):
        self._fitted = False

    def fit(self, years: np.ndarray, y_transformed: np.ndarray, X_exog: Optional[np.ndarray] = None):
        # Nothing to fit - this prior is stateless
        self._fitted = True

    def predict(self, years: Union[np.ndarray, float, int], X_exog: Optional[np.ndarray] = None) -> Union[np.ndarray, float]:
        if not self._fitted:
            raise RuntimeError("NeutralPrior must be fitted before prediction.")
        
        # Return 0 for scalar or array of zeros for array input
        if np.isscalar(years):
            return 0.0
        
        return np.zeros_like(np.asarray(years), dtype=float)

    def get_params(self) -> Dict:
        return {
            "description": "No trend prior - GP models entire signal"
        }

class AugmentedConsensusPrior(PowerGrowthPrior):
    """
    A Trend Prior that adjusts its slope based on the Future Trajectory of an exogenous driver.
    Ensures continuity by pivoting the new trend around the recent historical data (Anchor).
    
    Parameters
    ----------
    power : float
        Controls the concavity of the trend (1.0 = linear).
    anchor_window : int or None
        Number of recent observations to anchor the intercept to.
    min_annual_growth : float or None
        Minimum slope constraint (growth floor).
    exog_col_idx : int
        Index of the exogenous column to use for driver trajectory.
    driver_weight : float
        Weight for blending historical slope with driver-implied slope (0-1).
    memory_decay : float
        Decay factor for weighting historical observations (0 < decay <= 1).
        Values < 1.0 give more weight to recent observations.
        Value of 1.0 gives equal weight to all observations (uses TheilSen regression).
    """
    def __init__(
        self, 
        power: float = 1.0, 
        anchor_window: Optional[int] = 3, 
        min_annual_growth: Optional[float] = 0.02,
        exog_col_idx: int = 0,
        driver_weight: float = 0.5,
        memory_decay: float = 0.7,
        use_dynamic_weights: bool = False,
    ):
        super().__init__(power, anchor_window, min_annual_growth)
        self.exog_col_idx = exog_col_idx
        self.driver_weight = driver_weight
        self.memory_decay = memory_decay
        self.use_dynamic_weights = use_dynamic_weights

        # Learned Params
        self.slope_hist = None
        self.r2_score = None
        self.scale_ratio = None  

        # Anchor Coordinates (The "Launch Pad")
        self.anchor_x_val = None
        self.anchor_y_val = None
        
    def fit(self, years: np.ndarray, y_transformed: np.ndarray, X_exog: Optional[np.ndarray] = None):
        self.start_year_ref = np.min(years)
        t_relative = self._get_relative_time(years)
        X_power = np.power(t_relative, self.power).reshape(-1, 1)
        
        if self.memory_decay < 1.0:
            # Calculate age of each point (0 for newest, 1 for previous...)
            ages = np.max(years) - years
            weights = np.power(self.memory_decay, ages)
            print(f"Using memory decay weights: {weights}")
            
            # Use LinearRegression because it supports sample_weight
            lr_time = LinearRegression()
            lr_time.fit(X_power, y_transformed, sample_weight=weights)
        else:
            # Use Robust Regression (TheilSen) for equal weights
            lr_time = LinearRegression(random_state=42)
            lr_time.fit(X_power, y_transformed)

        self.r2_score = lr_time.score(X_power, y_transformed)
        self.slope_hist = float(lr_time.coef_[0])

        # 2. Learn Sensitivity (Elasticity)
        if X_exog is not None:
            if X_exog.ndim == 1:
                exog_series = X_exog.reshape(-1, 1)
            else:
                exog_series = X_exog[:, self.exog_col_idx].reshape(-1, 1)
            
            std_y = np.std(y_transformed)
            std_exog = np.std(exog_series)

            numerator = std_y if std_y > 1e-9 else np.abs(np.mean(y_transformed))
            denominator = std_exog if std_exog > 1e-9 else np.abs(np.mean(exog_series))
            
            if denominator < 1e-9:
                self.scale_ratio = 0.0 
            else:
                self.scale_ratio = numerator / denominator
        else:
            self.scale_ratio = 0.0

        # 3. Store the Anchor Point (The Launch Pad)
        # We need this to calculate the intercept dynamically in predict()
        self._store_anchor_point(y_transformed, X_power)
        
        # Calculate initial intercept for inspection
        self.intercept_b = self.anchor_y_val - (self.slope_hist * self.anchor_x_val)
        self.slope_a = self.slope_hist

    def predict(self, years: Union[np.ndarray, float, int], X_exog: Optional[np.ndarray] = None):
        if self.slope_hist is None:
            raise RuntimeError("AugmentedConsensusPrior must be fitted.")

        years_arr = np.array(years) if not np.isscalar(years) else np.array([years])
        t_relative = self._get_relative_time(years_arr)
        X_power = np.power(t_relative, self.power)

        current_slope = self.slope_hist
        
        # --- DYNAMIC SLOPE CALCULATION ---
        if X_exog is not None and len(years_arr) > 1:
            if X_exog.ndim == 1:
                exog_series = X_exog.reshape(-1, 1)
            else:
                exog_series = X_exog[:, self.exog_col_idx].reshape(-1, 1)
            
            lr_future = LinearRegression()
            X_power_reshaped = X_power.reshape(-1, 1)
            lr_future.fit(X_power_reshaped, exog_series)
            
            s_future_exog = float(lr_future.coef_[0])
            s_pull = s_future_exog * self.scale_ratio
            current_slope = self._compute_consensus_slope(self.slope_hist, s_pull)

            print(f"Dynamic slope calculation: s_future_exog={s_future_exog}, scale_ratio={self.scale_ratio}, s_pull={s_pull}")
            print(f"R2 Score: {self.r2_score}")
            print(f"Historical slope: {self.slope_hist}")
            print(f"Computed consensus slope: {current_slope}")

        # --- APPLY GROWTH FLOOR ---
        if self.min_annual_growth is not None:
            current_slope = max(current_slope, self.min_annual_growth)
            
        # --- DYNAMIC INTERCEPT CALCULATION ---
        # Recalculate 'b' so the new line passes through the stored Anchor Point.
        # b = y_anchor - (m_new * x_anchor)
        current_intercept = self.anchor_y_val - (current_slope * self.anchor_x_val)

        # Final Calculation
        result = (current_slope * X_power) + current_intercept

        if np.isscalar(years):
            return float(result[0])
        return result

    def _store_anchor_point(self, y, X_power):
        """
        Calculates and stores the average X and Y of the recent window.
        This point acts as the pivot for any future slope changes.
        """
        if self.anchor_window is not None:
            valid_window = min(self.anchor_window, len(y))
        else:
            valid_window = len(y)
        
        self.anchor_y_val = np.mean(y[-valid_window:])
        self.anchor_x_val = np.mean(X_power[-valid_window:])


    def _compute_consensus_slope(self, s_hist: float, s_pull: float) -> float:
        """
        Combines History and Future Pull.
        If use_dynamic_weights is True, the weight depends on how linear the history was.
        """
        if s_pull == 0.0:
            if s_hist < 0.0:
                if abs(s_hist) < 10**8:
                    return s_hist * 2
                else:
                    s_hist = s_hist * 0.8
            else:
                if self.r2_score < 0.3:
                    return -1 * s_hist * 0.2
            
        
        if self.r2_score > 0.6:
            if s_hist > 0.0 and s_pull > 0.0:
                s_hist =  s_hist * 1.5
            if s_hist < 0.0 and s_pull < 0.0:
                s_hist =  s_hist * 2
        
        if s_pull > 0 and s_hist < 0:
            s_pull = s_pull * 0.5

        if self.use_dynamic_weights:
            # Here, we treat R2 as the "Trust in History".
            trust_in_history = np.clip(self.r2_score, 0.05, 0.9) 
            
            # driver_weight becomes (1 - trust)
            dynamic_driver_weight = 1.0 - trust_in_history
            
            return (s_hist * (1 - dynamic_driver_weight)) + (s_pull * dynamic_driver_weight)
        
        # Fallback to static weight
        return (s_hist * (1 - self.driver_weight)) + (s_pull * self.driver_weight)

    def get_params(self) -> Dict:
        return {
            "slope_hist": self.slope_hist,
            "scale_ratio": self.scale_ratio,
            "power": self.power,
            "driver_weight": self.driver_weight,
            "memory_decay": self.memory_decay,
            "anchor_coords": (self.anchor_x_val, self.anchor_y_val)
        }


class BaseForecastModel(ABC):
    """
    Abstract base class for all annual-level forecasting models.

    -------------------------
    GENERAL MODEL EXPECTATIONS
    -------------------------
    A forecasting model must:

    1. Accept historical annual consumption levels `y`
       and optionally exogenous features `X` in `fit()`.

    2. Internally estimate + store “fitted parameters”
       (e.g., AR coefficients, trend parameters, etc.).

    3. Support one-step-ahead predictions via `predict()`.

    4. Support multi-step annual forecasting via `forecast_horizon()`,
       updating its own internal state after each predicted step
       (e.g., last_y, last_growth_rate, latent states, etc.).

    5. Provide full parameter reporting via `get_params()`,
       separating hyperparameters from fitted parameters.

    These methods define a **standard interface** so that higher-level
    forecasting pipelines can treat all models uniformly
    (Mean-Reverting, Linear, Interpolation, Machine-Learning-based, etc.).
    """


    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(
        self,
        *,
        y: Optional[np.ndarray] = None,
        monthly_matrix: Optional[np.ndarray] = None,
        years: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        **kwargs
    ) -> "BaseForecastModel":
        """
        Fit the model using historical data.

        Parameters
        ----------
        y : np.ndarray, shape (T,), optional
            Annual consumption levels or target variable.
            Most models (e.g., growth-rate-based) require strictly positive values.

        monthly_matrix : np.ndarray, optional
            Optional monthly-level data (shape: T x 12 or T x M).
            Some models may use monthly information for seasonality extraction,
            smoothing, nonlinear interpolation, etc.

        years : np.ndarray, shape (T,), optional
            Year indices corresponding to the y series.
            Needed for models that regress on year, detect structural breaks, etc.

        X : np.ndarray, optional
            Optional exogenous features aligned with `y`.
            Shape: (T, k) or (T,) for single-feature input.

        kwargs : dict
            Any additional hyperparameters or configuration flags specific
            to the concrete model.

        Returns
        -------
        self : BaseForecastModel
            The fitted instance, enabling method chaining.

        Model Responsibilities
        ----------------------
        - Validate inputs (lengths, missing values, monotonicity constraints, etc.).
        - Compute any transformations (log, growth rates, scaling).
        - Estimate needed parameters (e.g., μ, ρ, β, γ).
        - Store internal state for prediction (last_y, last_growth_rate, latent state).
        """
        raise NotImplementedError


    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    @abstractmethod
    def predict(self, X_next: Optional[np.ndarray] = None, **kwargs) -> float:
        """
        One-step-ahead forecast using the model’s internal fitted state.

        Parameters
        ----------
        X_next : np.ndarray, optional
            Exogenous features for the upcoming period.
            Shape may be:
                - (k,)    → 1D feature vector
                - (1, k)  → 2D row vector
            If the model does not use exogenous features, this may be ignored.

        Returns
        -------
        float
            Predicted level of the target variable for the next year.

        Model Responsibilities
        ----------------------
        - Use stored fitted parameters to compute next-period forecast.
        - Use stored internal state (e.g., last_y, last_growth_rate).
        - Apply transformations (inverse-log, exponentiation, etc.) as needed.
        """
        raise NotImplementedError


    # ------------------------------------------------------------------
    # MULTI-STEP FORECASTING
    # ------------------------------------------------------------------
    @abstractmethod
    def forecast_horizon(
        self,
        start_year: int,
        horizon: int,
        df_features,
        df_monthly,
        feature_config: dict,
        monthly_clients_lookup=None
    ) -> List[Tuple[int, float]]:
        """
        Produce multi-year forecasts by repeatedly calling `predict()`
        and updating the model's internal state between steps.

        Parameters
        ----------
        start_year : int
            The last observed year. Forecasting begins at start_year + 1.

        horizon : int
            Number of years to forecast ahead (e.g., 10 → forecast 10 future years).

        df_features : DataFrame
            DataFrame containing historical and future exogenous features.
            Used to construct X_next for each predicted year via feature builder.

        df_monthly : DataFrame
            Monthly-level data used for models that extract seasonality,
            smooth interpolation, or compute monthly-derived features.

        feature_config : dict
            Configuration dictionary passed to the feature builder
            (e.g., which features to compute, moving-average windows, client filters).

        monthly_clients_lookup : optional
            Additional lookup table required by some feature configurations.

        Returns
        -------
        List[Tuple[int, float]]
            List of (year, predicted_value) pairs.
            Example: [(2025, 1342.5), (2026, 1401.2), ...]

        Model Responsibilities
        ----------------------
        - Build feature vector for each forecasted year via external feature builder.
        - Call `predict()` for each step.
        - Update internal state (last_y, last_growth_rate, or model-specific states)
          so that forecasts are iterative and not independent.
        """
        raise NotImplementedError


    # ------------------------------------------------------------------
    # PARAMETER REPORTING
    # ------------------------------------------------------------------
    @abstractmethod
    def get_params(self) -> Dict:
        """
        Return all model parameters (hyperparameters + fitted parameters).

        Structure
        ---------
        {
            "hyper_params": { ... },     # configuration, model architecture, constraints
            "fitted_params": { ... }     # learned / optimized quantities
        }

        Model Responsibilities
        ----------------------
        - Separate hyperparameters (fixed at initialization)
          from fitted parameters learned in `fit()`.
        - Include any derived quantities (e.g., half-life from AR(1) coefficient).
        - Ensure all parameters are JSON-serializable for logging and model diagnostics.

        Returns
        -------
        dict
            Dictionary containing full parameter reporting.
        """
        raise NotImplementedError


class ParamIntrospectionMixin:
    """
    Mixin providing:
    - warnings for unused parameters
    """

    def _process_init(self, locals_dict, class_obj):
        """
        locals_dict = local variables of child class __init__ (locals())
        class_obj    = the class itself, e.g. MeanRevertingGrowthModel
        """

        # unknown extra parameters
        extra_kwargs = locals_dict.get("kwargs", {})

        # warn about unknown parameters
        for k in extra_kwargs:
            print(
                f"⚠️ [{class_obj.__name__}] Unused parameter '{k}' was passed "
                f"and will be ignored."
            )


class IntensityForecastWrapper(BaseForecastModel):
    """
    A Wrapper that converts a GaussianProcessForecastModel into an 'Intensity' model.
    
    It trains the internal GP model on (y / normalization_col) and 
    scales the output back up by (normalization_col) during prediction.
    """
    SUPPORTED_HYPERPARAMS = [
        "normalization_col",
        "kernel_key",
        "n_restarts_optimizer",
        "normalize_y",
        "use_log_transform",
        "alpha",
        "prior_config",
        "remove_outliers",
        "outlier_threshold"
    ]


    def __init__(self, 
                 normalization_col: str = Aliases.TOTAL_ACTIVE_CONTRATS, 
                 # GP parameters
                 kernel_key: Optional[str] = None,
                 n_restarts_optimizer: int = 10,
                 normalize_y: bool = True,
                 use_log_transform: bool = False,
                 alpha: float = 1e-10,
                 # Prior configuration as dict
                 prior_config: Optional[Dict] = None,
                 # Outlier detection
                 remove_outliers: bool = False,
                 outlier_threshold: float = 2.5,
                 **kwargs):
        """
        Wrapper that normalizes predictions by a column (e.g., total_active_contrats).
        
        Parameters
        ----------
        normalization_col : str
            Column name to use for normalization
        kernel_key : str, optional
            Key to select a kernel from KERNEL_REGISTRY. If None, uses default kernel.
        n_restarts_optimizer : int
            Number of restarts for GP optimizer
        normalize_y : bool
            Whether to normalize y in GP
        use_log_transform : bool
            Whether to apply log transform
        alpha : float
            GP alpha parameter
        prior_config : dict
            Dictionary containing prior configuration with 'type' and corresponding parameters.
        remove_outliers : bool
            Whether to detect and downweight outliers during GP fitting.
        outlier_threshold : float
            Z-score threshold for outlier detection (using MAD).
        """
        self.norm_col = normalization_col
        # Default prior config if not provided
        if prior_config is None:
            prior_config = {"type": "PowerGrowthPrior", "power": 0.5, "anchor_window": 3, "min_annual_growth": 0.0}

        self.model = GaussianProcessForecastModel(
            kernel_key=kernel_key,
            use_log_transform=use_log_transform,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            alpha=alpha,
            prior_config=prior_config,
            remove_outliers=remove_outliers,
            outlier_threshold=outlier_threshold
        )
            
        self.fitted_params = {}

    def fit(self, *, y: Optional[np.ndarray] = None, years: Optional[np.ndarray] = None, X: Optional[np.ndarray] = None, normalization_arr: Optional[np.ndarray] = None, **kwargs):
        """
        Calculates Intensity = y / normalization_arr
        Fits the internal model on Intensity.
        """
        if y is None or normalization_arr is None:
            raise ValueError(f"IntensityWrapper requires 'y' and 'normalization_arr' (values of {self.norm_col}).")
        
        print(normalization_arr)
        self.history_norm_arr = np.array(normalization_arr, dtype=float)

        # 1. Calculate Intensity (Avoid division by zero)
        safe_norm = np.array(normalization_arr, dtype=float)
        safe_norm[safe_norm == 0] = 1.0
        
        y_intensity = y / safe_norm
        print(y_intensity)
        print("###################################################")
        
        # 2. Fit the internal model on Intensity
        self.model.fit(y=y_intensity, years=years, X=X, **kwargs)
        
        # 3. Store params
        self.fitted_params = {
            **self.model.get_params()
        }
        return self

    def forecast_horizon(self, start_year: int, horizon: int, df_features: pd.DataFrame, df_monthly, feature_config, monthly_clients_lookup=None):
        """
        Gets Intensity forecast from internal model, then multiplies by Future Contracts.
        """
        # 1. Validation: Ensure we have the future values for the normalization column
        if self.norm_col not in df_features.columns:
            raise ValueError(f"Future df_features must contain '{self.norm_col}' to reverse the intensity normalization.")
            
        # 2. Get Intensity Forecast (Mean, Lower, Upper)
        intensity_results = self.model.forecast_horizon(
            start_year, horizon, df_features, df_monthly, feature_config, monthly_clients_lookup
        )
        
        # 3. Get Future Normalization Factors (Contracts)
        future_years = [r[0] for r in intensity_results]
        future_factors = []
        
        for yr in future_years:
            # Safe lookup for the specific year
            val = df_features.loc[df_features[Aliases.ANNEE] == yr, self.norm_col].values
            if len(val) == 0:
                raise ValueError(f"Missing '{self.norm_col}' data for future year {yr}")
            future_factors.append(val[0])
            
        # 4. Scale Back Up
        final_results = []
        for i, (yr, mean, lower, upper) in enumerate(intensity_results):
            factor = future_factors[i]
            
            final_mean = mean * factor
            final_lower = lower * factor
            final_upper = upper * factor
            
            final_results.append((yr, float(final_mean), float(final_lower), float(final_upper)))
            
        return final_results

    def get_params(self):
        out = {}
        out["hyper_params"] = {
            "normalization_col": self.norm_col,
            **self.model.hyper_params  # Include all GP hyper params
        }
        out["fitted_params"] = self.fitted_params
        return out
    
    # Inside IntensityForecastWrapper
    def predict(self, X_next: Optional[np.ndarray] = None, normalization_factor = None, **kwargs) -> float:
        """
        Predicts total consumption.
        Requires the normalization value (e.g. contracts) to be passed via kwargs.
        
        Example:
        model.predict(X_next, total_active_contrats=75)
        """
        # 1. Get the Intensity Prediction (Base Model)
        # We pass kwargs down in case the internal model also needs them
        predicted_intensity = self.model.predict(X_next, **kwargs)
        
        if normalization_factor is None:
            raise ValueError(
                f"IntensityForecastWrapper requires '{self.norm_col}' to be passed as a keyword argument "
                f"to predict(). \nUsage: model.predict(X_next, normalization_factor=123)"
            )
            
        # 3. Scale Up
        return predicted_intensity * float(normalization_factor)

    # Pass-through plotting to the internal model, 
    # BUT we might need to handle the scale difference. 
    # Usually easier to just plot the results returned by forecast_horizon externally.
    def plot_forecast(
        self, 
        forecast_results: List[Tuple[int, float, float, float]], 
        title: Optional[str] = None, 
        save_plot: bool = False,
        save_folder: str = ".",
        df_monthly: Optional[pd.DataFrame] = None,
        target_col: str = Aliases.CONSOMMATION_KWH,
        X_exog: Optional[np.ndarray] = None,
        normalization_arr_full: Optional[np.ndarray] = None
    ):
        """
        Visualizes the TOTAL Consumption Forecast.
        
        It reconstructs the historical total consumption by multiplying the 
        internal model's training intensity by the stored normalization array.
        
        Parameters
        ----------
        forecast_results : List[Tuple[int, float, float, float]]
            List of (year, mean, lower, upper) tuples from forecast_horizon.
        title : str, optional
            Plot title.
        save_plot : bool
            Whether to save the plot to disk.
        save_folder : str
            Folder to save the plot.
        df_monthly : pd.DataFrame, optional
            Monthly data for plotting actuals.
        target_col : str
            Column name for the target variable.
        X_exog : np.ndarray, optional
            Pre-built exogenous features for the full timeline (history + forecast years).
        normalization_arr_full : np.ndarray, optional
            Normalization values for the full timeline (history + forecast years).
            Used to scale trend from intensity space to total consumption.
        """
        # 1. Reconstruct Historical Total Consumption
        # The internal model stores 'train_y' as Intensity.
        # We stored 'train_norm_arr' (contracts) in self.model during fit (if using the class structure I gave earlier)
        # Or simpler: we saved it in the wrapper during fit.
        
        # We need to access the training data. 
        # Since 'fit' passed data to self.model, let's grab it from there.
        history_years = self.model.train_years
        history_intensity = self.model.train_y
        
        # We need the history normalization array (Contracts).
        # In the previous step, I added 'self.train_norm_arr' to the GPR fit method.
        # But to be safe and cleaner, the WRAPPER should store what it saw during fit.
        
        # Let's assume we update fit() to store self.history_norm_arr. 
        # (See updated fit method below to ensure this attribute exists).
        if hasattr(self, 'history_norm_arr') and self.history_norm_arr is not None:
            history_total = history_intensity * self.history_norm_arr
        else:
            # Fallback if attribute missing (shouldn't happen with correct fit)
            print("Warning: Could not reconstruct total history (missing normalization array). Plotting Intensity.")
            history_total = history_intensity

        # 2. Unpack Forecast
        f_years = [x[0] for x in forecast_results]
        f_mean = [x[1] for x in forecast_results]
        f_lower = [x[2] for x in forecast_results]
        f_upper = [x[3] for x in forecast_results]

        plt.figure(figsize=(10, 6))

        # 3. Plot History (Total)
        plt.scatter(history_years, history_total, color='black', s=40, label='Historical Data (Total)', zorder=5)
        plt.plot(history_years, history_total, color='black', linestyle=':', alpha=0.3)

        # 4. Plot Forecast (Total)
        plt.plot(f_years, f_mean, color='blue', linewidth=2, label='Forecast (Mean)', zorder=4)
        plt.fill_between(f_years, f_lower, f_upper, color='blue', alpha=0.15, label='95% Confidence Interval', zorder=1)

        # --- Plot the Underlying Trend (scaled by normalization) ---
        # IMPORTANT: Predict train and forecast years SEPARATELY to ensure consistent prior behavior
        # The AugmentedConsensusPrior behaves differently based on input data (dynamic slope calculation)
        if self.model.prior is not None:
            history_years_sorted = np.sort(np.unique(history_years))
            f_years_arr = np.array(f_years)
            n_history = len(history_years_sorted)
            n_forecast = len(f_years_arr)
            
            # Split X_exog into history and forecast parts if provided
            X_exog_history = None
            X_exog_forecast = None
            if X_exog is not None:
                X_exog_history = X_exog[:n_history]
                X_exog_forecast = X_exog[n_history:n_history + n_forecast]
            
            # Predict trend for training years (without dynamic slope adjustment)
            trend_vals_history = self.model.prior.predict(history_years_sorted, X_exog=X_exog_history)
            
            # Predict trend for forecast years (with dynamic slope adjustment using future X_exog)
            trend_vals_forecast = self.model.prior.predict(f_years_arr, X_exog=X_exog_forecast)
            
            # Combine results
            full_timeline_sorted = np.concatenate([history_years_sorted, f_years_arr])
            trend_vals = np.concatenate([trend_vals_history, trend_vals_forecast])
            
            if self.model.hyper_params["use_log_transform"]:
                trend_vals = np.exp(trend_vals)
            
            # Scale trend by normalization factors to get total consumption trend
            if normalization_arr_full is not None:
                trend_scaled = trend_vals * normalization_arr_full
                plt.plot(full_timeline_sorted, trend_scaled, color='gray', linestyle='--', alpha=0.5, label='Underlying Trend')

        # 5. Plot Actuals
        ground_truth_map = {}
        if df_monthly is not None:
             if target_col in df_monthly.columns and Aliases.ANNEE in df_monthly.columns:
                ground_truth_map = df_monthly.groupby(Aliases.ANNEE)[target_col].sum().to_dict()

        if ground_truth_map:
            act_years = []
            act_values = []
            for y in f_years:
                if y in ground_truth_map:
                    act_years.append(y)
                    act_values.append(ground_truth_map[y])
            if act_years:
                plt.scatter(act_years, act_values, color='green', s=60, marker='D', label='Actual Ground Truth', zorder=6)

        # Formatting
        plot_title = title if title else f"Total Consumption Forecast (Normalized by {self.norm_col})"
        plt.title(plot_title, fontsize=14)
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

        # 6. Saving Logic
        if save_plot:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            safe_name = re.sub(r'[^\w\s-]', '', plot_title).strip().replace(' ', '_')
            if not safe_name: safe_name = "forecast_plot"
            file_path = os.path.join(save_folder, f"{safe_name}.png")
            plt.savefig(file_path, dpi=300)
            print(f"Plot saved to: {file_path}")

        plt.show()


class GaussianProcessForecastModel(ParamIntrospectionMixin, BaseForecastModel):
    
    SUPPORTED_HYPERPARAMS = [
        "kernel_key", 
        "alpha", 
        "n_restarts_optimizer", 
        "normalize_y", 
        "use_log_transform",
        "prior_config",
        "remove_outliers", 
        "outlier_threshold"
    ]


    def __init__(self, 
                 kernel_key: Optional[str] = None, 
                 alpha=1e-10, 
                 n_restarts_optimizer=10, 
                 normalize_y=True, 
                 random_state=42, 
                 use_log_transform=True, 
                 prior_config: Optional[Dict] = None,
                 remove_outliers: bool = False,
                 outlier_threshold: float = 2.5,
                 **kwargs
                 ):
        """
        GPR Model for probabilistic extrapolation.
        
        Parameters
        ----------
        kernel_key : str, optional
            Key to select a kernel from KERNEL_REGISTRY. 
            Available keys: 'rbf_white', 'matern_white', 'rbf_dot_white', 'matern_smooth', 'rbf_long'
            If None, uses default kernel (rbf_white).
        alpha : float
            Value added to the diagonal of the kernel matrix during fitting.
        n_restarts_optimizer : int
            Number of restarts of the optimizer for finding the kernel's parameters.
        normalize_y : bool
            Whether to normalize the target values y by removing the mean and scaling.
        prior_config : dict
            Dictionary containing prior configuration with 'type' and corresponding parameters.
            Example: {'type': 'LinearGrowthPrior', 'min_annual_growth': 0.02, 'anchor_window': 3}
            Supported types: 
                - 'LinearGrowthPrior': Linear trend with optional growth floor
                - 'PowerGrowthPrior': Power function trend (concave growth)
                - 'FlatPrior': Horizontal line at mean/median
                - 'NeutralPrior': Zero output, GP models entire signal
                - 'AugmentedConsensusPrior': Adjusts slope based on exogenous driver trajectory
        """
        self._process_init(locals(), GaussianProcessForecastModel)

        # Default prior config if not provided
        if prior_config is None:
            prior_config = {"type": "LinearGrowthPrior", "min_annual_growth": 0.02, "anchor_window": 3}

        self.hyper_params = {
            "kernel_key": kernel_key,
            "alpha": alpha,
            "n_restarts_optimizer": n_restarts_optimizer,
            "normalize_y": normalize_y,
            "use_log_transform": use_log_transform,
            "prior_config": prior_config,
            "remove_outliers": remove_outliers,
            "outlier_threshold": outlier_threshold
        }
        self.random_state = random_state
        
        # Initialize prior based on configuration dictionary
        prior_type = prior_config.get("type", "PowerGrowthPrior")
        
        if prior_type == "LinearGrowthPrior":
            self.prior = LinearGrowthPrior(
                min_annual_growth=prior_config.get("min_annual_growth", 0.0),
                anchor_window=prior_config.get("anchor_window", 3)
            )
        elif prior_type == "PowerGrowthPrior":
            self.prior = PowerGrowthPrior(
                power=prior_config.get("power", 0.5),
                anchor_window=prior_config.get("anchor_window", 3),
                min_annual_growth=prior_config.get("min_annual_growth", 0.02)
            )
        elif prior_type == "FlatPrior":
            self.prior = FlatPrior(
                method=prior_config.get("method", "mean"),
                anchor_window=prior_config.get("anchor_window", None)
            )
        elif prior_type == "NeutralPrior":
            self.prior = NeutralPrior()
        elif prior_type == "AugmentedConsensusPrior":
            self.prior = AugmentedConsensusPrior(
                power=prior_config.get("power", 1.0),
                anchor_window=prior_config.get("anchor_window", 3),
                min_annual_growth=prior_config.get("min_annual_growth", 0.02),
                exog_col_idx=prior_config.get("exog_col_idx", 0),
                driver_weight=prior_config.get("driver_weight", 0.5),
                memory_decay=prior_config.get("memory_decay", 0.7),
                use_dynamic_weights=prior_config.get("use_dynamic_weights", True)
            )
        else:
            raise ValueError(f"Unknown prior type: {prior_type}. Choose from: LinearGrowthPrior, PowerGrowthPrior, FlatPrior, NeutralPrior, AugmentedConsensusPrior")
        
        # Internal Model State
        self.model = None
        self.fitted_params = {}
        self.scaler_x = StandardScaler()
        
        # Data State
        self.train_years = None
        self.train_y = None
                

    def fit(self, *, y: Optional[np.ndarray] = None, years: Optional[np.ndarray] = None, X: Optional[np.ndarray] = None, **kwargs):
        if y is None or years is None:
            raise ValueError("GaussianProcessForecastModel requires 'y' and 'years'.")
        
        # 1. Transform Target Variable (Exponential Logic)
        if self.hyper_params["use_log_transform"]:
            if np.any(y <= 0):
                raise ValueError("Log transform requires strictly positive y.")
            y_transformed = np.log(y)
        else:
            y_transformed = y

        self.train_years = years
        self.train_y = y

        # The GP model doesn't care HOW the trend is calculated.
        self.prior.fit(years, y_transformed, X)
        
        # Get the trend values for history
        trend_values = self.prior.predict(years, X)

        # Calculate Residuals (Data - Trend)
        residuals = y_transformed - trend_values

        # --- NEW: OUTLIER REJECTION LOGIC ---
        # If enabled, we calculate dynamic alpha (noise) values.
        # Points far from the trend get HIGH alpha (model ignores them).
        
        base_alpha = self.hyper_params["alpha"]
        
        if self.hyper_params["remove_outliers"]:
            # A. Calculate Robust Z-Scores using MAD (Median Absolute Deviation)
            # We use MAD because standard 'mean/std' is heavily skewed by the outliers themselves.
            median_resid = np.median(residuals)
            abs_diff = np.abs(residuals - median_resid)
            mad = np.median(abs_diff)
            
            if mad == 0:
                # Perfect fit or constant residuals, no outliers
                alpha_input = base_alpha
                self.outliers_detected = []
            else:
                # 0.6745 scales MAD to be comparable to Standard Deviation
                robust_z_scores = 0.6745 * (residuals - median_resid) / mad
                
                # B. Identify Outliers
                threshold = self.hyper_params["outlier_threshold"]
                outlier_mask = np.abs(robust_z_scores) > threshold
                
                # C. Construct Dynamic Alpha Array
                # Initialize all with base_alpha
                dynamic_alpha = np.full_like(residuals, base_alpha)
                
                # Penalty: If outlier, multiply noise by 10,000 (effectively removing it)
                dynamic_alpha[outlier_mask] = base_alpha * 10000.0
                
                alpha_input = dynamic_alpha
                self.outliers_detected = years[outlier_mask]
        else:
            # Standard behavior: single float applied to all points
            alpha_input = base_alpha
            self.outliers_detected = []

        # ------------------------------------

        X_years_raw = np.array(years).reshape(-1, 1)
        if X is not None:
             if X.ndim == 1: X = X.reshape(-1, 1)
             X_full = np.hstack([X_years_raw, X])
        else:
             X_full = X_years_raw

        # 2. Unified Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_full)

        # 3. Kernel Definition
        kernel = get_kernel(self.hyper_params["kernel_key"])
        print("lllllllllllllllllllllllllllll")
        print(kernel)

        if kernel is None:
            # Default kernel: RBF + WhiteKernel
            # Note: We keep the kernel's own noise level small, 
            # because 'alpha_input' handles the per-point noise now.
            kernel = (
                C(1.0, (1e-2, 1e3)) * RBF(length_scale=3.0, length_scale_bounds=(2.0, 6.0)) + 
                WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 0.5))
            )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_input,  # <--- PASSING THE DYNAMIC ARRAY OR FLOAT
            n_restarts_optimizer=self.hyper_params["n_restarts_optimizer"],
            normalize_y=self.hyper_params["normalize_y"],
            random_state=self.random_state
        )

        self.model.fit(X_scaled, residuals)
        
        self.fitted_params = {
            "learned_kernel": str(self.model.kernel_),
            "log_marginal_likelihood": float(self.model.log_marginal_likelihood()),
            "outliers_removed_count": len(self.outliers_detected),
            **self.prior.get_params()
        }
        
        return self


    def predict(self, X_next: Optional[np.ndarray] = None, **kwargs) -> float:
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict()")
        
        X_next = np.array(X_next).reshape(1, -1)
        year_val = X_next[0, 0] # Assume 1st col is Year
        
        # 1. Predict Trend
        trend_val = self.prior.predict(year_val, X_next)

        X_scaled = self.scaler.transform(X_next)
        resid_val = self.model.predict(X_scaled)[0]

        final_pred = trend_val + resid_val

        if self.hyper_params["use_log_transform"]:
            return float(np.exp(final_pred))
        
        return float(final_pred)

    def forecast_horizon(
        self, 
        start_year: int, 
        horizon: int, 
        df_features: pd.DataFrame, 
        df_monthly: pd.DataFrame, 
        feature_config: dict, 
        monthly_clients_lookup=None
    ) -> List[Tuple[int, float, float, float]]:
        
        future_years = np.arange(start_year + 1, start_year + 1 + horizon)
        
        # 1. Build Exogenous Features
        X_exog = build_growth_rate_features(
            years=future_years,
            df_features=df_features,
            clients_lookup=monthly_clients_lookup,
            df_monthly=df_monthly,
            use_clients=feature_config.get("use_clients", False),
            feature_block=feature_config.get("feature_block", []),
            use_pf=feature_config.get("use_pf", False),
            transforms=feature_config.get("transforms", ("lag_lchg",)),
            lags=feature_config.get("lags", (1,))
        )

        # 2. Build Unified Matrix
        X_years = future_years.reshape(-1, 1)
        
        if X_exog is not None and X_exog.size > 0:
            X_full = np.hstack([X_years, X_exog])
        else:
            X_full = X_years

        # 3. Scale using the Unified Scaler
        try:
            X_scaled = self.scaler.transform(X_full)
        except ValueError as e:
            raise ValueError(
                f"Feature dimension mismatch. Model expects {self.scaler.n_features_in_} features "
                f"(Years + Exog), but forecast generated {X_full.shape[1]} features."
            ) from e

        # 4. Predict
        resid_mean, resid_std = self.model.predict(X_scaled, return_std=True)
        
        trend_vals = self.prior.predict(future_years, X_exog)

        results = []
        for i, year in enumerate(future_years):
            print(f"Year: {year}, Trend: {trend_vals[i]}, Residual Mean: {resid_mean[i]}, Residual Std: {resid_std[i]}")
            y_mean = trend_vals[i] + resid_mean[i]
            y_std = resid_std[i] # The trend is deterministic, so uncertainty comes only from GP
            
            if self.hyper_params["use_log_transform"]:
                # The model predicted log(y). The error is in log-units.
                # CI_log = Mean_log +/- 1.96 * Std_log
                log_lower = y_mean - 1.96 * y_std
                log_upper = y_mean + 1.96 * y_std
                
                # Transform back to Real Units
                final_mean = np.exp(y_mean)
                final_lower = np.exp(log_lower)
                final_upper = np.exp(log_upper)
            else:
                final_mean = y_mean
                final_lower = y_mean - 1.96 * y_std
                final_upper = y_mean + 1.96 * y_std
            
            results.append((int(year), float(final_mean), float(final_lower), float(final_upper)))
            
        return results
    def get_params(self) -> Dict:
        return {
            "hyper_params": self.hyper_params,
            "fitted_params": self.fitted_params
        }
    

    def plot_forecast(
        self, 
        forecast_results: List[Tuple[int, float, float, float]], 
        title: Optional[str] = None, 
        save_plot: bool = False,
        save_folder: str = ".",
        df_monthly: Optional[pd.DataFrame] = None,
        target_col: str = Aliases.CONSOMMATION_KWH,
        X_exog: Optional[np.ndarray] = None
    ):
        """
        Visualizes the forecast, confidence intervals, actuals, AND the underlying Growth Trend.
        
        Parameters
        ----------
        forecast_results : List[Tuple[int, float, float, float]]
            List of (year, mean, lower, upper) tuples from forecast_horizon.
        title : str, optional
            Plot title.
        save_plot : bool
            Whether to save the plot to disk.
        save_folder : str
            Folder to save the plot.
        df_monthly : pd.DataFrame, optional
            Monthly data for plotting actuals.
        target_col : str
            Column name for the target variable.
        X_exog : np.ndarray, optional
            Pre-built exogenous features for the full timeline (history + forecast years).
            Should be aligned with the concatenation of train_years and forecast years.
        """
        # Unpack Forecast
        f_years = [x[0] for x in forecast_results]
        f_mean = [x[1] for x in forecast_results]
        f_lower = [x[2] for x in forecast_results]
        f_upper = [x[3] for x in forecast_results]

        plt.figure(figsize=(10, 6))

        # 1. Plot History (Training Data)
        plt.scatter(self.train_years, self.train_y, color='black', s=40, label='Historical Data', zorder=5)
        
        # 2. Plot Forecast Mean
        plt.plot(f_years, f_mean, color='blue', linewidth=2, label='Forecast (Mean)', zorder=4)

        # 3. Plot Confidence Intervals
        plt.fill_between(f_years, f_lower, f_upper, color='blue', alpha=0.15, label='95% Confidence Interval', zorder=1)

        # --- Plot the "Growth Floor" (The Forced Trend) ---
        # This helps you see if the model is respecting your min_annual_growth
        # IMPORTANT: Predict train and forecast years SEPARATELY to ensure consistent prior behavior
        # The AugmentedConsensusPrior behaves differently based on input data (dynamic slope calculation)
        if self.prior is not None:
            train_years_sorted = np.sort(np.unique(self.train_years))
            f_years_arr = np.array(f_years)
            n_train = len(train_years_sorted)
            n_forecast = len(f_years_arr)
            
            # Split X_exog into history and forecast parts if provided
            X_exog_history = None
            X_exog_forecast = None
            if X_exog is not None:
                X_exog_history = X_exog[:n_train]
                X_exog_forecast = X_exog[n_train:n_train + n_forecast]
            
            # Predict trend for training years (without dynamic slope adjustment)
            trend_vals_history = self.prior.predict(train_years_sorted, X_exog=X_exog_history)
            
            # Predict trend for forecast years (with dynamic slope adjustment using future X_exog)
            trend_vals_forecast = self.prior.predict(f_years_arr, X_exog=X_exog_forecast)
            
            # Combine results
            full_timeline_sorted = np.concatenate([train_years_sorted, f_years_arr])
            trend_vals = np.concatenate([trend_vals_history, trend_vals_forecast])
            
            if self.hyper_params["use_log_transform"]:
                trend_vals = np.exp(trend_vals)
                
            plt.plot(full_timeline_sorted, trend_vals, color='gray', linestyle='--', alpha=0.5, label='Underlying Trend')

        # 4. Plot Actuals
        ground_truth_map = {}
        if df_monthly is not None:
            if target_col in df_monthly.columns and Aliases.ANNEE in df_monthly.columns:
                annual_agg = df_monthly.groupby(Aliases.ANNEE)[target_col].sum()
                ground_truth_map = annual_agg.to_dict()

        if ground_truth_map:
            act_years = []
            act_values = []
            for y in f_years:
                if y in ground_truth_map:
                    act_years.append(y)
                    act_values.append(ground_truth_map[y])
            
            if act_years:
                plt.scatter(act_years, act_values, color='green', s=60, marker='D', label='Actual Ground Truth', zorder=6)

        # Formatting
        plot_title = title if title else "Probabilistic Forecast vs Actuals"
        plt.title(plot_title, fontsize=14)
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

        # 5. Saving Logic
        if save_plot:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            safe_name = re.sub(r'[^\w\s-]', '', plot_title).strip().replace(' ', '_')
            if not safe_name:
                safe_name = "forecast_plot"
                
            file_path = os.path.join(save_folder, f"{safe_name}.png")
            plt.savefig(file_path, dpi=300)
            print(f"Plot saved to: {file_path}")

        plt.show()


class MeanRevertingGrowthModel(ParamIntrospectionMixin):
    """
    Mean-reverting growth rate model with constrained optimization.
    
    Model: Δlog(y_{t}) = (1-ρ)μ + ρ*Δlog(y_{t-1}) + β*[Δlog(y_{t-1})]² + z_t'γ + ε
    
    Parameters:
        - μ: long-run mean growth rate
        - ρ: mean-reversion speed (constrained to [0, 1))
        - β: squared AR term coefficient
        - γ: coefficients for exogenous features
        - use_asymmetric_loss: whether to use asymmetric loss that penalizes underestimation
        - underestimation_penalty: multiplier for underestimation errors (e.g., 2.0 or 3.0)
    """
    SUPPORTED_HYPERPARAMS = {
        "include_ar",
        "include_exog",
        "include_ar_squared",
        "l2_penalty",
        "rho_bounds",
        "beta_bounds",
        "mu_bounds",
        "use_asymmetric_loss",
        "underestimation_penalty"
    }
    
    def __init__(self, 
                 include_ar: bool = True,
                 include_ar_squared: bool = False,
                 include_exog: bool = True,
                 l2_penalty: float = 10.0,
                 rho_bounds: Tuple[float, float] = (0.0, 1),
                 beta_bounds: Tuple[float, float] = (-1.0, 1.0),
                 mu_bounds: Tuple[float, float] = (0.0, None),
                 use_asymmetric_loss: bool = False,
                 underestimation_penalty: float = 2.0, 
                 **kwargs):
        
        self._process_init(locals(), MeanRevertingGrowthModel)

        # -----------------------------------------------------------
        # 3. Normal assignments
        # -----------------------------------------------------------
        self.include_ar = include_ar
        self.include_ar_squared = include_ar_squared
        self.include_exog = include_exog
        self.l2_penalty = l2_penalty
        self.rho_bounds = rho_bounds
        self.beta_bounds = beta_bounds
        self.mu_bounds = mu_bounds
        self.use_asymmetric_loss = use_asymmetric_loss
        self.underestimation_penalty = underestimation_penalty
        
        # Fitted parameters
        self.mu = None
        self.rho = None
        self.beta = None
        self.gamma = None
        self.scaler = None
        
        # For prediction
        self.last_growth_rate = None
        self.last_y = None
        
    def _objective(self, params: np.ndarray, growth_rates: np.ndarray, 
                   X_scaled: Optional[np.ndarray]) -> float:
        """
        Objective function: MSE (or asymmetric loss) + L2 penalty.
        
        params structure depends on configuration:
        - AR only: [μ, ρ]
        - AR + AR²: [μ, ρ, β]
        - Exog only: [μ, γ_1, ..., γ_k]
        - AR + Exog: [μ, ρ, γ_1, ..., γ_k]
        - AR + AR² + Exog: [μ, ρ, β, γ_1, ..., γ_k]
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
        
        if self.include_ar_squared:
            beta = params[idx]
            idx += 1
        else:
            beta = 0.0
        
        if self.include_exog and X_scaled is not None:
            gamma = params[idx:]
        else:
            gamma = np.array([])
        
        # Build predictions
        n = len(growth_rates) - 1
        y_pred = np.zeros(n)
        
        for i in range(n):
            # Δlog(y_{t+1}) = (1-ρ)μ + ρ*Δlog(y_t) + β*[Δlog(y_t)]² + z_t'γ
            y_pred[i] = (1 - rho) * mu + rho * growth_rates[i]
            
            if self.include_ar_squared:
                y_pred[i] += beta * (growth_rates[i] ** 2)
            
            if self.include_exog and X_scaled is not None:
                y_pred[i] += np.dot(X_scaled[i], gamma)
        
        y_true = growth_rates[1:]
        errors = y_true - y_pred
        
        # Compute loss based on mode
        if self.use_asymmetric_loss:
            # Asymmetric loss: penalize underestimation more heavily
            # When y_true > y_pred (underestimation), apply penalty multiplier
            # When y_true < y_pred (overestimation), use standard squared error
            squared_errors = errors ** 2
            underestimation_mask = errors > 0  # True where we underestimated
            
            # Apply penalty multiplier to underestimation errors
            weighted_errors = squared_errors.copy()
            weighted_errors[underestimation_mask] *= self.underestimation_penalty
            
            loss = np.mean(weighted_errors)
        else:
            # Standard MSE
            loss = np.mean(errors ** 2)
        
        # L2 penalty (exclude μ, ρ, and β from regularization, only penalize γ)
        if self.include_exog and len(gamma) > 0:
            l2_term = self.l2_penalty * np.sum(gamma ** 2)
        else:
            l2_term = 0.0
        
        return loss + l2_term
    
    def fit(self, *, y=None, X=None, **kwargs):
        """
        Fit the mean-reverting growth model.
        
        Args:
            y: Annual consumption levels (shape: T,)
            X: Exogenous features (shape: T, k) - optional
        """
        y = np.asarray(y, dtype=float).flatten()
        
        if len(y) < 3:
            raise ValueError("Need at least 3 years of data")
        
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
        bounds = [self.mu_bounds]  # μ unbounded
        
        if self.include_ar:
            params_init.append(0.5)  # ρ initial guess
            bounds.append(self.rho_bounds)
        
        if self.include_ar_squared:
            params_init.append(0.0)  # β initial guess
            bounds.append(self.beta_bounds)
        
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
        
        if self.include_ar_squared:
            self.beta = result.x[idx]
            idx += 1
        else:
            self.beta = 0.0
        
        if self.include_exog and X_scaled is not None:
            self.gamma = result.x[idx:]
        else:
            self.gamma = np.array([])
        
        # Store for prediction
        self.last_growth_rate = growth_rates[-1]
        self.last_y = y[-1]
        
        return self
    
    def predict(self, X_next: Optional[np.ndarray] = None, **kwargs) -> float:
        """
        Predict next year's consumption.
        
        Args:
            X_next: Exogenous features for next period
        
        Returns:
            Predicted consumption level
        """
        # Δlog(y_{t+1}) = (1-ρ)μ + ρ*Δlog(y_t) + β*[Δlog(y_t)]² + z_t'γ
        predicted_growth = (1 - self.rho) * self.mu
        
        if self.include_ar:
            predicted_growth += self.rho * self.last_growth_rate
        
        if self.include_ar_squared:
            predicted_growth += self.beta * (self.last_growth_rate ** 2)
        
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
    

    def forecast_horizon(
        self,
        start_year: int,
        horizon: int,
        df_features,
        df_monthly,
        feature_config: dict,
        monthly_clients_lookup=None,
    ):
        preds = []

        for h in range(1, horizon + 1):

            target_year = start_year + h

            # Build features for the forecast year
            x_next = build_growth_rate_features(
                years=[target_year],
                df_features=df_features,
                df_monthly=df_monthly,
                clients_lookup=monthly_clients_lookup,
                **feature_config
            )

            # Convert to numpy if available
            if x_next is not None:
                x_next = np.asarray(x_next, dtype=float)
                x_input = x_next[0]
            else:
                x_input = None

            # Predict annual consumption level
            y_pred = self.predict(x_input)

            preds.append((target_year, y_pred))

            # Update internal state for iterative forecasts
            predicted_growth_rate = np.log(y_pred) - np.log(self.last_y)
            self.last_growth_rate = predicted_growth_rate
            self.last_y = y_pred

        return preds

    def get_params(self) -> Dict:
        """Get model parameters split between hyper and fitted params."""
        hyper = {
            'include_ar': getattr(self, 'include_ar', None),
            'include_ar_squared': getattr(self, 'include_ar_squared', None),
            'include_exog': getattr(self, 'include_exog', None),
            'l2_penalty': getattr(self, 'l2_penalty', None),
            'rho_bounds': getattr(self, 'rho_bounds', None),
            'beta_bounds': getattr(self, 'beta_bounds', None),
            'use_asymmetric_loss': getattr(self, 'use_asymmetric_loss', None),
            'underestimation_penalty': getattr(self, 'underestimation_penalty', None),
        }

        fitted = {
            'mu': getattr(self, 'mu', None),
            'rho': getattr(self, 'rho', None),
            'beta': getattr(self, 'beta', None),
            'gamma': getattr(self, 'gamma', None),
        }

        # half-life is a derived fitted quantity for AR(1)
        if hyper.get('include_ar'):
            rho_val = fitted.get('rho')
            try:
                fitted['half_life'] = -np.log(2) / np.log(rho_val) if rho_val is not None and rho_val > 0 else np.inf
            except Exception:
                fitted['half_life'] = None

        return {
            'hyper_params': hyper,
            'fitted_params': fitted,
        }
