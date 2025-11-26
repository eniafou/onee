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
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Iterable, List, Mapping
import inspect
from onee.utils import add_annual_client_feature, add_yearly_feature
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from scipy.interpolate import lagrange, CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel as C, Matern




# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[0]

# Database paths
DB_REGIONAL_PATH = PROJECT_ROOT / 'data/ONEE_Regional_COMPLETE_2007_2023.db'
DB_DIST_PATH = PROJECT_ROOT / 'data/ONEE_Distributeurs_consumption.db'

# Analysis settings
TARGET_REGION = "Casablanca-Settat"
TARGET_ACTIVITY = "Administratif" #"Menages"
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

    if "annee" not in df.columns:
        raise ValueError("df_features must contain an 'annee' column.")

    cols = ["annee"] + feature_cols if feature_cols else ["annee"]
    df = (
        df[cols]
        .drop_duplicates(subset=["annee"])
        .sort_values("annee")
        .set_index("annee")
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
            feature_array, years_list, df_monthly, feature="puissance facturée", agg_method="sum"
        )

    return feature_array if feature_array.size else None

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
    def predict(self, X_next: Optional[np.ndarray] = None) -> float:
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


class GaussianProcessForecastModel(ParamIntrospectionMixin, BaseForecastModel):
    
    SUPPORTED_HYPERPARAMS = [
        "kernel", 
        "alpha", 
        "n_restarts_optimizer", 
        "normalize_y", 
        "random_state"
    ]

    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=10, normalize_y=True, random_state=42, use_log_transform=True):
        """
        GPR Model for probabilistic extrapolation.
        
        Parameters
        ----------
        kernel : sklearn.kernels.Kernel, optional
            The kernel specifying the covariance function. 
            Default: Constant * RBF (smooth) + DotProduct (trend) + White (noise)
        alpha : float
            Value added to the diagonal of the kernel matrix during fitting.
        n_restarts_optimizer : int
            Number of restarts of the optimizer for finding the kernel's parameters.
        normalize_y : bool
            Whether to normalize the target values y by removing the mean and scaling.
        """
        self._process_init(locals(), GaussianProcessForecastModel)

        self.hyper_params = {
            "kernel": kernel,
            "alpha": alpha,
            "n_restarts_optimizer": n_restarts_optimizer,
            "normalize_y": normalize_y,
            "random_state": random_state,
            "use_log_transform": use_log_transform
        }
        
        # Internal Model State
        self.model = None
        self.fitted_params = {}
        
        # Data State
        self.train_years = None
        self.train_y = None
        
        # Scaling State (CRITICAL for fixing exploding intervals)
        self.scaler_x = StandardScaler()
        self.train_features_scaled = None

    def fit(self, *, y: Optional[np.ndarray] = None, years: Optional[np.ndarray] = None, X: Optional[np.ndarray] = None, **kwargs):
        if y is None or years is None:
            raise ValueError("GaussianProcessForecastModel requires 'y' and 'years'.")
        
        # 1. Transform Target Variable (Exponential Logic)
        if self.hyper_params["use_log_transform"]:
            # We use log1p to be safe against zeros (log(1+x))
            # If your data is large (10^12), standard log is fine, but log1p is generally safer.
            if np.any(y <= 0):
                raise ValueError("Log transform requires strictly positive y (or non-negative for log1p).")
            self.train_y_transformed = np.log(y)
        else:
            self.train_y_transformed = y

        # 1. Build Unified Feature Matrix (Years + Exogenous)
        X_years = np.array(years).reshape(-1, 1)

        if X is not None:
             if X.ndim == 1: X = X.reshape(-1, 1)
             X_full = np.hstack([X_years, X])
        else:
             X_full = X_years

        # 2. Unified Scaling
        # Use a single scaler for the entire matrix to ensure consistency
        self.scaler = StandardScaler()
        self.train_features_scaled = self.scaler.fit_transform(X_full)

        self.train_years = years
        self.train_y = y

        # 3. Kernel Definition
        # Defaulting to Matern + WhiteKernel to prevent exploding extrapolation
        if self.hyper_params["kernel"] is None:
            # RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
            # kernel = (
            #     C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + 
            #     WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
            # )
            kernel = (C(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + 
                      DotProduct(sigma_0=1.0) + 
                      WhiteKernel(noise_level=1.0))
        else:
            kernel = self.hyper_params["kernel"]

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.hyper_params["alpha"],
            n_restarts_optimizer=self.hyper_params["n_restarts_optimizer"],
            normalize_y=self.hyper_params["normalize_y"],
            random_state=self.hyper_params["random_state"]
        )

        self.model.fit(self.train_features_scaled, self.train_y_transformed)
        
        self.fitted_params = {
            "learned_kernel": str(self.model.kernel_),
            "log_marginal_likelihood": float(self.model.log_marginal_likelihood())
        }
        
        return self

    def predict(self, X_next: Optional[np.ndarray] = None) -> float:
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict()")
        
        # Scale input using the unified scaler
        # X_next must match the shape [Years, Exog] used in fit
        X_next = np.array(X_next).reshape(1, -1)
        X_scaled = self.scaler.transform(X_next)

        pred = self.model.predict(X_scaled)

        if self.hyper_params["use_log_transform"]:
            return float(np.exp(pred[0]))
        
        return float(pred[0])

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
            feature_block=feature_config.get("feature_block", []),
            df_features=df_features,
            clients_lookup=monthly_clients_lookup,
            use_clients=feature_config.get("use_clients", False),
            df_monthly=df_monthly,
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
        y_mean, y_std = self.model.predict(X_scaled, return_std=True)
        
        results = []
        for i, year in enumerate(future_years):
            if self.hyper_params["use_log_transform"]:
                # The model predicted log(y). The error is in log-units.
                # CI_log = Mean_log +/- 1.96 * Std_log
                log_lower = y_mean[i] - 1.96 * y_std[i]
                log_upper = y_mean[i] + 1.96 * y_std[i]
                
                # Transform back to Real Units
                final_mean = np.exp(y_mean[i])
                final_lower = np.exp(log_lower)
                final_upper = np.exp(log_upper)
            else:
                final_mean = y_mean[i]
                final_lower = y_mean[i] - 1.96 * y_std[i]
                final_upper = y_mean[i] + 1.96 * y_std[i]
            
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
        target_col: str = "consommation_kwh"
    ):
        """
        Visualizes the forecast and optionally compares it against actuals derived 
        directly from monthly data.

        Parameters
        ----------
        df_monthly : pd.DataFrame, optional
            Raw monthly data containing 'annee' and 'target_col'. 
            If provided, it will be aggregated (Sum) to create annual ground truth.
        target_col : str
            The name of the value column in df_monthly to sum up (default: 'valeur').
        """
        import os
        import re

        # Unpack Forecast
        f_years = [x[0] for x in forecast_results]
        f_mean = [x[1] for x in forecast_results]
        f_lower = [x[2] for x in forecast_results]
        f_upper = [x[3] for x in forecast_results]

        plt.figure(figsize=(10, 6))

        # 1. Plot History (Training Data)
        plt.scatter(self.train_years, self.train_y, color='black', s=40, label='Historical Data', zorder=5)
        plt.plot(self.train_years, self.train_y, color='black', linestyle=':', alpha=0.3)

        # 2. Plot Forecast Mean
        plt.plot(f_years, f_mean, color='blue', linewidth=2, label='Forecast (Mean)', zorder=4)

        # 3. Plot Confidence Intervals
        plt.fill_between(f_years, f_lower, f_upper, color='blue', alpha=0.15, label='95% Confidence Interval', zorder=1)

        # 4. Logic to Prepare Actuals (Ground Truth)
        ground_truth_map = {}
        
        if df_monthly is not None:
            if target_col not in df_monthly.columns or "annee" not in df_monthly.columns:
                print(f"Warning: df_monthly is missing '{target_col}' or 'annee'. Skipping actuals plot.")
            else:
                # Group by Year and Sum
                annual_agg = df_monthly.groupby("annee")[target_col].sum()
                ground_truth_map = annual_agg.to_dict()

        # 5. Plot Actuals
        if ground_truth_map:
            # We only plot actuals that overlap with the FORECAST range 
            # (to check accuracy), or you can plot all to see the full picture.
            # Here we plot ANY actuals that exist in the forecast range.
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

        # 6. Saving Logic
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





class LocalInterpolationForecastModel(ParamIntrospectionMixin, BaseForecastModel):
    """
    Local curve-fitting interpolation-based forecast model.

    For each prediction step:
        - Take the last `window_size` years (or full history).
        - Fit several candidate models (linear, quadratic, spline, lagrange...).
        - Select the one with best metric (In-Sample RMSE or Cross-Validation Error).
        - Extrapolate 1 step forward.
        - Slide window (if not using full history) and repeat for multi-step forecasting.
    """
    SUPPORTED_HYPERPARAMS = {
        "window_size", "candidate_models", "weighted", "weight_decay", 
        "min_window", "positive_only", "use_full_history", "selection_mode",
        "fit_on_growth_rates" # NEW
    }

    def __init__(
        self,
        window_size: int = 5,
        candidate_models: List[str] = ("linear", "quadratic", "exponential", "logarithmic", "spline", "lagrange"),
        weighted: bool = True,
        weight_decay: float = 0.9,
        min_window: int = 3,
        positive_only: bool = False,
        use_full_history: bool = False,
        selection_mode: str = "cross_validation",
        fit_on_growth_rates: bool = True, # NEW
        **kwargs
    ):
        self._process_init(locals(), LocalInterpolationForecastModel)
        self.window_size = window_size
        self.candidate_models = list(candidate_models)
        self.weighted = weighted
        self.weight_decay = weight_decay
        self.min_window = min_window
        self.positive_only = positive_only
        self.use_full_history = use_full_history
        self.selection_mode = selection_mode
        self.fit_on_growth_rates = fit_on_growth_rates # NEW

        self.last_years = None
        self.last_values = None


    # ---------------------------------------------------------
    # Utility: compute weights
    # ---------------------------------------------------------
    def _compute_weights(self, n: int):
        if not self.weighted:
            return np.ones(n)
        # weights: [1, λ, λ^2, ...] from oldest to most recent
        exponents = np.arange(n-1, -1, -1)
        return np.power(self.weight_decay, exponents)

    # ---------------------------------------------------------
    # Fit individual candidate models
    # ---------------------------------------------------------
    def _fit_linear(self, years, values, weights):
        # y = a + b * t
        lr = LinearRegression()
        lr.fit(years.reshape(-1, 1), values, sample_weight=weights)
        def predict(t):
            return lr.predict(np.array([[t]]))[0]
        return predict

    def _fit_quadratic(self, years, values, weights):
        # y = a + b t + c t^2
        X = np.column_stack([years, years**2])
        lr = LinearRegression()
        lr.fit(X, values, sample_weight=weights)
        def predict(t):
            Xt = np.array([[t, t*t]])
            return lr.predict(Xt)[0]
        return predict

    def _fit_exponential(self, years, values, weights):
        # y = A e^(Bt) => log(y) = log(A) + B t
        if np.any(values <= 0):
            return None  # invalid for exponential
        lr = LinearRegression()
        lr.fit(years.reshape(-1, 1), np.log(values), sample_weight=weights)
        A = np.exp(lr.intercept_)
        B = lr.coef_[0]
        def predict(t):
            return A * np.exp(B * t)
        return predict

    def _fit_logarithmic(self, years, values, weights):
        # y = A + B ln(t)
        # Shift time to ensure log domain validity if years are 0-indexed or negative
        t_shift = 1 - np.min(years) if np.min(years) <= 0 else 0
        years_log = np.log(years + t_shift)
        
        lr = LinearRegression()
        lr.fit(years_log.reshape(-1, 1), values, sample_weight=weights)
        
        def predict(t):
            if (t + t_shift) <= 0: return values[-1] # Safety fallback
            return lr.predict(np.array([[np.log(t + t_shift)]]))[0]
        return predict

    def _fit_lagrange(self, years, values, weights):
        # Lagrange Interpolation (Exact fit)
        # Numerical stability fix: shift years to start at 0
        t0 = years[0]
        years_shifted = years - t0
        
        # scipy.interpolate.lagrange returns a numpy polynomial object
        poly = lagrange(years_shifted, values)
        
        def predict(t):
            return poly(t - t0)
        return predict

    def _fit_spline(self, years, values, weights):
        # Cubic Spline Interpolation (Exact fit, smoother than Lagrange)
        cs = CubicSpline(years, values, bc_type='natural')
        return lambda t: cs(t)

    # ---------------------------------------------------------
    # Model selection
    # ---------------------------------------------------------
    def _evaluate_in_sample(self, predict_fn, years, values, weights):
        """Standard RMSE on training data"""
        preds = np.array([predict_fn(t) for t in years])
        errors = preds - values
        return np.sqrt(np.average(errors**2, weights=weights))

    def _select_best_model(self, years, values):
        n = len(values)
        if n < self.min_window:
            return lambda t: values[-1], "constant"

        years = np.asarray(years, float)
        values = np.asarray(values, float)

        # Mapping names to functions
        fitters = {
            "linear": self._fit_linear,
            "quadratic": self._fit_quadratic,
            "exponential": self._fit_exponential,
            "logarithmic": self._fit_logarithmic,
            "lagrange": self._fit_lagrange,
            "spline": self._fit_spline,
        }

        # -----------------------------------------------------
        # STRATEGY 1: CROSS VALIDATION (The "Fair" Way)
        # -----------------------------------------------------
        if self.selection_mode == "cross_validation":
            # Train on T[:-1], Test on T[-1]
            train_y = years[:-1]
            train_v = values[:-1]
            test_y = years[-1]
            test_v = values[-1]
            
            # If not enough data for CV, fallback to linear on full data
            if len(train_v) < 2:
                weights = self._compute_weights(n)
                return self._fit_linear(years, values, weights), "linear_fallback"

            train_weights = self._compute_weights(len(train_v))
            
            best_name = "linear" # default
            best_error = np.inf

            for name in self.candidate_models:
                if name not in fitters: continue
                # interpolators need enough points
                if name in ["quadratic", "lagrange", "spline"] and len(train_v) < 3: continue 

                try:
                    fn = fitters[name](train_y, train_v, train_weights)
                    if fn is None: continue
                    
                    pred = fn(test_y)
                    error = abs(pred - test_v)
                    
                    if error < best_error:
                        best_error = error
                        best_name = name
                except:
                    continue
            
            # REFIT best model on ALL data
            final_weights = self._compute_weights(n)
            final_fn = fitters.get(best_name, self._fit_linear)(years, values, final_weights)
            return final_fn, best_name

        # -----------------------------------------------------
        # STRATEGY 2: IN-SAMPLE RMSE (The "Interpolation wins" Way)
        # -----------------------------------------------------
        else:
            weights = self._compute_weights(n)
            best_name = None
            best_fn = None
            best_rmse = np.inf

            for name in self.candidate_models:
                if name not in fitters: continue
                if name in ["quadratic", "lagrange", "spline"] and n < 3: continue

                try:
                    fn = fitters[name](years, values, weights)
                    if fn is None: continue
                    
                    rmse = self._evaluate_in_sample(fn, years, values, weights)
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_name = name
                        best_fn = fn
                except:
                    continue

            if best_fn is None:
                return lambda t: values[-1], "constant"

            return best_fn, best_name

    def fit(self, *, y=None, monthly_matrix=None, years=None, X=None, **kwargs):
        if y is None:
            raise ValueError("y must be provided")

        self.last_values = np.asarray(y, float)
        if years is not None:
            self.last_years = np.asarray(years, int)
        else:
            self.last_years = np.arange(len(self.last_values))
        return self
    
    def _to_growth_rates(self, years, values):
        """
        Converts:
           Years:  [2020, 2021, 2022]
           Values: [100,  110,  121]
        To:
           Years:  [2021, 2022] (Associated with the 'end' of the period)
           Rates:  [0.095, 0.095] (approx 10%) -> log(110/100), log(121/110)
        """
        if np.any(values <= 0):
            # Cannot take logs of non-positive numbers. 
            # Fallback strategy: return unchanged (or raise error)
            # Here we just disable the feature silently for safety
            return years, values, False

        log_vals = np.log(values)
        rates = np.diff(log_vals) # size becomes N-1
        rate_years = years[1:]    # align with the result year
        
        return rate_years, rates, True

    def _predict_one(self):
        values = self.last_values
        years = self.last_years

        # 1. Select Data Context
        if self.use_full_history:
            local_values = values
            local_years = years
        else:
            window = min(self.window_size, len(values))
            local_values = values[-window:]
            local_years = years[-window:]

        # 2. Check: Growth Rate Mode
        is_growth_mode = False
        if self.fit_on_growth_rates:
            # Transform data to rates
            # Note: We need at least 2 points to get 1 rate. 
            # If we have 3 points, we get 2 rates.
            rate_years, rates, success = self._to_growth_rates(local_years, local_values)
            if success and len(rates) >= self.min_window:
                local_years = rate_years
                local_values = rates
                is_growth_mode = True

        # 3. Fit and Predict
        # If in growth mode, we are predicting the NEXT RATE.
        # If normal mode, we are predicting the NEXT VALUE.
        predict_fn, model_name = self._select_best_model(local_years, local_values)
        
        next_t = years[-1] + 1
        predicted_result = predict_fn(next_t)

        # 4. Inverse Transform if needed
        if is_growth_mode:
            # We predicted a log-growth rate. Apply it to the last known real value.
            # y_{t+1} = y_t * exp(predicted_rate)
            last_real_val = values[-1]
            next_val = last_real_val * np.exp(predicted_result)
            model_name = f"{model_name}_growth" # Tag it for debugging
        else:
            next_val = predicted_result

        return next_val, model_name

    def predict(self, X_next=None) -> float:
        next_val, _ = self._predict_one()
        return float(next_val)

    
    def forecast_horizon(self, start_year, horizon, **kwargs):
        preds = []
        # We must clone the history because we will append predictions to it
        current_years = self.last_years.copy()
        current_vals = self.last_values.copy()

        for h in range(1, horizon + 1):
            # 1. Select Context
            if self.use_full_history:
                ctx_years = current_years
                ctx_vals = current_vals
            else:
                ctx_years = current_years[-self.window_size:]
                ctx_vals = current_vals[-self.window_size:]

            # 2. Handle Growth Mode logic manually here to recurse correctly
            is_growth_mode = False
            target_years = ctx_years
            target_vals = ctx_vals

            if self.fit_on_growth_rates:
                rate_years, rates, success = self._to_growth_rates(ctx_years, ctx_vals)
                if success and len(rates) >= self.min_window:
                    target_years = rate_years
                    target_vals = rates
                    is_growth_mode = True

            # 3. Predict
            next_val_fn, _ = self._select_best_model(target_years, target_vals)
            print(start_year)
            print(_)
            print("------------------")
            next_year = start_year + h
            
            raw_prediction = next_val_fn(next_year)

            # 4. Reconstruct
            if is_growth_mode:
                last_real_val = current_vals[-1]
                pred = last_real_val * np.exp(raw_prediction)
            else:
                pred = raw_prediction

            preds.append((next_year, pred))

            # 5. Update state for recursion
            current_years = np.append(current_years, next_year)
            current_vals = np.append(current_vals, pred)

        return preds


    def get_params(self) -> Dict:
        return {
            "hyper_params": {
                "window_size": self.window_size,
                "candidate_models": self.candidate_models,
                "weighted": self.weighted,
                "weight_decay": self.weight_decay,
                "min_window": self.min_window,
                "positive_only": self.positive_only,
                "use_full_history": self.use_full_history,
                "selection_mode": self.selection_mode,
                "fit_on_growth_rates": self.fit_on_growth_rates
            },
            "fitted_params": {},
        }

class RawMeanRevertingGrowthModel(ParamIntrospectionMixin, BaseForecastModel):
    """
    Mean-reverting AR(p) model using *raw differences*.

    Model:
        Δy_t = (1 - Σρ_j) * μ
                + Σ(ρ_j * Δy_{t-j})
                + β * (Δy_{t-1})²   [optional]
                + γᵀ X_t            [optional]
                + ε

    Reconstruction:
        y_t = y_{t-1} + Δy_t
    """

    SUPPORTED_HYPERPARAMS = {
        "include_ar",
        "ar_lags",
        "include_ar_squared",
        "include_exog",
        "l2_penalty",
        "beta_bounds",
        "use_asymmetric_loss",
        "underestimation_penalty",
    }


    def __init__(
        self,
        include_ar: bool = True,
        ar_lags: int = 1,
        include_ar_squared: bool = False,
        include_exog: bool = True,
        l2_penalty: float = 1.0,
        beta_bounds: Tuple[float, float] = (-1.0, 1.0),
        use_asymmetric_loss: bool = False,
        underestimation_penalty: float = 2.0,
        **kwargs
    ):
        self._process_init(locals(), RawMeanRevertingGrowthModel)

        self.include_ar = include_ar
        self.ar_lags = ar_lags
        self.include_ar_squared = include_ar_squared
        self.include_exog = include_exog
        self.l2_penalty = l2_penalty
        self.beta_bounds = beta_bounds
        self.use_asymmetric_loss = use_asymmetric_loss
        self.underestimation_penalty = underestimation_penalty

        # Fitted parameters
        self.mu = None
        self.rho = None       # vector, length p
        self.beta = None
        self.gamma = None
        self.scaler = None

        # State for forecasting
        self.last_y = None
        self.last_growths = None   # vector of last p differences

    # -------------------------------------------------------
    # Objective function
    # -------------------------------------------------------
    def _objective(self, params, diffs, X_scaled):
        """
        params = [mu, rho_1, ..., rho_p, (beta), gamma...]
        diffs: ΔPC values (length T-1)
        """
        idx = 0
        mu = params[idx]
        idx += 1

        # AR coefficients
        if self.include_ar:
            rho = params[idx: idx + self.ar_lags]
            idx += self.ar_lags
        else:
            rho = np.zeros(self.ar_lags)

        # squared term
        if self.include_ar_squared:
            beta = params[idx]
            idx += 1
        else:
            beta = 0.0

        # exog
        if self.include_exog and X_scaled is not None:
            gamma = params[idx:]
        else:
            gamma = np.zeros(0)

        p = self.ar_lags if self.include_ar else 1
        n = len(diffs) - p

        if n <= 0:
            return 1e10

        y_pred = np.zeros(n)
        for i in range(n):
            t = p + i

            # mean-reversion constant
            c = (1 - np.sum(rho)) * mu
            val = c

            # AR terms
            for j in range(self.ar_lags):
                val += rho[j] * diffs[t - 1 - j]

            # squared term
            if self.include_ar_squared:
                val += beta * (diffs[t - 1] ** 2)

            # exogenous
            if self.include_exog and X_scaled is not None:
                val += np.dot(X_scaled[t - 1], gamma)

            y_pred[i] = val

        y_true = diffs[p:]
        errors = y_true - y_pred

        if self.use_asymmetric_loss:
            se = errors ** 2
            mask = errors > 0
            se[mask] *= self.underestimation_penalty
            loss = np.mean(se)
        else:
            loss = np.mean(errors ** 2)

        # L2 on gamma
        if self.include_exog and len(gamma) > 0:
            loss += self.l2_penalty * np.sum(gamma ** 2)

        return loss

    # -------------------------------------------------------
    # Fit
    # -------------------------------------------------------
    def fit(self, *, y=None, monthly_matrix=None, years=None, X=None, **kwargs):
        """
        Fit using only PC levels (y is the PC values).
        """
        pc = np.asarray(y, float)
        if pc.ndim != 1:
            raise ValueError("PC levels must be a 1D vector")

        diffs = np.diff(pc)

        if len(diffs) < max(2, self.ar_lags + 1):
            raise ValueError("Not enough samples to fit raw-PC model")
    
        # ---------------- Exogenous ----------------------
        X_scaled = None
        if self.include_exog and X is not None:
            X = np.asarray(X, float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_aligned = X[1:]  # align exog to diffs

            # fill NaNs
            if np.isnan(X_aligned).any():
                col_means = np.nanmean(X_aligned, axis=0)
                for i in range(X_aligned.shape[1]):
                    X_aligned[np.isnan(X_aligned[:, i]), i] = col_means[i]

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_aligned)

        # ---------------- Parameter initialization --------------------
        mu_init = np.mean(diffs)
        params_init = [mu_init]
        bounds = [(None, None)]

        if self.include_ar:
            params_init.extend([0.3] * self.ar_lags)
            bounds.extend([(None, None)] * self.ar_lags)

        if self.include_ar_squared:
            params_init.append(0.0)
            bounds.append(self.beta_bounds)

        if self.include_exog and X_scaled is not None:
            k = X_scaled.shape[1]
            params_init.extend([0.0] * k)
            bounds.extend([(None, None)] * k)

        params_init = np.array(params_init)

        # ---------------- Optimize -----------------------
        result = minimize(
            fun=self._objective,
            x0=params_init,
            args=(diffs, X_scaled),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-5, "maxls": 50}
        )

        if not result.success:
            print(f"⚠️ RawPC model optimization warning: {result.message}")

        # ---------------- Extract parameters -----------------------
        idx = 0
        self.mu = result.x[idx]
        idx += 1

        if self.include_ar:
            self.rho = result.x[idx: idx + self.ar_lags]
            idx += self.ar_lags
        else:
            self.rho = np.zeros(self.ar_lags)

        if self.include_ar_squared:
            self.beta = result.x[idx]
            idx += 1
        else:
            self.beta = 0.0

        if self.include_exog and X_scaled is not None:
            self.gamma = result.x[idx:]
        else:
            self.gamma = np.zeros(0)

        # ---------------- Store last state for forecasting -----------------------
        self.last_y = pc[-1]
        p = self.ar_lags if self.include_ar else 1
        self.last_growths = diffs[-p:] if len(diffs) >= p else diffs

        return self

    # -------------------------------------------------------
    # Predict next PC value
    # -------------------------------------------------------
    def predict(self, X_next=None) -> float:
        """
        Predicts next PC level (not difference).
        """

        # AR(p) prediction of diff
        c = (1 - np.sum(self.rho)) * self.mu
        diff_pred = c

        # AR terms
        if self.include_ar and len(self.last_growths) > 0:
            for j in range(min(self.ar_lags, len(self.last_growths))):
                diff_pred += self.rho[j] * self.last_growths[-(j+1)]

        # squared term
        if self.include_ar_squared and len(self.last_growths) > 0:
            diff_pred += self.beta * (self.last_growths[-1] ** 2)

        # exogenous
        if self.include_exog and X_next is not None and self.scaler is not None:
            X_next = np.asarray(X_next, float).reshape(1, -1)
            X_next_scaled = self.scaler.transform(X_next)
            diff_pred += np.dot(X_next_scaled[0], self.gamma)

        # convert to PC level
        pc_pred = self.last_y + diff_pred
        return float(pc_pred)

    # -------------------------------------------------------
    # Multi-step forecasting
    # -------------------------------------------------------
    def forecast_horizon(
        self,
        start_year: int,
        horizon: int,
        df_features,
        df_monthly,
        feature_config,
        monthly_clients_lookup=None,
    ):
        preds = []
        for h in range(1, horizon + 1):
            target_year = start_year + h

            # Build exogenous features for target year
            X_next = build_growth_rate_features(
                years=[target_year],
                df_features=df_features,
                df_monthly=df_monthly,
                clients_lookup=monthly_clients_lookup,
                **feature_config
            )

            x_input = None
            if X_next is not None:
                X_next = np.asarray(X_next, float)
                x_input = X_next[0]

            pc_pred = self.predict(x_input)
            preds.append((target_year, pc_pred))

            # Update state
            diff_pred = pc_pred - self.last_y

            if self.include_ar:
                if self.ar_lags > 1:
                    self.last_growths = np.append(self.last_growths[1:], diff_pred)
                else:
                    self.last_growths = np.array([diff_pred])

            self.last_y = pc_pred

        return preds

    def get_params(self) -> Dict:
        return {
            "hyper_params": {
                "include_ar": self.include_ar,
                "ar_lags": self.ar_lags,
                "include_ar_squared": self.include_ar_squared,
                "include_exog": self.include_exog,
                "l2_penalty": self.l2_penalty,
                "beta_bounds": getattr(self, 'beta_bounds', None),
                "use_asymmetric_loss": getattr(self, 'use_asymmetric_loss', None),
                "underestimation_penalty": getattr(self, 'underestimation_penalty', None),
            },
            "fitted_params": {
                "mu": getattr(self, 'mu', None),
                "rho": getattr(self, 'rho', None),
                "beta": getattr(self, 'beta', None),
                "gamma": getattr(self, 'gamma', None),
            },
        }

class MeanRevertingGrowthModel(ParamIntrospectionMixin, BaseForecastModel):
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
    
    def predict(self, X_next: Optional[np.ndarray] = None) -> float:
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


class MeanRevertingGrowthModelARP(ParamIntrospectionMixin, BaseForecastModel):
    """
    Mean-reverting growth rate model with AR(p) structure and constrained optimization.
    
    Model: Δlog(y_{t}) = (1-Σρ_j)μ + Σ(ρ_j * Δlog(y_{t-j})) + β*[Δlog(y_{t-1})]² + z_t'γ + ε
    
    Parameters:
        - μ: long-run mean growth rate
        - ρ_j: AR coefficients for j=1,...,p lags (constrained for stationarity via PACF)
        - β: squared AR term coefficient
        - γ: coefficients for exogenous features
        - use_asymmetric_loss: whether to use asymmetric loss that penalizes underestimation
        - underestimation_penalty: multiplier for underestimation errors (e.g., 2.0 or 3.0)
    """

    SUPPORTED_HYPERPARAMS = {
    "include_ar",
    "ar_lags",
    "include_ar_squared",
    "include_exog",
    "l2_penalty",
    "beta_bounds",
    "mu_bounds",
    "use_asymmetric_loss",
    "underestimation_penalty",
}

    
    def __init__(self, 
                 include_ar: bool = True,
                 ar_lags: int = 3,
                 include_ar_squared: bool = False,
                 include_exog: bool = True,
                 l2_penalty: float = 1.0,
                 beta_bounds: Tuple[float, float] = (-1.0, 1.0),
                 mu_bounds: Tuple[float, float] = (0.0, None),
                 use_asymmetric_loss: bool = False,
                 underestimation_penalty: float = 2.0, **kwargs):
        
        self._process_init(locals(), MeanRevertingGrowthModelARP)

        self.include_ar = include_ar
        self.ar_lags = ar_lags
        self.include_ar_squared = include_ar_squared
        self.include_exog = include_exog
        self.l2_penalty = l2_penalty
        self.beta_bounds = beta_bounds
        self.mu_bounds = mu_bounds
        self.use_asymmetric_loss = use_asymmetric_loss
        self.underestimation_penalty = underestimation_penalty
        
        # Fitted parameters
        self.mu = None
        self.rho = None  # Now a vector of length ar_lags
        self.beta = None
        self.gamma = None
        self.scaler = None
        
        # For prediction
        self.last_growth_rates = None  # Now stores multiple lags
        self.last_y = None
        
    def _pacf_to_ar(self, pacf: np.ndarray) -> np.ndarray:
        """
        Convert partial autocorrelations to AR coefficients using Durbin-Levinson recursion.
        This ensures stationarity as long as |pacf[j]| < 1 for all j.
        
        Args:
            pacf: Partial autocorrelations (length p)
        
        Returns:
            AR coefficients (length p)
        """
        p = len(pacf)
        if p == 0:
            return np.array([])
        
        if p == 1:
            return pacf.copy()
        
        # Durbin-Levinson recursion
        ar_coeffs = np.zeros((p, p))
        ar_coeffs[0, 0] = pacf[0]
        
        for k in range(1, p):
            # New PACF value
            ar_coeffs[k, k] = pacf[k]
            
            # Update previous coefficients
            for j in range(k):
                ar_coeffs[k, j] = ar_coeffs[k-1, j] - pacf[k] * ar_coeffs[k-1, k-1-j]
        
        return ar_coeffs[p-1, :]
    
    def _ar_to_pacf(self, ar_coeffs: np.ndarray) -> np.ndarray:
        """
        Convert AR coefficients to partial autocorrelations (inverse of _pacf_to_ar).
        Used for initialization.
        
        Args:
            ar_coeffs: AR coefficients (length p)
        
        Returns:
            Partial autocorrelations (length p)
        """
        p = len(ar_coeffs)
        if p == 0:
            return np.array([])
        
        if p == 1:
            return ar_coeffs.copy()
        
        pacf = np.zeros(p)
        ar_temp = ar_coeffs.copy()
        
        for k in range(p-1, -1, -1):
            pacf[k] = ar_temp[k]
            
            if k > 0:
                for j in range(k):
                    ar_temp[j] = (ar_temp[j] + pacf[k] * ar_temp[k-1-j]) / (1 - pacf[k]**2 + 1e-10)
        
        return pacf
    
    def _objective(self, params: np.ndarray, growth_rates: np.ndarray, 
                   X_scaled: Optional[np.ndarray]) -> float:
        """
        Objective function: MSE (or asymmetric loss) + L2 penalty.
        
        params structure depends on configuration:
        - AR(p): [μ, pacf_1, ..., pacf_p]
        - AR(p) + AR²: [μ, pacf_1, ..., pacf_p, β]
        - Exog only: [μ, γ_1, ..., γ_k]
        - AR(p) + Exog: [μ, pacf_1, ..., pacf_p, γ_1, ..., γ_k]
        - AR(p) + AR² + Exog: [μ, pacf_1, ..., pacf_p, β, γ_1, ..., γ_k]
        - Neither: [μ]
        
        Note: We optimize over PACF parameters (transformed via tanh) to ensure stationarity.
        """
        
        idx = 0
        mu = params[idx]
        idx += 1
        
        # Extract and transform PACF to AR coefficients
        if self.include_ar:
            pacf_raw = params[idx:idx+self.ar_lags]
            pacf = np.tanh(pacf_raw)  # Constrain to (-1, 1)
            rho = self._pacf_to_ar(pacf)
            idx += self.ar_lags
        else:
            rho = np.array([])
        
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
        # We need at least ar_lags observations to start predicting
        p = self.ar_lags if self.include_ar else 0
        n = len(growth_rates) - p
        
        if n <= 0:
            return 1e10  # Not enough data
        
        y_pred = np.zeros(n)
        
        for i in range(n):
            # Start index in growth_rates for this prediction
            t = p + i
            
            # Δlog(y_{t+1}) = (1-Σρ_j)μ + Σ(ρ_j * Δlog(y_{t+1-j})) + β*[Δlog(y_t)]² + z_t'γ
            rho_sum = np.sum(rho) if len(rho) > 0 else 0.0
            y_pred[i] = (1 - rho_sum) * mu
            
            # Add AR terms
            if self.include_ar:
                for j in range(self.ar_lags):
                    y_pred[i] += rho[j] * growth_rates[t - 1 - j]
            
            # Add squared term (most recent lag only)
            if self.include_ar_squared:
                y_pred[i] += beta * (growth_rates[t - 1] ** 2)
            
            # Add exogenous terms
            if self.include_exog and X_scaled is not None:
                y_pred[i] += np.dot(X_scaled[t - 1], gamma)
        
        y_true = growth_rates[p:]
        errors = y_true - y_pred
        
        # Compute loss based on mode
        if self.use_asymmetric_loss:
            # Asymmetric loss: penalize underestimation more heavily
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
        # Compute growth rates
        growth_rates = np.diff(np.log(y))
        
        # Check if we have enough data
        min_obs = self.ar_lags + 1 if self.include_ar else 2
        if len(growth_rates) < min_obs:
            raise ValueError(f"Need at least {min_obs} years of data for ar_lags={self.ar_lags}")
        
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
            # Initialize PACF parameters (in raw space, before tanh)
            # Start with small values that correspond to moderate AR coefficients
            if self.ar_lags == 1:
                # For backward compatibility, initialize to ~0.5 after tanh
                pacf_init = [0.549]  # tanh(0.549) ≈ 0.5
            else:
                # For multiple lags, use decaying initialization
                pacf_init = [0.549 / (j + 1) for j in range(self.ar_lags)]
            
            params_init.extend(pacf_init)
            bounds.extend([(None, None)] * self.ar_lags)  # PACF unbounded (tanh constrains them)
        
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
            pacf_raw = result.x[idx:idx+self.ar_lags]
            pacf = np.tanh(pacf_raw)
            self.rho = self._pacf_to_ar(pacf)
            idx += self.ar_lags
        else:
            self.rho = np.array([])
        
        if self.include_ar_squared:
            self.beta = result.x[idx]
            idx += 1
        else:
            self.beta = 0.0
        if self.include_exog and X_scaled is not None:
            self.gamma = result.x[idx:]
        else:
            self.gamma = np.array([])
        
        # Store last p growth rates for prediction
        p = self.ar_lags if self.include_ar else 1
        self.last_growth_rates = growth_rates[-p:] if len(growth_rates) >= p else growth_rates
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
        # Δlog(y_{t+1}) = (1-Σρ_j)μ + Σ(ρ_j * Δlog(y_{t+1-j})) + β*[Δlog(y_t)]² + z_t'γ
        rho_sum = np.sum(self.rho) if len(self.rho) > 0 else 0.0
        predicted_growth = (1 - rho_sum) * self.mu
        
        # Add AR terms
        if self.include_ar and len(self.last_growth_rates) > 0:
            for j in range(min(self.ar_lags, len(self.last_growth_rates))):
                # last_growth_rates[-1] is most recent, [-2] is one before, etc.
                predicted_growth += self.rho[j] * self.last_growth_rates[-(j+1)]
        
        # Add squared term (most recent growth rate only)
        if self.include_ar_squared and len(self.last_growth_rates) > 0:
            predicted_growth += self.beta * (self.last_growth_rates[-1] ** 2)
        
        # Add exogenous terms
        if self.include_exog and X_next is not None and self.scaler is not None:
            X_next = np.asarray(X_next, dtype=float).flatten().reshape(1, -1)
            
            # Handle NaN
            if np.isnan(X_next).any() or np.isinf(X_next).any():
                print("Warning: X_next contains NaN or Inf. Cleaning it...", X_next)
                X_next = np.nan_to_num(X_next, nan=0.0, posinf=0.0, neginf=0.0)
            
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
            self.last_growth_rates = np.append(self.last_growth_rates, predicted_growth_rate)[-self.ar_lags:]
            self.last_y = y_pred

        return preds

    def get_params(self) -> Dict:
        """Get model parameters split into hyper and fitted groups."""
        hyper = {
            'include_ar': getattr(self, 'include_ar', None),
            'ar_lags': getattr(self, 'ar_lags', None),
            'include_ar_squared': getattr(self, 'include_ar_squared', None),
            'include_exog': getattr(self, 'include_exog', None),
            'l2_penalty': getattr(self, 'l2_penalty', None),
            'use_asymmetric_loss': getattr(self, 'use_asymmetric_loss', None),
            'underestimation_penalty': getattr(self, 'underestimation_penalty', None),
        }

        fitted = {
            'mu': getattr(self, 'mu', None),
            'rho': getattr(self, 'rho', None),
            'beta': getattr(self, 'beta', None),
            'gamma': getattr(self, 'gamma', None),
        }

        # Compute half-life for AR(1) if possible
        try:
            if hyper.get('include_ar') and hyper.get('ar_lags') == 1:
                rho_vec = fitted.get('rho')
                rho_val = None
                if rho_vec is not None:
                    # rho might be a list/array
                    rho_val = rho_vec[0] if hasattr(rho_vec, '__len__') and len(rho_vec) > 0 else rho_vec
                fitted['half_life'] = -np.log(2) / np.log(rho_val) if rho_val is not None and rho_val > 0 else np.inf
        except Exception:
            fitted['half_life'] = None

        return {
            'hyper_params': hyper,
            'fitted_params': fitted,
        }


class PCAMeanRevertingGrowthModel(ParamIntrospectionMixin, BaseForecastModel):
    """
    PCA-based mean-reverting growth rate model.
    
    Transforms 12 monthly values into n_pcs principal components, predicts each PC
    using the AR(p) growth model, then reconstructs the 12 monthly values.
    
    Model for each PC: Δlog(PC_{i,t}) = (1-Σρ_{i,j})μ_i + Σ(ρ_{i,j} * Δlog(PC_{i,t-j})) + β_i*[Δlog(PC_{i,t-1})]² + z_t'γ_i + ε
    
    Parameters:
        - n_pcs: Number of principal components to use (default 2)
        - pca_lambda: Power weight for PCA fitting (default 1.0, higher = more recent weight)
        - Each PC has its own MeanRevertingGrowthModel
    """

    SUPPORTED_HYPERPARAMS = {
    "n_pcs",
    "pca_lambda",
    
    # submodel parameters
    "include_ar",
    "ar_lags",
    "include_ar_squared",
    "include_exog",
    "l2_penalty",
    "beta_bounds",
    "use_asymmetric_loss",
    "underestimation_penalty",
}

        
    def __init__(self,
                 n_pcs: int = 2,
                 pca_lambda: float = 0.3,
                 include_ar: bool = True,
                 ar_lags: int = 1,
                 include_ar_squared: bool = False,
                 include_exog: bool = True,
                 l2_penalty: float = 1.0,
                 beta_bounds: Tuple[float, float] = (-1.0, 1.0),
                 use_asymmetric_loss: bool = False,
                 underestimation_penalty: float = 2.0, **kwargs):
        
        self._process_init(locals(), PCAMeanRevertingGrowthModel)


        self.n_pcs = n_pcs
        self.pca_lambda = pca_lambda
        
        # Store model parameters for each PC
        self.model_params = {
            'include_ar': include_ar,
            'ar_lags': ar_lags,
            'include_ar_squared': include_ar_squared,
            'include_exog': include_exog,
            'l2_penalty': l2_penalty,
            'beta_bounds': beta_bounds,
            'use_asymmetric_loss': use_asymmetric_loss,
            'underestimation_penalty': underestimation_penalty,
        }
        
        # PCA components
        self.scaler_pca = None
        self.pca_model = None
        self.effective_n_pcs = None
        
        # One model per PC component
        self.pc_models = []
        
        # Store years for weighting
        self.years = None
        
    def _compute_power_weights(self, years: np.ndarray, lambda_value: float) -> np.ndarray:
        """
        Compute power-based weights (lambda^k) so recent years have more influence.
        The most recent year receives exponent 0, the next receives 1, etc.
        """
        years = np.asarray(years)
        n_years = years.shape[0]
        if n_years == 0:
            return np.array([], dtype=float)

        if lambda_value is None:
            return np.ones(n_years, dtype=float)

        if lambda_value < 0:
            raise ValueError("lambda_value must be non-negative for power weights.")

        order = np.argsort(np.argsort(years))
        exponents = (n_years - 1) - order
        weights = np.power(lambda_value, exponents, dtype=float)

        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(n_years, dtype=float)

        return weights
    
    def _fit_weighted_pca(self, X_scaled: np.ndarray, weights: np.ndarray, n_components: int):
        """
        Fit PCA with sample weights.
        
        Args:
            X_scaled: Standardized data (n_samples, n_features)
            weights: Sample weights (n_samples,)
            n_components: Number of components
        
        Returns:
            PCA model
        """
        # Weighted covariance matrix
        X_weighted = X_scaled * np.sqrt(weights[:, np.newaxis])
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(X_weighted, full_matrices=False)
        
        # Store components
        n_components = min(n_components, len(S))
        
        pca_dict = {
            'components_': Vt[:n_components],
            'explained_variance_': (S[:n_components] ** 2) / (len(weights) - 1),
            'mean_': np.zeros(X_scaled.shape[1]),  # Already centered by StandardScaler
            'n_components_': n_components
        }
        
        return pca_dict
    
    def _weighted_pca_transform(self, pca_dict, X_scaled: np.ndarray) -> np.ndarray:
        """Transform data using fitted PCA."""
        return np.dot(X_scaled, pca_dict['components_'].T)
    
    def _weighted_pca_inverse_transform(self, pca_dict, X_pca: np.ndarray) -> np.ndarray:
        """Inverse transform from PC space to original space."""
        return np.dot(X_pca, pca_dict['components_'])
    
    def fit(self, *, y = None, monthly_matrix=None, years=None, X=None, **kwargs):
        """
        Fit the PCA-based growth model.
        
        Args:
            monthly_matrix: Monthly consumption data (shape: n_years, 12)
            years: Array of years corresponding to each row
            X: Exogenous features (shape: n_years, k) - optional
        """
        monthly_matrix = np.asarray(monthly_matrix, dtype=float)
        years = np.asarray(years, dtype=int)
        self.years = years
        
        if monthly_matrix.shape[0] != len(years):
            raise ValueError("Number of rows in monthly_matrix must match length of years")
        
        if monthly_matrix.shape[1] != 12:
            raise ValueError("monthly_matrix must have 12 columns (one per month)")
        
        # Step 1: Fit PCA on monthly data
        self.scaler_pca = StandardScaler()
        monthly_scaled = self.scaler_pca.fit_transform(monthly_matrix)
        
        # Compute sample weights based on years
        sample_weights = self._compute_power_weights(years, self.pca_lambda)
        
        # Fit weighted PCA
        self.pca_model = self._fit_weighted_pca(monthly_scaled, sample_weights, self.n_pcs)
        
        # Transform to PC space
        pc_scores = self._weighted_pca_transform(self.pca_model, monthly_scaled)
        self.effective_n_pcs = pc_scores.shape[1]
        
        # Step 2: Fit a growth model for each PC component
        self.pc_models = []
        
        for pc_idx in range(self.effective_n_pcs):
            # Get PC scores as "consumption levels"
            pc_levels = pc_scores[:, pc_idx]
            
            # Handle exogenous features for this PC
            # For PC0 (first component), use all features
            # For other PCs, you might want different features or same features
            X_pc = X if X is not None else None
            
            # Create and fit model for this PC
            model = RawMeanRevertingGrowthModel(**self.model_params)
            
            try:
                model.fit(y = pc_levels, X = X_pc)
                self.pc_models.append(model)
            except Exception as e:
                # print(f"⚠️  Warning: Could not fit model for PC{pc_idx}: {e}")
                # Create a dummy model that predicts the mean
                dummy_model = RawMeanRevertingGrowthModel(
                    include_ar=False,
                    include_exog=False
                )
                dummy_model.fit(y = pc_levels)
                self.pc_models.append(dummy_model)
        
        return self
    
    def _predict_monthly(self, X_next: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict next year's monthly consumption.
        
        Args:
            X_next: Exogenous features for next period (shape: k,) - optional
        
        Returns:
            Predicted monthly consumption (shape: 12,)
        """
        if self.pca_model is None or len(self.pc_models) == 0:
            raise ValueError("Model must be fitted before prediction")
        
        # Step 1: Predict each PC component
        predicted_pcs = np.zeros(self.effective_n_pcs)
        
        for pc_idx in range(self.effective_n_pcs):
            model = self.pc_models[pc_idx]
            
            # Predict PC level (not growth rate)
            X_pc = X_next if X_next is not None else None
            predicted_pcs[pc_idx] = model.predict(X_pc)
        
        # Step 2: Reconstruct monthly values from predicted PCs
        reconstructed_scaled = self._weighted_pca_inverse_transform(
            self.pca_model, 
            predicted_pcs.reshape(1, -1)
        )
        
        # Inverse transform to original scale
        reconstructed = self.scaler_pca.inverse_transform(reconstructed_scaled)[0]
        
        return reconstructed
    
    def predict(self, X_next: Optional[np.ndarray] = None) -> float:
        monthly_prediction = self._predict_monthly(X_next)
        return float(np.sum(monthly_prediction))
    
    def _forecast_horizon_monthly(
        self,
        start_year: int,
        horizon: int,
        df_features,
        df_monthly,
        feature_config: dict,
        monthly_clients_lookup=None,
    ):
        """
        Multi-step ahead forecasting.
        
        Args:
            start_year: Year to start forecasting from
            horizon: Number of years to forecast
            df_features: DataFrame with exogenous features
            df_monthly: DataFrame with monthly data
            feature_config: Configuration for feature building
            monthly_clients_lookup: Optional client data
        
        Returns:
            List of tuples (year, predicted_monthly_array)
        """
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

            # Predict monthly consumption (returns array of 12 values)
            monthly_pred = self._predict_monthly(x_input)
            
            preds.append((target_year, monthly_pred))

            # Update internal state for each PC model
            # Transform predicted monthly to PC space
            monthly_scaled = self.scaler_pca.transform(monthly_pred.reshape(1, -1))
            pc_scores_pred = self._weighted_pca_transform(self.pca_model, monthly_scaled)[0]
            
            # Update each PC model's state
            for pc_idx in range(self.effective_n_pcs):
                model = self.pc_models[pc_idx]
                
                # Get current and previous PC level
                current_pc_level = pc_scores_pred[pc_idx]
                previous_pc_level = model.last_y
                
                # Compute growth rate for this PC
                predicted_growth_rate = current_pc_level - previous_pc_level
                
                # Update the model's state
                # For AR(p), we need to update the vector of last growth rates
                if model.include_ar and model.ar_lags > 1:
                    # Shift the growth rates and add new one
                    model.last_growths = np.append(
                        model.last_growths[1:], 
                        predicted_growth_rate
                    )
                elif model.include_ar:
                    # For AR(1), just update the single value
                    model.last_growths = np.array([predicted_growth_rate])
                
                # Update last level
                model.last_y = current_pc_level

        return preds
    
    def forecast_horizon(
        self,
        start_year: int,
        horizon: int,
        df_features,
        df_monthly,
        feature_config: dict,
        monthly_clients_lookup=None,
    ):
        monthly_preds = self._forecast_horizon_monthly(
            start_year=start_year,
            horizon=horizon,
            df_features=df_features,
            df_monthly=df_monthly,
            feature_config=feature_config,
            monthly_clients_lookup=monthly_clients_lookup,
        )

        preds = []
        for target_year, monthly_pred in monthly_preds:
            annual_pred = np.sum(monthly_pred)
            preds.append((target_year, annual_pred))
        
        return preds

    def predict_with_actual_pcs(self, actual_monthly: np.ndarray) -> np.ndarray:
        """
        Helper method: Transform actual monthly data to PCs and reconstruct.
        Useful for debugging and understanding PCA reconstruction error.
        
        Args:
            actual_monthly: Actual monthly consumption (shape: 12,)
        
        Returns:
            Reconstructed monthly consumption (shape: 12,)
        """
        if self.pca_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Transform to PC space
        monthly_scaled = self.scaler_pca.transform(actual_monthly.reshape(1, -1))
        pc_scores = self._weighted_pca_transform(self.pca_model, monthly_scaled)
        
        # Reconstruct
        reconstructed_scaled = self._weighted_pca_inverse_transform(
            self.pca_model,
            pc_scores
        )
        reconstructed = self.scaler_pca.inverse_transform(reconstructed_scaled)[0]
        
        return reconstructed
    
    def get_pc_scores(self, monthly_matrix: np.ndarray) -> np.ndarray:
        """
        Transform monthly data to PC scores.
        
        Args:
            monthly_matrix: Monthly consumption data (shape: n_years, 12)
        
        Returns:
            PC scores (shape: n_years, n_pcs)
        """
        if self.pca_model is None:
            raise ValueError("Model must be fitted first")
        
        monthly_scaled = self.scaler_pca.transform(monthly_matrix)
        return self._weighted_pca_transform(self.pca_model, monthly_scaled)
    
    def get_params(self) -> Dict:
        """Return hyper and fitted parameters for the PCA-based model."""
        hyper = {
            'n_pcs': getattr(self, 'n_pcs', None),
            'pca_lambda': getattr(self, 'pca_lambda', None),
            'model_params': getattr(self, 'model_params', None),
        }

        fitted = {
            'effective_n_pcs': getattr(self, 'effective_n_pcs', None),
            'pc_models': [],
        }

        # Add parameters for each PC model (as nested param dicts)
        try:
            pc_models = getattr(self, 'pc_models', []) or []
            for model in pc_models:
                try:
                    fitted['pc_models'].append(model.get_params())
                except Exception:
                    fitted['pc_models'].append(None)
        except Exception:
            fitted['pc_models'] = None

        # Add PCA explained variance if available
        pca = getattr(self, 'pca_model', None)
        if pca is not None:
            try:
                ev = pca.get('explained_variance_', None) if isinstance(pca, dict) else getattr(pca, 'explained_variance_', None)
                fitted['explained_variance'] = ev
                if ev is not None:
                    total = np.sum(ev) if hasattr(ev, '__len__') else ev
                    fitted['explained_variance_ratio'] = (ev / total) if total else None
            except Exception:
                fitted['explained_variance'] = None
                fitted['explained_variance_ratio'] = None

        return {
            'hyper_params': hyper,
            'fitted_params': fitted,
        }
    
    def get_reconstruction_error(self, monthly_matrix: np.ndarray) -> Dict:
        """
        Calculate PCA reconstruction error.
        
        Args:
            monthly_matrix: Monthly consumption data (shape: n_years, 12)
        
        Returns:
            Dictionary with reconstruction metrics
        """
        if self.pca_model is None:
            raise ValueError("Model must be fitted first")
        
        # Transform and reconstruct
        monthly_scaled = self.scaler_pca.transform(monthly_matrix)
        pc_scores = self._weighted_pca_transform(self.pca_model, monthly_scaled)
        reconstructed_scaled = self._weighted_pca_inverse_transform(self.pca_model, pc_scores)
        reconstructed = self.scaler_pca.inverse_transform(reconstructed_scaled)
        
        # Calculate errors
        errors = monthly_matrix - reconstructed
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / (monthly_matrix + 1e-10))) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'max_error': np.max(np.abs(errors)),
        }


class AsymmetricAdaptiveTrendModel(ParamIntrospectionMixin, BaseForecastModel): # Inherit from your Mixins here
    """
    Asymmetric Adaptive Trend Model (AATM).
    
    Dynamics:
        1. Gap calculation: Gap_t = log(y_t) - Trend_t
        2. Asymmetric Reversion: α depends on whether Gap is + or -
        3. Adaptive Trend: Trend leans into positive gaps (bubbles/growth) 
           but resists negative gaps (crashes).

    Parameters:
        - mu_base: The fundamental long-run growth rate.
        - alpha_down: Reversion speed when price < trend (Crash recovery).
        - alpha_up: Reversion speed when price > trend (Bubble correction).
        - beta_adapt: How much the trend 'leans in' to price increases.
    """
    
    SUPPORTED_HYPERPARAMS = {
        "include_exog",
        "l2_penalty",
        "alpha_down_bounds",
        "alpha_up_bounds",
        "beta_adapt_bounds",
        "mu_bounds",
        "use_asymmetric_loss",
        "underestimation_penalty"
    }

    def __init__(self, 
                 include_exog: bool = True,
                 l2_penalty: float = 10.0,
                 alpha_down_bounds: Tuple[float, float] = (0.05, 0.5), # Default strict
                 alpha_up_bounds: Tuple[float, float] = (0.0, 0.2),   # Default lenient
                 beta_adapt_bounds: Tuple[float, float] = (0.0, 0.5),
                 mu_bounds: Tuple[float, float] = (0.0, None),
                 use_asymmetric_loss: bool = False,
                 underestimation_penalty: float = 2.0, 
                 **kwargs):
        
        self._process_init(locals(), AsymmetricAdaptiveTrendModel) # Uncomment if using your Mixin

        self.include_exog = include_exog
        self.l2_penalty = l2_penalty
        self.alpha_down_bounds = alpha_down_bounds
        self.alpha_up_bounds = alpha_up_bounds
        self.beta_adapt_bounds = beta_adapt_bounds
        self.mu_bounds = mu_bounds
        self.use_asymmetric_loss = use_asymmetric_loss
        self.underestimation_penalty = underestimation_penalty
        
        # Fitted parameters
        self.mu_base = None
        self.alpha_down = None
        self.alpha_up = None
        self.beta_adapt = None
        self.gamma = None
        self.scaler = None
        
        # State for prediction
        self.last_y = None
        self.last_trend_level = None
        self.last_dynamic_mu = None

    def _reconstruct_path(self, params: np.ndarray, log_y: np.ndarray, 
                          X_scaled: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal helper to simulate the trend path over history given specific parameters.
        Returns: (y_pred_log_diff, final_state_tuple)
        """
        # 1. Unpack Parameters
        idx = 0
        mu_base = params[idx]; idx += 1
        alpha_down = params[idx]; idx += 1
        alpha_up = params[idx]; idx += 1
        beta_adapt = params[idx]; idx += 1
        
        gamma = np.array([])
        if self.include_exog and X_scaled is not None:
            gamma = params[idx:]

        T = len(log_y)
        
        # Initialize State variables
        # We assume the trend started exactly where the price started
        trend_level = log_y[0] 
        dynamic_mu = mu_base
        
        predictions = []
        
        # 2. Iterate through history to build the dynamic trend
        # We predict t+1 based on information at t
        for t in range(T - 1):
            # Current state at t
            current_log_y = log_y[t]
            
            # A. Calculate Gap
            gap = current_log_y - trend_level
            
            # B. Determine Asymmetric Alpha
            if gap < 0:
                current_alpha = alpha_down
            else:
                current_alpha = alpha_up
                
            # C. Update Trend (Adaptive)
            # "Lean in" logic: if gap is positive, trend grows faster
            upside_pressure = max(0, gap)
            dynamic_mu = mu_base + (beta_adapt * upside_pressure)
            
            # The trend level for NEXT step (t+1)
            trend_level = trend_level + dynamic_mu
            
            # D. Predict Growth for t+1
            # Δlog(y) = Expected_Trend_Growth - Correction + Exog
            pred_change = dynamic_mu - (current_alpha * gap)
            
            if self.include_exog and X_scaled is not None:
                pred_change += np.dot(X_scaled[t], gamma)
                
            predictions.append(pred_change)

        return np.array(predictions), (trend_level, dynamic_mu)

    def _objective(self, params: np.ndarray, log_y: np.ndarray, 
                   X_scaled: Optional[np.ndarray]) -> float:
        
        # Reconstruct the path to get predictions
        y_pred_changes, _ = self._reconstruct_path(params, log_y, X_scaled)
        
        # True changes (actual growth rates)
        # log_y has length T, so diff is T-1
        y_true_changes = np.diff(log_y) 
        
        errors = y_true_changes - y_pred_changes
        
        # Compute Loss
        if self.use_asymmetric_loss:
            squared_errors = errors ** 2
            underestimation_mask = errors > 0 
            weighted_errors = squared_errors.copy()
            weighted_errors[underestimation_mask] *= self.underestimation_penalty
            loss = np.mean(weighted_errors)
        else:
            loss = np.mean(errors ** 2)
            
        # L2 Penalty on Gamma only
        idx_gamma_start = 4 # mu, a_down, a_up, beta
        if self.include_exog and len(params) > idx_gamma_start:
            gamma_params = params[idx_gamma_start:]
            l2_term = self.l2_penalty * np.sum(gamma_params ** 2)
        else:
            l2_term = 0.0
            
        return loss + l2_term

    def fit(self, *, y=None, X=None, **kwargs):
        y = np.asarray(y, dtype=float).flatten()
        log_y = np.log(y)
        
        # --- Exogenous Feature Handling (Same as your code) ---
        X_scaled = None
        if self.include_exog and X is not None:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1: X = X.reshape(-1, 1)
            X_aligned = X[1:] # Align with predictions
            
            if np.isnan(X_aligned).any():
                col_means = np.nanmean(X_aligned, axis=0)
                for i in range(X_aligned.shape[1]):
                    X_aligned[np.isnan(X_aligned[:, i]), i] = col_means[i]
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_aligned)

        # --- Initialization ---
        # Guess mu based on average growth
        mu_init = np.mean(np.diff(log_y))
        
        # [mu_base, alpha_down, alpha_up, beta_adapt]
        params_init = [mu_init, 0.5, 0.1, 0.1] 
        
        bounds = [
            self.mu_bounds,
            self.alpha_down_bounds,
            self.alpha_up_bounds,
            self.beta_adapt_bounds
        ]
        
        if self.include_exog and X_scaled is not None:
            k = X_scaled.shape[1]
            params_init.extend([0.0] * k)
            bounds.extend([(None, None)] * k)

        # --- Optimization ---
        result = minimize(
            fun=self._objective,
            x0=np.array(params_init),
            args=(log_y, X_scaled),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        # --- Store Parameters ---
        self.mu_base = result.x[0]
        self.alpha_down = result.x[1]
        self.alpha_up = result.x[2]
        self.beta_adapt = result.x[3]
        
        if self.include_exog and X_scaled is not None:
            self.gamma = result.x[4:]
        else:
            self.gamma = np.array([])

        # --- CRITICAL: SET FINAL STATE ---
        # We must run the path one last time to find out where the 
        # Trend Level ended up at time T, so we can predict T+1.
        _, final_state = self._reconstruct_path(result.x, log_y, X_scaled)
        
        self.last_trend_level = final_state[0] # Trend at T-1 (aligned for next step)
        self.last_dynamic_mu = final_state[1]  # Growth used for next step
        self.last_y = y[-1]
        
        return self

    def predict(self, X_next: Optional[np.ndarray] = None) -> float:
        """Predicts the next level y_{t+1} based on stored state."""
        
        # 1. Calculate Gap at the end of training (or previous step)
        last_log_y = np.log(self.last_y)
        gap = last_log_y - self.last_trend_level
        
        # 2. Determine Alpha
        if gap < 0:
            current_alpha = self.alpha_down
        else:
            current_alpha = self.alpha_up
            
        # 3. Calculate Prediction (Growth Rate)
        # Δlog(y) = μ_dynamic - α * Gap + Exog
        pred_growth = self.last_dynamic_mu - (current_alpha * gap)
        
        if self.include_exog and X_next is not None and self.scaler is not None:
            X_next = np.asarray(X_next, dtype=float).flatten().reshape(1, -1)
            if np.isnan(X_next).any(): X_next = np.nan_to_num(X_next, nan=0.0)
            X_next_scaled = self.scaler.transform(X_next)
            pred_growth += np.dot(X_next_scaled[0], self.gamma)
            
        # 4. Convert to Level
        y_pred = self.last_y * np.exp(pred_growth)
        
        return float(y_pred)

    def forecast_horizon(self, start_year: int, horizon: int, 
                        df_features, df_monthly, feature_config: dict, 
                        monthly_clients_lookup=None):
        """
        Iterative forecasting that updates the internal Trend state.
        """
        preds = []
        
        # We need to preserve the 'real' state to restore it after forecasting
        # so we don't corrupt the model if we call predict() again
        saved_y = self.last_y
        saved_trend = self.last_trend_level
        saved_mu = self.last_dynamic_mu

        for h in range(1, horizon + 1):
            target_year = start_year + h

            # (Placeholder function call from your snippet)
            # x_next = build_growth_rate_features(...) 
            # For this snippet, I assume x_next is None or handled externally
            x_input = None 

            # 1. Predict Price
            y_pred = self.predict(x_input)
            preds.append((target_year, y_pred))

            # 2. Update State for Next Iteration (t+2)
            # We must simulate the Trend update logic here!
            
            # A. Current Gap (based on the prediction we just made)
            log_y_pred = np.log(y_pred)
            # Note: predict() used last_trend_level. Now we update trend for next step.
            gap = np.log(self.last_y) - self.last_trend_level
            
            # B. Update Trend Expectations (Adaptive Logic)
            upside_pressure = max(0, gap)
            new_dynamic_mu = self.mu_base + (self.beta_adapt * upside_pressure)
            
            new_trend_level = self.last_trend_level + new_dynamic_mu
            
            # C. Commit State
            self.last_y = y_pred
            self.last_trend_level = new_trend_level
            self.last_dynamic_mu = new_dynamic_mu

        # Restore original state
        self.last_y = saved_y
        self.last_trend_level = saved_trend
        self.last_dynamic_mu = saved_mu

        return preds

    "include_exog",
    "l2_penalty",
    "alpha_down_bounds",
    "alpha_up_bounds",
    "beta_adapt_bounds",
    "mu_bounds",
    "use_asymmetric_loss",
    "underestimation_penalty"

    def get_params(self) -> Dict:
        hyper = {
            'include_exog': getattr(self, 'include_exog', None),
            'l2_penalty': getattr(self, 'l2_penalty', None),
            'alpha_down_bounds': getattr(self, 'alpha_down_bounds', None),
            'alpha_up_bounds': getattr(self, 'alpha_up_bounds', None),
            'beta_adapt_bounds': getattr(self, 'beta_adapt_bounds', None),
            'mu_bounds': getattr(self, 'mu_bounds', None),
            'use_asymmetric_loss': getattr(self, 'use_asymmetric_loss', None),
            'underestimation_penalty': getattr(self, 'underestimation_penalty', None),
        }

        fitted = {
            'mu_base': getattr(self, 'mu_base', None),
            'alpha_down': getattr(self, 'alpha_down', None),
            'alpha_up': getattr(self, 'alpha_up', None),
            'beta_adapt': getattr(self, 'beta_adapt', None),
            'gamma': getattr(self, 'gamma', None),
        }

        return {
            'hyper_params': hyper,
            'fitted_params': fitted,
        }



class FullyAdaptiveTrendModel(ParamIntrospectionMixin, BaseForecastModel):
    """
    Fully Adaptive Trend Model.
    
    Treats Increases and Decreases with separate "Trend Adaptation" parameters.
    Allows the trend line to bend UP for booms and DOWN for sustained crashes.
    """
    SUPPORTED_HYPERPARAMS = {
        "include_exog",
        "l2_penalty",
        "alpha_down_bounds",
        "alpha_up_bounds",
        "beta_up_bounds",
        "beta_down_bounds",
        "mu_bounds",
        "use_asymmetric_loss",
        "underestimation_penalty"
    }

    def __init__(self, 
                 include_exog: bool = True,
                 l2_penalty: float = 10.0,
                 # REVERSION SPEEDS (Alpha)
                 alpha_down_bounds: Tuple[float, float] = (0.05, 0.4), # Default: Slow/U-shape recovery
                 alpha_up_bounds: Tuple[float, float] = (0.0, 0.3),
                 # TREND ADAPTATION (Beta)
                 beta_up_bounds: Tuple[float, float] = (0.0, 0.5),    # How much to trust bubbles
                 beta_down_bounds: Tuple[float, float] = (0.0, 0.5),  # How much to trust crashes
                 mu_bounds: Tuple[float, float] = (0.0, None),
                 use_asymmetric_loss: bool = False,
                 underestimation_penalty: float = 2.0, 
                 **kwargs):
        
        self._process_init(locals(), AsymmetricAdaptiveTrendModel) # Uncomment if using your Mixin

        self.include_exog = include_exog
        self.l2_penalty = l2_penalty
        self.alpha_down_bounds = alpha_down_bounds
        self.alpha_up_bounds = alpha_up_bounds
        self.beta_up_bounds = beta_up_bounds
        self.beta_down_bounds = beta_down_bounds
        self.mu_bounds = mu_bounds
        self.use_asymmetric_loss = use_asymmetric_loss
        self.underestimation_penalty = underestimation_penalty
        
        # Fitted parameters
        self.mu_base = None
        self.alpha_down = None
        self.alpha_up = None
        self.beta_up = None
        self.beta_down = None
        self.gamma = None
        self.scaler = None
        
        # State
        self.last_y = None
        self.last_trend_level = None
        self.last_dynamic_mu = None

    def _reconstruct_path(self, params: np.ndarray, log_y: np.ndarray, 
                          X_scaled: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        
        # 1. Unpack Parameters (Order: mu, a_down, a_up, b_up, b_down, gamma...)
        idx = 0
        mu_base = params[idx]; idx += 1
        alpha_down = params[idx]; idx += 1
        alpha_up = params[idx]; idx += 1
        beta_up = params[idx]; idx += 1
        beta_down = params[idx]; idx += 1
        
        gamma = np.array([])
        if self.include_exog and X_scaled is not None:
            gamma = params[idx:]

        T = len(log_y)
        trend_level = log_y[0] 
        dynamic_mu = mu_base
        
        predictions = []
        
        for t in range(T - 1):
            current_log_y = log_y[t]
            gap = current_log_y - trend_level
            
            # --- STEP A: Determine Reversion Speed (Alpha) ---
            if gap < 0:
                current_alpha = alpha_down
            else:
                current_alpha = alpha_up
                
            # --- STEP B: Determine Trend Adaptation (Beta) ---
            # NEW: We allow the trend to react to both positive and negative gaps
            upside_pressure = max(0, gap)
            downside_pressure = min(0, gap) # This will be negative or 0
            
            # Update Mu: 
            # If gap is negative (crash), downside_pressure is negative.
            # beta_down * negative_value = reduces growth rate.
            dynamic_mu = mu_base + (beta_up * upside_pressure) + (beta_down * downside_pressure)
            
            # Update Trend Level
            trend_level = trend_level + dynamic_mu
            
            # --- STEP C: Predict ---
            pred_change = dynamic_mu - (current_alpha * gap)
            
            if self.include_exog and X_scaled is not None:
                pred_change += np.dot(X_scaled[t], gamma)
                
            predictions.append(pred_change)

        return np.array(predictions), (trend_level, dynamic_mu)

    def _objective(self, params: np.ndarray, log_y: np.ndarray, 
                   X_scaled: Optional[np.ndarray]) -> float:
        
        y_pred_changes, _ = self._reconstruct_path(params, log_y, X_scaled)
        y_true_changes = np.diff(log_y) 
        errors = y_true_changes - y_pred_changes
        
        if self.use_asymmetric_loss:
            squared_errors = errors ** 2
            underestimation_mask = errors > 0 
            weighted_errors = squared_errors.copy()
            weighted_errors[underestimation_mask] *= self.underestimation_penalty
            loss = np.mean(weighted_errors)
        else:
            loss = np.mean(errors ** 2)
            
        # L2 Penalty (Skip first 5 params: mu, a_d, a_u, b_u, b_d)
        idx_gamma_start = 5 
        if self.include_exog and len(params) > idx_gamma_start:
            gamma_params = params[idx_gamma_start:]
            l2_term = self.l2_penalty * np.sum(gamma_params ** 2)
        else:
            l2_term = 0.0
            
        return loss + l2_term

    def fit(self, *, y=None, X=None, **kwargs):
        y = np.asarray(y, dtype=float).flatten()
        log_y = np.log(y)
        
        # Exogenous handling (standard boilerplate)
        X_scaled = None
        if self.include_exog and X is not None:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1: X = X.reshape(-1, 1)
            X_aligned = X[1:] 
            if np.isnan(X_aligned).any():
                col_means = np.nanmean(X_aligned, axis=0)
                for i in range(X_aligned.shape[1]):
                    X_aligned[np.isnan(X_aligned[:, i]), i] = col_means[i]
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_aligned)

        # Init Params: [mu, alpha_down, alpha_up, beta_up, beta_down]
        mu_init = np.mean(np.diff(log_y))
        params_init = [mu_init, 0.2, 0.2, 0.1, 0.1] 
        
        bounds = [
            self.mu_bounds,
            self.alpha_down_bounds,
            self.alpha_up_bounds,
            self.beta_up_bounds,
            self.beta_down_bounds # New Bound
        ]
        
        if self.include_exog and X_scaled is not None:
            k = X_scaled.shape[1]
            params_init.extend([0.0] * k)
            bounds.extend([(None, None)] * k)

        result = minimize(
            fun=self._objective,
            x0=np.array(params_init),
            args=(log_y, X_scaled),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        self.mu_base = result.x[0]
        self.alpha_down = result.x[1]
        self.alpha_up = result.x[2]
        self.beta_up = result.x[3]
        self.beta_down = result.x[4]
        
        if self.include_exog and X_scaled is not None:
            self.gamma = result.x[5:]
        else:
            self.gamma = np.array([])

        _, final_state = self._reconstruct_path(result.x, log_y, X_scaled)
        self.last_trend_level = final_state[0]
        self.last_dynamic_mu = final_state[1]
        self.last_y = y[-1]
        
        return self

    def predict(self, X_next: Optional[np.ndarray] = None) -> float:
        last_log_y = np.log(self.last_y)
        gap = last_log_y - self.last_trend_level
        
        # Alpha Logic
        current_alpha = self.alpha_down if gap < 0 else self.alpha_up
            
        # Predict Growth
        pred_growth = self.last_dynamic_mu - (current_alpha * gap)
        
        if self.include_exog and X_next is not None and self.scaler is not None:
            X_next = np.asarray(X_next, dtype=float).flatten().reshape(1, -1)
            if np.isnan(X_next).any(): X_next = np.nan_to_num(X_next, nan=0.0)
            X_next_scaled = self.scaler.transform(X_next)
            pred_growth += np.dot(X_next_scaled[0], self.gamma)
            
        return float(self.last_y * np.exp(pred_growth))
    
    def forecast_horizon(
        self,
        start_year: int,
        horizon: int,
        df_features,
        df_monthly,
        feature_config: dict,
        monthly_clients_lookup=None,
    ):
        """
        Iterative forecasting that updates the internal Trend state based on predictions.
        
        It simulates the "Feedback Loop":
        1. Predict Price for Year X
        2. Calculate the Gap (Price - Trend)
        3. Adjust the Trend for Year X+1 (using beta_up/beta_down) based on that Gap.
        """
        preds = []
        
        # --- 1. STATE PRESERVATION ---
        # We save the model's actual state (from training) so we can restore it 
        # after the simulation. We don't want the forecast loop to permanently 
        # alter the fitted model.
        saved_y = self.last_y
        saved_trend = self.last_trend_level
        saved_mu = self.last_dynamic_mu

        # --- 2. FORECAST LOOP ---
        for h in range(1, horizon + 1):
            target_year = start_year + h

            # A. Prepare Features (Assumes your existing helper function)
            # You might need to import build_growth_rate_features or define it
            try:
                x_next = build_growth_rate_features(
                    years=[target_year],
                    df_features=df_features,
                    df_monthly=df_monthly,
                    clients_lookup=monthly_clients_lookup,
                    **feature_config
                )
                if x_next is not None:
                    x_next = np.asarray(x_next, dtype=float)
                    x_input = x_next[0]
                else:
                    x_input = None
            except NameError:
                # Fallback if function not defined in scope
                x_input = None

            # B. Predict Price (Step T+1)
            # This uses the current self.last_trend_level and self.last_dynamic_mu
            y_pred = self.predict(x_input)
            preds.append((target_year, y_pred))

            # C. UPDATE STATE FOR NEXT STEP (Step T+2)
            # We must simulate how the Trend would react to this prediction.
            
            # 1. Calculate the Gap created by our prediction
            # Note: self.last_trend_level is currently the trend for the prediction year
            log_y_pred = np.log(y_pred)
            gap = log_y_pred - self.last_trend_level
            
            # 2. Update Growth Expectation (Adaptive Logic)
            # This is where beta_up vs beta_down matters!
            upside_pressure = max(0, gap)
            downside_pressure = min(0, gap) # Negative or Zero
            
            new_dynamic_mu = self.mu_base + \
                             (self.beta_up * upside_pressure) + \
                             (self.beta_down * downside_pressure)
            
            # 3. Update Trend Level
            # The trend for the *next* year grows by this new mu
            new_trend_level = self.last_trend_level + new_dynamic_mu
            
            # 4. Commit Temporary State
            self.last_y = y_pred
            self.last_trend_level = new_trend_level
            self.last_dynamic_mu = new_dynamic_mu

        # --- 3. STATE RESTORATION ---
        self.last_y = saved_y
        self.last_trend_level = saved_trend
        self.last_dynamic_mu = saved_mu

        return preds
    
    def get_params(self) -> Dict:
        # REVERSION SPEEDS (Alpha)
                 
        hyper = {
            'include_exog': getattr(self, 'include_exog', None),
            'l2_penalty': getattr(self, 'l2_penalty', None),
            'alpha_down_bounds': getattr(self, 'alpha_down_bounds', None),
            'alpha_up_bounds': getattr(self, 'alpha_up_bounds', None),
            'beta_up_bounds': getattr(self, 'beta_up_bounds', None),
            'beta_down_bounds': getattr(self, 'beta_down_bounds', None),
            'mu_bounds': getattr(self, 'mu_bounds', None),
            'use_asymmetric_loss': getattr(self, 'use_asymmetric_loss', None),
            'underestimation_penalty': getattr(self, 'underestimation_penalty', None),
        }

        fitted = {
            'mu_base': getattr(self, 'mu_base', None),
            'alpha_down': getattr(self, 'alpha_down', None),
            'alpha_up': getattr(self, 'alpha_up', None),
            'beta_up': getattr(self, 'beta_up', None),
            'beta_down': getattr(self, 'beta_down', None),
            'gamma': getattr(self, 'gamma', None),
        }

        return {
            'hyper_params': hyper,
            'fitted_params': fitted,
        }
