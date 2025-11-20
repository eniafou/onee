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
from typing import Dict, Tuple, Optional, Iterable, List, Mapping
import inspect
from onee.utils import add_annual_client_feature, add_yearly_feature
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression





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
        pass

    @abstractmethod
    def predict(self, X_next: Optional[np.ndarray] = None) -> float:
        pass

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
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        pass



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




class LocalInterpolationForecastModel(ParamIntrospectionMixin, BaseForecastModel):
    """
    Local curve-fitting interpolation-based forecast model.

    For each prediction step:
        - Take the last `window_size` years.
        - Fit several candidate models (linear, quadratic, exponential, ...).
        - Select the one with lowest (weighted) RMSE.
        - Extrapolate 1 step forward.
        - Slide window and repeat for multi-step forecasting.

    This model is purely trend-based (ignores exogenous features for now).
    """
    SUPPORTED_HYPERPARAMS = {
        "window_size",
        "candidate_models",
        "weighted",
        "weight_decay",
        "min_window",
        "positive_only",
    }


    def __init__(
        self,
        window_size: int = 5,
        candidate_models: List[str] = ("linear", "quadratic", "exponential"),
        weighted: bool = True,
        weight_decay: float = 0.9,
        min_window: int = 2,
        positive_only: bool = False,
        **kwargs
    ):
        """
        Args:
            window_size: Number of past years used for curve fitting.
            candidate_models: Which curve types to consider.
            weighted: Whether to use exponentially decaying weights.
            weight_decay: Lambda for weights (recent points get more weight).
            min_window: Minimum required points to fit anything.
            positive_only: If True, exponential model is used only when y > 0.
        """
        self._process_init(locals(), LocalInterpolationForecastModel)

        self.window_size = window_size
        self.candidate_models = list(candidate_models)
        self.weighted = weighted
        self.weight_decay = weight_decay
        self.min_window = min_window
        self.positive_only = positive_only

        # store last values for recursive forecasting
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

    # ---------------------------------------------------------
    # Model selection
    # ---------------------------------------------------------
    def _evaluate_model(self, predict_fn, years, values, weights):
        preds = np.array([predict_fn(t) for t in years])
        errors = preds - values
        return np.sqrt(np.average(errors**2, weights=weights))

    def _select_best_model(self, years, values):
        n = len(values)
        if n < self.min_window:
            # fallback: constant model
            const = values[-1]
            return lambda t: const, "constant"

        years = np.asarray(years, float)
        values = np.asarray(values, float)
        weights = self._compute_weights(n)

        candidates = {}

        if "linear" in self.candidate_models:
            cand = self._fit_linear(years, values, weights)
            candidates["linear"] = cand

        if "quadratic" in self.candidate_models and n >= 3:
            cand = self._fit_quadratic(years, values, weights)
            candidates["quadratic"] = cand

        if "exponential" in self.candidate_models:
            if (not self.positive_only) or np.all(values > 0):
                cand = self._fit_exponential(years, values, weights)
                if cand is not None:
                    candidates["exponential"] = cand

        # evaluate RMSE
        best_name = None
        best_fn = None
        best_rmse = np.inf

        for name, fn in candidates.items():
            rmse = self._evaluate_model(fn, years, values, weights)
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_fn = fn

        if best_fn is None:
            # fallback to last value
            return lambda t: values[-1], "constant"

        return best_fn, best_name

    # ---------------------------------------------------------
    # Fit model
    # ---------------------------------------------------------
    def fit(
        self,
        *,
        y=None,
        monthly_matrix=None,
        years=None,
        X=None,
        **kwargs
    ):
        """
        Fit does not actually estimate global parameters.
        It only stores the history needed for future sliding-window fits.
        """
        if y is None:
            raise ValueError("y must be provided")

        self.last_values = np.asarray(y, float)
        if years is not None:
            self.last_years = np.asarray(years, int)
        else:
            # assume t = 0..T-1
            self.last_years = np.arange(len(self.last_values))
        return self

    # ---------------------------------------------------------
    # Predict one-step using local interpolation
    # ---------------------------------------------------------
    def _predict_one(self):
        values = self.last_values
        years = self.last_years

        window = min(self.window_size, len(values))
        local_values = values[-window:]
        local_years = years[-window:]

        # fit and select best local model
        predict_fn, model_name = self._select_best_model(local_years, local_values)

        # predict next point
        next_t = years[-1] + 1
        next_val = predict_fn(next_t)

        return next_val, model_name

    def predict(self, X_next=None) -> float:
        """
        Predict next year from the final state stored in fit().
        """
        next_val, _ = self._predict_one()
        return float(next_val)

    # ---------------------------------------------------------
    # Multi-step forecasting
    # ---------------------------------------------------------
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
        Multi-step recursive forecasting.
        (We ignore exogenous features for now.)
        """
        preds = []
        current_years = self.last_years.copy()
        current_vals = self.last_values.copy()

        for h in range(1, horizon + 1):
            next_val, model_name = self._select_best_model(
                current_years[-self.window_size:],
                current_vals[-self.window_size:]
            )

            next_year = start_year + h
            pred = next_val(next_year)

            preds.append((next_year, pred))

            # update state for recursion
            current_years = np.append(current_years, next_year)
            current_vals = np.append(current_vals, pred)

        return preds

    # ---------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------
    def get_params(self) -> Dict:
        return {
            "window_size": self.window_size,
            "candidate_models": self.candidate_models,
            "weighted": self.weighted,
            "weight_decay": self.weight_decay,
            "min_window": self.min_window,
            "positive_only": self.positive_only,
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
            "mu": self.mu,
            "rho": self.rho,
            "beta": self.beta,
            "gamma": self.gamma,
            "include_ar": self.include_ar,
            "ar_lags": self.ar_lags,
            "include_ar_squared": self.include_ar_squared,
            "include_exog": self.include_exog,
            "l2_penalty": self.l2_penalty,
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
    }
    
    def __init__(self, 
                 include_ar: bool = True,
                 include_ar_squared: bool = False,
                 include_exog: bool = True,
                 l2_penalty: float = 10.0,
                 rho_bounds: Tuple[float, float] = (0.0, 1),
                 beta_bounds: Tuple[float, float] = (-1.0, 1.0),
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
        bounds = [(None, None)]  # μ unbounded
        
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
        """Get model parameters."""
        params = {
            'mu': self.mu,
            'rho': self.rho,
            'beta': self.beta,
            'gamma': self.gamma,
            'include_ar': self.include_ar,
            'include_ar_squared': self.include_ar_squared,
            'include_exog': self.include_exog,
            'l2_penalty': self.l2_penalty,
            'use_asymmetric_loss': self.use_asymmetric_loss,
            'underestimation_penalty': self.underestimation_penalty,
        }
        
        if self.include_ar:
            params['half_life'] = -np.log(2) / np.log(self.rho) if self.rho > 0 else np.inf
        
        return params


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
                 use_asymmetric_loss: bool = False,
                 underestimation_penalty: float = 2.0, **kwargs):
        
        self._process_init(locals(), MeanRevertingGrowthModelARP)

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
        bounds = [(None, None)]  # μ unbounded
        
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
        """Get model parameters."""
        params = {
            'mu': self.mu,
            'rho': self.rho,  # Now a vector
            'beta': self.beta,
            'gamma': self.gamma,
            'include_ar': self.include_ar,
            'ar_lags': self.ar_lags,
            'include_ar_squared': self.include_ar_squared,
            'include_exog': self.include_exog,
            'l2_penalty': self.l2_penalty,
            'use_asymmetric_loss': self.use_asymmetric_loss,
            'underestimation_penalty': self.underestimation_penalty,
        }
        
        # Compute half-life for AR(1) case
        if self.include_ar and self.ar_lags == 1 and len(self.rho) > 0:
            rho_val = self.rho[0]
            params['half_life'] = -np.log(2) / np.log(rho_val) if rho_val > 0 else np.inf
        
        return params



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
        """Get model parameters."""
        params = {
            'n_pcs': self.n_pcs,
            'effective_n_pcs': self.effective_n_pcs,
            'pca_lambda': self.pca_lambda,
            'model_params': self.model_params,
        }
        
        # Add parameters for each PC model
        for pc_idx, model in enumerate(self.pc_models):
            params[f'pc{pc_idx}_params'] = model.get_params()
        
        # Add explained variance
        if self.pca_model is not None:
            params['explained_variance'] = self.pca_model['explained_variance_']
            params['explained_variance_ratio'] = (
                self.pca_model['explained_variance_'] / 
                np.sum(self.pca_model['explained_variance_'])
            )
        
        return params
    
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
        model.fit(y = y_train, X = X_train)
        
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
    final_model.fit(y = y_all, X = X_all)
    
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