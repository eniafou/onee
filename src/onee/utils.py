import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import plotly.express as px
from typing import Union, Callable, Iterable, Mapping


def custom_mean_absolute_percentage_error(y_true, y_pred, ignore_zeros=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if ignore_zeros:
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    # To avoid division by zero when ignore_zeros=False
    # we replace zeros with a very small number
    denominator = np.where(y_true == 0, np.finfo(float).eps, y_true)

    mape = np.mean(np.abs((y_true - y_pred) / denominator))
    return mape

def create_monthly_matrix(df, value_col):
    """Convert long format to year x month matrix"""
    pivot = df.pivot_table(
        index='annee',
        columns='mois',
        values=value_col,
        aggfunc='sum'
    )
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = 0
    return pivot[sorted(pivot.columns)].values


def plot_var_over_time(df, entity, entity_value, var="consommation"):
    subset = df[df[entity] == entity_value].copy()

    # Build a proper datetime column from year and month
    subset["date"] = pd.to_datetime(subset["annee"].astype(str) + '-' + subset["mois"].astype(str) + '-01')

    # Sort by date so the line is continuous
    subset = subset.sort_values("date")

    # Plot
    plt.figure(figsize=(12, 5))  # wider figure
    plt.plot(subset["date"], subset[var])
    plt.title(f"{entity} {entity_value}")
    plt.xlabel("Date (Year-Month)")
    plt.ylabel(var)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



import numpy as np
import pandas as pd

def add_exogenous_features(
    X_lags,
    years,
    feature_block,
    df_features,
    use_rate=False,
    change_type="log",   # "pct", "diff", or "log"
    fill_value=0.0,      # value to fill NaN/Inf after transformation
    log_epsilon=None     # if not None, computes ln(f+ε) - ln(f_prev+ε)
):
    """
    Add exogenous features (or their change) aligned to provided years.

    change_type:
      - "pct"  : (f_t - f_{t-1}) / f_{t-1}
      - "diff" : f_t - f_{t-1}
      - "log"  : ln(f_t) - ln(f_{t-1}) = ln(f_t / f_{t-1})
                 Requires f_t > 0 and f_{t-1} > 0 unless log_epsilon is set.
    """
    if not feature_block:
        return X_lags

    df = df_features.copy()
    missing = [c for c in feature_block + ['annee'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df_features: {missing}")

    df = df.sort_values('annee').reset_index(drop=True)

    if use_rate:
        F = df[feature_block].apply(pd.to_numeric, errors='coerce')

        if change_type == "pct":
            F_change = F.pct_change()
        elif change_type == "diff":
            F_change = F.diff()
        elif change_type == "log":
            if log_epsilon is None:
                # Invalid where non-positive; will be filled later
                F_log = np.log(F.where(F > 0))
            else:
                # Shift all values by ε before log; user is responsible for choosing ε
                F_log = np.log(F + log_epsilon)
            F_change = F_log.diff()
        else:
            raise ValueError("change_type must be 'pct', 'diff', or 'log'")

        F_change = F_change.replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        df_exog = pd.concat([df[['annee']], F_change], axis=1)

    else:
        F_raw = df[feature_block].apply(pd.to_numeric, errors='coerce').fillna(fill_value)
        df_exog = pd.concat([df[['annee']], F_raw], axis=1)

    year_to_vals = (
        df_exog.drop_duplicates(subset=['annee'])
               .set_index('annee')[feature_block]
               .to_dict(orient='index')
    )

    exog_data = []
    zero_row = [fill_value] * len(feature_block)

    for year in years:
        vals = year_to_vals.get(year)
        if vals is None:
            exog_data.append(zero_row)
        else:
            row = [fill_value if pd.isna(vals.get(col)) else vals.get(col) for col in feature_block]
            exog_data.append(row)

    exog_array = np.asarray(exog_data, dtype=float)
    return np.hstack([X_lags, exog_array])


def add_monthly_feature(X, years, df_monthly, feature = "temperature", agg_method = "mean"):
    """Add monthly feature values"""
    temp_data = []
    for year in years:
        year_temps = df_monthly[df_monthly['annee'] == year].groupby('mois')[feature].agg(agg_method)
        temps = []
        for m in range(1, 13):
            val = year_temps.get(m, 0)
            if pd.isna(val):
                val = 0
            temps.append(val)
        temp_data.append(temps)

    temp_array = np.array(temp_data)
    temp_array = np.nan_to_num(temp_array, nan=0.0)
    return np.hstack([X, temp_array])


def add_yearly_feature(X, years, df_yearly, feature="temperature", agg_method="mean"):
    """Add yearly feature values"""
    yearly_data = []
    for year in years:
        val = df_yearly[df_yearly['annee'] == year][feature].agg(agg_method)
        if pd.isna(val):
            val = 0
        yearly_data.append([val])

    yearly_array = np.array(yearly_data)
    yearly_array = np.nan_to_num(yearly_array, nan=0.0)
    return np.hstack([X, yearly_array])


def safe_metric(func, y_true, y_pred):
    """Safely compute metric ignoring NaNs"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return np.nan
    try:
        return func(y_true[mask], y_pred[mask])
    except Exception:
        return np.nan

def calculate_all_metrics(actual_monthly, pred_monthly, actual_annual, pred_annual):
    """Calculate comprehensive metrics safely, even with NaNs"""
    out = {
        'monthly_mae': safe_metric(mean_absolute_error, actual_monthly, pred_monthly),
        'monthly_mape': safe_metric(custom_mean_absolute_percentage_error, actual_monthly, pred_monthly) * 100,
        'monthly_r2': safe_metric(r2_score, actual_monthly, pred_monthly),
        'annual_mae': safe_metric(mean_absolute_error, actual_annual, pred_annual),
        'annual_mape': safe_metric(custom_mean_absolute_percentage_error, actual_annual, pred_annual) * 100,
        'annual_r2': safe_metric(r2_score, actual_annual, pred_annual),
    }
    return out

def calculate_all_annual_metrics(actual_annual, pred_annual):
    """Calculate comprehensive metrics safely, even with NaNs"""
    out = {
        'annual_mae': safe_metric(mean_absolute_error, actual_annual, pred_annual),
        'annual_mape': safe_metric(custom_mean_absolute_percentage_error, actual_annual, pred_annual) * 100,
        'annual_r2': safe_metric(r2_score, actual_annual, pred_annual),
    }
    return out

def select_best_model(results, r2_threshold):
    """Select best model: lowest MAE with R² > threshold"""
    valid_results = [r for r in results if r['annual_r2'] >= r2_threshold]

    if not valid_results:
        return max(results, key=lambda x: x['annual_r2'])

    return min(valid_results, key=lambda x: x['annual_mae'])

def safe_parse_date(x):
    """Try to parse a date or datetime string safely (return None if it fails)."""
    if pd.isna(x) or x in ['', None]:
        return None

    # If it's already a datetime or pandas Timestamp, return as datetime
    if isinstance(x, (datetime, pd.Timestamp)):
        return x if isinstance(x, datetime) else x.to_pydatetime()

    x = str(x).strip()

    # Try common date formats
    common_formats = [
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y",
        "%m/%d/%Y",          # US-style
        "%d %b %Y",          # e.g., 24 Nov 2022
        "%d %B %Y",          # e.g., 24 November 2022
    ]

    for fmt in common_formats:
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            continue

    return None  # could not parse

def get_move_in_year(df_c, not_started_yet_th = 20, strict = False):
    """
    Extract the first non-null 'Date d'emménagement' value from a DataFrame
    and return its year as an integer, or None if not available.
    """
    if 'Date d\'emménagement' not in df_c.columns:
        return None

    move_in_val = df_c['Date d\'emménagement'].dropna()
    if move_in_val.empty:
        return None

    move_in_date = safe_parse_date(move_in_val.iloc[0])
    if move_in_date is None:
        return None
    
    move_in_year = move_in_date.year

    if strict:
        return move_in_year

    move_out_year = safe_parse_date(df_c['Date de déménagement'].dropna().iloc[0]).year
    
    max_year = df_c["annee"].max()
    annual_consumption = df_c[df_c["annee"] == move_in_year]["consommation"].sum()
    while annual_consumption <= not_started_yet_th and (move_out_year is None or move_in_year < move_out_year):
        move_in_year+=1
        if move_in_year >= max_year:
            break
        annual_consumption = df_c[df_c["annee"] == move_in_year]["consommation"].sum()

    return move_in_year


def get_move_out_year(df_c):
    """
    Extract the last non-null 'Date de déménagement' value from a DataFrame
    and return its year as an integer, or None if not available.
    """
    if "Date de déménagement" not in df_c.columns:
        return None

    move_out_vals = df_c["Date de déménagement"].dropna()
    if move_out_vals.empty:
        return None

    move_out_date = safe_parse_date(move_out_vals.iloc[-1])
    if move_out_date is None:
        return None

    move_out_year = move_out_date.year

    return move_out_year

def add_annual_client_feature(
    X: np.ndarray, years: Iterable[int], series_lookup: Mapping[int, np.ndarray]
) -> np.ndarray:
    """
    Append the annual sum of monthly client predictions per year.
    """
    if not series_lookup:
        return X

    annual_values = []
    for year in years:
        series = series_lookup.get(int(year))
        if series is None:
            annual_values.append([0.0])
        else:
            annual_values.append([float(np.sum(series))])

    if not annual_values:
        return X

    annual_array = np.array(annual_values, dtype=float)
    return np.hstack([X, annual_array])


def plot_time_evolution(
    df: pd.DataFrame,
    year_col: str,
    month_col: str,
    value_col: str,
    freq: str = "monthly",               # "monthly" or "yearly"
    agg: Union[str, Callable] = "sum",    # e.g. "sum","mean","median","max","min" or a callable
    title: str = None
):
    """
    Plot the evolution of `value_col` over time using (year, month) columns.

    - If `freq="monthly"`: aggregates by year+month then plots monthly values.
    - If `freq="yearly"` : aggregates by year+month first, then aggregates those
      monthly values into yearly values using the same `agg`.

    Parameters
    ----------
    df : DataFrame containing at least [year_col, month_col, value_col]
    year_col, month_col : column names for year and month (month as 1–12)
    value_col : name of the variable to plot
    freq : "monthly" | "yearly"
    agg : aggregation function name or callable (sum/mean/median/max/min/…)
    title : optional plot title

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if freq not in {"monthly", "yearly"}:
        raise ValueError("freq must be 'monthly' or 'yearly'")

    # Ensure integers for year/month and build a first-of-month timestamp
    tmp = df[[year_col, month_col, value_col]].copy()
    tmp[year_col] = tmp[year_col].astype(int)
    tmp[month_col] = tmp[month_col].astype(int)
    tmp["date"] = pd.to_datetime(
        {"year": tmp[year_col], "month": tmp[month_col], "day": 1},
        errors="coerce"
    )

    # 1) Aggregate to monthly (handles duplicates per month safely)
    monthly = (
        tmp.groupby("date", as_index=False)
           .agg(**{value_col: (value_col, agg)})
           .sort_values("date")
    )

    if freq == "monthly":
        fig = px.line(
            monthly, x="date", y=value_col, markers=True,
            title=title or f"{value_col} — monthly ({agg})"
        )
        fig.update_layout(xaxis_title="Month", yaxis_title=value_col)
        return fig

    # 2) Aggregate monthly -> yearly
    yearly = (
        monthly.assign(year=monthly["date"].dt.year)
               .groupby("year", as_index=False)
               .agg(**{value_col: (value_col, agg)})
               .sort_values("year")
    )
    fig = px.line(
        yearly, x="year", y=value_col, markers=True,
        title=title or f"{value_col} — yearly ({agg} of monthly)"
    )
    fig.update_layout(xaxis_title="Year", yaxis_title=value_col)
    return fig
