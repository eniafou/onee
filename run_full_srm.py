from pathlib import Path
import pandas as pd
from onee.config.stf_config import ShortTermForecastConfig
from onee.config.ltf_config import LongTermForecastConfig
from onee.data.loader import DataLoader
from run_stf_srm import run_stf_srm_forecast, prepare_prediction_output as prepare_prediction_output_stf
from run_ltf_srm import run_ltf_srm_forecast, prepare_prediction_output as prepare_prediction_output_ltf
from onee.data.names import Aliases
# ...existing imports...
from full_forecast_utils import (
    get_latest_year_status,
    extrapolate_features,
    configure_stf_for_year,
    configure_ltf_for_year,
    rename_to_ltf_srm_results,
    rename_to_stf_srm_results,
    apply_consumption_adjustment
)


def compute_df_srm(df_regional: pd.DataFrame, df_dist: pd.DataFrame | None, 
                   var_cols: dict, target_variable: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute df_srm (Regional + Distributors combined) and update df_regional with distributors.
    
    First attempts to extract df_srm from an existing "Total" activity (which represents the full SRM).
    If "Total" doesn't exist, falls back to aggregating all activities + distributors to compute it.
    
    Args:
        df_regional: Regional DataFrame
        df_dist: Distributors DataFrame (can be None)
        var_cols: Dict with column names for regional and distributor variables
        target_variable: Name of the target variable column
        
    Returns:
        Tuple of (df_regional_updated, df_srm) where:
        - df_regional_updated: Regional data with "All distributers" added as a new activity (if distributors exist)
        - df_srm: DataFrame with [Annee, Mois, target_variable] columns (Regional + Distributors combined)
    """
    reg_var_col = var_cols["regional"]
    df_regional_updated = df_regional.copy()
    
    # Try to get df_srm from existing "Total" activity first (Total = full SRM)
    df_total_activity = df_regional[df_regional[Aliases.ACTIVITE] == "Total"]
    
    if not df_total_activity.empty:
        # "Total" activity exists, represents total regional consumption
        df_total_regional = (
            df_total_activity[[Aliases.ANNEE, Aliases.MOIS, reg_var_col]]
            .copy()
            .rename(columns={reg_var_col: target_variable})
        )
        
        # Combine with distributors to get df_srm
        if df_dist is not None and len(df_dist) > 0:
            dist_var_col = var_cols["distributor"]
            df_all_dist = (
                df_dist.groupby([Aliases.ANNEE, Aliases.MOIS])
                .agg({dist_var_col: 'sum'})
                .reset_index()
                .rename(columns={dist_var_col: target_variable})
            )
            
            # Add distributors as a new activity in df_regional_updated
            df_dist_as_activity = df_all_dist.copy()
            df_dist_as_activity[Aliases.ACTIVITE] = "All distributers"
            df_dist_as_activity = df_dist_as_activity.rename(columns={target_variable: reg_var_col})
            df_regional_updated = pd.concat([df_regional_updated, df_dist_as_activity], ignore_index=True)
            
            # Compute combined SRM (Regional total + Distributors)
            df_srm = (
                pd.concat(
                    [df_total_regional[[Aliases.ANNEE, Aliases.MOIS, target_variable]],
                     df_all_dist[[Aliases.ANNEE, Aliases.MOIS, target_variable]]],
                    ignore_index=True
                )
                .groupby([Aliases.ANNEE, Aliases.MOIS])
                .agg({target_variable: 'sum'})
                .reset_index()
            )
        else:
            df_srm = df_total_regional[[Aliases.ANNEE, Aliases.MOIS, target_variable]].copy()
    else:
        # "Total" doesn't exist, aggregate all activities (excluding "Total")
        df_non_total = df_regional[df_regional[Aliases.ACTIVITE] != "Total"]
        df_total_regional = (
            df_non_total.groupby([Aliases.ANNEE, Aliases.MOIS])
            .agg({reg_var_col: 'sum'})
            .reset_index()
            .rename(columns={reg_var_col: target_variable})
        )
        
        # Process distributors if available
        if df_dist is not None and len(df_dist) > 0:
            dist_var_col = var_cols["distributor"]
            df_all_dist = (
                df_dist.groupby([Aliases.ANNEE, Aliases.MOIS])
                .agg({dist_var_col: 'sum'})
                .reset_index()
                .rename(columns={dist_var_col: target_variable})
            )
            
            # Add distributors as a new activity in df_regional_updated
            df_dist_as_activity = df_all_dist.copy()
            df_dist_as_activity[Aliases.ACTIVITE] = "All distributers"
            df_dist_as_activity = df_dist_as_activity.rename(columns={target_variable: reg_var_col})
            df_regional_updated = pd.concat([df_regional_updated, df_dist_as_activity], ignore_index=True)
            
            # Compute combined SRM (Regional total + Distributors)
            df_srm = (
                pd.concat(
                    [df_total_regional[[Aliases.ANNEE, Aliases.MOIS, target_variable]],
                     df_all_dist[[Aliases.ANNEE, Aliases.MOIS, target_variable]]],
                    ignore_index=True
                )
                .groupby([Aliases.ANNEE, Aliases.MOIS])
                .agg({target_variable: 'sum'})
                .reset_index()
            )
        else:
            df_srm = df_total_regional[[Aliases.ANNEE, Aliases.MOIS, target_variable]].copy()
    
    return df_regional_updated, df_srm


def append_forecast(df_forecast: pd.DataFrame, df_srm: pd.DataFrame, 
                    df_regional: pd.DataFrame, target_variable: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Append forecast results to the SRM and regional DataFrames.
    
    This function updates existing records or appends new ones based on forecast data.
    For "Total" activity, updates df_srm. For other activities, updates df_regional.
    Only activities present in df_forecast are included in the output.
    
    Args:
        df_forecast: DataFrame with forecast results (must have Year, Month, Activity, target_variable columns)
        df_srm: SRM DataFrame to update
        df_regional: Regional DataFrame to update
        target_variable: Name of the target variable column
        
    Returns:
        Tuple of (updated df_srm, updated df_regional)
    """
    # Work with copies to avoid modifying original DataFrames unexpectedly
    df_srm = df_srm.copy()
    df_regional = df_regional.copy()
    
    # Get unique activities from df_forecast
    forecast_activities = df_forecast[Aliases.ACTIVITE].unique()
    
    # Filter df_regional to only include activities present in df_forecast
    non_total_activities = [a for a in forecast_activities if a != "Total"]
    if non_total_activities:
        df_regional = df_regional[df_regional[Aliases.ACTIVITE].isin(non_total_activities)]
    else:
        # No non-Total activities in forecast, return empty regional
        df_regional = df_regional.iloc[0:0]
    
    for _, row in df_forecast.iterrows():
        year = row[Aliases.ANNEE]
        month = row[Aliases.MOIS]
        activity = row[Aliases.ACTIVITE]
        value = row[target_variable]

        if activity == "Total":
            mask = (df_srm[Aliases.ANNEE] == year) & (df_srm[Aliases.MOIS] == month)
            if not df_srm[mask].empty:
                df_srm.loc[mask, target_variable] = value
            else:
                new_row = {Aliases.ANNEE: year, Aliases.MOIS: month, target_variable: value}
                df_srm = pd.concat([df_srm, pd.DataFrame([new_row])], ignore_index=True)
        else:
            mask = (
                (df_regional[Aliases.ANNEE] == year) & 
                (df_regional[Aliases.MOIS] == month) & 
                (df_regional[Aliases.ACTIVITE] == activity)
            )
            if not df_regional[mask].empty:
                df_regional.loc[mask, target_variable] = value
            else:
                new_row = {
                    Aliases.ANNEE: year, 
                    Aliases.MOIS: month, 
                    Aliases.ACTIVITE: activity, 
                    target_variable: value
                }
                df_regional = pd.concat([df_regional, pd.DataFrame([new_row])], ignore_index=True)

    return df_srm, df_regional


def correct_prediction_with_existant(df_forecast: pd.DataFrame, df_existant: pd.DataFrame, 
                                         target_variable: str) -> pd.DataFrame:
    """
    Correct SRM forecast predictions using existing data.
    
    For each activity in a region, compares the initial predictions with realized consumption
    and applies an adjustment factor to future predictions.
    
    Args:
        df_forecast: DataFrame with forecast predictions (STF SRM results format)
        df_existant: DataFrame with existing regional data (without Region column)
        target_variable: Name of the target variable column (should be Aliases.CONSOMMATION_KWH)
        
    Returns:
        Corrected forecast DataFrame with adjusted consumption values
    """
    # Make a copy to avoid modifying the original
    df_corrected = df_forecast.copy()
    
    # Get unique activities from forecast
    activity_groups = df_corrected.groupby(Aliases.ACTIVITE)
    for activite, group_forecast in activity_groups:
        # Filter existing data for this specific activity
        mask_existant = (df_existant[Aliases.ACTIVITE] == activite)
        df_activity_existant = df_existant[mask_existant].copy()
        
        if df_activity_existant.empty:
            # No existing data for this activity, skip correction
            continue
        
        # Get the forecast year (assuming all rows in group have same year)
        forecast_year = group_forecast[Aliases.ANNEE].iloc[0]
        
        # Filter existing data for the forecast year
        df_year_existant = df_activity_existant[
            df_activity_existant[Aliases.ANNEE] == forecast_year
        ].copy()
        
        if df_year_existant.empty:
            # No existing data for this year, skip correction
            continue
        
        # Sort both dataframes by month
        group_forecast_sorted = group_forecast.sort_values(Aliases.MOIS)
        df_year_existant_sorted = df_year_existant.sort_values(Aliases.MOIS)
        
        # Get the 12 months of initial predictions
        prediction_consomation_initiale = group_forecast_sorted[target_variable].tolist()
        
        # Ensure we have exactly 12 predictions
        if len(prediction_consomation_initiale) != 12:
            continue
        
        # Get realized consumption (only available months)
        consommation_realise = df_year_existant_sorted[target_variable].tolist()
        
        # Apply the adjustment
        try:
            correction_prediction = apply_consumption_adjustment(
                prediction_consomation_initiale, 
                consommation_realise
            )
        except (ValueError, ZeroDivisionError):
            # Skip correction if there's an error
            continue
        
        # Update the corrected dataframe with new values
        indices = group_forecast_sorted.index
        for i, idx in enumerate(indices):
            corrected_total = correction_prediction[i]
            
            # Update total consumption
            df_corrected.loc[idx, target_variable] = corrected_total
    
    return df_corrected

def _run_stf_forecast(config_stf: ShortTermForecastConfig, target_region: str, 
                      region_mode: int, df_regional: pd.DataFrame, 
                      df_features: pd.DataFrame, df_srm: pd.DataFrame) -> dict:
    """
    Execute short-term forecast for a region.
    
    Args:
        config_stf: Short-term forecast configuration
        target_region: Target region name
        region_mode: Region run mode
        df_regional: Regional data
        df_features: Feature data
        df_srm: SRM combined data
        
    Returns:
        Result dictionary from run_stf_srm_forecast
    """
    return run_stf_srm_forecast(
        config=config_stf,
        target_region=target_region,
        region_mode=region_mode,
        df_regional=df_regional,
        df_features=df_features,
        df_srm=df_srm
    )


def _run_ltf_forecast(config_ltf: LongTermForecastConfig, target_region: str,
                      df_regional: pd.DataFrame, df_features: pd.DataFrame, 
                      df_srm: pd.DataFrame) -> dict:
    """
    Execute long-term forecast for a region.
    
    Args:
        config_ltf: Long-term forecast configuration
        target_region: Target region name
        df_regional: Regional data
        df_features: Feature data
        df_srm: SRM combined data
        
    Returns:
        Result dictionary from run_ltf_srm_forecast
    """
    return run_ltf_srm_forecast(
        config=config_ltf,
        target_region=target_region,
        df_regional=df_regional,
        df_features=df_features,
        df_srm=df_srm,
        use_output_dir=False
    )


def _process_stf_result(result: dict, target_region: str) -> pd.DataFrame | None:
    """
    Process STF result and return formatted DataFrame.
    
    Args:
        result: Result dictionary from STF forecast
        target_region: Target region name
        
    Returns:
        Formatted DataFrame or None if result failed
    """
    if result.get('status') == 'success':
        df_prediction = prepare_prediction_output_stf(target_region, result['results'])
        df_prediction = df_prediction[df_prediction[Aliases.ACTIVITE] == "Total"]
        return df_prediction
    return None


def _process_ltf_result(result: dict, target_region: str) -> pd.DataFrame | None:
    """
    Process LTF result and return formatted DataFrame.
    
    Args:
        result: Result dictionary from LTF forecast
        target_region: Target region name
        
    Returns:
        Formatted DataFrame or None if result failed
    """
    if result.get('status') == 'success':
        return prepare_prediction_output_ltf(target_region, result['results'])
    return None


def configure_stf_for_year(config_stf: ShortTermForecastConfig, year: int) -> None:
    """
    Configure STF config for a specific evaluation year.
    
    Args:
        config_stf: STF config to modify (in-place)
        year: The year to set for evaluation
    """
    config_stf.evaluation.eval_years_start = year
    config_stf.evaluation.eval_years_end = year


def configure_ltf_for_year(config_ltf: LongTermForecastConfig, latest_year: int) -> None:
    """
    Configure LTF config to use latest_year as the base year for forecasting.
    
    Args:
        config_ltf: LTF config to modify (in-place)
        latest_year: The latest year of data to use as base
    """
    config_ltf.temporal.forecast_runs[0] = (
        config_ltf.temporal.forecast_runs[0][0],
        latest_year
    )


def _process_complete_year(config_stf: ShortTermForecastConfig, 
                           config_ltf: LongTermForecastConfig,
                           target_region: str, region_mode: int,
                           df_regional: pd.DataFrame, df_features: pd.DataFrame, 
                           df_srm: pd.DataFrame, latest_year: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Process forecasts when the latest year has complete data (month == 12).
    
    Args:
        config_stf: Short-term forecast configuration
        config_ltf: Long-term forecast configuration
        target_region: Target region name
        region_mode: Region run mode
        df_regional: Regional data
        df_features: Feature data
        df_srm: SRM combined data
        latest_year: The latest complete year
        
    Returns:
        Tuple of (STF result DataFrame, LTF result DataFrame)
    """
    # Configure and run STF for next year
    configure_stf_for_year(config_stf, latest_year + 1)
    stf_result = _run_stf_forecast(
        config_stf, target_region, region_mode, df_regional, df_features, df_srm
    )
    df_stf = _process_stf_result(stf_result, target_region)
    
    # Configure and run LTF
    configure_ltf_for_year(config_ltf, latest_year)
    ltf_result = _run_ltf_forecast(
        config_ltf, target_region, df_regional, df_features, df_srm
    )
    df_ltf = _process_ltf_result(ltf_result, target_region)
    
    return df_stf, df_ltf


def _process_incomplete_year(config_stf: ShortTermForecastConfig, 
                             config_ltf: LongTermForecastConfig,
                             target_region: str, region_mode: int,
                             df_regional: pd.DataFrame, df_features: pd.DataFrame, 
                             df_srm: pd.DataFrame, latest_year: int, 
                             latest_month: int, target_variable: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Process forecasts when the latest year has incomplete data (month < 12).
    
    This performs a two-step STF forecast:
    1. First, complete the current year (months after latest_month)
    2. Then, forecast the full next year
    
    Args:
        config_stf: Short-term forecast configuration
        config_ltf: Long-term forecast configuration
        target_region: Target region name
        region_mode: Region run mode
        df_regional: Regional data
        df_features: Feature data
        df_srm: SRM combined data
        latest_year: The latest year with data
        latest_month: The latest month with data
        target_variable: Name of the target variable
        
    Returns:
        Tuple of (STF result DataFrame, LTF result DataFrame)
    """
    # Step 1: Complete current year forecast
    configure_stf_for_year(config_stf, latest_year)
    result_current_year = _run_stf_forecast(
        config_stf, target_region, region_mode, df_regional, df_features, df_srm
    )
    
    if result_current_year.get('status') != 'success':
        print(f"Warning: STF forecast failed for {target_region} (current year)")
        return None, None
    # Prepare existing data for correction
    df_srm_copy = df_srm.copy()
    df_srm_copy[Aliases.ACTIVITE] = "Total"
    df_existant = pd.concat([df_regional.copy(), df_srm_copy], ignore_index=True)
    
    # Process current year result
    try:
        df_result_current, region_mode = prepare_prediction_output_stf(target_region, result_current_year['results'], return_used_mode = True)
    except ValueError as e:
        print("Error in preparing STF output:", e)
        df_result_current= prepare_prediction_output_stf(target_region, result_current_year['results'])

    df_result_current = correct_prediction_with_existant(df_result_current, df_existant, target_variable)

    # Update data with current year forecast for next year's forecast
    df_srm_updated, df_regional_updated = append_forecast(
        df_result_current, df_srm, df_regional, target_variable
    )

    # Step 2: Forecast next year
    configure_stf_for_year(config_stf, latest_year + 1)
    result_next_year = _run_stf_forecast(
        config_stf, target_region, region_mode, df_regional_updated, df_features, df_srm_updated
    )
    
    if result_next_year.get('status') != 'success':
        print(f"Warning: STF forecast failed for {target_region} (next year)")
        return None, None
    
    df_result_next = prepare_prediction_output_stf(target_region, result_next_year['results'])
    
    # Combine results: months after latest_month from current year + all of next year
    df_result_current = df_result_current[df_result_current[Aliases.ACTIVITE] == "Total"]
    df_result_next = df_result_next[df_result_next[Aliases.ACTIVITE] == "Total"]
    df_current_remaining = df_result_current[df_result_current[Aliases.MOIS] > latest_month]
    df_stf_combined = pd.concat([df_current_remaining, df_result_next], axis=0, ignore_index=True)
    
    # Run LTF forecast
    configure_ltf_for_year(config_ltf, latest_year)
    ltf_result = _run_ltf_forecast(
        config_ltf, target_region, df_regional_updated, df_features, df_srm_updated
    )
    df_ltf = _process_ltf_result(ltf_result, target_region)
    
    return df_stf_combined, df_ltf

def extrapolate_features(df_features: pd.DataFrame, latest_year: int) -> pd.DataFrame:
    if latest_year in df_features[Aliases.ANNEE].values:
        return df_features 
    df_features = df_features.sort_values(by=Aliases.ANNEE)
    last_year = df_features[Aliases.ANNEE].max()
    years_to_add = latest_year - last_year
    df_to_append = df_features[df_features[Aliases.ANNEE] == last_year].copy()
    for year in range(1, years_to_add + 1):
        new_year = last_year + year
        df_new = df_to_append.copy()
        df_new[Aliases.ANNEE] = new_year
        for col in df_features.columns:
            if col != Aliases.ANNEE and pd.api.types.is_numeric_dtype(df_features[col]):
                prev_val = df_features[df_features[Aliases.ANNEE] == last_year - 1][col].values[0]
                curr_val = df_features[df_features[Aliases.ANNEE] == last_year][col].values[0]
                if prev_val != 0:
                    growth_rate = (curr_val - prev_val) / prev_val
                    df_new[col] = df_new[col] * (1 + growth_rate)
        df_features = pd.concat([df_features, df_new], ignore_index=True)
    return df_features


def run_full_forecast_srm(regions_override: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run SRM forecasts for all configured regions.
    
    This function loads configuration, iterates through regions, and runs both
    short-term (STF) and long-term (LTF) forecasts for each region.
    
    Args:
        regions_override: Optional dict to override regions from config.
                         Keys are region names, values are region modes.
        
    Returns:
        Tuple of (df_stf_results, df_ltf_results) DataFrames containing all forecasts
    """
    # Load configuration
    config_path_stf = Path(__file__).parent / "configs/stf_srm.yaml"
    config_path_ltf = Path(__file__).parent / "configs/ltf_srm.yaml"
    config_stf = ShortTermForecastConfig.from_yaml(config_path_stf)
    config_ltf = LongTermForecastConfig.from_yaml(config_path_ltf)
    
    PROJECT_ROOT = config_stf.project.project_root
    VARIABLE = config_stf.data.variable
    REGIONS = regions_override if regions_override is not None else config_stf.data.regions
    
    # Initialize DataLoader
    data_loader = DataLoader(PROJECT_ROOT)
    
    # Collect results
    all_results_stf = []
    all_results_ltf = []

    for target_region, region_mode in list(REGIONS.items()):
        print(f"Loading data for {target_region}...")
        
        # Load region data
        df_regional, df_features, df_dist, var_cols = data_loader.load_srm_data(
            db_path=config_stf.project.project_root / config_stf.data.db_path,
            variable=VARIABLE,
            target_region=target_region,
        )

        # Get latest year and month from data
        latest_year, latest_month = get_latest_year_status(df_regional)
        if latest_year is None:
            print(f"Warning: No data found for {target_region}, skipping...")
            continue
        
        df_features = extrapolate_features(df_features, latest_year + 1)

        # Compute combined SRM data (Regional + Distributors) and update df_regional with distributors
        df_regional, df_srm = compute_df_srm(df_regional, df_dist, var_cols, VARIABLE)
        # Process based on whether the year is complete or not
        if latest_month == 12:
            df_stf, df_ltf = _process_complete_year(
                config_stf, config_ltf, target_region, region_mode,
                df_regional, df_features, df_srm, latest_year
            )
        else:
            df_stf, df_ltf = _process_incomplete_year(
                config_stf, config_ltf, target_region, region_mode,
                df_regional, df_features, df_srm, latest_year, latest_month, VARIABLE
            )
        
        # Collect results
        if df_stf is not None:
            all_results_stf.append(df_stf)
        if df_ltf is not None:
            all_results_ltf.append(df_ltf)

    # Combine all results
    df_results_stf = pd.concat(all_results_stf, axis=0, ignore_index=True) if all_results_stf else pd.DataFrame()
    df_results_ltf = pd.concat(all_results_ltf, axis=0, ignore_index=True) if all_results_ltf else pd.DataFrame()
    
    return rename_to_stf_srm_results(df_results_stf), rename_to_ltf_srm_results(df_results_ltf)

if __name__ == "__main__":
    df_stf, df_ltf = run_full_forecast_srm()
    df_stf.to_csv("stf_srm_results.csv", index=False, encoding="utf-8-sig")
    df_ltf.to_csv("ltf_srm_results.csv", index=False, encoding="utf-8-sig")
    print(f"STF results saved: {len(df_stf)} rows")
    print(f"LTF results saved: {len(df_ltf)} rows")