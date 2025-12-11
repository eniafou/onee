from pathlib import Path
import pandas as pd
from onee.config.stf_config import ShortTermForecastConfig
from onee.config.ltf_config import LongTermForecastConfig
from onee.data.loader import DataLoader
from run_stf_cd import run_stf_cd_forecast, prepare_prediction_output as prepare_prediction_output_stf, prepare_ca_output as prepare_ca_output_stf
from run_ltf_cd import run_ltf_cd_forecast, prepare_prediction_output as prepare_prediction_output_ltf, prepare_ca_output as prepare_ca_output_ltf
from onee.data.names import Aliases


def get_latest_year_status(df: pd.DataFrame) -> tuple[int | None, int | None]:
    """
    Get the latest year and month from the DataFrame.
    
    Args:
        df: DataFrame with Annee and Mois columns
        
    Returns:
        Tuple of (latest_year, latest_month) or (None, None) if DataFrame is empty
    """
    if df is None or df.empty:
        return None, None
    latest_year = df[Aliases.ANNEE].max()
    latest_month = df[df[Aliases.ANNEE] == latest_year][Aliases.MOIS].max()
    return latest_year, latest_month

def append_forecast(df_forecast: pd.DataFrame, df_contrats: pd.DataFrame, 
                    target_variable: str) -> pd.DataFrame:
    """
    Append forecast results to the contracts DataFrame.
    
    This function updates existing records or appends new ones based on forecast data.
    Only contracts present in df_forecast are included in the output.
    
    Args:
        df_forecast: DataFrame with forecast results
        df_contrats: Contracts DataFrame to update
        target_variable: Name of the target variable column
        
    Returns:
        Updated df_contrats DataFrame containing only contracts from df_forecast
    """
    # Get unique contracts from forecast
    forecast_contrats = df_forecast[Aliases.CONTRAT].dropna().unique()
    
    # Filter df_contrats to only include contracts that are in df_forecast
    df_contrats = df_contrats[df_contrats[Aliases.CONTRAT].isin(forecast_contrats)].copy()
    
    for _, row in df_forecast.iterrows():
        year = row[Aliases.ANNEE]
        month = row[Aliases.MOIS]
        contrat = row.get(Aliases.CONTRAT)
        value = row[target_variable]

        if contrat is not None:
            mask = (
                (df_contrats[Aliases.ANNEE] == year) & 
                (df_contrats[Aliases.MOIS] == month) & 
                (df_contrats[Aliases.CONTRAT] == contrat)
            )
            if not df_contrats[mask].empty:
                df_contrats.loc[mask, target_variable] = value
            else:
                new_row = {
                    Aliases.ANNEE: year, 
                    Aliases.MOIS: month, 
                    Aliases.CONTRAT: contrat, 
                    target_variable: value
                }
                # Copy other fields from the row if available
                for col in [Aliases.REGION, Aliases.PARTENAIRE, Aliases.ACTIVITE, Aliases.CONSOMMATION_ZCONHC, 
                            Aliases.CONSOMMATION_ZCONHL, Aliases.CONSOMMATION_ZCONHP, Aliases.PUISSANCE_FACTUREE, Aliases.NIVEAU_TENSION]:
                    if col in row:
                        new_row[col] = row[col]
                df_contrats = pd.concat([df_contrats, pd.DataFrame([new_row])], ignore_index=True)

    return df_contrats


def correct_prediction_with_existant(df_forecast: pd.DataFrame, df_existant: pd.DataFrame, 
                                     target_variable: str) -> pd.DataFrame:
    """
    Correct forecast predictions using existing data.
    
    TODO: Implement correction logic if needed (currently returns unchanged forecast).
    
    Args:
        df_forecast: DataFrame with forecast predictions
        df_existant: DataFrame with existing data to use for correction
        target_variable: Name of the target variable column
        
    Returns:
        Corrected forecast DataFrame
    """
    return df_forecast


def _run_stf_forecast(config_stf: ShortTermForecastConfig, 
                      df_contrats: pd.DataFrame, 
                      df_features: pd.DataFrame) -> dict:
    """
    Execute short-term forecast for CD.
    
    Args:
        config_stf: Short-term forecast configuration
        df_contrats: Contracts data
        df_features: Feature data
        
    Returns:
        Result dictionary from run_stf_cd_forecast
    """
    return run_stf_cd_forecast(
        config=config_stf,
        df_contrats=df_contrats,
        df_features=df_features
    )


def _run_ltf_forecast(config_ltf: LongTermForecastConfig,
                      df_contrats: pd.DataFrame, 
                      df_features: pd.DataFrame) -> dict:
    """
    Execute long-term forecast for CD.
    
    Args:
        config_ltf: Long-term forecast configuration
        df_contrats: Contracts data
        df_features: Feature data
        
    Returns:
        Result dictionary from run_ltf_cd_forecast
    """
    return run_ltf_cd_forecast(
        config=config_ltf,
        df_contrats=df_contrats,
        df_features=df_features,
        output_dir=None
    )


def _process_stf_result(result: dict, df_contrats: pd.DataFrame) -> pd.DataFrame | None:
    """
    Process STF result and return formatted DataFrame.
    
    Args:
        result: Result dictionary from STF forecast
        df_contrats: Contracts DataFrame for metadata lookup
        
    Returns:
        Formatted DataFrame or None if result failed
    """
    if result.get('status') == 'success':
        df_prediction = prepare_prediction_output_stf(result['results'], df_contrats)
        return prepare_ca_output_stf(df_prediction, df_contrats)
    return None


def _process_ltf_result(result: dict, df_contrats: pd.DataFrame) -> pd.DataFrame | None:
    """
    Process LTF result and return formatted DataFrame.
    
    Args:
        result: Result dictionary from LTF forecast
        df_contrats: Contracts DataFrame for metadata lookup
        
    Returns:
        Formatted DataFrame or None if result failed
    """
    if result.get('status') == 'success':
        df_prediction = prepare_prediction_output_ltf(result['results'])
        return prepare_ca_output_ltf(df_prediction, df_contrats)
    return None


def _configure_stf_for_year(config_stf: ShortTermForecastConfig, year: int) -> None:
    """
    Configure STF config for a specific evaluation year.
    
    Args:
        config_stf: STF config to modify (in-place)
        year: The year to set for evaluation
    """
    config_stf.evaluation.eval_years_start = year
    config_stf.evaluation.eval_years_end = year


def _configure_ltf_for_year(config_ltf: LongTermForecastConfig, latest_year: int) -> None:
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
                           df_contrats: pd.DataFrame, 
                           df_features: pd.DataFrame, 
                           latest_year: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Process forecasts when the latest year has complete data (month == 12).
    
    Args:
        config_stf: Short-term forecast configuration
        config_ltf: Long-term forecast configuration
        df_contrats: Contracts data
        df_features: Feature data
        latest_year: The latest complete year
        
    Returns:
        Tuple of (STF result DataFrame, LTF result DataFrame)
    """
    # Configure and run STF for next year
    _configure_stf_for_year(config_stf, latest_year + 1)
    stf_result = _run_stf_forecast(config_stf, df_contrats, df_features)
    df_stf = _process_stf_result(stf_result, df_contrats)
    
    # Configure and run LTF
    _configure_ltf_for_year(config_ltf, latest_year)
    ltf_result = _run_ltf_forecast(config_ltf, df_contrats, df_features)
    df_ltf = _process_ltf_result(ltf_result, df_contrats)
    
    return df_stf, df_ltf


def _process_incomplete_year(config_stf: ShortTermForecastConfig, 
                             config_ltf: LongTermForecastConfig,
                             df_contrats: pd.DataFrame, 
                             df_features: pd.DataFrame, 
                             latest_year: int, 
                             latest_month: int, 
                             target_variable: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Process forecasts when the latest year has incomplete data (month < 12).
    
    This performs a two-step STF forecast:
    1. First, complete the current year (months after latest_month)
    2. Then, forecast the full next year
    
    Args:
        config_stf: Short-term forecast configuration
        config_ltf: Long-term forecast configuration
        df_contrats: Contracts data
        df_features: Feature data
        latest_year: The latest year with data
        latest_month: The latest month with data
        target_variable: Name of the target variable
        
    Returns:
        Tuple of (STF result DataFrame, LTF result DataFrame)
    """
    # Step 1: Complete current year forecast
    _configure_stf_for_year(config_stf, latest_year)
    result_current_year = _run_stf_forecast(config_stf, df_contrats, df_features)
    
    if result_current_year.get('status') != 'success':
        print(f"Warning: STF forecast failed for CD (current year)")
        return None, None
    
    # Process current year result
    df_result_current = _process_stf_result(result_current_year, df_contrats)
    if df_result_current is None:
        return None, None
    
    df_result_current = correct_prediction_with_existant(df_result_current, df_contrats, target_variable)
    
    # Update data with current year forecast for next year's forecast
    df_contrats_updated = append_forecast(df_result_current, df_contrats, target_variable)
    
    # Step 2: Forecast next year
    _configure_stf_for_year(config_stf, latest_year + 1)
    result_next_year = _run_stf_forecast(config_stf, df_contrats_updated, df_features)
    
    if result_next_year.get('status') != 'success':
        print(f"Warning: STF forecast failed for CD (next year)")
        return None, None
    
    df_result_next = _process_stf_result(result_next_year, df_contrats_updated)
    if df_result_next is None:
        return None, None
    
    # Combine results: months after latest_month from current year + all of next year
    df_current_remaining = df_result_current[df_result_current[Aliases.MOIS] > latest_month]
    df_stf_combined = pd.concat([df_current_remaining, df_result_next], axis=0, ignore_index=True)
    
    # Run LTF forecast
    _configure_ltf_for_year(config_ltf, latest_year)
    ltf_result = _run_ltf_forecast(config_ltf, df_contrats_updated, df_features)
    df_ltf = _process_ltf_result(ltf_result, df_contrats_updated)
    
    return df_stf_combined, df_ltf


def extrapolate_features(df_features: pd.DataFrame, latest_year: int) -> pd.DataFrame:
    """
    Extrapolate features to cover the latest_year if needed.
    
    Args:
        df_features: Features DataFrame
        latest_year: The year to extrapolate to
        
    Returns:
        Updated features DataFrame with extrapolated values
    """
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


def run_full_forecast_cd() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run CD forecasts for all contracts.
    
    This function loads configuration, loads contract data, and runs both
    short-term (STF) and long-term (LTF) forecasts.
        
    Returns:
        Tuple of (df_stf_results, df_ltf_results) DataFrames containing all forecasts
    """
    # Load configuration
    config_path_stf = Path(__file__).parent / "configs/stf_cd.yaml"
    config_path_ltf = Path(__file__).parent / "configs/ltf_cd.yaml"
    config_stf = ShortTermForecastConfig.from_yaml(config_path_stf)
    config_ltf = LongTermForecastConfig.from_yaml(config_path_ltf)
    
    PROJECT_ROOT = config_stf.project.project_root
    VARIABLE = config_stf.data.variable
    
    # Initialize DataLoader
    data_loader = DataLoader(PROJECT_ROOT)
    
    print("Loading CD data...")
    
    # Load CD data (contracts and features)
    df_contrats, df_features = data_loader.load_cd_data(
        db_path=config_stf.project.project_root / config_stf.data.db_path,
    )

    # Get latest year and month from data
    latest_year, latest_month = get_latest_year_status(df_contrats)
    if latest_year is None:
        print("Warning: No data found for CD, exiting...")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Latest data: {latest_year}/{latest_month}")
    
    # Extrapolate features if needed
    df_features = extrapolate_features(df_features, latest_year + 1)

    # Process based on whether the year is complete or not
    if latest_month == 12:
        print(f"Processing complete year {latest_year}...")
        df_stf, df_ltf = _process_complete_year(
            config_stf, config_ltf, df_contrats, df_features, latest_year
        )
    else:
        print(f"Processing incomplete year {latest_year} (month {latest_month})...")
        df_stf, df_ltf = _process_incomplete_year(
            config_stf, config_ltf, df_contrats, df_features, 
            latest_year, latest_month, VARIABLE
        )
    
    # Return results (handle None cases)
    df_results_stf = df_stf if df_stf is not None else pd.DataFrame()
    df_results_ltf = df_ltf if df_ltf is not None else pd.DataFrame()
    
    return df_results_stf, df_results_ltf


if __name__ == "__main__":
    df_stf, df_ltf = run_full_forecast_cd()
    df_stf.to_csv("stf_cd_results.csv", index=False)
    df_ltf.to_csv("ltf_cd_results.csv", index=False)
    print(f"STF results saved: {len(df_stf)} rows")
    print(f"LTF results saved: {len(df_ltf)} rows")
