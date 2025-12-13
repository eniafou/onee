"""Shared utilities for SRM and CD forecast runners."""

import pandas as pd
from onee.config.stf_config import ShortTermForecastConfig
from onee.config.ltf_config import LongTermForecastConfig
from onee.data.names import Aliases, STFCDResultsColumns, STFSRMResultsColumns, LTFCDResultsColumns, LTFSRMResultsColumns


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


def configure_stf_for_year(config_stf: ShortTermForecastConfig, year: int) -> None:
    """
    Configure STF config for a specific evaluation year.
    
    Args:
        config_stf: STF config to modify (in-place)
        year: The year to set for evaluation
    """
    config_stf.evaluation.eval_years_start = year - 3
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


# ============================================================================
# Utility Functions for Output Tables
# ============================================================================

def rename_to_stf_srm_results(df):
    """
    Rename DataFrame columns from Aliases to STF_SRM_Results column names.
    
    Args:
        df: DataFrame with columns using Aliases naming convention
        
    Returns:
        DataFrame with columns renamed to STF_SRM_Results format
    """
    rename_map = {
        Aliases.REGION: STFSRMResultsColumns.REGION,
        Aliases.ACTIVITE: STFSRMResultsColumns.ACTIVITY,
        Aliases.ANNEE: STFSRMResultsColumns.YEAR,
        Aliases.MOIS: STFSRMResultsColumns.MONTH,
        Aliases.CONSOMMATION_KWH: STFSRMResultsColumns.CONSOMMATION,
    }
    return df.rename(columns=rename_map)


def rename_to_stf_cd_results(df):
    """
    Rename DataFrame columns from Aliases to STF_CD_Results column names.
    
    Args:
        df: DataFrame with columns using Aliases naming convention
        
    Returns:
        DataFrame with columns renamed to STF_CD_Results format
    """
    rename_map = {
        Aliases.REGION: STFCDResultsColumns.REGION,
        Aliases.PARTENAIRE: STFCDResultsColumns.PARTENAIRE,
        Aliases.CONTRAT: STFCDResultsColumns.CONTRAT,
        Aliases.ACTIVITE: STFCDResultsColumns.ACTIVITY,
        Aliases.ANNEE: STFCDResultsColumns.YEAR,
        Aliases.MOIS: STFCDResultsColumns.MONTH,
        Aliases.CONSOMMATION_KWH: STFCDResultsColumns.CONSOMMATION_TOTAL,
        Aliases.CONSOMMATION_ZCONHC: STFCDResultsColumns.CONSOMMATION_ZCONHC,
        Aliases.CONSOMMATION_ZCONHL: STFCDResultsColumns.CONSOMMATION_ZCONHL,
        Aliases.CONSOMMATION_ZCONHP: STFCDResultsColumns.CONSOMMATION_ZCONHP,
        Aliases.PUISSANCE_FACTUREE: STFCDResultsColumns.PUISSANCE_FACTUREE,
        Aliases.NIVEAU_TENSION: STFCDResultsColumns.NIVEAU_TENSION,
    }
    return df.rename(columns=rename_map)


def rename_to_ltf_srm_results(df):
    """
    Rename DataFrame columns from Aliases to LTF_SRM_Results column names.
    
    Args:
        df: DataFrame with columns using Aliases naming convention
        
    Returns:
        DataFrame with columns renamed to LTF_SRM_Results format
    """
    rename_map = {
        Aliases.REGION: LTFSRMResultsColumns.REGION,
        Aliases.ANNEE: LTFSRMResultsColumns.YEAR,
        Aliases.CONSOMMATION_KWH: LTFSRMResultsColumns.CONSOMMATION,
    }
    return df.rename(columns=rename_map)


def rename_to_ltf_cd_results(df):
    """
    Rename DataFrame columns from Aliases to LTF_CD_Results column names.
    
    Args:
        df: DataFrame with columns using Aliases naming convention
        
    Returns:
        DataFrame with columns renamed to LTF_CD_Results format
    """
    rename_map = {
        Aliases.REGION: LTFCDResultsColumns.REGION,
        Aliases.PARTENAIRE: LTFCDResultsColumns.PARTENAIRE,
        Aliases.CONTRAT: LTFCDResultsColumns.CONTRAT,
        Aliases.ACTIVITE: LTFCDResultsColumns.ACTIVITY,
        Aliases.ANNEE: LTFCDResultsColumns.YEAR,
        Aliases.CONSOMMATION_KWH: LTFCDResultsColumns.CONSOMMATION_TOTAL,
        Aliases.CONSOMMATION_ZCONHC: LTFCDResultsColumns.CONSOMMATION_ZCONHC,
        Aliases.CONSOMMATION_ZCONHL: LTFCDResultsColumns.CONSOMMATION_ZCONHL,
        Aliases.CONSOMMATION_ZCONHP: LTFCDResultsColumns.CONSOMMATION_ZCONHP,
        Aliases.PUISSANCE_FACTUREE: LTFCDResultsColumns.PUISSANCE_FACTUREE,
        Aliases.NIVEAU_TENSION: LTFCDResultsColumns.NIVEAU_TENSION,
    }
    return df.rename(columns=rename_map)


def apply_consumption_adjustment(prediction_consomation_initiale, consommation_realise):
    """
    Computes an adjustment factor based on the ratio between realised and initial predicted consumption
    for the first n elements, then generates a corrected prediction list of length 12.

    Parameters:
        prediction_consomation_initiale (list of float): List of 12 predicted values.
        consommation_realise (list of float): List of n realised values (n <= 12).

    Returns:
        correction_prediction (list of float): New list of 12 corrected predictions.
    """

    # Ensure valid input lengths
    if len(prediction_consomation_initiale) != 12:
        raise ValueError("prediction_consomation_initiale must contain exactly 12 elements.")

    n = len(consommation_realise)
    if n > 12:
        raise ValueError("consommation_realise cannot contain more than 12 elements.")

    # Compute average ratio (adjustment factor)
    ratios = [
        consommation_realise[i] / prediction_consomation_initiale[i]
        for i in range(n)
        if prediction_consomation_initiale[i] != 0  # Avoid division by zero
    ]
    adjustment_factor = sum(ratios) / len(ratios) if len(ratios) > 0 else 1.0  # fallback if no valid ratios

    # Build corrected prediction list
    correction_prediction = []

    # First n elements = realised consumption
    correction_prediction.extend(consommation_realise)

    # Remaining elements = predicted * adjustment factor
    for i in range(n, 12):
        corrected_value = prediction_consomation_initiale[i] * adjustment_factor
        correction_prediction.append(corrected_value)

    return correction_prediction
