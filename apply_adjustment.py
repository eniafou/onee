from adjustment import weather_adjustment, ev_adjustment
import pandas as pd

from onee.data.names import Aliases


def apply_corrections_ltf(
    df_predictions: pd.DataFrame, 
    corrections: pd.DataFrame, 
    correction_col_name: str = "correction"
) -> pd.DataFrame:
    """
    Apply corrections to long-term forecast predictions.
    
    Simply merges the corrections on region and year, then adds the correction
    to the consumption value.
    
    Parameters:
    -----------
    df_predictions : DataFrame
        LTF predictions with columns: region, annee, consommation_kwh
    corrections : DataFrame
        Corrections with columns: region, annee, correction
    correction_col_name : str
        Name for the correction column in the output (for tracking multiple corrections)
        
    Returns:
    --------
    DataFrame with original columns plus the correction column, and adjusted consommation_kwh
    """
    df_result = df_predictions.merge(
        corrections.rename(columns={'correction': correction_col_name}),
        on=[Aliases.REGION, Aliases.ANNEE],
        how='left'
    )
    
    # Fill missing corrections with 0
    df_result[correction_col_name] = df_result[correction_col_name].fillna(0)
    
    return df_result


def apply_corrections_stf(
    df_predictions: pd.DataFrame, 
    corrections: pd.DataFrame, 
    correction_col_name: str = "correction"
) -> pd.DataFrame:
    """
    Apply corrections to short-term forecast predictions.
    
    Distributes yearly corrections across months using the predicted consumption
    as a distribution key (percentage of yearly consumption).
    
    Parameters:
    -----------
    df_predictions : DataFrame
        STF predictions with columns: region, activite, annee, mois, consommation_kwh
        Note: activite column should contain only one value per region/year group
    corrections : DataFrame
        Yearly corrections with columns: region, annee, correction
    correction_col_name : str
        Name for the correction column in the output (for tracking multiple corrections)
        
    Returns:
    --------
    DataFrame with original columns plus the correction column, and adjusted consommation_kwh
    """
    df_result = df_predictions.copy()
    
    # Calculate yearly total consumption per region for distribution weights
    yearly_totals = (
        df_result.groupby([Aliases.REGION, Aliases.ANNEE])[Aliases.CONSOMMATION_KWH]
        .transform('sum')
    )
    
    # Calculate monthly distribution weight (percentage of yearly consumption)
    df_result['_monthly_weight'] = df_result[Aliases.CONSOMMATION_KWH] / yearly_totals
    
    # Handle division by zero (if yearly total is 0, use equal distribution)
    df_result['_monthly_weight'] = df_result['_monthly_weight'].fillna(1/12)
    
    # Merge with corrections
    df_result = df_result.merge(
        corrections.rename(columns={'correction': '_yearly_correction'}),
        on=[Aliases.REGION, Aliases.ANNEE],
        how='left'
    )
    
    # Fill missing corrections with 0
    df_result['_yearly_correction'] = df_result['_yearly_correction'].fillna(0)
    
    # Distribute yearly correction across months using weights
    df_result[correction_col_name] = df_result['_yearly_correction'] * df_result['_monthly_weight']
    
    # Drop temporary columns
    df_result = df_result.drop(columns=['_monthly_weight', '_yearly_correction'])
    
    return df_result


def apply_adjustments():
    db_path = # 
    df_srm_ltf_predictions = # region, annee, consommation_kwh
    df_srm_stf_predictions = # region, activite, annee, mois, consommation_kwh

    df_weather_diff = # region, annee, temp_min_diff, precipitation_sum_diff, et0_diff
    df_ev = # region, annee, ev

    weather_corrections = weather_adjustment(df_weather_diff, db_path) # region, annee, correction
    ev_corrections = ev_adjustment(df_ev, df_srm_ltf_predictions) # region, annee, correction

    df_srm_ltf_predictions = apply_corrections_ltf(df_srm_ltf_predictions, weather_corrections, correction_col_name="weather_correction")
    df_srm_stf_predictions = apply_corrections_stf(df_srm_stf_predictions, weather_corrections, correction_col_name="weather_correction")

    df_srm_ltf_predictions = apply_corrections_ltf(df_srm_ltf_predictions, ev_corrections, correction_col_name="ev_correction")
    df_srm_stf_predictions = apply_corrections_stf(df_srm_stf_predictions, ev_corrections, correction_col_name="ev_correction")

    return df_srm_ltf_predictions, df_srm_stf_predictions
