from pathlib import Path
from typing import Dict, Any

import pandas as pd

from onee.data.loader import DataLoader
from onee.data.names import Aliases


# Predefined weather sensitivity coefficients by region
WEATHER_COEFS: Dict[str, Dict[str, float]] = {'Béni Mellal-Khénifra': {'implicit_intercept': -761556999.0194256,
  'coef_unscaled_apparent_temperature_min': 150945.52494459832,
  'coef_unscaled_precipitation_sum': 695965.7652271345,
  'coef_unscaled_et0_fao_evapotranspiration_sum': 312589.59634202055},
 'Casablanca-Settat': {'implicit_intercept': -843349325.577105,
  'coef_unscaled_apparent_temperature_min': 56799106.10845571,
  'coef_unscaled_precipitation_sum': 580606.2664738528,
  'coef_unscaled_et0_fao_evapotranspiration_sum': -135632.51806734095},
 'Drâa-Tafilalet': {'implicit_intercept': -869168753.9888884,
  'coef_unscaled_apparent_temperature_min': -51561747.956962235,
  'coef_unscaled_precipitation_sum': -28065.201556676668,
  'coef_unscaled_et0_fao_evapotranspiration_sum': 703244.7969628241},
 'Fès-Meknès': {'implicit_intercept': -266981324.65040958,
  'coef_unscaled_apparent_temperature_min': -48271934.772982344,
  'coef_unscaled_precipitation_sum': 254007.85221659957,
  'coef_unscaled_et0_fao_evapotranspiration_sum': 427861.7030791526},
 'Laâyoune-Sakia El Hamra': {'implicit_intercept': -452163189.50732774,
  'coef_unscaled_apparent_temperature_min': 23777614.312995747,
  'coef_unscaled_precipitation_sum': 1424568.6105868893,
  'coef_unscaled_et0_fao_evapotranspiration_sum': 1225.4153371571465},
 'Marrakech-Safi': {'implicit_intercept': -859034964.2931361,
  'coef_unscaled_apparent_temperature_min': 23843194.044664875,
  'coef_unscaled_precipitation_sum': 151813.4221894953,
  'coef_unscaled_et0_fao_evapotranspiration_sum': 300419.78504665446},
 'Oriental': {'implicit_intercept': -965823292.7294213,
  'coef_unscaled_apparent_temperature_min': 92906001.74314561,
  'coef_unscaled_precipitation_sum': 234975.31276205444,
  'coef_unscaled_et0_fao_evapotranspiration_sum': -209013.88780028952},
 'Rabat-Salé-Kénitra': {'implicit_intercept': -198068003.8375749,
  'coef_unscaled_apparent_temperature_min': 3990694.8839878566,
  'coef_unscaled_precipitation_sum': 4478.622242531856,
  'coef_unscaled_et0_fao_evapotranspiration_sum': 77090.86797711515},
 'Souss-Massa': {'implicit_intercept': -2356575808.4518914,
  'coef_unscaled_apparent_temperature_min': 120356403.23536505,
  'coef_unscaled_precipitation_sum': 710014.7274596494,
  'coef_unscaled_et0_fao_evapotranspiration_sum': 142635.66650035518},
 'Tanger-Tétouan-Al Hoceïma': {'implicit_intercept': -926807170.2146193,
  'coef_unscaled_apparent_temperature_min': 8629097.307845984,
  'coef_unscaled_precipitation_sum': 125543.96345687777,
  'coef_unscaled_et0_fao_evapotranspiration_sum': 556269.2226169758}}


def weather_adjustment(df_weather_diff: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    """
    Calculate weather-based corrections per region and year.
    
    For each region and year, the correction is calculated as:
    correction = sum((latest_value + diff) * coef) + implicit_intercept
    
    Parameters:
    -----------
    df_weather_diff : DataFrame
        Weather differences by region and year.
        Required columns (using Aliases): 
        - region, annee, temp_min_diff, precipitation_sum_diff, et0_diff
    db_path : Path
        Path to database containing the Weather table for loading latest values.
        
    Returns:
    --------
    DataFrame with columns: region, annee, correction
    """
    # Load latest weather values from database
    loader = DataLoader(project_root=db_path.parent)
    df_weather_latests = loader.load_weather_latest(db_path)
    
    # Merge weather diff with latest values on region
    df_merged = df_weather_diff.merge(
        df_weather_latests,
        on=Aliases.REGION,
        how='left'
    )
    
    # Calculate correction for each row
    corrections = []
    for _, row in df_merged.iterrows():
        region = row[Aliases.REGION]
        
        if region not in WEATHER_COEFS:
            corrections.append(0.0)
            continue
        
        coefs = WEATHER_COEFS[region]
        
        # Calculate: sum((latest + diff) * coef) + intercept
        temp_min_contribution = (
            (row[Aliases.TEMP_MIN_LATEST] + row[Aliases.TEMP_MIN_DIFF]) *
            coefs['coef_unscaled_apparent_temperature_min']
        )
        precipitation_contribution = (
            (row[Aliases.PRECIPITATION_SUM_LATEST] + row[Aliases.PRECIPITATION_SUM_DIFF]) *
            coefs['coef_unscaled_precipitation_sum']
        )
        et0_contribution = (
            (row[Aliases.ET0_LATEST] + row[Aliases.ET0_DIFF]) *
            coefs['coef_unscaled_et0_fao_evapotranspiration_sum']
        )
        
        correction = (
            temp_min_contribution +
            precipitation_contribution +
            et0_contribution +
            coefs['implicit_intercept']
        )
        corrections.append(correction)
    
    df_merged['correction'] = corrections
    
    # Select output columns
    df_output = df_merged[[Aliases.REGION, Aliases.ANNEE, 'correction']].copy()
    
    return df_output


if __name__ == "__main__":
    # Test with mock data
    from pathlib import Path
    
    # Create mock df_weather_diff with sample regions and years
    df_weather_diff = pd.DataFrame({
        Aliases.REGION: [
            'Casablanca-Settat', 'Casablanca-Settat', 'Casablanca-Settat',
            'Rabat-Salé-Kénitra', 'Rabat-Salé-Kénitra', 'Rabat-Salé-Kénitra',
            'Marrakech-Safi', 'Marrakech-Safi', 'Marrakech-Safi',
        ],
        Aliases.ANNEE: [2024, 2025, 2026, 2024, 2025, 2026, 2024, 2025, 2026],
        Aliases.TEMP_MIN_DIFF: [0.5, 1.0, 1.5, 0.3, 0.8, 1.2, 0.7, 1.1, 1.6],
        Aliases.PRECIPITATION_SUM_DIFF: [-10, -20, -30, -5, -15, -25, -8, -18, -28],
        Aliases.ET0_DIFF: [5, 10, 15, 3, 8, 12, 6, 11, 16],
    })
    
    # Path to the database
    db_path = Path("data/all_data.db")
    
    print("Mock df_weather_diff:")
    print(df_weather_diff)
    print("\n" + "="*50 + "\n")
    
    # Run the adjustment
    df_result = weather_adjustment(df_weather_diff, db_path)
    
    print("Weather adjustment results:")
    print(df_result)