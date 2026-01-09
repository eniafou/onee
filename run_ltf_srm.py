# ─────────────────────────────────────────────────────────────
# run_horizon_forecast_srm.py (Clean + Correct)
# ─────────────────────────────────────────────────────────────

from onee.short_term_forecast_strategies import (
    create_monthly_matrix,
)
from onee.long_term_forecast_strategies import run_long_horizon_forecast, create_summary_dataframe
from onee.utils import fill_2020_with_avg, clean_name
from onee.config.ltf_config import LongTermForecastConfig
from onee.data.loader import DataLoader
from onee.data.names import Aliases, GRDValues

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
import sys

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def prepare_prediction_output(region, results_list):
    """
    Create a simple summary DataFrame with Region, Year, and Predicted_Annual for a single region.
    
    Args:
        region: Region name
        results_list: List of forecast result dictionaries for this region
        
    Returns:
        pandas DataFrame with Region, Year, and Consommation columns
    """
    df_summary_records = []
    
    if not results_list:
        return pd.DataFrame(columns=[Aliases.REGION, Aliases.ANNEE, Aliases.CONSOMMATION_KWH])
    
    # Build historical consumption lookup for safety net (from actuals in results)
    # Used to replace negative predictions with last year's actual value
    historical_consumption = {}
    for r in results_list:
        actuals = r.get("actuals", [])
        forecast_years = r.get("forecast_years", [])
        region_name = r.get("region", region)
        for y, actual in zip(forecast_years, actuals):
            if actual is not None:
                historical_consumption[(region_name, y)] = actual
    
    for r in results_list:
        for y, v in zip(r["forecast_years"], r["pred_annual"]):
            region_name = r.get("region", region)
            predicted = v
            
            # Safety net: if predicted consumption is negative, use last year's value
            if predicted < 0:
                last_year_key = (region_name, y - 1)
                fallback_value = historical_consumption.get(last_year_key)
                if fallback_value is not None and fallback_value >= 0:
                    predicted = fallback_value
                else:
                    # If no valid last year data, use 0 as last resort
                    predicted = 0
            
            df_summary_records.append(
                {
                    Aliases.REGION: region_name,
                    Aliases.ANNEE: y,
                    Aliases.CONSOMMATION_KWH: predicted
                }
            )
    
    return pd.DataFrame(df_summary_records)



# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
def run_ltf_srm_forecast(config, target_region, df_regional, df_features, df_srm, use_output_dir=False):
    """
    Execute LTF SRM forecast for a single region and return results
    
    Args:
        config: LongTermForecastConfig object (already configured)
        target_region: Name of the region to forecast
        df_regional: Regional DataFrame
        df_features: Features DataFrame
        df_srm: Pre-computed SRM DataFrame (Regional + Distributors combined), with columns [Annee, Mois, target_variable]
        use_output_dir: Boolean to control output directory usage. If True, creates and uses output_dir. If False, passes None.
        
    Returns:
        dict with:
            - status: 'success' or 'error'
            - results: List of forecast results for this region
            - error: Error message if status is 'error'
    """
    try:
        # Extract config values
        RUN_LEVELS = set(config.data.run_levels)
        forecast_runs = config.temporal.forecast_runs
        
        # Get model registry
        MODEL_REGISTRY = config.get_model_registry()
        
        # Initialize DataLoader for client predictions
        data_loader = DataLoader(config.project.project_root)

        print(f"\n{'='*80}\n🌍 REGION: {target_region}\n{'='*80}")

        # Create output_dir if use_output_dir is True, otherwise None
        if use_output_dir:
            output_dir = config.get_output_dir(target_region)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None

        # Lookup of previously computed monthly client predictions
        client_predictions_lookup = data_loader.load_client_prediction_lookup(
            target_region
        )
        all_results = []

        for train_start, train_end in forecast_runs:
            print(
                f"\n🧭 Training {train_start}→{train_end}, horizon={config.temporal.horizon}"
            )

            # ────────────────────────────────────────────────────────────────
            # LEVEL 1: SRM
            # ────────────────────────────────────────────────────────────────
            if 1 in RUN_LEVELS:
                print(
                    f"\n{'#'*60}\nLEVEL 1: SRM (Regional + Distributors)\n{'#'*60}"
                )
                DataLoader.aggregate_predictions(
                    client_predictions_lookup,
                    "SRM_Regional_Plus_Dist",
                    ["Total_Regional", "All_Distributors"],
                )
                
                df_srm_filtered = df_srm[df_srm[Aliases.ANNEE] >= train_start].copy()
                if config.data.impute_2020:
                    df_srm_filtered = fill_2020_with_avg(df_srm_filtered, config.data.target_variable)

                df_train = df_srm_filtered[df_srm_filtered[Aliases.ANNEE] <= train_end]
                monthly_matrix = create_monthly_matrix(
                    df_train, value_col=config.data.target_variable
                )
                years = np.sort(df_train[Aliases.ANNEE].unique())

                res = run_long_horizon_forecast(
                    monthly_matrix=monthly_matrix,
                    years=years,
                    df_features=df_features,
                    df_monthly=df_srm_filtered,
                    config=config,
                    MODEL_REGISTRY=MODEL_REGISTRY,
                    monthly_clients_lookup=client_predictions_lookup.get(
                        "SRM_Regional_Plus_Dist", {}
                    ),
                    region_entity = f"{target_region} - SRM (Regional + Distributors)",
                    save_folder=output_dir if output_dir else None
                )

                forecast_years = [y for y, _ in res["horizon_predictions"]]
                pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                actuals = []
                percent_errors = []
                for y, v in zip(forecast_years, pred_annual):
                    actual = None
                    percent_error = None
                    mask = df_srm_filtered[Aliases.ANNEE] == y
                    if not df_srm_filtered.loc[mask, config.data.target_variable].empty:
                        actual = df_srm_filtered.loc[mask, config.data.target_variable].sum()
                        percent_error = (
                            (v - actual) / actual * 100 if actual != 0 else None
                        )

                    actuals.append(actual)
                    percent_errors.append(percent_error)

                all_results.append(
                    {
                        "region": target_region,
                        "train_start": train_start,
                        "train_end": train_end,
                        "level": "SRM_Regional_Plus_Dist",
                        "forecast_years": forecast_years,
                        "pred_annual": pred_annual,
                        "pred_monthly": res["monthly_forecasts"],
                        "run_parameters": res.get("run_parameters", {}),
                        "actuals": actuals,
                        "percent_errors": percent_errors,
                    }
                )
        print(f"\n✅ LTF SRM Forecast completed for {target_region}")
        
        return {
            'status': 'success',
            'results': all_results
        }
        
    except Exception as e:
        print(f"\n❌ LTF SRM Forecast failed for {target_region}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'results': [],
            'error': str(e)
        }


if __name__ == "__main__":
    # Load configuration from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ltf_srm.yaml"
    
    # Load configuration
    config = LongTermForecastConfig.from_yaml(config_path)
    
    # Override temporal settings if needed (e.g., latest_year_in_data=2023, horizon=5)
    latest_year_in_data = 2023
    horizon = 5
    if latest_year_in_data is not None:
        config.temporal.forecast_runs[0] = (
            config.temporal.forecast_runs[0][0],
            latest_year_in_data
        )
        config.temporal.horizon = horizon if horizon is not None else config.temporal.horizon
    
    REGIONS = config.data.regions
    
    # Initialize DataLoader
    data_loader = DataLoader(config.project.project_root)
    
    # Create output directories for each region
    output_dirs = {}
    for TARGET_REGION in REGIONS:
        output_dir = config.get_output_dir(TARGET_REGION)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[TARGET_REGION] = output_dir
    
    # Run forecast for each region and collect results
    all_results_by_region = {}
    overall_status = 'success'
    
    for TARGET_REGION in REGIONS:
        print(f"Loading data for {TARGET_REGION}...")
        df_regional, df_features, df_dist, var_cols = data_loader.load_srm_data(
            db_path=config.project.project_root / config.data.db_path,
            variable=config.data.target_variable,
            target_region=TARGET_REGION,
        )
        
        # Compute df_srm (Regional + Distributors)
        reg_var_col = var_cols["regional"]
        df_total_regional = (
            df_regional.groupby([Aliases.ANNEE, Aliases.MOIS])
            .agg({reg_var_col: 'sum'})
            .reset_index()
            .rename(columns={reg_var_col: config.data.target_variable})
        )
        
        if df_dist is not None and len(df_dist) > 0:
            dist_var_col = var_cols["distributor"]
            df_all_dist = (
                df_dist.groupby([Aliases.ANNEE, Aliases.MOIS])
                .agg({dist_var_col: 'sum'})
                .reset_index()
                .rename(columns={dist_var_col: config.data.target_variable})
            )
            df_srm = (
                pd.concat(
                    [df_total_regional[[Aliases.ANNEE, Aliases.MOIS, config.data.target_variable]],
                     df_all_dist[[Aliases.ANNEE, Aliases.MOIS, config.data.target_variable]]],
                    ignore_index=True
                )
                .groupby([Aliases.ANNEE, Aliases.MOIS])
                .agg({config.data.target_variable: 'sum'})
                .reset_index()
            )
        else:
            df_srm = df_total_regional[[Aliases.ANNEE, Aliases.MOIS, config.data.target_variable]].copy()
        
        # Run forecast for this region
        result = run_ltf_srm_forecast(
            config=config,
            target_region=TARGET_REGION,
            df_regional=df_regional,
            df_features=df_features,
            df_srm=df_srm,
            use_output_dir=False
        )
        
        if result['status'] == 'success':
            all_results_by_region[TARGET_REGION] = result['results']
        else:
            overall_status = 'error'
            all_results_by_region[TARGET_REGION] = []
    
    
    # If successful, save outputs to disk
    for TARGET_REGION, all_results in all_results_by_region.items():
        output_dir = output_dirs[TARGET_REGION]
        
        # Save pickle file
        with open(
            output_dir / f"{clean_name(TARGET_REGION)}_{config.data.target_variable}_{config.project.exp_name}.pkl",
            "wb",
        ) as f:
            pickle.dump(all_results, f)
        
        # Create and save Excel summary
        df_summary = create_summary_dataframe(all_results)
        out_xlsx = (
            output_dir / f"summary_{clean_name(TARGET_REGION)}_{config.data.target_variable}_{config.project.exp_name}.xlsx"
        )
        df_summary.to_excel(out_xlsx, index=False)
        print(f"\n📁 Saved horizon forecasts to {out_xlsx}")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)
