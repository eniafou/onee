# ─────────────────────────────────────────────────────────────
# run_horizon_forecast_srm.py (Clean + Correct)
# ─────────────────────────────────────────────────────────────

from short_term_forecast_strategies import (
    create_monthly_matrix,
)
from long_term_forecast_strategies import run_long_horizon_forecast
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

def create_summary_dataframe(all_results):
    """
    Create a summary DataFrame from forecast results
    
    Args:
        all_results: List of forecast result dictionaries
        
    Returns:
        pandas DataFrame with flattened summary records
    """
    df_summary_records = []
    for r in all_results:
        for y, v, actual, percent_error in zip(
            r["forecast_years"], r["pred_annual"], r["actuals"], r["percent_errors"]
        ):
            # Flatten run parameters for the summary sheet
            rp = r.get("run_parameters", {}) or {}
            growth = rp.get("growth_model", {}) or {}
            feature_config = rp.get("feature_config", {}) or {}

            df_summary_records.append(
                {
                    "Region": r["region"],
                    "Train_Start": r.get("train_start"),
                    "Train_End": r.get("train_end"),
                    "Level": r["level"],
                    "Year": y,
                    "Predicted_Annual": v,
                    "Actual_Annual": actual,
                    "Percent_Error": percent_error,
                    **feature_config,
                    **growth
                }
            )

    return pd.DataFrame(df_summary_records)


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

def run_ltf_srm_forecast(config_path="configs/ltf_srm.yaml", use_output_dir=False):
    """
    Execute LTF SRM forecast and return results
    
    Args:
        config_path: Path to YAML configuration file
        use_output_dir: Boolean to control output directory usage. If True, creates and uses output_dir. If False, passes None.
        
    Returns:
        dict with:
            - status: 'success' or 'error'
            - results: Dict mapping region names to list of forecast results
            - error: Error message if status is 'error'
    """
    try:
        # Load configuration from YAML
        config = LongTermForecastConfig.from_yaml(config_path)
        
        # Extract config values
        REGIONS = config.data.regions
        RUN_LEVELS = set(config.data.run_levels)
        forecast_runs = config.temporal.forecast_runs
        
        # Get model registry
        MODEL_REGISTRY = config.get_model_registry()
        
        # Initialize DataLoader
        data_loader = DataLoader(config.project.project_root)
        
        all_results_by_region = {}

        for TARGET_REGION in REGIONS:
            print(f"\n{'='*80}\n🌍 REGION: {TARGET_REGION}\n{'='*80}")

            # Create output_dir if use_output_dir is True, otherwise None
            if use_output_dir:
                output_dir = config.get_output_dir(TARGET_REGION)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = None

            # Load data using DataLoader
            df_regional, df_features, df_dist, var_cols = data_loader.load_srm_data(
                db_path=config.project.project_root / config.data.db_path,
                variable=config.data.target_variable,
                target_region=TARGET_REGION,
            )
            
            reg_var_col = var_cols["regional"]
            dist_var_col = var_cols["distributor"]

            # Lookup of previously computed monthly client predictions
            client_predictions_lookup = data_loader.load_client_prediction_lookup(
                TARGET_REGION
            )
            all_results = []

            activities = sorted(df_regional[Aliases.ACTIVITE].unique())

            for train_start, train_end in forecast_runs:
                print(
                    f"\n🧭 Training {train_start}→{train_end}, horizon={config.temporal.horizon}"
                )

                # ────────────────────────────────────────────────────────────────
                # LEVEL 1: SRM
                # ────────────────────────────────────────────────────────────────
                if 1 in RUN_LEVELS:        
                    if len(df_dist) >= 1:                    
                        print(
                            f"\n{'#'*60}\nLEVEL 7: SRM (Regional + Distributors)\n{'#'*60}"
                        )
                        DataLoader.aggregate_predictions(
                            client_predictions_lookup,
                            "SRM_Regional_Plus_Dist",
                            ["Total_Regional", "All_Distributors"],
                        )
                        df_all_dist = (
                            df_dist.groupby([Aliases.ANNEE, Aliases.MOIS])
                            .agg({dist_var_col: "sum"})
                            .reset_index()
                            .rename(columns={dist_var_col: config.data.target_variable})
                        )
                        df_total_regional = (
                            df_regional.groupby([Aliases.ANNEE, Aliases.MOIS])
                            .agg({reg_var_col: "sum"})
                            .reset_index()
                            .rename(columns={reg_var_col: config.data.target_variable})
                        )
                        df_srm = (
                            pd.concat([df_total_regional, df_all_dist], ignore_index=True)
                            .groupby([Aliases.ANNEE, Aliases.MOIS])
                            .agg({config.data.target_variable: "sum"})
                            .reset_index()
                        )

                        # if config.data.impute_2020:
                        #     df_srm = fill_2020_with_avg(df_srm, config.data.target_variable)

                        df_srm = df_srm[df_srm[Aliases.ANNEE] >= 2013]
                        df_train = df_srm[df_srm[Aliases.ANNEE] <= train_end]
                        monthly_matrix = create_monthly_matrix(
                            df_train, value_col=config.data.target_variable
                        )
                        years = np.sort(df_train[Aliases.ANNEE].unique())

                        res = run_long_horizon_forecast(
                            monthly_matrix=monthly_matrix,
                            years=years,
                            df_features=df_features,
                            df_monthly=df_srm,
                            config=config,
                            MODEL_REGISTRY=MODEL_REGISTRY,
                            monthly_clients_lookup=client_predictions_lookup.get(
                                "SRM_Regional_Plus_Dist", {}
                            ),
                            region_entity = f"{TARGET_REGION} - SRM (Regional + Distributors)",
                            save_folder=output_dir if output_dir else None
                        )

                        forecast_years = [y for y, _ in res["horizon_predictions"]]
                        pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                        actuals = []
                        percent_errors = []
                        for y, v in zip(forecast_years, pred_annual):
                            actual = None
                            percent_error = None
                            mask = df_srm[Aliases.ANNEE] == y
                            if not df_srm.loc[mask, config.data.target_variable].empty:
                                actual = df_srm.loc[mask, config.data.target_variable].sum()
                                percent_error = (
                                    (v - actual) / actual * 100 if actual != 0 else None
                                )

                            actuals.append(actual)
                            percent_errors.append(percent_error)

                        all_results.append(
                            {
                                "region": TARGET_REGION,
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
                    else:
                        print(f"\n{'#'*60}\nLEVEL 4: TOTAL REGIONAL\n{'#'*60}")
                        df_total_regional = (
                            df_regional.groupby([Aliases.ANNEE, Aliases.MOIS])
                            .agg({reg_var_col: "sum"})
                            .reset_index()
                            .rename(columns={reg_var_col: config.data.target_variable})
                        )
                        if config.data.impute_2020:
                            df_total_regional = fill_2020_with_avg(df_total_regional, reg_var_col)
                        DataLoader.aggregate_predictions(
                            client_predictions_lookup,
                            "Total_Regional",
                            [f"Activity_{a}" for a in activities],
                        )

                        df_train = df_total_regional[
                            (df_total_regional[Aliases.ANNEE] >= train_start) & (df_total_regional[Aliases.ANNEE] <= train_end)
                        ]
                        monthly_matrix = create_monthly_matrix(
                            df_train, value_col=config.data.target_variable
                        )
                        years = np.sort(df_train[Aliases.ANNEE].unique())

                        res = run_long_horizon_forecast(
                            monthly_matrix=monthly_matrix,
                            years=years,
                            df_features=df_features,
                            df_monthly=df_total_regional,
                            config=config,
                            MODEL_REGISTRY=MODEL_REGISTRY,
                            monthly_clients_lookup=client_predictions_lookup.get(
                                "Total_Regional", {}
                            ),
                            region_entity = f"{TARGET_REGION} - TOTAL REGIONAL",
                            save_folder=output_dir if output_dir else None,
                        )

                        forecast_years = [y for y, _ in res["horizon_predictions"]]
                        pred_annual = [float(v) for _, v in res["horizon_predictions"]]

                        actuals = []
                        percent_errors = []
                        for y, v in zip(forecast_years, pred_annual):
                            actual = None
                            percent_error = None
                            mask = df_total_regional[Aliases.ANNEE] == y
                            if not df_total_regional.loc[mask, reg_var_col].empty:
                                actual = df_total_regional.loc[mask, reg_var_col].sum()
                                percent_error = (
                                    (v - actual) / actual * 100 if actual != 0 else None
                                )
                            actuals.append(actual)
                            percent_errors.append(percent_error)

                        all_results.append(
                            {
                                "region": TARGET_REGION,
                                "train_start": train_start,
                                "train_end": train_end,
                                "level": "Total_Regional",
                                "forecast_years": forecast_years,
                                "pred_annual": pred_annual,
                                "pred_monthly": res["monthly_forecasts"],
                                "run_parameters": res.get("run_parameters", {}),
                                "actuals": actuals,
                                "percent_errors": percent_errors,
                            }
                        )
                
            # Store results for this region
            all_results_by_region[TARGET_REGION] = all_results

        print(f"\n✅ LTF SRM Forecast completed: {len(all_results_by_region)} regions")
        
        return {
            'status': 'success',
            'results': all_results_by_region
        }
        
    except Exception as e:
        print(f"\n❌ LTF SRM Forecast failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'results': {},
            'error': str(e)
        }


if __name__ == "__main__":
    # Load configuration from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ltf_srm.yaml"
    
    # Reload config to create output directories
    config = LongTermForecastConfig.from_yaml(config_path)
    REGIONS = config.data.regions
    
    # Create output directories for each region
    output_dirs = {}
    for TARGET_REGION in REGIONS:
        output_dir = config.get_output_dir(TARGET_REGION)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[TARGET_REGION] = output_dir
    
    # Run the forecast with use_output_dir=True
    result = run_ltf_srm_forecast(config_path=config_path, use_output_dir=True)
    
    # If successful, save outputs to disk
    if result['status'] == 'success':
        all_results_by_region = result['results']
        
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
