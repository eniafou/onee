# ─────────────────────────────────────────────────────────────
# run_horizon_forecast_cd.py (Clean + Correct)
# ─────────────────────────────────────────────────────────────

import os
from short_term_forecast_strategies import (
    create_monthly_matrix,
)
from long_term_forecast_strategies import run_long_horizon_forecast, create_summary_dataframe
from onee.utils import fill_2020_with_avg, get_move_in_year, clean_name
from onee.config.ltf_config import LongTermForecastConfig
from onee.data.loader import DataLoader
from onee.data.names import Aliases


import numpy as np
import pandas as pd
import sqlite3
import pickle
from pathlib import Path
import warnings
import sys

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

def run_ltf_cd_forecast(config_path="configs/ltf_cd.yaml", output_dir=None):
    """
    Execute LTF CD forecast and return results
    
    Args:
        config_path: Path to YAML configuration file
        project_root: Path object for project root directory (optional)
        output_dir: Path to output directory (optional). If provided, passes to save_folder. If None, passes None.
        
    Returns:
        dict with:
            - status: 'success' or 'error'
            - results: List of forecast results
            - error: Error message if status is 'error'
    """
    try:
        # Load configuration from YAML
        config = LongTermForecastConfig.from_yaml(config_path)
        
        # Extract config values
        RUN_LEVELS = set(config.data.run_levels)
        forecast_runs = config.temporal.forecast_runs
        
        # Get model registry
        MODEL_REGISTRY = config.get_model_registry()
        
        # Initialize DataLoader
        data_loader = DataLoader(config.project.project_root)

        # Load data using DataLoader
        df_contrats, df_features = data_loader.load_cd_data(
            db_path=config.project.project_root / config.data.db_path,
        )

        all_results = []

        for train_start, train_end in forecast_runs:
            print(
                f"\n🧭 Training {train_start}→{train_end}, horizon={config.temporal.horizon}"
            )

            if 1 in RUN_LEVELS:
                print(f"\n{'#'*60}")
                print(f"LEVEL 1: REGION")
                print(f"{'#'*60}")

                regions = sorted(df_contrats[Aliases.REGION].unique())

                # --- Process regions ---
                for region in regions:
                    df_region = df_contrats[df_contrats[Aliases.REGION] == region].copy()
                    df_region = df_region.groupby([Aliases.ANNEE, Aliases.MOIS]).agg({
                        config.data.target_variable: 'sum',
                        Aliases.PUISSANCE_FACTUREE: 'sum'
                    }).reset_index()
                    
                    df_train = df_region[
                        (df_region[Aliases.ANNEE] >= train_start) & (df_region[Aliases.ANNEE] <= train_end)
                    ]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.data.target_variable
                    )
                    years = np.sort(df_train[Aliases.ANNEE].unique())

                    df_activite_features = data_loader.compute_contract_features(df_contrats, entity_col=Aliases.REGION, end_year=train_end + config.temporal.horizon)
                    df_r = df_activite_features[df_activite_features[Aliases.REGION] == region]
                    df_r = df_r.merge(df_features, on=Aliases.ANNEE)
                     
                    res = run_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_r,
                        df_monthly=df_region,
                        config=config,
                        MODEL_REGISTRY=MODEL_REGISTRY,
                        region_entity=f"CD - région {region}",
                        save_folder=output_dir / f"region_{region}" if output_dir else None,
                    )
                    
                    forecast_years = [y for y, _ in res["horizon_predictions"]]
                    pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                    actuals = []
                    percent_errors = []
                    for y, v in zip(forecast_years, pred_annual):
                        actual = None
                        percent_error = None
                        mask = df_region[Aliases.ANNEE] == y
                        if not df_region.loc[mask, config.data.target_variable].empty:
                            actual = df_region.loc[mask, config.data.target_variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "train_start": train_start,
                            "train_end": train_end,
                            "level": f"Région_{region}",
                            "forecast_years": forecast_years,
                            "pred_annual": pred_annual,
                            "pred_monthly": res["monthly_forecasts"],
                            "run_parameters": res.get("run_parameters", {}),
                            "actuals": actuals,
                            "percent_errors": percent_errors,
                        }
                    )

            if 2 in RUN_LEVELS:
                print(f"\n{'#'*60}")
                print(f"LEVEL 3: ACTIVITE")
                print(f"{'#'*60}")

                activites = sorted(df_contrats[Aliases.ACTIVITE].unique())



                # --- Process established activities ---
                for activite in activites:
                    df_activite = df_contrats[df_contrats[Aliases.ACTIVITE] == activite].copy()
                    df_activite = df_activite.groupby([Aliases.ANNEE, Aliases.MOIS]).agg({
                        config.data.target_variable: 'sum',
                        Aliases.PUISSANCE_FACTUREE: 'sum'
                    }).reset_index()
                    
                    df_train = df_activite[
                        (df_activite[Aliases.ANNEE] >= train_start) & (df_activite[Aliases.ANNEE] <= train_end)
                    ]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.data.target_variable
                    )
                    years = np.sort(df_train[Aliases.ANNEE].unique())

                    df_activite_features = data_loader.compute_contract_features(df_contrats, entity_col=Aliases.ACTIVITE, end_year=train_end + config.temporal.horizon)
                    df_a = df_activite_features[df_activite_features[Aliases.ACTIVITE] == activite]
                    df_a = df_a.merge(df_features, on=Aliases.ANNEE)
     
                    res = run_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_a,
                        df_monthly=df_activite,
                        config=config,
                        MODEL_REGISTRY=MODEL_REGISTRY,
                        region_entity = f"CD - activité {activite}",
                        save_folder=output_dir / f"{activite}" if output_dir else None,
                    )
                    
                    forecast_years = [y for y, _ in res["horizon_predictions"]]
                    pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                    actuals = []
                    percent_errors = []
                    for y, v in zip(forecast_years, pred_annual):
                        actual = None
                        percent_error = None
                        mask = df_activite[Aliases.ANNEE] == y
                        if not df_activite.loc[mask, config.data.target_variable].empty:
                            actual = df_activite.loc[mask, config.data.target_variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "train_start": train_start,
                            "train_end": train_end,
                            "level": f"Activité_{activite}",
                            "forecast_years": forecast_years,
                            "pred_annual": pred_annual,
                            "pred_monthly": res["monthly_forecasts"],
                            "run_parameters": res.get("run_parameters", {}),
                            "actuals": actuals,
                            "percent_errors": percent_errors,
                        }
                    )

        print(f"\n✅ LTF CD Forecast completed: {len(all_results)} results")
        
        return {
            'status': 'success',
            'results': all_results
        }
        
    except Exception as e:
        print(f"\n❌ LTF CD Forecast failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'results': [],
            'error': str(e)
        }


if __name__ == "__main__":
    # Load configuration from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ltf_cd.yaml"
    
    # Reload config to create output directory
    config = LongTermForecastConfig.from_yaml(config_path)
    
    # Create output directory
    output_dir = (
        config.project.project_root
        / config.project.output_base_dir
        / f"{config.project.exp_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the forecast with output_dir
    result = run_ltf_cd_forecast(config_path=config_path, output_dir=output_dir)
    
    # If successful, save outputs to disk
    if result['status'] == 'success':
        all_results = result['results']
        
        # Save pickle file
        with open(
            output_dir / f"{config.data.target_variable}_{config.project.exp_name}.pkl",
            "wb",
        ) as f:
            pickle.dump(all_results, f)
        
        # Create and save Excel summary
        df_summary = create_summary_dataframe(all_results)
        out_xlsx = (
            output_dir / f"summary_{config.data.target_variable}_{config.project.exp_name}.xlsx"
        )
        df_summary.to_excel(out_xlsx, index=False)
        print(f"\n📁 Saved horizon forecasts to {out_xlsx}")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)
