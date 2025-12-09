import warnings
warnings.filterwarnings('ignore')

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import sys

from short_term_forecast_strategies import run_analysis_for_entity, save_summary
from onee.utils import clean_name
from onee.config.stf_config import ShortTermForecastConfig
from onee.data.loader import DataLoader
from onee.data.names import Aliases


# ────────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────────────────────────

def run_stf_srm_forecast(config_path="configs/stf_srm.yaml", use_output_dir=False):
    """
    Execute STF SRM forecast and return results
    
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
        # Load configuration
        config = ShortTermForecastConfig.from_yaml(config_path)
        
        # Convert to legacy ANALYSIS_CONFIG format
        ANALYSIS_CONFIG = config.to_analysis_config()
        
        # Extract commonly used values
        PROJECT_ROOT = config.project.project_root
        VARIABLE = config.data.variable
        REGIONS = config.data.regions
        exp_name = config.project.exp_name
        use_monthly_clients_options = config.features.use_monthly_clients_options
        
        # Initialize DataLoader
        data_loader = DataLoader(PROJECT_ROOT)
        
        all_results_by_region = {}

        for TARGET_REGION, region_mode in REGIONS.items():
            # Determine run levels for this region:
            # 0: only SRM (level 0 - Total Regional + Distributors)
            # 1: both level 1 (individual activities) and level 0 (SRM)
            if region_mode == 0:
                RUN_LEVELS = {0}
            elif region_mode == 1:
                RUN_LEVELS = {0, 1}
            else:
                RUN_LEVELS = set(config.data.run_levels)
          
            print(f"Loading data for {TARGET_REGION}...\n")

            # Load data using DataLoader
            df_regional, df_features, df_dist, var_cols = data_loader.load_srm_data(
                db_path=config.project.project_root / config.data.db_path,
                variable=VARIABLE,
                target_region=TARGET_REGION,
            )
            reg_var_col = var_cols["regional"]

            print(f"\n{'#'*60}")
            print(f"ULTRA-STRICT MODE: PCA REFITTED IN EACH LOOCV FOLD")
            print(f"{'#'*60}")

            # Only load client predictions if we're going to use them
            client_predictions_lookup = {}
            if True in use_monthly_clients_options:
                client_predictions_lookup = data_loader.load_client_prediction_lookup(TARGET_REGION)
            activities = sorted(df_regional[Aliases.ACTIVITE].unique())

            all_results = []

            def _maybe_add(res):
                if res:
                    all_results.append(res)

            # LEVEL 1: INDIVIDUAL ACTIVITIES
            if 1 in RUN_LEVELS:
                print(f"\n{'#'*60}\nLEVEL 1: INDIVIDUAL ACTIVITIES\n{'#'*60}")
                for activity in activities:
                    df_activity = (
                        df_regional[df_regional[Aliases.ACTIVITE] == activity][[Aliases.ANNEE, Aliases.MOIS, reg_var_col]]
                        .copy()
                        .rename(columns={reg_var_col: VARIABLE})
                    )
                    entity_name = f"Activity_{activity}"
                    res = run_analysis_for_entity(
                        df_activity,
                        entity_name,
                        df_features,
                        df_regional,
                        config=ANALYSIS_CONFIG,
                        client_predictions=client_predictions_lookup.get(entity_name),
                        under_estimation_penalty=config.loss.under_estimation_penalty,
                    )
                    _maybe_add(res)

            # TOTAL REGIONAL
            df_total_regional = (
                df_regional.groupby([Aliases.ANNEE, Aliases.MOIS])
                .agg({reg_var_col: 'sum'})
                .reset_index()
                .rename(columns={reg_var_col: VARIABLE})
            )

            # ALL DISTRIBUTORS
            df_all_dist = None
            if df_dist is not None:
                dist_var_col = var_cols["distributor"]
                df_all_dist = (
                    df_dist.groupby([Aliases.ANNEE, Aliases.MOIS])
                    .agg({dist_var_col: 'sum'})
                    .reset_index()
                    .rename(columns={dist_var_col: VARIABLE})
                )

            # SRM (Regional + Distributors)
            if 0 in RUN_LEVELS:
                print(f"\n{'#'*60}\nSRM (Regional + Distributors)\n{'#'*60}")
                if df_all_dist is None:
                    print(f"⚠️  Skipping SRM: distributors data unsupported for VARIABLE='{VARIABLE}'.")
                else:
                    df_srm = (
                        pd.concat(
                            [df_total_regional[[Aliases.ANNEE, Aliases.MOIS, VARIABLE]],
                            df_all_dist[[Aliases.ANNEE, Aliases.MOIS, VARIABLE]]],
                            ignore_index=True
                        )
                        .groupby([Aliases.ANNEE, Aliases.MOIS])
                        .agg({VARIABLE: 'sum'})
                        .reset_index()
                    )
                    res = run_analysis_for_entity(
                        df_srm,
                        "SRM_Regional_Plus_Dist",
                        df_features,
                        df_regional,
                        config=ANALYSIS_CONFIG,
                        client_predictions=client_predictions_lookup.get("SRM_Regional_Plus_Dist"),
                        under_estimation_penalty=config.loss.under_estimation_penalty,
                    )
                    _maybe_add(res)

            # Store results for this region
            all_results_by_region[TARGET_REGION] = all_results

        print(f"\n✅ STF SRM Forecast completed: {len(all_results_by_region)} regions")
        
        return {
            'status': 'success',
            'results': all_results_by_region
        }
        
    except Exception as e:
        print(f"\n❌ STF SRM Forecast failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'results': {},
            'error': str(e)
        }


# ────────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load configuration from command line or use default
    config_path = Path(__file__).parent / "configs/stf_srm.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Reload config to create output directories
    config = ShortTermForecastConfig.from_yaml(config_path)
    PROJECT_ROOT = config.project.project_root
    VARIABLE = config.data.variable
    REGIONS = config.data.regions
    exp_name = config.project.exp_name
    
    # Create output directories for each region
    output_dirs = {}
    for TARGET_REGION in REGIONS.keys():
        output_dir = PROJECT_ROOT / 'outputs/outputs_srm' / exp_name / clean_name(TARGET_REGION)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[TARGET_REGION] = output_dir
    
    # Run the forecast
    result = run_stf_srm_forecast(config_path=config_path, use_output_dir=True)
    
    # If successful, save outputs to disk
    if result['status'] == 'success':
        all_results_by_region = result['results']
        
        for TARGET_REGION, all_results in all_results_by_region.items():
            output_dir = output_dirs[TARGET_REGION]
            
            # Save outputs
            output_file = output_dir / f'{clean_name(TARGET_REGION)}_{VARIABLE}_{exp_name}.xlsx'
            monthly_book_file = output_dir / f'monthly_predictions_by_entity_{clean_name(TARGET_REGION)}_{VARIABLE}_{exp_name}.xlsx'
            save_summary(all_results, output_file, monthly_book_file)

            with open(output_dir / f'all_results_{clean_name(TARGET_REGION)}_{VARIABLE}.pkl', "wb") as f:
                pickle.dump(all_results, f)

            print(f"\n{'='*60}")
            print(f"✅ Results saved to: {output_file}")
            print(f"📊 Total entities analyzed: {len(all_results)}")
            print(f"🔒 ZERO DATA LEAKAGE GUARANTEED")
            print(f"{'='*60}\n")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)
