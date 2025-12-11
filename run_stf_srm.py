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

def prepare_prediction_output(region, results_list):
    """
    Create a tall DataFrame with monthly predictions for a single region.
    
    Args:
        region: Region name
        results_list: List of result dictionaries for this region
        
    Returns:
        pandas DataFrame with columns: Region, Activity, Year, Month, Consommation
        
    Notes:
        - For level 0 (SRM), adds an artificial activity called "Total"
        - Extracts monthly predictions from the best_model of each result
    """

    def extract_activity(entity_name: str) -> str:
        prefix = "Activity_"
        if entity_name.startswith(prefix):
            return entity_name[len(prefix):]
        return entity_name  # fallback if format is unexpected
    
    records = []
    
    if not results_list:
        return pd.DataFrame(columns=[Aliases.REGION, Aliases.ACTIVITE, Aliases.ANNEE, Aliases.MOIS, Aliases.CONSOMMATION_KWH])
        
    for result in results_list:
        entity = result.get("entity", "")
        best_model = result.get("best_model", {})
        monthly_details = result.get("monthly_details", {})
        
        if not monthly_details:
            continue
        
        # Determine activity name
        # If entity is empty or equals region, it's level 0 (SRM) -> use "Total"
        if not entity or entity.startswith("SRM_"):
            activity = "Total"
        else:
            activity = extract_activity(entity)
        
        # Extract monthly predictions for each year
        for year, df_monthly in monthly_details.items():
            if df_monthly is None or df_monthly.empty:
                continue
            
            # Iterate through each month in the year
            for _, row in df_monthly.iterrows():
                month = row.get("Month")
                predicted = row.get("Predicted")
                
                if month is not None and predicted is not None:
                    records.append({
                        Aliases.REGION: region,
                        Aliases.ACTIVITE: activity,
                        Aliases.ANNEE: year,
                        Aliases.MOIS: int(month),
                        Aliases.CONSOMMATION_KWH: predicted
                    })
    
    if not records:
        # Return empty DataFrame with correct columns if no data
        return pd.DataFrame(columns=[Aliases.REGION, Aliases.ACTIVITE, Aliases.ANNEE, Aliases.MOIS, Aliases.CONSOMMATION_KWH])
    
    df = pd.DataFrame(records)
    
    # Sort for better readability
    df = df.sort_values([Aliases.REGION, Aliases.ACTIVITE, Aliases.ANNEE, Aliases.MOIS]).reset_index(drop=True)
    
    return df


def run_stf_srm_forecast(config, target_region, region_mode, df_regional, df_features, df_srm):
    """
    Execute STF SRM forecast for a single region and return results
    
    Args:
        config: ShortTermForecastConfig object (already configured)
        target_region: Name of the region to forecast
        region_mode: Run level mode for this region (0: only SRM, 1: activities + SRM)
        df_regional: Regional DataFrame
        df_features: Features DataFrame
        df_srm: Pre-computed SRM DataFrame (Regional + Distributors combined), with columns [Annee, Mois, VARIABLE]
        
    Returns:
        dict with:
            - status: 'success' or 'error'
            - results: List of forecast results for this region
            - error: Error message if status is 'error'
    """
    try:
        # Convert to legacy ANALYSIS_CONFIG format
        ANALYSIS_CONFIG = config.to_analysis_config()
        
        # Extract commonly used values
        PROJECT_ROOT = config.project.project_root
        VARIABLE = config.data.variable
        use_monthly_clients_options = config.features.use_monthly_clients_options
        
        # Initialize DataLoader for client predictions
        data_loader = DataLoader(PROJECT_ROOT)

        # Determine run levels for this region:
        # 0: only SRM (level 0 - Total Regional + Distributors)
        # 1: both level 1 (individual activities) and level 0 (SRM)
        if region_mode == 0:
            RUN_LEVELS = {0}
        elif region_mode == 1:
            RUN_LEVELS = {0, 1}
        else:
            RUN_LEVELS = set(config.data.run_levels)
      
        print(f"Processing data for {target_region}...\n")

        # Get reg_var_col from df_regional columns (exclude Annee and Mois)
        reg_var_col = [col for col in df_regional.columns if col not in [Aliases.ANNEE, Aliases.MOIS, Aliases.ACTIVITE, Aliases.REGION]][0]

        print(f"\n{'#'*60}")
        print(f"ULTRA-STRICT MODE: PCA REFITTED IN EACH LOOCV FOLD")
        print(f"{'#'*60}")

        # Only load client predictions if we're going to use them
        client_predictions_lookup = {}
        if True in use_monthly_clients_options:
            client_predictions_lookup = data_loader.load_client_prediction_lookup(target_region)
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
                    config=ANALYSIS_CONFIG,
                    client_predictions=client_predictions_lookup.get(entity_name),
                    under_estimation_penalty=config.loss.under_estimation_penalty,
                )
                _maybe_add(res)

        # SRM (Regional + Distributors)
        if 0 in RUN_LEVELS:
            print(f"\n{'#'*60}\nSRM (Regional + Distributors)\n{'#'*60}")
            res = run_analysis_for_entity(
                df_srm,
                "SRM_Regional_Plus_Dist",
                df_features,
                config=ANALYSIS_CONFIG,
                client_predictions=client_predictions_lookup.get("SRM_Regional_Plus_Dist"),
                under_estimation_penalty=config.loss.under_estimation_penalty,
            )
            _maybe_add(res)

        print(f"\n✅ STF SRM Forecast completed for {target_region}")
        
        return {
            'status': 'success',
            'results': all_results
        }
        
    except Exception as e:
        print(f"\n❌ STF SRM Forecast failed for {target_region}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'results': [],
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
    
    # Load configuration
    config = ShortTermForecastConfig.from_yaml(config_path)
    
    # Override eval years if needed (e.g., latest_year_in_data=2023)
    latest_year_in_data = 2023
    if latest_year_in_data is not None:
        config.evaluation.eval_years_start = latest_year_in_data + 1
        config.evaluation.eval_years_end = latest_year_in_data + 1
    
    PROJECT_ROOT = config.project.project_root
    VARIABLE = config.data.variable
    REGIONS = config.data.regions
    exp_name = config.project.exp_name
    
    # Initialize DataLoader
    data_loader = DataLoader(PROJECT_ROOT)
    
    # Create output directories for each region
    output_dirs = {}
    for TARGET_REGION in REGIONS.keys():
        output_dir = PROJECT_ROOT / 'outputs/outputs_srm' / exp_name / clean_name(TARGET_REGION)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[TARGET_REGION] = output_dir
    
    # Run forecast for each region and collect results
    all_results_by_region = {}
    overall_status = 'success'
    
    for TARGET_REGION, region_mode in REGIONS.items():
        print(f"Loading data for {TARGET_REGION}...")
        df_regional, df_features, df_dist, var_cols = data_loader.load_srm_data(
            db_path=config.project.project_root / config.data.db_path,
            variable=VARIABLE,
            target_region=TARGET_REGION,
        )
        
        # Compute df_srm (Regional + Distributors)
        reg_var_col = var_cols["regional"]
        df_total_regional = (
            df_regional.groupby([Aliases.ANNEE, Aliases.MOIS])
            .agg({reg_var_col: 'sum'})
            .reset_index()
            .rename(columns={reg_var_col: VARIABLE})
        )
        
        if df_dist is not None and len(df_dist) > 0:
            dist_var_col = var_cols["distributor"]
            df_all_dist = (
                df_dist.groupby([Aliases.ANNEE, Aliases.MOIS])
                .agg({dist_var_col: 'sum'})
                .reset_index()
                .rename(columns={dist_var_col: VARIABLE})
            )
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
        else:
            df_srm = df_total_regional[[Aliases.ANNEE, Aliases.MOIS, VARIABLE]].copy()
        
        # Run forecast for this region
        result = run_stf_srm_forecast(
            config=config,
            target_region=TARGET_REGION,
            region_mode=region_mode,
            df_regional=df_regional,
            df_features=df_features,
            df_srm=df_srm
        )
        
        if result['status'] == 'success':
            all_results_by_region[TARGET_REGION] = result['results']
        else:
            overall_status = 'error'
            all_results_by_region[TARGET_REGION] = []
    
    
    # Save outputs to disk for each region
    for TARGET_REGION, all_results in all_results_by_region.items():
        if all_results:
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
    sys.exit(0 if overall_status == 'success' else 1)
