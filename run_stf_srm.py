import warnings
warnings.filterwarnings('ignore')

import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from short_term_forecast_strategies import run_analysis_for_entity, save_summary
from onee.utils import clean_name
from onee.config.stf_config import ShortTermForecastConfig
from onee.data.loader import DataLoader
from onee.data.names import Aliases

# ────────────────────────────────────────────────────────────────────────────────
# LOAD CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
config_path = Path(__file__).parent / "configs/stf_srm.yaml"
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

# ────────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for TARGET_REGION, region_mode in REGIONS.items():
        # Determine run levels for this region:
        # 0: only SRM (level 0 - Total Regional + Distributors)
        # 1: both level 1 (individual activities) and level 0 (SRM)
        if region_mode == 0:
            RUN_LEVELS = {0}
        elif region_mode == 1:
            RUN_LEVELS = {0, 1}
        else:
            RUN_LEVELS = {0}  # Default to SRM only
      
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

        # Create output directories
        output_dir = PROJECT_ROOT / 'outputs_srm' / exp_name / clean_name(TARGET_REGION)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save outputs
        output_file = output_dir / f'{clean_name(TARGET_REGION)}_{VARIABLE}_{exp_name}.xlsx'
        monthly_book_file = output_dir / f'monthly_predictions_by_entity_{clean_name(TARGET_REGION)}_{VARIABLE}_{exp_name}.xlsx'
        save_summary(all_results, output_file, monthly_book_file)

        with open(output_dir / f'all_results_{clean_name(TARGET_REGION)}_{VARIABLE}.pkl', "wb") as f:#_{exp_name}
            pickle.dump(all_results, f)

        print(f"\n{'='*60}")
        print(f"✅ Results saved to: {output_file}")
        print(f"📊 Total entities analyzed: {len(all_results)}")
        print(f"🔒 ZERO DATA LEAKAGE GUARANTEED")
        print(f"{'='*60}\n")
