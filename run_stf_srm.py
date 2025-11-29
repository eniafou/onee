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

# ────────────────────────────────────────────────────────────────────────────────
# LOAD CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
config_path = Path(__file__).parent / "configs/stf_srm.yaml"
config = ShortTermForecastConfig.from_yaml(config_path)

# Convert to legacy ANALYSIS_CONFIG format
ANALYSIS_CONFIG = config.to_analysis_config()
print(ANALYSIS_CONFIG)

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
    # Choose which analysis parts (levels) to run
    # 1: Activities, 2: Aggregated BT, 3: Aggregated MT, 4: Total Regional,
    # 5: Individual Distributors, 6: All Distributors, 7: SRM (Regional+Dist)
    RUN_LEVELS = config.data.run_levels

    for TARGET_REGION in REGIONS:
        print(f"Loading data for {TARGET_REGION}...\n")

        # DB paths
        db_regional_path = config.data.db_regional
        db_dist_path = config.data.db_distributors

        # Load data using DataLoader
        df_regional, df_features, df_dist, var_cols = data_loader.load_srm_data(
            db_regional_path=db_regional_path,
            db_dist_path=db_dist_path,
            variable=VARIABLE,
            target_region=TARGET_REGION,
        )
        print(df_regional.head())
        reg_var_col = var_cols["regional"]

        print(f"\n{'#'*60}")
        print(f"ULTRA-STRICT MODE: PCA REFITTED IN EACH LOOCV FOLD")
        print(f"{'#'*60}")

        # Only load client predictions if we're going to use them
        client_predictions_lookup = {}
        if True in use_monthly_clients_options:
            client_predictions_lookup = data_loader.load_client_prediction_lookup(TARGET_REGION)
        activities = sorted(df_regional['activite'].unique())

        all_results = []

        def _maybe_add(res):
            if res:
                all_results.append(res)

        # LEVEL 1: INDIVIDUAL ACTIVITIES
        if 1 in RUN_LEVELS:
            print(f"\n{'#'*60}\nLEVEL 1: INDIVIDUAL ACTIVITIES\n{'#'*60}")
            for activity in activities:
                df_activity = (
                    df_regional[df_regional['activite'] == activity][['annee', 'mois', reg_var_col]]
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

        # LEVEL 2: AGGREGATED BT
        if 2 in RUN_LEVELS:
            print(f"\n{'#'*60}\nLEVEL 2: AGGREGATED BT\n{'#'*60}")
            mt_activities = ["Administratif_mt", "Agricole", "Industriel", "Résidentiel", "Tertiaire"]
            bt_activities = [a for a in activities if a not in mt_activities]
            df_bt = df_regional[df_regional['activite'].isin(bt_activities)]
            if df_bt.empty:
                print("⚠️  Skipping: no BT activities found.")
            else:
                df_bt_agg = (
                    df_bt.groupby(['annee', 'mois'])
                    .agg({reg_var_col: 'sum'})
                    .reset_index()
                    .rename(columns={reg_var_col: VARIABLE})
                )
                DataLoader.aggregate_predictions(
                    client_predictions_lookup,
                    "Aggregated_BT",
                    [f"Activity_{a}" for a in bt_activities],
                )
                res = run_analysis_for_entity(
                    df_bt_agg,
                    "Aggregated_BT",
                    df_features,
                    df_bt_agg,
                    config=ANALYSIS_CONFIG,
                    client_predictions=client_predictions_lookup.get("Aggregated_BT"),
                    under_estimation_penalty=config.loss.under_estimation_penalty,
                )
                _maybe_add(res)

        # LEVEL 3: AGGREGATED MT
        if 3 in RUN_LEVELS:
            print(f"\n{'#'*60}\nLEVEL 3: AGGREGATED MT\n{'#'*60}")
            df_mt = df_regional[df_regional['activite'].isin(mt_activities)]
            if df_mt.empty:
                print("⚠️  Skipping: no MT activities found.")
            else:
                df_mt_agg = (
                    df_mt.groupby(['annee', 'mois'])
                    .agg({reg_var_col: 'sum'})
                    .reset_index()
                    .rename(columns={reg_var_col: VARIABLE})
                )
                DataLoader.aggregate_predictions(
                    client_predictions_lookup,
                    "Aggregated_MT",
                    [f"Activity_{a}" for a in mt_activities],
                )
                res = run_analysis_for_entity(
                    df_mt_agg,
                    "Aggregated_MT",
                    df_features,
                    df_mt_agg,
                    config=ANALYSIS_CONFIG,
                    client_predictions=client_predictions_lookup.get("Aggregated_MT"),
                    under_estimation_penalty=config.loss.under_estimation_penalty,
                )
                _maybe_add(res)

        # LEVEL 4: TOTAL REGIONAL
        print_total_regional = 4 in RUN_LEVELS
        df_total_regional = (
            df_regional.groupby(['annee', 'mois'])
            .agg({reg_var_col: 'sum'})
            .reset_index()
            .rename(columns={reg_var_col: VARIABLE})
        )
        if print_total_regional:
            print(f"\n{'#'*60}\nLEVEL 4: TOTAL REGIONAL\n{'#'*60}")
            DataLoader.aggregate_predictions(
                client_predictions_lookup,
                "Total_Regional",
                [f"Activity_{a}" for a in activities],
            )
            res = run_analysis_for_entity(
                df_total_regional,
                "Total_Regional",
                df_features,
                df_total_regional,
                config=ANALYSIS_CONFIG,
                client_predictions=client_predictions_lookup.get("Total_Regional"),
                under_estimation_penalty=config.loss.under_estimation_penalty,
            )
            _maybe_add(res)

        # LEVEL 5: INDIVIDUAL DISTRIBUTORS
        if 5 in RUN_LEVELS:
            print(f"\n{'#'*60}\nLEVEL 5: INDIVIDUAL DISTRIBUTORS\n{'#'*60}")
            if df_dist is None:
                print(f"⚠️  Skipping: distributors data unsupported for VARIABLE='{VARIABLE}'.")
            else:
                dist_var_col = var_cols["distributor"]
                distributors = sorted(df_dist['distributeur'].unique())
                for distributor in distributors:
                    df_distributor = (
                        df_dist[df_dist['distributeur'] == distributor][['annee', 'mois', dist_var_col]]
                        .copy()
                        .rename(columns={dist_var_col: VARIABLE})
                    )
                    safe_name = clean_name(distributor)
                    entity_name = f"Distributor_{safe_name}"
                    res = run_analysis_for_entity(
                        df_distributor,
                        entity_name,
                        df_features,
                        df_regional,
                        config=ANALYSIS_CONFIG,
                        client_predictions=client_predictions_lookup.get(entity_name),
                        under_estimation_penalty=config.loss.under_estimation_penalty,
                    )
                    _maybe_add(res)

        # LEVEL 6: ALL DISTRIBUTORS
        df_all_dist = None
        if df_dist is not None:
            dist_var_col = var_cols["distributor"]
            df_all_dist = (
                df_dist.groupby(['annee', 'mois'])
                .agg({dist_var_col: 'sum'})
                .reset_index()
                .rename(columns={dist_var_col: VARIABLE})
            )
        if 6 in RUN_LEVELS:
            print(f"\n{'#'*60}\nLEVEL 6: ALL DISTRIBUTORS COMBINED\n{'#'*60}")
            if df_all_dist is None:
                print(f"⚠️  Skipping: distributors data unsupported for VARIABLE='{VARIABLE}'.")
            else:
                res = run_analysis_for_entity(
                    df_all_dist,
                    "All_Distributors",
                    df_features,
                    df_regional,
                    config=ANALYSIS_CONFIG,
                    client_predictions=client_predictions_lookup.get("All_Distributors"),
                    under_estimation_penalty=config.loss.under_estimation_penalty,
                )
                _maybe_add(res)

        # LEVEL 7: SRM (Regional + Distributors)
        if 7 in RUN_LEVELS:
            print(f"\n{'#'*60}\nLEVEL 7: SRM (Regional + Distributors)\n{'#'*60}")
            if df_all_dist is None:
                print(f"⚠️  Skipping SRM: distributors data unsupported for VARIABLE='{VARIABLE}'.")
            else:
                df_srm = (
                    pd.concat(
                        [df_total_regional[['annee', 'mois', VARIABLE]],
                        df_all_dist[['annee', 'mois', VARIABLE]]],
                        ignore_index=True
                    )
                    .groupby(['annee', 'mois'])
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
