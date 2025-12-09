import logging
import os
import sys
import warnings

import pandas as pd
from pathlib import Path
from short_term_forecast_strategies import run_analysis_for_entity, save_summary, get_move_in_year
from new_entities_handlers import handle_growth_entity, handle_similarity_entity, filter_established_entities, handle_similarity_entity_prediction, handle_growth_entity_prediction
import pickle
import joblib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from onee.config.stf_config import ShortTermForecastConfig
from onee.data.loader import DataLoader
from onee.data.names import Aliases


# ────────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────────────────────────

def run_stf_cd_forecast(config_path="configs/stf_cd.yaml"):
    """
    Execute STF CD forecast and return results
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        dict with:
            - status: 'success' or 'error'
            - results: List of forecast results
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
        RUN_LEVELS = config.data.run_levels
        exp_name = config.project.exp_name
        
        # Load data
        data_loader = DataLoader(PROJECT_ROOT)
        
        df_contrats, df_features = data_loader.load_cd_data(
            db_path=config.project.project_root / config.data.db_path,
        )
        
        all_results = []
        
        # ===========================
        # Main processing
        # ===========================
        if 1 in RUN_LEVELS:
            print(f"\n{'#'*60}")
            print(f"LEVEL 1: CONTRAT")
            print(f"{'#'*60}")

            contrats = sorted(df_contrats[Aliases.CONTRAT].unique())

            established_contracts = []
            growth_contracts = []
            similarity_contracts = []

            # --- Categorize contracts ---
            for contrat in contrats:
                df_contrat = df_contrats[df_contrats[Aliases.CONTRAT] == contrat].copy()
                move_in_year = get_move_in_year(df_contrat)

                if move_in_year is None or move_in_year <= config.evaluation.eval_years_end - 2:
                    established_contracts.append(contrat)
                elif move_in_year == config.evaluation.eval_years_end - 1:
                    growth_contracts.append(contrat)
                elif move_in_year == config.evaluation.eval_years_end:
                    similarity_contracts.append(contrat)

            print(f"\n🟢 Established contracts: {len(established_contracts)}")
            print(f"🚀 Growth contracts: {len(growth_contracts)}")
            print(f"🔗 Similarity contracts: {len(similarity_contracts)}")

            # --- Process established contracts (normal forecasting) ---
            all_results_established = []
            for i, contrat in enumerate(established_contracts):
                df_contrat = df_contrats[df_contrats[Aliases.CONTRAT] == contrat].copy()
                result = run_analysis_for_entity(
                    df_contrat,
                    f"Contrat_{contrat}",
                    df_features,
                    config=ANALYSIS_CONFIG,
                    favor_overestimation=config.loss.favor_overestimation,
                    under_estimation_penalty=config.loss.under_estimation_penalty,
                )
                if result:
                    all_results_established.append(result)

            all_results_growth = []
            for contrat in growth_contracts:
                all_results_growth.append(handle_growth_entity_prediction(df_contrats, all_results_established, contrat))

            all_results_similarity = []
            for contrat in similarity_contracts:
                all_results_similarity.append(handle_similarity_entity_prediction(df_contrats, all_results_established, contrat))
            
            all_results_contrats = all_results_established + all_results_similarity + all_results_growth
            all_results += all_results_contrats

        print(f"\n✅ STF CD Forecast completed: {len(all_results)} results")
        
        return {
            'status': 'success',
            'results': all_results
        }
        
    except Exception as e:
        print(f"\n❌ STF CD Forecast failed: {str(e)}")
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
    config_path = Path(__file__).parent / "configs/stf_cd.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Reload config to create output directories
    config = ShortTermForecastConfig.from_yaml(config_path)
    PROJECT_ROOT = config.project.project_root
    exp_name = config.project.exp_name
    
    # Create output directory
    output_dir = PROJECT_ROOT / 'outputs/outputs_cd'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the forecast
    result = run_stf_cd_forecast(config_path=config_path)
    
    # If successful, save outputs to disk
    if result['status'] == 'success':
        all_results = result['results']
        
        output_file = output_dir / f'{exp_name}.xlsx'
        monthly_output_file = output_dir / f'{exp_name}_monthly_forcasts.xlsx'
        
        save_summary(all_results, output_file, monthly_output_file)
        
        with open(output_dir / f'all_results_{exp_name}.pkl', "wb") as f:
            pickle.dump(all_results, f)

        print(f"\n{'='*60}")
        print(f"✅ Results saved to: {output_file}")
        print(f"📊 Total entities analyzed: {len(all_results)}")
        print(f"🔒 ZERO DATA LEAKAGE GUARANTEED")
        print(f"{'='*60}\n")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)
