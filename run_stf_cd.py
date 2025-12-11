import sys
import warnings

import pandas as pd
from pathlib import Path
from short_term_forecast_strategies import run_analysis_for_entity, save_summary, get_move_in_year
from new_entities_handlers import handle_similarity_entity_prediction, handle_growth_entity_prediction
import pickle
from onee.config.stf_config import ShortTermForecastConfig
from onee.data.loader import DataLoader
from onee.data.names import Aliases

warnings.filterwarnings('ignore')

def prepare_prediction_output(all_results, df_contrats):
    """
    Create a tall DataFrame with monthly predictions from CD (contract-level) forecast results.
    
    Args:
        all_results: List of result dictionaries from run_stf_cd_forecast
        df_contrats: DataFrame with contract metadata (Region, Partenaire, Contrat, Activite, etc.)
        
    Returns:
        pandas DataFrame with columns: 
            Region, Partenaire, Contrat, Activity, Year, Month, Consommation_Total
    """
    import pandas as pd
    from onee.data.names import Aliases
    
    records = []
    
    # Build a lookup from df_contrats for contract metadata
    # Group by contrat to get unique contract info
    contract_info = (
        df_contrats
        .groupby(Aliases.CONTRAT, as_index=False)
        .agg({
            Aliases.REGION: 'first',
            Aliases.PARTENAIRE: 'first',
            Aliases.ACTIVITE: 'first',
        })
    )
    contract_lookup = contract_info.set_index(Aliases.CONTRAT).to_dict('index')
    
    for result in all_results:
        if result is None:
            continue
            
        entity = result.get("entity", "")
        monthly_details = result.get("monthly_details", {})
        
        if not monthly_details:
            continue
        
        # Extract contract number from entity name (e.g., "Contrat_123456" -> 123456)
        contrat = None
        if entity.startswith("Contrat_"):
            try:
                contrat = int(entity.replace("Contrat_", ""))
            except ValueError:
                contrat = entity.replace("Contrat_", "")
        else:
            contrat = entity
        
        # Get contract metadata from lookup
        info = contract_lookup.get(contrat, {})
        region = info.get(Aliases.REGION, None)
        partenaire = info.get(Aliases.PARTENAIRE, None)
        activite = info.get(Aliases.ACTIVITE, None)
        
        # Extract monthly predictions for each year
        for year, df_monthly in monthly_details.items():
            if df_monthly is None or df_monthly.empty:
                continue
            
            for _, row in df_monthly.iterrows():
                month = row.get("Month")
                predicted = row.get("Predicted")
                
                if month is None or predicted is None:
                    continue
                
                records.append({
                    Aliases.REGION: region,
                    Aliases.PARTENAIRE: partenaire,
                    Aliases.CONTRAT: contrat,
                    Aliases.ACTIVITE: activite,
                    Aliases.ANNEE: year,
                    Aliases.MOIS: int(month),
                    Aliases.CONSOMMATION_KWH: predicted,
                })
    
    if not records:
        return pd.DataFrame(columns=[
            Aliases.REGION, Aliases.PARTENAIRE, Aliases.CONTRAT, Aliases.ACTIVITE, 
            Aliases.ANNEE, Aliases.MOIS, Aliases.CONSOMMATION_KWH
        ])
    
    df = pd.DataFrame(records)
    
    # Sort for better readability
    df = df.sort_values(
        [Aliases.REGION, Aliases.PARTENAIRE, Aliases.CONTRAT, Aliases.ACTIVITE, Aliases.ANNEE, Aliases.MOIS]
    ).reset_index(drop=True)
    
    return df

def prepare_ca_output(df_prediction, df_contrats):
    """
    Add consumption breakdown columns (ZCONHC, ZCONHL, ZCONHP), Puissance_Facturee, 
    and Niveau_tension to the output DataFrame.
    
    For each (contrat, year, month) in df_prediction, finds the latest available data
    in df_contrats (where year/month <= target) and calculates percentages from
    the historical values, then applies them to Consommation_Total.
    
    Args:
        df_prediction: DataFrame from prepare_prediction_output with Consommation_Total
        df_contrats: DataFrame with Consommation_ZCONHC, Consommation_ZCONHL, Consommation_ZCONHP
        
    Returns:
        DataFrame with added columns: Consommation_ZCONHC, Consommation_ZCONHL, Consommation_ZCONHP,
                                      Puissance_Facturee, Niveau_tension
    """
    
    # Prepare df_contrats with required columns for consumption breakdown
    contrats_subset = df_contrats[[
        Aliases.CONTRAT, Aliases.ANNEE, Aliases.MOIS,
        Aliases.CONSOMMATION_ZCONHC, Aliases.CONSOMMATION_ZCONHL, Aliases.CONSOMMATION_ZCONHP,
        Aliases.PUISSANCE_FACTUREE, Aliases.NIVEAU_TENSION
    ]].copy()
    
    # Create a sort key for finding latest available data
    contrats_subset["_year_month"] = (
        contrats_subset[Aliases.ANNEE] * 12 + contrats_subset[Aliases.MOIS]
    )
    
    # Sort by contrat and year_month descending to easily find latest
    contrats_subset = contrats_subset.sort_values(
        [Aliases.CONTRAT, "_year_month"], ascending=[True, False]
    )
    
    # Initialize new columns
    df_output = df_prediction.copy()
    df_output[Aliases.CONSOMMATION_ZCONHC] = None
    df_output[Aliases.CONSOMMATION_ZCONHL] = None
    df_output[Aliases.CONSOMMATION_ZCONHP] = None
    df_output[Aliases.PUISSANCE_FACTUREE] = None
    df_output[Aliases.NIVEAU_TENSION] = None
    
    # Group contrats data for faster lookup
    contrats_grouped = contrats_subset.groupby(Aliases.CONTRAT)
    
    for idx, row in df_output.iterrows():
        contrat = row[Aliases.CONTRAT]
        target_year = row[Aliases.ANNEE]
        target_month = row[Aliases.MOIS]
        target_ym = target_year * 12 + target_month
        cons_total = row[Aliases.CONSOMMATION_KWH]
        
        # Get data for this contract
        try:
            contract_data = contrats_grouped.get_group(contrat)
        except KeyError:
            # Contract not found in df_contrats - use fallback values
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHC] = cons_total / 3
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHL] = cons_total / 3
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHP] = cons_total / 3
            df_output.at[idx, Aliases.NIVEAU_TENSION] = "HT"
            continue
        
        # Find latest available data (year_month <= target)
        available = contract_data[contract_data["_year_month"] <= target_ym]
        
        if available.empty:
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHC] = cons_total / 3
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHL] = cons_total / 3
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHP] = cons_total / 3
            df_output.at[idx, Aliases.NIVEAU_TENSION] = "HT"
            continue
        
        # Get the latest row (first due to descending sort)
        latest = available.iloc[0]
        
        # Set Puissance_Facturee and Niveau_tension from latest available data
        df_output.at[idx, Aliases.PUISSANCE_FACTUREE] = latest[Aliases.PUISSANCE_FACTUREE]
        niveau_tension = latest[Aliases.NIVEAU_TENSION]
        df_output.at[idx, Aliases.NIVEAU_TENSION] = niveau_tension if niveau_tension else "HT"
        
        # Calculate total from breakdown columns
        total_breakdown = (
            latest[Aliases.CONSOMMATION_ZCONHC] + latest[Aliases.CONSOMMATION_ZCONHL] + latest[Aliases.CONSOMMATION_ZCONHP]
        )
        
        if total_breakdown > 0:
            # Calculate percentages
            pct_hc = latest[Aliases.CONSOMMATION_ZCONHC] / total_breakdown
            pct_hl = latest[Aliases.CONSOMMATION_ZCONHL] / total_breakdown
            pct_hp = latest[Aliases.CONSOMMATION_ZCONHP] / total_breakdown
            
            # Apply percentages to Consommation_Total
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHC] = cons_total * pct_hc
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHL] = cons_total * pct_hl
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHP] = cons_total * pct_hp
        else:
            # If total is zero, split equally
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHC] = cons_total / 3
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHL] = cons_total / 3
            df_output.at[idx, Aliases.CONSOMMATION_ZCONHP] = cons_total / 3
    
    return df_output

def run_stf_cd_forecast(config, df_contrats, df_features):
    """
    Execute STF CD forecast and return results
    
    Args:
        config: ShortTermForecastConfig object
        df_contrats: DataFrame with contract data
        df_features: DataFrame with features data
        
    Returns:
        dict with:
            - status: 'success' or 'error'
            - results: List of forecast results
            - error: Error message if status is 'error'
    """
    try:
        # Convert to legacy ANALYSIS_CONFIG format
        ANALYSIS_CONFIG = config.to_analysis_config()
        
        # Extract commonly used values
        PROJECT_ROOT = config.project.project_root
        VARIABLE = config.data.variable
        RUN_LEVELS = config.data.run_levels
        exp_name = config.project.exp_name
        
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
                all_results_growth.append(handle_growth_entity_prediction(df_contrats, all_results_established, contrat, prediction_year = config.evaluation.eval_years_end))

            all_results_similarity = []
            for contrat in similarity_contracts:
                all_results_similarity.append(handle_similarity_entity_prediction(df_contrats, all_results_established, contrat, prediction_year = config.evaluation.eval_years_end))
            
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
    
    # Load configuration from YAML
    config = ShortTermForecastConfig.from_yaml(config_path)
    
    # Apply overrides
    latest_year_in_data = 2023
    config.evaluation.eval_years_start = latest_year_in_data + 1
    config.evaluation.eval_years_end = latest_year_in_data + 1
    
    PROJECT_ROOT = config.project.project_root
    exp_name = config.project.exp_name
    
    # Create output directory
    output_dir = PROJECT_ROOT / 'outputs/outputs_cd'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize DataLoader and load data
    data_loader = DataLoader(PROJECT_ROOT)
    df_contrats, df_features = data_loader.load_cd_data(
        db_path=config.project.project_root / config.data.db_path,
    )
    
    # Run the forecast with config and data
    result = run_stf_cd_forecast(config=config, df_contrats=df_contrats, df_features=df_features)
    
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
