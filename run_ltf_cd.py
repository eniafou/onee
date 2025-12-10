# ─────────────────────────────────────────────────────────────
# run_horizon_forecast_cd.py (Clean + Correct)
# ─────────────────────────────────────────────────────────────

import os
import run_stf_srm
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
def prepare_prediction_output(all_results):
    """
    Create a simple summary DataFrame with Region, Year, and Predicted_Annual
    
    Args:
        all_results: List of forecast result dictionaries
        
    Returns:
        pandas DataFrame with Region, Year, and Predicted_Annual columns
    """
    df_summary_records = []
    for r in all_results:
        for y, v in zip(r["forecast_years"], r["pred_annual"]):
            df_summary_records.append(
                {
                    "Activity": r.get("activity"),
                    "Year": y,
                    "Consommation_Total": v
                }
            )
    
    return pd.DataFrame(df_summary_records)


def prepare_ca_output(df_prediction, df_contrats):
    """
    For each unique (Activity, Year) in df_prediction, find all contracts still operating 
    (move_out_year >= year) and distribute Consommation_Total based on Puissance_Facturee.
    Then apply consumption breakdown percentages.
    
    Args:
        df_prediction: DataFrame with Activity, Year, and Consommation_Total columns
        df_contrats: DataFrame with contract data including DATE_DEMENAGEMENT
        
    Returns:
        DataFrame with columns: Region, Partenaire, Contrat, Activity, Year,
                               Consommation_Total, Consommation_ZCONHC, Consommation_ZCONHL, 
                               Consommation_ZCONHP, Puissance_Facturee, Niveau_tension
    """
    from onee.utils import get_move_out_year
    
    # Column names for the output breakdown
    ZCONHC = "Consommation_ZCONHC"
    ZCONHL = "Consommation_ZCONHL"
    ZCONHP = "Consommation_ZCONHP"
    
    # Get unique contracts and compute move_out_year for each
    contracts = df_contrats[Aliases.CONTRAT].unique()
    
    contract_move_out = {}
    for contrat in contracts:
        df_c = df_contrats[df_contrats[Aliases.CONTRAT] == contrat]
        move_out_year = get_move_out_year(df_c)
        contract_move_out[contrat] = move_out_year
    
    # Build contract metadata lookup (Region, Partenaire, Activity)
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
    
    # Aggregate df_contrats by year (sum monthly values per contract per year)
    contrats_yearly = df_contrats.groupby(
        [Aliases.CONTRAT, Aliases.ANNEE], as_index=False
    ).agg({
        Aliases.CONSOMMATION_ZCONHC: 'sum',
        Aliases.CONSOMMATION_ZCONHL: 'sum',
        Aliases.CONSOMMATION_ZCONHP: 'sum',
        Aliases.PUISSANCE_FACTUREE: 'sum',
        Aliases.NIVEAU_TENSION: 'first',  # Take first value for niveau_tension
    })
    
    # Sort by contrat and year descending to easily find latest
    contrats_yearly = contrats_yearly.sort_values(
        [Aliases.CONTRAT, Aliases.ANNEE], ascending=[True, False]
    )
    
    # Group contrats data for faster lookup
    contrats_grouped = contrats_yearly.groupby(Aliases.CONTRAT)
    
    records = []
    
    # Process each (Activity, Year) from predictions
    for _, pred_row in df_prediction.iterrows():
        activity = pred_row["Activity"]
        year = pred_row["Year"]
        total_consumption = pred_row["Consommation_Total"]
        
        # Find all contracts for this activity that are still operating (move_out_year >= year)
        active_contracts = []
        for contrat, move_out in contract_move_out.items():
            info = contract_lookup.get(contrat, {})
            contrat_activity = info.get(Aliases.ACTIVITE, None)
            if contrat_activity == activity and (move_out is None or move_out >= year):
                active_contracts.append(contrat)
        
        # First pass: collect Puissance_Facturee and other data for all active contracts
        contract_data_list = []
        for contrat in active_contracts:
            info = contract_lookup.get(contrat, {})
            region = info.get(Aliases.REGION, None)
            partenaire = info.get(Aliases.PARTENAIRE, None)
            activite = info.get(Aliases.ACTIVITE, None)
            
            pf_value = 0.0
            niveau_tension = "HT"
            pct_hc = 1/3
            pct_hl = 1/3
            pct_hp = 1/3
            
            try:
                contract_data = contrats_grouped.get_group(contrat)
                # Find latest available year data (year <= target year)
                available = contract_data[contract_data[Aliases.ANNEE] <= year]
                
                if not available.empty:
                    latest = available.iloc[0]  # First row is latest due to descending sort
                    pf_value = latest[Aliases.PUISSANCE_FACTUREE] or 0.0
                    nt = latest[Aliases.NIVEAU_TENSION]
                    niveau_tension = nt if nt else "HT"
                    
                    # Calculate breakdown percentages from yearly aggregates
                    zconhc_hist = latest[Aliases.CONSOMMATION_ZCONHC] or 0.0
                    zconhl_hist = latest[Aliases.CONSOMMATION_ZCONHL] or 0.0
                    zconhp_hist = latest[Aliases.CONSOMMATION_ZCONHP] or 0.0
                    total_breakdown = zconhc_hist + zconhl_hist + zconhp_hist
                    
                    if total_breakdown > 0:
                        pct_hc = zconhc_hist / total_breakdown
                        pct_hl = zconhl_hist / total_breakdown
                        pct_hp = zconhp_hist / total_breakdown
                    
            except KeyError:
                pass
            
            contract_data_list.append({
                "contrat": contrat,
                "region": region,
                "partenaire": partenaire,
                "activite": activite,
                "pf_value": pf_value,
                "niveau_tension": niveau_tension,
                "pct_hc": pct_hc,
                "pct_hl": pct_hl,
                "pct_hp": pct_hp,
            })
        
        # Calculate total Puissance_Facturee for this activity-year
        total_pf = sum(c["pf_value"] for c in contract_data_list)
        
        # Second pass: distribute consumption and apply percentages
        for c in contract_data_list:
            # Calculate contract's share of consumption based on Puissance_Facturee
            if total_pf > 0:
                pf_share = c["pf_value"] / total_pf
            else:
                # If no Puissance_Facturee data, distribute equally
                pf_share = 1 / len(contract_data_list) if contract_data_list else 0
            
            contract_consumption = total_consumption * pf_share
            
            # Apply breakdown percentages to get ZCONHC, ZCONHL, ZCONHP
            records.append({
                "Region": c["region"],
                "Partenaire": c["partenaire"],
                "Contrat": c["contrat"],
                "Activity": c["activite"],
                "Year": year,
                "Consommation_Total": contract_consumption,
                ZCONHC: contract_consumption * c["pct_hc"],
                ZCONHL: contract_consumption * c["pct_hl"],
                ZCONHP: contract_consumption * c["pct_hp"],
                "Puissance_Facturee": c["pf_value"],
                "Niveau_tension": c["niveau_tension"]
            })
    
    if not records:
        return pd.DataFrame(columns=[
            "Region", "Partenaire", "Contrat", "Activity", "Year",
            "Consommation_Total", ZCONHC, ZCONHL, ZCONHP, 
            "Puissance_Facturee", "Niveau_tension"
        ])
    
    df_output = pd.DataFrame(records)
    
    # Sort for better readability
    df_output = df_output.sort_values(
        ["Region", "Partenaire", "Contrat", "Activity", "Year"]
    ).reset_index(drop=True)
    
    return df_output



# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

def run_ltf_cd_forecast(config_path="configs/ltf_cd.yaml",  latest_year_in_data=None, horizon=5, output_dir=None,):
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

        if latest_year_in_data is not None:
            config.temporal.forecast_runs[0] = (
                config.temporal.forecast_runs[0][0],
                latest_year_in_data
            )
            config.temporal.horizon = horizon if horizon is not None else config.temporal.horizon
        
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
                    df_a = df_a.merge(df_features, on=Aliases.ANNEE, how="outer")
                    print(df_a.tail())

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
                            "activity": activite,
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
    result = run_ltf_cd_forecast(config_path=config_path, output_dir=None, latest_year_in_data=2023)
    df_prediction = prepare_prediction_output(result['results'])
    data_loader = DataLoader(config.project.project_root)
    df_contrats, _ = data_loader.load_cd_data(
            db_path=config.project.project_root / config.data.db_path,)
    final_df = prepare_ca_output(df_prediction, df_contrats)
    final_df.to_csv("test_final_df.csv", index=False)
    # If successful, save outputs to disk
    # if result['status'] == 'success':
    #     all_results = result['results']
        
    #     # Save pickle file
    #     with open(
    #         output_dir / f"{config.data.target_variable}_{config.project.exp_name}.pkl",
    #         "wb",
    #     ) as f:
    #         pickle.dump(all_results, f)
        
    #     # Create and save Excel summary
    #     df_summary = create_summary_dataframe(all_results)
    #     out_xlsx = (
    #         output_dir / f"summary_{config.data.target_variable}_{config.project.exp_name}.xlsx"
    #     )
    #     df_summary.to_excel(out_xlsx, index=False)
    #     print(f"\n📁 Saved horizon forecasts to {out_xlsx}")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)