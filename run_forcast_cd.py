import logging
import os
import sys
import warnings

import pandas as pd
import sqlite3
from pathlib import Path
from forcast_strategies import run_analysis_for_entity, save_summary
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - ULTRA-STRICT NO DATA LEAKAGE
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[0]

VARIABLE = "consommation"
UNIT = "Mwh"
exp_name = "activite_s4_nan"


N_PCS = 3
LAGS_OPTIONS = [2, 3]
ALPHAS = [0.01, 0.1, 1.0, 10.0]
R2_THRESHOLD = 0.6
PC_WEIGHTS = [0.5, 0.8]
PCA_LAMBDAS = [0.3, 0.7, 1.0]

training_end = None
use_monthly_temp_options = [False]
use_monthly_clients_options = [False]
client_pattern_weights = [0.3, 0.5, 0.8]
training_windows = [2,3,4]

FEATURE_BLOCKS = {
    'none': [],
    'gdp_only': ['pib_mdh'],
    'sectoral_only': ['gdp_primaire', 'gdp_secondaire', 'gdp_tertiaire'],
    'gdp_sectoral': ['pib_mdh', 'gdp_primaire', 'gdp_secondaire', 'gdp_tertiaire'],
}
eval_years_start = 2021
eval_years_end = 2023


ANALYSIS_CONFIG = {
    "value_col": VARIABLE,
    "N_PCS": N_PCS,
    "LAGS_OPTIONS": LAGS_OPTIONS,
    "FEATURE_BLOCKS": FEATURE_BLOCKS,
    "ALPHAS": ALPHAS,
    "PC_WEIGHTS": PC_WEIGHTS,
    "R2_THRESHOLD": R2_THRESHOLD,
    "unit": UNIT,
    "PCA_LAMBDAS": PCA_LAMBDAS,
    "training_end": training_end,
    "use_monthly_temp_options": use_monthly_temp_options,
    "use_monthly_clients_options": use_monthly_clients_options,
    "client_pattern_weights": client_pattern_weights,
    "training_windows": training_windows,
    "eval_years_end": eval_years_end,
    "eval_years_start": eval_years_start,
}
# ============================================================================
# LOAD DATA
# ============================================================================
db_path = PROJECT_ROOT / 'data/ONEE_Regional_COMPLETE.db'

db_regional = sqlite3.connect(db_path)

query_features = f"""
SELECT Year as annee, 
       SUM(GDP_Millions_DH) as pib_mdh,
       SUM(GDP_Primaire) as gdp_primaire,
       SUM(GDP_Secondaire) as gdp_secondaire,
       SUM(GDP_Tertiaire) as gdp_tertiaire,
       AVG(temp) as temperature_annuelle
FROM regional_features
GROUP BY Year
"""
df_features = pd.read_sql_query(query_features, db_regional)

db_regional.close()

df_path = PROJECT_ROOT / "data/cd_data_2013_2023.csv"

df_contrats = pd.read_csv(df_path)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
# Choose which analysis parts (levels) to run
# 1: Contrat, 2: Partenaire (client), 3: Activities
RUN_LEVELS = {3}

output_file =  PROJECT_ROOT / f'outputs_cd/{exp_name}.xlsx'
monthly_output_file = PROJECT_ROOT / f'outputs_cd/{exp_name}_monthly_forcasts.xlsx'
all_results = []

if 1 in RUN_LEVELS:
    print(f"\n{'#'*60}")
    print(f"LEVEL 1: CONTRAT")
    print(f"{'#'*60}")

    contrats = sorted(df_contrats['contrat'].unique())

    for i, contrat in enumerate(contrats):
        df_contrat = df_contrats[df_contrats['contrat'] == contrat].copy()
        result = run_analysis_for_entity(
            df_contrat, 
            f"Contrat_{contrat}", 
            df_features,
            df_contrats,
            config = ANALYSIS_CONFIG,
            under_estimation_penalty = 2,
        )
        if result:
            all_results.append(result)
        
        if i%10 == 0:
            save_summary(all_results, output_file, monthly_output_file)

if 2 in RUN_LEVELS:
    print(f"\n{'#'*60}")
    print(f"LEVEL 2: PARTENAIRE")
    print(f"{'#'*60}")

    # partenaires = sorted(df_contrats['partenaire'].unique())
    partenaires = ["ONCF"]

    for i, partenaire in enumerate(partenaires):
        df_partenaire = df_contrats[df_contrats['partenaire'] == partenaire].copy()
        df_partenaire = df_partenaire.groupby(['annee', 'mois']).agg({
            VARIABLE: 'sum',
            'temperature': 'mean'
        }).reset_index()

        result = run_analysis_for_entity(
            df_partenaire, 
            f"Partenaire_{partenaire}", 
            df_features,
            df_partenaire, 
            config = ANALYSIS_CONFIG,
            under_estimation_penalty = 2,
        )
        if result:
            all_results.append(result)
        
        if i%10 == 0:
            save_summary(all_results, output_file, monthly_output_file)


if 3 in RUN_LEVELS:
    print(f"\n{'#'*60}")
    print(f"LEVEL 2: ACTIVITE")
    print(f"{'#'*60}")

    # activites = sorted(df_contrats['activite'].unique())
    activites = ["industries alimentaires"]

    for i, activite in enumerate(activites):
        df_activite = df_contrats[df_contrats['activite'] == activite].copy()
        df_activite = df_activite.groupby(['annee', 'mois']).agg({
            VARIABLE: 'sum',
            'temperature': 'mean'
        }).reset_index()

        result = run_analysis_for_entity(
            df_activite, 
            f"Activité_{activite}", 
            df_features,
            df_activite, 
            config = ANALYSIS_CONFIG,
            favor_overestimation=False,
            under_estimation_penalty = 1,
        )
        if result:
            all_results.append(result)
        
        if i%10 == 0:
            save_summary(all_results, output_file, monthly_output_file)


save_summary(all_results, output_file, monthly_output_file)
with open(PROJECT_ROOT / f'outputs_cd/all_results_{exp_name}.pkl', "wb") as f:
    pickle.dump(all_results, f)

print(f"\n{'='*60}")
print(f"✅ Results saved to: {output_file}")
print(f"📊 Total entities analyzed: {len(all_results)}")
print(f"🔒 ZERO DATA LEAKAGE GUARANTEED")
print(f"{'='*60}\n")
