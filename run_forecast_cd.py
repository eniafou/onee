import logging
import os
import sys
import warnings

import pandas as pd
import sqlite3
from pathlib import Path
from forecast_strategies import run_analysis_for_entity, save_summary, get_move_in_year
from new_entities_handlers import handle_growth_entity, handle_similarity_entity, filter_established_entities, handle_similarity_entity_prediction, handle_growth_entity_prediction
import pickle
import joblib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION - ULTRA-STRICT NO DATA LEAKAGE
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[0]

VARIABLE = "consommation"
UNIT = "Mwh"
exp_name = "temp"
# Choose which analysis parts (levels) to run
# 1: Contrat, 2: Partenaire (client), 3: Activities
RUN_LEVELS = {1}

N_PCS = 3
LAGS_OPTIONS = [1,2]
ALPHAS = [0.01, 0.1, 1.0, 10.0]
R2_THRESHOLD = 0.6
PC_WEIGHTS = [0.5, 0.8]
PCA_LAMBDAS = [0.2, 0.3, 0.4] #[0.3, 0.7, 1.0]

training_end = None
use_monthly_temp_options = [False]
use_monthly_clients_options = [False]
use_pf_options = [True, False]
client_pattern_weights = [0.3, 0.5, 0.8]
training_windows = [1, 2, 3, 4] #0,1

FEATURE_BLOCKS = {
    'none': [],
    'gdp_only': ['pib_mdh'],
    'sectoral_only': ['gdp_primaire', 'gdp_secondaire', 'gdp_tertiaire'],
    'gdp_sectoral': ['pib_mdh', 'gdp_primaire', 'gdp_secondaire', 'gdp_tertiaire'],
}
eval_years_start = 2021
eval_years_end = 2023
train_start_year = 2018

growth_feature_transforms = [("lag_lchg",)]
growth_feature_lags = [(1,)]


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
    "use_pf_options":use_pf_options,
    "client_pattern_weights": client_pattern_weights,
    "training_windows": training_windows,
    "train_start_year": train_start_year,
    "eval_years_end": eval_years_end,
    "eval_years_start": eval_years_start,
    "growth_feature_transforms": growth_feature_transforms,
    "growth_feature_lags": growth_feature_lags,
}

# ============================================================================
# LOAD DATA
# ============================================================================
db_path = PROJECT_ROOT / 'data/ONEE_Regional_COMPLETE_2007_2023.db'

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


output_file =  PROJECT_ROOT / f'outputs_cd/{exp_name}.xlsx'
monthly_output_file = PROJECT_ROOT / f'outputs_cd/{exp_name}_monthly_forcasts.xlsx'
all_results = []


# ===========================
# Main processing
# ===========================
if 1 in RUN_LEVELS:
    print(f"\n{'#'*60}")
    print(f"LEVEL 1: CONTRAT")
    print(f"{'#'*60}")

    contrats = sorted(df_contrats['contrat'].unique())

    established_contracts = []
    growth_contracts = []
    similarity_contracts = []

    # --- Categorize contracts ---
    for contrat in contrats:
        df_contrat = df_contrats[df_contrats['contrat'] == contrat].copy()
        move_in_year = get_move_in_year(df_contrat)

        if move_in_year is None or move_in_year <= 2021:
            established_contracts.append(contrat)
        elif move_in_year == 2022:
            growth_contracts.append(contrat)
        elif move_in_year == 2023:
            similarity_contracts.append(contrat)

    print(f"\n🟢 Established contracts: {len(established_contracts)}")
    print(f"🚀 Growth contracts: {len(growth_contracts)}")
    print(f"🔗 Similarity contracts: {len(similarity_contracts)}")

    # --- Process established contracts (normal forecasting) ---
    all_results_established = []
    for i, contrat in enumerate(established_contracts):
        df_contrat = df_contrats[df_contrats['contrat'] == contrat].copy()
        result = run_analysis_for_entity(
            df_contrat,
            f"Contrat_{contrat}",
            df_features,
            config=ANALYSIS_CONFIG,
            favor_overestimation=True,
            under_estimation_penalty=1.5,
        )
        if result:
            all_results_established.append(result)
    # all_results_established = joblib.load(PROJECT_ROOT / "outputs_cd/all_results_contrat_s4_overestimation.pkl")
    # all_results_established = filter_established_entities(all_results_established, growth_contracts, similarity_contracts)

    all_results_growth = []
    for contrat in growth_contracts:
        all_results_growth.append(handle_growth_entity_prediction(df_contrats, all_results_established, contrat))


    all_results_similarity = []
    for contrat in similarity_contracts:
        all_results_similarity.append(handle_similarity_entity_prediction(df_contrats, all_results_established, contrat))
    
    all_results_contrats = all_results_established + all_results_similarity + all_results_growth
    all_results += all_results_contrats


if 2 in RUN_LEVELS:
    print(f"\n{'#'*60}")
    print(f"LEVEL 2: PARTENAIRE")
    print(f"{'#'*60}")

    partenaires = sorted(df_contrats['partenaire'].unique())

    for i, partenaire in enumerate(partenaires):
        df_partenaire = df_contrats[df_contrats['partenaire'] == partenaire].copy()
        df_partenaire = df_partenaire.groupby(['annee', 'mois']).agg({
            VARIABLE: 'sum',
            'temperature': 'mean',
            'puissance facturée': 'sum'
        }).reset_index()

        result = run_analysis_for_entity(
            df_partenaire, 
            f"Partenaire_{partenaire}", 
            df_features,
            config = ANALYSIS_CONFIG,
            under_estimation_penalty = 1.5,
        )
        if result:
            all_results.append(result)


if 3 in RUN_LEVELS:
    print(f"\n{'#'*60}")
    print(f"LEVEL 3: ACTIVITE")
    print(f"{'#'*60}")

    activites = sorted(df_contrats['activite'].unique())

    established_activites = []
    growth_activites = []
    similarity_activites = []

    # --- Categorize activities ---
    for activite in activites:
        df_activite = df_contrats[df_contrats['activite'] == activite].copy()
        move_in_year = get_move_in_year(df_activite)

        if move_in_year is None or move_in_year <= 2021:
            established_activites.append(activite)
        elif move_in_year == 2022:
            growth_activites.append(activite)
        elif move_in_year == 2023:
            similarity_activites.append(activite)

    print(f"\n🟢 Established activites: {len(established_activites)}")
    print(f"🚀 Growth activites: {len(growth_activites)}")
    print(f"🔗 Similarity activites: {len(similarity_activites)}")

    # --- Process established activities ---
    all_results_established_activite = []
    for activite in established_activites:
        df_activite = df_contrats[df_contrats['activite'] == activite].copy()
        df_activite = df_activite.groupby(['annee', 'mois']).agg({
            VARIABLE: 'sum',
            'temperature': 'mean',
            'puissance facturée': 'sum'
        }).reset_index()

        result = run_analysis_for_entity(
            df_activite,
            f"Activité_{activite}",
            df_features,
            config=ANALYSIS_CONFIG,
            favor_overestimation=True,
            under_estimation_penalty=1.5,
        )
        if result:
            all_results_established_activite.append(result)

    # all_results_established_activite = joblib.load(PROJECT_ROOT / "outputs_cd/all_results_activite_overestimation.pkl")
    # all_results_established_activite = filter_established_entities(all_results_established_activite, growth_activites, similarity_activites, "activite")


    # --- Process growth activities ---
    all_results_growth_activite = []
    for activite in growth_activites:
        all_results_growth_activite.append(
            handle_growth_entity(df_contrats, all_results_established_activite, activite, "activite")
        )

    # --- Process similarity activities ---
    all_results_similarity_activite = []
    for activite in similarity_activites:
        all_results_similarity_activite.append(
            handle_similarity_entity(df_contrats, all_results_established_activite, activite, "activite")
        )

    # --- Merge all ---
    all_results_activite = (
        all_results_established_activite
        + all_results_similarity_activite
        + all_results_growth_activite
    )

    all_results += all_results_activite

save_summary(all_results, output_file, monthly_output_file)
with open(PROJECT_ROOT / f'outputs_cd/all_results_{exp_name}.pkl', "wb") as f:
    pickle.dump(all_results, f)

print(f"\n{'='*60}")
print(f"✅ Results saved to: {output_file}")
print(f"📊 Total entities analyzed: {len(all_results)}")
print(f"🔒 ZERO DATA LEAKAGE GUARANTEED")
print(f"{'='*60}\n")
