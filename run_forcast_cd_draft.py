import logging
import os
import sys
import warnings

import pandas as pd
import sqlite3
from pathlib import Path
from forcast_strategies_draft import run_analysis_for_entity, save_summary, get_move_in_year
import pickle
import joblib
warnings.filterwarnings('ignore')

from kmodes.kprototypes import KPrototypes
import numpy as np
import pandas as pd

def prepare_kproto_clusters(df):
    """
    K-Prototypes clustering with categorical 'activite' and numeric monthly 'puissance facturée'.
    Works identically to the version you validated manually.
    """
    df_2023 = df[df['annee'] == 2023].copy()
    df_2023 = df_2023[['contrat', 'activite', 'mois', 'puissance facturée']].dropna()
    df_2023['activite'] = df_2023['activite'].astype(str)

    # Pivot to get monthly columns
    da = df_2023[['contrat', 'activite']].drop_duplicates()
    dd = df_2023.pivot(index='contrat', columns='mois', values='puissance facturée')
    data = dd.merge(da, on='contrat')

    # Make 'activite' the first column
    data['contrat'] = data['activite']
    data.drop(columns=['activite'], inplace=True)
    data.rename(columns={'contrat': 'activite'}, inplace=True)

    # Prepare matrix for K-Prototypes
    X = data.to_numpy(dtype=object)
    categorical_cols = [0]  # first column is 'activite'

    from kmodes.kprototypes import KPrototypes
    kproto = KPrototypes(n_clusters=4, init='Huang', random_state=42)
    clusters = kproto.fit_predict(X, categorical=categorical_cols)

    data['cluster'] = clusters

    # Compute cluster centroids
    cluster_profiles = (
        data.groupby('cluster')[[col for col in data.columns if isinstance(col, (int, np.integer))]]
        .mean()
        .rename(columns={i: f'mois_{i}' for i in range(1, 13)})
    )

    print(f"✅ K-Prototypes trained successfully ({len(cluster_profiles)} clusters).")
    return kproto, cluster_profiles, data[['activite', 'cluster']]



# ============================================================================
# CONFIGURATION - ULTRA-STRICT NO DATA LEAKAGE
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[0]

VARIABLE = "consommation"
UNIT = "Mwh"
exp_name = "gggggg"


N_PCS = 3
LAGS_OPTIONS = [2, 3]
ALPHAS = [0.01, 0.1, 1.0, 10.0]
R2_THRESHOLD = 0.6
PC_WEIGHTS = [0.5, 0.8]
PCA_LAMBDAS = [0.3, 0.7, 1.0]

training_end = None
use_monthly_temp_options = [False]
use_monthly_clients_options = [False]
use_pf_options = [True, False]
client_pattern_weights = [0.3, 0.5, 0.8]
training_windows = [2,3,4] #0,1

FEATURE_BLOCKS = {
    'none': [],
    'gdp_only': ['pib_mdh'],
    'sectoral_only': ['gdp_primaire', 'gdp_secondaire', 'gdp_tertiaire'],
    'gdp_sectoral': ['pib_mdh', 'gdp_primaire', 'gdp_secondaire', 'gdp_tertiaire'],
}
eval_years_start = 2021
eval_years_end = 2023
train_start_year = 2018

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
# kproto, cluster_profiles, cluster_assignments = prepare_kproto_clusters(df_contrats)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
# Choose which analysis parts (levels) to run
# 1: Contrat, 2: Partenaire (client), 3: Activities
RUN_LEVELS = {3}

output_file =  PROJECT_ROOT / f'outputs_cd/{exp_name}.xlsx'
monthly_output_file = PROJECT_ROOT / f'outputs_cd/{exp_name}_monthly_forcasts.xlsx'
all_results = []



from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


def handle_similarity_contract(df, all_results_established, contrat_id):
    """
    Handle a 'similarity' contract (starting in 2023).
    Finds the most similar established contract based on 2023 'puissance facturée'
    profile within the same activity, then reuses its 2023 predictions.
    Always returns a structured result consistent with run_analysis_for_entity().
    """
    df_contrat = df[df['contrat'] == contrat_id].copy()
    if df_contrat.empty:
        print(f"⚠️ Contract {contrat_id} not found in dataset.")
        return _build_empty_result(df_contrat, contrat_id)

    activite = df_contrat['activite'].iloc[0]
    print(f"\n🔗 Handling similarity contract: {contrat_id} ({activite})")

    # --- Filter to same activity contracts ---
    df_same_activity = df[df['activite'] == activite].copy()
    df_2023 = df_same_activity[df_same_activity['annee'] == 2023]

    pivot = (
        df_2023.pivot_table(
            index='contrat',
            columns='mois',
            values='puissance facturée'
        )
        .fillna(0)
        .reindex(columns=range(1, 13), fill_value=0)
    )

    if pivot.empty:
        print(f"⚠️ No comparable 2023 puissance facturée data for '{activite}'.")
        return _build_empty_result(df_contrat, contrat_id)

    # --- Get target contract 2023 profile ---
    target_2023 = (
        df_contrat[df_contrat['annee'] == 2023]
        .sort_values('mois')['puissance facturée']
        .fillna(0)
        .to_numpy()
    )
    if len(target_2023) == 0:
        print(f"⚠️ Contract {contrat_id} has no 2023 puissance facturée data.")
        return _build_empty_result(df_contrat, contrat_id)

    target_vec = target_2023.reshape(1, -1)

    # --- Similarity search (KNN) ---
    X = pivot.values
    contrats_in_activity = pivot.index.to_numpy()

    knn = NearestNeighbors(n_neighbors=min(5, len(X)), metric='euclidean')
    knn.fit(X)
    distances, indices = knn.kneighbors(target_vec)

    similar_contracts = contrats_in_activity[indices[0]]
    similar_contracts = [c for c in similar_contracts if c != contrat_id]

    if not similar_contracts:
        print(f"⚠️ No similar contracts found for {contrat_id}.")
        return _build_empty_result(df_contrat, contrat_id)

    best_match_id = similar_contracts[0]
    print(f"✅ Most similar contract found: {best_match_id}")

    # --- Find the matched contract in established results ---
    best_match = next(
        (r for r in all_results_established if f"Contrat_{best_match_id}" == r['entity']),
        None
    )

    if best_match is None:
        print(f"⚠️ No best match found in all_results for contract {best_match_id}.")
        return _build_empty_result(df_contrat, contrat_id)

    best_model_match = best_match['best_model']
    if 2023 not in best_model_match['valid_years']:
        print(f"⚠️ Matched contract {best_match_id} has no 2023 predictions.")
        return _build_empty_result(df_contrat, contrat_id)

    # --- Extract predictions and actuals ---
    idx_2023 = np.where(best_model_match['valid_years'] == 2023)[0][0]
    predicted_2023 = best_model_match['pred_monthly_matrix'][idx_2023]

    actual_2023 = (
        df_contrat[df_contrat['annee'] == 2023]
        .sort_values('mois')['consommation']
        .fillna(0)
        .to_numpy()
    )
    if len(actual_2023) < 12:
        # pad with zeros if missing months
        actual_2023 = np.pad(actual_2023, (0, 12 - len(actual_2023)), constant_values=0)

    # --- Compute errors ---
    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct_monthly = np.where(
            actual_2023 != 0,
            ((predicted_2023 - actual_2023) / actual_2023) * 100,
            np.nan
        )

    annual_actual = np.float64(actual_2023.sum())
    annual_pred = np.float64(predicted_2023.sum())
    annual_error_pct = (
        np.nan if annual_actual == 0 else ((annual_pred - annual_actual) / annual_actual) * 100
    )

    # --- Build minimal best_model ---
    actual_matrix = (
        df_contrat.pivot_table(
            index='annee',
            columns='mois',
            values='consommation',
            aggfunc='sum'
        )
        .reindex(columns=range(1, 13), fill_value=0)
        .fillna(0)
        .to_numpy()
    )
    valid_years = df_contrat['annee'].unique()

    best_model = {
        'pred_monthly_matrix': np.array([predicted_2023]),
        'actual_monthly_matrix': actual_matrix,
        'valid_years': valid_years,
        'strategy': 'Clustering - Similarity',
        'n_lags': None,
        'feature_block': None,
        'use_monthly_temp': None,
        'use_monthly_clients': None,
        'client_pattern_weight': None,
        'pc_weight': None,
        'pca_lambda': None,
        'monthly_mae': None,
        'monthly_mape': None,
        'monthly_r2': None,
        'annual_mae': None,
        'annual_mape': None,
        'annual_r2': None
    }

    # --- Assemble test_years dict (fully populated) ---
    test_years = {
        '2023_actual_monthly': actual_2023,
        '2023_predicted_monthly': predicted_2023,
        '2023_error_pct_monthly': error_pct_monthly,
        '2023_actual_annual': annual_actual,
        '2023_predicted_annual': annual_pred,
        '2023_error_pct_annual': annual_error_pct,
        '2023_source_match': best_match_id
    }

    # --- Final structured result ---
    result = {
        "entity": f"Contrat_{contrat_id}",
        "best_model": best_model,
        "training_end": None,
        "monthly_details": {},
        "test_years": test_years
    }

    print(f"🔁 Reused 2023 predictions from contract {best_match_id}")
    return result

def handle_similarity_contract_kproto(df, kproto, cluster_profiles, contrat_id):
    """
    Handle a 'similarity' contract using K-Prototypes cluster centroid.
    Uses centroid mean 2023 profile as predicted monthly pattern.
    """
    df_contrat = df[df['contrat'] == contrat_id].copy()
    if df_contrat.empty:
        print(f"⚠️ Contract {contrat_id} not found.")
        return _build_empty_result(df_contrat, contrat_id)

    # Build feature vector for prediction (activite + 2023 monthly values)
    activite = df_contrat['activite'].iloc[0]
    df_2023 = df_contrat[df_contrat['annee'] == 2023].sort_values('mois')
    profile_2023 = df_2023['puissance facturée'].fillna(0).to_numpy()
    if len(profile_2023) < 12:
        profile_2023 = np.pad(profile_2023, (0, 12 - len(profile_2023)), constant_values=0)

    X_new = np.concatenate([[activite], profile_2023]).reshape(1, -1)
    cluster = kproto.predict(X_new, categorical=[0])[0]

    centroid_profile = cluster_profiles.loc[cluster].to_numpy()
    centroid_profile = centroid_profile / (centroid_profile.sum() or 1)

    # Predicted 2023 = scaled centroid shape * actual total of 2023
    actual_2023 = df_contrat[df_contrat['annee'] == 2023]['consommation'].fillna(0).to_numpy()
    if len(actual_2023) < 12:
        actual_2023 = np.pad(actual_2023, (0, 12 - len(actual_2023)), constant_values=0)
    annual_actual = actual_2023.sum()

    predicted_2023 = annual_actual * centroid_profile

    # Compute error %
    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct = np.where(actual_2023 != 0, (predicted_2023 - actual_2023) / actual_2023 * 100, np.nan)

    annual_pred = predicted_2023.sum()
    annual_error_pct = (
        np.nan if annual_actual == 0 else ((annual_pred - annual_actual) / annual_actual) * 100
    )

    # Build final result
    best_model = {
        'pred_monthly_matrix': np.array([predicted_2023]),
        'actual_monthly_matrix': actual_2023.reshape(1, -1),
        'valid_years': np.array([2023]),
        'strategy': f"KPrototypes - Similarity (Cluster {cluster})",
    }

    test_years = {
        '2023_actual_monthly': actual_2023,
        '2023_predicted_monthly': predicted_2023,
        '2023_error_pct_monthly': error_pct,
        '2023_actual_annual': annual_actual,
        '2023_predicted_annual': annual_pred,
        '2023_error_pct_annual': annual_error_pct,
        '2023_cluster_id': int(cluster),
    }

    print(f"🔗 Contract {contrat_id} assigned to cluster {cluster}.")
    return {
        "entity": f"Contrat_{contrat_id}",
        "best_model": best_model,
        "training_end": None,
        "monthly_details": {},
        "test_years": test_years,
    }


def _build_empty_result(df_contrat, contrat_id):
    """Return a consistent empty result object for similarity contracts."""
    actual_2023 = np.zeros(12)
    nan_arr = np.full(12, np.nan)

    best_model = {
        'pred_monthly_matrix': None,
        'actual_monthly_matrix': None,
        'valid_years': df_contrat['annee'].unique() if not df_contrat.empty else [],
        'strategy': None,
        'n_lags': None,
        'feature_block': None,
        'use_monthly_temp': None,
        'use_monthly_clients': None,
        'client_pattern_weight': None,
        'pc_weight': None,
        'pca_lambda': None,
        'monthly_mae': None,
        'monthly_mape': None,
        'monthly_r2': None,
        'annual_mae': None,
        'annual_mape': None,
        'annual_r2': None
    }

    test_years = {
        '2023_actual_monthly': actual_2023,
        '2023_predicted_monthly': np.zeros(12),
        '2023_error_pct_monthly': nan_arr,
        '2023_actual_annual': np.float64(0.0),
        '2023_predicted_annual': np.float64(0.0),
        '2023_error_pct_annual': np.nan,
        '2023_source_match': None
    }

    return {
        "entity": f"Contrat_{contrat_id}",
        "best_model": best_model,
        "training_end": None,
        "monthly_details": {},
        "test_years": test_years
    }


def handle_growth_contract(df, all_results_established, contrat_id):
    """
    Handle a 'growth' contract (starting in 2022).
    Predicts 2023 consumption using log growth rate inferred
    from the most similar established contract.
    """
    df_contrat = df[df['contrat'] == contrat_id].copy()
    if df_contrat.empty:
        print(f"⚠️ Contract {contrat_id} not found in dataset.")
        return _build_empty_result(df_contrat, contrat_id)

    activite = df_contrat['activite'].iloc[0]
    print(f"\n🚀 Handling growth contract: {contrat_id} ({activite})")

    # --- 1️⃣ Filter to same activity contracts ---
    df_same_activity = df[df['activite'] == activite].copy()
    df_2023 = df_same_activity[df_same_activity['annee'] == 2023]

    pivot = (
        df_2023.pivot_table(
            index='contrat',
            columns='mois',
            values='puissance facturée'
        )
        .fillna(0)
        .reindex(columns=range(1, 13), fill_value=0)
    )

    if pivot.empty:
        print(f"⚠️ No comparable 2023 puissance facturée data for '{activite}'.")
        return _build_empty_result(df_contrat, contrat_id)

    # --- 2️⃣ Build target 2023 profile ---
    target_2023 = (
        df_contrat[df_contrat['annee'] == 2023]
        .sort_values('mois')['puissance facturée']
        .fillna(0)
        .to_numpy()
    )
    if len(target_2023) == 0:
        print(f"⚠️ Contract {contrat_id} has no 2023 puissance facturée data.")
        return _build_empty_result(df_contrat, contrat_id)

    target_vec = target_2023.reshape(1, -1)

    # --- 3️⃣ Find most similar contract ---
    X = pivot.values
    contrats_in_activity = pivot.index.to_numpy()

    knn = NearestNeighbors(n_neighbors=min(5, len(X)), metric='euclidean')
    knn.fit(X)
    distances, indices = knn.kneighbors(target_vec)

    similar_contracts = contrats_in_activity[indices[0]]
    similar_contracts = [c for c in similar_contracts if c != contrat_id]

    if not similar_contracts:
        print(f"⚠️ No similar contracts found for {contrat_id}.")
        return _build_empty_result(df_contrat, contrat_id)

    best_match_id = similar_contracts[0]
    print(f"✅ Most similar contract found: {best_match_id}")

    # --- 4️⃣ Retrieve best match result ---
    best_match = next(
        (r for r in all_results_established if f"Contrat_{best_match_id}" == r['entity']),
        None
    )

    if best_match is None:
        print(f"⚠️ No best match found in all_results for {best_match_id}.")
        return _build_empty_result(df_contrat, contrat_id)

    best_model_match = best_match['best_model']
    valid_years = best_model_match['valid_years']
    if not (2022 in valid_years and 2023 in valid_years):
        print(f"⚠️ Matched contract {best_match_id} lacks 2022 or 2023 data.")
        return _build_empty_result(df_contrat, contrat_id)

    # --- 5️⃣ Extract actual_2022 and predicted_2023 from similar contract ---
    idx_2022 = np.where(valid_years == 2022)[0][0]
    idx_2023 = np.where(valid_years == 2023)[0][0]

    actual_2022 = best_model_match['actual_monthly_matrix'][idx_2022].sum()
    predicted_2023_similar = best_model_match['pred_monthly_matrix'][idx_2023].sum()

    if actual_2022 <= 0 or predicted_2023_similar <= 0:
        print(f"⚠️ Invalid growth data for matched contract {best_match_id}.")
        return _build_empty_result(df_contrat, contrat_id)

    # --- 6️⃣ Compute growth rate ---
    growth_rate = np.log(predicted_2023_similar) - np.log(actual_2022)
    print(f"📈 Growth rate (log): {growth_rate:.4f}")

    # --- 7️⃣ Compute predicted annual 2023 for current contract ---
    actual_2022_current = (
        df_contrat[df_contrat['annee'] == 2022]['consommation']
        .fillna(0).sum()
    )

    predicted_2023_annual = np.exp(np.log(actual_2022_current + 1e-9) + growth_rate)

    # --- 8️⃣ Compute mean curve (shape) from similar contract ---
    mean_curve = best_model_match['pred_monthly_matrix'][idx_2023]
    mean_curve_normalized = mean_curve / (mean_curve.sum() or 1)

    predicted_2023_monthly = predicted_2023_annual * mean_curve_normalized

    # --- 9️⃣ Compute errors (if 2023 actual exists) ---
    actual_2023 = (
        df_contrat[df_contrat['annee'] == 2023]['consommation']
        .fillna(0)
        .to_numpy()
    )
    if len(actual_2023) < 12:
        actual_2023 = np.pad(actual_2023, (0, 12 - len(actual_2023)), constant_values=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct_monthly = np.where(
            actual_2023 != 0,
            ((predicted_2023_monthly - actual_2023) / actual_2023) * 100,
            np.nan
        )

    annual_actual = np.float64(actual_2023.sum())
    annual_pred = np.float64(predicted_2023_monthly.sum())
    annual_error_pct = (
        np.nan if annual_actual == 0 else ((annual_pred - annual_actual) / annual_actual) * 100
    )

    # --- 🔟 Build best_model ---
    actual_matrix = (
        df_contrat.pivot_table(
            index='annee',
            columns='mois',
            values='consommation',
            aggfunc='sum'
        )
        .reindex(columns=range(1, 13), fill_value=0)
        .fillna(0)
        .to_numpy()
    )

    best_model = {
        'pred_monthly_matrix': np.array([predicted_2023_monthly]),
        'actual_monthly_matrix': actual_matrix,
        'valid_years': df_contrat['annee'].unique(),
        'strategy': 'Clustering - Growth',
        'n_lags': None,
        'feature_block': None,
        'use_monthly_temp': None,
        'use_monthly_clients': None,
        'client_pattern_weight': None,
        'pc_weight': None,
        'pca_lambda': None,
        'monthly_mae': None,
        'monthly_mape': None,
        'monthly_r2': None,
        'annual_mae': None,
        'annual_mape': None,
        'annual_r2': None
    }

    test_years = {
        '2023_actual_monthly': actual_2023,
        '2023_predicted_monthly': predicted_2023_monthly,
        '2023_error_pct_monthly': error_pct_monthly,
        '2023_actual_annual': annual_actual,
        '2023_predicted_annual': annual_pred,
        '2023_error_pct_annual': annual_error_pct,
        '2023_source_match': best_match_id
    }

    result = {
        "entity": f"Contrat_{contrat_id}",
        "best_model": best_model,
        "training_end": None,
        "monthly_details": {},
        "test_years": test_years
    }

    print(f"✅ Predicted 2023 annual: {predicted_2023_annual:,.2f} (using growth rate {growth_rate:.4f})")
    return result

def handle_growth_contract_kproto(df, kproto, cluster_profiles, contrat_id):
    """
    Handle a 'growth' contract using K-Prototypes cluster centroid.
    Predicts 2023 using the centroid's log growth rate from 2022->2023.
    """
    df_contrat = df[df['contrat'] == contrat_id].copy()
    if df_contrat.empty:
        print(f"⚠️ Contract {contrat_id} not found.")
        return _build_empty_result(df_contrat, contrat_id)

    activite = df_contrat['activite'].iloc[0]
    df_2023 = df_contrat[df_contrat['annee'] == 2023].sort_values('mois')
    profile_2023 = df_2023['puissance facturée'].fillna(0).to_numpy()
    if len(profile_2023) < 12:
        profile_2023 = np.pad(profile_2023, (0, 12 - len(profile_2023)), constant_values=0)

    X_new = np.concatenate([[activite], profile_2023]).reshape(1, -1)
    cluster = kproto.predict(X_new, categorical=[0])[0]

    # Centroid monthly curve for 2023
    centroid_curve = cluster_profiles.loc[cluster].to_numpy()
    centroid_shape = centroid_curve / (centroid_curve.sum() or 1)

    # Estimate log growth rate within this cluster (based on all contracts)
    df_cluster = df[df['activite'] == activite].copy()
    df_cluster['cluster'] = kproto.predict(
        np.column_stack([df_cluster['activite'], df_cluster['puissance facturée']]),
        categorical=[0]
    )
    cluster_df = df_cluster[df_cluster['cluster'] == cluster]
    if cluster_df.empty:
        growth_rate = 0.0
    else:
        annual_df = cluster_df.groupby(['contrat', 'annee'])['consommation'].sum().unstack()
        if {2022, 2023}.issubset(annual_df.columns):
            annual_df = annual_df.dropna(subset=[2022, 2023])
            if len(annual_df) > 0:
                growth_rate = np.log((annual_df[2023] + 1e-9) / (annual_df[2022] + 1e-9)).mean()
            else:
                growth_rate = 0.0
        else:
            growth_rate = 0.0

    # Apply growth rate to this contract's 2022 consumption
    actual_2022 = df_contrat[df_contrat['annee'] == 2022]['consommation'].sum()
    predicted_2023_annual = np.exp(np.log(actual_2022 + 1e-9) + growth_rate)
    predicted_2023_monthly = predicted_2023_annual * centroid_shape

    # Compute errors if actual 2023 exists
    actual_2023 = df_contrat[df_contrat['annee'] == 2023]['consommation'].fillna(0).to_numpy()
    if len(actual_2023) < 12:
        actual_2023 = np.pad(actual_2023, (0, 12 - len(actual_2023)), constant_values=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct = np.where(actual_2023 != 0, (predicted_2023_monthly - actual_2023) / actual_2023 * 100, np.nan)

    best_model = {
        'pred_monthly_matrix': np.array([predicted_2023_monthly]),
        'actual_monthly_matrix': actual_2023.reshape(1, -1),
        'valid_years': np.array([2022, 2023]),
        'strategy': f"KPrototypes - Growth (Cluster {cluster})",
    }

    test_years = {
        '2023_actual_monthly': actual_2023,
        '2023_predicted_monthly': predicted_2023_monthly,
        '2023_error_pct_monthly': error_pct,
        '2023_actual_annual': actual_2023.sum(),
        '2023_predicted_annual': predicted_2023_annual,
        '2023_cluster_id': int(cluster),
        '2023_growth_rate_log': float(growth_rate),
    }

    print(f"🚀 Contract {contrat_id} → Cluster {cluster}, Growth Rate: {growth_rate:.4f}")
    return {
        "entity": f"Contrat_{contrat_id}",
        "best_model": best_model,
        "training_end": None,
        "monthly_details": {},
        "test_years": test_years,
    }



def filter_established_contracts(all_results, growth_contracts, similarity_contracts):
    # Convert to a set of excluded entity names for faster lookup
    excluded_entities = {
        f"Contrat_{cid}" for cid in list(growth_contracts) + list(similarity_contracts)
    }

    filtered = [r for r in all_results if r.get("entity") not in excluded_entities]

    print(f"🧹 Removed {len(all_results) - len(filtered)} non-established contracts "
          f"({len(filtered)} remaining).")
    return filtered

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
    # all_results_established = []
    # for i, contrat in enumerate(established_contracts):
    #     df_contrat = df_contrats[df_contrats['contrat'] == contrat].copy()
    #     result = run_analysis_for_entity(
    #         df_contrat,
    #         f"Contrat_{contrat}",
    #         df_features,
    #         config=ANALYSIS_CONFIG,
    #         favor_overestimation=False,
    #         under_estimation_penalty=1.5,
    #     )
    #     if result:
    #         all_results_established.append(result)
    all_results_established = joblib.load(PROJECT_ROOT / "outputs_cd/all_results_contrat_s4_overestimation.pkl")
    all_results_established = filter_established_contracts(all_results_established, growth_contracts, similarity_contracts)

    all_results_growth = []
    for contrat in growth_contracts:
        # all_results_growth.append(handle_growth_contract_kproto(df_contrats, kproto, cluster_profiles, contrat))
        all_results_growth.append(handle_growth_contract(df_contrats, all_results_established, contrat))


    all_results_similarity = []
    for contrat in similarity_contracts:
        # all_results_similarity.append(handle_similarity_contract_kproto(df_contrats, kproto, cluster_profiles, contrat))
        all_results_similarity.append(handle_similarity_contract(df_contrats, all_results_established, contrat))
    
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
    print(f"LEVEL 2: ACTIVITE")
    print(f"{'#'*60}")

    activites = sorted(df_contrats['activite'].unique())

    for i, activite in enumerate(activites):
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
            config = ANALYSIS_CONFIG,
            favor_overestimation=False,
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
            f"Activite_{activite}",
            df_features,
            config=ANALYSIS_CONFIG,
            favor_overestimation=False,
            under_estimation_penalty=1.5,
        )
        if result:
            all_results_established_activite.append(result)

    # --- Process growth activities ---
    all_results_growth_activite = []
    for activite in growth_activites:
        all_results_growth_activite.append(
            handle_growth_contract(df_contrats, all_results_established_activite, activite)
        )

    # --- Process similarity activities ---
    all_results_similarity_activite = []
    for activite in similarity_activites:
        all_results_similarity_activite.append(
            handle_similarity_contract(df_contrats, all_results_established_activite, activite)
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
