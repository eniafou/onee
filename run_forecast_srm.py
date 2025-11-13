import warnings
warnings.filterwarnings('ignore')

import sqlite3
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from forecast_strategies import run_analysis_for_entity, save_summary

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG AREA
# ────────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[0]

UNIT = "Kwh"
VARIABLE = "consommation_kwh"              # supported examples: "nbr_clients", "consommation_kwh"
exp_name = "2017"

N_PCS = 3
LAGS_OPTIONS = [2, 3]
ALPHAS = [0.1, 1.0, 10.0]
R2_THRESHOLD = 0.6
PC_WEIGHTS = [0.2, 0.5]
PCA_LAMBDAS = [0.3, 0.7, 1.0]

training_end = None
use_monthly_temp_options = [False]
use_monthly_clients_options = [False] #[True, False]
client_pattern_weights = [0.3, 0.5, 0.8]
training_windows = [4, 7, 10]
FEATURE_BLOCKS = {
    'none': [],
    'gdp_only': ['pib_mdh'],
    'sectoral_only': ['gdp_primaire', 'gdp_secondaire', 'gdp_tertiaire'],
    'gdp_sectoral': ['pib_mdh', 'gdp_primaire', 'gdp_secondaire', 'gdp_tertiaire'],
}
eval_years_start = 2015
eval_years_end = 2017
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
    "client_pattern_weights": client_pattern_weights,
    "training_windows": training_windows,
    "eval_years_end": eval_years_end,
    "eval_years_start": eval_years_start,
    "growth_feature_transforms": growth_feature_transforms,
    "growth_feature_lags": growth_feature_lags,
}



# ────────────────────────────────────────────────────────────────────────────────
# Predefined query & column specs per VARIABLE
# ────────────────────────────────────────────────────────────────────────────────
VARIABLE_SPECS = {
    # Regional source: monthly_data (ONEE_Regional_COMPLETE.db)
    "nbr_clients": {
        "regional_select_expr": '"Nbr Clients" as nbr_clients',
        "regional_var_col": "nbr_clients",
        # Distributors do not provide client counts → disable
        "distributor_supported": False,
        "distributor_select_expr": None,
        "distributor_var_col": None,
    },
    "consommation_kwh": {
        "regional_select_expr": 'MWh * 1000 as consommation_kwh',
        "regional_var_col": "consommation_kwh",
        # Distributors source: consumption (ONEE_Distributeurs_consumption.db)
        "distributor_supported": True,
        "distributor_select_expr": 'SUM(consumption_value) as consommation_kwh',
        "distributor_var_col": "consommation_kwh",
    },
}


def load_client_prediction_lookup(region_name: str) -> dict[str, dict[int, np.ndarray]]:
    """
    Load monthly nbr_clients predictions produced by the dedicated pipeline.
    Returns a dict mapping entity name -> {year: np.ndarray(shape (12,))}
    """
    results_path = PROJECT_ROOT / f"outputs_srm/nbr_clients/all_results_{clean_name(region_name)}_nbr_clients.pkl"
    if not results_path.exists():
        print(f"⚠️  No precomputed client predictions found at {results_path}.")
        return {}

    try:
        with results_path.open("rb") as fp:
            raw_results = pickle.load(fp)
    except Exception as exc:
        print(f"⚠️  Failed to load client predictions: {exc}")
        return {}

    lookup: dict[str, dict[int, np.ndarray]] = {}
    for entry in raw_results:
        entity_name = entry.get("entity")
        if entity_name is None:
            continue
        model_info = entry.get("best_model") or entry
        valid_years = model_info.get("valid_years")
        pred_matrix = model_info.get("pred_monthly_matrix")
        if valid_years is None or pred_matrix is None:
            continue

        entity_preds: dict[int, np.ndarray] = {}
        for idx, year in enumerate(valid_years):
            try:
                year_int = int(year)
                monthly_values = np.asarray(pred_matrix[idx], dtype=float).reshape(-1)
            except Exception:
                continue
            if monthly_values.size != 12:
                continue
            entity_preds[year_int] = monthly_values.copy()

        if entity_preds:
            lookup[entity_name] = entity_preds

    return lookup


def aggregate_predictions(
    lookup: dict[str, dict[int, np.ndarray]],
    target_name: str,
    component_names: list[str],
) -> None:
    """
    Combine monthly predictions from `component_names` and store them in `lookup[target_name]`.
    Missing components are skipped; only years with at least one component prediction are kept.
    """
    combined: dict[int, np.ndarray] = {}

    for component in component_names:
        component_preds = lookup.get(component)
        if not component_preds:
            continue

        for year, values in component_preds.items():
            combined.setdefault(year, np.zeros(12, dtype=float))
            combined[year] = combined[year] + np.asarray(values, dtype=float)

    if combined:
        lookup[target_name] = combined


def get_queries_for(variable: str, target_region: str):
    """
    Return (query_regional, query_dist, query_features, var_cols)
    - Any query can be None if not needed/supported for this VARIABLE.
    """
    if variable not in VARIABLE_SPECS:
        raise ValueError(f"Unsupported VARIABLE='{variable}'. Available: {list(VARIABLE_SPECS)}")
    spec = VARIABLE_SPECS[variable]

    query_regional_mt = f"""
        SELECT
            Year  as annee,
            Month as mois,
            Activity as activite,
            {spec['regional_select_expr']}
        FROM monthly_data_mt
        WHERE Region = '{target_region}'
        ORDER BY Year, Month, Activity
    """

    query_regional_bt = f"""
        SELECT
            Year  as annee,
            Month as mois,
            Activity as activite,
            {spec['regional_select_expr']}
        FROM monthly_data_bt
        WHERE Region = '{target_region}'
        ORDER BY Year, Month, Activity
    """

    query_dist = None
    if spec["distributor_supported"]:
        query_dist = f"""
            SELECT
                year as annee,
                month as mois,
                distributeur,
                {spec['distributor_select_expr']}
            FROM consumption
            WHERE region = '{target_region}'
            GROUP BY year, month, distributeur
            ORDER BY year, month, distributeur
        """

    query_features = f"""
        SELECT
            Year as annee,
            AVG(GDP_Millions_DH) as pib_mdh,
            AVG(GDP_Primaire)    as gdp_primaire,
            AVG(GDP_Secondaire)  as gdp_secondaire,
            AVG(GDP_Tertiaire)   as gdp_tertiaire,
            AVG(temp)            as temperature_annuelle
        FROM regional_features
        WHERE Region = '{target_region}'
        GROUP BY Year
    """

    var_cols = {
        "regional": spec["regional_var_col"],
        "distributor": spec["distributor_var_col"],
    }
    return query_regional_mt, query_regional_bt, query_dist, query_features, var_cols


def require_columns(df: pd.DataFrame, cols: list[str], ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {ctx}: {missing}")


def clean_name(name: str) -> str:
    return (
        name.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

# ────────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    REGIONS = ["Béni Mellal-Khénifra", "Casablanca-Settat", "Drâa-Tafilalet", "Fès-Meknès", "Laâyoune-Sakia El Hamra", "Marrakech-Safi", "Oriental", "Rabat-Salé-Kénitra", "Tanger-Tétouan-Al Hoceïma", "Souss-Massa"]

    # Choose which analysis parts (levels) to run
    # 1: Activities, 2: Aggregated BT, 3: Aggregated MT, 4: Total Regional,
    # 5: Individual Distributors, 6: All Distributors, 7: SRM (Regional+Dist)
    # RUN_LEVELS = {1, 4, 5, 6, 7}
    RUN_LEVELS = {1,4,5,6,7}

    for TARGET_REGION in REGIONS:
        print(f"Loading data for {TARGET_REGION}...\n")

        # DB paths
        db_regional_path = PROJECT_ROOT / 'data/ONEE_Regional_COMPLETE_2007_2023.db'
        db_dist_path     = PROJECT_ROOT / 'data/ONEE_Distributeurs_consumption.db'

        db_regional = sqlite3.connect(db_regional_path)
        db_dist     = sqlite3.connect(db_dist_path)

        # Predefined queries for chosen VARIABLE
        q_regional_mt, q_regional_bt, q_dist, q_features, var_cols = get_queries_for(VARIABLE, TARGET_REGION)

        # Load data (regional + features are always needed)
        df_regional_mt = pd.read_sql_query(q_regional_mt, db_regional)
        df_regional_bt = pd.read_sql_query(q_regional_bt, db_regional)
        df_features = pd.read_sql_query(q_features, db_regional)

        # Optional distributors (None if unsupported by VARIABLE)
        df_dist = pd.read_sql_query(q_dist, db_dist) if q_dist is not None else None

        db_regional.close()
        db_dist.close()

        df_regional_mt['activite'] = df_regional_mt['activite'].replace("Administratif", "Administratif_mt")
        df_regional = pd.concat([df_regional_bt, df_regional_mt])

        # Basic checks & cleaning
        reg_var_col = var_cols["regional"]
        require_columns(df_regional, ["annee", "mois", "activite", reg_var_col], "df_regional")
        df_regional = df_regional.copy()
        df_regional[reg_var_col] = df_regional[reg_var_col].fillna(0)

        if df_dist is not None:
            dist_var_col = var_cols["distributor"]
            require_columns(df_dist, ["annee", "mois", "distributeur", dist_var_col], "df_dist")
            df_dist = df_dist.copy()
            df_dist[dist_var_col] = df_dist[dist_var_col].fillna(0)

        print(f"\n{'#'*60}")
        print(f"ULTRA-STRICT MODE: PCA REFITTED IN EACH LOOCV FOLD")
        print(f"{'#'*60}")

        # Only load client predictions if we're going to use them
        client_predictions_lookup = {}
        if True in use_monthly_clients_options:
            client_predictions_lookup = load_client_prediction_lookup(TARGET_REGION)
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
                    under_estimation_penalty = 2,
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
                aggregate_predictions(
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
                    under_estimation_penalty = 2,
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
                aggregate_predictions(
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
                    under_estimation_penalty = 2,
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
            aggregate_predictions(
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
                under_estimation_penalty = 3,
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
                        under_estimation_penalty = 3,
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
                    under_estimation_penalty = 3,
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
                    under_estimation_penalty = 3,
                )
                _maybe_add(res)

        # Create output directories
        output_dir = PROJECT_ROOT / 'outputs_srm' / exp_name / clean_name(TARGET_REGION)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save outputs
        output_file = output_dir / f'{clean_name(TARGET_REGION)}_{VARIABLE}_{exp_name}.xlsx'
        monthly_book_file = output_dir / f'monthly_predictions_by_entity_{clean_name(TARGET_REGION)}_{VARIABLE}_{exp_name}.xlsx'
        save_summary(all_results, output_file, monthly_book_file)

        with open(output_dir / f'all_results_{clean_name(TARGET_REGION)}_{VARIABLE}_{exp_name}.pkl', "wb") as f:
            pickle.dump(all_results, f)

        print(f"\n{'='*60}")
        print(f"✅ Results saved to: {output_file}")
        print(f"📊 Total entities analyzed: {len(all_results)}")
        print(f"🔒 ZERO DATA LEAKAGE GUARANTEED")
        print(f"{'='*60}\n")
