import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from onee.utils import get_move_in_year
from onee.data.names import Aliases
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def _build_monthly_details(prediction_year, actual_monthly, predicted_monthly, error_pct_monthly):
    """Build monthly_details dict matching run_analysis_for_entity format.
    
    Returns a dict: {prediction_year: DataFrame with Month, Actual, Predicted, Error_%}
    """
    months = np.arange(1, 13)
    data = {
        "Month": months,
        "Actual": np.asarray(actual_monthly, dtype=float),
        "Predicted": np.asarray(predicted_monthly, dtype=float),
        "Error_%": np.asarray(error_pct_monthly, dtype=float),
    }
    return {prediction_year: pd.DataFrame(data)}


def _extract_pf_pa_ratio_cons(first_year):
    """Extract PF (12), PA (12), optionally PA/PF ratio (12) and consumption (12).

    Returns a tuple (pf, pa, ratio, cons).
    If any series has fewer than 12 months, pads with zeros to ensure length 12.
    Returns None only if all values sum to 0.
    """
    pf = first_year.sort_values(Aliases.MOIS)[Aliases.PUISSANCE_FACTUREE].fillna(0).values[:12]
    pa = first_year.sort_values(Aliases.MOIS)[Aliases.PUISSANCE_APPELEE].fillna(0).values[:12]
    cons = first_year.sort_values(Aliases.MOIS)[Aliases.CONSOMMATION_KWH].fillna(0).values[:12]

    # Pad with zeros if fewer than 12 months
    if len(pf) < 12:
        pf = np.pad(pf, (0, 12 - len(pf)), constant_values=0)
    if len(pa) < 12:
        pa = np.pad(pa, (0, 12 - len(pa)), constant_values=0)
    if len(cons) < 12:
        cons = np.pad(cons, (0, 12 - len(cons)), constant_values=0)

    # Compute pa/pf ratio, set to 0 if pf==0
    ratio_pa_pf = np.zeros_like(pf)
    nonzero_pf = pf != 0
    ratio_pa_pf[nonzero_pf] = pa[nonzero_pf] / pf[nonzero_pf]

    ratio_pf_pa = np.zeros_like(pa)
    nonzero_pa = pa != 0
    ratio_pf_pa[nonzero_pa] = pf[nonzero_pa] / pa[nonzero_pa]
    
    return pf, pa, ratio_pa_pf, cons


def _collect_starting_samples(df_scope, prediction_year):
    """Collect training samples (X_list, y_list) from a dataframe scope.

    For each contrat in df_scope, find the move-in start year, extract the
    first 12 months using _extract_pf_pa_ratio_cons and return lists of
    feature vectors (24 = 12 PF + 12 PA/PF ratios) and targets (12 months cons).
    """
    contrats = df_scope[Aliases.CONTRAT].unique()
    X_list, y_list = [], []

    for c in contrats:
        sub = df_scope[df_scope[Aliases.CONTRAT] == c].sort_values([Aliases.ANNEE, Aliases.MOIS]) 
        if sub.empty or sub[Aliases.DATE_EMMENAGEMENT].isna().all():
            continue

        start_year = get_move_in_year(sub)
        if pd.isna(start_year) or start_year == prediction_year or start_year < 2018: 
            continue

        first_year = sub[sub[Aliases.ANNEE] == start_year].copy()
        if first_year.empty:
            continue

        res = _extract_pf_pa_ratio_cons(first_year)
        if res is None:
            continue
        
        pf, pa, ratio, cons = res
        if pf.sum() == 0:
            continue
        
        X_list.append(pf)
        y_list.append(cons)

    return X_list, y_list

def _collect_starting_samples_growth(df_scope, prediction_year, use_ratio=False):
    contrats = df_scope[Aliases.CONTRAT].unique()
    X_list, y_list = [], []

    for c in contrats:
        sub = df_scope[df_scope[Aliases.CONTRAT] == c].sort_values([Aliases.ANNEE, Aliases.MOIS])
        if sub.empty or sub[Aliases.DATE_EMMENAGEMENT].isna().all():
            continue

        start_year = get_move_in_year(sub)
        if pd.isna(start_year) or (start_year + 1) >= prediction_year or start_year < 2018:
            continue

        first_year = sub[sub[Aliases.ANNEE] == start_year].copy()
        second_year = sub[sub[Aliases.ANNEE] == start_year + 1].copy()
        if first_year.empty or second_year.empty:
            continue

        res = _extract_pf_pa_ratio_cons(first_year)
        if res is None:
            continue
        pf, pa, ratio, cons = res
        if pf.sum() == 0:
            continue

        cons_next = second_year.sort_values(Aliases.MOIS)[Aliases.CONSOMMATION_KWH].fillna(0).values[:12]
        if len(cons_next) < 12:
            continue

        total_1 = cons.sum()
        total_2 = cons_next.sum()
        if total_1 == 0 or total_2 == 0:
            continue

        growth_rate = np.log(total_2 + 1e-9) - np.log(total_1 + 1e-9)
        X_list.append(np.concatenate([pf, ratio]))
        y_list.append(growth_rate)

    return X_list, y_list



def handle_similarity_entity_prediction(df, all_results_established, entity_id, prediction_year, level_type=None, save_csv=False):
    """
    Generic handler for similarity entities (prediction_year starters).
    Replaced clustering with a small predictive model:
    predict the starting 12-month consumption based on
    the starting 12-month puissance facturée and optionally the PA/PF ratio (pa/pf).
    By default `use_ratio` is False and only PF months are used as features.
    """
    if level_type is None:
        level_type = Aliases.CONTRAT
    if level_type not in [Aliases.CONTRAT, Aliases.ACTIVITE]:
        raise ValueError(f"level_type must be either '{Aliases.CONTRAT}' or '{Aliases.ACTIVITE}'.")

    df_entity = df[df[level_type] == entity_id].copy()
    if df_entity.empty:
        print(f"⚠️ {level_type.capitalize()} {entity_id} not found.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    label = level_type.capitalize()
    print(f"\n🔗 Handling similarity {label}: {entity_id}")

    # === Filter by activity if possible ===
    if level_type == Aliases.CONTRAT:
        activite = df_entity[Aliases.ACTIVITE].iloc[0]
        df_same_activity = df[df[Aliases.ACTIVITE] == activite].copy()
        contrats_in_activity = df_same_activity[Aliases.CONTRAT].unique().tolist()
        if len(contrats_in_activity) <= 3:
            df_train_scope = df.copy()
            print("⚠️ Only 3 contrat in this activity → using all data for training.")
        else:
            df_train_scope = df_same_activity.copy()
            print(f"✅ Training model within activity {activite}.")
    else:
        df_train_scope = df.copy()
        print(f"✅ Training model on all activities (level_type={level_type}).")

    # --- Extract training samples ---
    # Use helper to collect X_list and y_list (features depend on use_ratio)
    X_list, y_list = _collect_starting_samples(df_train_scope, prediction_year)

    if len(X_list) < 3:
        print("⚠️ Not enough data to train the model, using all of the data if possible.")
        X_list, y_list = _collect_starting_samples(df.copy(), prediction_year)

    X = np.vstack(X_list)
    Y = np.vstack(y_list)

    # Combine X and Y into one dataframe
    all_data = np.hstack([X, Y])
    # Build appropriate column names depending on whether ratio is used
    feature_cols = [f'PF_M{i+1}' for i in range(12)]

    columns = feature_cols + [f'Cons_M{i+1}' for i in range(12)]
    df_all = pd.DataFrame(all_data, columns=columns)

    print(f"\n📊 Training samples: {len(X)}, Features: {X.shape[1]}, Targets: {Y.shape[1]}")

    # === Train predictive model ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
    model.fit(X_scaled, Y)

    print("✅ Model trained successfully.")

    # === Prepare input for target entity ===
    sub_entity = df_entity.sort_values([Aliases.ANNEE, Aliases.MOIS]).copy()
    start_year_entity = get_move_in_year(sub_entity)
    print(f"start year: {start_year_entity}")

    first_year_entity = sub_entity[sub_entity[Aliases.ANNEE] == start_year_entity].copy()
    if first_year_entity.empty:
        print(f"⚠️ No data for start year {start_year_entity} for {entity_id}.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)
    # extract features for the target entity (respecting use_ratio)
    res_e = _extract_pf_pa_ratio_cons(first_year_entity)
    if res_e is None:
        print(f"⚠️ Missing PF or PA data for {entity_id}.")
        return handle_similarity_entity(df, all_results_established, entity_id, prediction_year, level_type, k=1)

    pf_e, pa_e, ratio_e, cons_e = res_e

    # append a placeholder row to df_all so CSV shows the input

    df_all.loc[len(df_all)] = list(pf_e) + [np.nan] * 12
    

    # build X_new matching training features
    X_new = pf_e.reshape(1, -1)

    X_new_scaled = scaler.transform(X_new)

    if X_new.sum() == 0:
        print(f"0 values in pf, defaulting to 0.")
        raw_pred = np.zeros((1, X_new.shape[1]))
    else:
        raw_pred = model.predict(X_new_scaled).flatten()
        if save_csv:
            df_all.to_csv(f"{entity_id}.csv")
    
    neg_count = (raw_pred < 0).sum()
    if neg_count:
        print(f"⚠️ {neg_count} negative values were clamped to 0.")
    predicted_year_cons = np.clip(raw_pred, 0, None)

    print(f"✅ Predicted starting consumption for {entity_id}: {np.round(predicted_year_cons,2)}")

    # === Compute errors if prediction_year actual exists ===
    actual_year_cons = (
        df_entity[df_entity[Aliases.ANNEE] == prediction_year]
        .sort_values(Aliases.MOIS)[Aliases.CONSOMMATION_KWH]
        .fillna(0)
        .to_numpy()
    )
    if len(actual_year_cons) < 12:
        actual_year_cons = np.pad(actual_year_cons, (0, 12 - len(actual_year_cons)), constant_values=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct_monthly = np.where(
            actual_year_cons != 0,
            ((predicted_year_cons - actual_year_cons) / actual_year_cons) * 100,
            np.nan,
        )

    annual_actual = np.float64(actual_year_cons.sum())
    annual_pred = np.float64(predicted_year_cons.sum())
    annual_error_pct = (
        np.nan if annual_actual == 0 else ((annual_pred - annual_actual) / annual_actual) * 100
    )

    monthly_details = _build_monthly_details(
        prediction_year, actual_year_cons, predicted_year_cons, error_pct_monthly
    )

    result = {
        "entity": f"{label}_{entity_id}",
        "best_model": {
            "pred_monthly_matrix": np.array([predicted_year_cons]),
            "actual_monthly_matrix": np.array([actual_year_cons]),
            "valid_years": df_entity[Aliases.ANNEE].unique(),
            "strategy": f"Predictive Start Consumption ({label})",
        },
        "monthly_details": monthly_details,
        "test_years": {
            f"{prediction_year}_actual_monthly": actual_year_cons,
            f"{prediction_year}_predicted_monthly": predicted_year_cons,
            f"{prediction_year}_error_pct_monthly": error_pct_monthly,
            f"{prediction_year}_actual_annual": annual_actual,
            f"{prediction_year}_predicted_annual": annual_pred,
            f"{prediction_year}_error_pct_annual": annual_error_pct,
            f"{prediction_year}_source_match": [],  # no longer based on similarity
        },
    }

    print(f"🎯 Completed predictive estimation for {entity_id}")
    return result

def handle_growth_entity_prediction(df, all_results_established, entity_id, prediction_year, level_type=Aliases.CONTRAT, save_csv=False):
    """
    Predicts the growth rate of consumption (prediction_year vs prediction_year-1) for a contract or activity
    using a regression model trained on starting contracts' PF & PA features.
    Fallbacks gracefully to handle_growth_entity or empty result when needed.
    """
    if level_type not in [Aliases.CONTRAT, Aliases.ACTIVITE]:
        raise ValueError(f"level_type must be either '{Aliases.CONTRAT}' or '{Aliases.ACTIVITE}'.")

    df_entity = df[df[level_type] == entity_id].copy()
    if df_entity.empty:
        print(f"⚠️ {level_type.capitalize()} {entity_id} not found.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    label = level_type.capitalize()
    print(f"\n🚀 Handling predictive growth {label}: {entity_id}")

    # === Filter by activity if possible ===
    if level_type == Aliases.CONTRAT:
        activite = df_entity[Aliases.ACTIVITE].iloc[0]
        df_same_activity = df[df[Aliases.ACTIVITE] == activite].copy()
        contrats_in_activity = df_same_activity[Aliases.CONTRAT].unique().tolist()
        df_train_scope = (
            df.copy() if len(contrats_in_activity) <= 3 else df_same_activity.copy()
        )
    else:
        df_train_scope = df.copy()

    # === Prepare training data: PF, PA, growth ===
    X_list, y_list = _collect_starting_samples_growth(df_train_scope, prediction_year)

    if len(X_list) < 5:
        print("⚠️ Not enough training data for predictive growth in this activity→ fallback to full training.")
        X_list, y_list = _collect_starting_samples_growth(df.copy(), prediction_year)

    X = np.vstack(X_list)
    y = np.array(y_list)

    
    
    # === Train regression model ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    # === Prepare input for target entity ===
    sub_entity = df_entity.sort_values([Aliases.ANNEE, Aliases.MOIS]).copy()
    start_year_entity = get_move_in_year(sub_entity)
    first_year_entity = sub_entity[sub_entity[Aliases.ANNEE] == start_year_entity].copy()
    if first_year_entity.empty:
        print(f"⚠️ No valid start year data for {entity_id}.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    res_e = _extract_pf_pa_ratio_cons(first_year_entity)
    if res_e is None:
        print(f"⚠️ Missing PF/PA data for {entity_id}.")
        return handle_growth_entity(df, all_results_established, entity_id, prediction_year, level_type)

    pf_e, pa_e, ratio_e, cons_e = res_e
    X_new = np.concatenate([pf_e, ratio_e]).reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)
    predicted_growth = model.predict(X_new_scaled)[0]

    if predicted_growth > 2:
        predicted_growth = 2
    elif predicted_growth < -2:
        predicted_growth = -2

    print(f"📈 Predicted log growth rate: {predicted_growth:.4f}")

    # === Save training data to CSV ===
    feature_cols = [f'PF_M{i+1}' for i in range(12)] + [f'PA_M{i+1}' for i in range(12)]
    df_all = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=feature_cols + ['growth_rate'])
    print(f"📊 Training growth model with {len(X)} samples, {X.shape[1]} features.")

    df_all.loc[len(df_all)] = list(pf_e) + list(ratio_e) + [np.nan]
    if save_csv:
        df_all.to_csv(f"{entity_id}_growth_training.csv", index=False)
        print(f"💾 Saved training data to {entity_id}_growth_training.csv")

    # === Apply predicted growth rate ===
    prev_year = prediction_year - 1
    actual_prev_year_current = df_entity[df_entity[Aliases.ANNEE] == prev_year][Aliases.CONSOMMATION_KWH].sum()
    predicted_year_annual = np.exp(np.log(actual_prev_year_current + 1e-9) + predicted_growth)

    # === Reuse monthly shape logic from handle_growth_entity ===
    try:
        if level_type == Aliases.CONTRAT:
            df_pred_year = df[df[Aliases.ANNEE] == prediction_year]
            df_pred_year = df_pred_year[df_pred_year[Aliases.ACTIVITE] == activite]
        else:
            df_pred_year = df[df[Aliases.ANNEE] == prediction_year]

        pivot = (
            df_pred_year.pivot_table(
                index=level_type,
                columns=Aliases.MOIS,
                values=Aliases.PUISSANCE_FACTUREE,
                aggfunc="sum"
            )
            .fillna(0)
            .reindex(columns=range(1, 13), fill_value=0)
        )
        if not pivot.empty:
            Xp = pivot.values
            candidates = pivot.index.to_numpy()
            knn = NearestNeighbors(n_neighbors=min(3, len(Xp)), metric="euclidean")
            knn.fit(Xp)
            target_vec = (
                df_entity[df_entity[Aliases.ANNEE] == prediction_year]
                .sort_values(Aliases.MOIS)[Aliases.PUISSANCE_FACTUREE]
                .fillna(0)
                .to_numpy()
                .reshape(1, -1)
            )
            distances, indices = knn.kneighbors(target_vec)
            similar_entities = [c for c in candidates[indices[0]] if c != entity_id][:3]
            weights = np.array([0.6, 0.3, 0.1])[: len(similar_entities)]
            weights /= weights.sum()

            weighted_mean_curve = np.zeros(12)
            for sid, w in zip(similar_entities, weights):
                match = next(
                    (r for r in all_results_established if r["entity"] == f"{label}_{sid}"),
                    None,
                )
                if match is not None:
                    best_model = match["best_model"]
                    idx_pred_year = np.where(best_model["valid_years"] == prediction_year)[0][0]
                    weighted_mean_curve += w * best_model["pred_monthly_matrix"][idx_pred_year]

            mean_curve_normalized = weighted_mean_curve / (weighted_mean_curve.sum() or 1)
            predicted_year_monthly = predicted_year_annual * mean_curve_normalized
        else:
            predicted_year_monthly = np.repeat(predicted_year_annual / 12, 12)
    except Exception as e:
        print(f"⚠️ Failed to compute monthly shape: {e}")
        predicted_year_monthly = np.repeat(predicted_year_annual / 12, 12)

    # === Compute actuals and errors ===
    actual_year_cons = (
        df_entity[df_entity[Aliases.ANNEE] == prediction_year][Aliases.CONSOMMATION_KWH]
        .fillna(0)
        .to_numpy()
    )
    if len(actual_year_cons) < 12:
        actual_year_cons = np.pad(actual_year_cons, (0, 12 - len(actual_year_cons)), constant_values=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct_monthly = np.where(
            actual_year_cons != 0,
            ((predicted_year_monthly - actual_year_cons) / actual_year_cons) * 100,
            np.nan,
        )

    annual_actual = np.float64(actual_year_cons.sum())
    annual_pred = np.float64(predicted_year_monthly.sum())
    annual_error_pct = (
        np.nan if annual_actual == 0 else ((annual_pred - annual_actual) / annual_actual) * 100
    )

    best_model_new = {
        "pred_monthly_matrix": np.array([predicted_year_monthly]),
        "actual_monthly_matrix": np.array([actual_year_cons]),
        "valid_years": df_entity[Aliases.ANNEE].unique(),
        "strategy": f"Predictive - Growth ({label})",
    }

    monthly_details = _build_monthly_details(
        prediction_year, actual_year_cons, predicted_year_monthly, error_pct_monthly
    )

    result = {
        "entity": f"{label}_{entity_id}",
        "best_model": best_model_new,
        "monthly_details": monthly_details,
        "test_years": {
            f"{prediction_year}_actual_monthly": actual_year_cons,
            f"{prediction_year}_predicted_monthly": predicted_year_monthly,
            f"{prediction_year}_error_pct_monthly": error_pct_monthly,
            f"{prediction_year}_actual_annual": annual_actual,
            f"{prediction_year}_predicted_annual": annual_pred,
            f"{prediction_year}_error_pct_annual": annual_error_pct,
            f"{prediction_year}_source_match": similar_entities if 'similar_entities' in locals() else [],
        },
    }

    print(f"✅ Predicted {prediction_year} annual (growth-based): {predicted_year_annual:,.2f}")
    return result


def _build_empty_result(df_entity, entity_id, prediction_year, level_type=Aliases.CONTRAT):
    """Return a consistent empty result object for similarity contracts."""
    actual_year_cons = (
        df_entity[df_entity[Aliases.ANNEE] == prediction_year][Aliases.CONSOMMATION_KWH]
        .fillna(0)
        .to_numpy()
    )
    annual_actual = np.float64(actual_year_cons.sum())
    nan_arr = np.full(12, np.nan)

    best_model = {
        'pred_monthly_matrix': None,
        'actual_monthly_matrix': None,
        'valid_years': df_entity[Aliases.ANNEE].unique() if not df_entity.empty else [],
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
        f'{prediction_year}_actual_monthly': actual_year_cons,
        f'{prediction_year}_predicted_monthly': np.zeros(12),
        f'{prediction_year}_error_pct_monthly': nan_arr,
        f'{prediction_year}_actual_annual': annual_actual,
        f'{prediction_year}_predicted_annual': np.float64(0.0),
        f'{prediction_year}_error_pct_annual': np.nan,
        f'{prediction_year}_source_match': None
    }

    monthly_details = _build_monthly_details(
        prediction_year, actual_year_cons, np.zeros(12), nan_arr
    )

    return {
        "entity": f'{ "Contrat" if level_type == Aliases.CONTRAT else "Activité"}_{entity_id}',
        "best_model": best_model,
        "training_end": None,
        "monthly_details": monthly_details,
        "test_years": test_years
    }



def handle_growth_entity(df, all_results_established, entity_id, prediction_year, level_type=Aliases.CONTRAT):
    """
    Generic handler for growth entities (either contracts or activities).
    - For contracts: filters within same activity unless no peers exist.
    - For activities: compares directly via puissance facturée vectors.
    """
    if level_type not in [Aliases.CONTRAT, Aliases.ACTIVITE]:
        raise ValueError(f"level_type must be either '{Aliases.CONTRAT}' or '{Aliases.ACTIVITE}'.")

    df_entity = df[df[level_type] == entity_id].copy()
    if df_entity.empty:
        print(f"⚠️ {level_type.capitalize()} {entity_id} not found.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    label = level_type.capitalize()
    prev_year = prediction_year - 1
    print(f"\n🚀 Handling growth {label}: {entity_id}")

    # === Determine comparison subset ===
    if level_type == Aliases.CONTRAT:
        activite = df_entity[Aliases.ACTIVITE].iloc[0]
        df_same_activity = df[df[Aliases.ACTIVITE] == activite].copy()

        contrats_in_activity = df_same_activity[Aliases.CONTRAT].unique().tolist()
        if len(contrats_in_activity) <= 1:
            # No other contracts in same activity → use all data
            df_pred_year = df[df[Aliases.ANNEE] == prediction_year].copy()
        else:
            df_pred_year = df_same_activity[df_same_activity[Aliases.ANNEE] == prediction_year].copy()
    else:
        df_pred_year = df[df[Aliases.ANNEE] == prediction_year].copy()

    # === Build pivot for similarity ===
    pivot = (
        df_pred_year.pivot_table(
            index=level_type,
            columns=Aliases.MOIS,
            values=Aliases.PUISSANCE_FACTUREE,
            aggfunc="sum"
        )
        .fillna(0)
        .reindex(columns=range(1, 13), fill_value=0)
    )

    if pivot.empty:
        print(f"⚠️ No comparable {prediction_year} data for {entity_id}.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    target_vec = (
        df_entity[df_entity[Aliases.ANNEE] == prediction_year]
        .sort_values(Aliases.MOIS)[Aliases.PUISSANCE_FACTUREE]
        .fillna(0)
        .to_numpy()
        .reshape(1, -1)
    )
    if target_vec.size == 0:
        print(f"⚠️ {label} {entity_id} has no {prediction_year} puissance facturée data.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    # === Find nearest neighbors ===
    X = pivot.values
    candidates = pivot.index.to_numpy()
    knn = NearestNeighbors(n_neighbors=min(5, len(X)), metric="euclidean")
    knn.fit(X)
    distances, indices = knn.kneighbors(target_vec)

    similar_entities = [c for c in candidates[indices[0]] if c != entity_id]
    if not similar_entities:
        print(f"⚠️ No similar {level_type}s found for {entity_id}.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    # === Use top 3 similar entities with weights ===
    top_k = min(3, len(similar_entities))
    top_similar_ids = similar_entities[:top_k]
    
    # Adjust weights to sum to 1 based on available entities
    base_weights = [0.6, 0.3, 0.1]
    weights = np.array(base_weights[:top_k])
    weights = weights / weights.sum()
    
    print(f"✅ Using top {top_k} similar {level_type}s: {top_similar_ids} with weights {weights}")

    # === Find corresponding established results ===
    prefix = "Contrat" if level_type == Aliases.CONTRAT else "Activité"
    
    valid_matches = []
    valid_weights = []
    
    for i, match_id in enumerate(top_similar_ids):
        match_result = next(
            (r for r in all_results_established if r["entity"] == f"{prefix}_{match_id}"),
            None,
        )
        if match_result is None:
            print(f"⚠️ No established match found for {match_id}.")
            continue
            
        best_model = match_result["best_model"]
        valid_years = best_model["valid_years"]
        
        if not (prev_year in valid_years and prediction_year in valid_years):
            print(f"⚠️ Matched {match_id} lacks {prev_year} or {prediction_year} data.")
            continue
            
        valid_matches.append((match_id, best_model))
        valid_weights.append(weights[i])
    
    if not valid_matches:
        print(f"⚠️ No valid matches found with required data.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)
    
    # Renormalize weights for valid matches
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()

    # === Compute weighted growth rate ===
    weighted_growth_rate = 0.0
    
    for (match_id, best_model), weight in zip(valid_matches, valid_weights):
        valid_years = best_model["valid_years"]
        idx_prev_year = np.where(valid_years == prev_year)[0][0]
        idx_pred_year = np.where(valid_years == prediction_year)[0][0]
        actual_prev_year = best_model["actual_monthly_matrix"][idx_prev_year].sum()
        predicted_year_similar = best_model["pred_monthly_matrix"][idx_pred_year].sum()
        
        growth_rate = np.log(predicted_year_similar + 1e-9) - np.log(actual_prev_year + 1e-9)
        weighted_growth_rate += weight * growth_rate
        print(f"  {match_id}: growth_rate={growth_rate:.4f}, weight={weight:.2f}")
    
    print(f"📈 Weighted growth rate (log): {weighted_growth_rate:.4f}")

    actual_prev_year_current = df_entity[df_entity[Aliases.ANNEE] == prev_year][Aliases.CONSOMMATION_KWH].sum()
    predicted_year_annual = np.exp(np.log(actual_prev_year_current + 1e-9) + weighted_growth_rate)

    # === Compute weighted mean curve ===
    weighted_mean_curve = np.zeros(12)
    
    for (match_id, best_model), weight in zip(valid_matches, valid_weights):
        valid_years = best_model["valid_years"]
        idx_pred_year = np.where(valid_years == prediction_year)[0][0]
        mean_curve = best_model["pred_monthly_matrix"][idx_pred_year]
        weighted_mean_curve += weight * mean_curve
    
    mean_curve_normalized = weighted_mean_curve / (weighted_mean_curve.sum() or 1)
    predicted_year_monthly = predicted_year_annual * mean_curve_normalized

    # === Compute actuals and errors ===
    actual_year_cons = (
        df_entity[df_entity[Aliases.ANNEE] == prediction_year][Aliases.CONSOMMATION_KWH]
        .fillna(0)
        .to_numpy()
    )
    if len(actual_year_cons) < 12:
        actual_year_cons = np.pad(actual_year_cons, (0, 12 - len(actual_year_cons)), constant_values=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct_monthly = np.where(
            actual_year_cons != 0,
            ((predicted_year_monthly - actual_year_cons) / actual_year_cons) * 100,
            np.nan,
        )

    annual_actual = np.float64(actual_year_cons.sum())
    annual_pred = np.float64(predicted_year_monthly.sum())
    annual_error_pct = (
        np.nan if annual_actual == 0 else ((annual_pred - annual_actual) / annual_actual) * 100
    )

    # === Assemble output ===
    best_model_new = {
        "pred_monthly_matrix": np.array([predicted_year_monthly]),
        "actual_monthly_matrix": np.array([actual_year_cons]),
        "valid_years": df_entity[Aliases.ANNEE].unique(),
        "strategy": f"Clustering - Growth ({label})",
    }

    monthly_details = _build_monthly_details(
        prediction_year, actual_year_cons, predicted_year_monthly, error_pct_monthly
    )

    result = {
        "entity": f"{label}_{entity_id}",
        "best_model": best_model_new,
        "monthly_details": monthly_details,
        "test_years": {
            f"{prediction_year}_actual_monthly": actual_year_cons,
            f"{prediction_year}_predicted_monthly": predicted_year_monthly,
            f"{prediction_year}_error_pct_monthly": error_pct_monthly,
            f"{prediction_year}_actual_annual": annual_actual,
            f"{prediction_year}_predicted_annual": annual_pred,
            f"{prediction_year}_error_pct_annual": annual_error_pct,
            f"{prediction_year}_source_match": [match_id for match_id, _ in valid_matches],
        },
    }

    print(f"✅ Predicted {prediction_year} annual: {predicted_year_annual:,.2f}")
    return result


def handle_similarity_entity(df, all_results_established, entity_id, prediction_year, level_type=Aliases.CONTRAT, k=3):
    """
    Generic handler for similarity entities (prediction_year starters).
    - For contracts: filter within same activity (fallback to all if alone).
    - For activities: similarity based on puissance facturée vectors.
    """
    if level_type not in [Aliases.CONTRAT, Aliases.ACTIVITE]:
        raise ValueError(f"level_type must be either '{Aliases.CONTRAT}' or '{Aliases.ACTIVITE}'.")

    df_entity = df[df[level_type] == entity_id].copy()
    if df_entity.empty:
        print(f"⚠️ {level_type.capitalize()} {entity_id} not found.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    label = level_type.capitalize()
    print(f"\n🔗 Handling similarity {label}: {entity_id}")

    # === Determine comparison scope ===
    if level_type == Aliases.CONTRAT:
        activite = df_entity[Aliases.ACTIVITE].iloc[0]
        df_same_activity = df[df[Aliases.ACTIVITE] == activite].copy()
        contrats_in_activity = df_same_activity[Aliases.CONTRAT].unique().tolist()
        if len(contrats_in_activity) <= 1:
            df_pred_year = df[df[Aliases.ANNEE] == prediction_year].copy()
        else:
            df_pred_year = df_same_activity[df_same_activity[Aliases.ANNEE] == prediction_year].copy()
    else:
        df_pred_year = df[df[Aliases.ANNEE] == prediction_year].copy()

    pivot = (
        df_pred_year.pivot_table(
            index=level_type,
            columns=Aliases.MOIS,
            values=Aliases.PUISSANCE_FACTUREE,
            aggfunc="sum"
        )
        .fillna(0)
        .reindex(columns=range(1, 13), fill_value=0)
    )

    if pivot.empty:
        print(f"⚠️ No comparable {prediction_year} puissance facturée data for {entity_id}.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    target_vec = (
        df_entity[df_entity[Aliases.ANNEE] == prediction_year]
        .sort_values(Aliases.MOIS)[Aliases.PUISSANCE_FACTUREE]
        .fillna(0)
        .to_numpy()
        .reshape(1, -1)
    )
    if target_vec.sum() == 0:
        print(f"⚠️ {entity_id} doesn't have any {prediction_year} puissance facturée data.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)
    
    X = pivot.values
    candidates = pivot.index.to_numpy()
    knn = NearestNeighbors(n_neighbors=min(5, len(X)), metric="euclidean")
    knn.fit(X)
    distances, indices = knn.kneighbors(target_vec)

    similar_entities = [c for c in candidates[indices[0]] if c != entity_id]
    if not similar_entities:
        print(f"⚠️ No similar {level_type}s found for {entity_id}.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)

    # === Use top k similar entities with weights ===
    top_k = min(k, len(similar_entities))
    top_similar_ids = similar_entities[:top_k]
    
    # Adjust weights to sum to 1 based on available entities
    base_weights = [0.6, 0.3, 0.1]
    weights = np.array(base_weights[:top_k])
    weights = weights / weights.sum()
    
    print(f"✅ Using top {top_k} similar {label}s: {top_similar_ids} with weights {weights}")

    prefix = "Contrat" if level_type == Aliases.CONTRAT else "Activité"
    
    valid_matches = []
    valid_weights = []
    
    for i, match_id in enumerate(top_similar_ids):
        match_result = next(
            (r for r in all_results_established if r["entity"] == f"{prefix}_{match_id}"),
            None,
        )

        if match_result is None:
            print(f"⚠️ No established match found for {match_id}.")
            continue

        best_model = match_result["best_model"]
        if prediction_year not in best_model["valid_years"]:
            print(f"⚠️ Matched {match_id} has no {prediction_year} predictions.")
            continue

        valid_matches.append((match_id, best_model))
        valid_weights.append(weights[i])

    if not valid_matches:
        print(f"⚠️ No valid matches found with {prediction_year} predictions.")
        return _build_empty_result(df_entity, entity_id, prediction_year, level_type)
    
    # Renormalize weights for valid matches
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()

    # === Compute weighted prediction ===
    predicted_year_cons = np.zeros(12)
    
    for (match_id, best_model), weight in zip(valid_matches, valid_weights):
        idx_pred_year = np.where(best_model["valid_years"] == prediction_year)[0][0]
        pred_monthly = best_model["pred_monthly_matrix"][idx_pred_year]
        predicted_year_cons += weight * pred_monthly
        print(f"  {match_id}: weight={weight:.2f}")

    actual_year_cons = (
        df_entity[df_entity[Aliases.ANNEE] == prediction_year]
        .sort_values(Aliases.MOIS)[Aliases.CONSOMMATION_KWH]
        .fillna(0)
        .to_numpy()
    )
    if len(actual_year_cons) < 12:
        actual_year_cons = np.pad(actual_year_cons, (0, 12 - len(actual_year_cons)), constant_values=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct_monthly = np.where(
            actual_year_cons != 0,
            ((predicted_year_cons - actual_year_cons) / actual_year_cons) * 100,
            np.nan,
        )

    annual_actual = np.float64(actual_year_cons.sum())
    annual_pred = np.float64(predicted_year_cons.sum())
    annual_error_pct = (
        np.nan if annual_actual == 0 else ((annual_pred - annual_actual) / annual_actual) * 100
    )

    monthly_details = _build_monthly_details(
        prediction_year, actual_year_cons, predicted_year_cons, error_pct_monthly
    )

    result = {
        "entity": f"{label}_{entity_id}",
        "best_model": {
            "pred_monthly_matrix": np.array([predicted_year_cons]),
            "actual_monthly_matrix": np.array([actual_year_cons]),
            "valid_years": df_entity[Aliases.ANNEE].unique(),
            "strategy": f"Clustering - Similarity ({label})",
        },
        "monthly_details": monthly_details,
        "test_years": {
            f"{prediction_year}_actual_monthly": actual_year_cons,
            f"{prediction_year}_predicted_monthly": predicted_year_cons,
            f"{prediction_year}_error_pct_monthly": error_pct_monthly,
            f"{prediction_year}_actual_annual": annual_actual,
            f"{prediction_year}_predicted_annual": annual_pred,
            f"{prediction_year}_error_pct_annual": annual_error_pct,
            f"{prediction_year}_source_match": [match_id for match_id, _ in valid_matches],
        },
    }

    print(f"🔁 Reused weighted {prediction_year} predictions from {len(valid_matches)} {label}(s)")
    return result