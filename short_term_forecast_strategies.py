from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from onee.growth_rate_model import MeanRevertingGrowthModel, build_growth_rate_features
from onee.utils import (
    add_monthly_feature,
    add_exogenous_features,
    create_monthly_matrix,
    calculate_all_metrics,
    select_best_model,
    add_yearly_feature,
    get_move_in_year,
    add_annual_client_feature
)
from onee.data.names import Aliases
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Mapping, Optional
from itertools import product


def compute_power_weights(years, lambda_value):
    """
    Compute power-based weights (lambda^k) so recent years have more influence.
    The most recent year receives exponent 0, the next receives 1, etc.
    """
    years = np.asarray(years)
    n_years = years.shape[0]
    if n_years == 0:
        return np.array([], dtype=float)

    if lambda_value is None:
        return np.ones(n_years, dtype=float)

    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative for power weights.")

    order = np.argsort(np.argsort(years))
    exponents = (n_years - 1) - order
    weights = np.power(lambda_value, exponents, dtype=float)

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(n_years, dtype=float)

    return weights


def fit_weighted_pca(X, sample_weights, n_components):
    """
    Fit PCA on standardized data using sample weights.
    Returns a lightweight model dict with mean and components.
    """
    X = np.asarray(X, dtype=float)
    sample_weights = np.asarray(sample_weights, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")

    n_samples = X.shape[0]
    if n_samples != sample_weights.shape[0]:
        raise ValueError("Length of sample_weights must match number of samples in X.")

    if np.any(sample_weights < 0):
        raise ValueError("sample_weights must be non-negative.")

    weight_sum = sample_weights.sum()
    if weight_sum <= 0:
        sample_weights = np.ones(n_samples, dtype=float)
        weight_sum = float(n_samples)

    mean = np.average(X, axis=0, weights=sample_weights)
    centered = X - mean
    weighted = centered * np.sqrt(sample_weights)[:, None]

    cov = (weighted.T @ weighted) / weight_sum
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], a_min=0.0, a_max=None)
    eigvecs = eigvecs[:, order]

    max_components = min(n_components, eigvecs.shape[1])
    components = eigvecs[:, :max_components].T

    return {
        "mean": mean,
        "components": components,
        "explained_variance": eigvals[:max_components],
        "n_components": components.shape[0],
    }


def weighted_pca_transform(model, X):
    centered = np.asarray(X, dtype=float) - model["mean"]
    return centered @ model["components"].T


def weighted_pca_inverse_transform(model, scores):
    scores = np.asarray(scores, dtype=float)
    return (scores @ model["components"]) + model["mean"]


def select_best_with_fallback(results, r2_threshold):
    """
    Select the best model among `results` using the shared selection logic.
    Falls back to the highest R2 if the threshold filter rejects every model.
    """
    if not results:
        return None

    try:
        return select_best_model(results, r2_threshold)
    except ValueError:
        # Fall back to the highest annual R2 score available
        return max(results, key=lambda r: r.get("annual_r2", float("-inf")))


def _weighted_quantile(values, weights, tau):
    """
    Compute the weighted tau-quantile of 'values' with nonnegative 'weights'.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = values[mask]
    w = weights[mask]
    if v.size == 0:
        return 0.0
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(cdf, tau, side="left")
    idx = np.clip(idx, 0, len(v) - 1)
    return v[idx]

def _fit_lambda_pinball(y, p, tau=0.75, eps=1e-8):
    """
    Closed-form lambda under pinball loss: lambda is the weighted tau-quantile
    of (y/p - 1) with weights p, assuming p>0. Clips/filters safely.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    
    # Filter out invalid predictions
    valid_mask = (p > eps) & np.isfinite(p) & np.isfinite(y)
    
    if not valid_mask.any():
        # No valid data points
        return 0.0
    
    y_valid = y[valid_mask]
    p_valid = p[valid_mask]
    
    z = (y_valid / p_valid) - 1.0
    w = p_valid
    
    lam = _weighted_quantile(z, w, tau)
    
    # Optional: ensure we don't flip sign crazily
    # lam = np.clip(lam, -0.5, 2.0)
    
    if not np.isfinite(lam):
        lam = 0.0
    
    return lam


def build_ensemble_result(
    best_models,
    *,
    weights=None,
    strategy_name="Strategy5_Ensemble",
    tau=0.75,
    use_annual=False,
):
    """
    Combine multiple strategy outputs into a weighted ensemble, then apply an
    asymmetric-upward-bias calibration factor (1 + lambda), where lambda is learned
    by minimizing a pinball (quantile) loss with tau > 0.5 (penalizes underestimates more).
    
    This version prevents future data leakage by training weights and lambda on past data only.
    """
    models = [m for m in best_models if m is not None]
    if len(models) < 2:
        return None

    shared_years = set(models[0]["valid_years"])
    for model in models[1:]:
        shared_years &= set(model["valid_years"])
    if not shared_years:
        return None

    ordered_years = sorted(shared_years)
    model_preds, model_actuals = [], []

    for model in models:
        idx = [np.where(model["valid_years"] == year)[0][0] for year in ordered_years]
        model_preds.append(model["pred_monthly_matrix"][idx])  # (n_years, 12)
        model_actuals.append(model["actual_monthly_matrix"][idx])

    model_preds = np.stack(model_preds)  # (n_models, n_years, 12)
    model_actuals = np.stack(model_actuals)

    # Use mean of actuals across models as reference
    reference_actual = model_actuals.mean(axis=0)  # (n_years, 12)

    # Initialize ensemble predictions and tracking arrays
    ensemble_preds = np.zeros((len(ordered_years), 12))
    lambda_values = []
    weights_history = []

    # Iterate over each year
    for i in range(len(ordered_years)):
        if i == 0:
            # For the first year, use equal weights and no lambda adjustment
            current_weights = np.ones(len(models), dtype=float) / len(models)
            ensemble_preds[i] = np.tensordot(current_weights, model_preds[:, i, :], axes=(0, 0))
            lambda_values.append(0.0)
            weights_history.append(current_weights)
        else:
            # Use only data from previous years (0 to i-1) for training
            train_preds = model_preds[:, :i, :]  # (n_models, i, 12)
            train_actuals = reference_actual[:i, :]  # (i, 12)

            # Learn ensemble weights based on past data
            if weights is None:
                preds_flat = train_preds.transpose(1, 2, 0).reshape(-1, len(models))
                actual_flat = train_actuals.flatten()
                try:
                    learned_weights = np.linalg.lstsq(preds_flat, actual_flat, rcond=None)[0]
                except np.linalg.LinAlgError:
                    learned_weights = np.ones(len(models), dtype=float)

                # Clip and normalize weights
                learned_weights = np.clip(learned_weights, 0.0, None)
                if not np.isfinite(learned_weights).all() or learned_weights.sum() <= 0:
                    learned_weights = np.ones(len(models), dtype=float)
                learned_weights = learned_weights / learned_weights.sum()
            else:
                learned_weights = np.asarray(weights, dtype=float)
                if learned_weights.sum() == 0:
                    learned_weights = np.ones(len(models), dtype=float)
                learned_weights = learned_weights / learned_weights.sum()

            weights_history.append(learned_weights)

            # Create unbiased ensemble prediction for current year using learned weights
            pred_monthly_unbiased = np.tensordot(learned_weights, model_preds[:, i, :], axes=(0, 0))

            # Learn lambda (asymmetric bias) based on past data
            ensemble_train_preds = np.tensordot(learned_weights, train_preds, axes=(0, 0))  # (i, 12)

            if use_annual:
                y_fit = train_actuals.sum(axis=1)  # (i,)
                p_fit = ensemble_train_preds.sum(axis=1)  # (i,)
            else:
                y_fit = train_actuals.flatten()  # (i*12,)
                p_fit = ensemble_train_preds.flatten()  # (i*12,)

            lam = _fit_lambda_pinball(y_fit, p_fit, tau=float(tau))
            lambda_values.append(lam)

            # Apply the bias factor to current year's prediction
            ensemble_preds[i] = (1.0 + lam) * pred_monthly_unbiased

    # Calculate metrics on all predictions
    actual_annual = reference_actual.sum(axis=1)
    pred_annual = ensemble_preds.sum(axis=1)

    metrics = calculate_all_metrics(
        reference_actual.flatten(),
        ensemble_preds.flatten(),
        actual_annual,
        pred_annual,
    )

    component_strategies = [m["strategy"] for m in models]
    component_details = [
        {
            "strategy": m["strategy"],
            "n_lags": m.get("n_lags"),
            "feature_block": m.get("feature_block"),
        }
        for m in models
    ]

    # Use the final learned weights (or average of all weights)
    final_weights = weights_history[-1] if weights_history else np.ones(len(models)) / len(models)
    final_lambda = lambda_values[-1] if lambda_values else 0.0

    return {
        "strategy": strategy_name,
        "n_lags": tuple(m.get("n_lags") for m in models),
        "training_window": tuple(m.get("training_window", "N/A") for m in models),
        "feature_block": tuple(m.get("feature_block", "N/A") for m in models),
        "use_monthly_temp": any(m.get("use_monthly_temp") for m in models),
        "use_monthly_clients": any(m.get("use_monthly_clients") for m in models),
        "pc_weight": tuple(m.get("pc_weight", "N/A") for m in models),
        "pca_lambda": tuple(m.get("pca_lambda", "N/A") for m in models),
        "client_pattern_weight": tuple(
            m.get("client_pattern_weight", "N/A") for m in models
        ),
        "ensemble_weights": final_weights.tolist(),
        "ensemble_reference_strategies": component_strategies,
        "ensemble_component_details": component_details,
        "pred_monthly_matrix": ensemble_preds,
        "actual_monthly_matrix": reference_actual,
        "valid_years": np.array(ordered_years),
        "bias_lambda": final_lambda,
        "bias_tau": float(tau),
        "bias_fit_target": "annual" if use_annual else "monthly",
        "lambda_history": lambda_values,
        "weights_history": [w.tolist() for w in weights_history],
        **metrics,
    }

def _normalize_monthly_lookup(
    series_lookup: Optional[Mapping[int, Iterable[float]]],
) -> Dict[int, np.ndarray]:
    """
    Ensure the monthly series lookup maps year -> ndarray of length 12.
    Missing or malformed entries are dropped.
    """
    if not series_lookup:
        return {}

    normalized: Dict[int, np.ndarray] = {}
    for raw_year, values in series_lookup.items():
        try:
            year = int(raw_year)
        except (TypeError, ValueError):
            continue

        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != 12:
            continue

        normalized[year] = arr.copy()

    return normalized


def add_monthly_client_feature(
    X: np.ndarray, years: Iterable[int], series_lookup: Mapping[int, np.ndarray]
) -> np.ndarray:
    """
    Append a 12-month feature vector per year from `series_lookup`.
    Falls back to zeros when a year is missing.
    """
    if not series_lookup:
        return X

    rows = []
    for year in years:
        series = series_lookup.get(int(year))
        if series is None:
            rows.append(np.zeros(12, dtype=float))
        else:
            rows.append(series)

    if not rows:
        return X

    series_array = np.vstack(rows)
    return np.hstack([X, series_array])







def strategy1_ultra_strict_loocv(
    monthly_matrix,
    years,
    df_features,
    df_monthly,
    n_pcs,
    lags_options,
    feature_blocks,
    alphas,
    use_monthly_temp_options,
    use_pf_options = [False],
    monthly_clients_lookup=None,
    use_monthly_clients_options=[False],
    pca_lambdas=[1.0],
    training_end=None,
):
    """
    Strategy 1: PC-based prediction with ULTRA-STRICT LOOCV
    PCA is refitted in each fold using ONLY training data
    """
    results = []
    n_years = len(years)

    use_monthly_clients_options = (
        use_monthly_clients_options if monthly_clients_lookup else [False]
    )

    clients_lookup = monthly_clients_lookup or {}

    for n_lags, (fb_name, fb_features), lambda_value, use_temp, use_clients, use_pf in product(
        lags_options,
        feature_blocks.items(),
        pca_lambdas,
        use_monthly_temp_options,
        use_monthly_clients_options,
        use_pf_options
    ):
        if n_years <= n_lags:
            continue

        if use_clients and not clients_lookup:
            continue

        # Store predictions for all years
        all_predictions = []
        all_actuals = []
        valid_years_list = []

        # LOOCV: Leave one year out at a time
        for test_idx in range(n_lags, n_years):
            fold_training_end = (
                training_end if training_end is not None else years[test_idx] - 1
            )

            # Training data: all years except test year
            train_indices = [
                i
                for i in range(n_years)
                if (i != test_idx and years[i] <= fold_training_end)
            ]
            train_matrix = monthly_matrix[train_indices]

            # Fit PCA on TRAINING data only with power weights
            scaler_pca = StandardScaler()
            train_scaled = scaler_pca.fit_transform(train_matrix)
            train_years = np.array([years[idx] for idx in train_indices])
            sample_weights = compute_power_weights(train_years, lambda_value)
            pca_model = fit_weighted_pca(train_scaled, sample_weights, n_pcs)
            train_pc_scores = weighted_pca_transform(pca_model, train_scaled)
            effective_n_pcs = train_pc_scores.shape[1]

            if effective_n_pcs == 0:
                continue

            # Predict each PC component
            predicted_pcs = []

            for pc_idx in range(effective_n_pcs):
                # Create lagged features from training PC scores
                X_train_lags = []
                y_train = []
                train_years_for_pc = []

                for i in range(n_lags, len(train_pc_scores)):
                    lags = train_pc_scores[i - n_lags : i, pc_idx]
                    X_train_lags.append(lags)
                    y_train.append(train_pc_scores[i, pc_idx])
                    # Map back to actual year
                    actual_train_idx = train_indices[i]
                    train_years_for_pc.append(years[actual_train_idx])

                if len(X_train_lags) == 0:
                    predicted_pcs.append(0)
                    continue

                X_train_lags = np.array(X_train_lags).reshape(len(X_train_lags), -1)
                y_train = np.array(y_train)

                # Add exogenous features
                X_train = add_exogenous_features(
                    X_train_lags,
                    train_years_for_pc,
                    fb_features,
                    df_features,
                )

                # Add monthly temperature for PC2 and PC3
                if use_temp and pc_idx > 0:
                    X_train = add_monthly_feature(
                        X_train, train_years_for_pc, df_monthly
                    )
                if use_clients and pc_idx > 0:
                    X_train = add_monthly_client_feature(
                        X_train, train_years_for_pc, clients_lookup
                    )
                if use_pf and pc_idx > 0:
                    X_train = add_monthly_feature(
                        X_train, train_years_for_pc, df_monthly, feature=Aliases.PUISSANCE_FACTUREE, agg_method="sum"
                    )

                # Create test features (using past n_lags years)
                if test_idx < n_lags:
                    predicted_pcs.append(0)
                    continue

                # Get lagged PC scores for test year
                test_lag_indices = [i for i in range(test_idx - n_lags, test_idx)]
                test_lags_pc = []

                for lag_idx in test_lag_indices:
                    if lag_idx == test_idx:
                        # Can't use test year itself
                        predicted_pcs.append(0)
                        break
                    # Transform lag year to PC space
                    lag_scaled = scaler_pca.transform(
                        monthly_matrix[lag_idx : lag_idx + 1]
                    )
                    lag_pc = weighted_pca_transform(pca_model, lag_scaled)[0, pc_idx]
                    test_lags_pc.append(lag_pc)

                if len(test_lags_pc) != n_lags:
                    predicted_pcs.append(0)
                    continue

                X_test = np.array(test_lags_pc).reshape(1, -1)

                # Add exogenous features for test year
                X_test = add_exogenous_features(
                    X_test,
                    [years[test_idx]],
                    fb_features,
                    df_features,
                )

                # Add monthly temperature for test year
                if use_temp and pc_idx > 0:
                    X_test = add_monthly_feature(X_test, [years[test_idx]], df_monthly)
                if use_pf and pc_idx > 0:
                    X_test = add_monthly_feature(X_test, [years[test_idx]], df_monthly, feature=Aliases.PUISSANCE_FACTUREE, agg_method="sum")
                if use_clients and pc_idx > 0:
                    X_test = add_monthly_client_feature(
                        X_test,
                        [years[test_idx]],
                        clients_lookup,
                    )

                # Select best alpha using nested CV on training data
                best_alpha = alphas[0]
                best_mae = float("inf")
                if len(X_train) >= 2:
                    for alpha in alphas:
                        fold_errors = []
                        for i in range(len(X_train)):
                            X_tr = np.delete(X_train, i, axis=0)
                            y_tr = np.delete(y_train, i)
                            X_val = X_train[i : i + 1]
                            y_val = y_train[i]

                            scaler_x = StandardScaler()
                            X_tr_scaled = scaler_x.fit_transform(X_tr)
                            X_val_scaled = scaler_x.transform(X_val)

                            model = Ridge(alpha=alpha)
                            model.fit(X_tr_scaled, y_tr)
                            pred = model.predict(X_val_scaled)[0]
                            fold_errors.append(abs(pred - y_val))

                        mae = np.mean(fold_errors)
                        if mae < best_mae:
                            best_mae = mae
                            best_alpha = alpha

                # Train final model with best alpha
                scaler_x = StandardScaler()
                X_train_scaled = scaler_x.fit_transform(X_train)
                X_test_scaled = scaler_x.transform(X_test)

                model = Ridge(alpha=best_alpha)
                model.fit(X_train_scaled, y_train)

                pred_pc = model.predict(X_test_scaled)[0]
                predicted_pcs.append(pred_pc)

            # Reconstruct monthly consumption from predicted PCs
            predicted_pcs = np.array(predicted_pcs)
            reconstructed = weighted_pca_inverse_transform(
                pca_model, predicted_pcs.reshape(1, -1)
            )
            reconstructed = scaler_pca.inverse_transform(reconstructed)[0]

            all_predictions.append(reconstructed)
            all_actuals.append(monthly_matrix[test_idx])
            valid_years_list.append(years[test_idx])

        if len(all_predictions) == 0:
            continue

        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        actual_monthly = all_actuals.flatten()
        pred_monthly = all_predictions.flatten()
        actual_annual = all_actuals.sum(axis=1)
        pred_annual = all_predictions.sum(axis=1)

        metrics = calculate_all_metrics(
            actual_monthly, pred_monthly, actual_annual, pred_annual
        )

        results.append(
            {
                "strategy": "Strategy1_PC_UltraStrict",
                "n_lags": n_lags,
                "feature_block": fb_name,
                "use_monthly_temp": use_temp,
                "use_monthly_clients": bool(use_clients),
                "pca_lambda": lambda_value,
                "pc_weight": None,
                "pred_monthly_matrix": all_predictions,
                "actual_monthly_matrix": all_actuals,
                "valid_years": np.array(valid_years_list),
                **metrics,
            }
        )

    return results


def strategy2_ultra_strict_loocv(
    monthly_matrix,
    years,
    df_features,
    df_monthly,
    lags_options,
    feature_blocks,
    alphas,
    use_pf_options = [False],
    monthly_clients_lookup=None,
    use_monthly_clients_options=[False],
    client_pattern_weights=None,
    training_end=None,
):
    """
    Strategy 2: Direct annual prediction with mean curve distribution
    PCA refitted in each fold for distribution pattern
    """
    results = []
    n_years = len(years)
    annual_consumption = monthly_matrix.sum(axis=1)

    use_monthly_clients_options = (
        use_monthly_clients_options if monthly_clients_lookup else [False]
    )
    clients_lookup = monthly_clients_lookup or {}

    for n_lags, (fb_name, fb_features), use_clients, use_pf in product(
        lags_options, feature_blocks.items(), use_monthly_clients_options, use_pf_options
    ):
        if n_years <= n_lags:
            continue

        weight_options = (
            client_pattern_weights if (use_clients and clients_lookup) else [None]
        )
        predictions_by_weight = {w: [] for w in weight_options}
        actuals = []
        valid_years_list = []

        # LOOCV
        for test_idx in range(n_lags, n_years):
            fold_training_end = (
                training_end if training_end is not None else years[test_idx] - 1
            )
            # Training data
            train_indices = [
                i
                for i in range(n_years)
                if (i != test_idx and years[i] <= fold_training_end)
            ]
            train_matrix = monthly_matrix[train_indices]
            train_annual = annual_consumption[train_indices]

            # Create lagged features
            X_train_lags = []
            y_train = []
            train_years_list = []

            for i in range(n_lags, len(train_annual)):
                lags = train_annual[i - n_lags : i]
                X_train_lags.append(lags)
                y_train.append(train_annual[i])
                actual_train_idx = train_indices[i]
                train_years_list.append(years[actual_train_idx])

            if len(X_train_lags) == 0:
                continue

            X_train_lags = np.array(X_train_lags)
            y_train = np.array(y_train)

            # Add exogenous features
            X_train = add_exogenous_features(
                X_train_lags, train_years_list, fb_features, df_features
            )
            if use_clients and clients_lookup:
                X_train = add_annual_client_feature(
                    X_train, train_years_list, clients_lookup
                )
            if use_pf:
                X_train = add_yearly_feature(
                    X_train, train_years_list, df_monthly, feature=Aliases.PUISSANCE_FACTUREE, agg_method="sum"
                )

            # Create test features
            test_lags = annual_consumption[test_idx - n_lags : test_idx]
            X_test = test_lags.reshape(1, -1)
            X_test = add_exogenous_features(
                X_test, [years[test_idx]], fb_features, df_features
            )
            if use_clients and clients_lookup:
                X_test = add_annual_client_feature(
                    X_test, [years[test_idx]], clients_lookup
                )
            if use_pf:
                X_test = add_yearly_feature(
                    X_test, [years[test_idx]], df_monthly, feature=Aliases.PUISSANCE_FACTUREE, agg_method="sum"
                )

            if years[test_idx] <= fold_training_end + 1:
                # Select best alpha
                best_alpha = alphas[0]
                best_mae = float("inf")
                if len(X_train) >= 2:
                    for alpha in alphas:
                        fold_errors = []
                        for i in range(len(X_train)):
                            X_tr = np.delete(X_train, i, axis=0)
                            y_tr = np.delete(y_train, i)
                            X_val = X_train[i : i + 1]
                            y_val = y_train[i]

                            scaler_x = StandardScaler()
                            X_tr_scaled = scaler_x.fit_transform(X_tr)
                            X_val_scaled = scaler_x.transform(X_val)

                            model = Ridge(alpha=alpha)
                            model.fit(X_tr_scaled, y_tr)
                            pred = model.predict(X_val_scaled)[0]
                            fold_errors.append(abs(pred - y_val))

                        mae = np.mean(fold_errors)
                        if mae < best_mae:
                            best_mae = mae
                            best_alpha = alpha

                # Train final model
                scaler_x = StandardScaler()
                X_train_scaled = scaler_x.fit_transform(X_train)
                X_test_scaled = scaler_x.transform(X_test)

                model = Ridge(alpha=best_alpha)
                model.fit(X_train_scaled, y_train)

            pred_annual = model.predict(X_test_scaled)[0]

            # Distribute using blended pattern
            mean_curve = train_matrix.mean(axis=0)
            mean_curve_normalized = mean_curve / mean_curve.sum()

            client_curve = None
            if use_clients and clients_lookup:
                client_curve = clients_lookup.get(int(years[test_idx]))
            if client_curve is not None and client_curve.sum() > 0:
                client_curve_normalized = client_curve / client_curve.sum()
            else:
                client_curve_normalized = mean_curve_normalized

            actuals.append(monthly_matrix[test_idx])
            valid_years_list.append(years[test_idx])

            for weight in weight_options:
                if weight is None:
                    blended_pattern = mean_curve_normalized
                else:
                    blended_pattern = (
                        1 - weight
                    ) * mean_curve_normalized + weight * client_curve_normalized
                pred_monthly = pred_annual * blended_pattern
                predictions_by_weight[weight].append(pred_monthly)

        for weight in weight_options:
            all_predictions = np.array(predictions_by_weight[weight])
            if all_predictions.size == 0:
                continue

            all_actuals = np.array(actuals)

            actual_monthly = all_actuals.flatten()
            pred_monthly = all_predictions.flatten()
            actual_annual = all_actuals.sum(axis=1)
            pred_annual = all_predictions.sum(axis=1)

            metrics = calculate_all_metrics(
                actual_monthly, pred_monthly, actual_annual, pred_annual
            )

            results.append(
                {
                    "strategy": "Strategy2_MeanCurve_UltraStrict",
                    "n_lags": n_lags,
                    "feature_block": fb_name,
                    "use_monthly_temp": False,
                    "use_monthly_clients": bool(use_clients and clients_lookup),
                    "client_pattern_weight": weight if weight is not None else None,
                    "pc_weight": None,
                    "pred_monthly_matrix": all_predictions,
                    "actual_monthly_matrix": all_actuals,
                    "valid_years": np.array(valid_years_list),
                    **metrics,
                }
            )

    return results


def strategy3_ultra_strict_loocv(
    monthly_matrix,
    years,
    df_features,
    df_monthly,
    n_pcs,
    lags_options,
    feature_blocks,
    alphas,
    pc_weights,
    use_monthly_temp_options,
    use_pf_options = [False],
    monthly_clients_lookup=None,
    use_monthly_clients_options=[False],
    client_pattern_weights=None,
    pca_lambdas=[1.0],
    training_end=None,
):
    """
    Strategy 3 FIXED: Predict PCs + Weighted Historical Pattern

    1. Predict PC scores for test year (no leakage)
    2. Reconstruct pattern from predicted PCs
    3. Calculate weighted mean curve from training years (recent = more weight)
    4. Combine both patterns with pc_weight
    5. Predict annual consumption
    6. Distribute using combined pattern
    """
    results = []
    n_years = len(years)
    annual_consumption = monthly_matrix.sum(axis=1)

    use_monthly_clients_options = (
        use_monthly_clients_options if monthly_clients_lookup else [False]
    )
    clients_lookup = monthly_clients_lookup or {}

    for (
        n_lags,
        (fb_name, fb_features),
        pc_weight,
        lambda_value,
        use_clients,
        use_temp,
        use_pf,
    ) in product(
        lags_options,
        feature_blocks.items(),
        pc_weights,
        pca_lambdas,
        use_monthly_clients_options,
        use_monthly_temp_options,
        use_pf_options,
    ):
        if n_years <= n_lags:
            continue

        weight_options = (
            client_pattern_weights if (use_clients and clients_lookup) else [None]
        )
        predictions_by_weight = {w: [] for w in weight_options}
        actuals = []
        valid_years_list = []

        # LOOCV
        for test_idx in range(n_lags, n_years):
            fold_training_end = (
                training_end if training_end is not None else years[test_idx] - 1
            )
            train_indices = [
                i
                for i in range(n_years)
                if (i != test_idx and years[i] <= fold_training_end)
            ]
            train_matrix = monthly_matrix[train_indices]
            train_annual = annual_consumption[train_indices]

            # Fit PCA on training data only with power weights
            scaler_pca = StandardScaler()
            train_scaled = scaler_pca.fit_transform(train_matrix)
            train_years = np.array([years[idx] for idx in train_indices])
            sample_weights = compute_power_weights(train_years, lambda_value)
            pca_model = fit_weighted_pca(train_scaled, sample_weights, n_pcs)
            train_pc_scores = weighted_pca_transform(pca_model, train_scaled)
            effective_n_pcs = train_pc_scores.shape[1]

            if effective_n_pcs == 0:
                continue

            # Predict each PC for test year
            predicted_pcs = []

            for pc_idx in range(effective_n_pcs):
                X_train_lags = []
                y_train_pc = []
                train_years_for_pc = []

                for i in range(n_lags, len(train_pc_scores)):
                    lags = train_pc_scores[i - n_lags : i, pc_idx]
                    X_train_lags.append(lags)
                    y_train_pc.append(train_pc_scores[i, pc_idx])
                    actual_train_idx = train_indices[i]
                    train_years_for_pc.append(years[actual_train_idx])

                if len(X_train_lags) == 0:
                    predicted_pcs.append(0)
                    continue

                X_train_lags = np.array(X_train_lags).reshape(len(X_train_lags), -1)
                y_train_pc = np.array(y_train_pc)

                X_train = add_exogenous_features(
                    X_train_lags,
                    train_years_for_pc,
                    fb_features,
                    df_features,
                )
                if pc_idx > 0 and clients_lookup and use_clients:
                    X_train = add_monthly_client_feature(
                        X_train, train_years_for_pc, clients_lookup
                    )

                if use_temp and pc_idx > 0:
                    X_train = add_monthly_feature(
                        X_train, train_years_for_pc, df_monthly
                    )
                if use_pf and pc_idx > 0:
                    X_train = add_monthly_feature(
                        X_train, train_years_for_pc, df_monthly, feature=Aliases.PUISSANCE_FACTUREE, agg_method="sum"
                    )

                test_lag_indices = list(range(test_idx - n_lags, test_idx))
                test_lags_pc = []

                for lag_idx in test_lag_indices:
                    lag_scaled = scaler_pca.transform(
                        monthly_matrix[lag_idx : lag_idx + 1]
                    )
                    lag_pc = weighted_pca_transform(pca_model, lag_scaled)[0, pc_idx]
                    test_lags_pc.append(lag_pc)

                if len(test_lags_pc) != n_lags:
                    predicted_pcs.append(0)
                    continue

                X_test = np.array(test_lags_pc).reshape(1, -1)
                X_test = add_exogenous_features(
                    X_test, [years[test_idx]], fb_features, df_features
                )

                if pc_idx > 0 and clients_lookup and use_clients:
                    X_test = add_monthly_client_feature(
                        X_test, [years[test_idx]], clients_lookup
                    )

                if use_temp and pc_idx > 0:
                    X_test = add_monthly_feature(X_test, [years[test_idx]], df_monthly)
                
                if use_pf and pc_idx > 0:
                    X_test = add_monthly_feature(X_test, [years[test_idx]], df_monthly, feature=Aliases.PUISSANCE_FACTUREE, agg_method="sum")

                best_alpha = alphas[0]
                best_mae = float("inf")

                if len(X_train) >= 2:
                    for alpha in alphas:
                        fold_errors = []
                        for i in range(len(X_train)):
                            X_tr = np.delete(X_train, i, axis=0)
                            y_tr = np.delete(y_train_pc, i)
                            X_val = X_train[i : i + 1]
                            y_val = y_train_pc[i]

                            scaler_x = StandardScaler()
                            X_tr_scaled = scaler_x.fit_transform(X_tr)
                            X_val_scaled = scaler_x.transform(X_val)

                            model = Ridge(alpha=alpha)
                            model.fit(X_tr_scaled, y_tr)
                            pred = model.predict(X_val_scaled)[0]
                            fold_errors.append(abs(pred - y_val))

                        mae = np.mean(fold_errors)
                        if mae < best_mae:
                            best_mae = mae
                            best_alpha = alpha

                scaler_x = StandardScaler()
                X_train_scaled = scaler_x.fit_transform(X_train)
                X_test_scaled = scaler_x.transform(X_test)

                model = Ridge(alpha=best_alpha)
                model.fit(X_train_scaled, y_train_pc)

                pred_pc = model.predict(X_test_scaled)[0]
                predicted_pcs.append(pred_pc)

            predicted_pcs = np.array(predicted_pcs)
            predicted_pattern = weighted_pca_inverse_transform(
                pca_model, predicted_pcs.reshape(1, -1)
            )
            predicted_pattern = scaler_pca.inverse_transform(predicted_pattern)[0]

            if predicted_pattern.sum() > 0:
                predicted_pattern_normalized = (
                    predicted_pattern / predicted_pattern.sum()
                )
            else:
                predicted_pattern_normalized = np.ones(12) / 12

            decay_rate = 0.1

            train_years_list = [years[i] for i in train_indices]
            test_year = years[test_idx]

            weights = []
            for train_year in train_years_list:
                years_ago = test_year - train_year
                weight = np.exp(-decay_rate * years_ago)
                weights.append(weight)

            weights = np.array(weights)
            weights = weights / weights.sum()

            weighted_mean_curve = np.zeros(12)
            for i, train_idx in enumerate(train_indices):
                weighted_mean_curve += weights[i] * train_matrix[i]

            weighted_mean_curve_normalized = (
                weighted_mean_curve / weighted_mean_curve.sum()
            )

            client_curve = None
            if use_clients and clients_lookup:
                client_curve = clients_lookup.get(int(test_year))
            if client_curve is not None and client_curve.sum() > 0:
                client_curve_normalized = client_curve / client_curve.sum()
            else:
                client_curve_normalized = weighted_mean_curve_normalized

            X_train_annual_lags = []
            y_train_annual = []
            train_years_annual = []

            for i in range(n_lags, len(train_annual)):
                lags = train_annual[i - n_lags : i]
                X_train_annual_lags.append(lags)
                y_train_annual.append(train_annual[i])
                actual_train_idx = train_indices[i]
                train_years_annual.append(years[actual_train_idx])

            if len(X_train_annual_lags) == 0:
                continue

            X_train_annual_lags = np.array(X_train_annual_lags)
            y_train_annual = np.array(y_train_annual)

            X_train_annual = add_exogenous_features(
                X_train_annual_lags,
                train_years_annual,
                fb_features,
                df_features,
            )
            if use_clients and clients_lookup:
                X_train_annual = add_annual_client_feature(
                    X_train_annual, train_years_annual, clients_lookup
                )
            if use_pf:
                X_train_annual = add_yearly_feature(
                    X_train_annual, train_years_annual, df_monthly, feature=Aliases.PUISSANCE_FACTUREE, agg_method="sum"
                )

            test_lags_annual = annual_consumption[test_idx - n_lags : test_idx]
            X_test_annual = test_lags_annual.reshape(1, -1)
            X_test_annual = add_exogenous_features(
                X_test_annual,
                [years[test_idx]],
                fb_features,
                df_features,
            )

            if use_clients and clients_lookup:
                X_test_annual = add_annual_client_feature(
                    X_test_annual, [years[test_idx]], clients_lookup
                )
            if use_pf:
                X_test_annual = add_yearly_feature(
                    X_test_annual, [years[test_idx]], df_monthly, feature=Aliases.PUISSANCE_FACTUREE, agg_method="sum"
                )

            if years[test_idx] <= fold_training_end + 1:
                best_alpha_annual = alphas[0]
                best_mae_annual = float("inf")
                if len(X_train_annual) >= 2:
                    for alpha in alphas:
                        fold_errors = []
                        for i in range(len(X_train_annual)):
                            X_tr = np.delete(X_train_annual, i, axis=0)
                            y_tr = np.delete(y_train_annual, i)
                            X_val = X_train_annual[i : i + 1]
                            y_val = y_train_annual[i]

                            scaler_x = StandardScaler()
                            X_tr_scaled = scaler_x.fit_transform(X_tr)
                            X_val_scaled = scaler_x.transform(X_val)

                            model = Ridge(alpha=alpha)
                            model.fit(X_tr_scaled, y_tr)
                            pred = model.predict(X_val_scaled)[0]
                            fold_errors.append(abs(pred - y_val))

                        mae = np.mean(fold_errors)
                        if mae < best_mae_annual:
                            best_mae_annual = mae
                            best_alpha_annual = alpha

                scaler_annual = StandardScaler()
                X_train_annual_scaled = scaler_annual.fit_transform(X_train_annual)
                X_test_annual_scaled = scaler_annual.transform(X_test_annual)

                model_annual = Ridge(alpha=best_alpha_annual)
                model_annual.fit(X_train_annual_scaled, y_train_annual)

            pred_annual = model_annual.predict(X_test_annual_scaled)[0]

            actuals.append(monthly_matrix[test_idx])
            valid_years_list.append(years[test_idx])

            for weight in weight_options:
                if weight is None:
                    enhanced_mean_curve = weighted_mean_curve_normalized
                else:
                    enhanced_mean_curve = (
                        (1 - weight) * weighted_mean_curve_normalized
                        + weight * client_curve_normalized
                    )

                combined_pattern = (
                    pc_weight * predicted_pattern_normalized
                    + (1 - pc_weight) * enhanced_mean_curve
                )

                pred_monthly = pred_annual * combined_pattern
                predictions_by_weight[weight].append(pred_monthly)

        for weight in weight_options:
            all_predictions = np.array(predictions_by_weight[weight])
            if all_predictions.size == 0:
                continue

            all_actuals = np.array(actuals)

            actual_monthly = all_actuals.flatten()
            pred_monthly = all_predictions.flatten()
            actual_annual = all_actuals.sum(axis=1)
            pred_annual = all_predictions.sum(axis=1)

            metrics = calculate_all_metrics(
                actual_monthly, pred_monthly, actual_annual, pred_annual
            )

            results.append(
                {
                    "strategy": "Strategy3_PredictedPC_WeightedHistory",
                    "n_lags": n_lags,
                    "feature_block": fb_name,
                    "use_monthly_temp": False,
                    "use_monthly_clients": bool(use_clients and clients_lookup),
                    "pca_lambda": lambda_value,
                    "pc_weight": pc_weight,
                    "client_pattern_weight": weight if weight is not None else None,
                    "pred_monthly_matrix": all_predictions,
                    "actual_monthly_matrix": all_actuals,
                    "valid_years": np.array(valid_years_list),
                    **metrics,
                }
            )

    return results


def strategy4_ultra_strict_loocv(
    monthly_matrix,
    years,
    df_features,
    df_monthly,
    feature_blocks,
    monthly_clients_lookup=None,
    client_pattern_weights=None,
    use_monthly_clients_options=[False],
    use_pf_options = [False],
    growth_feature_transforms: Iterable[Iterable[str]] = (("level",),),
    growth_feature_lags: Iterable[Iterable[int]] = ((1,),),
    training_windows=[10],
):

    results = []
    n_years = len(years)
    annual_consumption = monthly_matrix.sum(axis=1)

    use_monthly_clients_options = (
        use_monthly_clients_options if monthly_clients_lookup else [False]
    )
    clients_lookup = monthly_clients_lookup or {}

    # Normalize/compatibility: callers may pass a single iterable of transforms/lags
    # (e.g. ("lag_lchg",)) or a list of such iterables. We want a list of tuples
    # where each element is an iterable of transforms or lags respectively.
    if isinstance(growth_feature_transforms, (list, tuple)) and all(
        isinstance(x, str) for x in growth_feature_transforms
    ):
        transforms_options = [tuple(growth_feature_transforms)]
    else:
        transforms_options = [tuple(x) for x in growth_feature_transforms]

    if isinstance(growth_feature_lags, (list, tuple)) and all(
        isinstance(x, (int, np.integer)) for x in growth_feature_lags
    ):
        lags_options = [tuple(growth_feature_lags)]
    else:
        lags_options = [tuple(x) for x in growth_feature_lags]

    # Iterate over combinations including growth feature transform/lags option-sets
    for (
        (fb_name, fb_features),
        training_window,
        use_clients,
        use_pf,
        transforms_option,
        lags_option,
    ) in product(
        feature_blocks.items(),
        training_windows,
        use_monthly_clients_options,
        use_pf_options,
        transforms_options,
        lags_options,
    ):
        weight_options = (
            client_pattern_weights if (use_clients and clients_lookup) else [None]
        )
        predictions_by_weight = {w: [] for w in weight_options}
        actuals = []
        valid_years_list = []
        growth_model = None

        # LOOCV
        for test_idx in range(2, n_years):
            # Training data
            if training_window is not None:
                start_idx = max(0, test_idx - training_window)
            else:
                start_idx = 0
            train_indices = [
                i for i in range(n_years) if (i < test_idx and i >= start_idx)
            ]
            train_matrix = monthly_matrix[train_indices]
            train_annual = annual_consumption[train_indices]
            train_years = [years[idx] for idx in train_indices]

            if len(train_years) < 3:
                continue

            train_features = build_growth_rate_features(
                train_years,
                fb_features,
                df_features,
                clients_lookup=clients_lookup if use_clients else None,
                use_clients = bool(use_clients and clients_lookup),
                df_monthly=df_monthly,
                use_pf=use_pf,
                transforms=transforms_option,
                lags=lags_option,
            )
            include_exog = bool(
                train_features is not None and train_features.shape[1] > 0
            )

            if include_exog:
                train_features_array = np.asarray(train_features, dtype=float)
                feature_dim = train_features_array.shape[1]
            else:
                train_features_array = None
                feature_dim = 0

            train_years_array = np.asarray(train_years, dtype=int)
            sort_idx = np.argsort(train_years_array)
            train_years_array = train_years_array[sort_idx]
            train_annual_sorted = np.asarray(train_annual, dtype=float)[sort_idx]
            if include_exog:
                train_features_array = train_features_array[sort_idx]

            test_year = int(years[test_idx])
            test_features = build_growth_rate_features(
                [test_year],
                fb_features,
                df_features,
                clients_lookup=clients_lookup if use_clients else None,
                use_clients=bool(use_clients and clients_lookup),
                df_monthly=df_monthly,
                use_pf=use_pf,
                transforms=transforms_option,
                lags=lags_option,
            )
    
            if include_exog:
                if test_features is None or test_features.shape[1] != feature_dim:
                    test_features_array = np.zeros((1, feature_dim), dtype=float)
                else:
                    test_features_array = np.asarray(test_features, dtype=float)
            else:
                test_features_array = None



            growth_model = MeanRevertingGrowthModel(
                include_ar=True,
                include_exog=include_exog,
            )
            try:
                growth_model.fit(
                    y = train_annual_sorted,
                    X = train_features_array if include_exog else None,
                )
            except ValueError:
                growth_model = None

            if growth_model is None:
                continue

            if include_exog:
                x_next = test_features_array[0]
            else:
                x_next = None

            pred_annual = growth_model.predict(x_next)

            # Distribute using blended pattern
            mean_curve = train_matrix.mean(axis=0)
            mean_curve_normalized = mean_curve / mean_curve.sum()

            client_curve = None
            if use_clients and clients_lookup:
                client_curve = clients_lookup.get(int(years[test_idx]))
            if client_curve is not None and client_curve.sum() > 0:
                client_curve_normalized = client_curve / client_curve.sum()
            else:
                client_curve_normalized = mean_curve_normalized

            actuals.append(monthly_matrix[test_idx])
            valid_years_list.append(years[test_idx])

            for weight in weight_options:
                if weight is None:
                    blended_pattern = mean_curve_normalized
                else:
                    blended_pattern = (
                        1 - weight
                    ) * mean_curve_normalized + weight * client_curve_normalized
                pred_monthly = pred_annual * blended_pattern
                predictions_by_weight[weight].append(pred_monthly)

        for weight in weight_options:
            all_predictions = np.array(predictions_by_weight[weight])
            if all_predictions.size == 0:
                continue

            all_actuals = np.array(actuals)

            actual_monthly = all_actuals.flatten()
            pred_monthly = all_predictions.flatten()
            actual_annual = all_actuals.sum(axis=1)
            pred_annual = all_predictions.sum(axis=1)

            metrics = calculate_all_metrics(
                actual_monthly, pred_monthly, actual_annual, pred_annual
            )

            results.append(
                {
                    "strategy": "Strategy4_growth_rate",
                    "n_lags": None,
                    "feature_block": fb_name,
                    "use_monthly_temp": False,
                    "use_pf": use_pf,
                    "use_monthly_clients": bool(use_clients and clients_lookup),
                    "client_pattern_weight": weight if weight is not None else None,
                    "pc_weight": None,
                    "growth_feature_transforms": tuple(transforms_option),
                    "growth_feature_lags": tuple(lags_option),
                    "pred_monthly_matrix": all_predictions,
                    "actual_monthly_matrix": all_actuals,
                    "valid_years": np.array(valid_years_list),
                    "training_window": training_window,
                    **metrics,
                }
            )

    return results


def compute_empirical_CIs(best, target_year=2023, alpha=0.05):
    """
    Build non-parametric prediction intervals using LOOCV residuals.
    Returns:
      monthly_ci: (12, 2) array of [lower, upper] per month in same UNIT as predictions
      annual_ci:  (2,) array of [lower, upper] for annual total
    """
    valid_years = np.array(best["valid_years"])
    actual = np.array(best["actual_monthly_matrix"])  # shape: (n_years, 12)
    pred = np.array(best["pred_monthly_matrix"])  # shape: (n_years, 12)

    # Exclude target year from residual pool if present
    mask = valid_years != target_year
    if mask.sum() == 0:
        # Fallback: if there's only the target year, we cannot compute CIs
        return None, None

    resid_monthly = actual[mask] - pred[mask]  # residuals in original unit
    # Percentile bounds for residuals
    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)

    # For monthly CIs, do percentiles month-by-month
    resid_lo = np.percentile(resid_monthly, lo_q, axis=0)
    resid_hi = np.percentile(resid_monthly, hi_q, axis=0)

    monthly_ci_residuals = np.vstack([resid_lo, resid_hi]).T  # (12,2)

    # Annual residuals (sum over months per year)
    resid_annual = resid_monthly.sum(axis=1)
    annual_lo = np.percentile(resid_annual, lo_q)
    annual_hi = np.percentile(resid_annual, hi_q)

    return monthly_ci_residuals, np.array([annual_lo, annual_hi])

def run_analysis_for_entity(
    df,
    entity_name,
    df_features,
    df_monthly=None,
    *,
    config: dict,  # <- everything comes from here
    client_predictions=None,
    favor_overestimation=True,
    under_estimation_penalty=2,
    **kwargs,
):
    """
    Required keys in `config`:
      - value_col (str)
      - N_PCS (int)
      - LAGS_OPTIONS (list[int])
      - FEATURE_BLOCKS (dict[str, list[str]])
      - ALPHAS (list[float])
      - PC_WEIGHTS (list[float])
      - PCA_LAMBDAS (list[float], optional; defaults to [1.0])
      - R2_THRESHOLD (float)
      - unit (str)   # optional; defaults to 'kWh'
      - growth_feature_transforms (Iterable[str], optional)
      - growth_feature_lags (Iterable[int], optional)
    """
    # --- minimal validation (kept very lightweight) ---
    required = [
        "value_col",
        "N_PCS",
        "LAGS_OPTIONS",
        "FEATURE_BLOCKS",
        "ALPHAS",
        "PC_WEIGHTS",
        "R2_THRESHOLD",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    value_col = config["value_col"]
    N_PCS = config["N_PCS"]
    LAGS_OPTIONS = config["LAGS_OPTIONS"]
    FEATURE_BLOCKS = config["FEATURE_BLOCKS"]
    ALPHAS = config["ALPHAS"]
    PC_WEIGHTS = config["PC_WEIGHTS"]
    PCA_LAMBDAS = config.get("PCA_LAMBDAS", [0.3, 0.7, 1.0])
    R2_THRESHOLD = config["R2_THRESHOLD"]
    unit = config.get("unit", "kWh")
    training_end = config.get("training_end", 2020)
    use_monthly_temp_options = config.get("use_monthly_temp_options", [False])
    use_monthly_clients_options = config.get("use_monthly_clients_options", [False])
    use_pf_options = config.get("use_pf_options", [False])
    client_pattern_weights = config.get("client_pattern_weights", [0.3, 0.5, 0.8])
    growth_feature_transforms = config.get("growth_feature_transforms", [("lag_lchg",)])
    growth_feature_lags = config.get("growth_feature_lags", [(1,)])
    training_windows = config.get("training_windows", [4, 7, 10])
    train_start_year = config.get("train_start_year", 2007)
    eval_years_start = config.get("eval_years_start", 2021)
    eval_years_end = config.get("eval_years_end", 2023)

    print(f"\n{'='*60}")
    print(f"ANALYZING: {entity_name}")
    print(f"{'='*60}")

    move_in_year = get_move_in_year(df)
    if move_in_year is not None:
        move_in_year = move_in_year - 2
        # Ensure valid_data only includes years after the move-in date
        valid_data = df[df[Aliases.ANNEE].between(max(train_start_year, move_in_year), eval_years_end)]
        print(f"📅 Move-in year detected: {move_in_year} → filtering from {max(train_start_year, move_in_year)} to {eval_years_end}")
    else:
        valid_data = df[df[Aliases.ANNEE].between(train_start_year, eval_years_end)]

    if valid_data.empty:
        print(f"⚠️ No valid data for {entity_name} after applying move-in year filter.")
        return None

    monthly_matrix = create_monthly_matrix(valid_data, value_col=value_col)
    years = np.array(sorted(valid_data[Aliases.ANNEE].unique()))
    clients_lookup = _normalize_monthly_lookup(client_predictions)

    # Run all strategies with ULTRA-STRICT LOOCV
    s1_results = strategy1_ultra_strict_loocv(
        monthly_matrix=monthly_matrix,
        years=years,
        df_features=df_features,
        df_monthly=df_monthly if df_monthly is not None else df,
        n_pcs=N_PCS,
        lags_options=LAGS_OPTIONS,
        feature_blocks=FEATURE_BLOCKS,
        alphas=ALPHAS,
        use_monthly_temp_options=use_monthly_temp_options,
        use_monthly_clients_options=use_monthly_clients_options,
        use_pf_options=use_pf_options,
        monthly_clients_lookup=clients_lookup,
        pca_lambdas=PCA_LAMBDAS,
        training_end=training_end,
    )

    s2_results = strategy2_ultra_strict_loocv(
        monthly_matrix=monthly_matrix,
        years=years,
        df_features=df_features,
        df_monthly=df_monthly if df_monthly is not None else df,
        lags_options=LAGS_OPTIONS,
        feature_blocks=FEATURE_BLOCKS,
        alphas=ALPHAS,
        monthly_clients_lookup=clients_lookup,
        use_monthly_clients_options=use_monthly_clients_options,
        use_pf_options=use_pf_options,
        client_pattern_weights=client_pattern_weights,
        training_end=training_end,
    )

    s3_results = strategy3_ultra_strict_loocv(
        monthly_matrix=monthly_matrix,
        years=years,
        df_features=df_features,
        df_monthly=df_monthly if df_monthly is not None else df,
        n_pcs=N_PCS,
        lags_options=LAGS_OPTIONS,
        feature_blocks=FEATURE_BLOCKS,
        alphas=ALPHAS,
        pc_weights=PC_WEIGHTS,
        use_monthly_temp_options=use_monthly_temp_options,
        use_monthly_clients_options=use_monthly_clients_options,
        use_pf_options=use_pf_options,
        monthly_clients_lookup=clients_lookup,
        client_pattern_weights=client_pattern_weights,
        pca_lambdas=PCA_LAMBDAS,
        training_end=training_end,
    )
    
    s4_results = strategy4_ultra_strict_loocv(
        monthly_matrix=monthly_matrix,
        years=years,
        df_features=df_features,
        df_monthly=df_monthly if df_monthly is not None else df,
        feature_blocks=FEATURE_BLOCKS,
        monthly_clients_lookup=clients_lookup,
        use_monthly_clients_options=use_monthly_clients_options,
        use_pf_options=use_pf_options,
        client_pattern_weights=client_pattern_weights,
        growth_feature_transforms=growth_feature_transforms,
        growth_feature_lags=growth_feature_lags,
        training_windows=training_windows,
    )

    ensemble_results = []
    best_s1 = select_best_with_fallback(s1_results, R2_THRESHOLD)
    best_s2 = select_best_with_fallback(s2_results, R2_THRESHOLD)
    best_s3 = select_best_with_fallback(s3_results, R2_THRESHOLD)
    best_s4 = select_best_with_fallback(s4_results, R2_THRESHOLD)

    # Function to validate model predictions
    def is_valid_model(model):
        if model is None:
            return False
        pred_matrix = model.get('pred_monthly_matrix')
        if pred_matrix is None:
            return False
        # Check for NaN or Inf values in predictions
        return not (np.isnan(pred_matrix).any() or np.isinf(pred_matrix).any())

    # Filter out invalid models
    valid_models = []
    model_names = ['Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4']
    for model, name in zip([best_s1, best_s2, best_s3, best_s4], model_names):
        if not is_valid_model(model):
            print(f"⚠️  {name} produced invalid predictions (NaN or Inf) - excluding from ensemble")
        else:
            valid_models.append(model)

    if len(valid_models) >= 2:  # Require at least 2 valid models for ensemble
        tau = under_estimation_penalty / (1 + under_estimation_penalty)
        ensemble_candidate = build_ensemble_result(
            valid_models, tau=tau, use_annual=False
        )
        if ensemble_candidate is not None:
            ensemble_results.append(ensemble_candidate)
            print(f"✓ Created ensemble using {len(valid_models)} valid strategies")

    # Select best
    if favor_overestimation and len(ensemble_results) > 0:
        all_results = ensemble_results
    else:
        all_results = (
            ensemble_results + s1_results + s2_results + s3_results + s4_results
        )

    if len(all_results) == 0:
        print("⚠️  No valid results")
        return None

    best = select_best_model(all_results, R2_THRESHOLD)

    # Print best model performance
    print(f"\n🏆 BEST MODEL: {best['strategy']}")
    config_parts = [
        f"lags={best['n_lags']}",
        f"features={best['feature_block']}",
    ]
    if best.get("pc_weight") is not None:
        config_parts.append(f"pc_weight={best['pc_weight']}")
    if best.get("pca_lambda") is not None:
        config_parts.append(f"pca_lambda={best['pca_lambda']}")
    if best.get("use_monthly_temp"):
        config_parts.append(f"monthly_temp={best['use_monthly_temp']}")
    if best.get("use_monthly_clients"):
        config_parts.append(f"monthly_clients={best['use_monthly_clients']}")
    if best.get("client_pattern_weight") is not None:
        config_parts.append(f"client_pattern_weight={best['client_pattern_weight']}")
    if best.get("ensemble_reference_strategies"):
        ref = "/".join(best["ensemble_reference_strategies"])
        config_parts.append(f"ensemble_of={ref}")
    if best.get("ensemble_weights"):
        config_parts.append(f"ensemble_weights={best['ensemble_weights']}")
    print("   Config: " + ", ".join(config_parts))

    print(f"\n   📊 ANNUAL METRICS:")
    print(f"      MAE:  {best['annual_mae']:,.2f} {unit}")
    print(f"      MAPE: {best['annual_mape']:.2f}%")
    print(f"      R²:   {best['annual_r2']:.4f}")

    print(f"\n   📊 MONTHLY METRICS:")
    print(f"      MAE:  {best['monthly_mae']:,.2f} {unit}")
    print(f"      MAPE: {best['monthly_mape']:.2f}%")
    print(f"      R²:   {best['monthly_r2']:.4f}")

    monthly_details = (
        {}
    )  # year -> DataFrame (Month, Actual, Predicted, Error_%, [PI_low, PI_high])

    # Latest year predictions
    test_years = {}

    monthly_ci_residuals, annual_ci_residuals = compute_empirical_CIs(
        best, target_year=eval_years_end, alpha=0.05
    )

    if eval_years_end in best["valid_years"]:
        idx_latest = np.where(best["valid_years"] == eval_years_end)[0][0]
        actual_latest = best["actual_monthly_matrix"][idx_latest]
        pred_latest = best["pred_monthly_matrix"][idx_latest]
        error_pct_latest = []

        if monthly_ci_residuals is not None:
            monthly_ci_latest = np.zeros((12, 2))
            for m in range(12):
                lo_resid, hi_resid = monthly_ci_residuals[m]
                monthly_ci_latest[m, 0] = pred_latest[m] + lo_resid
                monthly_ci_latest[m, 1] = pred_latest[m] + hi_resid
            # Annual CI
            annual_ci_latest = None
            if annual_ci_residuals is not None:
                annual_ci_latest = np.array(
                    [
                        pred_latest.sum() + annual_ci_residuals[0],
                        pred_latest.sum() + annual_ci_residuals[1],
                    ]
                )
        else:
            monthly_ci_latest = None
            annual_ci_latest = None

        print(f"\n   📅 {eval_years_end} MONTHLY PREDICTIONS:")
        print(f"      Month | Actual ({unit}) | Predicted ({unit}) | Error (%)")
        print(f"      {'-'*55}")
        for m in range(12):
            denom = actual_latest[m]
            error_pct = (
                np.nan if denom == 0 else ((pred_latest[m] - denom) / denom) * 100
            )
            error_pct_latest.append(error_pct)
            print(
                f"      {m+1:2d}    | {actual_latest[m]:12,.0f} | {pred_latest[m]:15,.0f} | {error_pct:7.2f}%"
            )

        test_years[f"{eval_years_end}_actual_monthly"] = actual_latest
        test_years[f"{eval_years_end}_predicted_monthly"] = pred_latest
        test_years[f"{eval_years_end}_error_pct_monthly"] = np.array(error_pct_latest)

        annual_actual = actual_latest.sum()
        annual_pred = pred_latest.sum()
        annual_error_pct = (
            np.nan
            if annual_actual == 0
            else ((annual_pred - annual_actual) / annual_actual) * 100
        )

        test_years[f"{eval_years_end}_actual_annual"] = annual_actual
        test_years[f"{eval_years_end}_predicted_annual"] = annual_pred
        test_years[f"{eval_years_end}_error_pct_annual"] = annual_error_pct

        if annual_ci_latest is not None:
            test_years[f"{eval_years_end}_predicted_monthly_ci95"] = monthly_ci_latest
            test_years[f"{eval_years_end}_min_annual_ci95"] = annual_ci_latest[0]
            test_years[f"{eval_years_end}_max_annual_ci95"] = annual_ci_latest[1]

        print(f"      {'-'*90}" if monthly_ci_latest is not None else f"      {'-'*55}")
        if annual_ci_latest is not None:
            print(
                f"      TOTAL | {annual_actual:12,.0f} | {annual_pred:15,.0f} | "
                f"95% PI [{annual_ci_latest[0]:,.0f}, {annual_ci_latest[1]:,.0f}] | {annual_error_pct:7.2f}%"
            )
        else:
            print(
                f"      TOTAL | {annual_actual:12,.0f} | {annual_pred:15,.0f} | {annual_error_pct:7.2f}%"
            )

        # DataFrame for latest year (with optional PI)
        months = np.arange(1, 13)
        data_latest = {
            "Month": months,
            "Actual": actual_latest,
            "Predicted": pred_latest,
            "Error_%": np.array(error_pct_latest, dtype=float),
        }
        if monthly_ci_latest is not None:
            data_latest["PI_low"] = monthly_ci_latest[:, 0]
            data_latest["PI_high"] = monthly_ci_latest[:, 1]
        monthly_details[eval_years_end] = pd.DataFrame(data_latest)

    if "valid_years" in best:
        for i in range(eval_years_start, eval_years_end):
            if i in best["valid_years"]:
                idx_prev = np.where(best["valid_years"] == i)[0][0]
                actual_prev = best["actual_monthly_matrix"][idx_prev]
                pred_prev = best["pred_monthly_matrix"][idx_prev]

                test_years[f"{i}_actual_annual"] = actual_prev.sum()
                test_years[f"{i}_predicted_annual"] = pred_prev.sum()

                with np.errstate(divide="ignore", invalid="ignore"):
                    err_prev = np.where(
                        actual_prev != 0,
                        (pred_prev - actual_prev) / actual_prev * 100,
                        np.nan,
                    )
                monthly_details[i] = pd.DataFrame(
                    {
                        "Month": np.arange(1, 13),
                        "Actual": actual_prev,
                        "Predicted": pred_prev,
                        "Error_%": err_prev.astype(float),
                    }
                )

    result = {
        "entity": entity_name,
        "best_model": best,
        "training_end": training_end,
        "monthly_details": monthly_details,
    }
    if test_years:
        result["test_years"] = test_years

    return result


def save_summary(all_results, output_file, monthly_book_file=None):
    """
    Writes the global summary Excel to `output_file` (same behavior),
    and if `monthly_book_file` is provided, also writes ONE Excel workbook where:
      - Each sheet corresponds to an entity.
      - The sheet contains a single tall table: Year, Month, Actual, Predicted, Error_%[, PI_low, PI_high]
        (PI columns present but NaN if not available for that year).
    """
    # ---- global summary (unchanged) ----
    summary_data = []
    for result in all_results:
        best = result.get("best_model", {})
        row = {
            "Entity": result.get("entity"),
            "Strategy": best.get("strategy"),
            "Lags": best.get("n_lags"),
            "Features": best.get("feature_block"),
            "PC_Weight": best.get("pc_weight", "N/A"),
            "PCA_Lambda": best.get("pca_lambda", "N/A"),
            "Monthly_Temp": best.get("use_monthly_temp", False),
            "Monthly_Clients": best.get("use_monthly_clients", False),
            "Client_Pattern_Weight": best.get("client_pattern_weight", "N/A"),
            "Ensemble_Components": (
                ", ".join(best.get("ensemble_reference_strategies", []))
                if best.get("ensemble_reference_strategies")
                else "N/A"
            ),
            "Ensemble_Weights": best.get("ensemble_weights", "N/A"),
            "Training_window": best.get("training_window", "N/A"),
            "Annual_MAE": best.get("annual_mae"),
            "Annual_MAPE": best.get("annual_mape"),
            "Annual_R2": best.get("annual_r2"),
            "Monthly_MAE": best.get("monthly_mae"),
            "Monthly_MAPE": best.get("monthly_mape"),
            "Monthly_R2": best.get("monthly_r2"),
        }
        test_years = result.get("test_years", {})
        if test_years:
            row.update(
                {
                    k: v
                    for k, v in test_years.items()
                    if k.endswith("annual_ci95")
                }
            )
            row.update(
                {k: v for k, v in test_years.items() if k.endswith("annual")}
            )
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(output_file, index=False)

    # ---- ONE monthly workbook: sheet per entity ----
    if monthly_book_file is not None:
        with pd.ExcelWriter(monthly_book_file, engine="xlsxwriter") as writer:
            used_sheet_names = set()

            for result in all_results:
                entity = result.get("entity")
                monthly_details = result.get("monthly_details", {})
                if not monthly_details:
                    continue

                # Concatenate all years into one tall table
                frames = []
                for y, df_y in sorted(monthly_details.items()):
                    df_y2 = df_y.copy()
                    # Ensure PI columns exist for unioned schema
                    for col in ["PI_low", "PI_high"]:
                        if col not in df_y2.columns:
                            df_y2[col] = np.nan
                    df_y2.insert(0, Aliases.ANNEE, y)
                    frames.append(df_y2)

                if not frames:
                    continue

                df_entity = pd.concat(frames, ignore_index=True)
                # Consistent column order
                cols = [
                    Aliases.ANNEE,
                    "Month",
                    "Actual",
                    "Predicted",
                    "Error_%",
                    "PI_low",
                    "PI_high",
                ]
                df_entity = df_entity[cols]

                # Sanitize & uniquify sheet name (<=31 chars)
                safe = "".join(
                    c if c.isalnum() or c in "._- " else "_" for c in entity
                ).strip()
                sheet = safe[:31] if safe else "Entity"
                base = sheet
                idx = 1
                while sheet in used_sheet_names:
                    suffix = f"_{idx}"
                    sheet = base[: 31 - len(suffix)] + suffix
                    idx += 1
                used_sheet_names.add(sheet)

                df_entity.to_excel(writer, sheet_name=sheet, index=False)

            # Optional index sheet
            index_rows = [{"Entity": r["entity"]} for r in all_results]
            if index_rows:
                pd.DataFrame(index_rows).to_excel(
                    writer, sheet_name="README", index=False
                )
