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

    # Initialize weights if not provided
    if weights is None:
        weights = np.ones(len(models), dtype=float)

    # Prepare to store predictions and coefficients
    ensemble_preds = np.zeros((len(ordered_years), 12))
    lambda_values = []

    # Iterate over each year (starting from the second year)
    for i in range(1, len(ordered_years)):
        # Only use data up to the previous year for training
        train_years = ordered_years[:i]

        # Get model predictions and actuals for training period
        train_idx = [np.where(model["valid_years"] == year)[0][0] for year in train_years]
        train_preds = model_preds[:, train_idx]
        train_actuals = model_actuals[:, train_idx]

        # Learn ensemble weights based on past data
        preds_flat = train_preds.transpose(1, 2, 0).reshape(-1, len(models))
        actual_flat = train_actuals.mean(axis=0).flatten()
        try:
            learned_weights = np.linalg.lstsq(preds_flat, actual_flat, rcond=None)[0]
        except np.linalg.LinAlgError:
            learned_weights = np.ones(len(models), dtype=float)

        # Clip and normalize weights
        learned_weights = np.clip(learned_weights, 0.0, None)
        if not np.isfinite(learned_weights).all() or learned_weights.sum() <= 0:
            learned_weights = np.ones(len(models), dtype=float)
        learned_weights /= learned_weights.sum()

        # Apply learned weights to the predictions
        ensemble_preds[i] = np.tensordot(learned_weights, train_preds, axes=(0, 0)).mean(axis=0)

        # Learn lambda (asymmetric bias) for this year based on past data
        actual_monthly = train_actuals.mean(axis=0)  # (n_years, 12)
        pred_monthly_unbiased = ensemble_preds[i]

        if use_annual:
            y_fit = actual_monthly.sum(axis=1)  # (n_years,)
            p_fit = pred_monthly_unbiased.sum(axis=1)  # (n_years,)
        else:
            y_fit = actual_monthly.flatten()  # (n_years*12,)
            p_fit = pred_monthly_unbiased.flatten()  # (n_years*12,)

        lam = _fit_lambda_pinball(y_fit, p_fit, tau=float(tau))
        lambda_values.append(lam)

        # Apply the bias factor to predictions
        ensemble_preds[i] *= (1.0 + lam)

    # Now calculate all metrics based on the final unbiased ensemble prediction
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
        "ensemble_weights": weights.tolist(),
        "ensemble_reference_strategies": component_strategies,
        "ensemble_component_details": component_details,
        "pred_monthly_matrix": ensemble_preds,
        "actual_monthly_matrix": actual_monthly,
        "valid_years": np.array(ordered_years),
        "bias_lambda": float(lam),
        "bias_tau": float(tau),
        "bias_fit_target": "annual" if use_annual else "monthly",
        **metrics,
    }
