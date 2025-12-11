from itertools import product
from onee.growth_rate_model import (
    build_growth_rate_features,
)
from onee.utils import select_best_model, calculate_all_annual_metrics
from onee.data.names import Aliases
import numpy as np
import pandas as pd

def run_annual_loocv_grid_search(
    monthly_matrix,
    years,
    df_features,
    df_monthly,
    monthly_clients_lookup,
    GrowthModelClass,
    feature_building_grid,
    model_hyperparameter_grid,
    metric_fn,
    verbose=True,
):
    """
    Perform annual-only LOOCV grid search over model and feature-building configurations.

    Returns:
        A list of dictionaries, each containing:
            - model_config
            - feature_config
            - pred_annual_values
            - actual_annual_values
            - valid_years
            - flat metrics
    """

    n_years = len(years)
    annual_consumption = monthly_matrix.sum(axis=1)

    # -------------------------------------------------------------------------
    # Extract grid dicts from pydantic objects
    # -------------------------------------------------------------------------
    fg = feature_building_grid.model_dump()
    mg = model_hyperparameter_grid.model_dump()

    mg = {
        k: v for k, v in mg.items() 
        if k in GrowthModelClass.SUPPORTED_HYPERPARAMS
    }
    # -------------------------------------------------------------------------
    # training_window MUST be handled separately
    # -------------------------------------------------------------------------
    if "training_window" not in fg:
        print("⚠️ feature_building_grid does not contain 'training_window'.")
        training_windows = [None]
    else:
        training_windows = fg.pop("training_window")

    # All remaining fields in fg are hyperparameters for build_growth_rate_features
    feature_param_names = list(fg.keys())
    feature_param_values = list(fg.values())

    # All fields in mg are hyperparameters for GrowthModelClass
    model_param_names = list(mg.keys())
    model_param_values = list(mg.values())

    results = []

    # -------------------------------------------------------------------------
    # GRID SEARCH
    # -------------------------------------------------------------------------
    for training_window in training_windows:
        # Loop over feature-building parameters (Cartesian product)
        for feature_values in product(*feature_param_values):
            # Build the feature configuration dict for this iteration
            feature_config_dict = dict(zip(feature_param_names, feature_values))
            feature_config_dict["training_window"] = training_window

            # Loop over model hyperparameters (Cartesian product)
            for model_values in product(*model_param_values):
                
                model_config_dict = dict(zip(model_param_names, model_values))
                # Storage for LOOCV
                pred_annual_list = []
                actual_annual_list = []
                valid_years_list = []

                # -------------------------------------------------------------
                # LOOCV — predict year t using train < t
                # -------------------------------------------------------------
                for test_idx in range(2, n_years):

                    # Determine training range
                    if training_window is not None:
                        start_idx = max(0, test_idx - training_window)
                    else:
                        start_idx = 0

                    train_indices = [
                        i for i in range(n_years)
                        if (i < test_idx and i >= start_idx)
                    ]

                    # Need at least 3 annual observations for stable fitting
                    if len(train_indices) < 3:
                        if verbose:
                            print(f"Skipping year {years[test_idx]} due to insufficient training data.")
                        continue

                    train_years = [years[i] for i in train_indices]
                    train_annual = annual_consumption[train_indices]

                    # ---------------------------------------------------------
                    # Build training features
                    # ---------------------------------------------------------
                    train_features = build_growth_rate_features(
                        train_years,
                        df_features=df_features,
                        df_monthly=df_monthly,
                        clients_lookup=monthly_clients_lookup,
                        **feature_config_dict
                    )
                    # Convert to numpy
                    if train_features is not None:
                        train_features_array = np.asarray(train_features, float)
                    else:
                        train_features_array = None

                    # Sort by year for safety
                    train_years_array = np.asarray(train_years, int)
                    sort_idx = np.argsort(train_years_array)
                    train_years_array = train_years_array[sort_idx]
                    train_annual_sorted = np.asarray(train_annual, float)[sort_idx]
                    train_monthly_sorted = monthly_matrix[train_indices][sort_idx]

                    if train_features_array is not None:
                        train_features_array = train_features_array[sort_idx]
                    # ---------------------------------------------------------
                    # Train model
                    # ---------------------------------------------------------
                    try:
                        model = GrowthModelClass(**model_config_dict)
                        
                        # Get normalization array if normalization_col is specified
                        normalization_arr = None
                        if "normalization_col" in model_config_dict:
                            norm_col = model_config_dict["normalization_col"]
                            norm_df = df_features[[Aliases.ANNEE, norm_col]].drop_duplicates()
                            norm_df = norm_df.set_index(Aliases.ANNEE).loc[train_years_array]
                            normalization_arr = norm_df[norm_col].values
                        
                        model.fit(y = train_annual_sorted, X = train_features_array, years = train_years_array, monthly_matrix=train_monthly_sorted, normalization_arr=normalization_arr)
                    except Exception as e:
                        if verbose:
                            print("Error training model:", e)
                        continue  # skip this fold if model fails

                    # ---------------------------------------------------------
                    # Build test features
                    # ---------------------------------------------------------
                    test_year = int(years[test_idx])
                    test_features = build_growth_rate_features(
                        [test_year],
                        df_features=df_features,
                        df_monthly=df_monthly,
                        clients_lookup=monthly_clients_lookup,
                        **feature_config_dict
                    )

                    if test_features is not None:
                        test_features_array = np.asarray(test_features, float)[0]
                    else:
                        test_features_array = None

                    # ---------------------------------------------------------
                    # Predict annual
                    # ---------------------------------------------------------
                    try:
                        if GrowthModelClass.__name__ in ["GaussianProcessForecastModel", "IntensityForecastWrapper"]:
                            if test_features_array is None:
                                test_features_array = np.array(test_year).reshape(1, -1)
                            else:
                                test_features_array = np.hstack([np.array(test_year), test_features_array])
                        
                        if "normalization_col" not in model_config_dict:
                            value = None
                        else:
                            value = df_features.loc[df_features[Aliases.ANNEE] == test_year].get(model_config_dict["normalization_col"])
                            if value is not None:
                                if  len(value) != 1: 
                                    raise ValueError("Expected exactly one matching row.")
                                value = value.item()
                        pred_annual = model.predict(test_features_array, normalization_factor = value)
                    except Exception as e:
                        if verbose:
                            print(f"Error predicting year {test_year}:", e)
                        continue

                    actual_annual = annual_consumption[test_idx]

                    pred_annual_list.append(pred_annual)
                    actual_annual_list.append(actual_annual)
                    valid_years_list.append(test_year)

                # End LOOCV loop ------------------------------------------------

                # Skip if no valid folds
                if len(pred_annual_list) == 0:
                    if verbose:
                        print("No valid LOOCV folds for this config.")
                    continue

                pred_arr = np.array(pred_annual_list)
                act_arr = np.array(actual_annual_list)
                valid_years_arr = np.array(valid_years_list)

                # -------------------------------------------------------------
                # Compute metrics
                # -------------------------------------------------------------
                metrics = metric_fn(
                    actual_annual=act_arr,
                    pred_annual=pred_arr,
                )

                # -------------------------------------------------------------
                # Store result
                # -------------------------------------------------------------
                results.append(
                    {   
                        "model_name": GrowthModelClass.__name__,
                        "model_class": GrowthModelClass,   
                        "model_config": model.get_params(),
                        "feature_config": feature_config_dict,
                        "pred_annual_values": pred_arr,
                        "actual_annual_values": act_arr,
                        "valid_years": valid_years_arr,
                        **metrics,
                    }
                )

    return results



def run_long_horizon_forecast(
    monthly_matrix,
    years,
    df_features,
    df_monthly,
    config,
    MODEL_REGISTRY,
    monthly_clients_lookup=None,
    region_entity = "",
    save_folder: str = ".",
):
    # Step 1. Run LOOCV search across all param combinations from ForecastConfig
    all_results = []

    for model_config in config.models.models:
        model_name = model_config.model_type
        model_class = MODEL_REGISTRY[model_name]
        results = run_annual_loocv_grid_search(
            monthly_matrix=monthly_matrix,
            years=years,
            df_features=df_features,
            df_monthly=df_monthly,
            monthly_clients_lookup=monthly_clients_lookup,
            GrowthModelClass=model_class,
            feature_building_grid=config.features,
            model_hyperparameter_grid=model_config,
            metric_fn=calculate_all_annual_metrics,
        )
        all_results+=results

    best_result = select_best_model(all_results, r2_threshold=config.models.r2_threshold)

    print(
        f"\n🏆 Best model: {best_result['model_name']} | R²={best_result.get('annual_r2', 0):.3f}"
    )
    training_window = best_result["feature_config"].get("training_window", None)

    # Determine which years to use
    n_years = len(years)

    if training_window is None:
        # Use all years
        train_start = 0
    else:
        # Use only the last N years (handle short dataset edge case)
        train_start = max(0, n_years - training_window)

    train_years = years[train_start:]
    monthly_matrix = monthly_matrix[train_start:]
    train_annual = monthly_matrix.sum(axis=1)

    growth_features = build_growth_rate_features(
        years=train_years,
        df_features=df_features,
        clients_lookup=monthly_clients_lookup,
        df_monthly=df_monthly,
        **best_result["feature_config"],   # includes feature_block, transforms, lags, etc.
    )
    BestModelClass = best_result["model_class"]

    growth_model = BestModelClass(
        **best_result["model_config"]["hyper_params"]
    )

    # Get normalization array if normalization_col is specified in hyper_params
    normalization_arr = None
    hyper_params = best_result["model_config"]["hyper_params"]
    if "normalization_col" in hyper_params:
        norm_col = hyper_params["normalization_col"]
        norm_df = df_features[[Aliases.ANNEE, norm_col]].drop_duplicates()
        norm_df = norm_df.set_index(Aliases.ANNEE).loc[train_years]
        normalization_arr = norm_df[norm_col].values

    growth_model.fit(y = train_annual, X = growth_features, years = train_years, monthly_matrix=monthly_matrix, normalization_arr=normalization_arr)

    # Step 5. Forecast horizon years using best params
    last_year = int(max(years))

    horizon_preds = growth_model.forecast_horizon(
        df_features=df_features,
        start_year=last_year,
        horizon=config.temporal.horizon,
        monthly_clients_lookup=monthly_clients_lookup,
        df_monthly=df_monthly,
        feature_config=best_result["feature_config"]
    )

    # Support two possible shapes for `horizon_preds` returned by models:
    # - List[Tuple[int, float]]
    # - List[Tuple[int, float, float, float]]  (year, pred, lower, upper)
    processed_horizon_preds = []  # List[Tuple[int, float]]
    confidence_intervals = []     # List[Tuple[float, float]] for items that include CI

    for item in horizon_preds:
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                year = int(item[0])
                pred = float(item[1])
                processed_horizon_preds.append((year, pred))
                if len(item) >= 4:
                    # Collect the two floats as a confidence interval (lower, upper)
                    confidence_intervals.append((float(item[2]), float(item[3])))
            else:
                raise ValueError("Each horizon prediction item must contain at least (year, pred)")
        else:
            raise TypeError("Each horizon prediction must be a tuple or list")

    # Step 6. Distribute annual forecasts to months using only (year, pred)
    mean_curve = monthly_matrix.mean(axis=0)
    mean_curve_norm = mean_curve / mean_curve.sum()
    future_monthly = [val * mean_curve_norm for _, val in processed_horizon_preds]

    if confidence_intervals:
        # Build exogenous features for full timeline (history + forecast) for trend visualization
        forecast_years = [year for year, _ in processed_horizon_preds]
        full_timeline = np.concatenate([train_years, forecast_years])
        full_timeline_sorted = np.sort(np.unique(full_timeline))
        
        X_exog_full = build_growth_rate_features(
            years=full_timeline_sorted,
            df_features=df_features,
            clients_lookup=monthly_clients_lookup,
            df_monthly=df_monthly,
            **best_result["feature_config"],
        )
        
        # Build normalization array for full timeline if using IntensityForecastWrapper
        normalization_arr_full = None
        if "normalization_col" in hyper_params:
            norm_col = hyper_params["normalization_col"]
            norm_df = df_features[[Aliases.ANNEE, norm_col]].drop_duplicates()
            norm_df = norm_df.set_index(Aliases.ANNEE)
            normalization_arr_full = np.array([
                norm_df.loc[yr, norm_col] if yr in norm_df.index else np.nan
                for yr in full_timeline_sorted
            ])
        
        if save_folder:
            growth_model.plot_forecast(
                horizon_preds, 
                title=region_entity,
                save_plot=True,
                save_folder=save_folder,
                df_monthly=df_monthly,
                X_exog=X_exog_full,
                normalization_arr_full=normalization_arr_full
            )

    return {
        "horizon_predictions": processed_horizon_preds,
        "horizon_confidence_intervals": confidence_intervals,
        "monthly_forecasts": future_monthly,
        "run_parameters": {
            "feature_config": best_result["feature_config"],
            "growth_model": {"model_name":best_result['model_name'],**best_result["model_config"]["hyper_params"], **best_result["model_config"]["fitted_params"]},
        },
    }


def create_summary_dataframe(all_results):
    """
    Create a summary DataFrame from forecast results
    
    Args:
        all_results: List of forecast result dictionaries
        
    Returns:
        pandas DataFrame with flattened summary records
    """
    df_summary_records = []
    for r in all_results:
        for y, v, actual, percent_error in zip(
            r["forecast_years"], r["pred_annual"], r["actuals"], r["percent_errors"]
        ):
            # Flatten run parameters for the summary sheet
            rp = r.get("run_parameters", {}) or {}
            growth = rp.get("growth_model", {}) or {}
            feature_config = rp.get("feature_config", {}) or {}

            df_summary_records.append(
                {
                    "Region": r.get("region"),
                    "Train_Start": r.get("train_start"),
                    "Train_End": r.get("train_end"),
                    "Level": r.get("level"),
                    "Year": y,
                    "Predicted_Annual": v,
                    "Actual_Annual": actual,
                    "Percent_Error": percent_error,
                    **feature_config,
                    **growth
                }
            )

    return pd.DataFrame(df_summary_records)
