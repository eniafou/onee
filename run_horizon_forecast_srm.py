# ─────────────────────────────────────────────────────────────
# run_horizon_forecast_srm.py (Clean + Correct)
# ─────────────────────────────────────────────────────────────

from growth_rate_model import MeanRevertingGrowthModelARP, MeanRevertingGrowthModel, PCAMeanRevertingGrowthModel, RawMeanRevertingGrowthModel, LocalInterpolationForecastModel
from forecast_strategies import (
    build_growth_rate_features,
    create_monthly_matrix,
)
from onee.utils import select_best_model, calculate_all_annual_metrics
from run_forecast_srm import (
    clean_name,
    get_queries_for,
    aggregate_predictions,
    require_columns,
    load_client_prediction_lookup,
)
import numpy as np
import pandas as pd
import sqlite3
import pickle
from pathlib import Path
import warnings

from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional
from itertools import product

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION OBJECT
# ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "MeanRevertingGrowthModel": MeanRevertingGrowthModel,
    "MeanRevertingGrowthModelARP": MeanRevertingGrowthModelARP,
    "PCAMeanRevertingGrowthModel": PCAMeanRevertingGrowthModel,
    "RawMeanRevertingGrowthModel": RawMeanRevertingGrowthModel,
    "LocalInterpolationForecastModel": LocalInterpolationForecastModel,
}

class GeneralParams(BaseModel):
    project_root: Path
    exp_name: str
    horizon: int
    variable: str = "consommation_kwh"
    unit: str = "Kwh"
    r2_threshold: float = 0.6
    model_classes: List[str] = Field(
        default_factory=lambda: ["MeanRevertingGrowthModel", "MeanRevertingGrowthModelARP"] # "MeanRevertingGrowthModelARP", "MeanRevertingGrowthModel", "PCAMeanRevertingGrowthModel", "RawMeanRevertingGrowthModel", "LocalInterpolationForecastModel"
    )


class FeatureBuildingGrid(BaseModel):
    transforms: Tuple[Tuple[str, ...], ...] = (
        # ("lchg",),
        # ("level",),
        ("lchg", "level"),
        # ("lag_lchg",),
    )
    lags: Tuple[Tuple[int, ...], ...] = ((1),)
    feature_block: List[List[str]] = Field(
        default_factory=lambda: [
            [],
            ["pib_mdh"],
            ["gdp_primaire", "gdp_secondaire", "gdp_tertiaire"],
            ["pib_mdh", "gdp_primaire", "gdp_secondaire", "gdp_tertiaire"],
        ]
    )
    use_pf: List[bool] = Field(default_factory=lambda: [False])
    use_clients: List[bool] = Field(default_factory=lambda: [True, False])
    training_window: List[Optional[int]] = Field(default_factory=lambda: [None, 3, 5])


class ModelHyperparameterGrid(BaseModel):
    
    # Shared parameters across both models
    include_ar: List[bool] = Field(default_factory=lambda: [True])
    include_exog: List[bool] = Field(default_factory=lambda: [True])
    include_ar_squared: List[bool] = Field(default_factory=lambda: [True, False])

    # AR(p)-only parameter (ignored by basic model, warning issued)
    ar_lags: List[int] = Field(default_factory=lambda: [3, 5])

    # Regularization
    l2_penalty: List[float] = Field(default_factory=lambda: [0.0, 10])
    # beta_bounds: List[Tuple[float, float]] = Field(
    #     default_factory=lambda: [(-1.0, 1.0)]
    # )

    rho_bounds: List[Tuple[float, float]] = Field(
        default_factory=lambda: [(0.00, 1.0)]
    )

    # Asymmetric loss
    use_asymmetric_loss: List[bool] = Field(
        default_factory=lambda: [False, True]
    )
    # underestimation_penalty: List[float] = Field(
    #     default_factory=lambda: [1.0, 2.0, 3.0]
    # )

    # PCA-based model
    n_pcs: List[int] = Field(default_factory=lambda: [2, 3])
    pca_lambda: List[float] = Field(default_factory=lambda: [0.3, 0.9])

    # Local interpolation model params
    window_size: List[int] = Field(default_factory=lambda: [3, 5, 7])
    weighted: List[bool] = Field(default_factory=lambda: [True])
    weight_decay: List[float] = Field(default_factory=lambda: [0.5, 0.9])



class ForecastConfig(BaseModel):
    general_params: GeneralParams
    feature_building_grid: FeatureBuildingGrid = FeatureBuildingGrid()
    model_hyperparameter_grid: ModelHyperparameterGrid = ModelHyperparameterGrid()

 


def run_annual_loocv_grid_search(
    monthly_matrix,
    years,
    df_features,
    df_monthly,
    monthly_clients_lookup,
    GrowthModelClass,
    feature_building_grid,
    model_hyperparameter_grid,
    metric_fn,     # A function to compute metrics (actual_annual, pred_annual) → dict
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
    fg = feature_building_grid.dict()
    mg = model_hyperparameter_grid.dict()

    mg = {
        k: v for k, v in mg.items() 
        if k in GrowthModelClass.SUPPORTED_HYPERPARAMS
    }

    if GrowthModelClass.__name__ == "LocalInterpolationForecastModel":
        fg = {"feature_block": [[]], "training_window": [None]}

    # print(mg)
    # print(fg)
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
                        model.fit(y = train_annual_sorted, X = train_features_array, years = train_years, monthly_matrix=train_monthly_sorted)
                    except Exception:
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
                        pred_annual = model.predict(test_features_array)
                    except Exception:
                        continue

                    actual_annual = annual_consumption[test_idx]

                    pred_annual_list.append(pred_annual)
                    actual_annual_list.append(actual_annual)
                    valid_years_list.append(test_year)

                # End LOOCV loop ------------------------------------------------

                # Skip if no valid folds
                if len(pred_annual_list) == 0:
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
    config: ForecastConfig,
    monthly_clients_lookup=None,
):
    # Step 1. Run LOOCV search across all param combinations from ForecastConfig
    all_results = []

    for model_name in config.general_params.model_classes:
        model_class = MODEL_REGISTRY[model_name]
        results = run_annual_loocv_grid_search(
            monthly_matrix=monthly_matrix,
            years=years,
            df_features=df_features,
            df_monthly=df_monthly,
            monthly_clients_lookup=monthly_clients_lookup,
            GrowthModelClass=model_class,
            feature_building_grid=config.feature_building_grid,
            model_hyperparameter_grid=config.model_hyperparameter_grid,
            metric_fn=calculate_all_annual_metrics,
        )
        all_results+=results

    best_result = select_best_model(all_results, r2_threshold=config.general_params.r2_threshold)

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
        **best_result["feature_config"],   # includes feature_blocks, transforms, lags, etc.
    )
    BestModelClass = best_result["model_class"]

    growth_model = BestModelClass(
        **best_result["model_config"]
    )

    growth_model.fit(y = train_annual, X = growth_features, years = train_years, monthly_matrix=monthly_matrix)

    # Step 5. Forecast horizon years using best params
    last_year = int(max(years))

    horizon_preds = growth_model.forecast_horizon(
        df_features=df_features,
        start_year=last_year,
        horizon=config.general_params.horizon,
        monthly_clients_lookup=monthly_clients_lookup,
        df_monthly=df_monthly,
        feature_config=best_result["feature_config"]
    )

    # Step 6. Distribute annual forecasts to months
    mean_curve = monthly_matrix.mean(axis=0)
    mean_curve_norm = mean_curve / mean_curve.sum()
    future_monthly = [val * mean_curve_norm for _, val in horizon_preds]

    return {
        "horizon_predictions": horizon_preds,
        "monthly_forecasts": future_monthly,
        "run_parameters": {
            "feature_config": best_result["feature_config"],
            "growth_model": {"model_name":best_result['model_name'],**best_result["model_config"]},
        },
    }


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = ForecastConfig(
        general_params=GeneralParams(
            project_root=Path(__file__).resolve().parents[0],
            exp_name="srm_arp",
            horizon=5
        )
    )

    REGIONS = [
        "Béni Mellal-Khénifra", "Casablanca-Settat", "Drâa-Tafilalet", "Fès-Meknès",
        "Laâyoune-Sakia El Hamra",
        "Marrakech-Safi",
        "Oriental",
        "Rabat-Salé-Kénitra",
        "Tanger-Tétouan-Al Hoceïma",
        "Souss-Massa",
    ]

    REGIONS = ["Casablanca-Settat"]
    # Choose which analysis parts (levels) to run
    # 1: Activities, 2: Aggregated BT, 3: Aggregated MT, 4: Total Regional,
    # 5: Individual Distributors, 6: All Distributors, 7: SRM (Regional+Dist)
    RUN_LEVELS = {4}
    forecast_types = {
        # "forward": (2007, 2023),
        "backtest": (2007, 2020),
    }

    for TARGET_REGION in REGIONS:
        print(f"\n{'='*80}\n🌍 REGION: {TARGET_REGION}\n{'='*80}")

        db_regional = sqlite3.connect(
            config.general_params.project_root / "data/ONEE_Regional_COMPLETE_2007_2023.db"
        )
        db_dist = sqlite3.connect(
            config.general_params.project_root / "data/ONEE_Distributeurs_consumption.db"
        )

        q_regional_mt, q_regional_bt, q_dist, q_features, var_cols = get_queries_for(
            config.general_params.variable, TARGET_REGION
        )
        df_regional_mt = pd.read_sql_query(q_regional_mt, db_regional)
        df_regional_bt = pd.read_sql_query(q_regional_bt, db_regional)
        df_features = pd.read_sql_query(q_features, db_regional)
        df_dist = pd.read_sql_query(q_dist, db_dist) if q_dist is not None else None

        db_regional.close()
        db_dist.close()

        df_regional_mt["activite"] = df_regional_mt["activite"].replace(
            "Administratif", "Administratif_mt"
        )
        df_regional = pd.concat([df_regional_bt, df_regional_mt])

        reg_var_col = var_cols["regional"]
        require_columns(

            df_regional, ["annee", "mois", "activite", reg_var_col], "df_regional"
        )
        df_regional[reg_var_col] = df_regional[reg_var_col].fillna(0)

        if df_dist is not None:
            dist_var_col = var_cols["distributor"]
            require_columns(
                df_dist, ["annee", "mois", "distributeur", dist_var_col], "df_dist"
            )
            df_dist[dist_var_col] = df_dist[dist_var_col].fillna(0)

        # Lookup of previously computed monthly client predictions
        client_predictions_lookup = load_client_prediction_lookup(TARGET_REGION)
        all_results = []

        activities = sorted(df_regional["activite"].unique())
        mt_activities = [
            "Administratif_mt",
            "Agricole",
            "Industriel",
            "Résidentiel",
            "Tertiaire",
        ]
        bt_activities = [a for a in activities if a not in mt_activities]

        for mode, (train_start, train_end) in forecast_types.items():
            print(
                f"\n🧭 MODE: {mode.upper()} — training {train_start}→{train_end}, horizon={config.general_params.horizon}"
            )

            # ────────────────────────────────────────────────────────────────
            # LEVEL 1: INDIVIDUAL ACTIVITIES
            # ────────────────────────────────────────────────────────────────
            if 1 in RUN_LEVELS:
                print(f"\n{'#'*60}\nLEVEL 1: INDIVIDUAL ACTIVITIES\n{'#'*60}")
                for activity in activities:
                    entity_name = f"Activity_{activity}"
                    print(f"\n{'#'*30}\n{entity_name}\n{'#'*30}")
                    df_activity = (
                        df_regional[df_regional["activite"] == activity][
                            ["annee", "mois", reg_var_col]
                        ]
                        .copy()
                        .rename(columns={reg_var_col: config.general_params.variable})
                    )
                    df_train = df_activity[df_activity["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.general_params.variable
                    )
                    years = np.sort(df_train["annee"].unique())

                    res = run_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_activity,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            f"Activity_{activity}", {}
                        ),
                    )

                    # store monthly forecasts for later aggregation
                    if "monthly_forecasts" in res:
                        client_predictions_lookup[entity_name] = {
                            y: np.array(m)
                            for y, m in zip(
                                [y for y, _ in res["horizon_predictions"]],
                                res["monthly_forecasts"],
                            )
                        }

                    forecast_years = [y for y, _ in res["horizon_predictions"]]
                    pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                    actuals = []
                    percent_errors = []
                    for y, v in zip(forecast_years, pred_annual):
                        actual = None
                        percent_error = None
                        mask = df_activity["annee"] == y
                        if not df_activity.loc[mask, reg_var_col].empty:
                            actual = df_activity.loc[mask, reg_var_col].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "region": TARGET_REGION,
                            "mode": mode,
                            "level": entity_name,
                            "forecast_years": forecast_years,
                            "pred_annual": pred_annual,
                            "pred_monthly": res["monthly_forecasts"],
                            "run_parameters": res.get("run_parameters", {}),
                            "actuals": actuals,
                            "percent_errors": percent_errors,
                        }
                    )

            # ────────────────────────────────────────────────────────────────
            # LEVEL 2: AGGREGATED BT
            # ────────────────────────────────────────────────────────────────
            if 2 in RUN_LEVELS:
                print(f"\n{'#'*60}\nLEVEL 2: AGGREGATED BT\n{'#'*60}")
                df_bt = df_regional[df_regional["activite"].isin(bt_activities)]
                if not df_bt.empty:
                    df_bt_agg = (
                        df_bt.groupby(["annee", "mois"])
                        .agg({reg_var_col: "sum"})
                        .reset_index()
                        .rename(columns={reg_var_col: config.general_params.variable})
                    )

                    # Combine predictions from Level 1
                    aggregate_predictions(
                        client_predictions_lookup,
                        "Aggregated_BT",
                        [f"Activity_{a}" for a in bt_activities],
                    )

                    df_train = df_bt_agg[df_bt_agg["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.general_params.variable
                    )
                    years = np.sort(df_train["annee"].unique())

                    res = run_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_bt_agg,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            "Aggregated_BT", {}
                        ),
                    )

                    # Update aggregated lookup
                    if "monthly_forecasts" in res:
                        client_predictions_lookup["Aggregated_BT"] = {
                            y: np.array(m)
                            for y, m in zip(
                                [y for y, _ in res["horizon_predictions"]],
                                res["monthly_forecasts"],
                            )
                        }

                    forecast_years = [y for y, _ in res["horizon_predictions"]]
                    pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                    actuals = []
                    percent_errors = []
                    for y, v in zip(forecast_years, pred_annual):
                        actual = None
                        percent_error = None
                        mask = df_bt_agg["annee"] == y
                        if not df_bt_agg.loc[mask, config.general_params.variable].empty:
                            actual = df_bt_agg.loc[mask, config.general_params.variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "region": TARGET_REGION,
                            "mode": mode,
                            "level": "Aggregated_BT",
                            "forecast_years": forecast_years,
                            "pred_annual": pred_annual,
                            "pred_monthly": res["monthly_forecasts"],
                            "run_parameters": res.get("run_parameters", {}),
                            "actuals": actuals,
                            "percent_errors": percent_errors,
                        }
                    )

            # ────────────────────────────────────────────────────────────────
            # LEVEL 3: AGGREGATED MT
            # ────────────────────────────────────────────────────────────────
            if 3 in RUN_LEVELS:
                print(f"\n{'#'*60}\nLEVEL 3: AGGREGATED MT\n{'#'*60}")
                df_mt = df_regional[df_regional["activite"].isin(mt_activities)]
                if not df_mt.empty:
                    df_mt_agg = (
                        df_mt.groupby(["annee", "mois"])
                        .agg({reg_var_col: "sum"})
                        .reset_index()
                        .rename(columns={reg_var_col: config.general_params.variable})
                    )

                    aggregate_predictions(
                        client_predictions_lookup,
                        "Aggregated_MT",
                        [f"Activity_{a}" for a in mt_activities],
                    )

                    df_train = df_mt_agg[df_mt_agg["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.general_params.variable
                    )
                    years = np.sort(df_train["annee"].unique())

                    res = run_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_mt_agg,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            "Aggregated_MT", {}
                        ),
                    )

                    if "monthly_forecasts" in res:
                        client_predictions_lookup["Aggregated_MT"] = {
                            y: np.array(m)
                            for y, m in zip(
                                [y for y, _ in res["horizon_predictions"]],
                                res["monthly_forecasts"],
                            )
                        }

                    forecast_years = [y for y, _ in res["horizon_predictions"]]
                    pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                    actuals = []
                    percent_errors = []
                    for y, v in zip(forecast_years, pred_annual):
                        actual = None
                        percent_error = None
                        mask = df_mt_agg["annee"] == y
                        if not df_mt_agg.loc[mask, config.general_params.variable].empty:
                            actual = df_mt_agg.loc[mask, config.general_params.variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "region": TARGET_REGION,
                            "mode": mode,
                            "level": "Aggregated_MT",
                            "forecast_years": forecast_years,
                            "pred_annual": pred_annual,
                            "pred_monthly": res["monthly_forecasts"],
                            "run_parameters": res.get("run_parameters", {}),
                            "actuals": actuals,
                            "percent_errors": percent_errors,
                        }
                    )

            # ────────────────────────────────────────────────────────────────
            # LEVEL 4: TOTAL REGIONAL
            # ────────────────────────────────────────────────────────────────
            if 4 in RUN_LEVELS:
                print(f"\n{'#'*60}\nLEVEL 4: TOTAL REGIONAL\n{'#'*60}")
                df_total_regional = (
                    df_regional.groupby(["annee", "mois"])
                    .agg({reg_var_col: "sum"})
                    .reset_index()
                    .rename(columns={reg_var_col: config.general_params.variable})
                )

                aggregate_predictions(
                    client_predictions_lookup,
                    "Total_Regional",
                    [f"Activity_{a}" for a in activities],
                )

                df_train = df_total_regional[df_total_regional["annee"] <= train_end]
                monthly_matrix = create_monthly_matrix(
                    df_train, value_col=config.general_params.variable
                )
                years = np.sort(df_train["annee"].unique())

                res = run_long_horizon_forecast(
                    monthly_matrix=monthly_matrix,
                    years=years,
                    df_features=df_features,
                    df_monthly=df_regional,
                    config=config,
                    monthly_clients_lookup=client_predictions_lookup.get(
                        "Total_Regional", {}
                    ),
                )

                if "monthly_forecasts" in res:
                    client_predictions_lookup["Total_Regional"] = {
                        y: np.array(m)
                        for y, m in zip(
                            [y for y, _ in res["horizon_predictions"]],
                            res["monthly_forecasts"],
                        )
                    }

                forecast_years = [y for y, _ in res["horizon_predictions"]]
                pred_annual = [float(v) for _, v in res["horizon_predictions"]]

                actuals = []
                percent_errors = []
                for y, v in zip(forecast_years, pred_annual):
                    actual = None
                    percent_error = None
                    mask = df_total_regional["annee"] == y
                    if not df_total_regional.loc[mask, reg_var_col].empty:
                        actual = df_total_regional.loc[mask, reg_var_col].sum()
                        percent_error = (
                            (v - actual) / actual * 100 if actual != 0 else None
                        )
                    actuals.append(actual)
                    percent_errors.append(percent_error)

                all_results.append(
                    {
                        "region": TARGET_REGION,
                        "mode": mode,
                        "level": "Total_Regional",
                        "forecast_years": forecast_years,
                        "pred_annual": pred_annual,
                        "pred_monthly": res["monthly_forecasts"],
                        "run_parameters": res.get("run_parameters", {}),
                        "actuals": actuals,
                        "percent_errors": percent_errors,
                    }
                )

            # ────────────────────────────────────────────────────────────────
            # LEVEL 5–7: DISTRIBUTORS + SRM
            # ────────────────────────────────────────────────────────────────
            if df_dist is not None:
                # Level 5: Individual Distributors
                if 5 in RUN_LEVELS:
                    print(f"\n{'#'*60}\nLEVEL 5: INDIVIDUAL DISTRIBUTORS\n{'#'*60}")
                    for distributor in sorted(df_dist["distributeur"].unique()):
                        df_distributor = (
                            df_dist[df_dist["distributeur"] == distributor][
                                ["annee", "mois", dist_var_col]
                            ]
                            .copy()
                            .rename(columns={dist_var_col: config.general_params.variable})
                        )
                        df_train = df_distributor[df_distributor["annee"] <= train_end]
                        monthly_matrix = create_monthly_matrix(
                            df_train, value_col=config.general_params.variable
                        )
                        years = np.sort(df_train["annee"].unique())

                        res = run_long_horizon_forecast(
                            monthly_matrix=monthly_matrix,
                            years=years,
                            df_features=df_features,
                            df_monthly=df_distributor,
                            config=config,
                            monthly_clients_lookup=client_predictions_lookup.get(
                                f"Distributor_{distributor}", {}
                            ),
                        )

                        entity_name = f"Distributor_{distributor}"
                        if "monthly_forecasts" in res:
                            client_predictions_lookup[entity_name] = {
                                y: np.array(m)
                                for y, m in zip(
                                    [y for y, _ in res["horizon_predictions"]],
                                    res["monthly_forecasts"],
                                )
                            }

                        forecast_years = [y for y, _ in res["horizon_predictions"]]
                        pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                        actuals = []
                        percent_errors = []
                        for y, v in zip(forecast_years, pred_annual):
                            actual = None
                            percent_error = None
                            mask = df_distributor["annee"] == y
                            if not df_distributor.loc[mask, config.general_params.variable].empty:
                                actual = df_distributor.loc[mask, config.general_params.variable].sum()
                                percent_error = (
                                    (v - actual) / actual * 100 if actual != 0 else None
                                )

                            actuals.append(actual)
                            percent_errors.append(percent_error)

                        all_results.append(
                            {
                                "region": TARGET_REGION,
                                "mode": mode,
                                "level": entity_name,
                                "forecast_years": forecast_years,
                                "pred_annual": pred_annual,
                                "pred_monthly": res["monthly_forecasts"],
                                "run_parameters": res.get("run_parameters", {}),
                                "actuals": actuals,
                                "percent_errors": percent_errors,
                            }
                        )

                # Level 6: All Distributors combined
                if 6 in RUN_LEVELS:
                    print(f"\n{'#'*60}\nLEVEL 6: ALL DISTRIBUTORS COMBINED\n{'#'*60}")
                    aggregate_predictions(
                        client_predictions_lookup,
                        "All_Distributors",
                        [
                            f"Distributor_{d}"
                            for d in sorted(df_dist["distributeur"].unique())
                        ],
                    )
                    df_all_dist = (
                        df_dist.groupby(["annee", "mois"])
                        .agg({dist_var_col: "sum"})
                        .reset_index()
                        .rename(columns={dist_var_col: config.general_params.variable})
                    )
                    df_train = df_all_dist[df_all_dist["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.general_params.variable
                    )
                    years = np.sort(df_train["annee"].unique())

                    res = run_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_all_dist,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            "All_Distributors", {}
                        ),
                    )

                    forecast_years = [y for y, _ in res["horizon_predictions"]]
                    pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                    actuals = []
                    percent_errors = []
                    for y, v in zip(forecast_years, pred_annual):
                        actual = None
                        percent_error = None
                        mask = df_all_dist["annee"] == y
                        if not df_all_dist.loc[mask, config.general_params.variable].empty:
                            actual = df_all_dist.loc[mask, config.general_params.variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "region": TARGET_REGION,
                            "mode": mode,
                            "level": "All_Distributors",
                            "forecast_years": forecast_years,
                            "pred_annual": pred_annual,
                            "pred_monthly": res["monthly_forecasts"],
                            "run_parameters": res.get("run_parameters", {}),
                            "actuals": actuals,
                            "percent_errors": percent_errors,
                        }
                    )

                # Level 7: SRM
                if 7 in RUN_LEVELS:
                    print(
                        f"\n{'#'*60}\nLEVEL 7: SRM (Regional + Distributors)\n{'#'*60}"
                    )
                    aggregate_predictions(
                        client_predictions_lookup,
                        "SRM_Regional_Plus_Dist",
                        ["Total_Regional", "All_Distributors"],
                    )
                    df_all_dist = (
                        df_dist.groupby(["annee", "mois"])
                        .agg({dist_var_col: "sum"})
                        .reset_index()
                        .rename(columns={dist_var_col: config.general_params.variable})
                    )
                    df_total_regional = (
                        df_regional.groupby(["annee", "mois"])
                        .agg({reg_var_col: "sum"})
                        .reset_index()
                        .rename(columns={reg_var_col: config.general_params.variable})
                    )
                    df_srm = (
                        pd.concat([df_total_regional, df_all_dist], ignore_index=True)
                        .groupby(["annee", "mois"])
                        .agg({config.general_params.variable: "sum"})
                        .reset_index()
                    )
                    df_train = df_srm[df_srm["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.general_params.variable
                    )
                    years = np.sort(df_train["annee"].unique())

                    res = run_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_srm,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            "SRM_Regional_Plus_Dist", {}
                        ),
                    )

                    forecast_years = [y for y, _ in res["horizon_predictions"]]
                    pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                    actuals = []
                    percent_errors = []
                    for y, v in zip(forecast_years, pred_annual):
                        actual = None
                        percent_error = None
                        mask = df_srm["annee"] == y
                        if not df_srm.loc[mask, config.general_params.variable].empty:
                            actual = df_srm.loc[mask, config.general_params.variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "region": TARGET_REGION,
                            "mode": mode,
                            "level": "SRM_Regional_Plus_Dist",
                            "forecast_years": forecast_years,
                            "pred_annual": pred_annual,
                            "pred_monthly": res["monthly_forecasts"],
                            "run_parameters": res.get("run_parameters", {}),
                            "actuals": actuals,
                            "percent_errors": percent_errors,
                        }
                    )

        # ────────────────────────────────────────────────────────────────
        # OUTPUTS
        # ────────────────────────────────────────────────────────────────
        output_dir = (
            config.general_params.project_root
            / "outputs_horizon"
            / f"{config.general_params.exp_name}"
            / clean_name(TARGET_REGION)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(
            output_dir
            / f"{clean_name(TARGET_REGION)}_{config.general_params.variable}_{config.general_params.exp_name}.pkl",
            "wb",
        ) as f:
            pickle.dump(all_results, f)

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
                        "Region": r["region"],
                        "Mode": r["mode"],
                        "Level": r["level"],
                        "Year": y,
                        "Predicted_Annual": v,
                        "Actual_Annual": actual,
                        "Percent_Error": percent_error,
                        **feature_config,
                        **growth
                    }
                )

        df_summary = pd.DataFrame(df_summary_records)
        out_xlsx = (
            output_dir
            / f"summary_{clean_name(TARGET_REGION)}_{config.general_params.variable}_{config.general_params.exp_name}.xlsx"
        )
        df_summary.to_excel(out_xlsx, index=False)
        print(f"\n📁 Saved horizon forecasts to {out_xlsx}")
