# ─────────────────────────────────────────────────────────────
# run_horizon_forecast_srm.py (Clean + Correct)
# ─────────────────────────────────────────────────────────────

from onee.growth_rate_model import (MeanRevertingGrowthModelARP, 
                               MeanRevertingGrowthModel, 
                               PCAMeanRevertingGrowthModel, 
                               RawMeanRevertingGrowthModel, 
                               LocalInterpolationForecastModel, 
                               AsymmetricAdaptiveTrendModel,
                               GaussianProcessForecastModel)
from forecast_strategies import (
    create_monthly_matrix,
)
from horizon_forecast_strategies import run_long_horizon_forecast
from onee.utils import fill_2020_with_avg
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
    "AsymmetricAdaptiveTrendModel": AsymmetricAdaptiveTrendModel,
    "GaussianProcessForecastModel": GaussianProcessForecastModel
}

class GeneralParams(BaseModel):
    project_root: Path
    exp_name: str
    horizon: int
    variable: str = "consommation_kwh"
    unit: str = "Kwh"
    r2_threshold: float = 0.6
    model_classes: List[str] = Field(
        default_factory=lambda: ["GaussianProcessForecastModel"] # "GaussianProcessForecastModel", "MeanRevertingGrowthModelARP", "MeanRevertingGrowthModel", "PCAMeanRevertingGrowthModel", "RawMeanRevertingGrowthModel", "LocalInterpolationForecastModel", "AsymmetricAdaptiveTrendModel", ""
    )
    impute_2020: bool = False


class FeatureBuildingGrid(BaseModel):
    transforms: Tuple[Tuple[str, ...], ...] = (
        ("lchg",),
        ("lchg", "lag_lchg"),
        # ("level"),
        # ("level", "lag"),
    )
    lags: Tuple[Tuple[int, ...], ...] = ((1,2),)
    feature_block: List[List[str]] = Field(
        default_factory=lambda: [
            [],
            ["pib_mdh"],
            ["gdp_primaire", "gdp_secondaire", "gdp_tertiaire"],
            ["pib_mdh", "gdp_primaire", "gdp_secondaire", "gdp_tertiaire"],
        ]
    )
    use_pf: List[bool] = Field(default_factory=lambda: [False])
    use_clients: List[bool] = Field(default_factory=lambda: [True])
    training_window: List[Optional[int]] = Field(default_factory=lambda: [None])


class ModelHyperparameterGrid(BaseModel):
    
    # Shared parameters across both models
    include_ar: List[bool] = Field(default_factory=lambda: [True])
    include_exog: List[bool] = Field(default_factory=lambda: [True])
    include_ar_squared: List[bool] = Field(default_factory=lambda: [True, False])

    # AR(p)-only parameter (ignored by basic model, warning issued)
    ar_lags: List[int] = Field(default_factory=lambda: [2, 3, 5])

    # Regularization
    l2_penalty: List[float] = Field(default_factory=lambda: [0.0])
    # beta_bounds: List[Tuple[float, float]] = Field(
    #     default_factory=lambda: [(-1.0, 1.0)]
    # )

    # rho_bounds: List[Tuple[float, float]] = Field(
    #     default_factory=lambda: [(0.0, 1.0)]
    # )

    # Asymmetric loss
    use_asymmetric_loss: List[bool] = Field(
        default_factory=lambda: [True]
    )
    underestimation_penalty: List[float] = Field(
        default_factory=lambda: [2.0, 5.0]
    )

    # PCA-based model
    n_pcs: List[int] = Field(default_factory=lambda: [2, 3])
    pca_lambda: List[float] = Field(default_factory=lambda: [0.3, 0.9])

    # Local interpolation model params
    window_size: List[int] = Field(default_factory=lambda: [3])
    weight_decay: List[float] = Field(default_factory=lambda: [0.6, 1])
    selection_mode: List[str] = Field(
        default_factory=lambda: ["in_sample"] # "cross_validation", "in_sample"
    )
    fit_on_growth_rates: List[bool] = Field(
        default_factory=lambda: [True]
    )
    use_full_history: List[bool] = Field(
        default_factory=lambda: [True]
    )

    # Gaussian Process model params
    n_restarts_optimizer: List[int] = Field(default_factory=lambda: [10])
    normalize_y: List[bool] = Field(default_factory=lambda: [True, False])



class ForecastConfig(BaseModel):
    general_params: GeneralParams
    feature_building_grid: FeatureBuildingGrid = FeatureBuildingGrid()
    model_hyperparameter_grid: ModelHyperparameterGrid = ModelHyperparameterGrid()

 



# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = ForecastConfig(
        general_params=GeneralParams(
            project_root=Path(__file__).resolve().parents[0],
            exp_name="gp_decreasing_growth",
            horizon=5
        )
    )

    REGIONS = [
        "Casablanca-Settat",
        "Béni Mellal-Khénifra",
        "Drâa-Tafilalet", 
        "Fès-Meknès",
        "Laâyoune-Sakia El Hamra",
        "Marrakech-Safi",
        "Oriental",
        "Rabat-Salé-Kénitra",
        "Tanger-Tétouan-Al Hoceïma",
        "Souss-Massa",
    ]

    # Choose which analysis parts (levels) to run
    # 1: Activities, 2: Aggregated BT, 3: Aggregated MT, 4: Total Regional,
    # 5: Individual Distributors, 6: All Distributors, 7: SRM (Regional+Dist)
    RUN_LEVELS = {4,6,7}

    forecast_types = {
        # "forward": (2007, 2023),
        "backtest": (2007, 2018),
    }

    for TARGET_REGION in REGIONS:
        print(f"\n{'='*80}\n🌍 REGION: {TARGET_REGION}\n{'='*80}")

        output_dir = (
            config.general_params.project_root
            / "outputs_horizon"
            / f"{config.general_params.exp_name}"
            / clean_name(TARGET_REGION)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

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

        df_regional_bt = df_regional_bt[df_regional_bt["annee"] >= 2013]
        df_regional_mt = df_regional_mt[df_regional_mt["annee"] >= 2013]
        df_features = df_features[df_features["annee"] >= 2013]
        if df_dist is not None:
            df_dist = df_dist[df_dist["annee"] >= 2013]

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
                    if config.general_params.impute_2020:
                        df_activity = fill_2020_with_avg(df_activity, reg_var_col)

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
                        MODEL_REGISTRY=MODEL_REGISTRY,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            f"Activity_{activity}", {}
                        ),
                    )

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
                        MODEL_REGISTRY=MODEL_REGISTRY,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            "Aggregated_BT", {}
                        ),
                    )

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
                        MODEL_REGISTRY=MODEL_REGISTRY,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            "Aggregated_MT", {}
                        ),
                    )

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
                if config.general_params.impute_2020:
                    df_total_regional = fill_2020_with_avg(df_total_regional, reg_var_col)
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
                    df_monthly=df_total_regional,
                    config=config,
                    MODEL_REGISTRY=MODEL_REGISTRY,
                    monthly_clients_lookup=client_predictions_lookup.get(
                        "Total_Regional", {}
                    ),
                    region_entity = f"{TARGET_REGION} - TOTAL REGIONAL",
                    save_folder=output_dir,
                )

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
            if len(df_dist) >= 1:
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
                        if config.general_params.impute_2020:
                            df_distributor = fill_2020_with_avg(
                                df_distributor, dist_var_col
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
                            MODEL_REGISTRY=MODEL_REGISTRY,
                            monthly_clients_lookup=client_predictions_lookup.get(
                                f"Distributor_{distributor}", {}
                            ),
                        )

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
                                "level": f"Distributor_{distributor}",
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
                    if config.general_params.impute_2020:
                        df_all_dist = fill_2020_with_avg(df_all_dist, dist_var_col)
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
                        MODEL_REGISTRY=MODEL_REGISTRY,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            "All_Distributors", {}
                        ),
                        region_entity = f"{TARGET_REGION} - ALL DISTRIBUTORS COMBINED",
                        save_folder=output_dir
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

                    # if config.general_params.impute_2020:
                    #     df_srm = fill_2020_with_avg(df_srm, config.general_params.variable)

                    df_srm = df_srm[df_srm["annee"] >= 2013]
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
                        MODEL_REGISTRY=MODEL_REGISTRY,
                        monthly_clients_lookup=client_predictions_lookup.get(
                            "SRM_Regional_Plus_Dist", {}
                        ),
                        region_entity = f"{TARGET_REGION} - SRM (Regional + Distributors)",
                        save_folder=output_dir
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
