# ─────────────────────────────────────────────────────────────
# run_horizon_forecast_srm.py (Clean + Correct)
# ─────────────────────────────────────────────────────────────

from growth_rate_model import MeanRevertingGrowthModel
from forecast_strategies import (
    build_growth_rate_features,
    strategy4_ultra_strict_loocv,
    create_monthly_matrix,
)
from onee.utils import select_best_model
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

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION OBJECT
# ─────────────────────────────────────────────────────────────

class ForecastConfig:
    def __init__(
        self,
        project_root,
        variable="consommation_kwh",
        unit="Kwh",
        horizon=5,
        growth_model_l2=10.0,
        rho_bounds=(0.0, 1.0),
        feature_transforms=(("lag_lchg",),),#(("lag_lchg",), ("lchg",)),
        feature_lags=((1,),),#((1,), (1, 2)),
        feature_blocks=None,
        use_pf_options=[False],
        training_windows=[None],
        r2_threshold=0.6,
        client_pattern_weights = [0.3, 0.5, 0.8],
        use_client_options=[False]
    ):
        self.project_root = Path(project_root)
        self.variable = variable
        self.unit = unit
        self.horizon = horizon
        self.growth_model_l2 = growth_model_l2
        self.rho_bounds = rho_bounds
        self.feature_transforms = feature_transforms
        self.feature_lags = feature_lags
        self.feature_blocks = feature_blocks or {
            "none": [],
            "gdp_only": ["pib_mdh"],
            "sectoral_only": ["gdp_primaire", "gdp_secondaire", "gdp_tertiaire"],
            "gdp_sectoral": ["pib_mdh", "gdp_primaire", "gdp_secondaire", "gdp_tertiaire"],
        }
        self.use_pf_options = use_pf_options
        self.training_windows = training_windows
        self.r2_threshold = r2_threshold
        self.client_pattern_weights = client_pattern_weights
        self.use_client_options = use_client_options


# ─────────────────────────────────────────────────────────────
# MODEL FITTING + FORECAST FUNCTIONS
# ─────────────────────────────────────────────────────────────

def fit_growth_model_from_history(annual_series, features, config: ForecastConfig):
    model = MeanRevertingGrowthModel(
        include_ar=True,
        include_exog=features is not None and features.shape[1] > 0,
        l2_penalty=config.growth_model_l2,
        rho_bounds=config.rho_bounds,
        use_asymmetric_loss=False,
        include_ar_squared=True,
    )
    model.fit(annual_series, features)
    return model


def forecast_horizon(
    growth_model,
    df_features,
    start_year,
    fb_features,
    monthly_clients_lookup,
    use_clients,
    df_monthly,
    use_pf,
    feature_transforms,
    feature_lags
):
    """Forecast multiple years ahead using the provided growth model.

    This function accepts all parameters required by `build_growth_rate_features`
    so it can construct exogenous features for each forecast year.
    """
    preds = []
    for h in range(1, config.horizon + 1):
        target_year = start_year + h
        x_next = build_growth_rate_features(
            years=[target_year],
            feature_block=fb_features,
            df_features=df_features,
            clients_lookup=monthly_clients_lookup,
            use_clients=use_clients,
            df_monthly=df_monthly,
            use_pf=use_pf,
            transforms=feature_transforms,
            lags=feature_lags,
        )
        x_next = np.asarray(x_next, dtype=float) if x_next is not None else None
        y_pred = growth_model.predict(x_next[0] if x_next is not None else None)
        preds.append((target_year, y_pred))

        predicted_growth_rate = np.log(y_pred) - np.log(growth_model.last_y)
        growth_model.last_growth_rate = predicted_growth_rate
        growth_model.last_y = y_pred

    return preds


def strategy4_long_horizon_forecast(
    monthly_matrix,
    years,
    df_features,
    df_monthly,
    config: ForecastConfig,
    monthly_clients_lookup=None,
):
    # Step 1. Run LOOCV search across all param combinations from ForecastConfig
    base_results = strategy4_ultra_strict_loocv(
        monthly_matrix=monthly_matrix,
        years=years,
        df_features=df_features,
        df_monthly=df_monthly,
        feature_blocks=config.feature_blocks,
        monthly_clients_lookup=monthly_clients_lookup,
        use_monthly_clients_options=config.use_client_options if monthly_clients_lookup else [False],
        use_pf_options=config.use_pf_options,
        client_pattern_weights=config.client_pattern_weights,
        growth_feature_transforms=config.feature_transforms,
        growth_feature_lags=config.feature_lags,
        training_windows=config.training_windows,
    )

    best_result = select_best_model(base_results, r2_threshold=config.r2_threshold)
    print(f"\n🏆 Best model: {best_result['feature_block']} | R²={best_result.get('r2_annual', 0):.3f}")

    # Step 2. Extract best-found parameters from LOOCV
    best_transforms = best_result.get("growth_feature_transforms", ("lag_lchg",))
    best_lags = best_result.get("growth_feature_lags", (1,))
    best_use_pf = best_result.get("use_pf", False)
    best_use_clients = best_result.get("use_monthly_clients", False)
    best_fb_name = best_result["feature_block"]

    # Step 3. Build training features with best parameters
    growth_features = build_growth_rate_features(
        years=years,
        feature_block=config.feature_blocks[best_fb_name],
        df_features=df_features,
        clients_lookup=monthly_clients_lookup,
        use_clients=best_use_clients,
        df_monthly=df_monthly,
        use_pf=best_use_pf,
        transforms=best_transforms,
        lags=best_lags,
    )

    # Step 4. Fit final model with chosen params
    growth_model = fit_growth_model_from_history(
        annual_series=monthly_matrix.sum(axis=1),
        features=growth_features,
        config=config,
    )

    print("ρ (rho):", growth_model.rho)
    print("μ (mu):", growth_model.mu)

    # Step 5. Forecast horizon years using best params
    last_year = int(max(years))
    config.feature_transforms = best_transforms
    config.feature_lags = best_lags
    horizon_preds = forecast_horizon(
        growth_model=growth_model,
        df_features=df_features,
        start_year=last_year,
        fb_features=config.feature_blocks[best_fb_name],
        monthly_clients_lookup=monthly_clients_lookup,
        use_clients=best_use_clients,
        df_monthly=df_monthly,
        use_pf=best_use_pf,
        feature_transforms=best_transforms,
        feature_lags=best_lags,
    )

    # Step 6. Distribute annual forecasts to months
    mean_curve = monthly_matrix.mean(axis=0)
    mean_curve_norm = mean_curve / mean_curve.sum()
    future_monthly = [val * mean_curve_norm for _, val in horizon_preds]

    return {
        "base_result": best_result,
        "horizon_predictions": horizon_preds,
        "monthly_forecasts": future_monthly,
        "run_parameters": {
            "feature_block_name": best_fb_name,
            "feature_block": config.feature_blocks[best_fb_name],
            "use_pf": bool(best_use_pf),
            "use_clients": bool(best_use_clients),
            "feature_transforms": best_transforms,
            "feature_lags": best_lags,
            "growth_model": {
                "rho": getattr(growth_model, "rho", None),
                "mu": getattr(growth_model, "mu", None),
                "beta": getattr(growth_model, "beta", None),
                "gamma": getattr(growth_model, "gamma", None),
            },
            "use_squared": getattr(growth_model, "include_ar_squared", False),
            "use_asymmetric_loss": getattr(growth_model, "use_asymmetric_loss", False),
        },
    }


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = ForecastConfig(
        project_root=Path(__file__).resolve().parents[0],
        horizon=5,
    )

    REGIONS = [
        "Casablanca-Settat",
    ]
    # Choose which analysis parts (levels) to run
    # 1: Activities, 2: Aggregated BT, 3: Aggregated MT, 4: Total Regional,
    # 5: Individual Distributors, 6: All Distributors, 7: SRM (Regional+Dist)
    RUN_LEVELS = {1, 4}
    forecast_types = {
        # "forward": (2007, 2023),
        "backtest": (2007, 2018),
    }

    for TARGET_REGION in REGIONS:
        print(f"\n{'='*80}\n🌍 REGION: {TARGET_REGION}\n{'='*80}")

        db_regional = sqlite3.connect(config.project_root / "data/ONEE_Regional_COMPLETE_2007_2023.db")
        db_dist = sqlite3.connect(config.project_root / "data/ONEE_Distributeurs_consumption.db")

        q_regional_mt, q_regional_bt, q_dist, q_features, var_cols = get_queries_for(config.variable, TARGET_REGION)
        df_regional_mt = pd.read_sql_query(q_regional_mt, db_regional)
        df_regional_bt = pd.read_sql_query(q_regional_bt, db_regional)
        df_features = pd.read_sql_query(q_features, db_regional)
        df_dist = pd.read_sql_query(q_dist, db_dist) if q_dist is not None else None

        db_regional.close()
        db_dist.close()

        df_regional_mt["activite"] = df_regional_mt["activite"].replace("Administratif", "Administratif_mt")
        df_regional = pd.concat([df_regional_bt, df_regional_mt])

        reg_var_col = var_cols["regional"]
        require_columns(df_regional, ["annee", "mois", "activite", reg_var_col], "df_regional")
        df_regional[reg_var_col] = df_regional[reg_var_col].fillna(0)

        if df_dist is not None:
            dist_var_col = var_cols["distributor"]
            require_columns(df_dist, ["annee", "mois", "distributeur", dist_var_col], "df_dist")
            df_dist[dist_var_col] = df_dist[dist_var_col].fillna(0)

        # Lookup of previously computed monthly client predictions
        client_predictions_lookup = load_client_prediction_lookup(TARGET_REGION)
        all_results = []

        activities = sorted(df_regional["activite"].unique())
        mt_activities = ["Administratif_mt", "Agricole", "Industriel", "Résidentiel", "Tertiaire"]
        bt_activities = [a for a in activities if a not in mt_activities]

        for mode, (train_start, train_end) in forecast_types.items():
            print(f"\n🧭 MODE: {mode.upper()} — training {train_start}→{train_end}, horizon={config.horizon}")

            # ────────────────────────────────────────────────────────────────
            # LEVEL 1: INDIVIDUAL ACTIVITIES
            # ────────────────────────────────────────────────────────────────
            if 1 in RUN_LEVELS:
                print(f"\n{'#'*60}\nLEVEL 1: INDIVIDUAL ACTIVITIES\n{'#'*60}")
                for activity in activities:
                    df_activity = (
                        df_regional[df_regional["activite"] == activity][["annee", "mois", reg_var_col]]
                        .copy()
                        .rename(columns={reg_var_col: config.variable})
                    )
                    df_train = df_activity[df_activity["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(df_train, value_col=config.variable)
                    years = np.sort(df_train["annee"].unique())

                    res = strategy4_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_activity,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get(f"Activity_{activity}", {}),
                    )

                    entity_name = f"Activity_{activity}"
                    # store monthly forecasts for later aggregation
                    if "monthly_forecasts" in res:
                        client_predictions_lookup[entity_name] = {
                            y: np.array(m)
                            for y, m in zip(
                                [y for y, _ in res["horizon_predictions"]],
                                res["monthly_forecasts"]
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
                            percent_error = (v - actual)/actual * 100 if actual != 0 else None
                    
                        actuals.append(actual)
                        percent_errors.append(percent_error)


                    all_results.append({
                        "region": TARGET_REGION,
                        "mode": mode,
                        "level": entity_name,
                        "forecast_years":forecast_years,
                        "pred_annual": pred_annual,
                        "pred_monthly": res["monthly_forecasts"],
                        "run_parameters": res.get("run_parameters", {}),
                        "actuals":actuals,
                        "percent_errors":percent_errors
                    })

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
                        .rename(columns={reg_var_col: config.variable})
                    )

                    # Combine predictions from Level 1
                    aggregate_predictions(
                        client_predictions_lookup,
                        "Aggregated_BT",
                        [f"Activity_{a}" for a in bt_activities],
                    )

                    df_train = df_bt_agg[df_bt_agg["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(df_train, value_col=config.variable)
                    years = np.sort(df_train["annee"].unique())

                    res = strategy4_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_bt_agg,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get("Aggregated_BT", {}),
                    )

                    # Update aggregated lookup
                    if "monthly_forecasts" in res:
                        client_predictions_lookup["Aggregated_BT"] = {
                            y: np.array(m)
                            for y, m in zip(
                                [y for y, _ in res["horizon_predictions"]],
                                res["monthly_forecasts"]
                            )
                        }

                    all_results.append({
                        "region": TARGET_REGION,
                        "mode": mode,
                        "level": "Aggregated_BT",
                        "forecast_years": [y for y, _ in res["horizon_predictions"]],
                        "pred_annual": [float(v) for _, v in res["horizon_predictions"]],
                        "pred_monthly": res["monthly_forecasts"],
                        "run_parameters": res.get("run_parameters", {}),
                    })

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
                        .rename(columns={reg_var_col: config.variable})
                    )

                    aggregate_predictions(
                        client_predictions_lookup,
                        "Aggregated_MT",
                        [f"Activity_{a}" for a in mt_activities],
                    )

                    df_train = df_mt_agg[df_mt_agg["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(df_train, value_col=config.variable)
                    years = np.sort(df_train["annee"].unique())

                    res = strategy4_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_mt_agg,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get("Aggregated_MT", {}),
                    )

                    if "monthly_forecasts" in res:
                        client_predictions_lookup["Aggregated_MT"] = {
                            y: np.array(m)
                            for y, m in zip(
                                [y for y, _ in res["horizon_predictions"]],
                                res["monthly_forecasts"]
                            )
                        }

                    all_results.append({
                        "region": TARGET_REGION,
                        "mode": mode,
                        "level": "Aggregated_MT",
                        "forecast_years": [y for y, _ in res["horizon_predictions"]],
                        "pred_annual": [float(v) for _, v in res["horizon_predictions"]],
                        "pred_monthly": res["monthly_forecasts"],
                        "run_parameters": res.get("run_parameters", {}),
                    })

            # ────────────────────────────────────────────────────────────────
            # LEVEL 4: TOTAL REGIONAL
            # ────────────────────────────────────────────────────────────────
            if 4 in RUN_LEVELS:
                print(f"\n{'#'*60}\nLEVEL 4: TOTAL REGIONAL\n{'#'*60}")
                df_total_regional = (
                    df_regional.groupby(["annee", "mois"])
                    .agg({reg_var_col: "sum"})
                    .reset_index()
                    .rename(columns={reg_var_col: config.variable})
                )

                aggregate_predictions(
                    client_predictions_lookup,
                    "Total_Regional",
                    [f"Activity_{a}" for a in activities],
                )

                df_train = df_total_regional[df_total_regional["annee"] <= train_end]
                monthly_matrix = create_monthly_matrix(df_train, value_col=config.variable)
                years = np.sort(df_train["annee"].unique())

                res = strategy4_long_horizon_forecast(
                    monthly_matrix=monthly_matrix,
                    years=years,
                    df_features=df_features,
                    df_monthly=df_regional,
                    config=config,
                    monthly_clients_lookup=client_predictions_lookup.get("Total_Regional", {}),
                )

                if "monthly_forecasts" in res:
                    client_predictions_lookup["Total_Regional"] = {
                        y: np.array(m)
                        for y, m in zip(
                            [y for y, _ in res["horizon_predictions"]],
                            res["monthly_forecasts"]
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
                        percent_error = (v - actual)/actual * 100 if actual != 0 else None
                    actuals.append(actual)
                    percent_errors.append(percent_error)


                all_results.append({
                    "region": TARGET_REGION,
                    "mode": mode,
                    "level": "Total_Regional",
                    "forecast_years": forecast_years,
                    "pred_annual": pred_annual,
                    "pred_monthly": res["monthly_forecasts"],
                    "run_parameters": res.get("run_parameters", {}),
                    "actuals":actuals,
                    "percent_errors":percent_errors
                })

            # ────────────────────────────────────────────────────────────────
            # LEVEL 5–7: DISTRIBUTORS + SRM
            # ────────────────────────────────────────────────────────────────
            if df_dist is not None:
                # Level 5: Individual Distributors
                if 5 in RUN_LEVELS:
                    print(f"\n{'#'*60}\nLEVEL 5: INDIVIDUAL DISTRIBUTORS\n{'#'*60}")
                    for distributor in sorted(df_dist["distributeur"].unique()):
                        df_distributor = (
                            df_dist[df_dist["distributeur"] == distributor][["annee", "mois", dist_var_col]]
                            .copy()
                            .rename(columns={dist_var_col: config.variable})
                        )
                        df_train = df_distributor[df_distributor["annee"] <= train_end]
                        monthly_matrix = create_monthly_matrix(df_train, value_col=config.variable)
                        years = np.sort(df_train["annee"].unique())

                        res = strategy4_long_horizon_forecast(
                            monthly_matrix=monthly_matrix,
                            years=years,
                            df_features=df_features,
                            df_monthly=df_distributor,
                            config=config,
                            monthly_clients_lookup=client_predictions_lookup.get(f"Distributor_{distributor}", {}),
                        )

                        entity_name = f"Distributor_{distributor}"
                        if "monthly_forecasts" in res:
                            client_predictions_lookup[entity_name] = {
                                y: np.array(m)
                                for y, m in zip(
                                    [y for y, _ in res["horizon_predictions"]],
                                    res["monthly_forecasts"]
                                )
                            }

                        all_results.append({
                            "region": TARGET_REGION,
                            "mode": mode,
                            "level": entity_name,
                            "forecast_years": [y for y, _ in res["horizon_predictions"]],
                            "pred_annual": [float(v) for _, v in res["horizon_predictions"]],
                            "pred_monthly": res["monthly_forecasts"],
                            "run_parameters": res.get("run_parameters", {}),
                        })

                # Level 6: All Distributors combined
                if 6 in RUN_LEVELS:
                    print(f"\n{'#'*60}\nLEVEL 6: ALL DISTRIBUTORS COMBINED\n{'#'*60}")
                    aggregate_predictions(
                        client_predictions_lookup,
                        "All_Distributors",
                        [f"Distributor_{d}" for d in sorted(df_dist["distributeur"].unique())],
                    )
                    df_all_dist = (
                        df_dist.groupby(["annee", "mois"])
                        .agg({dist_var_col: "sum"})
                        .reset_index()
                        .rename(columns={dist_var_col: config.variable})
                    )
                    df_train = df_all_dist[df_all_dist["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(df_train, value_col=config.variable)
                    years = np.sort(df_train["annee"].unique())

                    res = strategy4_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_all_dist,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get("All_Distributors", {}),
                    )

                    all_results.append({
                        "region": TARGET_REGION,
                        "mode": mode,
                        "level": "All_Distributors",
                        "forecast_years": [y for y, _ in res["horizon_predictions"]],
                        "pred_annual": [float(v) for _, v in res["horizon_predictions"]],
                        "pred_monthly": res["monthly_forecasts"],
                        "run_parameters": res.get("run_parameters", {}),
                    })

                # Level 7: SRM
                if 7 in RUN_LEVELS:
                    print(f"\n{'#'*60}\nLEVEL 7: SRM (Regional + Distributors)\n{'#'*60}")
                    aggregate_predictions(
                        client_predictions_lookup,
                        "SRM_Regional_Plus_Dist",
                        ["Total_Regional", "All_Distributors"],
                    )
                    df_all_dist = (
                        df_dist.groupby(["annee", "mois"])
                        .agg({dist_var_col: "sum"})
                        .reset_index()
                        .rename(columns={dist_var_col: config.variable})
                    )
                    df_total_regional = (
                        df_regional.groupby(["annee", "mois"])
                        .agg({reg_var_col: "sum"})
                        .reset_index()
                        .rename(columns={reg_var_col: config.variable})
                    )
                    df_srm = (
                        pd.concat([df_total_regional, df_all_dist], ignore_index=True)
                        .groupby(["annee", "mois"])
                        .agg({config.variable: "sum"})
                        .reset_index()
                    )
                    df_train = df_srm[df_srm["annee"] <= train_end]
                    monthly_matrix = create_monthly_matrix(df_train, value_col=config.variable)
                    years = np.sort(df_train["annee"].unique())

                    res = strategy4_long_horizon_forecast(
                        monthly_matrix=monthly_matrix,
                        years=years,
                        df_features=df_features,
                        df_monthly=df_srm,
                        config=config,
                        monthly_clients_lookup=client_predictions_lookup.get("SRM_Regional_Plus_Dist", {}),
                    )

                    all_results.append({
                        "region": TARGET_REGION,
                        "mode": mode,
                        "level": "SRM_Regional_Plus_Dist",
                        "forecast_years": [y for y, _ in res["horizon_predictions"]],
                        "pred_annual": [float(v) for _, v in res["horizon_predictions"]],
                        "pred_monthly": res["monthly_forecasts"],
                        "run_parameters": res.get("run_parameters", {}),
                    })

        # ────────────────────────────────────────────────────────────────
        # OUTPUTS
        # ────────────────────────────────────────────────────────────────
        output_dir = config.project_root / "outputs_horizon" / f"horizon_{config.horizon}y" / clean_name(TARGET_REGION)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / f"{clean_name(TARGET_REGION)}_{config.variable}_horizon_{config.horizon}y.pkl", "wb") as f:
            pickle.dump(all_results, f)

        df_summary_records = []
        for r in all_results:
            for y, v, actual, percent_error in zip(r["forecast_years"], r["pred_annual"], r["actuals"], r["percent_errors"]):
                # Flatten run parameters for the summary sheet
                rp = r.get("run_parameters", {}) or {}
                growth = rp.get("growth_model", {}) or {}
                df_summary_records.append({
                    "Region": r["region"],
                    "Mode": r["mode"],
                    "Level": r["level"],
                    "Year": y,
                    "Predicted_Annual": v,
                    "Actual_Annual": actual,
                    "Percent_Error":percent_error,
                    "Feature_Block": rp.get("feature_block_name"),
                    "Use_PF": rp.get("use_pf"),
                    "Use_squared": rp.get("use_squared"),
                    "Use_asymmetric_loss": rp.get("use_asymmetric_loss"),
                    "Use_Clients": rp.get("use_clients"),
                    "Feature_Transforms": (None if rp.get("feature_transforms") is None else str(rp.get("feature_transforms"))),
                    "Feature_Lags": (None if rp.get("feature_lags") is None else str(rp.get("feature_lags"))),
                    "Growth_Rho": growth.get("rho"),
                    "Growth_Mu": growth.get("mu"),
                    "Growth_Beta": growth.get("beta"),
                    "Growth_Gamma": growth.get("gamma")
                })

        df_summary = pd.DataFrame(df_summary_records)
        out_xlsx = output_dir / f"summary_{clean_name(TARGET_REGION)}_{config.variable}_horizon_{config.horizon}y.xlsx"
        df_summary.to_excel(out_xlsx, index=False)
        print(f"\n📁 Saved horizon forecasts to {out_xlsx}")

