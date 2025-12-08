# ─────────────────────────────────────────────────────────────
# run_horizon_forecast_srm.py (Clean + Correct)
# ─────────────────────────────────────────────────────────────

from short_term_forecast_strategies import (
    create_monthly_matrix,
)
from long_term_forecast_strategies import run_long_horizon_forecast
from onee.utils import fill_2020_with_avg, clean_name
from onee.config.ltf_config import LongTermForecastConfig
from onee.data.loader import DataLoader
from onee.data.names import Aliases, GRDValues

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
import sys

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load configuration from YAML
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ltf_srm.yaml"
    config = LongTermForecastConfig.from_yaml(config_path)
    
    # Resolve project root to absolute path
    if not config.project.project_root.is_absolute():
        config.project.project_root = Path(__file__).resolve().parent
    
    # Extract config values
    REGIONS = config.data.regions
    RUN_LEVELS = set(config.data.run_levels)
    forecast_runs = config.temporal.forecast_runs
    
    # Get model registry
    MODEL_REGISTRY = config.get_model_registry()
    
    # Initialize DataLoader
    data_loader = DataLoader(config.project.project_root)

    for TARGET_REGION in REGIONS:
        print(f"\n{'='*80}\n🌍 REGION: {TARGET_REGION}\n{'='*80}")

        output_dir = config.get_output_dir(TARGET_REGION)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data using DataLoader
        df_regional, df_features, df_dist, var_cols = data_loader.load_srm_data(
            db_path=config.project.project_root / config.data.db_path,
            variable=config.data.target_variable,
            target_region=TARGET_REGION,
        )
        
        reg_var_col = var_cols["regional"]
        dist_var_col = var_cols["distributor"]

        # Lookup of previously computed monthly client predictions
        client_predictions_lookup = data_loader.load_client_prediction_lookup(
            TARGET_REGION
        )
        all_results = []

        activities = sorted(df_regional[Aliases.ACTIVITE].unique())
        mt_activities = [
            GRDValues.ACTIVITY_ADMINISTRATIF_MT,
            GRDValues.ACTIVITY_AGRICOLE,
            GRDValues.ACTIVITY_INDUSTRIEL,
            GRDValues.ACTIVITY_RESIDENTIEL,
            GRDValues.ACTIVITY_TERTIAIRE,
        ]
        bt_activities = [a for a in activities if a not in mt_activities]

        for train_start, train_end in forecast_runs:
            print(
                f"\n🧭 Training {train_start}→{train_end}, horizon={config.temporal.horizon}"
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
                        df_regional[df_regional[Aliases.ACTIVITE] == activity][
                            [Aliases.ANNEE, Aliases.MOIS, reg_var_col]
                        ]
                        .copy()
                        .rename(columns={reg_var_col: config.data.target_variable})
                    )
                    if config.data.impute_2020:
                        df_activity = fill_2020_with_avg(df_activity, reg_var_col)

                    df_train = df_activity[
                        (df_activity[Aliases.ANNEE] >= train_start) & (df_activity[Aliases.ANNEE] <= train_end)
                    ]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.data.target_variable
                    )
                    years = np.sort(df_train[Aliases.ANNEE].unique())

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
                        mask = df_activity[Aliases.ANNEE] == y
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
                            "train_start": train_start,
                            "train_end": train_end,
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
                df_bt = df_regional[df_regional[Aliases.ACTIVITE].isin(bt_activities)]
                if not df_bt.empty:
                    df_bt_agg = (
                        df_bt.groupby([Aliases.ANNEE, Aliases.MOIS])
                        .agg({reg_var_col: "sum"})
                        .reset_index()
                        .rename(columns={reg_var_col: config.data.target_variable})
                    )

                    # Combine predictions from Level 1
                    DataLoader.aggregate_predictions(
                        client_predictions_lookup,
                        "Aggregated_BT",
                        [f"Activity_{a}" for a in bt_activities],
                    )

                    df_train = df_bt_agg[
                        (df_bt_agg[Aliases.ANNEE] >= train_start) & (df_bt_agg[Aliases.ANNEE] <= train_end)
                    ]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.data.target_variable
                    )
                    years = np.sort(df_train[Aliases.ANNEE].unique())

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
                        mask = df_bt_agg[Aliases.ANNEE] == y
                        if not df_bt_agg.loc[mask, config.data.target_variable].empty:
                            actual = df_bt_agg.loc[mask, config.data.target_variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "region": TARGET_REGION,
                            "train_start": train_start,
                            "train_end": train_end,
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
                df_mt = df_regional[df_regional[Aliases.ACTIVITE].isin(mt_activities)]
                if not df_mt.empty:
                    df_mt_agg = (
                        df_mt.groupby([Aliases.ANNEE, Aliases.MOIS])
                        .agg({reg_var_col: "sum"})
                        .reset_index()
                        .rename(columns={reg_var_col: config.data.target_variable})
                    )

                    DataLoader.aggregate_predictions(
                        client_predictions_lookup,
                        "Aggregated_MT",
                        [f"Activity_{a}" for a in mt_activities],
                    )

                    df_train = df_mt_agg[
                        (df_mt_agg[Aliases.ANNEE] >= train_start) & (df_mt_agg[Aliases.ANNEE] <= train_end)
                    ]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.data.target_variable
                    )
                    years = np.sort(df_train[Aliases.ANNEE].unique())

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
                        mask = df_mt_agg[Aliases.ANNEE] == y
                        if not df_mt_agg.loc[mask, config.data.target_variable].empty:
                            actual = df_mt_agg.loc[mask, config.data.target_variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "region": TARGET_REGION,
                            "train_start": train_start,
                            "train_end": train_end,
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
                    df_regional.groupby([Aliases.ANNEE, Aliases.MOIS])
                    .agg({reg_var_col: "sum"})
                    .reset_index()
                    .rename(columns={reg_var_col: config.data.target_variable})
                )
                if config.data.impute_2020:
                    df_total_regional = fill_2020_with_avg(df_total_regional, reg_var_col)
                DataLoader.aggregate_predictions(
                    client_predictions_lookup,
                    "Total_Regional",
                    [f"Activity_{a}" for a in activities],
                )

                df_train = df_total_regional[
                    (df_total_regional[Aliases.ANNEE] >= train_start) & (df_total_regional[Aliases.ANNEE] <= train_end)
                ]
                monthly_matrix = create_monthly_matrix(
                    df_train, value_col=config.data.target_variable
                )
                years = np.sort(df_train[Aliases.ANNEE].unique())

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
                    mask = df_total_regional[Aliases.ANNEE] == y
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
                        "train_start": train_start,
                        "train_end": train_end,
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
                    for distributor in sorted(df_dist[Aliases.DISTRIBUTEUR].unique()):
                        df_distributor = (
                            df_dist[df_dist[Aliases.DISTRIBUTEUR] == distributor][
                                [Aliases.ANNEE, Aliases.MOIS, dist_var_col]
                            ]
                            .copy()
                            .rename(columns={dist_var_col: config.data.target_variable})
                        )
                        if config.data.impute_2020:
                            df_distributor = fill_2020_with_avg(
                                df_distributor, dist_var_col
                            )
                        df_train = df_distributor[
                            (df_distributor[Aliases.ANNEE] >= train_start) & (df_distributor[Aliases.ANNEE] <= train_end)
                        ]
                        monthly_matrix = create_monthly_matrix(
                            df_train, value_col=config.data.target_variable
                        )
                        years = np.sort(df_train[Aliases.ANNEE].unique())

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
                            mask = df_distributor[Aliases.ANNEE] == y
                            if not df_distributor.loc[mask, config.data.target_variable].empty:
                                actual = df_distributor.loc[mask, config.data.target_variable].sum()
                                percent_error = (
                                    (v - actual) / actual * 100 if actual != 0 else None
                                )

                            actuals.append(actual)
                            percent_errors.append(percent_error)

                        all_results.append(
                            {
                                "region": TARGET_REGION,
                                "train_start": train_start,
                                "train_end": train_end,
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
                    DataLoader.aggregate_predictions(
                        client_predictions_lookup,
                        "All_Distributors",
                        [
                            f"Distributor_{d}"
                            for d in sorted(df_dist[Aliases.DISTRIBUTEUR].unique())
                        ],
                    )
                    df_all_dist = (
                        df_dist.groupby([Aliases.ANNEE, Aliases.MOIS])
                        .agg({dist_var_col: "sum"})
                        .reset_index()
                        .rename(columns={dist_var_col: config.data.target_variable})
                    )
                    if config.data.impute_2020:
                        df_all_dist_agg = fill_2020_with_avg(df_all_dist_agg, dist_var_col)
                    df_train = df_all_dist[
                        (df_all_dist[Aliases.ANNEE] >= train_start) & (df_all_dist[Aliases.ANNEE] <= train_end)
                    ]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.data.target_variable
                    )
                    years = np.sort(df_train[Aliases.ANNEE].unique())

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
                        mask = df_all_dist[Aliases.ANNEE] == y
                        if not df_all_dist.loc[mask, config.data.target_variable].empty:
                            actual = df_all_dist.loc[mask, config.data.target_variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "region": TARGET_REGION,
                            "train_start": train_start,
                            "train_end": train_end,
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
                    DataLoader.aggregate_predictions(
                        client_predictions_lookup,
                        "SRM_Regional_Plus_Dist",
                        ["Total_Regional", "All_Distributors"],
                    )
                    df_all_dist = (
                        df_dist.groupby([Aliases.ANNEE, Aliases.MOIS])
                        .agg({dist_var_col: "sum"})
                        .reset_index()
                        .rename(columns={dist_var_col: config.data.target_variable})
                    )
                    df_total_regional = (
                        df_regional.groupby([Aliases.ANNEE, Aliases.MOIS])
                        .agg({reg_var_col: "sum"})
                        .reset_index()
                        .rename(columns={reg_var_col: config.data.target_variable})
                    )
                    df_srm = (
                        pd.concat([df_total_regional, df_all_dist], ignore_index=True)
                        .groupby([Aliases.ANNEE, Aliases.MOIS])
                        .agg({config.data.target_variable: "sum"})
                        .reset_index()
                    )

                    # if config.data.impute_2020:
                    #     df_srm = fill_2020_with_avg(df_srm, config.data.target_variable)

                    df_srm = df_srm[df_srm[Aliases.ANNEE] >= 2013]
                    df_train = df_srm[df_srm[Aliases.ANNEE] <= train_end]
                    monthly_matrix = create_monthly_matrix(
                        df_train, value_col=config.data.target_variable
                    )
                    years = np.sort(df_train[Aliases.ANNEE].unique())

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
                        mask = df_srm[Aliases.ANNEE] == y
                        if not df_srm.loc[mask, config.data.target_variable].empty:
                            actual = df_srm.loc[mask, config.data.target_variable].sum()
                            percent_error = (
                                (v - actual) / actual * 100 if actual != 0 else None
                            )

                        actuals.append(actual)
                        percent_errors.append(percent_error)

                    all_results.append(
                        {
                            "region": TARGET_REGION,
                            "train_start": train_start,
                            "train_end": train_end,
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
            / f"{clean_name(TARGET_REGION)}_{config.data.target_variable}_{config.project.exp_name}.pkl",
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
                        "Train_Start": r.get("train_start"),
                        "Train_End": r.get("train_end"),
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
            / f"summary_{clean_name(TARGET_REGION)}_{config.data.target_variable}_{config.project.exp_name}.xlsx"
        )
        df_summary.to_excel(out_xlsx, index=False)
        print(f"\n📁 Saved horizon forecasts to {out_xlsx}")
