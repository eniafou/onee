# ─────────────────────────────────────────────────────────────
# run_horizon_forecast_srm.py (Clean + Correct)
# ─────────────────────────────────────────────────────────────

from onee.growth_rate_model import MeanRevertingGrowthModelARP, MeanRevertingGrowthModel, PCAMeanRevertingGrowthModel, RawMeanRevertingGrowthModel, LocalInterpolationForecastModel, AsymmetricAdaptiveTrendModel, FullyAdaptiveTrendModel, GaussianProcessForecastModel, IntensityForecastWrapper
from forecast_strategies import (
    create_monthly_matrix,
)
from horizon_forecast_strategies import run_long_horizon_forecast
from onee.utils import fill_2020_with_avg, get_move_in_year
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
    "FullyAdaptiveTrendModel": FullyAdaptiveTrendModel,
    "GaussianProcessForecastModel": GaussianProcessForecastModel,
    "IntensityForecastWrapper": IntensityForecastWrapper,
}

class GeneralParams(BaseModel):
    project_root: Path
    exp_name: str
    horizon: int
    variable: str = "consommation"
    unit: str = "Kwh"
    r2_threshold: float = 0.6
    model_classes: List[str] = Field(
        default_factory=lambda: ["IntensityForecastWrapper"] # "MeanRevertingGrowthModelARP", "MeanRevertingGrowthModel", "PCAMeanRevertingGrowthModel", "RawMeanRevertingGrowthModel", "LocalInterpolationForecastModel", "AsymmetricAdaptiveTrendModel", "FullyAdaptiveTrendModel"
    )


class FeatureBuildingGrid(BaseModel):
    transforms: Tuple[Tuple[str, ...], ...] = (
        # ("lchg",),
        # ("lchg", "lag_lchg"),
        ("level",),
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
    use_pf: List[bool] = Field(default_factory=lambda: [True])
    use_clients: List[bool] = Field(default_factory=lambda: [False])
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
    
    mu_bounds: List[Tuple[float, float]] = Field(
        default_factory=lambda: [(None, None)]
    )
    alpha_down_bounds: List[Tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 0.05)]  # Default strict
    )

    alpha_up_bounds: List[Tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 0.05)]  # Default lenient
    )

    beta_adapt_bounds: List[Tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 0.5)]
    )

    beta_down_bounds: List[Tuple[float, float]] = Field(
        default_factory=lambda: [(4, 6)]  # Default strict
    )

    beta_up_bounds: List[Tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 1.5)]  # Default lenient
    )

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
    normalization_col: List[str] = Field(
        default_factory=lambda: ["total_active_contrats"]
    )



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
            exp_name="gp_1",
            horizon=5
        )
    )

    # Choose which analysis parts (levels) to run
    # 1: Contrat, 2: Partenaire (client), 3: Activities
    RUN_LEVELS = {3}

    forecast_types = {
        # "forward": (2007, 2023),
        "backtest": (2007, 2018),
    }

    
    db_regional_path = config.general_params.project_root / 'data/ONEE_Regional_COMPLETE_2007_2023.db'
    db_cd_path = config.general_params.project_root / "data/cd_database_2013_2023.db"

    db_regional = sqlite3.connect(db_regional_path)
    db_cd = sqlite3.connect(db_cd_path)

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

    query_contrats = "SELECT * from cd"
    df_contrats = pd.read_sql_query(query_contrats, db_cd)

    query_activite_features = "SELECT * from active_contrats_features"
    df_activite_features = pd.read_sql_query(query_activite_features, db_cd)

    db_regional.close()
    db_cd.close()

    all_results = []

    for mode, (train_start, train_end) in forecast_types.items():
        print(
            f"\n🧭 MODE: {mode.upper()} — training {train_start}→{train_end}, horizon={config.general_params.horizon}"
        )

        output_dir = (
            config.general_params.project_root
            / "outputs_horizon_cd"
            / f"{config.general_params.exp_name}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # ────────────────────────────────────────────────────────────────
        # LEVEL 1: INDIVIDUAL ACTIVITIES
        # ────────────────────────────────────────────────────────────────
        if 1 in RUN_LEVELS:
            print(f"\n{'#'*60}\nLEVEL 1: CONTRAT\n{'#'*60}")
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


            all_results_established = []
            for i, contrat in enumerate(established_contracts):
                df_contrat = df_contrats[df_contrats['contrat'] == contrat].copy()
                # df_activity = fill_2020_with_avg(df_activity, config.general_params.variable)
                df_train = df_contrat[df_contrat["annee"] <= train_end]
                monthly_matrix = create_monthly_matrix(
                    df_train, value_col=config.general_params.variable
                )
                years = np.sort(df_train["annee"].unique())

                res = run_long_horizon_forecast(
                    monthly_matrix=monthly_matrix,
                    years=years,
                    df_features=df_features,
                    df_monthly=df_contrat,
                    config=config,
                    MODEL_REGISTRY=MODEL_REGISTRY,
                )

                forecast_years = [y for y, _ in res["horizon_predictions"]]
                pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                actuals = []
                percent_errors = []
                for y, v in zip(forecast_years, pred_annual):
                    actual = None
                    percent_error = None
                    mask = df_contrat["annee"] == y
                    if not df_contrat.loc[mask, config.general_params.variable].empty:
                        actual = df_contrat.loc[mask, config.general_params.variable].sum()
                        percent_error = (
                            (v - actual) / actual * 100 if actual != 0 else None
                        )

                    actuals.append(actual)
                    percent_errors.append(percent_error)

                all_results.append(
                    {
                        "mode": mode,
                        "level": f"Contrat_{contrat}",
                        "forecast_years": forecast_years,
                        "pred_annual": pred_annual,
                        "pred_monthly": res["monthly_forecasts"],
                        "run_parameters": res.get("run_parameters", {}),
                        "actuals": actuals,
                        "percent_errors": percent_errors,
                    }
                )
            all_results_contrats = all_results_established
            all_results += all_results_contrats
        
        if 3 in RUN_LEVELS:
            print(f"\n{'#'*60}")
            print(f"LEVEL 3: ACTIVITE")
            print(f"{'#'*60}")

            activites = sorted(df_contrats['activite'].unique())
            # activites = ["ADMINISTRATION PUBLIQUE"]

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
                print("###############################", activite)
                df_activite = df_contrats[df_contrats['activite'] == activite].copy()
                df_activite = df_activite.groupby(['annee', 'mois']).agg({
                    config.general_params.variable: 'sum',
                    'temperature': 'mean',
                    'puissance facturée': 'sum'
                }).reset_index()
                
                df_train = df_activite[df_activite["annee"] <= train_end]
                monthly_matrix = create_monthly_matrix(
                    df_train, value_col=config.general_params.variable
                )
                years = np.sort(df_train["annee"].unique())

                df_a = df_activite_features[df_activite_features["activite"] == activite]
                df_a = df_a.merge(df_features, on ="annee")

                config.feature_building_grid.feature_block = [
                    # [],
                    ['total_active_contrats', 'just_started', 'two_years_old', 'three_years_old', 'more_than_3_years_old', 'pib_mdh'],
                    # ["pib_mdh", "gdp_primaire", "gdp_secondaire", "gdp_tertiaire"],
                ]
                res = run_long_horizon_forecast(
                    monthly_matrix=monthly_matrix,
                    years=years,
                    df_features=df_a,
                    df_monthly=df_activite,
                    config=config,
                    MODEL_REGISTRY=MODEL_REGISTRY,
                    region_entity = f"CD - activité {activite}",
                    save_folder=output_dir / f"{activite}",
                )
                
                forecast_years = [y for y, _ in res["horizon_predictions"]]
                pred_annual = [float(v) for _, v in res["horizon_predictions"]]
                actuals = []
                percent_errors = []
                for y, v in zip(forecast_years, pred_annual):
                    actual = None
                    percent_error = None
                    mask = df_activite["annee"] == y
                    if not df_activite.loc[mask, config.general_params.variable].empty:
                        actual = df_activite.loc[mask, config.general_params.variable].sum()
                        percent_error = (
                            (v - actual) / actual * 100 if actual != 0 else None
                        )

                    actuals.append(actual)
                    percent_errors.append(percent_error)

                all_results_established_activite.append(
                    {
                        "mode": mode,
                        "level": f"Activité_{activite}",
                        "forecast_years": forecast_years,
                        "pred_annual": pred_annual,
                        "pred_monthly": res["monthly_forecasts"],
                        "run_parameters": res.get("run_parameters", {}),
                        "actuals": actuals,
                        "percent_errors": percent_errors,
                    }
                )

        

            # --- Merge all ---
            all_results_activite = (
                all_results_established_activite
            )

            all_results += all_results_activite

    # ────────────────────────────────────────────────────────────────
    # OUTPUTS
    # ────────────────────────────────────────────────────────────────
    with open(
        output_dir
        / f"{config.general_params.variable}_{config.general_params.exp_name}.pkl",
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
        / f"summary_{config.general_params.variable}_{config.general_params.exp_name}.xlsx"
    )
    df_summary.to_excel(out_xlsx, index=False)
    print(f"\n📁 Saved horizon forecasts to {out_xlsx}")
