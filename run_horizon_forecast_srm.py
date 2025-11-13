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
        feature_transforms=(("lag_lchg",), ("lchg",)),
        feature_lags=((1,), (1, 2)),
        feature_blocks=None,
        use_pf_options=[False],
        training_windows=[10],
        r2_threshold=0.6,
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


# ─────────────────────────────────────────────────────────────
# MODEL FITTING + FORECAST FUNCTIONS
# ─────────────────────────────────────────────────────────────

def fit_growth_model_from_history(annual_series, features, config: ForecastConfig):
    model = MeanRevertingGrowthModel(
        include_ar=True,
        include_exog=features is not None and features.shape[1] > 0,
        l2_penalty=config.growth_model_l2,
        rho_bounds=config.rho_bounds,
    )
    model.fit(annual_series, features)
    return model


def forecast_horizon(growth_model, df_features, start_year, config: ForecastConfig, fb_features=None):
    preds = []
    for h in range(1, config.horizon + 1):
        target_year = start_year + h
        x_next = build_growth_rate_features(
            years=[target_year],
            feature_block=fb_features,
            df_features=df_features,
            transforms=config.feature_transforms,
            lags=config.feature_lags,
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
        use_monthly_clients_options=[False, True] if monthly_clients_lookup else [False],
        use_pf_options=config.use_pf_options,
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
    print("coef_:", getattr(growth_model, "coef_", None))

    # Step 5. Forecast horizon years using best params
    last_year = int(max(years))
    config.feature_transforms = best_transforms
    config.feature_lags = best_lags
    horizon_preds = forecast_horizon(
        growth_model=growth_model,
        df_features=df_features,
        start_year=last_year,
        config=config,
        fb_features=config.feature_blocks[best_fb_name],
    )

    # Step 6. Distribute annual forecasts to months
    mean_curve = monthly_matrix.mean(axis=0)
    mean_curve_norm = mean_curve / mean_curve.sum()
    future_monthly = [val * mean_curve_norm for _, val in horizon_preds]

    return {
        "base_result": best_result,
        "horizon_predictions": horizon_preds,
        "monthly_forecasts": future_monthly,
    }


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = ForecastConfig(
        project_root=Path(__file__).resolve().parents[0],
        horizon=5,
    )

    REGIONS = ["Casablanca-Settat"]
    forecast_types = {
        "forward": (2007, 2023),
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

        client_predictions_lookup = load_client_prediction_lookup(TARGET_REGION)
        all_results = []

        for mode, (train_start, train_end) in forecast_types.items():
            print(f"\n🧭 MODE: {mode.upper()} — training {train_start}→{train_end}, horizon={config.horizon}")
            df_total_regional = (
                df_regional.groupby(["annee", "mois"])
                .agg({reg_var_col: "sum"})
                .reset_index()
                .rename(columns={reg_var_col: config.variable})
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
                monthly_clients_lookup=client_predictions_lookup,
            )

            all_results.append({
                "region": TARGET_REGION,
                "mode": mode,
                "level": "Total_Regional",
                "forecast_years": [y for y, _ in res["horizon_predictions"]],
                "pred_annual": [float(v) for _, v in res["horizon_predictions"]],
                "pred_monthly": res["monthly_forecasts"],
            })

            print(f"✅ Completed {mode} forecast for {TARGET_REGION}")

        output_dir = config.project_root / "outputs_horizon" / f"horizon_{config.horizon}y" / clean_name(TARGET_REGION)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / f"{clean_name(TARGET_REGION)}_{config.variable}_horizon_{config.horizon}y.pkl", "wb") as f:
            pickle.dump(all_results, f)

        df_summary_records = []
        for r in all_results:
            for y, v in zip(r["forecast_years"], r["pred_annual"]):
                actual = None
                if r["mode"] == "backtest":
                    mask = df_regional["annee"] == y
                    if not df_regional.loc[mask, reg_var_col].empty:
                        actual = df_regional.loc[mask, reg_var_col].sum()
                df_summary_records.append({
                    "Region": r["region"],
                    "Mode": r["mode"],
                    "Level": r["level"],
                    "Year": y,
                    "Predicted_Annual": v,
                    "Actual_Annual": actual,
                })

        df_summary = pd.DataFrame(df_summary_records)
        out_xlsx = output_dir / f"summary_{clean_name(TARGET_REGION)}_{config.variable}_horizon_{config.horizon}y.xlsx"
        df_summary.to_excel(out_xlsx, index=False)
        print(f"\n📁 Saved horizon forecasts to {out_xlsx}")