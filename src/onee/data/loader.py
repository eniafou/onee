"""
Centralized data loading module for ONEE forecasting project.

This module provides a unified DataLoader class that handles all database operations,
query generation, and data preprocessing for both SRM (regional) and CD (contracts) data.
"""

import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from onee.utils import clean_name, require_columns
from onee.data.names import (
    Tables, GRDColumns, ExogenousColumns, CDColumns,
    Aliases, GRDValues, build_variable_specs
)
from onee.utils import get_move_in_year, get_move_out_year

class DataLoader:
    """
    Centralized data loader for ONEE forecasting pipelines.
    
    Handles:
    - Database connections and queries
    - Data loading for SRM (regional) and CD (contracts)
    - Client prediction lookup management
    - Prediction aggregation
    """
    
    # Variable specifications for different target variables
    # Now loaded dynamically from names.py
    VARIABLE_SPECS = build_variable_specs()
    
    def __init__(self, project_root: Path):
        """
        Initialize the DataLoader.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        
    def _connect_db(self, db_path: Path) -> sqlite3.Connection:
        """Create a database connection."""
        return sqlite3.connect(db_path)
    
    # ============================================================================
    # SRM (Regional) Data Loading
    # ============================================================================
    
    def get_queries_for(self, variable: str, target_region: str) -> Tuple[str, str, Optional[str], str, Dict[str, str]]:
        """
        Generate SQL queries for a specific variable and region.
        
        Args:
            variable: Target variable (e.g., 'nbr_clients', 'consommation_kwh')
            target_region: Target region name
            
        Returns:
            Tuple of (query_regional_mt, query_regional_bt, query_dist, query_features, var_cols)
        """
        if variable not in self.VARIABLE_SPECS:
            raise ValueError(f"Unsupported VARIABLE='{variable}'. Available: {list(self.VARIABLE_SPECS)}")
        
        spec = self.VARIABLE_SPECS[variable]
        
        query_regional_mt = f"""
            SELECT
                {GRDColumns.YEAR} as {Aliases.ANNEE},
                {GRDColumns.MONTH} as {Aliases.MOIS},
                {GRDColumns.ACTIVITY} as {Aliases.ACTIVITE},
                {spec['regional_select_expr']}
            FROM {Tables.GRD}
            WHERE {GRDColumns.GRD} = '{GRDValues.GRD_SRM}' AND {GRDColumns.CLASS} = '{GRDValues.CLASS_MT}' AND {GRDColumns.REGION} = '{target_region}'
            ORDER BY {GRDColumns.YEAR}, {GRDColumns.MONTH}, {GRDColumns.ACTIVITY}
        """
        
        query_regional_bt = f"""
            SELECT
                {GRDColumns.YEAR} as {Aliases.ANNEE},
                {GRDColumns.MONTH} as {Aliases.MOIS},
                {GRDColumns.ACTIVITY} as {Aliases.ACTIVITE},
                {spec['regional_select_expr']}
            FROM {Tables.GRD}
            WHERE {GRDColumns.GRD} = '{GRDValues.GRD_SRM}' AND {GRDColumns.CLASS} = '{GRDValues.CLASS_BT}' AND {GRDColumns.REGION} = '{target_region}'
            ORDER BY {GRDColumns.YEAR}, {GRDColumns.MONTH}, {GRDColumns.ACTIVITY}
        """
        
        query_dist = None
        if spec["distributor_supported"]:
            query_dist = f"""
                SELECT
                    {GRDColumns.YEAR} as {Aliases.ANNEE},
                    {GRDColumns.MONTH} as {Aliases.MOIS},
                    {GRDColumns.GRD} as {Aliases.DISTRIBUTEUR},
                    {spec['distributor_select_expr']}
                FROM {Tables.GRD}
                WHERE {GRDColumns.GRD} != '{GRDValues.GRD_SRM}' AND {GRDColumns.REGION} = '{target_region}'
                GROUP BY {GRDColumns.YEAR}, {GRDColumns.MONTH}, {GRDColumns.GRD}
                ORDER BY {GRDColumns.YEAR}, {GRDColumns.MONTH}, {GRDColumns.GRD}
            """
        
        query_features = f"""
            SELECT
                {ExogenousColumns.YEAR} as {Aliases.ANNEE},
                AVG({ExogenousColumns.PIB_MDH}) as {Aliases.PIB_MDH},
                AVG({ExogenousColumns.PRIMAIRE}) as {Aliases.GDP_PRIMAIRE},
                AVG({ExogenousColumns.SECONDAIRE}) as {Aliases.GDP_SECONDAIRE},
                AVG({ExogenousColumns.TERTIAIRE}) as {Aliases.GDP_TERTIAIRE}
            FROM {Tables.EXOGENOUS_DATA}
            WHERE {ExogenousColumns.REGION} = '{target_region}'
            GROUP BY {ExogenousColumns.YEAR}
        """
        
        var_cols = {
            "regional": spec["regional_var_col"],
            "distributor": spec["distributor_var_col"],
        }
        
        return query_regional_mt, query_regional_bt, query_dist, query_features, var_cols
    
    def load_srm_data(
        self,
        db_path: Path,
        variable: str,
        target_region: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, str]]:
        """
        Load SRM (regional) data for a specific region and variable.
        
        Args:
            db_regional_path: Path to regional database
            db_dist_path: Path to distributors database
            variable: Target variable
            target_region: Target region name
            
        Returns:
            Tuple of (df_regional, df_features, df_dist, var_cols)
        """
        # Generate queries
        q_regional_mt, q_regional_bt, q_dist, q_features, var_cols = self.get_queries_for(variable, target_region)
        
        # Connect to databases
        db = self._connect_db(db_path)
        
        # Load regional data
        df_regional_mt = pd.read_sql_query(q_regional_mt, db)
        df_regional_bt = pd.read_sql_query(q_regional_bt, db)
        df_features = pd.read_sql_query(q_features, db)
        
        # Load distributor data if supported
        df_dist = pd.read_sql_query(q_dist, db) if q_dist is not None else None
        
        # Close connections
        db.close()
        
        # Combine MT and BT data
        df_regional_mt[Aliases.ACTIVITE] = df_regional_mt[Aliases.ACTIVITE].replace(
            GRDValues.ACTIVITY_ADMINISTRATIF, GRDValues.ACTIVITY_ADMINISTRATIF_MT
        )
        df_regional = pd.concat([df_regional_bt, df_regional_mt])
        
        # Validate and clean
        reg_var_col = var_cols["regional"]
        require_columns(df_regional, [Aliases.ANNEE, Aliases.MOIS, Aliases.ACTIVITE, reg_var_col], "df_regional")
        df_regional[reg_var_col] = df_regional[reg_var_col].fillna(0)
        
        if df_dist is not None:
            dist_var_col = var_cols["distributor"]
            require_columns(df_dist, [Aliases.ANNEE, Aliases.MOIS, Aliases.DISTRIBUTEUR, dist_var_col], "df_dist")
            df_dist[dist_var_col] = df_dist[dist_var_col].fillna(0)
        
        return df_regional, df_features, df_dist, var_cols
    
    # ============================================================================
    # CD (Contracts) Data Loading
    # ============================================================================
    
    def load_cd_data(
        self,
        db_path: Path,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load CD (contracts) data.
        
        Args:
            db_regional_path: Path to regional database
            db_cd_path: Path to CD database
            include_activite_features: Whether to load activity features
            
        Returns:
            Tuple of (df_contrats, df_features, df_activite_features)
        """
        # Connect to databases
        db = self._connect_db(db_path)
        
        # Load features from regional database
        query_features = f"""
        SELECT {ExogenousColumns.YEAR} as {Aliases.ANNEE}, 
               SUM({ExogenousColumns.PIB_MDH}) as {Aliases.PIB_MDH},
               SUM({ExogenousColumns.PRIMAIRE}) as {Aliases.GDP_PRIMAIRE},
               SUM({ExogenousColumns.SECONDAIRE}) as {Aliases.GDP_SECONDAIRE},
               SUM({ExogenousColumns.TERTIAIRE}) as {Aliases.GDP_TERTIAIRE}
        FROM {Tables.EXOGENOUS_DATA}
        GROUP BY {ExogenousColumns.YEAR}
        """
        df_features = pd.read_sql_query(query_features, db)
        
        # Load contracts data
        query_contrats = f"""
        SELECT 
        {CDColumns.REGION} as {Aliases.REGION},
        {CDColumns.PARTENAIRE} as {Aliases.PARTENAIRE}, 
        {CDColumns.NUMERO_DE_CONTRAT} as {Aliases.CONTRAT}, 
        {CDColumns.YEAR} as {Aliases.ANNEE}, 
        {CDColumns.MONTH} as {Aliases.MOIS}, 
        {CDColumns.ACTIVITE} as {Aliases.ACTIVITE}, 
        {CDColumns.CONSOMMATION_KWH} as {Aliases.CONSOMMATION_KWH},
        {CDColumns.CONSOMMATION_ZCONHC} as {Aliases.CONSOMMATION_ZCONHC},
        {CDColumns.CONSOMMATION_ZCONHL} as {Aliases.CONSOMMATION_ZCONHL},
        {CDColumns.CONSOMMATION_ZCONHP} as {Aliases.CONSOMMATION_ZCONHP},
        {CDColumns.NIVEAU_TENSION} as {Aliases.NIVEAU_TENSION},
        {CDColumns.PUISSANCE_FACTUREE} as '{Aliases.PUISSANCE_FACTUREE}',
        {CDColumns.PUISSANCE_APPELEE} as '{Aliases.PUISSANCE_APPELEE}',
        {CDColumns.DATE_EMMENAGEMENT} as '{Aliases.DATE_EMMENAGEMENT}',
        {CDColumns.DATE_DEMENAGEMENT} as '{Aliases.DATE_DEMENAGEMENT}'
        FROM {Tables.CD}
        """
        df_contrats = pd.read_sql_query(query_contrats, db)
        
        
        db.close()
        
        return df_contrats, df_features
    
    def compute_contract_features(self, df, entity_col, end_year=None, future_just_started=0):
        """
        Compute contract features aggregated by entity (activite, region, etc.) and year.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing contract data
        entity_col : str, default="activite"
            Column name to group by (e.g., "activite", "region")
        end_year : int, optional
            If provided, forecast values until this year
        future_just_started : int, default=0
            Number of new contracts starting each future year
        
        Returns:
        --------
        pd.DataFrame
            Aggregated features by entity and year
        """
        
        # Pre-compute start/end years and aggregate puissance for each contract per year
        contract_puissance = (
            df.groupby([Aliases.CONTRAT, Aliases.ANNEE])
            .agg({
                Aliases.PUISSANCE_FACTUREE: "sum",
                Aliases.PUISSANCE_APPELEE: "sum",
                entity_col: "first"
            })
            .reset_index()
        )
        
        # Pre-compute start and end years for each contract (done once)
        contract_years = (
            df.groupby(Aliases.CONTRAT)
            .apply(lambda g: pd.Series({
                "start_year": get_move_in_year(g),
                "finish_year": get_move_out_year(g),
                entity_col: g[entity_col].iloc[0]
            }))
            .reset_index()
        )
        
        # Remove contracts with incomplete info
        contract_years = contract_years.dropna(subset=["start_year", "finish_year"])
        contract_years["start_year"] = contract_years["start_year"].astype(int)
        contract_years["finish_year"] = contract_years["finish_year"].astype(int)
        
        # Merge puissance data with contract years
        contract_data = contract_years.merge(contract_puissance, on=[Aliases.CONTRAT, entity_col], how="left")
        
        def add_features_optimized(entity_value, annee, contract_data):
            # Filter contracts for this entity that are active in this year
            mask = (
                (contract_data[entity_col] == entity_value) &
                (contract_data["start_year"] <= annee) &
                (contract_data["finish_year"] >= annee) &
                (contract_data[Aliases.ANNEE] == annee)
            )
            active_contracts = contract_data[mask]
            
            # Calculate age for each contract
            ages = annee - active_contracts["start_year"]
            
            # Create age masks
            just_started_mask = ages == 0
            two_years_mask = ages == 1
            three_years_mask = ages == 2
            more_than_3_mask = ages > 2
            
            return pd.Series({
                entity_col: entity_value,
                Aliases.ANNEE: annee,
                # Contract counts
                Aliases.TOTAL_ACTIVE_CONTRATS: len(active_contracts),
                Aliases.JUST_STARTED: just_started_mask.sum(),
                Aliases.TWO_YEARS_OLD: two_years_mask.sum(),
                Aliases.THREE_YEARS_OLD: three_years_mask.sum(),
                Aliases.MORE_THAN_3_YEARS_OLD: more_than_3_mask.sum(),
                # Puissance facturée
                Aliases.PUISSANCE_FACTUREE_TOTAL: active_contracts[Aliases.PUISSANCE_FACTUREE].sum(),
                Aliases.PUISSANCE_FACTUREE_JUST_STARTED: active_contracts.loc[just_started_mask, Aliases.PUISSANCE_FACTUREE].sum(),
                Aliases.PUISSANCE_FACTUREE_TWO_YEARS_OLD: active_contracts.loc[two_years_mask, Aliases.PUISSANCE_FACTUREE].sum(),
                Aliases.PUISSANCE_FACTUREE_THREE_YEARS_OLD: active_contracts.loc[three_years_mask, Aliases.PUISSANCE_FACTUREE].sum(),
                Aliases.PUISSANCE_FACTUREE_MORE_THAN_3_YEARS_OLD: active_contracts.loc[more_than_3_mask, Aliases.PUISSANCE_FACTUREE].sum(),
                # Puissance appelée
                Aliases.PUISSANCE_APPELEE_TOTAL: active_contracts[Aliases.PUISSANCE_APPELEE].sum(),
                Aliases.PUISSANCE_APPELEE_JUST_STARTED: active_contracts.loc[just_started_mask, Aliases.PUISSANCE_APPELEE].sum(),
                Aliases.PUISSANCE_APPELEE_TWO_YEARS_OLD: active_contracts.loc[two_years_mask, Aliases.PUISSANCE_APPELEE].sum(),
                Aliases.PUISSANCE_APPELEE_THREE_YEARS_OLD: active_contracts.loc[three_years_mask, Aliases.PUISSANCE_APPELEE].sum(),
                Aliases.PUISSANCE_APPELEE_MORE_THAN_3_YEARS_OLD: active_contracts.loc[more_than_3_mask, Aliases.PUISSANCE_APPELEE].sum(),
            })
        
        def get_new_year_values(past_values, entity_value, annee, contract_years):
            """
            Calculate values for future years by aging contracts and removing finished ones.
            """
            # Get contracts that will finish this year for this entity
            finished_mask = (
                (contract_years[entity_col] == entity_value) &
                (contract_years["finish_year"] == annee - 1)
            )
            finished_contracts = contract_years[finished_mask]
            
            # Calculate ages of finished contracts (at the time they finish)
            finished_ages = (annee - 1) - finished_contracts["start_year"]
            
            # Count how many contracts finish in each age category
            finished_counts = {
                "just_started": (finished_ages == 0).sum(),
                "two_years_old": (finished_ages == 1).sum(),
                "three_years_old": (finished_ages == 2).sum(),
                "more_than_3_years_old": (finished_ages > 2).sum()
            }
            
            out = {}
            out[entity_col] = entity_value
            out[Aliases.ANNEE] = annee
            
            # Assuming no new contracts start in future years (or use parameter)
            out[Aliases.JUST_STARTED] = future_just_started
            
            # Age existing contracts by 1 year and remove finished ones
            out[Aliases.TWO_YEARS_OLD] = past_values[Aliases.JUST_STARTED] - finished_counts["just_started"]
            out[Aliases.THREE_YEARS_OLD] = past_values[Aliases.TWO_YEARS_OLD] - finished_counts["two_years_old"]
            out[Aliases.MORE_THAN_3_YEARS_OLD] = (
                past_values[Aliases.MORE_THAN_3_YEARS_OLD] + 
                past_values[Aliases.THREE_YEARS_OLD] - 
                finished_counts["three_years_old"] - 
                finished_counts["more_than_3_years_old"]
            )
            
            # Recalculate total
            out[Aliases.TOTAL_ACTIVE_CONTRATS] = (
                out[Aliases.JUST_STARTED] + 
                out[Aliases.TWO_YEARS_OLD] + 
                out[Aliases.THREE_YEARS_OLD] + 
                out[Aliases.MORE_THAN_3_YEARS_OLD]
            )
            
            # Puissance values - age them similarly
            out[Aliases.PUISSANCE_FACTUREE_JUST_STARTED] = 0  # No new contracts
            out[Aliases.PUISSANCE_FACTUREE_TWO_YEARS_OLD] = past_values[Aliases.PUISSANCE_FACTUREE_JUST_STARTED]
            out[Aliases.PUISSANCE_FACTUREE_THREE_YEARS_OLD] = past_values[Aliases.PUISSANCE_FACTUREE_TWO_YEARS_OLD]
            out[Aliases.PUISSANCE_FACTUREE_MORE_THAN_3_YEARS_OLD] = (
                past_values[Aliases.PUISSANCE_FACTUREE_MORE_THAN_3_YEARS_OLD] + 
                past_values[Aliases.PUISSANCE_FACTUREE_THREE_YEARS_OLD]
            )
            out[Aliases.PUISSANCE_FACTUREE_TOTAL] = (
                out[Aliases.PUISSANCE_FACTUREE_JUST_STARTED] +
                out[Aliases.PUISSANCE_FACTUREE_TWO_YEARS_OLD] +
                out[Aliases.PUISSANCE_FACTUREE_THREE_YEARS_OLD] +
                out[Aliases.PUISSANCE_FACTUREE_MORE_THAN_3_YEARS_OLD]
            )
            
            # Same for puissance_appelee
            out[Aliases.PUISSANCE_APPELEE_JUST_STARTED] = 0
            out[Aliases.PUISSANCE_APPELEE_TWO_YEARS_OLD] = past_values[Aliases.PUISSANCE_APPELEE_JUST_STARTED]
            out[Aliases.PUISSANCE_APPELEE_THREE_YEARS_OLD] = past_values[Aliases.PUISSANCE_APPELEE_TWO_YEARS_OLD]
            out[Aliases.PUISSANCE_APPELEE_MORE_THAN_3_YEARS_OLD] = (
                past_values[Aliases.PUISSANCE_APPELEE_MORE_THAN_3_YEARS_OLD] + 
                past_values[Aliases.PUISSANCE_APPELEE_THREE_YEARS_OLD]
            )
            out[Aliases.PUISSANCE_APPELEE_TOTAL] = (
                out[Aliases.PUISSANCE_APPELEE_JUST_STARTED] +
                out[Aliases.PUISSANCE_APPELEE_TWO_YEARS_OLD] +
                out[Aliases.PUISSANCE_APPELEE_THREE_YEARS_OLD] +
                out[Aliases.PUISSANCE_APPELEE_MORE_THAN_3_YEARS_OLD]
            )
            
            return out
        
        # Get all unique combinations of entity and annee
        entities = df[entity_col].unique()
        years = df[Aliases.ANNEE].unique()
        max_year_in_data = int(years.max())
        
        # Generate results for existing data
        result_df = pd.DataFrame([
            add_features_optimized(entity, annee, contract_data)
            for entity in entities
            for annee in sorted(years)
        ])
        
        # If end_year is provided, forecast future years
        if end_year is not None and end_year > max_year_in_data:
            future_rows = []
            
            for entity in entities:
                # Get the last year's values for this entity
                entity_data = result_df[result_df[entity_col] == entity].sort_values(Aliases.ANNEE)
                
                if len(entity_data) == 0:
                    continue
                    
                last_values = entity_data.iloc[-1].to_dict()
                
                # Generate values for each future year
                for future_year in range(max_year_in_data + 1, end_year + 1):
                    new_values = get_new_year_values(last_values, entity, future_year, contract_years)
                    future_rows.append(new_values)
                    last_values = new_values  # Use this year's values for next year
            
            # Append future rows to result
            if future_rows:
                future_df = pd.DataFrame(future_rows)
                result_df = pd.concat([result_df, future_df], ignore_index=True)
        
        return result_df

    # ============================================================================
    # Client Prediction Management
    # ============================================================================
    
    def load_client_prediction_lookup(self, region_name: str) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Load monthly client predictions from precomputed results.
        
        Args:
            region_name: Region name
            variable: Variable name (default: "nbr_clients")
            
        Returns:
            Dictionary mapping entity name -> {year: monthly_predictions}
        """
        variable: str = Aliases.NBR_CLIENTS
        results_path = self.project_root / f"{variable}/all_results_{clean_name(region_name)}_{variable}.pkl"
        
        if not results_path.exists():
            print(f"⚠️  No precomputed client predictions found at {results_path}.")
            return {}
        
        try:
            with results_path.open("rb") as fp:
                raw_results = pickle.load(fp)
        except Exception as exc:
            print(f"⚠️  Failed to load client predictions: {exc}")
            return {}
        
        lookup: Dict[str, Dict[int, np.ndarray]] = {}
        
        for entry in raw_results:
            entity_name = entry.get("entity")
            if entity_name is None:
                continue
            
            model_info = entry.get("best_model") or entry
            valid_years = model_info.get("valid_years")
            pred_matrix = model_info.get("pred_monthly_matrix")
            
            if valid_years is None or pred_matrix is None:
                continue
            
            entity_preds: Dict[int, np.ndarray] = {}
            for idx, year in enumerate(valid_years):
                try:
                    year_int = int(year)
                    monthly_values = np.asarray(pred_matrix[idx], dtype=float).reshape(-1)
                except Exception:
                    continue
                
                if monthly_values.size != 12:
                    continue
                
                entity_preds[year_int] = monthly_values.copy()
            
            if entity_preds:
                lookup[entity_name] = entity_preds
        
        return lookup
    
    @staticmethod
    def aggregate_predictions(
        lookup: Dict[str, Dict[int, np.ndarray]],
        target_name: str,
        component_names: List[str],
    ) -> None:
        """
        Aggregate monthly predictions from multiple components.
        
        Args:
            lookup: Dictionary of predictions (modified in-place)
            target_name: Name of the aggregated entity
            component_names: List of component entity names to aggregate
        """
        if target_name in lookup:
            return
        
        combined: Dict[int, np.ndarray] = {}
        
        for component in component_names:
            component_preds = lookup.get(component)
            if not component_preds:
                continue
            
            for year, values in component_preds.items():
                combined.setdefault(year, np.zeros(12, dtype=float))
                combined[year] = combined[year] + np.asarray(values, dtype=float)
        
        if combined:
            lookup[target_name] = combined
