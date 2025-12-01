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
    Tables, GRDColumns, ExogenousColumns, CDColumns, ActiveContratsColumns,
    Aliases, GRDValues, build_variable_specs
)


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
        include_activite_features: bool = False,
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
        {CDColumns.PARTENAIRE} as {Aliases.PARTENAIRE}, 
        {CDColumns.NUMERO_DE_CONTRAT} as {Aliases.CONTRAT}, 
        {CDColumns.YEAR} as {Aliases.ANNEE}, 
        {CDColumns.MONTH} as {Aliases.MOIS}, 
        {CDColumns.ACTIVITE} as {Aliases.ACTIVITE}, 
        {CDColumns.CONSOMMATION_KWH} as {Aliases.CONSOMMATION_KWH},
        {CDColumns.PUISSANCE_FACTUREE} as '{Aliases.PUISSANCE_FACTUREE}',
        {CDColumns.PUISSANCE_APPELEE} as '{Aliases.PUISSANCE_APPELEE}',
        {CDColumns.DATE_EMMENAGEMENT} as '{Aliases.DATE_EMMENAGEMENT}',
        {CDColumns.DATE_DEMENAGEMENT} as '{Aliases.DATE_DEMENAGEMENT}'
        FROM {Tables.CD}
        """
        df_contrats = pd.read_sql_query(query_contrats, db)
        
        # Optionally load activity features
        df_activite_features = None
        if include_activite_features:
            query_activite_features = f"""
            SELECT 
            {ActiveContratsColumns.ANNEE} as {Aliases.ANNEE}, 
            {ActiveContratsColumns.ACTIVITE} as {Aliases.ACTIVITE},
            {ActiveContratsColumns.TOTAL_ACTIVE_CONTRATS} as {Aliases.TOTAL_ACTIVE_CONTRATS},
            {ActiveContratsColumns.JUST_STARTED} as {Aliases.JUST_STARTED},
            {ActiveContratsColumns.TWO_YEARS_OLD} as {Aliases.TWO_YEARS_OLD},
            {ActiveContratsColumns.THREE_YEARS_OLD} as {Aliases.THREE_YEARS_OLD},
            {ActiveContratsColumns.MORE_THAN_3_YEARS_OLD} as {Aliases.MORE_THAN_3_YEARS_OLD},
            {ActiveContratsColumns.PUISSANCE_FACTUREE_TOTAL} as {Aliases.PUISSANCE_FACTUREE_TOTAL},
            {ActiveContratsColumns.PUISSANCE_FACTUREE_JUST_STARTED} as {Aliases.PUISSANCE_FACTUREE_JUST_STARTED},
            {ActiveContratsColumns.PUISSANCE_FACTUREE_TWO_YEARS_OLD} as {Aliases.PUISSANCE_FACTUREE_TWO_YEARS_OLD},
            {ActiveContratsColumns.PUISSANCE_FACTUREE_THREE_YEARS_OLD} as {Aliases.PUISSANCE_FACTUREE_THREE_YEARS_OLD},
            {ActiveContratsColumns.PUISSANCE_FACTUREE_MORE_THAN_3_YEARS_OLD} as {Aliases.PUISSANCE_FACTUREE_MORE_THAN_3_YEARS_OLD}
            FROM {Tables.ACTIVE_CONTRATS_FEATURES}
            """
            df_activite_features = pd.read_sql_query(query_activite_features, db)
        
        db.close()
        
        return df_contrats, df_features, df_activite_features
    
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
