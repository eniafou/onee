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
    VARIABLE_SPECS = {
        "nbr_clients": {
            "regional_select_expr": '"Nbr Clients" as nbr_clients',
            "regional_var_col": "nbr_clients",
            "distributor_supported": False,
            "distributor_select_expr": None,
            "distributor_var_col": None,
        },
        "consommation_kwh": {
            "regional_select_expr": 'Consommation_Kwh as consommation_kwh',
            "regional_var_col": "consommation_kwh",
            "distributor_supported": True,
            "distributor_select_expr": 'SUM(Consommation_Kwh) as consommation_kwh',
            "distributor_var_col": "consommation_kwh",
        },
    }
    
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
                Year  as annee,
                Month as mois,
                Activity as activite,
                {spec['regional_select_expr']}
            FROM GRD
            WHERE GRD = 'SRM' AND Class = 'MT' AND Region = '{target_region}'
            ORDER BY Year, Month, Activity
        """
        
        query_regional_bt = f"""
            SELECT
                Year  as annee,
                Month as mois,
                Activity as activite,
                {spec['regional_select_expr']}
            FROM GRD
            WHERE GRD = 'SRM' AND Class = 'BT' AND Region = '{target_region}'
            ORDER BY Year, Month, Activity
        """
        
        query_dist = None
        if spec["distributor_supported"]:
            query_dist = f"""
                SELECT
                    Year as annee,
                    Month as mois,
                    GRD as distributeur,
                    {spec['distributor_select_expr']}
                FROM GRD
                WHERE GRD != 'SRM' AND Region = '{target_region}'
                GROUP BY Year, Month, GRD
                ORDER BY Year, Month, GRD
            """
        
        query_features = f"""
            SELECT
                Year as annee,
                AVG(PIB_MDH) as pib_mdh,
                AVG(Primaire)    as gdp_primaire,
                AVG(Secondaire)  as gdp_secondaire,
                AVG(Tertiaire)   as gdp_tertiaire
            FROM Exogenous_Data
            WHERE Region = '{target_region}'
            GROUP BY Year
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
        df_regional_mt['activite'] = df_regional_mt['activite'].replace("Administratif", "Administratif_mt")
        df_regional = pd.concat([df_regional_bt, df_regional_mt])
        
        # Validate and clean
        reg_var_col = var_cols["regional"]
        require_columns(df_regional, ["annee", "mois", "activite", reg_var_col], "df_regional")
        df_regional[reg_var_col] = df_regional[reg_var_col].fillna(0)
        
        if df_dist is not None:
            dist_var_col = var_cols["distributor"]
            require_columns(df_dist, ["annee", "mois", "distributeur", dist_var_col], "df_dist")
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
        query_features = """
        SELECT Year as annee, 
               SUM(PIB_MDH) as pib_mdh,
               SUM(Primaire) as gdp_primaire,
               SUM(Secondaire) as gdp_secondaire,
               SUM(Tertiaire) as gdp_tertiaire
        FROM Exogenous_Data
        GROUP BY Year
        """
        df_features = pd.read_sql_query(query_features, db)
        
        # Load contracts data
        query_contrats = """
        SELECT 
        Partenaire as partenaire, 
        Numero_de_contrat as contrat, 
        year as annee, 
        month as mois, 
        Activite as activite, 
        Consommation_Kwh as consommation_kwh,
        Puissance_facturee as 'puissance facturée',
        Puissance_appelee as 'puissance appelée',
        Date_emmenagement as 'Date d''emménagement',
        Date_demenagement as 'Date de déménagement'
        FROM CD
        """
        df_contrats = pd.read_sql_query(query_contrats, db)
        
        # Optionally load activity features
        df_activite_features = None
        if include_activite_features:
            query_activite_features = """
            SELECT 
            annee as annee, 
            activite as activite,
            total_active_contrats as total_active_contrats,
            just_started as just_started,
            two_years_old as two_years_old,
            three_years_old as three_years_old,
            more_than_3_years_old as more_than_3_years_old
            from Active_Contrats_Features
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
        variable: str = "nbr_clients"
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
