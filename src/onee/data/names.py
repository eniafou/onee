"""
Database schema definitions for ONEE forecasting project.

This module centralizes all table names, column names, and SQL aliases used
throughout the codebase. Changes to database schema should only be made here.
"""

from typing import Dict


# ============================================================================
# Table Names
# ============================================================================

class Tables:
    """Database table names."""
    GRD = "GRD"
    EXOGENOUS_DATA = "Exogenous_Data"
    CD = "CD"
    ACTIVE_CONTRATS_FEATURES = "Active_Contrats_Features"


# ============================================================================
# Column Names - GRD Table
# ============================================================================

class GRDColumns:
    """Column names for GRD table."""
    YEAR = "Year"
    MONTH = "Month"
    ACTIVITY = "Activity"
    NBR_CLIENTS = '"Nbr Clients"'  # Quoted because of space
    CONSOMMATION_KWH = "Consommation_Kwh"
    GRD = "GRD"
    CLASS = "Class"
    REGION = "Region"


# ============================================================================
# Column Names - Exogenous Data Table
# ============================================================================

class ExogenousColumns:
    """Column names for Exogenous_Data table."""
    YEAR = "Year"
    REGION = "Region"
    PIB_MDH = "PIB_MDH"
    PRIMAIRE = "Primaire"
    SECONDAIRE = "Secondaire"
    TERTIAIRE = "Tertiaire"


# ============================================================================
# Column Names - CD (Contracts) Table
# ============================================================================

class CDColumns:
    """Column names for CD table."""
    PARTENAIRE = "Partenaire"
    NUMERO_DE_CONTRAT = "Numero_de_contrat"
    YEAR = "year"
    MONTH = "month"
    ACTIVITE = "Activite"
    CONSOMMATION_KWH = "Consommation_Kwh"
    PUISSANCE_FACTUREE = "Puissance_facturee"
    PUISSANCE_APPELEE = "Puissance_appelee"
    DATE_EMMENAGEMENT = "Date_emmenagement"
    DATE_DEMENAGEMENT = "Date_demenagement"


# ============================================================================
# Column Names - Active Contrats Features Table
# ============================================================================

class ActiveContratsColumns:
    """Column names for Active_Contrats_Features table."""
    ANNEE = "annee"
    ACTIVITE = "activite"
    TOTAL_ACTIVE_CONTRATS = "total_active_contrats"
    JUST_STARTED = "just_started"
    TWO_YEARS_OLD = "two_years_old"
    THREE_YEARS_OLD = "three_years_old"
    MORE_THAN_3_YEARS_OLD = "more_than_3_years_old"


# ============================================================================
# SQL Query Aliases (used throughout codebase)
# ============================================================================

class Aliases:
    """Standard aliases used in SQL queries and DataFrame columns."""
    # Time columns
    ANNEE = "annee"
    MOIS = "mois"
    
    # Entity identifiers
    ACTIVITE = "activite"
    DISTRIBUTEUR = "distributeur"
    PARTENAIRE = "partenaire"
    CONTRAT = "contrat"
    
    # Metrics
    NBR_CLIENTS = "nbr_clients"
    CONSOMMATION_KWH = "consommation_kwh"
    PUISSANCE_FACTUREE = "puissance facturée"
    PUISSANCE_APPELEE = "puissance appelée"
    
    # Economic indicators
    PIB_MDH = "pib_mdh"
    GDP_PRIMAIRE = "gdp_primaire"
    GDP_SECONDAIRE = "gdp_secondaire"
    GDP_TERTIAIRE = "gdp_tertiaire"
    
    # Date fields
    DATE_EMMENAGEMENT = "Date_emmenagement"
    DATE_DEMENAGEMENT = "Date_demenagement"
    
    # Activity features
    TOTAL_ACTIVE_CONTRATS = "total_active_contrats"
    JUST_STARTED = "just_started"
    TWO_YEARS_OLD = "two_years_old"
    THREE_YEARS_OLD = "three_years_old"
    MORE_THAN_3_YEARS_OLD = "more_than_3_years_old"


# ============================================================================
# GRD Values (Classification constants)
# ============================================================================

class GRDValues:
    """Standard values for GRD-related classifications."""
    GRD_SRM = "SRM"
    CLASS_MT = "MT"
    CLASS_BT = "BT"
    ACTIVITY_ADMINISTRATIF = "Administratif"
    ACTIVITY_ADMINISTRATIF_MT = "Administratif_mt"
    ACTIVITY_AGRICOLE = "Agricole"
    ACTIVITY_INDUSTRIEL = "Industriel"
    ACTIVITY_RESIDENTIEL = "Résidentiel"
    ACTIVITY_TERTIAIRE = "Tertiaire"


# ============================================================================
# Helper Functions
# ============================================================================

def build_variable_specs() -> Dict[str, Dict[str, any]]:
    """
    Build variable specifications using centralized names.
    
    Returns:
        Dictionary mapping variable names to their specifications
    """
    return {
        Aliases.NBR_CLIENTS: {
            "regional_select_expr": f"{GRDColumns.NBR_CLIENTS} as {Aliases.NBR_CLIENTS}",
            "regional_var_col": Aliases.NBR_CLIENTS,
            "distributor_supported": False,
            "distributor_select_expr": None,
            "distributor_var_col": None,
        },
        Aliases.CONSOMMATION_KWH: {
            "regional_select_expr": f"{GRDColumns.CONSOMMATION_KWH} as {Aliases.CONSOMMATION_KWH}",
            "regional_var_col": Aliases.CONSOMMATION_KWH,
            "distributor_supported": True,
            "distributor_select_expr": f"SUM({GRDColumns.CONSOMMATION_KWH}) as {Aliases.CONSOMMATION_KWH}",
            "distributor_var_col": Aliases.CONSOMMATION_KWH,
        },
    }
