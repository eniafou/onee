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
    WEATHER = "Weather"

    STF_SRM_RESULTS = "STF_SRM_Results"
    STF_CD_RESULTS = "STF_CD_Results"
    LTF_SRM_RESULTS = "LTF_SRM_Results"
    LTF_CD_RESULTS = "LTF_CD_Results"


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
# Column Names - Weather Table
# ============================================================================

class WeatherColumns:
    """Column names for Weather table."""
    REGION = "Region"
    YEAR = "Year"
    MONTH = "Month"
    APPARENT_TEMPERATURE_MIN = "apparent_temperature_min"
    PRECIPITATION_SUM = "precipitation_sum"
    ET0_FAO_EVAPOTRANSPIRATION_SUM = "et0_fao_evapotranspiration_sum"


# ============================================================================
# Column Names - CD (Contracts) Table
# ============================================================================
class CDColumns:
    """Column names for CD table."""
    REGION = "Region"
    PARTENAIRE = "Partenaire"
    NUMERO_DE_CONTRAT = "Numero_de_contrat"
    YEAR = "year"
    MONTH = "month"
    ACTIVITE = "Activite"
    CONSOMMATION_KWH = "Consommation_Kwh"
    CONSOMMATION_ZCONHC = "Consommation_ZCONHC"
    CONSOMMATION_ZCONHL = "Consommation_ZCONHL"
    CONSOMMATION_ZCONHP = "Consommation_ZCONHP"
    PUISSANCE_FACTUREE = "Puissance_facturee"
    PUISSANCE_APPELEE = "Puissance_appelee"
    DATE_EMMENAGEMENT = "Date_emmenagement"
    DATE_DEMENAGEMENT = "Date_demenagement"
    NIVEAU_TENSION = "Niveau_de_tension"

# ============================================================================
# Column Names - STF SRM Results Table
# ============================================================================
class STFSRMResultsColumns:
    """Column names for STF_SRM_Results output table."""
    REGION = "Region"
    ACTIVITY = "Activity"
    YEAR = "Year"
    MONTH = "Month"
    CONSOMMATION = "Consommation"


# ============================================================================
# Column Names - STF CD Results Table
# ============================================================================
class STFCDResultsColumns:
    """Column names for STF_CD_Results output table."""
    REGION = "Region"
    PARTENAIRE = "Partenaire"
    CONTRAT = "Contrat"
    ACTIVITY = "Activity"
    YEAR = "Year"
    MONTH = "Month"
    CONSOMMATION_TOTAL = "Consommation_Total"
    CONSOMMATION_ZCONHC = "Consommation_ZCONHC"
    CONSOMMATION_ZCONHL = "Consommation_ZCONHL"
    CONSOMMATION_ZCONHP = "Consommation_ZCONHP"
    PUISSANCE_FACTUREE = "Puissance_Facturee"
    NIVEAU_TENSION = "Niveau_tension"


# ============================================================================
# Column Names - LTF SRM Results Table
# ============================================================================
class LTFSRMResultsColumns:
    """Column names for LTF_SRM_Results output table."""
    REGION = "Region"
    YEAR = "Year"
    CONSOMMATION = "Consommation"


# ============================================================================
# Column Names - LTF CD Results Table
# ============================================================================
class LTFCDResultsColumns:
    """Column names for LTF_CD_Results output table."""
    REGION = "Region"
    PARTENAIRE = "Partenaire"
    CONTRAT = "Contrat"
    ACTIVITY = "Activity"
    YEAR = "Year"
    CONSOMMATION_TOTAL = "Consommation_Total"
    CONSOMMATION_ZCONHC = "Consommation_ZCONHC"
    CONSOMMATION_ZCONHL = "Consommation_ZCONHL"
    CONSOMMATION_ZCONHP = "Consommation_ZCONHP"
    PUISSANCE_FACTUREE = "Puissance_Facturee"
    NIVEAU_TENSION = "Niveau_tension"


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
    REGION = "region"
    
    # Metrics
    NBR_CLIENTS = "nbr_clients"
    CONSOMMATION_KWH = "consommation_kwh"
    PUISSANCE_FACTUREE = "puissance_facturee"
    PUISSANCE_APPELEE = "puissance_appelee"
    NIVEAU_TENSION = "niveau_tension"
    CONSOMMATION_ZCONHC = "consommation_zconhc"
    CONSOMMATION_ZCONHL = "consommation_zconhl"
    CONSOMMATION_ZCONHP = "consommation_zconhp"
    
    # Economic indicators
    PIB_MDH = "pib_mdh"
    GDP_PRIMAIRE = "gdp_primaire"
    GDP_SECONDAIRE = "gdp_secondaire"
    GDP_TERTIAIRE = "gdp_tertiaire"
    
    # Date fields
    DATE_EMMENAGEMENT = "date_emmenagement"
    DATE_DEMENAGEMENT = "date_demenagement"
    
    # Weather sensitivity coefficients
    COEF_TEMP_MIN = "coef_temp_min"
    COEF_PRECIPITATION = "coef_precipitation"
    COEF_ET0 = "coef_et0"
    PCT_CONTRIBUTION = "pct_contribution"
    IMPLICIT_INTERCEPT = "implicit_intercept"
    
    # Weather variables
    TEMP_MIN_DIFF = "temp_min_diff"
    PRECIPITATION_SUM_DIFF = "precipitation_sum_diff"
    ET0_DIFF = "et0_diff"
    TEMP_MIN_LATEST = "temp_min_latest"
    PRECIPITATION_SUM_LATEST = "precipitation_sum_latest"
    ET0_LATEST = "et0_latest"
    TEMP_MIN_CORRECTION = "temp_min_correction"
    PRECIPITATION_SUM_CORRECTION = "precipitation_sum_correction"
    ET0_CORRECTION = "et0_correction"
    
    # Activity features
    TOTAL_ACTIVE_CONTRATS = "total_active_contrats"
    JUST_STARTED = "just_started"
    TWO_YEARS_OLD = "two_years_old"
    THREE_YEARS_OLD = "three_years_old"
    MORE_THAN_3_YEARS_OLD = "more_than_3_years_old"
    PUISSANCE_FACTUREE_TOTAL = "puissance_facturee_total"
    PUISSANCE_FACTUREE_JUST_STARTED = "puissance_facturee_just_started"
    PUISSANCE_FACTUREE_TWO_YEARS_OLD = "puissance_facturee_two_years_old"
    PUISSANCE_FACTUREE_THREE_YEARS_OLD = "puissance_facturee_three_years_old"
    PUISSANCE_FACTUREE_MORE_THAN_3_YEARS_OLD = "puissance_facturee_more_than_3_years_old"
    PUISSANCE_APPELEE_TOTAL = "puissance_appelee_total"
    PUISSANCE_APPELEE_JUST_STARTED = "puissance_appelee_just_started"
    PUISSANCE_APPELEE_TWO_YEARS_OLD = "puissance_appelee_two_years_old"
    PUISSANCE_APPELEE_THREE_YEARS_OLD = "puissance_appelee_three_years_old"
    PUISSANCE_APPELEE_MORE_THAN_3_YEARS_OLD = "puissance_appelee_more_than_3_years_old"


# ============================================================================
# GRD Values (Classification constants)
# ============================================================================

class GRDValues:
    """Standard values for GRD-related classifications."""
    GRD_SRM = "SRM"
    CLASS_MT = "MT"
    CLASS_BT = "BT"
    CLASS_TOTAL = "Total"
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
