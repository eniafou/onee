"""
Configuration Schema for short term Forecast System
Organized by functional groups for clarity
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
import yaml
from onee.data.names import Aliases


# ═══════════════════════════════════════════════════════════════════════
# PROJECT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class ProjectConfig(BaseModel):
    """Project paths and experiment identification"""
    project_root: Path
    exp_name: str
    
    @field_validator('project_root', mode='before')
    @classmethod
    def resolve_project_root(cls, v):
        if isinstance(v, str):
            return Path(v).resolve()
        return v


# ═══════════════════════════════════════════════════════════════════════
# DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class DataConfig(BaseModel):
    """Data sources, target variables, and processing"""
    variable: str = Aliases.CONSOMMATION_KWH
    unit: str = "Kwh"
    
    # For SRM: list of regions; for CD: None
    regions: Optional[List[str]] = None
    
    # Run levels determine which analysis parts to execute
    run_levels: List[int]
    
    # Database paths (relative to project_root)
    db_path: str = "data/all_data.db"


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class EvaluationConfig(BaseModel):
    """Evaluation period and training configuration"""
    eval_years_start: int = 2021
    eval_years_end: int = 2023
    train_start_year: Optional[int] = None
    training_end: Optional[int] = None  # If set, limits training to this year


# ═══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class FeatureConfig(BaseModel):
    """Feature engineering parameters"""
    feature_blocks: Dict[str, List[str]] = Field(
        default={
            'none': [],
            'gdp_only': [Aliases.PIB_MDH],
            'sectoral_only': [Aliases.GDP_PRIMAIRE, Aliases.GDP_SECONDAIRE, Aliases.GDP_TERTIAIRE],
            'gdp_sectoral': [Aliases.PIB_MDH, Aliases.GDP_PRIMAIRE, Aliases.GDP_SECONDAIRE, Aliases.GDP_TERTIAIRE],
        }
    )
    
    # Growth rate feature configuration
    growth_feature_transforms: List[Tuple[str, ...]] = Field(
        default=[("lag_lchg",)]
    )
    growth_feature_lags: List[Tuple[int, ...]] = Field(
        default=[(1,)]
    )
    
    # Monthly data usage options
    use_monthly_temp_options: List[bool] = Field(default=[False])
    use_monthly_clients_options: List[bool] = Field(default=[False])
    use_pf_options: List[bool] = Field(default=[True, False])
    
    @field_validator('growth_feature_transforms', 'growth_feature_lags', mode='before')
    @classmethod
    def convert_to_list_of_tuples(cls, v):
        """Convert lists to list of tuples"""
        if isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], list):
                return [tuple(item) for item in v]
        return v


# ═══════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════

class ModelHyperparametersConfig(BaseModel):
    """models hyperparameters"""
    # PCA configuration
    n_pcs: int = 3
    lags_options: List[int] = Field(default=[1, 2])
    alphas: List[float] = Field(default=[0.01, 0.1, 1.0, 10.0])
    pc_weights: List[float] = Field(default=[0.5, 0.8])
    pca_lambdas: List[float] = Field(default=[0.2, 0.3, 0.4])
    
    # Training configuration
    training_windows: List[int] = Field(default=[1, 2, 3, 4])
    
    # Pattern weights
    client_pattern_weights: List[float] = Field(default=[0.3, 0.5, 0.8])
    
    # Model selection
    r2_threshold: float = 0.6


# ═══════════════════════════════════════════════════════════════════════
# LOSS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class LossConfig(BaseModel):
    """Asymmetric loss configuration for different entity types"""
    favor_overestimation: bool = False
    under_estimation_penalty: float = 1.5
    
    # Entity-specific penalties (optional overrides)
    penalties_by_level: Optional[Dict[int, float]] = None


# ═══════════════════════════════════════════════════════════════════════
# ROOT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class ShortTermForecastConfig(BaseModel):
    """Root configuration for short term forecasting"""
    project: ProjectConfig
    data: DataConfig
    evaluation: EvaluationConfig
    features: FeatureConfig
    model: ModelHyperparametersConfig
    loss: LossConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ShortTermForecastConfig":
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str | Path):
        """Save configuration to YAML file"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            config_dict = self.model_dump(mode='python')
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def to_analysis_config(self) -> dict:
        """
        Convert to the legacy ANALYSIS_CONFIG format for backward compatibility
        """
        return {
            "value_col": self.data.variable,
            "N_PCS": self.model.n_pcs,
            "LAGS_OPTIONS": self.model.lags_options,
            "FEATURE_BLOCKS": self.features.feature_blocks,
            "ALPHAS": self.model.alphas,
            "PC_WEIGHTS": self.model.pc_weights,
            "R2_THRESHOLD": self.model.r2_threshold,
            "unit": self.data.unit,
            "PCA_LAMBDAS": self.model.pca_lambdas,
            "training_end": self.evaluation.training_end,
            "use_monthly_temp_options": self.features.use_monthly_temp_options,
            "use_monthly_clients_options": self.features.use_monthly_clients_options,
            "use_pf_options": self.features.use_pf_options,
            "client_pattern_weights": self.model.client_pattern_weights,
            "training_windows": self.model.training_windows,
            "train_start_year": self.evaluation.train_start_year,
            "eval_years_end": self.evaluation.eval_years_end,
            "eval_years_start": self.evaluation.eval_years_start,
            "growth_feature_transforms": self.features.growth_feature_transforms,
            "growth_feature_lags": self.features.growth_feature_lags,
        }
