"""
Configuration Schema for Long-Term Forecast System
Five-Pillar Architecture: Project, Data, Temporal, Features, Model
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal, Any, Dict, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
from onee.data.names import Aliases


# ═══════════════════════════════════════════════════════════════════════
# PILLAR 1: PROJECT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class ProjectConfig(BaseModel):
    """Meta-data, paths, experiment identification"""
    project_root: Path
    exp_name: str
    output_base_dir: str = "outputs_horizon"  # Base directory name
    
    @field_validator('project_root', mode='before')
    @classmethod
    def resolve_project_root(cls, v):
        if isinstance(v, str):
            return Path(v).resolve()
        return v


# ═══════════════════════════════════════════════════════════════════════
# PILLAR 2: DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class DataConfig(BaseModel):
    """Data sources, regions, run levels, target variables"""
    target_variable: str = Aliases.CONSOMMATION_KWH
    unit: str = "Kwh"
    regions: Optional[List[str]] = None  # For SRM; None for CD
    run_levels: List[int]  # Which analysis levels to execute
    impute_2020: bool = False
    
    # Database paths (relative to project_root or absolute)
    db_path: str = "data/all_data.db"


# ═══════════════════════════════════════════════════════════════════════
# PILLAR 3: TEMPORAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class TemporalConfig(BaseModel):
    """Forecast horizons, training windows"""
    horizon: int = 5  # Number of years to forecast
    forecast_runs: List[Tuple[int, int]]  # List of (train_start, train_end) pairs
    
    @field_validator('forecast_runs', mode='before')
    @classmethod
    def convert_forecast_runs(cls, v):
        """Convert list of lists to list of tuples"""
        if isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], list):
                return [tuple(run) for run in v]
        return v


# ═══════════════════════════════════════════════════════════════════════
# PILLAR 4: FEATURES CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class FeaturesConfig(BaseModel):
    """Feature engineering: lags, transforms, toggles"""
    transforms: List[List[str]] = Field(
        default=[["lchg"], ["lchg", "lag_lchg"]],
        description="Feature transform combinations"
    )
    lags: List[List[int]] = Field(
        default=[[1, 2]],
        description="Lag configurations"
    )
    feature_block: List[List[str]] = Field(
        default=[
            [],
            [Aliases.PIB_MDH],
            [Aliases.GDP_PRIMAIRE, Aliases.GDP_SECONDAIRE, Aliases.GDP_TERTIAIRE],
            [Aliases.PIB_MDH, Aliases.GDP_PRIMAIRE, Aliases.GDP_SECONDAIRE, Aliases.GDP_TERTIAIRE],
        ],
        description="Exogenous feature combinations to try",
    )
    use_pf: List[bool] = Field(default=[False], description="Use power factor")
    use_clients: List[bool] = Field(default=[True], description="Use client counts")
    training_window: List[Optional[int]] = Field(
        default=[None],
        description="Rolling training window (None = use all history)"
    )
    
    @field_validator('transforms', 'lags', mode='before')
    @classmethod
    def convert_to_list_of_lists(cls, v):
        """Ensure nested lists for tuple-like structures"""
        if isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], tuple):
                return [list(item) for item in v]
        return v


# ═══════════════════════════════════════════════════════════════════════
# PILLAR 5: MODEL CONFIGURATION (with Discriminated Union)
# ═══════════════════════════════════════════════════════════════════════

class BaseModelConfig(BaseModel):
    """Shared parameters across all model types"""
    model_type: str  # Discriminator field



class GaussianProcessForecastModelConfig(BaseModelConfig):
    """Configuration for Gaussian Process Forecast Model"""
    model_type: Literal["GaussianProcessForecastModel"] = "GaussianProcessForecastModel"
    
    # GP-specific parameters
    kernel_key: List[Optional[str]] = Field(
        default=[None],
        description="Kernel key from KERNEL_REGISTRY. Options: 'rbf_white', 'matern_white', 'rbf_dot_white', 'matern_smooth', 'rbf_long', or None for default."
    )
    n_restarts_optimizer: List[int] = Field(default=[10])
    normalize_y: List[bool] = Field(default=[True, False])
    use_log_transform: List[bool] = Field(default=[True])
    alpha: List[float] = Field(default=[1e-10])
    
    # Outlier detection
    remove_outliers: List[bool] = Field(
        default=[False],
        description="Whether to detect and downweight outliers during GP fitting."
    )
    outlier_threshold: List[float] = Field(
        default=[2.5],
        description="Z-score threshold for outlier detection (using MAD)."
    )
    
    # Prior configuration - list of prior config dicts to try
    prior_config: List[Dict[str, Any]] = Field(
        default=[{"type": "PowerGrowthPrior", "power": 0.5, "anchor_window": 3, "min_annual_growth": 0.02}],
        description="List of prior configurations. Each dict should have 'type' and corresponding params."
    )


class IntensityForecastWrapperConfig(BaseModelConfig):
    """Configuration for Intensity Forecast Wrapper (CD-specific)"""
    model_type: Literal["IntensityForecastWrapper"] = "IntensityForecastWrapper"
    
    # Normalization
    normalization_col: List[str] = Field(default=[Aliases.TOTAL_ACTIVE_CONTRATS], description="Column for normalization")
    
    # GP parameters for internal model
    kernel_key: List[Optional[str]] = Field(
        default=[None],
        description="Kernel key from KERNEL_REGISTRY. Options: 'rbf_white', 'matern_white', 'rbf_dot_white', 'matern_smooth', 'rbf_long', or None for default."
    )
    n_restarts_optimizer: List[int] = Field(default=[10])
    normalize_y: List[bool] = Field(default=[True, False])
    use_log_transform: List[bool] = Field(default=[False])
    alpha: List[float] = Field(default=[1e-10])
    
    # Outlier detection
    remove_outliers: List[bool] = Field(
        default=[False],
        description="Whether to detect and downweight outliers during GP fitting."
    )
    outlier_threshold: List[float] = Field(
        default=[2.5],
        description="Z-score threshold for outlier detection (using MAD)."
    )
    
    # Prior configuration for internal GP model - list of prior config dicts
    prior_config: List[Dict[str, Any]] = Field(
        default=[{"type": "PowerGrowthPrior", "power": 0.5, "anchor_window": 3, "min_annual_growth": 0.02}],
        description="List of prior configurations for internal model. Each dict should have 'type' and corresponding params."
    )


# Discriminated Union for Model Configs
ModelConfigUnion = Annotated[
    Union[
        GaussianProcessForecastModelConfig,
        IntensityForecastWrapperConfig
    ],
    Field(discriminator='model_type')
]


class ModelsConfig(BaseModel):
    """Container for multiple model configurations"""
    models: List[ModelConfigUnion]
    r2_threshold: float = Field(default=0.6, description="R² threshold for model selection")


# ═══════════════════════════════════════════════════════════════════════
# ROOT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class LongTermForecastConfig(BaseModel):
    """Root configuration combining all 5 pillars for Long-Term forecasting"""
    project: ProjectConfig
    data: DataConfig
    temporal: TemporalConfig
    features: FeaturesConfig
    models: ModelsConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "LongTermForecastConfig":
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
            # Convert to dict, handling Path objects
            config_dict = self.model_dump(mode='python')
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def get_model_registry(self) -> Dict[str, type]:
        """Build model registry from model classes"""
        # This will be populated at runtime by importing actual model classes
        from onee.growth_rate_model import (
            GaussianProcessForecastModel,
            IntensityForecastWrapper
        )
        
        registry = {
            "GaussianProcessForecastModel": GaussianProcessForecastModel,
            "IntensityForecastWrapper": IntensityForecastWrapper,
        }
        
        return registry
    
    def get_output_dir(self, region: Optional[str] = None) -> Path:
        """Compute output directory path"""
        from onee.utils import clean_name
        
        base = self.project.project_root / self.project.output_base_dir / self.project.exp_name
        if region:
            return base / clean_name(region)
        return base
