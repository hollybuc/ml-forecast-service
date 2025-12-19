"""Project-level configuration schema."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .base import BaseConfig, SplitConfig, ForecastConfig
from .preprocessing import PreprocessingConfig
from .model_configs import ARIMAConfig, ProphetConfig, XGBoostConfig, LSTMConfig


class ProjectConfig(BaseModel):
    """Complete project configuration - can be saved/loaded."""

    # Metadata
    config_version: str = Field("1.0", description="Configuration schema version")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when config was created",
    )
    project_name: Optional[str] = Field(None, description="Optional project name")
    description: Optional[str] = Field(None, description="Optional project description")

    # Configurations from each step
    base_config: Optional[BaseConfig] = Field(None, description="Basic configuration")
    preprocessing_config: Optional[PreprocessingConfig] = Field(
        None, description="Preprocessing configuration"
    )
    split_config: Optional[SplitConfig] = Field(None, description="Data split configuration")
    forecast_config: Optional[ForecastConfig] = Field(
        None, description="Forecast configuration"
    )

    # Selected models and their configs
    selected_models: list[str] = Field(
        default_factory=list, description="List of selected model names"
    )

    # Model-specific configs (stored as dict)
    arima_config: Optional[ARIMAConfig] = Field(None, description="ARIMA configuration")
    prophet_config: Optional[ProphetConfig] = Field(None, description="Prophet configuration")
    xgboost_config: Optional[XGBoostConfig] = Field(None, description="XGBoost configuration")
    lstm_config: Optional[LSTMConfig] = Field(None, description="LSTM configuration")

    class Config:
        json_schema_extra = {
            "example": {
                "config_version": "1.0",
                "created_at": "2025-01-01T10:00:00",
                "project_name": "Sales Forecast",
                "description": "Daily sales forecasting project",
                "base_config": {
                    "date_column": "date",
                    "target_column": "sales",
                    "frequency": "D",
                },
                "preprocessing_config": {
                    "fill_missing_dates": True,
                    "missing_values_method": "forward_fill",
                    "outliers_method": "clip",
                },
                "split_config": {
                    "method": "temporal",
                    "train_size": 0.7,
                    "val_size": 0.15,
                    "test_size": 0.15,
                },
                "forecast_config": {"horizon": 30, "confidence_intervals": [0.95, 0.80]},
                "selected_models": ["prophet", "xgboost"],
                "prophet_config": {
                    "growth": "linear",
                    "yearly_seasonality": "auto",
                    "weekly_seasonality": "auto",
                },
                "xgboost_config": {
                    "max_depth": 6,
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                },
            }
        }
