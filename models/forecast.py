"""Forecast result schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class ForecastResult(BaseModel):
    """Forecast result for a single model."""

    model_name: str = Field(..., description="Name of the model used for forecast")
    dates: list[str] = Field(..., description="List of forecast dates (ISO format)")
    values: list[float] = Field(..., description="Forecasted values")

    # Confidence intervals
    lower_95: Optional[list[float]] = Field(
        None, description="Lower bound of 95% confidence interval"
    )
    upper_95: Optional[list[float]] = Field(
        None, description="Upper bound of 95% confidence interval"
    )
    lower_80: Optional[list[float]] = Field(
        None, description="Lower bound of 80% confidence interval"
    )
    upper_80: Optional[list[float]] = Field(
        None, description="Upper bound of 80% confidence interval"
    )
    
    # Metadata for grouped models
    metadata: dict = Field(
        default_factory=dict, description="Additional metadata (e.g., group information)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "xgboost",
                "dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "values": [1420.5, 1385.2, 1450.8],
                "lower_95": [1180.0, 1130.0, 1190.0],
                "upper_95": [1660.0, 1640.0, 1710.0],
                "lower_80": [1280.0, 1240.0, 1300.0],
                "upper_80": [1560.0, 1530.0, 1600.0],
            }
        }


class ForecastRequest(BaseModel):
    """Request for generating forecast."""

    model_selection: str = Field(
        ...,
        description="Which model(s) to use: 'best', 'all', 'ensemble', or specific model name",
    )
    horizon: int = Field(..., ge=1, description="Number of steps to forecast")
    confidence_intervals: list[float] = Field(
        default_factory=lambda: [0.95, 0.80],
        description="Confidence interval levels",
    )

    # Future regressors (if needed)
    future_regressors: Optional[dict] = Field(
        None, description="Future values for regressors (for Prophet, XGBoost with features)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_selection": "best",
                "horizon": 30,
                "confidence_intervals": [0.95, 0.80],
                "future_regressors": {
                    "temperature": [15.2, 16.1, 14.8],
                    "is_promo": [0, 1, 0],
                },
            }
        }
