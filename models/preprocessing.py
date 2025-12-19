"""Preprocessing configuration schemas."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    # Missing dates handling
    fill_missing_dates: bool = Field(
        True, description="Fill missing dates in the time series"
    )

    # Missing values handling
    missing_values_method: Literal[
        "forward_fill", "backward_fill", "interpolate", "mean", "median", "drop"
    ] = Field("forward_fill", description="Method for handling missing values")

    # Outliers handling
    outliers_method: Literal["clip", "remove", "winsorize", "none"] = Field(
        "none", description="Method for handling outliers"
    )
    outliers_threshold: float = Field(
        3.0, ge=1.0, le=10.0, description="IQR threshold for outlier detection"
    )

    # Transformations
    transformation: Literal["none", "log", "sqrt", "box-cox"] = Field(
        "none", description="Transformation to apply to target variable"
    )

    # Feature engineering
    create_date_features: bool = Field(
        True, description="Create date-based features (year, month, day, etc.)"
    )
    create_lag_features: bool = Field(
        False, description="Create lag features (for ML models)"
    )
    lag_periods: list[int] = Field(
        default_factory=lambda: [1, 7, 30], description="Lag periods to create"
    )
    create_rolling_features: bool = Field(
        False, description="Create rolling window features"
    )
    rolling_windows: list[int] = Field(
        default_factory=lambda: [7, 14, 30], description="Rolling window sizes"
    )
    create_cyclical_features: bool = Field(
        True, description="Create cyclical features (sin/cos)"
    )

    # Holiday features
    add_holidays: bool = Field(True, description="Add holiday features")
    holiday_country: str = Field("PL", description="Country code for holidays")

    class Config:
        json_schema_extra = {
            "example": {
                "fill_missing_dates": True,
                "missing_values_method": "forward_fill",
                "outliers_method": "clip",
                "outliers_threshold": 3.0,
                "transformation": "log",
                "create_date_features": True,
                "create_lag_features": False,
                "lag_periods": [1, 7, 30],
                "create_rolling_features": False,
                "rolling_windows": [7, 14, 30],
                "create_cyclical_features": True,
                "add_holidays": True,
                "holiday_country": "PL",
            }
        }
