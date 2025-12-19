"""Base configuration schemas."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Basic configuration - common for all models."""

    # REQUIRED
    date_column: str = Field(..., description="Name of the date/datetime column")
    target_column: str = Field(..., description="Name of the target variable column")
    frequency: Literal["H", "D", "W", "M", "Q", "Y"] = Field(
        ..., description="Time series frequency (auto-detected)"
    )

    # OPTIONAL
    group_column: Optional[str] = Field(None, description="Name of the grouping column")
    feature_columns: list[str] = Field(
        default_factory=list, description="List of feature column names"
    )
    selected_groups: list[str] = Field(
        default_factory=list, description="List of selected groups to forecast"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "date_column": "date",
                "target_column": "sales",
                "frequency": "D",
                "group_column": None,
                "feature_columns": ["temperature", "is_holiday"],
            }
        }


class SplitConfig(BaseModel):
    """Data split configuration."""

    method: Literal["temporal", "cv"] = Field(
        "temporal", description="Split method: temporal or cross-validation"
    )

    # For temporal split
    train_size: float = Field(0.7, ge=0.1, le=0.9, description="Training set size")
    val_size: float = Field(0.15, ge=0.0, le=0.5, description="Validation set size")
    test_size: float = Field(0.15, ge=0.1, le=0.5, description="Test set size")

    # For cross-validation
    n_splits: int = Field(5, ge=2, le=10, description="Number of CV splits")
    cv_type: Literal["expanding", "sliding"] = Field(
        "expanding", description="CV type: expanding or sliding window"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "method": "temporal",
                "train_size": 0.7,
                "val_size": 0.15,
                "test_size": 0.15,
            }
        }


class ForecastConfig(BaseModel):
    """Forecast configuration."""

    horizon: int = Field(30, ge=1, description="Number of steps to forecast")
    confidence_intervals: list[float] = Field(
        default_factory=lambda: [0.95, 0.80],
        description="Confidence interval levels",
    )

    class Config:
        json_schema_extra = {
            "example": {"horizon": 30, "confidence_intervals": [0.95, 0.80]}
        }
