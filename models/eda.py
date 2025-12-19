"""EDA (Exploratory Data Analysis) schemas."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class EDAReport(BaseModel):
    """Exploratory data analysis report."""

    # Date range
    date_min: str = Field(..., description="Minimum date in dataset")
    date_max: str = Field(..., description="Maximum date in dataset")
    n_observations: int = Field(..., description="Total number of observations")
    detected_frequency: str = Field(..., description="Detected time series frequency")

    # Missing data
    missing_dates: list[str] = Field(
        default_factory=list, description="List of missing dates"
    )
    missing_dates_pct: float = Field(..., description="Percentage of missing dates")
    missing_values: int = Field(..., description="Number of missing target values")
    missing_values_pct: float = Field(
        ..., description="Percentage of missing target values"
    )

    # Target statistics
    target_stats: dict = Field(..., description="Target variable statistics")
    zeros_count: int = Field(..., description="Number of zero values")
    negative_count: int = Field(..., description="Number of negative values")

    # Patterns
    trend: Literal["increasing", "decreasing", "stable", "unknown"] = Field(
        ..., description="Detected trend direction"
    )
    seasonality_detected: list[str] = Field(
        default_factory=list, description="Detected seasonality patterns"
    )
    is_stationary: bool = Field(..., description="Stationarity test result (ADF)")
    adf_pvalue: float = Field(..., description="ADF test p-value")

    # Outliers
    outliers_count: int = Field(..., description="Number of outliers detected")
    outliers_pct: float = Field(..., description="Percentage of outliers")
    outliers_indices: list[int] = Field(
        default_factory=list, description="Indices of outliers"
    )

    # Groups (if group_column exists)
    groups: Optional[list[str]] = Field(None, description="List of unique groups")
    groups_count: Optional[int] = Field(None, description="Number of groups")
    groups_stats: Optional[dict] = Field(None, description="Per-group statistics (target_stats, outliers, missing_data)")

    # Alerts and recommendations
    alerts: list[str] = Field(default_factory=list, description="Data quality alerts")
    recommendations: list[str] = Field(
        default_factory=list, description="Model recommendations"
    )

    # Chart data (optional - for visualization)
    chart_data: Optional[dict] = Field(
        None,
        description="Data for charts (time series, distribution, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "date_min": "2020-01-01",
                "date_max": "2023-12-31",
                "n_observations": 1461,
                "detected_frequency": "D",
                "missing_dates": [],
                "missing_dates_pct": 0.0,
                "missing_values": 5,
                "missing_values_pct": 0.34,
                "target_stats": {
                    "mean": 1250.5,
                    "median": 1200.0,
                    "std": 350.2,
                    "min": 100.0,
                    "max": 3500.0,
                },
                "zeros_count": 0,
                "negative_count": 0,
                "trend": "increasing",
                "seasonality_detected": ["weekly", "yearly"],
                "is_stationary": False,
                "adf_pvalue": 0.15,
                "outliers_count": 12,
                "outliers_pct": 0.82,
                "outliers_indices": [45, 123, 456],
                "alerts": ["5% missing values detected"],
                "recommendations": ["Consider SARIMA for seasonal data"],
            }
        }
