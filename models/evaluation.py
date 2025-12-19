"""Evaluation and metrics schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    """Metrics for a single model."""

    model_name: str = Field(..., description="Name of the model")

    # Primary metrics
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    smape: float = Field(..., description="Symmetric Mean Absolute Percentage Error")

    # Secondary metrics
    r2: float = Field(..., description="R-squared (coefficient of determination)")
    medae: float = Field(..., description="Median Absolute Error")
    mase: float = Field(..., description="Mean Absolute Scaled Error")

    # Training info
    training_time_seconds: float = Field(..., description="Training time in seconds")

    # Rank (assigned after comparison)
    rank: Optional[int] = Field(None, description="Rank among all models")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "xgboost",
                "mae": 115.2,
                "rmse": 152.1,
                "mape": 7.8,
                "smape": 7.5,
                "r2": 0.91,
                "medae": 98.5,
                "mase": 0.85,
                "training_time_seconds": 12.5,
                "rank": 1,
            }
        }


class EvaluationReport(BaseModel):
    """Evaluation report for all trained models."""

    models_metrics: list[ModelMetrics] = Field(
        ..., description="Metrics for each model"
    )
    best_model: str = Field(..., description="Name of the best performing model")
    ranking_metric: str = Field(
        ..., description="Metric used for ranking (e.g., 'mape')"
    )

    # Warnings and insights
    warnings: list[str] = Field(
        default_factory=list, description="Warnings about model performance"
    )
    insights: list[str] = Field(
        default_factory=list, description="Insights about results"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "models_metrics": [
                    {
                        "model_name": "xgboost",
                        "mae": 115.2,
                        "rmse": 152.1,
                        "mape": 7.8,
                        "smape": 7.5,
                        "r2": 0.91,
                        "medae": 98.5,
                        "mase": 0.85,
                        "training_time_seconds": 12.5,
                        "rank": 1,
                    },
                    {
                        "model_name": "prophet",
                        "mae": 127.4,
                        "rmse": 168.2,
                        "mape": 8.7,
                        "smape": 8.3,
                        "r2": 0.89,
                        "medae": 105.2,
                        "mase": 0.92,
                        "training_time_seconds": 8.2,
                        "rank": 2,
                    },
                ],
                "best_model": "xgboost",
                "ranking_metric": "mape",
                "warnings": ["Large errors on weekends - consider adding features"],
                "insights": ["Residuals are normally distributed", "No autocorrelation in residuals"],
            }
        }
