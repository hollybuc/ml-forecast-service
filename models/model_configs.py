"""Model-specific configuration schemas."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class ARIMAConfig(BaseModel):
    """ARIMA/SARIMA model configuration."""

    # Auto or manual
    auto: bool = Field(True, description="Auto-tune ARIMA parameters")

    # Manual parameters (if auto=False)
    p: int = Field(1, ge=0, le=5, description="AR order")
    d: int = Field(1, ge=0, le=2, description="Differencing order")
    q: int = Field(1, ge=0, le=5, description="MA order")

    # Seasonal parameters
    seasonal: bool = Field(True, description="Use SARIMA (seasonal)")
    seasonal_period: Optional[int] = Field(
        None, ge=2, description="Seasonal period (auto-detected if None)"
    )
    P: int = Field(1, ge=0, le=2, description="Seasonal AR order")
    D: int = Field(1, ge=0, le=1, description="Seasonal differencing order")
    Q: int = Field(1, ge=0, le=2, description="Seasonal MA order")

    class Config:
        json_schema_extra = {
            "example": {
                "auto": True,
                "p": 1,
                "d": 1,
                "q": 1,
                "seasonal": True,
                "seasonal_period": 7,
                "P": 1,
                "D": 1,
                "Q": 1,
            }
        }


class ProphetConfig(BaseModel):
    """Prophet model configuration."""

    # Growth
    growth: Literal["linear", "logistic"] = Field(
        "linear", description="Growth trend type"
    )
    cap: Optional[float] = Field(None, description="Carrying capacity (for logistic)")
    floor: Optional[float] = Field(None, description="Floor value (for logistic)")

    # Seasonality
    yearly_seasonality: Literal["auto", True, False] = Field(
        "auto", description="Yearly seasonality"
    )
    weekly_seasonality: Literal["auto", True, False] = Field(
        "auto", description="Weekly seasonality"
    )
    daily_seasonality: Literal["auto", True, False] = Field(
        "auto", description="Daily seasonality"
    )

    # Changepoints
    changepoint_prior_scale: float = Field(
        0.05, ge=0.001, le=0.5, description="Changepoint flexibility"
    )
    seasonality_prior_scale: float = Field(
        10.0, ge=0.01, le=100.0, description="Seasonality strength"
    )

    # Regressors
    add_country_holidays: bool = Field(True, description="Add country holidays")
    country: str = Field("PL", description="Country for holidays")

    class Config:
        json_schema_extra = {
            "example": {
                "growth": "linear",
                "yearly_seasonality": "auto",
                "weekly_seasonality": "auto",
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "add_country_holidays": True,
                "country": "PL",
            }
        }


class XGBoostConfig(BaseModel):
    """XGBoost model configuration."""

    # Tree parameters
    max_depth: int = Field(6, ge=3, le=10, description="Maximum tree depth")
    n_estimators: int = Field(100, ge=10, le=1000, description="Number of trees")
    learning_rate: float = Field(
        0.1, ge=0.001, le=1.0, description="Learning rate (eta)"
    )

    # Regularization
    reg_alpha: float = Field(0.0, ge=0.0, le=1.0, description="L1 regularization")
    reg_lambda: float = Field(1.0, ge=0.0, le=10.0, description="L2 regularization")

    # Other
    subsample: float = Field(
        1.0, ge=0.5, le=1.0, description="Subsample ratio of training instances"
    )
    colsample_bytree: float = Field(
        1.0, ge=0.5, le=1.0, description="Subsample ratio of columns"
    )

    # Early stopping
    early_stopping_rounds: int = Field(
        10, ge=5, le=50, description="Early stopping rounds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "max_depth": 6,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "early_stopping_rounds": 10,
            }
        }


class LSTMConfig(BaseModel):
    """LSTM model configuration."""

    # Architecture
    hidden_size: int = Field(64, ge=16, le=256, description="LSTM hidden size")
    num_layers: int = Field(2, ge=1, le=4, description="Number of LSTM layers")
    dropout: float = Field(0.2, ge=0.0, le=0.5, description="Dropout rate")

    # Training
    batch_size: int = Field(32, ge=8, le=128, description="Batch size")
    epochs: int = Field(100, ge=10, le=500, description="Number of epochs")
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1, description="Learning rate")

    # Sequence
    sequence_length: int = Field(
        30, ge=7, le=90, description="Input sequence length (lookback)"
    )

    # Early stopping
    patience: int = Field(10, ge=5, le=50, description="Early stopping patience")

    class Config:
        json_schema_extra = {
            "example": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "sequence_length": 30,
                "patience": 10,
            }
        }
