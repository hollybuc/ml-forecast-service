"""Enumerations for model types and methods."""

from enum import Enum


class ModelType(str, Enum):
    """Available model types."""

    # Statistical models
    ARIMA = "arima"
    SARIMA = "sarima"
    ETS = "ets"
    PROPHET = "prophet"

    # Machine Learning models
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"

    # Deep Learning models
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"

    # Baseline models
    NAIVE = "naive"
    SEASONAL_NAIVE = "seasonal_naive"
    MOVING_AVERAGE = "moving_average"


class MissingMethod(str, Enum):
    """Methods for handling missing values."""

    DROP = "drop"
    INTERPOLATE = "interpolate"
    FFILL = "ffill"
    BFILL = "bfill"
    MEAN = "mean"
    MEDIAN = "median"


class OutlierMethod(str, Enum):
    """Methods for handling outliers."""

    NONE = "none"
    CLIP = "clip"
    MEDIAN = "median"
    REMOVE = "remove"


class TransformMethod(str, Enum):
    """Transformation methods for target variable."""

    NONE = "none"
    LOG = "log"
    SQRT = "sqrt"
    BOXCOX = "box-cox"
