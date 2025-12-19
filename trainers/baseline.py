"""Baseline model trainers (Naive, Seasonal Naive)."""

import pandas as pd
import numpy as np
from typing import Dict, Any
import pickle
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


class NaiveTrainer:
    """Naive baseline model (last value forecast)."""

    def __init__(self):
        """Initialize Naive trainer."""
        self.model_name = "naive"
        self.last_value = None
        self.model = None  # For compatibility with worker

    def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train Naive model.

        Args:
            train_df: Training DataFrame.
            target_column: Target column name.
            **kwargs: Additional arguments (ignored).

        Returns:
            Training results dictionary.
        """
        logger.info("Training Naive baseline model")

        # "Training" is just storing the last value
        self.last_value = train_df[target_column].iloc[-1]
        
        # Store model state for pickling
        self.model = {
            "model_name": self.model_name,
            "last_value": float(self.last_value),
        }

        logger.info(f"Naive model trained. Last value: {self.last_value:.2f}")

        return {
            "model_name": self.model_name,
            "status": "success",
            "n_train": len(train_df),
            "last_value": float(self.last_value),
        }

    def predict(
        self,
        horizon: int,
        date_column: str = None,
        last_date: pd.Timestamp = None,
        frequency: str = "D",
    ) -> Dict[str, Any]:
        """Generate forecast.

        Args:
            horizon: Number of steps to forecast.
            date_column: Date column name (optional).
            last_date: Last date in training data (optional).
            frequency: Time series frequency.

        Returns:
            Dictionary with predictions.
        """
        logger.info(f"Generating {horizon}-step Naive forecast")

        # Naive forecast: repeat last value
        predictions = np.full(horizon, self.last_value)

        # Generate dates if provided
        dates = None
        if last_date is not None:
            dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq=frequency,
            )

        return {
            "predictions": predictions.tolist(),
            "dates": dates.tolist() if dates is not None else None,
        }

    def save(self, path: str):
        """Save model to file.

        Args:
            path: Path to save model.
        """
        # Extract model_id from path (e.g., "models/naive_region_North" -> "naive_region_North")
        path_obj = Path(path)
        model_id = path_obj.name if path_obj.name else self.model_name
        model_path = path_obj / f"{model_id}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump({"last_value": self.last_value}, f)

        logger.info(f"Naive model saved to {model_path}")

    def load(self, path: str):
        """Load model from file.

        Args:
            path: Path to load model from.
        """
        model_path = Path(path) / f"{self.model_name}.pkl"

        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.last_value = data["last_value"]

        logger.info(f"Naive model loaded from {model_path}")


class SeasonalNaiveTrainer:
    """Seasonal Naive baseline model."""

    def __init__(self):
        """Initialize Seasonal Naive trainer."""
        self.model_name = "seasonal_naive"
        self.seasonal_values = None
        self.seasonal_period = None
        self.model = None  # For compatibility with worker

    def train(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        seasonal_period: int = 7,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train Seasonal Naive model.

        Args:
            train_df: Training DataFrame.
            target_column: Target column name.
            seasonal_period: Seasonal period (e.g., 7 for weekly).
            **kwargs: Additional arguments (ignored).

        Returns:
            Training results dictionary.
        """
        logger.info(f"Training Seasonal Naive model (period={seasonal_period})")

        self.seasonal_period = seasonal_period

        # Store last season values
        self.seasonal_values = train_df[target_column].iloc[-seasonal_period:].values
        
        # Store model state for pickling
        self.model = {
            "model_name": self.model_name,
            "seasonal_values": self.seasonal_values.tolist(),
            "seasonal_period": seasonal_period,
        }

        logger.info(
            f"Seasonal Naive model trained. Period: {seasonal_period}, "
            f"Mean value: {self.seasonal_values.mean():.2f}"
        )

        return {
            "model_name": self.model_name,
            "status": "success",
            "n_train": len(train_df),
            "seasonal_period": seasonal_period,
            "mean_seasonal_value": float(self.seasonal_values.mean()),
        }

    def predict(
        self,
        horizon: int,
        date_column: str = None,
        last_date: pd.Timestamp = None,
        frequency: str = "D",
    ) -> Dict[str, Any]:
        """Generate forecast.

        Args:
            horizon: Number of steps to forecast.
            date_column: Date column name (optional).
            last_date: Last date in training data (optional).
            frequency: Time series frequency.

        Returns:
            Dictionary with predictions.
        """
        logger.info(f"Generating {horizon}-step Seasonal Naive forecast")

        # Seasonal naive forecast: repeat seasonal pattern
        predictions = np.tile(
            self.seasonal_values, (horizon // self.seasonal_period) + 1
        )[:horizon]

        # Generate dates if provided
        dates = None
        if last_date is not None:
            dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq=frequency,
            )

        return {
            "predictions": predictions.tolist(),
            "dates": dates.tolist() if dates is not None else None,
        }

    def save(self, path: str):
        """Save model to file.

        Args:
            path: Path to save model.
        """
        # Extract model_id from path (e.g., "models/seasonal_naive_region_North" -> "seasonal_naive_region_North")
        path_obj = Path(path)
        model_id = path_obj.name if path_obj.name else self.model_name
        model_path = path_obj / f"{model_id}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "seasonal_values": self.seasonal_values,
                    "seasonal_period": self.seasonal_period,
                },
                f,
            )

        logger.info(f"Seasonal Naive model saved to {model_path}")

    def load(self, path: str):
        """Load model from file.

        Args:
            path: Path to load model from.
        """
        model_path = Path(path) / f"{self.model_name}.pkl"

        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.seasonal_values = data["seasonal_values"]
            self.seasonal_period = data["seasonal_period"]

        logger.info(f"Seasonal Naive model loaded from {model_path}")
