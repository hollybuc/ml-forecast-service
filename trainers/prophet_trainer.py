"""Prophet model trainer."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import pickle
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


class ProphetTrainer:
    """Facebook Prophet model trainer."""

    def __init__(self):
        """Initialize Prophet trainer."""
        self.model_name = "prophet"
        self.model = None
        self.config = None

    def train(
        self,
        train_df: pd.DataFrame,
        date_column: str,
        target_column: str,
        feature_columns: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        frequency: str = 'D',
        **kwargs,
    ) -> Dict[str, Any]:
        """Train Prophet model.

        Args:
            train_df: Training DataFrame.
            date_column: Date column name.
            target_column: Target column name.
            feature_columns: Feature column names (regressors).
            config: Prophet configuration.
            **kwargs: Additional arguments (ignored).

        Returns:
            Training results dictionary.
        """
        logger.info("Training Prophet model")
        
        # Import Prophet lazily to avoid hanging on Windows
        try:
            from prophet import Prophet
        except ImportError as e:
            logger.error(f"Failed to import Prophet: {str(e)}")
            return {
                "model_name": self.model_name,
                "status": "failed",
                "error": f"Prophet not installed: {str(e)}",
            }

        # Default config
        if config is None:
            config = {}

        self.config = config

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame(
            {
                "ds": train_df[date_column],
                "y": train_df[target_column],
            }
        )
        # Ensure datetime + sorted (Prophet is sensitive to ordering)
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

        # Add regressors if provided
        regressors = []
        if feature_columns:
            for col in feature_columns:
                if col in train_df.columns:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(train_df[col]):
                        prophet_df[col] = train_df[col]
                        regressors.append(col)
                    else:
                        logger.warning(f"Skipping non-numeric regressor for Prophet: {col}")

        # Decide seasonality defaults based on history length.
        # Prophet's "auto" can disable yearly seasonality for ~<2y histories, which hurts many retail-like series.
        history_days = None
        try:
            history_days = int((prophet_df["ds"].max() - prophet_df["ds"].min()).days)
        except Exception:
            history_days = None

        yearly_cfg = config.get("yearly_seasonality", "auto")
        if yearly_cfg == "auto" and history_days is not None and history_days >= 365:
            yearly_cfg = True

        # Holidays:
        # This project often already supplies an explicit `is_holiday` regressor.
        # Adding a country calendar by default can inject the *wrong* holiday pattern (e.g. PL) and degrade accuracy.
        add_country_holidays = config.get("add_country_holidays", False)

        # Initialize Prophet with improved defaults for daily sales data
        self.model = Prophet(
            growth=config.get("growth", "linear"),
            yearly_seasonality=self._parse_seasonality(yearly_cfg),
            weekly_seasonality=self._parse_seasonality(config.get("weekly_seasonality", "auto")),
            daily_seasonality=self._parse_seasonality(config.get("daily_seasonality", False)),
            # Increased from 0.05 to allow more flexibility in trend changes
            changepoint_prior_scale=config.get("changepoint_prior_scale", 0.2),
            # Reduced from 10.0 to prevent overfitting to seasonal patterns
            seasonality_prior_scale=config.get("seasonality_prior_scale", 5.0),
            # Multiplicative often works better for sales-like series whose seasonal amplitude scales with level.
            seasonality_mode=config.get("seasonality_mode", "multiplicative"),
            # Increased from 25 (default) to detect more changepoints
            n_changepoints=config.get("n_changepoints", 50),
            # Use more robust optimization
            interval_width=config.get("interval_width", 0.8),
            mcmc_samples=config.get("mcmc_samples", 0),
        )

        # Add country holidays if requested
        if add_country_holidays:
            country = config.get("country", "PL")
            self.model.add_country_holidays(country_name=country)
            logger.info(f"Added holidays for country: {country}")

        # Add regressors
        for regressor in regressors:
            self.model.add_regressor(regressor)
            logger.info(f"Added regressor: {regressor}")

        # Train model
        try:
            self.model.fit(prophet_df)
            logger.info("Prophet model trained successfully")

            return {
                "model_name": self.model_name,
                "status": "success",
                "n_train": len(train_df),
                "regressors": regressors,
                "config": config,
            }

        except Exception as e:
            logger.error(f"Prophet training failed: {str(e)}")
            return {
                "model_name": self.model_name,
                "status": "failed",
                "error": str(e),
            }

    def _parse_seasonality(self, value: Any) -> bool:
        """Parse seasonality parameter.

        Args:
            value: Seasonality value ("auto", True, False, or int).

        Returns:
            Boolean or int.
        """
        if value == "auto":
            return "auto"
        elif isinstance(value, bool):
            return value
        elif isinstance(value, int):
            return value
        else:
            return False

    def predict(
        self,
        horizon: int,
        future_regressors: Optional[pd.DataFrame] = None,
        frequency: str = 'D',
        include_history: bool = False
    ) -> Dict[str, Any]:
        """Generate forecast.

        Args:
            horizon: Number of steps to forecast.
            future_regressors: DataFrame with future regressor values.
            frequency: Frequency of the future dataframe (e.g., 'D' for daily, 'H' for hourly).
            include_history: Whether to include historical predictions in output.

        Returns:
            Dictionary with predictions.
        """
        logger.info(f"Generating {horizon}-step Prophet forecast")

        if self.model is None:
            raise ValueError("Model not trained yet")

        # Generate future dataframe
        future = self.model.make_future_dataframe(periods=horizon, freq=frequency)

        # Add future regressors if provided
        if future_regressors is not None and not future_regressors.empty:
            # Ensure ds column is datetime
            if 'ds' in future_regressors.columns:
                future_regressors = future_regressors.copy()
                future_regressors['ds'] = pd.to_datetime(future_regressors['ds'])
                
                # Handle NaN values in future regressors BEFORE mapping
                regressor_cols = [col for col in future_regressors.columns if col != 'ds']
                for col in regressor_cols:
                    if future_regressors[col].isna().any():
                        logger.warning(f"Future regressor '{col}' contains NaN values. Filling with forward/backward fill.")
                        future_regressors[col] = future_regressors[col].ffill().bfill()
                        # If still NaN, fill with mean
                        if future_regressors[col].isna().any():
                            mean_val = future_regressors[col].mean()
                            if pd.isna(mean_val):
                                mean_val = 0.0
                            logger.warning(f"Future regressor '{col}' still has NaN. Filling with {mean_val}")
                            future_regressors[col] = future_regressors[col].fillna(mean_val)
                
                # Merge future regressors by date
                # This will add regressor columns to the future dataframe
                for col in future_regressors.columns:
                    if col != "ds":
                        # Create a mapping from date to regressor value
                        regressor_map = dict(zip(future_regressors['ds'], future_regressors[col]))
                        # Apply to future dataframe (only updates matching dates, leaves others as NaN)
                        future[col] = future['ds'].map(regressor_map)
            else:
                # If no 'ds' column, assume future_regressors is aligned with forecast period
                for col in future_regressors.columns:
                    if len(future_regressors) >= horizon:
                        # Only update the last 'horizon' rows (forecast period)
                        future.loc[future.index >= len(future) - horizon, col] = future_regressors[col].values[:horizon]
        
        # Fill NaN values in regressors with forward fill (use last known value)
        # This is necessary because Prophet requires all regressor values
        if future_regressors is not None and not future_regressors.empty:
            regressor_cols = [col for col in future_regressors.columns if col != 'ds']
            for col in regressor_cols:
                if col in future.columns:
                    # Use ffill() and bfill() instead of deprecated fillna(method=...)
                    future[col] = future[col].ffill().bfill()

        # Generate forecast
        try:
            forecast = self.model.predict(future)

            # Extract forecast values
            if not include_history:
                forecast = forecast.tail(horizon)

            return {
                "predictions": forecast["yhat"].tolist(),
                "lower_bound": forecast["yhat_lower"].tolist(),
                "upper_bound": forecast["yhat_upper"].tolist(),
                "dates": forecast["ds"].tolist(),
                "components": {
                    "trend": forecast["trend"].tolist(),
                },
            }

        except Exception as e:
            logger.error(f"Prophet prediction failed: {str(e)}")
            raise

    def update(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
    ):
        """Update model by re-fitting on expanded data.
        
        Args:
            df: Combined historical data (e.g. Train + Val).
            date_column: Date column name.
            target_column: Target column name.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Prophet needs a fresh fit to update its history and trend baseline
        logger.info(f"Updating Prophet model with {len(df)} total observations")
        
        prophet_df = pd.DataFrame(
            {
                "ds": df[date_column],
                "y": df[target_column],
            }
        )
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

        # Re-add regressors if they exist in df
        regressors = [col for col in self.model.extra_regressors.keys()]
        for reg in regressors:
            if reg in df.columns:
                # Handle NaN values in regressors - Prophet cannot handle NaN
                if df[reg].isna().any():
                    logger.warning(f"Regressor '{reg}' contains NaN values. Filling with forward/backward fill.")
                    prophet_df[reg] = df[reg].ffill().bfill()
                    # If still NaN after ffill/bfill, fill with mean
                    if prophet_df[reg].isna().any():
                        mean_val = prophet_df[reg].mean()
                        if pd.isna(mean_val):
                            mean_val = 0.0
                        logger.warning(f"Regressor '{reg}' still has NaN after ffill/bfill. Filling with {mean_val}")
                        prophet_df[reg] = prophet_df[reg].fillna(mean_val)
                else:
                    prophet_df[reg] = df[reg]

        try:
            # We must re-fit to update the model state
            self.model.fit(prophet_df)
            logger.info("Prophet model update (re-fit) successful")
        except Exception as e:
            logger.error(f"Failed to update Prophet model: {e}")
            logger.warning("Continuing with original model state (no update applied)")
            # Don't raise - continue with the original model state
            # This allows prediction to proceed even if update fails

    def save(self, path: str):
        """Save model to file.

        Args:
            path: Path to save model.
        """
        # Extract model_id from path (e.g., "models/prophet_region_North" -> "prophet_region_North")
        path_obj = Path(path)
        model_id = path_obj.name if path_obj.name else self.model_name
        model_path = path_obj / f"{model_id}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "config": self.config,
                },
                f,
            )

        logger.info(f"Prophet model saved to {model_path}")

    def load(self, path: str):
        """Load model from file.

        Args:
            path: Path to load model from.
        """
        model_path = Path(path) / f"{self.model_name}.pkl"

        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.config = data.get("config", {})

        logger.info(f"Prophet model loaded from {model_path}")
