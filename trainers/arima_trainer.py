"""ARIMA/SARIMAX model trainer with auto parameter selection and exogenous regressors."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import pickle
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging

logger = logging.getLogger(__name__)


class ARIMATrainer:
    """ARIMA/SARIMAX model trainer with auto_arima and exogenous regressors support."""

    def __init__(self):
        """Initialize ARIMA trainer."""
        self.model_name = "arima"
        self.model = None
        self.model_fit = None
        self.config = None
        self.is_seasonal = False
        self.feature_columns = None
        self.best_order = None
        self.best_seasonal_order = None

    def train(
        self,
        train_df: pd.DataFrame,
        date_column: str,
        target_column: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train ARIMA/SARIMAX model with optional auto parameter selection and exogenous variables.

        Args:
            train_df: Training DataFrame.
            date_column: Date column name.
            target_column: Target column name.
            config: ARIMA configuration.
            **kwargs: Additional arguments (feature_columns, etc.).

        Returns:
            Training results dictionary.
        """
        logger.info("Training ARIMA/SARIMAX model")

        # Default config
        if config is None:
            config = {}

        self.config = config

        # Get feature columns for exogenous regressors
        self.feature_columns = kwargs.get("feature_columns", [])
        
        # Prepare time series
        ts = train_df.set_index(date_column)[target_column]
        
        # Prepare exogenous variables if provided
        exog = None
        if self.feature_columns:
            logger.info(f"Using exogenous regressors: {self.feature_columns}")
            exog = train_df.set_index(date_column)[self.feature_columns]

        # Get parameters
        auto = config.get("auto", True)
        seasonal = config.get("seasonal", False)
        self.is_seasonal = seasonal

        if auto:
            logger.info("Using auto ARIMA to find optimal parameters...")
            try:
                # Import pmdarima
                from pmdarima import auto_arima
                
                # Determine seasonal period
                if seasonal:
                    # Try 'm' first (from UI), then 'seasonal_period' (legacy), then auto-detect
                    seasonal_period = config.get("m", config.get("seasonal_period", None))
                    if seasonal_period is None:
                        # Auto-detect from frequency
                        freq = train_df[date_column].diff().mode()[0]
                        if freq == pd.Timedelta(days=1):
                            seasonal_period = 7  # Weekly for daily data
                        elif freq == pd.Timedelta(days=7):
                            seasonal_period = 52  # Yearly for weekly data
                        elif freq >= pd.Timedelta(days=28) and freq <= pd.Timedelta(days=31):
                            seasonal_period = 12  # Yearly for monthly data
                        else:
                            seasonal_period = 7  # Default
                    logger.info(f"Using seasonal period: {seasonal_period}")
                else:
                    seasonal_period = 1
                
                # Get search parameters from config - increased defaults for better fit
                max_p = config.get("max_p", 7)  # Increased from 5
                max_d = config.get("max_d", 2)
                max_q = config.get("max_q", 7)  # Increased from 5
                max_P = config.get("max_P", 3)  # Increased from 2
                max_Q = config.get("max_Q", 3)  # Increased from 2
                
                # Run auto_arima
                logger.info(f"Searching for best ARIMA parameters (max_p={max_p}, max_d={max_d}, max_q={max_q})...")
                try:
                    auto_model = auto_arima(
                        ts,
                        exogenous=exog,
                        # Allow simpler models (including p=0/q=0). For some series this avoids "flatline" forecasts.
                        start_p=0, max_p=max_p,
                        start_q=0, max_q=max_q,
                        d=None, max_d=max_d,  # Auto-detect differencing with max limit
                        start_P=0, max_P=max_P,
                        start_Q=0, max_Q=max_Q,
                        D=None,  # Auto-detect seasonal differencing
                        m=seasonal_period if seasonal else 1,
                        seasonal=seasonal,
                        trace=False,  # Disable trace for cleaner logs
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,  # Use stepwise for faster search
                        information_criterion='aic',  # Use AIC for model selection
                        # Ensure intercept is allowed; otherwise differenced models can collapse to near-constant forecasts.
                        with_intercept=True,
                        random_state=42,
                        n_fits=100,  # Increased from 50 for more thorough search
                    )
                except (ValueError, np.linalg.LinAlgError) as e:
                    # If auto-detection fails (e.g., singular matrix in ADF test),
                    # use manual differencing parameters
                    logger.warning(f"Auto-detection failed ({str(e)}), using manual differencing")
                    auto_model = auto_arima(
                        ts,
                        exogenous=exog,
                        start_p=0, max_p=5,
                        start_q=0, max_q=5,
                        d=1,  # Manual differencing
                        start_P=0, max_P=2,
                        start_Q=0, max_Q=2,
                        D=1 if seasonal else 0,  # Manual seasonal differencing
                        m=seasonal_period if seasonal else 1,
                        seasonal=seasonal,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        random_state=42,
                        n_fits=50,
                        test='kpss',  # Use KPSS test instead of ADF
                    )
                
                # Extract best parameters
                self.best_order = auto_model.order
                self.best_seasonal_order = auto_model.seasonal_order if seasonal else (0, 0, 0, 0)
                
                logger.info(f"Best ARIMA order: {self.best_order}")
                if seasonal:
                    logger.info(f"Best seasonal order: {self.best_seasonal_order}")
                
                # Use the fitted model from auto_arima
                self.model_fit = auto_model
                
                # Get model statistics
                aic = self.model_fit.aic()
                bic = self.model_fit.bic()
                
                logger.info(f"Model AIC: {aic:.2f}, BIC: {bic:.2f}")
                
            except ImportError:
                logger.warning("pmdarima not installed, falling back to simple default")
                # Fallback to simple default
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 7) if seasonal else (0, 0, 0, 0)
                self.best_order = order
                self.best_seasonal_order = seasonal_order
                
                # Train with statsmodels
                if seasonal or exog is not None:
                    self.model = SARIMAX(
                        ts,
                        exog=exog,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                else:
                    self.model = ARIMA(ts, exog=exog, order=order)
                
                self.model_fit = self.model.fit()
                aic = self.model_fit.aic
                bic = self.model_fit.bic
                
        else:
            # Manual parameters
            p = config.get("p", 1)
            d = config.get("d", 1)
            q = config.get("q", 1)
            order = (p, d, q)
            self.best_order = order

            if seasonal:
                P = config.get("P", 1)
                D = config.get("D", 1)
                Q = config.get("Q", 1)
                s = config.get("seasonal_period", 7)
                seasonal_order = (P, D, Q, s)
                self.best_seasonal_order = seasonal_order
            else:
                seasonal_order = (0, 0, 0, 0)
                self.best_seasonal_order = seasonal_order

            # Train model
            logger.info(f"Training with manual parameters: order={order}, seasonal_order={seasonal_order}")
            
            if seasonal or exog is not None:
                self.model = SARIMAX(
                    ts,
                    exog=exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                self.model = ARIMA(ts, exog=exog, order=order)

            # Fit model
            self.model_fit = self.model.fit()
            aic = self.model_fit.aic
            bic = self.model_fit.bic

        logger.info("ARIMA/SARIMAX model trained successfully")

        return {
            "model_name": self.model_name,
            "status": "success",
            "n_train": len(train_df),
            "order": self.best_order,
            "seasonal_order": self.best_seasonal_order if seasonal else None,
            "aic": float(aic),
            "bic": float(bic),
            "config": config,
            "feature_columns": self.feature_columns,
            "uses_exog": len(self.feature_columns) > 0 if self.feature_columns else False,
        }

    def predict(
        self,
        horizon: int,
        last_date: pd.Timestamp,
        frequency: str = "D",
        exog_future: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate forecast with optional future exogenous variables.

        Args:
            horizon: Number of steps to forecast.
            last_date: Last date in training data.
            frequency: Time series frequency.
            exog_future: Future exogenous variables (required if model was trained with exog).

        Returns:
            Dictionary with predictions.
        """
        logger.info(f"Generating {horizon}-step ARIMA forecast")

        if self.model_fit is None:
            raise ValueError("Model not trained yet")

        try:
            # Check if exogenous variables are needed
            if self.feature_columns and exog_future is None:
                logger.warning("Model was trained with exogenous variables but none provided for prediction")
            
            # Log exogenous info for debugging
            if exog_future is not None:
                logger.info(f"Predicting with exog shape: {exog_future.shape if hasattr(exog_future, 'shape') else len(exog_future)}")
            
            # Generate forecast
            if hasattr(self.model_fit, 'predict'):  # pmdarima model
                forecast = self.model_fit.predict(n_periods=horizon, exogenous=exog_future)
                
                # Get confidence intervals
                forecast_result = self.model_fit.predict(
                    n_periods=horizon,
                    exogenous=exog_future,
                    return_conf_int=True,
                    alpha=0.05
                )
                if isinstance(forecast_result, tuple):
                    forecast, conf_int = forecast_result
                else:
                    # Fallback if confidence intervals not available
                    conf_int = None
                    
            else:  # statsmodels model
                forecast = self.model_fit.forecast(steps=horizon, exog=exog_future)
                
                # Get confidence intervals
                forecast_obj = self.model_fit.get_forecast(steps=horizon, exog=exog_future)
                conf_int = forecast_obj.conf_int()

            # Generate forecast dates
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq=frequency,
            )

            # Ensure forecast is a numpy array or list of correct length
            if hasattr(forecast, 'values'):
                forecast_values = forecast.values
            elif hasattr(forecast, 'tolist'):
                forecast_values = forecast.tolist()
            else:
                forecast_values = list(forecast)
            
            # Ensure we only return the requested horizon
            forecast_values = forecast_values[:horizon] if len(forecast_values) > horizon else forecast_values

            result = {
                "predictions": forecast_values,
                "dates": [d.isoformat() for d in forecast_dates],
            }
            
            # Add confidence intervals if available
            if conf_int is not None:
                if isinstance(conf_int, np.ndarray):
                    result["lower_bound"] = conf_int[:, 0].tolist()
                    result["upper_bound"] = conf_int[:, 1].tolist()
                else:
                    result["lower_bound"] = conf_int.iloc[:, 0].tolist()
                    result["upper_bound"] = conf_int.iloc[:, 1].tolist()
            
            return result

        except Exception as e:
            logger.error(f"ARIMA prediction failed: {str(e)}")
            raise

    def save(self, path: str):
        """Save model to file.

        Args:
            path: Path to save model.
        """
        # Extract model_id from path (e.g., "models/arima_region_North" -> "arima_region_North")
        path_obj = Path(path)
        model_id = path_obj.name if path_obj.name else self.model_name
        model_path = path_obj / f"{model_id}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model_fit": self.model_fit,
                    "config": self.config,
                    "is_seasonal": self.is_seasonal,
                    "feature_columns": self.feature_columns,
                    "best_order": self.best_order,
                    "best_seasonal_order": self.best_seasonal_order,
                },
                f,
            )

        logger.info(f"ARIMA model saved to {model_path}")

    def load(self, path: str):
        """Load model from file.

        Args:
            path: Path to load model from.
        """
        model_path = Path(path) / f"{self.model_name}.pkl"

        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.model_fit = data["model_fit"]
            self.config = data.get("config", {})
            self.is_seasonal = data.get("is_seasonal", False)
            self.feature_columns = data.get("feature_columns", [])
            self.best_order = data.get("best_order")
            self.best_seasonal_order = data.get("best_seasonal_order")

        logger.info(f"ARIMA model loaded from {model_path}")
