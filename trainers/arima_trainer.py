"""ARIMA/SARIMAX model trainer with auto parameter selection and exogenous regressors."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import pickle
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from services.feature_engineering import add_automated_temporal_features
import logging
import pmdarima
from pmdarima.arima import auto_arima

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
        self.auto_feature_columns = []
        self.best_order = None
        self.best_seasonal_order = None
        self.scaler = None

    def train(
        self,
        train_df: pd.DataFrame,
        date_column: str,
        target_column: str,
        config: Optional[Dict[str, Any]] = None,
        frequency: str = 'D',
        log_callback: Optional[callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train ARIMA/SARIMAX model with optional auto parameter selection and exogenous variables.

        Args:
            train_df: Training DataFrame.
            date_column: Date column name.
            target_column: Target column name.
            config: ARIMA configuration.
            log_callback: Optional callback for real-time logging (str -> None).
            **kwargs: Additional arguments (feature_columns, etc.).

        Returns:
            Training results dictionary.
        """
        def _log(msg, level="info"):
            if level == "info":
                logger.info(msg)
            elif level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            elif level == "debug":
                logger.debug(msg)
            if log_callback:
                log_callback(msg, level)

        _log(f"Training ARIMA/SARIMAX model (pmdarima version: {pmdarima.__version__})")
        _log(f"Target: {target_column}, Frequency: {frequency}")
        
        # Default config
        if config is None:
            config = {}

        self.config = config

        # Get feature columns for exogenous regressors
        self.feature_columns = kwargs.get("feature_columns", [])
        
        # Prepare time series
        ts = train_df.set_index(date_column)[target_column]

        # LOGGING ENHANCEMENT: Input Diagnostics
        ts_mean = ts.mean()
        ts_std = ts.std()
        ts_min = ts.min()
        ts_max = ts.max()
        _log(f"ARIMA Input Data Profile: rows={len(ts)}, mean={ts_mean:.2f}, std={ts_std:.2f}, range=[{ts_min:.2f}, {ts_max:.2f}]")
        
        # Automatically generate temporal features for daily data if enabled
        auto_features = config.get("auto_features", True)
        self.auto_feature_columns = []
        
        if auto_features:
            # The features may have already been added by redis_worker.py
            # If so, add_automated_temporal_features will skip re-generation
            train_df, new_feats = add_automated_temporal_features(train_df, date_column, config)
            
            # Identify all 'feat_...' columns present in the dataframe
            self.auto_feature_columns = [col for col in train_df.columns if col.startswith('feat_')]
            if self.auto_feature_columns:
                _log(f"ARIMA using automated temporal features: {self.auto_feature_columns}")

        # Prepare exogenous variables if provided or generated
        exog_cols = (self.feature_columns or []) + self.auto_feature_columns
        exog = None
        self.scaler = None
        
        if exog_cols:
            _log(f"Using exogenous regressors: {exog_cols}")
            exog_df = train_df.set_index(date_column)[exog_cols]
            
            # LOGGING ENHANCEMENT: Sparsity Check (Pre-scaling)
            for col in exog_cols:
                n_zero = (exog_df[col] == 0).sum()
                n_nan = exog_df[col].isna().sum()
                _log(f"  Pre-scale Signal '{col}': zeros={n_zero}/{len(exog_df)}, NaNs={n_nan}/{len(exog_df)}")

            # Scale exogenous variables to prevent optimization issues
            self.scaler = StandardScaler()
            exog_scaled = pd.DataFrame(
                self.scaler.fit_transform(exog_df),
                columns=exog_df.columns,
                index=exog_df.index
            )
            exog = exog_scaled

        # Get parameters
        auto = config.get("auto", True)
        seasonal = config.get("seasonal", False)
        self.is_seasonal = seasonal

        if auto:
            _log("Using auto ARIMA to find optimal parameters...")
            try:
                # Determine seasonal period
                if seasonal:
                    # Try 'm' first (from UI), then 'seasonal_period' (legacy), then auto-detect
                    seasonal_period = config.get("m", config.get("seasonal_period", None))
                    if seasonal_period is None:
                        # Auto-detect from frequency
                        if frequency == 'H':
                            seasonal_period = 24  # Daily seasonality for hourly data
                        elif frequency == 'D':
                            seasonal_period = 7   # Weekly seasonality for daily data
                        elif frequency == 'W' or frequency.startswith('W-'):
                            seasonal_period = 52  # Yearly seasonality for weekly data
                        elif frequency == 'M' or frequency == 'MS':
                            seasonal_period = 12  # Yearly seasonality for monthly data
                        else:
                            # Fallback to mode detection if frequency string is unknown
                            try:
                                freq_mode = train_df[date_column].diff().mode()[0]
                                if freq_mode == pd.Timedelta(days=1):
                                    seasonal_period = 7
                                elif freq_mode == pd.Timedelta(days=7):
                                    seasonal_period = 52
                                else:
                                    seasonal_period = 7
                            except:
                                seasonal_period = 7
                    _log(f"Using seasonal period: {seasonal_period} for frequency {frequency}")
                else:
                    seasonal_period = 1
                
                # Search strategy: Relaxed to allow for better pattern capture
                max_p = config.get("max_p", 5)
                max_d = config.get("max_d", 2)
                max_q = config.get("max_q", 5)
                max_P = config.get("max_P", 2)
                max_Q = config.get("max_Q", 2)
                
                _log(f"Auto-ARIMA search strategy: m={seasonal_period}, max_p={max_p}, max_q={max_q}, max_d={max_d}")
                
                if exog is not None:
                    _log(f"Training with exogenous features. Shape: {exog.shape}")

                # Use exhaustive search for datasets under 1000 points
                use_stepwise = len(ts) >= 1000
                if not use_stepwise:
                    _log("Using exhaustive search (stepwise=False) for better parameter selection")
                
                auto_model = auto_arima(
                    ts,
                    X=exog,
                    start_p=0, max_p=max_p,
                    start_q=0, max_q=max_q,
                    d=None, max_d=max_d,
                    start_P=0, max_P=max_P,
                    start_Q=0, max_Q=max_Q,
                    D=None,
                    m=seasonal_period if seasonal else 1,
                    seasonal=seasonal,
                    stationary=False, # Let auto_arima decide stationarity
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=use_stepwise,
                    information_criterion='aic',
                    with_intercept="auto",
                    random_state=42,
                    n_fits=150,
                    seasonal_test='ch' if seasonal else None,
                )
                
                # Extract best parameters
                self.best_order = auto_model.order
                self.best_seasonal_order = auto_model.seasonal_order if seasonal else (0, 0, 0, 0)
                
                _log(f"Best ARIMA order: {self.best_order}")
                if seasonal:
                    _log(f"Best seasonal order: {self.best_seasonal_order}")
                
                # Use the fitted model from auto_arima
                self.model_fit = auto_model
                
                # Get model statistics
                aic = self.model_fit.aic()
                bic = self.model_fit.bic()
                
                _log(f"Model AIC: {aic:.2f}, BIC: {bic:.2f}")
                
            except Exception as e:
                _log(f"Could not log auto parameters: {e}", level="debug")
                # Fallback to simple default if auto_arima fails or pmdarima not installed
                _log("Auto ARIMA failed or pmdarima not installed, falling back to simple default", level="warning")
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
            _log(f"Training with manual parameters: order={order}, seasonal_order={seasonal_order}")
            
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
            
        # LOGGING ENHANCEMENT: Universal Coefficient Visibility
        try:
            # Handle both pmdarima and statsmodels param access
            if hasattr(self.model_fit, 'params') and callable(self.model_fit.params):
                raw_params = self.model_fit.params()
            elif hasattr(self.model_fit, 'params'):
                raw_params = self.model_fit.params
            else:
                raw_params = {}

            # Extract exogenous coefficients
            if isinstance(raw_params, (pd.Series, dict, np.ndarray)) and exog is not None:
                # pmdarima.params() returns a numpy array, but we need names.
                # In pmdarima, the order usually matches [intercept, exog..., ar..., ma...]
                param_names = []
                if hasattr(self.model_fit, 'arima_res_'): # pmdarima
                    param_names = self.model_fit.arima_res_.param_names
                elif hasattr(self.model_fit, 'param_names'): # statsmodels
                    param_names = self.model_fit.param_names
                
                if param_names and len(raw_params) == len(param_names):
                    # Find overlap with our exogenous columns
                    coef_map = {}
                    for name, val in zip(param_names, raw_params):
                        # Match exactly or as prefix (statsmodels sometimes prefixes)
                        matched_col = next((c for c in exog.columns if c == name or name == f"x{exog.columns.get_loc(c)}"), None)
                        if matched_col:
                            coef_map[matched_col] = float(val)
                    
                    if coef_map:
                        _log(f"Model coefficients (Exogenous): {coef_map}")
                    else:
                        _log(f"Model params (names available): {dict(zip(param_names, raw_params))}", level="debug")
                else:
                    _log(f"Model params (raw): {raw_params}", level="debug")
        except Exception as e:
            _log(f"Could not log detailed parameters: {e}", level="debug")

        _log("ARIMA/SARIMAX model trained successfully")

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
            "auto_feature_columns": self.auto_feature_columns,
            "uses_exog": len(exog_cols) > 0 if exog_cols else False,
        }

    def predict(
        self,
        horizon: int,
        last_date: pd.Timestamp = None,
        frequency: str = "D",
        exog_future: Optional[pd.DataFrame] = None,
        log_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Generate forecast with optional future exogenous variables.

        Args:
            horizon: Number of steps to forecast.
            last_date: Last date in training data.
            frequency: Time series frequency.
            exog_future: Future exogenous variables (required if model was trained with exog).
            log_callback: Optional callback for real-time logging (str -> None).

        Returns:
            Dictionary with predictions.
        """
        def _log(msg, level="info"):
            if level == "info":
                logger.info(msg)
            elif level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            elif level == "debug":
                logger.debug(msg)
            if log_callback:
                log_callback(msg, level)

        if self.model_fit is None:
            raise ValueError("Model not trained yet")
        
        _log(f"Generating {horizon}-step ARIMA forecast")

        if self.model_fit is None:
            raise ValueError("Model not trained yet")

        try:
            # Generate automated features for the forecast horizon if they were used in training
            if self.auto_feature_columns:
                if exog_future is None:
                    exog_future = pd.DataFrame(index=range(horizon))
                
                # Define frequency offset
                try:
                    offset = pd.tseries.frequencies.to_offset(frequency)
                except:
                    offset = pd.Timedelta(days=1)

                # Generate dates for future
                future_dates = pd.date_range(
                    start=last_date + offset,
                    periods=horizon,
                    freq=frequency,
                )
                
                # Add automated features - Day of Week (Cyclic)
                if 'feat_dofw_sin' in self.auto_feature_columns:
                    exog_future['feat_dofw_sin'] = np.sin(2 * np.pi * future_dates.dayofweek / 7)
                if 'feat_dofw_cos' in self.auto_feature_columns:
                    exog_future['feat_dofw_cos'] = np.cos(2 * np.pi * future_dates.dayofweek / 7)
                
                # Month (Cyclic)
                if 'feat_month_sin' in self.auto_feature_columns:
                    exog_future['feat_month_sin'] = np.sin(2 * np.pi * (future_dates.month - 1) / 12)
                if 'feat_month_cos' in self.auto_feature_columns:
                    exog_future['feat_month_cos'] = np.cos(2 * np.pi * (future_dates.month - 1) / 12)
                
                # Fourier terms (Yearly)
                day_of_year = future_dates.dayofyear
                for k in range(1, 4):
                    if f'feat_sin_year_{k}' in self.auto_feature_columns:
                        exog_future[f'feat_sin_year_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
                    if f'feat_cos_year_{k}' in self.auto_feature_columns:
                        exog_future[f'feat_cos_year_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)

            # Check if exogenous variables are needed
            exog_needed = (self.feature_columns or []) + (self.auto_feature_columns or [])
            if exog_needed:
                if exog_future is None:
                    logger.warning("Model was trained with exogenous variables but none provided for prediction")
                else:
                    # Filter and order columns
                    exog_future = exog_future[exog_needed]
                    
                    # Scale according to training data
                    if self.scaler is not None:
                        exog_future_scaled = pd.DataFrame(
                            self.scaler.transform(exog_future),
                            columns=exog_future.columns,
                            index=exog_future.index
                        )
                        exog_future = exog_future_scaled
            
            # Log exogenous info for debugging
            if exog_future is not None:
                logger.info(f"Predicting with exog shape: {exog_future.shape if hasattr(exog_future, 'shape') else len(exog_future)}")
            
            # Generate forecast
            if hasattr(self.model_fit, 'predict'):  # pmdarima model
                # Get confidence intervals
                forecast_result = self.model_fit.predict(
                    n_periods=horizon,
                    X=exog_future,
                    return_conf_int=True,
                    alpha=0.05
                )
                if isinstance(forecast_result, tuple):
                    forecast, conf_int = forecast_result
                else:
                    # Fallback if confidence intervals not available
                    forecast = forecast_result
                    conf_int = None
                    
            else:  # statsmodels model
                forecast = self.model_fit.forecast(steps=horizon, exog=exog_future)
                
                # Get confidence intervals
                forecast_obj = self.model_fit.get_forecast(steps=horizon, exog=exog_future)
                conf_int = forecast_obj.conf_int()

            # Generate forecast dates
            try:
                offset = pd.tseries.frequencies.to_offset(frequency)
            except:
                offset = pd.Timedelta(days=1)

            forecast_dates = pd.date_range(
                start=last_date + offset,
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
            
            # LOGGING ENHANCEMENT: Prediction Diagnostics
            pred_vals = np.array(forecast_values)
            _log(f"ARIMA Prediction Profile: mean={pred_vals.mean():.2f}, std={pred_vals.std():.2f}, range=[{pred_vals.min():.2f}, {pred_vals.max():.2f}]")
            _log(f"ARIMA Sample Predictions (first 5): {forecast_values[:5]}")
            
            return result

        except Exception as e:
            logger.error(f"ARIMA prediction failed: {str(e)}")
            raise

    def update(
        self,
        df: pd.DataFrame,
        target_column: str,
        exog_columns: Optional[list[str]] = None,
        log_callback: Optional[callable] = None,
    ):
        """Update model fit with new observations.

        Args:
            df: New data to append.
            target_column: Target column name.
            exog_columns: Exogenous feature columns (must match training).
            log_callback: Optional callback for real-time logging (str -> None).
        """
        def _log(msg, level="info"):
            if level == "info":
                logger.info(msg)
            elif level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            elif level == "debug":
                logger.debug(msg)
            if log_callback:
                log_callback(msg, level)

        if self.model_fit is None:
            raise ValueError("Model not trained yet")

        y_val = df[target_column].values
        exog_val = df[exog_columns].values if exog_columns else None

        try:
            if hasattr(self.model_fit, "update"): # pmdarima
                self.model_fit.update(y_val, X=exog_val)
                _log(f"ARIMA model updated with {len(y_val)} validation observations")
            else:
                # Statsmodels doesn't have a simple update() exactly like pmdarima
                # For now, we'll just log and skip for pure statsmodels if used
                _log(f"Model {type(self.model_fit)} does not support granular updates. State might be slightly stale.", level="warning")
        except Exception as e:
            _log(f"Failed to update ARIMA model: {e}", level="error")

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
                    "auto_feature_columns": self.auto_feature_columns,
                    "best_order": self.best_order,
                    "best_seasonal_order": self.best_seasonal_order,
                    "scaler": self.scaler,
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
            self.auto_feature_columns = data.get("auto_feature_columns", [])
            self.best_order = data.get("best_order")
            self.best_seasonal_order = data.get("best_seasonal_order")
            self.scaler = data.get("scaler")

        logger.info(f"ARIMA model loaded from {model_path}")
