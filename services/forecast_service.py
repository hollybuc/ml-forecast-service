"""Forecast service for generating predictions from trained models."""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import timedelta
from services.feature_engineering import add_automated_temporal_features
import logging

logger = logging.getLogger(__name__)


class ForecastService:
    """Service for generating forecasts from trained models."""

    def __init__(self, models_path: str):
        """Initialize forecast service.
        
        Args:
            models_path: Path to directory containing trained models
        """
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

    def generate_forecast(
        self,
        model_path: str,
        data_path: str,
        date_column: str,
        target_column: str,
        horizon: int,
        frequency: str = 'D',
    ) -> Dict[str, Any]:
        """Generate forecast using a trained model.
        
        Args:
            model_path: Path to the trained model file
            data_path: Path to the historical data CSV
            date_column: Name of the date column
            target_column: Name of the target column
            horizon: Number of periods to forecast
            frequency: Data frequency ('D', 'H', 'W', 'M')
            
        Returns:
            Dictionary containing forecast results
        """
        logger.info(f"Generating forecast: model={model_path}, horizon={horizon}")
        
        # Load model
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load historical data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded CSV columns: {df.columns.tolist()}")
        logger.info(f"Date column: {date_column}, Target column: {target_column}")
        logger.info(f"First few rows:\n{df.head()}")
        
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # Validate columns
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")
        
        # Keep ALL columns - models may need feature columns
        logger.info(f"Full dataframe shape: {df.shape}, columns: {df.columns.tolist()}")
        logger.info(f"Target column dtype: {df[target_column].dtype}")
        
        # Determine model type from filename
        # Format: {project_id}_{model_type}_{timestamp}.pkl or {project_id}_{model_type}_{group}_{timestamp}.pkl
        model_name = model_file.stem
        parts = model_name.split('_')
        
        # Extract model type (second part) and group (if present)
        if len(parts) >= 2:
            model_type = parts[1]
            # Check if there's a group name (parts[2] is not a timestamp)
            group_value = None
            if len(parts) >= 4 and not parts[2].isdigit():
                group_value = parts[2]
                logger.info(f"Model type: {model_type}, Group: {group_value}")
            else:
                logger.info(f"Model type: {model_type} (no grouping)")
        else:
            model_type = 'unknown'
            group_value = None
            logger.warning(f"Could not parse model name: {model_name}")
        
        # Generate forecast based on model type
        if model_type == 'naive':
            predictions, lower_95, upper_95 = self._forecast_naive(
                model, df, target_column, horizon
            )
        elif model_type == 'seasonal':
            predictions, lower_95, upper_95 = self._forecast_seasonal_naive(
                model, df, target_column, horizon
            )
        elif model_type == 'prophet':
            predictions, lower_95, upper_95 = self._forecast_prophet(
                model, df, date_column, horizon
            )
        elif model_type in ['xgboost', 'lightgbm']:
            predictions, lower_95, upper_95 = self._forecast_tree_model(
                model, df, date_column, target_column, horizon
            )
        elif model_type == 'arima':
            predictions, lower_95, upper_95 = self._forecast_arima(
                model, horizon, df
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Generate future dates
        last_date = df[date_column].max()
        future_dates = self._generate_future_dates(last_date, horizon, frequency)
        
        # Include ALL historical data for complete context
        historical_df = df
        historical_dates = historical_df[date_column].tolist()
        historical_values = historical_df[target_column].tolist()
        
        # Create result
        result = {
            'model_name': model_name,
            'model_type': model_type,
            'dates': [d.isoformat() for d in future_dates],
            'predictions': predictions.tolist(),
            'lower_95': lower_95.tolist() if lower_95 is not None else None,
            'upper_95': upper_95.tolist() if upper_95 is not None else None,
            'historical_dates': [d.isoformat() for d in historical_dates],
            'historical_values': historical_values,
            'horizon': horizon,
            'frequency': frequency,
        }
        
        logger.info(f"Forecast generated successfully: {len(predictions)} predictions, {len(historical_values)} historical points")
        return result

    def _forecast_naive(
        self,
        model: Any,
        df: pd.DataFrame,
        target_column: str,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast using Naive model."""
        # Extract the actual model from the ModelManager wrapper
        if isinstance(model, dict) and 'model' in model:
            naive_model = model['model']
        else:
            naive_model = model
        
        # Get last_value from the naive model
        if isinstance(naive_model, dict):
            last_value = naive_model.get('last_value')
        else:
            last_value = getattr(naive_model, 'last_value', None)
        
        if last_value is None:
            raise ValueError("Naive model does not have 'last_value' attribute")
        
        predictions = np.full(horizon, last_value)
        
        # Simple CI based on historical std
        historical_std = df[target_column].std()
        lower_95 = predictions - 1.96 * historical_std
        upper_95 = predictions + 1.96 * historical_std
        
        return predictions, lower_95, upper_95

    def _forecast_seasonal_naive(
        self,
        model: Any,
        df: pd.DataFrame,
        target_column: str,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast using Seasonal Naive model."""
        # Extract the actual model from the ModelManager wrapper
        if isinstance(model, dict) and 'model' in model:
            seasonal_model = model['model']
        else:
            seasonal_model = model
        
        # Get seasonal_period from the seasonal naive model
        if isinstance(seasonal_model, dict):
            seasonal_period = seasonal_model.get('seasonal_period', 7)
        else:
            seasonal_period = getattr(seasonal_model, 'seasonal_period', 7)
        
        # Repeat last seasonal pattern
        last_values = df[target_column].tail(seasonal_period).values
        predictions = np.tile(last_values, (horizon // seasonal_period) + 1)[:horizon]
        
        # Simple CI
        historical_std = df[target_column].std()
        lower_95 = predictions - 1.96 * historical_std
        upper_95 = predictions + 1.96 * historical_std
        
        return predictions, lower_95, upper_95

    def _forecast_prophet(
        self,
        model: Any,
        df: pd.DataFrame,
        date_column: str,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast using Prophet model."""
        # Extract Prophet model - handle both direct Prophet object and wrapped dict
        prophet_model = model
        
        # If it's a dict (from ModelManager), extract the model
        if isinstance(model, dict):
            prophet_model = model.get('model')
        
        # If it still has a 'model' attribute, extract it
        if hasattr(prophet_model, 'model') and prophet_model.model is not None:
            prophet_model = prophet_model.model
        
        if prophet_model is None:
            raise ValueError("Failed to extract Prophet model from saved structure")
        
        logger.info(f"Prophet model type: {type(prophet_model)}")
        
        try:
            from prophet import Prophet
            
            # Verify it's actually a Prophet model
            if not isinstance(prophet_model, Prophet):
                logger.error(f"Expected Prophet model, got {type(prophet_model)}")
                raise ValueError(f"Invalid Prophet model type: {type(prophet_model)}")
            
            # Generate future dataframe
            future = prophet_model.make_future_dataframe(periods=horizon, include_history=False)
            
            # Add regressors if they were used during training
            if hasattr(prophet_model, 'extra_regressors') and prophet_model.extra_regressors:
                logger.info(f"Prophet model uses regressors: {list(prophet_model.extra_regressors.keys())}")
                # For simplicity, fill regressors with last known values
                for regressor in prophet_model.extra_regressors.keys():
                    if regressor in df.columns:
                        last_value = df[regressor].iloc[-1]
                        future[regressor] = last_value
                        logger.info(f"Added regressor '{regressor}' with value {last_value}")
            
            # Generate forecast
            forecast = prophet_model.predict(future)
            predictions = forecast['yhat'].values
            lower_95 = forecast['yhat_lower'].values
            upper_95 = forecast['yhat_upper'].values
            
            return predictions, lower_95, upper_95
            
        except ImportError:
            raise ValueError("Prophet not installed")
        except Exception as e:
            logger.error(f"Prophet forecast error: {str(e)}")
            raise

    def _forecast_tree_model(
        self,
        model: Any,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast using Tree-based model (XGBoost/LightGBM)."""
        # Extract the saved model data
        # ModelManager wraps everything, so structure is: {model: {...}, model_name, metadata, ...}
        # For tree models, the inner dict has: {model: <actual>, config, feature_names, time_interval}
        if isinstance(model, dict):
            # First level: extract from ModelManager wrapper
            inner = model.get('model')
            
            # Check if it's a tree model dict (has feature_names)
            if isinstance(inner, dict) and 'feature_names' in inner:
                # Tree model: extract all components
                trained_model = inner.get('model')
                feature_names = inner.get('feature_names', [])
                time_interval = inner.get('time_interval', pd.Timedelta(days=1))
                config = inner.get('config', {})
            else:
                # Not a tree model or already extracted
                trained_model = inner
                feature_names = []
                time_interval = pd.Timedelta(days=1)
                config = {}
        else:
            # Direct model object (shouldn't happen)
            trained_model = model
            feature_names = []
            time_interval = pd.Timedelta(days=1)
            config = {}
        
        if trained_model is None:
            raise ValueError("Failed to extract tree model from saved structure")
        
        logger.info(f"Tree model type: {type(trained_model)}, features: {len(feature_names)}")
        
        # Recreate the trainer to use its predict method which handles feature engineering
        # Determine model type
        model_type_str = str(type(trained_model))
        if 'xgboost' in model_type_str.lower() or 'XGBRegressor' in model_type_str:
            from trainers.xgboost_trainer import XGBoostTrainer
            trainer = XGBoostTrainer()
        elif 'lightgbm' in model_type_str.lower() or 'lgbm' in model_type_str.lower() or 'LGBMRegressor' in model_type_str:
            from trainers.lgbm_trainer import LGBMTrainer
            trainer = LGBMTrainer()
        else:
            raise ValueError(f"Unknown tree model type: {type(trained_model)}")
        
        # Set the trainer's model and metadata
        trainer.model = trained_model
        trainer.feature_names = feature_names
        trainer.time_interval = time_interval
        trainer.config = config
        
        # Use the trainer's predict method which handles feature engineering
        result = trainer.predict(df, date_column, target_column, horizon)
        predictions = np.array(result['predictions'])
        
        # Simple CI based on historical std
        historical_std = df[target_column].std()
        lower_95 = predictions - 1.96 * historical_std
        upper_95 = predictions + 1.96 * historical_std
        
        return predictions, lower_95, upper_95

    def _forecast_arima(
        self,
        model: Any,
        horizon: int,
        df: pd.DataFrame = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast using ARIMA model."""
        try:
            # Extract ARIMA model_fit - handle both direct object and wrapped dict
            model_fit = model
            
            # If it's a dict (from ModelManager), extract the model
            if isinstance(model, dict):
                model_fit = model.get('model_fit') or model.get('model')
            
            # If it still has a 'model_fit' attribute, extract it
            if hasattr(model_fit, 'model_fit') and model_fit.model_fit is not None:
                model_fit = model_fit.model_fit
            
            if model_fit is None:
                raise ValueError("Failed to extract ARIMA model_fit from saved structure")
            
            logger.info(f"ARIMA model_fit type: {type(model_fit)}")
            
            # Generate forecast
            # Check if it's pmdarima (has predict with n_periods) or statsmodels (has forecast)
            model_type_str = str(type(model_fit))
            
            # Check if model needs exogenous variables
            exog_future = None
            
            # DEBUG LOGGING: Show what's in the model dict
            if isinstance(model, dict):
                logger.info(f"Model dict keys: {list(model.keys())}")
            
            # Try to get model metadata to understand which features were used
            # ARIMA trainer saves feature_columns and auto_feature_columns at TOP LEVEL of dict
            if isinstance(model, dict):
                feature_columns = model.get('feature_columns', [])
                auto_feature_columns = model.get('auto_feature_columns', [])
                scaler = model.get('scaler', None)
                logger.info(f"Extracted from model dict: feature_columns={feature_columns}, auto_feature_columns={auto_feature_columns}, scaler={'present' if scaler else 'None'}")
            else:
                feature_columns = getattr(model_fit, 'feature_columns', [])
                auto_feature_columns = getattr(model_fit, 'auto_feature_columns', [])
                scaler = getattr(model_fit, 'scaler', None)
                logger.info(f"Extracted from model_fit: feature_columns={feature_columns}, auto_feature_columns={auto_feature_columns}, scaler={'present' if scaler else 'None'}")
            
            # Check if model was trained with exog
            model_has_exog = False
            if hasattr(model_fit, 'model') and hasattr(model_fit.model, 'exog'):
                model_has_exog = model_fit.model.exog is not None
            elif hasattr(model_fit, 'exog_'):
                model_has_exog = model_fit.exog_ is not None
            
            logger.info(f"Model has exog check: {model_has_exog}")
            
            if model_has_exog:
                logger.info("ARIMA model was trained with exogenous variables, preparing future exog")
                logger.info(f"Feature columns from training: {feature_columns}")
                logger.info(f"Auto feature columns: {auto_feature_columns}")
                
                if df is not None:
                    # Generate future dates for the forecast horizon
                    date_col = df.columns[0]  # Assume first column is date
                    last_date = pd.to_datetime(df[date_col].max())
                    
                    # Infer frequency from data
                    date_series = pd.to_datetime(df[date_col])
                    freq_mode = date_series.diff().mode()
                    if not freq_mode.empty:
                        freq_delta = freq_mode[0]
                    else:
                        freq_delta = pd.Timedelta(days=1)
                    
                    # Generate future dates
                    future_dates = pd.date_range(
                        start=last_date + freq_delta,
                        periods=horizon,
                        freq=freq_delta
                    )
                    
                    # Create future dataframe
                    future_df = pd.DataFrame({date_col: future_dates})
                    
                    # Generate automated temporal features (Fourier, DOFW, Month)
                    if auto_feature_columns:
                        logger.info(f"Generating {len(auto_feature_columns)} automated temporal features")
                        future_df, _ = add_automated_temporal_features(future_df, date_col)
                    
                    # Forward-fill user-provided features from last known values
                    user_features = [f for f in feature_columns if f not in auto_feature_columns]
                    if user_features:
                        logger.info(f"Forward-filling {len(user_features)} user features: {user_features}")
                        for feat in user_features:
                            if feat in df.columns:
                                last_value = df[feat].iloc[-1]
                                future_df[feat] = last_value
                            else:
                                logger.warning(f"Feature '{feat}' not found in data, filling with 0")
                                future_df[feat] = 0
                    
                    # Combine all exog columns in the correct order
                    all_exog_cols = user_features + auto_feature_columns
                    exog_future_df = future_df[all_exog_cols]
                    
                    # Apply scaler if it was saved
                    if scaler is not None:
                        logger.info("Applying saved StandardScaler to future exog")
                        exog_future = scaler.transform(exog_future_df)
                    else:
                        exog_future = exog_future_df.values
                    
                    logger.info(f"Generated future exog shape: {exog_future.shape}")
                else:
                    logger.error("Model requires exog but no historical data provided")
                    raise ValueError("ARIMA model requires exogenous variables but no data was provided")
            
            if 'pmdarima' in model_type_str or 'arima.ARIMA' in model_type_str:
                # pmdarima ARIMA model
                logger.info("Using pmdarima ARIMA forecast method")
                try:
                    predictions, conf_int = model_fit.predict(
                        n_periods=horizon, 
                        X=exog_future,
                        return_conf_int=True, 
                        alpha=0.05
                    )
                    predictions = np.array(predictions)
                    lower_95 = conf_int[:, 0]
                    upper_95 = conf_int[:, 1]
                except Exception as e:
                    logger.warning(f"Confidence interval extraction failed: {e}, using predictions only")
                    predictions = model_fit.predict(n_periods=horizon, X=exog_future)
                    predictions = np.array(predictions)
                    # Fallback CI
                    std_err = np.std(predictions) if len(predictions) > 1 else abs(predictions[0]) * 0.1
                    lower_95 = predictions - 1.96 * std_err
                    upper_95 = predictions + 1.96 * std_err
            else:
                # statsmodels ARIMA model
                logger.info("Using statsmodels ARIMA forecast method")
                forecast_obj = model_fit.get_forecast(steps=horizon, exog=exog_future)
                forecast_result = forecast_obj.predicted_mean
                predictions = forecast_result.values if hasattr(forecast_result, 'values') else np.array(forecast_result)
                
                # Get prediction intervals (95% confidence)
                try:
                    pred_int = forecast_obj.conf_int(alpha=0.05)
                    lower_95 = pred_int.iloc[:, 0].values
                    upper_95 = pred_int.iloc[:, 1].values
                except Exception as e:
                    logger.warning(f"Confidence interval extraction failed: {e}")
                    # Fallback CI
                    std_err = np.std(predictions) if len(predictions) > 1 else abs(predictions[0]) * 0.1
                    lower_95 = predictions - 1.96 * std_err
                    upper_95 = predictions + 1.96 * std_err
            
            return np.array(predictions), np.array(lower_95), np.array(upper_95)
            
        except Exception as e:
            logger.error(f"ARIMA forecast error: {str(e)}")
            logger.error(f"Model structure: {model}")
            raise

    def _generate_future_dates(
        self,
        last_date: pd.Timestamp,
        horizon: int,
        frequency: str,
    ) -> List[pd.Timestamp]:
        """Generate future dates based on frequency."""
        if frequency == 'D':
            dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]
        elif frequency == 'H':
            dates = [last_date + timedelta(hours=i + 1) for i in range(horizon)]
        elif frequency == 'W':
            dates = [last_date + timedelta(weeks=i + 1) for i in range(horizon)]
        elif frequency == 'M':
            dates = pd.date_range(start=last_date, periods=horizon + 1, freq='MS')[1:].tolist()
        else:
            # Default to daily
            dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]
        
        return dates
