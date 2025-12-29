"""LightGBM model trainer with feature engineering."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import pickle
from pathlib import Path

import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)


class LGBMTrainer:
    """LightGBM regression model trainer."""

    def __init__(self):
        """Initialize LightGBM trainer."""
        self.model_name = "lightgbm"
        self.model = None
        self.config = None
        self.feature_names = None
        self.time_interval = pd.Timedelta(days=1)
        self.target_differencing = True  # High-impact fix for level-shift issues
        self.last_train_value = 0.0

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        date_column: str,
        target_column: str,
        feature_columns: Optional[list[str]] = None,
        group_column: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train LightGBM model.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            date_column: Date column name.
            target_column: Target column name.
            feature_columns: Additional feature column names.
            config: LightGBM configuration.
            **kwargs: Additional arguments.

        Returns:
            Training results dictionary.
        """
        logger.info("Training LightGBM model")

        # Default config
        if config is None:
            config = {}

        self.config = config

        # Detect time frequency
        try:
            # Try to infer frequency from the first few rows
            sorted_dates = train_df[date_column].sort_values()
            self.time_interval = sorted_dates.diff().mode()[0]
            logger.info(f"Detected time interval: {self.time_interval}")
        except Exception as e:
            logger.warning(f"Could not detect time interval, defaulting to 1 day: {e}")
            self.time_interval = pd.Timedelta(days=1)

        # Feature engineering
        logger.info("Performing feature engineering...")
        train_features = self._create_features(
            train_df, date_column, target_column, feature_columns, group_column
        )
        val_features = (
            self._create_features(val_df, date_column, target_column, feature_columns, group_column)
            if len(val_df) > 0
            else None
        )

        # Prepare X, y
        X_train, y_train = self._prepare_xy(train_features, target_column)

        # Apply target differencing if enabled
        # We predict y[t] - y[t-1] (delta)
        # Note: lag_1 column in features already stores y[t-1]
        self.target_differencing = config.get("target_differencing", True)
        self.last_train_value = train_df[target_column].iloc[-1]

        if self.target_differencing:
            if f"lag_1" in X_train.columns:
                y_train = y_train - X_train["lag_1"]
                logger.info("Using target differencing (predicting deltas)")
            else:
                logger.warning("Target differencing enabled but 'lag_1' missing. Disabling.")
                self.target_differencing = False

        if val_features is not None:
            X_val, y_val = self._prepare_xy(val_features, target_column)
            if self.target_differencing:
                y_val = y_val - X_val["lag_1"]
        else:
            X_val, y_val = None, None

        # Store feature names
        self.feature_names = X_train.columns.tolist()
        logger.info(f"Created {len(self.feature_names)} features")

        # Initialize LightGBM model with improved defaults for time series
        params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "num_leaves": config.get("num_leaves", 64),  # Increased from 31 for more complex patterns
            "max_depth": config.get("max_depth", 8),  # Set explicit depth limit
            "learning_rate": config.get("learning_rate", 0.05),  # Reduced for better generalization
            "n_estimators": config.get("n_estimators", 500),  # Increased significantly (early stopping will prevent overfitting)
            "reg_alpha": config.get("reg_alpha", 0.1),  # L1 regularization
            "reg_lambda": config.get("reg_lambda", 1.0),  # L2 regularization
            "subsample": config.get("subsample", 0.8),  # Use 80% of data per tree
            "colsample_bytree": config.get("colsample_bytree", 0.8),  # Use 80% of features per tree
            "min_child_samples": config.get("min_child_samples", 20),  # Prevent overfitting on small samples
            "random_state": 42,
            "verbosity": -1,
        }

        self.model = lgb.LGBMRegressor(**params)

        # Train model
        try:
            if X_val is not None:
                # Train with validation set and early stopping
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=config.get("early_stopping_rounds", 20),  # Increased patience
                        verbose=False,
                    ),
                ]

                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    callbacks=callbacks,
                )

                best_iteration = self.model.best_iteration_
                val_score = self.model.best_score_["valid_0"]["rmse"]

                logger.info(
                    f"LightGBM trained with early stopping. "
                    f"Best iteration: {best_iteration}, Val RMSE: {val_score:.4f}"
                )
            else:
                # Train without validation
                self.model.fit(X_train, y_train)
                logger.info("LightGBM trained (no validation set)")

            # Feature importance
            feature_importance = dict(
                zip(
                    self.feature_names,
                    self.model.feature_importances_.tolist(),
                )
            )

            # Sort by importance
            feature_importance = dict(
                sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

            return {
                "model_name": self.model_name,
                "status": "success",
                "n_train": len(train_df),
                "n_val": len(val_df) if val_df is not None else 0,
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names,
                "feature_importance": feature_importance,
                "config": config,
            }

        except Exception as e:
            logger.error(f"LightGBM training failed: {str(e)}")
            return {
                "model_name": self.model_name,
                "status": "failed",
                "error": str(e),
            }

    def _create_features(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        feature_columns: Optional[list[str]] = None,
        group_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Create features for LightGBM.

        Args:
            df: Input DataFrame.
            date_column: Date column name.
            target_column: Target column name.
            feature_columns: Additional feature column names.
            group_column: Column for grouping time series.

        Returns:
            DataFrame with features.
        """
        result_df = df.copy()
        
        # Get configurable feature parameters from config (set during training)
        lags = self.config.get('lags', [1, 7, 14, 28]) if hasattr(self, 'config') and self.config else [1, 7, 14, 28]
        rolling_windows = self.config.get('rolling_windows', [7, 14, 30]) if hasattr(self, 'config') and self.config else [7, 14, 30]
        use_ewm = self.config.get('use_ewm', True) if hasattr(self, 'config') and self.config else True
        ewm_span = self.config.get('ewm_span', 7) if hasattr(self, 'config') and self.config else 7
        
        logger.info(f"Creating features with lags={lags}, rolling_windows={rolling_windows}, ewm_span={ewm_span}")

        # If group column exists, create features within each group
        if group_column and group_column in df.columns:
            logger.info(f"Creating features per group: {group_column}")
            group_dfs = []
            
            for group_value in df[group_column].unique():
                group_df = df[df[group_column] == group_value].copy()
                group_df = group_df.sort_values(date_column).reset_index(drop=True)
                
                # Configurable lag features
                for lag in lags:
                    group_df[f"lag_{lag}"] = group_df[target_column].shift(lag)

                # Configurable rolling features
                for window in rolling_windows:
                    group_df[f"rolling_mean_{window}"] = (
                        group_df[target_column].shift(1).rolling(window=window).mean()
                    )
                    group_df[f"rolling_std_{window}"] = (
                        group_df[target_column].shift(1).rolling(window=window).std()
                    )

                # Optional exponential weighted mean
                if use_ewm:
                    group_df[f"ewm_{ewm_span}"] = group_df[target_column].shift(1).ewm(span=ewm_span).mean()
                
                group_dfs.append(group_df)
            
            result_df = pd.concat(group_dfs, ignore_index=True)
        else:
            # No grouping - create features on entire dataset
            # Configurable lag features
            for lag in lags:
                result_df[f"lag_{lag}"] = result_df[target_column].shift(lag)

            # Configurable rolling features
            for window in rolling_windows:
                result_df[f"rolling_mean_{window}"] = (
                    result_df[target_column].shift(1).rolling(window=window).mean()
                )
                result_df[f"rolling_std_{window}"] = (
                    result_df[target_column].shift(1).rolling(window=window).std()
                )

            # Optional exponential weighted mean
            if use_ewm:
                result_df[f"ewm_{ewm_span}"] = result_df[target_column].shift(1).ewm(span=ewm_span).mean()

        # Include additional features if provided
        if feature_columns:
            for col in feature_columns:
                if col in df.columns and col != date_column and col != target_column:
                    # Feature already exists in df
                    pass

        # Time features
        if pd.api.types.is_datetime64_any_dtype(df[date_column]):
            result_df["month"] = result_df[date_column].dt.month
            result_df["day_of_week"] = result_df[date_column].dt.dayofweek
            result_df["day_of_year"] = result_df[date_column].dt.dayofyear

        # Drop rows with NaN (from lags/rolling)
        result_df = result_df.dropna()

        return result_df

    def _prepare_xy(
        self, df: pd.DataFrame, target_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare X and y for training.

        Args:
            df: Features DataFrame.
            target_column: Target column name.

        Returns:
            Tuple of (X, y).
        """
        # Exclude target and date columns
        feature_cols = []
        for col in df.columns:
            if col == target_column:
                continue
            if col.endswith("_date") or col == "date":
                continue

            # LightGBM handles numeric and boolean naturally
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
                df[col]
            ):
                feature_cols.append(col)
            # LightGBM can also handle categorical if properly typed, but for safety in this version
            # we'll stick to numeric unless we explicitly handle categorical conversion.
            # For now, let's keep it consistent with XGBoost implementation
            elif pd.api.types.is_categorical_dtype(
                df[col]
            ) or pd.api.types.is_object_dtype(df[col]):
                # Potentially could be converted to categorical
                logger.warning(f"Skipping non-numeric column: {col}")
                continue

        X = df[feature_cols].copy()
        y = df[target_column]

        return X, y

    def predict(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        horizon: int,
    ) -> Dict[str, Any]:
        """Generate forecast.

        Args:
            df: Historical DataFrame.
            date_column: Date column name.
            target_column: Target column name.
            horizon: Number of steps to forecast.

        Returns:
            Dictionary with predictions.
        """
        logger.info(f"Generating {horizon}-step LightGBM forecast")

        if self.model is None:
            raise ValueError("Model not trained yet")

        predictions = []
        current_df = df.copy()

        # Iterative prediction (one step at a time)
        for step in range(horizon):
            # Create features for current step
            features_df = self._create_features(
                current_df, date_column, target_column, None
            )

            if len(features_df) == 0:
                raise ValueError("Not enough data to create features")

            # Get last row features
            # Ensure we only select features that were used during training
            try:
                X_last = features_df[self.feature_names].iloc[[-1]]
            except KeyError as e:
                missing = list(set(self.feature_names) - set(features_df.columns))
                logger.error(f"Missing features for prediction: {missing}")
                raise ValueError(f"Missing features: {missing}")

            # Predict (model predicts delta if target_differencing is True)
            pred_val = self.model.predict(X_last)[0]

            # Re-integrate if differencing was used
            if self.target_differencing:
                last_val = current_df[target_column].iloc[-1]
                final_pred = last_val + pred_val
            else:
                final_pred = pred_val

            predictions.append(final_pred)

            # Append prediction to dataframe for next iteration
            last_date = current_df[date_column].iloc[-1]

            # Use detected time interval instead of hardcoded 1 day
            next_date = last_date + self.time_interval

            new_row = current_df.iloc[[-1]].copy()
            new_row[date_column] = next_date
            new_row[target_column] = final_pred  # Use prediction for lag features
            
            # IMPROVED: Use actual historical values from same period last year
            # This preserves realistic variation in exogenous features
            historical_data = current_df.head(-1).copy()  # Exclude the last row we just copied
            if len(historical_data) > 0:
                historical_data[date_column] = pd.to_datetime(historical_data[date_column])
                
                # Try to find data from roughly the same time last year (365 days ago)
                # This captures seasonality better than averaging
                if isinstance(next_date, pd.Timestamp):
                    # Calculate date one year ago
                    try:
                        target_date_last_year = next_date - pd.Timedelta(days=365)
                        # Find closest date in history (within 7 days)
                        date_diffs = abs((historical_data[date_column] - target_date_last_year).dt.days)
                        if date_diffs.min() <= 7:  # Found a close match
                            closest_idx = date_diffs.idxmin()
                            # Use values from that historical period
                            for col in new_row.columns:
                                if col not in [date_column, target_column] and col in historical_data.columns:
                                    new_row[col] = historical_data.loc[closest_idx, col]
                            logger.debug(f"Using historical values from {historical_data.loc[closest_idx, date_column]} for forecast date {next_date}")
                        else:
                            # Fallback: use recent same day-of-week pattern (but actual values, not averages)
                            dow = next_date.dayofweek
                            same_dow_data = historical_data[historical_data[date_column].dt.dayofweek == dow]
                            if len(same_dow_data) > 0:
                                # Use the most recent occurrence of this weekday
                                most_recent_idx = same_dow_data[date_column].idxmax()
                                for col in new_row.columns:
                                    if col not in [date_column, target_column] and col in same_dow_data.columns:
                                        new_row[col] = same_dow_data.loc[most_recent_idx, col]
                    except:
                        # If any error, just keep the last row's values
                        pass

            current_df = pd.concat([current_df, new_row], ignore_index=True)

        # Generate forecast dates
        last_date = df[date_column].iloc[-1]

        # Generator dates using the interval
        forecast_dates = [
            last_date + self.time_interval * (i + 1) for i in range(horizon)
        ]

        return {
            "predictions": [float(p) for p in predictions],
            "dates": [d.isoformat() for d in forecast_dates],
        }

    def save(self, path: str):
        """Save model to file.

        Args:
            path: Path to save model.
        """
        path_obj = Path(path)
        model_id = path_obj.name if path_obj.name else self.model_name
        model_path = path_obj / f"{model_id}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "config": self.config,
                    "feature_names": self.feature_names,
                    "time_interval": self.time_interval,
                },
                f,
            )

        logger.info(f"LightGBM model saved to {model_path}")

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
            self.feature_names = data.get("feature_names", [])
            self.time_interval = data.get("time_interval", pd.Timedelta(days=1))

        logger.info(f"LightGBM model loaded from {model_path}")
