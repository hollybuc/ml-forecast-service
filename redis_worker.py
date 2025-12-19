"""
Redis-based worker for ML Forecast Service.
Polls Redis queue for training and forecast jobs and processes them.
"""

import os
import json
import time
import logging
import redis
import traceback
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Redis keys/channels
ML_JOBS_QUEUE = "ml:jobs:queue"
ML_FORECAST_JOBS_QUEUE = "ml:forecast:queue"
ML_EDA_JOBS_QUEUE = "ml:eda:queue"
ML_PREPROCESSING_JOBS_QUEUE = "ml:preprocessing:queue"
ML_EVALUATION_JOBS_QUEUE = "ml:evaluation:queue"
ML_JOBS_UPDATE_CHANNEL = "ml:jobs:update"
ML_JOB_DATA_PREFIX = "ml:job:data:"
ML_JOB_LOGS_PREFIX = "ml:job:logs:"

# Initialize Redis clients
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")


def publish_progress(job_id: str, status: str, message: str, percentage: int = 0, data: Dict[str, Any] = None):
    """Publish job progress to Redis channel."""
    try:
        event_data = {
            "event": "progress",
            "jobId": job_id,
            "data": {
                "status": status,
                "message": message,
                "percentage": percentage,
                **(data or {})
            }
        }
        redis_client.publish(ML_JOBS_UPDATE_CHANNEL, json.dumps(event_data))
        logger.info(f"[{job_id}] Progress: {status} - {message} ({percentage}%)")
    except Exception as e:
        logger.error(f"Failed to publish progress for job {job_id}: {e}")


def append_log(job_id: str, message: str, level: str = "info"):
    """Append log message to job logs in Redis."""
    try:
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message
        }
        redis_client.rpush(f"{ML_JOB_LOGS_PREFIX}{job_id}", json.dumps(log_entry))
        logger.info(f"[{job_id}] {level.upper()}: {message}")
    except Exception as e:
        logger.error(f"Failed to append log for job {job_id}: {e}")


def _safe_dt_series(values) -> pd.Series:
    """Convert array-like to datetime Series safely."""
    return pd.to_datetime(pd.Series(values), errors="coerce")


def _summarize_time_index(dates: pd.Series) -> Dict[str, Any]:
    """Summarize a datetime-like Series for logging/debugging."""
    s = pd.to_datetime(dates, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    s = s.sort_values()
    diffs = s.diff().dropna()
    # Mode of diffs can be empty if only 1 date
    mode_diff = diffs.mode().iloc[0] if not diffs.empty and not diffs.mode().empty else None
    summary: Dict[str, Any] = {
        "count": int(len(s)),
        "start": str(s.iloc[0]),
        "end": str(s.iloc[-1]),
        "n_unique": int(s.nunique()),
        "n_duplicates": int(len(s) - s.nunique()),
        "mode_step": str(mode_diff) if mode_diff is not None else None,
    }
    if len(s) >= 2:
        # Expected daily grid check (best-effort)
        expected = pd.date_range(start=s.iloc[0], end=s.iloc[-1], freq="D")
        summary["expected_daily_points"] = int(len(expected))
        summary["missing_daily_points"] = int(len(expected.difference(pd.DatetimeIndex(s.unique()))))
    return summary


def _summarize_numeric(df: pd.DataFrame, cols: list[str]) -> Dict[str, Any]:
    """Summarize numeric columns for logging/debugging."""
    out: Dict[str, Any] = {}
    for c in cols:
        if c not in df.columns:
            out[c] = {"present": False}
            continue
        s = df[c]
        out[c] = {
            "present": True,
            "dtype": str(s.dtype),
            "n": int(len(s)),
            "n_nan": int(pd.isna(s).sum()),
        }
        if pd.api.types.is_numeric_dtype(s):
            out[c].update(
                {
                    "min": float(np.nanmin(s.values)) if len(s) else None,
                    "max": float(np.nanmax(s.values)) if len(s) else None,
                    "mean": float(np.nanmean(s.values)) if len(s) else None,
                    "std": float(np.nanstd(s.values)) if len(s) else None,
                }
            )
    return out


def process_training_job(job_data: Dict[str, Any]):
    """Process a training job."""
    job_id = job_data.get("jobId")
    
    try:
        append_log(job_id, "Starting ML training job...")
        publish_progress(job_id, "running", "Initializing training...", 10)
        
        # Import ML trainers (lazy import to avoid startup delays)
        from trainers.prophet_trainer import ProphetTrainer
        from trainers.arima_trainer import ARIMATrainer
        from trainers.xgboost_trainer import XGBoostTrainer
        from trainers.lgbm_trainer import LGBMTrainer
        from trainers.baseline import NaiveTrainer, SeasonalNaiveTrainer
        from services.data_loader import DataLoader
        from services.model_manager import ModelManager
        
        # Extract job parameters
        dataset_path = job_data.get("datasetPath")
        models = job_data.get("models", [])
        config = job_data.get("config", {})
        target_column = job_data.get("targetColumn", "value")
        date_column = job_data.get("dateColumn", "date")
        feature_columns = job_data.get("featureColumns", [])
        training_result_id = job_data.get("trainingResultId")
        
        append_log(job_id, f"Loading dataset from {dataset_path}...")
        publish_progress(job_id, "running", "Loading dataset...", 20)
        
        # Load dataset
        data_loader = DataLoader()
        df = data_loader.load_csv(dataset_path)
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Ensure date column is datetime
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column).reset_index(drop=True)
        
        append_log(job_id, f"Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")
        # High-signal dataset diagnostics (helps explain bad metrics)
        try:
            date_summary = _summarize_time_index(df[date_column]) if date_column in df.columns else {"missing_date_column": True}
            append_log(job_id, f"Dataset date diagnostics: {date_summary}")
            # Show quick target stats too
            if target_column in df.columns:
                tgt_stats = _summarize_numeric(df, [target_column]).get(target_column, {})
                append_log(job_id, f"Target '{target_column}' stats: {tgt_stats}")
            if feature_columns:
                feat_stats = _summarize_numeric(df, feature_columns)
                append_log(job_id, f"Feature stats: {feat_stats}")
        except Exception as diag_err:
            append_log(job_id, f"WARNING: Failed dataset diagnostics: {diag_err}", "warning")
        
        # Prepare data
        append_log(job_id, "Preparing data for training...")
        publish_progress(job_id, "running", "Preparing data...", 30)
        
        # Get split configuration from job data (default to 70/15/15)
        train_ratio = config.get("trainRatio", 0.7)
        val_ratio = config.get("valRatio", 0.15)
        test_ratio = config.get("testRatio", 0.15)
        
        # Get group column if present
        group_column = job_data.get("groupColumn")
        
        # Split data - handle grouped data properly
        if group_column and group_column in df.columns:
            # For grouped data, split within each group
            append_log(job_id, f"Splitting data by group: {group_column}")
            train_dfs = []
            val_dfs = []
            test_dfs = []
            
            for group_value in df[group_column].unique():
                group_df = df[df[group_column] == group_value].copy()
                group_df = group_df.sort_values(date_column).reset_index(drop=True)
                
                train_size = int(len(group_df) * train_ratio)
                val_size = int(len(group_df) * val_ratio)
                
                train_dfs.append(group_df[:train_size])
                val_dfs.append(group_df[train_size:train_size + val_size])
                test_dfs.append(group_df[train_size + val_size:])
            
            train_df = pd.concat(train_dfs, ignore_index=True)
            val_df = pd.concat(val_dfs, ignore_index=True)
            test_df = pd.concat(test_dfs, ignore_index=True)
        else:
            # Simple temporal split for non-grouped data
            train_size = int(len(df) * train_ratio)
            val_size = int(len(df) * val_ratio)
            
            train_df = df[:train_size].copy()
            val_df = df[train_size:train_size + val_size].copy()
            test_df = df[train_size + val_size:].copy()
        
        append_log(job_id, f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Determine if we're training per-group models or a single model
        if group_column and group_column in df.columns:
            group_values = sorted(df[group_column].unique())
            append_log(job_id, f"Training separate models for {len(group_values)} groups: {group_values}")
        else:
            group_values = [None]  # Single model for all data
        
        # Train models
        results = {}
        model_manager = ModelManager()
        
        trainer_map = {
            "prophet": ProphetTrainer,
            "arima": ARIMATrainer,
            "xgboost": XGBoostTrainer,
            "lightgbm": LGBMTrainer,
            "naive": NaiveTrainer,
            "seasonal_naive": SeasonalNaiveTrainer,
        }
        
        total_models = len(models) * len(group_values)
        model_counter = 0
        
        for model_name in models:
            model_lower = model_name.lower()
            
            if model_lower not in trainer_map:
                append_log(job_id, f"Unknown model: {model_name}", "warning")
                continue
            
            # Train model for each group (or once if no grouping)
            for group_value in group_values:
                model_counter += 1
                progress = 40 + int((model_counter / total_models) * 50)
                
                # Create group-specific model name
                if group_value is not None:
                    full_model_name = f"{model_name}_{group_value}"
                    append_log(job_id, f"Training {model_name} for group '{group_value}'...")
                    publish_progress(job_id, "running", f"Training {model_name} ({group_value})...", progress)
                    
                    # Extract group-specific data
                    group_train_df = train_df[train_df[group_column] == group_value].copy()
                    group_val_df = val_df[val_df[group_column] == group_value].copy()
                    group_test_df = test_df[test_df[group_column] == group_value].copy()
                else:
                    full_model_name = model_name
                    append_log(job_id, f"Training {model_name}...")
                    publish_progress(job_id, "running", f"Training {model_name}...", progress)
                    
                    # Use all data
                    group_train_df = train_df.copy()
                    group_val_df = val_df.copy()
                    group_test_df = test_df.copy()

                # Group split diagnostics (date ranges + continuity)
                try:
                    train_dates = _summarize_time_index(group_train_df[date_column]) if date_column in group_train_df.columns else {"missing_date_column": True}
                    val_dates = _summarize_time_index(group_val_df[date_column]) if date_column in group_val_df.columns else {"missing_date_column": True}
                    test_dates = _summarize_time_index(group_test_df[date_column]) if date_column in group_test_df.columns else {"missing_date_column": True}
                    append_log(job_id, f"Group '{group_value}' split date diagnostics: train={train_dates}, val={val_dates}, test={test_dates}")
                except Exception as diag_err:
                    append_log(job_id, f"WARNING: Failed group split diagnostics for {group_value}: {diag_err}", "warning")
            
                try:
                    trainer_class = trainer_map[model_lower]
                    trainer = trainer_class()
                    
                    # Get model-specific configuration
                    model_configs = job_data.get("modelConfigs", {})
                    model_config = model_configs.get(model_lower, {})
                    
                    # Train model with correct parameters for each model type
                    # NOTE: For grouped models, we DON'T pass group_column to trainers
                    # because we're already training on single-group data
                    if model_lower in ['xgboost', 'lightgbm']:
                        # Tree models need validation set and feature columns
                        trainer.train(
                            train_df=group_train_df,
                            val_df=group_val_df,
                            date_column=date_column,
                            target_column=target_column,
                            feature_columns=feature_columns,
                            group_column=None,  # No grouping needed - already filtered
                            config=model_config
                        )
                    elif model_lower == 'prophet':
                        # Prophet needs explicit parameter names
                        trainer.train(
                            train_df=group_train_df,
                            date_column=date_column,
                            target_column=target_column,
                            feature_columns=feature_columns,
                            config=model_config
                        )
                    elif model_lower == 'arima':
                        # ARIMA needs explicit parameter names
                        # Enable seasonal ARIMA by default for daily data
                        arima_config = model_config.copy() if model_config else {}
                        if 'seasonal' not in arima_config:
                            arima_config['seasonal'] = True  # Enable SARIMAX
                            arima_config['m'] = 7  # Weekly seasonality for daily data
                        
                        trainer.train(
                            train_df=group_train_df,
                            date_column=date_column,
                            target_column=target_column,
                            config=arima_config,
                            feature_columns=feature_columns
                        )
                    elif model_lower in ['naive', 'seasonal_naive']:
                        # Baseline models only need train_df and target
                        trainer.train(
                            train_df=group_train_df,
                            target_column=target_column,
                            seasonal_period=model_config.get('seasonal_period', 7)
                        )
                    else:
                        # Fallback
                        trainer.train(
                            train_df=group_train_df,
                            target_column=target_column,
                            date_column=date_column,
                            config=model_config
                        )
                
                    # Save model
                    # For XGBoost/LightGBM, save complete trainer state (model + metadata)
                    # For ARIMA, save model_fit (the fitted results)
                    # For others, save model
                    if model_lower in ['xgboost', 'lightgbm']:
                        model_to_save = {
                            'model': trainer.model,
                            'config': trainer.config,
                            'feature_names': trainer.feature_names,
                            'time_interval': trainer.time_interval,
                        }
                    else:
                        model_to_save = getattr(trainer, 'model_fit', None) or trainer.model
                    
                    model_path = model_manager.save_model(
                        model_to_save,
                        model_name=full_model_name,  # Use group-specific name
                        project_id=job_data.get("projectId"),
                        metadata={
                            "training_result_id": training_result_id,
                            "job_id": job_id,
                            "group_value": str(group_value) if group_value is not None else None,
                            "train_size": len(group_train_df),
                            "val_size": len(group_val_df),
                            "test_size": len(group_test_df),
                        }
                    )
                
                    # Evaluate model on test set
                    metrics = None
                    try:
                        # IMPORTANT:
                        # Our data is split temporally into train / val / test.
                        # The test window starts AFTER the validation window.
                        # For multi-step forecasting models (Prophet/ARIMA/baselines), if we forecast only
                        # len(test) steps starting at the end of TRAIN, those predictions align to the VAL
                        # period (and only the very last point may overlap with TEST).
                        # That misalignment will produce extremely poor metrics (often negative R²).
                        #
                        # Fix: forecast through (val + test), then score ONLY the last len(test) points.
                        val_len = len(group_val_df)
                        test_len = len(group_test_df)
                        eval_horizon = val_len + test_len

                        if test_len == 0:
                            raise ValueError("Test set is empty; cannot evaluate model")

                        # Build a combined evaluation window (val + test) for exogenous/regressor inputs
                        eval_df = pd.concat([group_val_df, group_test_df], ignore_index=True)
                        if date_column in eval_df.columns:
                            eval_df = eval_df.sort_values(date_column).reset_index(drop=True)

                        # Build the expected future date index (what our forecasters assume)
                        # This matters if the dataset has missing days or irregular spacing.
                        last_date = pd.Timestamp(group_train_df[date_column].iloc[-1])
                        future_dates = pd.date_range(
                            start=last_date + pd.Timedelta(days=1),
                            periods=eval_horizon,
                            freq='D',
                        )

                        append_log(
                            job_id,
                            f"Eval setup: val_len={val_len}, test_len={test_len}, eval_horizon={eval_horizon}, "
                            f"train_end={str(last_date)}, future_start={str(future_dates[0])}, future_end={str(future_dates[-1])}",
                        )
                        # Check how well eval_df covers the expected future date grid
                        try:
                            eval_dates_summary = _summarize_time_index(eval_df[date_column]) if date_column in eval_df.columns else {"missing_date_column": True}
                            append_log(job_id, f"Eval window (val+test) date diagnostics: {eval_dates_summary}")
                            if feature_columns:
                                eval_feat_stats = _summarize_numeric(eval_df, feature_columns)
                                append_log(job_id, f"Eval window feature stats: {eval_feat_stats}")
                        except Exception as diag_err:
                            append_log(job_id, f"WARNING: Failed eval diagnostics: {diag_err}", "warning")

                        # Different trainers have different predict signatures
                        if model_lower == 'prophet':
                            # Prophet: predict(horizon, future_regressors) -> {"predictions": [...], "dates": [...], ...}
                            # Prophet needs future regressor values for prediction
                            future_regressors = None
                            if feature_columns:
                                # Align regressors to the expected future date index
                                # IMPORTANT: ProphetTrainer.make_future_dataframe(include_history=True) requires
                                # regressor values for history too. Provide (train history + future) to avoid
                                # bfill/ffill washing out regressors.
                                reg_source = pd.concat([group_train_df, eval_df], ignore_index=True)
                                reg = reg_source[[date_column] + feature_columns].copy()
                                reg[date_column] = pd.to_datetime(reg[date_column])
                                reg = reg.drop_duplicates(subset=[date_column], keep='last').set_index(date_column)
                                # Build full index = history dates + forecast dates
                                history_dates = pd.to_datetime(group_train_df[date_column], errors="coerce").dropna().sort_values().unique()
                                full_index = pd.DatetimeIndex(history_dates).append(pd.DatetimeIndex(future_dates))
                                reg = reg.reindex(full_index).ffill().bfill()
                                future_regressors = reg.reset_index().rename(columns={date_column: 'ds'})
                            result = trainer.predict(
                                horizon=eval_horizon,
                                future_regressors=future_regressors
                            )
                            predictions_all = np.array(result['predictions']) if isinstance(result, dict) else np.array(result)
                            pred_dates_all = pd.to_datetime(result.get('dates')) if isinstance(result, dict) and 'dates' in result else future_dates
                        elif model_lower == 'arima':
                            # ARIMA/SARIMAX: predict with optional exogenous features
                            # Prepare future exogenous variables if model was trained with them
                            exog_future = None
                            if feature_columns and hasattr(trainer, 'feature_columns') and trainer.feature_columns:
                                # Align exogenous features to the expected future date index
                                ex = eval_df[[date_column] + feature_columns].copy()
                                ex[date_column] = pd.to_datetime(ex[date_column])
                                ex = ex.drop_duplicates(subset=[date_column], keep='last').set_index(date_column)
                                ex = ex.reindex(future_dates).ffill().bfill()
                                exog_future = ex[feature_columns]
                                append_log(job_id, f"  Using exogenous features for ARIMA: {feature_columns}")
                                append_log(job_id, f"  Exog shape: {exog_future.shape}, Eval horizon: {eval_horizon} (val={val_len}, test={test_len})")
                            
                            result = trainer.predict(
                                horizon=eval_horizon, 
                                last_date=last_date, 
                                frequency='D',
                                exog_future=exog_future
                            )
                            predictions_all = np.array(result['predictions']) if isinstance(result, dict) else np.array(result)
                            pred_dates_all = pd.to_datetime(result.get('dates')) if isinstance(result, dict) and 'dates' in result else future_dates
                        elif model_lower in ['naive', 'seasonal_naive']:
                            # Naive: predict(horizon, last_date) -> {"predictions": [...], "dates": [...]}
                            result = trainer.predict(horizon=eval_horizon, last_date=last_date, frequency='D')
                            predictions_all = np.array(result['predictions']) if isinstance(result, dict) else np.array(result)
                            pred_dates_all = pd.to_datetime(result.get('dates')) if isinstance(result, dict) and 'dates' in result else future_dates
                        elif model_lower in ['xgboost', 'lightgbm']:
                            # XGBoost/LightGBM: These are NOT autoregressive models!
                            # They need the actual exogenous feature values from the test period.
                            # We combine train + val + test data, create features, then predict on test rows.
                            
                            # Ensure data is sorted chronologically
                            group_train_df = group_train_df.sort_values(date_column).reset_index(drop=True)
                            group_val_df = group_val_df.sort_values(date_column).reset_index(drop=True)
                            group_test_df = group_test_df.sort_values(date_column).reset_index(drop=True)
                            
                            # Combine train, val, and test data so lags flow correctly into test
                            combined_df = pd.concat([group_train_df, group_val_df, group_test_df], ignore_index=True)
                            combined_df = combined_df.sort_values(date_column).reset_index(drop=True)
                            
                            append_log(job_id, f"  Combined data: {len(combined_df)} rows (train: {len(group_train_df)}, val: {len(group_val_df)}, test: {len(group_test_df)})")
                            
                            # Create features on combined data (this ensures lags from train flow into test)
                            combined_features = trainer._create_features(
                                combined_df,
                                date_column,
                                target_column,
                                feature_columns,
                                None  # No group column here as we're already working with group-specific data
                            )
                            
                            append_log(job_id, f"  After feature engineering: {len(combined_features)} rows (lost {len(combined_df) - len(combined_features)} to NaN)")
                            
                            # Find where test data starts in the combined features
                            # Use date range to identify test portion
                            test_start_date = group_test_df[date_column].min()
                            test_mask = combined_features[date_column] >= test_start_date
                            test_features = combined_features[test_mask].copy()
                            
                            append_log(job_id, f"  Test features: {len(test_features)} rows")
                            
                            if len(test_features) == 0:
                                raise ValueError("No test features available after feature engineering")
                            
                            # Predict on test features
                            X_test = test_features[trainer.feature_names]
                            predictions = trainer.model.predict(X_test)
                            
                            # Get corresponding actuals (from the same rows in test_features)
                            actuals = test_features[target_column].values
                        else:
                            append_log(job_id, f"Unknown model type for prediction: {full_model_name}", "warning")
                            continue
                        
                        # For non-tree models, get actuals from group_test_df
                        if model_lower not in ['xgboost', 'lightgbm']:
                            # Align predictions to test dates by join (handles missing days safely)
                            test_dates = pd.to_datetime(group_test_df[date_column].values)
                            actual_series = pd.Series(group_test_df[target_column].values, index=test_dates).sort_index()

                            pred_series = pd.Series(
                                predictions_all[:len(pred_dates_all)],
                                index=pd.to_datetime(pred_dates_all[:len(predictions_all)]),
                            ).sort_index()

                            aligned = actual_series.to_frame('actual').join(pred_series.to_frame('pred'), how='inner')
                            if aligned.empty:
                                raise ValueError("No overlapping dates between predictions and test set (check date continuity/frequency)")

                            # Warn if we couldn't align all test points (usually indicates missing days)
                            if len(aligned) < len(actual_series):
                                append_log(job_id, f"  WARNING: Only {len(aligned)}/{len(actual_series)} test points aligned by date. Missing/irregular dates may exist.", "warning")

                            actuals = aligned['actual'].to_numpy()
                            predictions = aligned['pred'].to_numpy()

                            # Always log alignment + a small sample of the join to debug quickly
                            try:
                                head_sample = aligned.head(3).to_dict(orient="index")
                                tail_sample = aligned.tail(3).to_dict(orient="index")
                                append_log(job_id, f"  Alignment: matched_points={len(aligned)}, test_points={len(actual_series)}")
                                append_log(job_id, f"  Aligned sample head(3): {head_sample}")
                                append_log(job_id, f"  Aligned sample tail(3): {tail_sample}")
                            except Exception as sample_err:
                                append_log(job_id, f"  WARNING: Failed to log aligned samples: {sample_err}", "warning")
                        
                        # Ensure predictions and actuals have the same length
                        min_len = min(len(predictions), len(actuals))
                        predictions = predictions[:min_len]
                        actuals = actuals[:min_len]
                        
                        if len(predictions) == 0 or len(actuals) == 0:
                            raise ValueError("No predictions or actuals to evaluate")
                        
                        # Diagnostic logging for Prophet/ARIMA
                        if model_lower in ['prophet', 'arima']:
                            pred_mean = np.mean(predictions)
                            pred_std = np.std(predictions)
                            actual_mean = np.mean(actuals)
                            actual_std = np.std(actuals)
                            append_log(job_id, f"  {model_lower.upper()} Diagnostics:")
                            append_log(job_id, f"    Predictions: mean={pred_mean:.2f}, std={pred_std:.2f}, range=[{np.min(predictions):.2f}, {np.max(predictions):.2f}]")
                            append_log(job_id, f"    Actuals: mean={actual_mean:.2f}, std={actual_std:.2f}, range=[{np.min(actuals):.2f}, {np.max(actuals):.2f}]")
                            if pred_std < 10:
                                append_log(job_id, f"    WARNING: Predictions have very low variance (std={pred_std:.2f})!", "warning")
                        
                        # Calculate residuals
                        residuals = actuals - predictions
                        
                        # Calculate metrics with safety checks
                        mae = mean_absolute_error(actuals, predictions)
                        rmse = np.sqrt(mean_squared_error(actuals, predictions))
                        
                        # MAPE with zero-division protection
                        non_zero_mask = actuals != 0
                        if np.sum(non_zero_mask) > 0:
                            mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
                        else:
                            mape = 0.0
                        
                        # SMAPE with zero-division protection
                        denominator = np.abs(actuals) + np.abs(predictions)
                        non_zero_denom = denominator != 0
                        if np.sum(non_zero_denom) > 0:
                            smape = np.mean(2 * np.abs(predictions[non_zero_denom] - actuals[non_zero_denom]) / denominator[non_zero_denom]) * 100
                        else:
                            smape = 0.0
                        
                        r2 = r2_score(actuals, predictions)
                        medae = median_absolute_error(actuals, predictions)
                    
                        # MASE (Mean Absolute Scaled Error) - use training data for naive baseline
                        if len(group_train_df) > 1:
                            train_actuals = group_train_df[target_column].values
                            naive_error = np.mean(np.abs(train_actuals[1:] - train_actuals[:-1]))
                            mase = mae / naive_error if naive_error != 0 else 0
                        else:
                            mase = 0
                        
                        metrics = {
                            "mae": float(mae),
                            "rmse": float(rmse),
                            "mape": float(mape),
                            "smape": float(smape),
                            "r2": float(r2),
                            "medae": float(medae),
                            "mase": float(mase),
                            "residuals": residuals.tolist(),
                            "predictions": predictions.tolist(),
                            "actuals": actuals.tolist(),
                        }
                        
                        # Add feature importance for tree models
                        if model_lower in ['xgboost', 'lightgbm'] and hasattr(trainer, 'feature_names'):
                            feature_importance = dict(
                                zip(
                                    trainer.feature_names,
                                    trainer.model.feature_importances_.tolist(),
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
                            metrics["feature_importance"] = feature_importance
                        
                        group_label = f" ({group_value})" if group_value is not None else ""
                        append_log(job_id, f"{model_name}{group_label} metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
                    except Exception as metric_error:
                        append_log(job_id, f"Failed to calculate metrics for {full_model_name}: {str(metric_error)}", "warning")
                        logger.warning(f"Metric calculation error: {traceback.format_exc()}")
                    
                    results[full_model_name] = {
                        "modelType": full_model_name,
                        "model_path": model_path,
                        "metrics": metrics,
                        "group_value": str(group_value) if group_value is not None else None,
                        "train_size": len(group_train_df),
                        "val_size": len(group_val_df),
                        "test_size": len(group_test_df),
                    }
                    
                    append_log(job_id, f"{full_model_name} trained successfully. Model saved to: {model_path}")
                    
                except Exception as e:
                    append_log(job_id, f"Failed to train {full_model_name}: {str(e)}", "error")
                    logger.error(f"Error training {full_model_name}: {traceback.format_exc()}")
        
        # Convert results dict to array for easier processing in backend
        results_array = list(results.values())
        
        # Store results in Redis for NestJS to pick up
        result_key = f"ml:job:result:{job_id}"
        redis_client.set(result_key, json.dumps(results_array), ex=3600)  # Expire after 1 hour
        
        append_log(job_id, "Training completed successfully!")
        publish_progress(job_id, "completed", "Training completed", 100, {"results": results_array})
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        append_log(job_id, error_msg, "error")
        publish_progress(job_id, "error", error_msg, 0)
        logger.error(f"Error processing job {job_id}: {traceback.format_exc()}")


def process_forecast_job(job_data: Dict[str, Any]):
    """Process a forecast generation job."""
    job_id = job_data.get("jobId")
    forecast_id = job_data.get("forecastId")
    
    try:
        logger.info(f"[{job_id}] Starting forecast generation")
        publish_progress(job_id, "running", "Starting forecast generation", 10, {
            "jobType": "forecast",
            "forecastId": forecast_id
        })
        
        # Extract job parameters
        model_path = job_data.get("modelPath")
        dataset_path = job_data.get("datasetPath")
        date_column = job_data.get("dateColumn")
        target_column = job_data.get("targetColumn")
        horizon = job_data.get("horizon", 30)
        frequency = job_data.get("frequency", "D")
        
        logger.info(f"[{job_id}] Forecast params: model={model_path}, horizon={horizon}")
        
        # Import forecast service
        from services.forecast_service import ForecastService
        
        models_path = os.getenv("MODELS_PATH", "/shared/models")
        forecast_service = ForecastService(models_path)
        
        publish_progress(job_id, "running", "Generating predictions", 50, {
            "jobType": "forecast",
            "forecastId": forecast_id
        })
        
        # Generate forecast
        result = forecast_service.generate_forecast(
            model_path=model_path,
            data_path=dataset_path,
            date_column=date_column,
            target_column=target_column,
            horizon=horizon,
            frequency=frequency,
        )
        
        logger.info(f"[{job_id}] Forecast generated successfully")
        publish_progress(job_id, "running", "Forecast completed", 90, {
            "jobType": "forecast",
            "forecastId": forecast_id
        })
        
        # Publish completion with results
        event_data = {
            "event": "progress",
            "jobId": job_id,
            "data": {
                "status": "completed",
                "message": "Forecast generation completed",
                "percentage": 100,
                "jobType": "forecast",
                "forecastId": forecast_id,
                "results": result,
                "predictions": result.get("predictions", [])
            }
        }
        redis_client.publish(ML_JOBS_UPDATE_CHANNEL, json.dumps(event_data))
        
        logger.info(f"[{job_id}] INFO: Forecast completed successfully!")
        
    except Exception as e:
        error_msg = f"Forecast generation failed: {str(e)}"
        logger.error(f"[{job_id}] ERROR: {error_msg}")
        logger.error(traceback.format_exc())
        
        publish_progress(
            job_id,
            "error",
            error_msg,
            0,
            {
                "jobType": "forecast",
                "forecastId": forecast_id
            }
        )


def process_eda_job(job_data: Dict[str, Any]):
    """Process EDA (Exploratory Data Analysis) job."""
    from services.eda_service import EDAService
    
    job_id = job_data.get("jobId")
    eda_report_id = job_data.get("edaReportId")
    dataset_path = job_data.get("datasetPath")
    date_column = job_data.get("dateColumn")
    target_column = job_data.get("targetColumn")
    feature_columns = job_data.get("featureColumns", [])
    
    logger.info(f"[{job_id}] Starting EDA analysis")
    
    try:
        # Publish progress: starting
        publish_progress(
            job_id,
            "running",
            "Starting EDA analysis",
            10
        )
        
        # Run EDA analysis
        logger.info(f"[{job_id}] Analyzing dataset: {dataset_path}")
        eda_service = EDAService()
        
        start_time = time.time()
        eda_results = eda_service.analyze(
            dataset_path=dataset_path,
            date_column=date_column,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        processing_time = time.time() - start_time
        
        # Add processing time to metadata
        if 'metadata' not in eda_results:
            eda_results['metadata'] = {}
        eda_results['metadata']['processingTime'] = processing_time
        
        # Publish progress: completed
        publish_progress(
            job_id,
            "completed",
            f"EDA analysis completed in {processing_time:.2f}s",
            100,
            {
                "edaReportId": eda_report_id,
                "results": eda_results,
            }
        )
        
        logger.info(f"[{job_id}] EDA analysis completed successfully")
        
    except Exception as e:
        error_msg = f"EDA analysis failed: {str(e)}"
        logger.error(f"[{job_id}] ERROR: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Publish error
        publish_progress(
            job_id,
            "error",
            error_msg,
            0,
            {
                "edaReportId": eda_report_id,
                "error": error_msg,
            }
        )


def process_preprocessing_job(job_data: Dict[str, Any]):
    """Process data preprocessing job."""
    from services.preprocessing_service import PreprocessingService
    
    job_id = job_data.get("jobId")
    preprocessing_id = job_data.get("preprocessingId")
    dataset_path = job_data.get("datasetPath")
    date_column = job_data.get("dateColumn")
    target_column = job_data.get("targetColumn")
    config = job_data.get("config", {})
    
    logger.info(f"[{job_id}] Starting data preprocessing")
    
    try:
        # Publish progress: starting
        publish_progress(
            job_id,
            "processing",
            "Starting data preprocessing",
            10
        )
        
        # Run preprocessing
        logger.info(f"[{job_id}] Preprocessing dataset: {dataset_path}")
        preprocessing_service = PreprocessingService()
        
        start_time = time.time()
        results = preprocessing_service.preprocess(
            dataset_path=dataset_path,
            date_column=date_column,
            target_column=target_column,
            config=config,
        )
        processing_time = time.time() - start_time
        
        # Add processing time to results
        results['results']['processingTime'] = processing_time
        
        # Publish progress: completed
        publish_progress(
            job_id,
            "completed",
            f"Preprocessing completed in {processing_time:.2f}s",
            100,
            {
                "preprocessingId": preprocessing_id,
                "results": results,
            }
        )
        
        logger.info(f"[{job_id}] Preprocessing completed successfully")
        
    except Exception as e:
        error_msg = f"Preprocessing failed: {str(e)}"
        logger.error(f"[{job_id}] ERROR: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Publish error
        publish_progress(
            job_id,
            "error",
            error_msg,
            0,
            {
                "preprocessingId": preprocessing_id,
                "error": error_msg,
            }
        )


def process_evaluation_job(job_data: Dict[str, Any]):
    """Process an evaluation job."""
    job_id = job_data.get("jobId")
    evaluation_id = job_data.get("evaluationId")
    
    try:
        append_log(job_id, "Starting model evaluation...")
        publish_progress(job_id, "running", "Initializing evaluation...", 10)
        
        # Import evaluation service
        from services.evaluation_service import EvaluationService
        
        # Extract job parameters
        project_id = job_data.get("projectId")
        ranking_metric = job_data.get("rankingMetric", "mape")
        training_results = job_data.get("trainingResults", [])
        
        append_log(job_id, f"Evaluating {len(training_results)} models...")
        publish_progress(job_id, "running", "Calculating metrics...", 30)
        
        # Run evaluation
        evaluation_service = EvaluationService()
        start_time = time.time()
        
        results = evaluation_service.evaluate_models(
            project_id=project_id,
            training_results=training_results,
            ranking_metric=ranking_metric,
        )
        
        processing_time = time.time() - start_time
        results['metadata']['processingTime'] = processing_time
        
        append_log(job_id, f"Evaluation completed in {processing_time:.2f}s")
        
        # Publish completion
        publish_progress(
            job_id,
            "completed",
            f"Evaluation completed. Best model: {results['bestModel']}",
            100,
            {
                "evaluationId": evaluation_id,
                "bestModel": results['bestModel'],
                "modelResults": results['modelResults'],
                "summary": results['summary'],
                "metadata": results['metadata'],
            }
        )
        
        logger.info(f"[{job_id}] Evaluation completed successfully")
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        logger.error(f"[{job_id}] ERROR: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Publish error
        publish_progress(
            job_id,
            "error",
            error_msg,
            0,
            {
                "evaluationId": evaluation_id,
                "error": error_msg,
            }
        )


def poll_queue():
    """Poll Redis queue for jobs."""
    logger.info("Starting Redis worker - polling for jobs...")
    logger.info(f"Listening on queues: {ML_JOBS_QUEUE}, {ML_FORECAST_JOBS_QUEUE}, {ML_EDA_JOBS_QUEUE}, {ML_PREPROCESSING_JOBS_QUEUE}, {ML_EVALUATION_JOBS_QUEUE}")
    
    while True:
        try:
            # Block for 1 second waiting for a job from any queue
            job_raw = redis_client.brpop([ML_JOBS_QUEUE, ML_FORECAST_JOBS_QUEUE, ML_EDA_JOBS_QUEUE, ML_PREPROCESSING_JOBS_QUEUE, ML_EVALUATION_JOBS_QUEUE], timeout=1)
            
            if job_raw is None:
                continue
            
            # Parse job
            queue_name, job_json = job_raw
            job_data = json.loads(job_json)
            job_id = job_data.get("jobId")
            
            logger.info(f"Received job from {queue_name}: {job_id}")
            
            # Route to appropriate handler based on queue
            if queue_name == ML_JOBS_QUEUE:
                # Fetch full job data from Redis (for training jobs)
                job_data_key = f"{ML_JOB_DATA_PREFIX}{job_id}"
                job_data_json = redis_client.get(job_data_key)
                
                if not job_data_json:
                    logger.error(f"Job data not found for job {job_id}")
                    continue
                
                job_data = json.loads(job_data_json)
                process_training_job(job_data)
            elif queue_name == ML_FORECAST_JOBS_QUEUE:
                process_forecast_job(job_data)
            elif queue_name == ML_EDA_JOBS_QUEUE:
                process_eda_job(job_data)
            elif queue_name == ML_PREPROCESSING_JOBS_QUEUE:
                process_preprocessing_job(job_data)
            elif queue_name == ML_EVALUATION_JOBS_QUEUE:
                process_evaluation_job(job_data)
            else:
                logger.error(f"Unknown queue: {queue_name}")
            
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in worker loop: {e}")
            logger.error(traceback.format_exc())
            time.sleep(1)  # Avoid tight loop on persistent errors


if __name__ == "__main__":
    poll_queue()

