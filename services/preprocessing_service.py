"""
Data Preprocessing Service

Handles various data preprocessing tasks including:
- Missing value handling
- Outlier detection and treatment
- Data transformations
- Feature engineering
- Differencing for stationarity
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.ensemble import IsolationForest
import logging
import os

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Service for preprocessing time series data"""

    def preprocess(
        self,
        dataset_path: str,
        date_column: str,
        target_column: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Preprocess the dataset according to the configuration

        Args:
            dataset_path: Path to the CSV file
            date_column: Name of the date column
            target_column: Name of the target variable column
            config: Preprocessing configuration

        Returns:
            Dictionary with preprocessing results and processed dataset path
        """
        logger.info(f"Starting preprocessing for: {dataset_path}")

        # Load data
        df = pd.read_csv(dataset_path)
        original_row_count = len(df)
        logger.info(f"Loaded {original_row_count} rows")

        # Parse dates
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        # Track changes
        missing_values_handled = 0
        outliers_handled = 0
        features_created = []

        # 1. Handle missing values
        if config.get('missingValueStrategy'):
            df, missing_count = self._handle_missing_values(
                df, target_column, config['missingValueStrategy']
            )
            missing_values_handled = missing_count

        # 2. Handle outliers
        if config.get('outlierStrategy') and config['outlierStrategy']['method'] != 'none':
            df, outlier_count = self._handle_outliers(
                df, target_column, config['outlierStrategy']
            )
            outliers_handled = outlier_count

        # 3. Apply transformation
        if config.get('transformation') and config['transformation']['method'] != 'none':
            df = self._apply_transformation(
                df, target_column, config['transformation']
            )

        # 4. Feature engineering
        if config.get('featureEngineering'):
            df, new_features = self._engineer_features(
                df, date_column, target_column, config['featureEngineering']
            )
            features_created = new_features

        # 5. Apply differencing
        if config.get('differencing') and config['differencing'].get('enabled'):
            df = self._apply_differencing(
                df, target_column, config['differencing']
            )

        # Save processed dataset
        processed_filename = f"processed_{os.path.basename(dataset_path)}"
        processed_path = os.path.join('/shared/datasets', processed_filename)
        df.to_csv(processed_path, index=False)
        logger.info(f"Processed dataset saved to: {processed_path}")

        processed_row_count = len(df)
        removed_rows = original_row_count - processed_row_count

        return {
            'processedDatasetPath': processed_path,
            'results': {
                'originalRowCount': original_row_count,
                'processedRowCount': processed_row_count,
                'removedRows': removed_rows,
                'missingValuesHandled': missing_values_handled,
                'outliersHandled': outliers_handled,
                'featuresCreated': features_created,
            },
        }

    def _handle_missing_values(
        self, df: pd.DataFrame, target_column: str, strategy: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Handle missing values according to strategy"""
        method = strategy.get('method', 'forward_fill')
        columns = strategy.get('columns', [target_column])

        missing_count = df[columns].isna().sum().sum()

        if missing_count == 0:
            return df, 0

        logger.info(f"Handling {missing_count} missing values using method: {method}")

        if method == 'drop':
            df = df.dropna(subset=columns)
        elif method == 'forward_fill':
            df[columns] = df[columns].fillna(method='ffill')
        elif method == 'backward_fill':
            df[columns] = df[columns].fillna(method='bfill')
        elif method == 'interpolate':
            df[columns] = df[columns].interpolate(method='linear')
        elif method == 'mean':
            for col in columns:
                df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            for col in columns:
                df[col] = df[col].fillna(df[col].median())

        return df, missing_count

    def _handle_outliers(
        self, df: pd.DataFrame, target_column: str, strategy: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, int]:
        """Handle outliers according to strategy"""
        method = strategy.get('method', 'iqr')
        action = strategy.get('action', 'remove')
        threshold = strategy.get('threshold', 1.5)

        values = df[target_column].values
        outlier_mask = np.zeros(len(values), dtype=bool)

        if method == 'iqr':
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (values < lower_bound) | (values > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            outlier_mask = z_scores > threshold

        elif method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_predictions = iso_forest.fit_predict(values.reshape(-1, 1))
            outlier_mask = outlier_predictions == -1

        outlier_count = np.sum(outlier_mask)
        logger.info(f"Detected {outlier_count} outliers using method: {method}")

        if outlier_count == 0:
            return df, 0

        if action == 'remove':
            df = df[~outlier_mask]
        elif action == 'clip':
            if method == 'iqr':
                df.loc[outlier_mask, target_column] = np.clip(
                    df.loc[outlier_mask, target_column], lower_bound, upper_bound
                )
        elif action == 'winsorize':
            lower_percentile = np.percentile(values, 5)
            upper_percentile = np.percentile(values, 95)
            df.loc[outlier_mask, target_column] = np.clip(
                df.loc[outlier_mask, target_column],
                lower_percentile,
                upper_percentile,
            )

        return df, outlier_count

    def _apply_transformation(
        self, df: pd.DataFrame, target_column: str, transformation: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply transformation to data"""
        method = transformation.get('method', 'none')
        columns = transformation.get('columns', [target_column])

        if method == 'none':
            return df

        logger.info(f"Applying transformation: {method}")

        for col in columns:
            values = df[col].values

            if method == 'log':
                # Add small constant to avoid log(0)
                df[col] = np.log1p(values)
            elif method == 'sqrt':
                df[col] = np.sqrt(np.abs(values)) * np.sign(values)
            elif method == 'box_cox':
                # Box-Cox requires positive values
                if np.all(values > 0):
                    df[col] = boxcox1p(values, 0)
                else:
                    logger.warning(f"Box-Cox requires positive values for {col}, skipping")
            elif method == 'yeo_johnson':
                pt = PowerTransformer(method='yeo-johnson')
                df[col] = pt.fit_transform(values.reshape(-1, 1)).flatten()
            elif method == 'standardize':
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(values.reshape(-1, 1)).flatten()
            elif method == 'normalize':
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(values.reshape(-1, 1)).flatten()

        return df

    def _engineer_features(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Engineer new features"""
        features_created = []

        # Create lag features
        if config.get('createLags'):
            lag_periods = config.get('lagPeriods', [1, 7, 14])
            for lag in lag_periods:
                feature_name = f'{target_column}_lag_{lag}'
                df[feature_name] = df[target_column].shift(lag)
                features_created.append(feature_name)
            logger.info(f"Created {len(lag_periods)} lag features")

        # Create rolling features
        if config.get('createRollingFeatures'):
            windows = config.get('rollingWindows', [7, 14, 30])
            for window in windows:
                # Rolling mean
                feature_name = f'{target_column}_rolling_mean_{window}'
                df[feature_name] = df[target_column].rolling(window=window).mean()
                features_created.append(feature_name)

                # Rolling std
                feature_name = f'{target_column}_rolling_std_{window}'
                df[feature_name] = df[target_column].rolling(window=window).std()
                features_created.append(feature_name)
            logger.info(f"Created {len(windows) * 2} rolling features")

        # Create date features
        if config.get('createDateFeatures'):
            date_features = config.get('dateFeatures', [
                'day_of_week', 'month', 'quarter', 'year', 'is_weekend'
            ])

            if 'day_of_week' in date_features:
                df['day_of_week'] = df[date_column].dt.dayofweek
                features_created.append('day_of_week')

            if 'month' in date_features:
                df['month'] = df[date_column].dt.month
                features_created.append('month')

            if 'quarter' in date_features:
                df['quarter'] = df[date_column].dt.quarter
                features_created.append('quarter')

            if 'year' in date_features:
                df['year'] = df[date_column].dt.year
                features_created.append('year')

            if 'is_weekend' in date_features:
                df['is_weekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)
                features_created.append('is_weekend')

            logger.info(f"Created {len(date_features)} date features")

        # Drop rows with NaN values created by feature engineering
        df = df.dropna()

        return df, features_created

    def _apply_differencing(
        self, df: pd.DataFrame, target_column: str, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply differencing to make series stationary"""
        order = config.get('order', 1)
        seasonal = config.get('seasonal', False)
        seasonal_period = config.get('seasonalPeriod', 7)

        logger.info(f"Applying differencing of order {order}")

        # Apply regular differencing
        for i in range(order):
            df[f'{target_column}_diff_{i+1}'] = df[target_column].diff()

        # Apply seasonal differencing
        if seasonal:
            logger.info(f"Applying seasonal differencing with period {seasonal_period}")
            df[f'{target_column}_seasonal_diff'] = df[target_column].diff(seasonal_period)

        # Drop NaN values created by differencing
        df = df.dropna()

        return df
