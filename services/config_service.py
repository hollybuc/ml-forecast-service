"""Configuration service for base config auto-detection and validation."""

import pandas as pd
from typing import Tuple, Optional, List, Dict

from backend.core.validators import DataValidator
from backend.core.exceptions import DataValidationError, ConfigurationError
from backend.core.logging import get_logger
from backend.models import BaseConfig

logger = get_logger(__name__)


class ConfigService:
    """Service for handling configuration logic."""

    def __init__(self):
        """Initialize config service."""
        self.validator = DataValidator()

    def auto_detect_config(self, df: pd.DataFrame) -> Dict:
        """Auto-detect base configuration from DataFrame.

        Args:
            df: Pandas DataFrame with uploaded data.

        Returns:
            Dictionary with auto-detected configuration suggestions.

        Raises:
            DataValidationError: If auto-detection fails.
        """
        logger.info("Starting auto-detection of base configuration")

        try:
            # Detect datetime columns
            datetime_columns = self.validator.detect_datetime_columns(df)
            logger.info(f"Detected datetime columns: {datetime_columns}")

            if not datetime_columns:
                raise DataValidationError(
                    "No datetime columns found in data. At least one datetime column is required.",
                    details={"available_columns": df.columns.tolist()},
                )

            # Detect numeric columns
            numeric_columns = self.validator.detect_numeric_columns(df)
            logger.info(f"Detected numeric columns: {numeric_columns}")

            if not numeric_columns:
                raise DataValidationError(
                    "No numeric columns found in data. At least one numeric column is required for target variable.",
                    details={"available_columns": df.columns.tolist()},
                )

            # Auto-select best date column (prefer columns with 'date', 'time', 'timestamp' in name)
            date_column = self._select_best_date_column(datetime_columns, df)

            # Auto-select best target column (prefer columns with 'target', 'value', 'sales', 'y' in name)
            target_column = self._select_best_target_column(numeric_columns, df)

            # Detect frequency
            frequency = self.validator.detect_frequency(df, date_column)
            logger.info(f"Detected frequency: {frequency}")

            # Detect potential group columns (categorical with moderate cardinality)
            group_column_suggestions = self._detect_group_columns(df)
            logger.info(f"Suggested group columns: {group_column_suggestions}")

            # Detect potential feature columns (exclude date and target)
            feature_column_suggestions = self._detect_feature_columns(
                df, date_column, target_column
            )
            logger.info(f"Suggested feature columns: {feature_column_suggestions}")

            auto_config = {
                "date_column": date_column,
                "target_column": target_column,
                "frequency": frequency,
                "group_column": group_column_suggestions[0] if group_column_suggestions else None,
                "feature_columns": feature_column_suggestions,
                "all_columns": {
                    "datetime_columns": datetime_columns,
                    "numeric_columns": numeric_columns,
                    "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
                },
            }

            logger.info("Auto-detection completed successfully")
            return auto_config

        except Exception as e:
            logger.error(f"Auto-detection failed: {str(e)}")
            raise

    def validate_config(
        self,
        df: pd.DataFrame,
        config: BaseConfig,
    ) -> Tuple[bool, List[str]]:
        """Validate base configuration.

        Args:
            df: Pandas DataFrame with uploaded data.
            config: Base configuration to validate.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        logger.info(f"Validating base configuration: {config}")
        errors = []

        # Validate date column
        is_valid, error_msg = self.validator.validate_date_column(df, config.date_column)
        if not is_valid:
            errors.append(error_msg)

        # Validate target column
        is_valid, error_msg = self.validator.validate_target_column(df, config.target_column)
        if not is_valid:
            errors.append(error_msg)

        # Validate group column (if provided)
        if config.group_column:
            is_valid, error_msg = self.validator.validate_column_exists(df, config.group_column)
            if not is_valid:
                errors.append(error_msg)

        # Validate feature columns (if provided)
        for feature_col in config.feature_columns:
            is_valid, error_msg = self.validator.validate_column_exists(df, feature_col)
            if not is_valid:
                errors.append(error_msg)

        # Validate that group_column is not in feature_columns
        if config.group_column and config.group_column in config.feature_columns:
            errors.append(
                f"Group column '{config.group_column}' cannot be used as a feature column. "
                f"Please remove it from feature columns."
            )

        # Validate that date_column and target_column are not in feature_columns
        if config.date_column in config.feature_columns:
            errors.append(
                f"Date column '{config.date_column}' cannot be used as a feature column. "
                f"Please remove it from feature columns."
            )
        
        if config.target_column in config.feature_columns:
            errors.append(
                f"Target column '{config.target_column}' cannot be used as a feature column. "
                f"Please remove it from feature columns."
            )

        # Check for duplicate dates
        has_duplicates, duplicate_indices = self.validator.check_duplicates(
            df, config.date_column, config.group_column
        )
        if has_duplicates:
            errors.append(
                f"Found {len(duplicate_indices)} duplicate dates"
                + (f" (per group)" if config.group_column else "")
            )

        # Validate frequency
        if config.frequency not in ["H", "D", "W", "M", "Q", "Y"]:
            errors.append(f"Invalid frequency: {config.frequency}. Must be one of: H, D, W, M, Q, Y")

        is_valid = len(errors) == 0
        logger.info(f"Validation result: {'PASSED' if is_valid else 'FAILED'}")

        if not is_valid:
            logger.warning(f"Validation errors: {errors}")

        return is_valid, errors

    def _select_best_date_column(self, datetime_columns: List[str], df: pd.DataFrame) -> str:
        """Select the best date column from available datetime columns.

        Args:
            datetime_columns: List of datetime column names.
            df: Pandas DataFrame.

        Returns:
            Best date column name.
        """
        # Priority keywords for date column
        date_keywords = ["date", "datetime", "timestamp", "time", "ds"]

        # Check for columns with priority keywords
        for keyword in date_keywords:
            for col in datetime_columns:
                if keyword in col.lower():
                    logger.info(f"Selected date column '{col}' based on keyword '{keyword}'")
                    return col

        # If no priority keyword found, return first datetime column
        logger.info(f"Selected date column '{datetime_columns[0]}' (first available)")
        return datetime_columns[0]

    def _select_best_target_column(self, numeric_columns: List[str], df: pd.DataFrame) -> str:
        """Select the best target column from available numeric columns.

        Args:
            numeric_columns: List of numeric column names.
            df: Pandas DataFrame.

        Returns:
            Best target column name.
        """
        # Priority keywords for target column
        target_keywords = ["target", "y", "value", "sales", "revenue", "amount", "qty", "quantity"]

        # Check for columns with priority keywords
        for keyword in target_keywords:
            for col in numeric_columns:
                if keyword in col.lower():
                    logger.info(f"Selected target column '{col}' based on keyword '{keyword}'")
                    return col

        # If no priority keyword found, return first numeric column
        logger.info(f"Selected target column '{numeric_columns[0]}' (first available)")
        return numeric_columns[0]

    def _detect_group_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect potential group columns (categorical with moderate cardinality).

        Args:
            df: Pandas DataFrame.

        Returns:
            List of potential group column names.
        """
        group_cols = []

        # Get categorical columns (exclude datetime)
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            # Skip if column looks like a date
            if any(keyword in col.lower() for keyword in ["date", "time", "timestamp"]):
                logger.debug(f"Skipping '{col}' - looks like a date column")
                continue

            # Check cardinality (number of unique values)
            n_unique = df[col].nunique()
            n_total = len(df)

            # Good group column: 2-50 unique values, or <5% cardinality
            if 2 <= n_unique <= 50 or (n_unique / n_total) < 0.05:
                group_cols.append(col)
                logger.debug(f"Column '{col}' is a potential group column (cardinality: {n_unique})")

        return group_cols

    def _detect_feature_columns(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
    ) -> List[str]:
        """Detect potential feature columns.

        Args:
            df: Pandas DataFrame.
            date_column: Date column name.
            target_column: Target column name.

        Returns:
            List of potential feature column names.
        """
        # Exclude date and target columns
        exclude_cols = {date_column, target_column}

        # Get numeric and categorical columns (potential features)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        all_feature_cols = numeric_cols + categorical_cols

        # Filter out excluded columns
        feature_cols = [col for col in all_feature_cols if col not in exclude_cols]

        logger.debug(f"Detected {len(feature_cols)} potential feature columns")
        return feature_cols


# Global config service instance
config_service = ConfigService()


def get_config_service() -> ConfigService:
    """Get config service instance (for dependency injection).

    Returns:
        ConfigService instance.
    """
    return config_service
