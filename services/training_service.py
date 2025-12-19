"""Training service for data preparation and split."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

from backend.core.logging import get_logger
from backend.core.exceptions import DataValidationError
from backend.core.database import DuckDBManager
from backend.models import BaseConfig, SplitConfig

logger = get_logger(__name__)


class TrainingService:
    """Service for preparing data for model training."""

    def __init__(self):
        """Initialize training service."""
        pass

    def prepare_training_data(
        self,
        db: DuckDBManager,
        base_config: BaseConfig,
        split_config: SplitConfig,
        model_id: str,
    ) -> Dict[str, Any]:
        """Prepare data for model training.

        Args:
            db: DuckDB manager instance.
            base_config: Base configuration.
            split_config: Split configuration.
            model_id: Model identifier.

        Returns:
            Dictionary with prepared data and metadata.

        Raises:
            DataValidationError: If data preparation fails.
        """
        logger.info(f"Preparing training data for model: {model_id}")

        try:
            # Load processed data
            if not db.table_exists("processed_data"):
                raise DataValidationError(
                    "No processed data found. Please complete preprocessing first.",
                    details={"missing_table": "processed_data"},
                )

            df = db.fetch_df("SELECT * FROM processed_data")
            logger.info(f"Loaded {len(df)} rows of processed data")

            # Parse date column
            df[base_config.date_column] = pd.to_datetime(df[base_config.date_column])

            # Sort by date
            df = df.sort_values(base_config.date_column)

            # Split data
            # Split data
            if base_config.group_column and base_config.group_column in df.columns:
                train_df, val_df, test_df = self._split_data_grouped(
                    df, split_config, base_config.group_column
                )
            else:
                train_df, val_df, test_df = self._split_data(df, split_config)

            logger.info(
                f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )

            # Prepare result
            result = {
                "train": train_df,
                "val": val_df,
                "test": test_df,
                "full": df,
                "metadata": {
                    "n_train": len(train_df),
                    "n_val": len(val_df),
                    "n_test": len(test_df),
                    "n_total": len(df),
                    "date_column": base_config.date_column,
                    "target_column": base_config.target_column,
                    "feature_columns": base_config.feature_columns,
                    "group_column": base_config.group_column,
                    "frequency": base_config.frequency,
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise DataValidationError(
                "Failed to prepare training data",
                details={"error": str(e), "model_id": model_id},
            )

    def _split_data(
        self, df: pd.DataFrame, split_config: SplitConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets (Global Split).

        Args:
            df: DataFrame to split.
            split_config: Split configuration.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        if split_config.method == "temporal":
            return self._temporal_split(df, split_config)
        else:
            # Cross-validation not implemented yet
            raise NotImplementedError("Cross-validation split not implemented yet")

    def _split_data_grouped(
        self, df: pd.DataFrame, split_config: SplitConfig, group_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets Per Group (Local Split).

        Args:
            df: DataFrame to split.
            split_config: Split configuration.
            group_col: Group column name.

        Returns:
            Tuple of concatenated (train_df, val_df, test_df).
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Iterate over each group and split locally
        for _, group_df in df.groupby(group_col):
            # Sort individual group by date just to be safe (though global sort should've handled it)
            # We assume df coming in is already sorted globally, which usually implies local sort too if stable,
            # but usually explicit sort is safer or assumed.
            # group_df is a view/copy.

            # Apply standard split to this group's subset
            if split_config.method == "temporal":
                tr, va, te = self._temporal_split(group_df, split_config)
                train_dfs.append(tr)
                val_dfs.append(va)
                test_dfs.append(te)
            else:
                raise NotImplementedError("Cross-validation split not implemented yet")

        # Concatenate all parts
        # If any list is empty (e.g. no groups?), create empty DFs
        if not train_dfs:
            empty = pd.DataFrame(columns=df.columns)
            return empty, empty, empty

        train_all = pd.concat(train_dfs).sort_index()
        val_all = (
            pd.concat(val_dfs).sort_index()
            if val_dfs
            else pd.DataFrame(columns=df.columns)
        )
        test_all = pd.concat(test_dfs).sort_index()

        return train_all, val_all, test_all

    def _temporal_split(
        self, df: pd.DataFrame, split_config: SplitConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal train/val/test split.

        Args:
            df: DataFrame to split.
            split_config: Split configuration.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        n = len(df)

        # Calculate split indices
        train_end = int(n * split_config.train_size)
        val_end = train_end + int(n * split_config.val_size)

        # Split
        train_df = df.iloc[:train_end].copy()
        val_df = (
            df.iloc[train_end:val_end].copy()
            if split_config.val_size > 0
            else pd.DataFrame()
        )
        test_df = df.iloc[val_end:].copy()

        # Only log for large global chunks, might be noisy for groups
        # logger.info(f"Temporal split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df

    def save_training_results(
        self,
        db: DuckDBManager,
        model_id: str,
        results: Dict[str, Any],
    ):
        """Save training results to database.

        Args:
            db: DuckDB manager instance.
            model_id: Model identifier.
            results: Training results dictionary.
        """
        logger.info(f"Saving training results for model: {model_id}")

        try:
            import json
            import numpy as np

            # Create table if not exists
            db.execute("""
                CREATE TABLE IF NOT EXISTS training_results (
                    model_id VARCHAR,
                    timestamp TIMESTAMP,
                    results JSON,
                    PRIMARY KEY (model_id, timestamp)
                )
            """)

            # Convert numpy types to native Python types
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            # Save results
            results_clean = convert_numpy_types(results)
            results_json = json.dumps(results_clean)
            timestamp = datetime.now().isoformat()

            db.execute(
                """
                INSERT INTO training_results (model_id, timestamp, results)
                VALUES (?, ?, ?)
                """,
                (model_id, timestamp, results_json),
            )

            logger.info(f"Training results saved for model: {model_id}")

        except Exception as e:
            logger.error(f"Error saving training results: {str(e)}")
            raise DataValidationError(
                "Failed to save training results",
                details={"error": str(e), "model_id": model_id},
            )


# Global service instance
training_service = TrainingService()


def get_training_service() -> TrainingService:
    """Get training service instance (for dependency injection).

    Returns:
        TrainingService instance.
    """
    return training_service
