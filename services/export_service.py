"""Export service for exporting forecasts, models, and configurations."""

import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from io import BytesIO, StringIO

from backend.core.logging import get_logger
from backend.core.exceptions import DataValidationError, ResourceNotFoundError
from backend.core.database import DuckDBManager
from backend.models import ForecastResult

logger = get_logger(__name__)


class ExportService:
    """Service for exporting forecasts, models, and configurations."""

    def __init__(self):
        """Initialize export service."""
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def export_forecast_to_csv(
        self,
        db: DuckDBManager,
        include_history: bool = False,
        include_intervals: bool = True,
        include_all_models: bool = False,
    ) -> str:
        """Export forecast to CSV format.

        Args:
            db: DuckDB manager instance.
            include_history: Include historical data.
            include_intervals: Include confidence intervals.
            include_all_models: Include all models or just best.

        Returns:
            CSV string.

        Raises:
            ResourceNotFoundError: If forecast data not found.
        """
        logger.info("Exporting forecast to CSV")

        # Load forecast data
        forecasts = self._load_forecast_data(db)

        # Convert to DataFrame
        df = self._forecasts_to_dataframe(
            forecasts,
            db=db,
            include_intervals=include_intervals,
            include_all_models=include_all_models,
        )

        # Include history if requested
        if include_history:
            history_df = self._load_historical_data(db)
            df = pd.concat([history_df, df], ignore_index=True)

        # Convert to CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        logger.info(f"Exported {len(df)} rows to CSV")
        return csv_content

    def export_forecast_to_excel(
        self,
        db: DuckDBManager,
        include_history: bool = False,
        include_intervals: bool = True,
        include_all_models: bool = False,
    ) -> bytes:
        """Export forecast to Excel format.

        Args:
            db: DuckDB manager instance.
            include_history: Include historical data.
            include_intervals: Include confidence intervals.
            include_all_models: Include all models or just best.

        Returns:
            Excel file as bytes.

        Raises:
            ResourceNotFoundError: If forecast data not found.
        """
        logger.info("Exporting forecast to Excel")

        # Load forecast data
        forecasts = self._load_forecast_data(db)

        # Convert to DataFrame
        forecast_df = self._forecasts_to_dataframe(
            forecasts,
            db=db,
            include_intervals=include_intervals,
            include_all_models=include_all_models,
        )

        # Create Excel file with multiple sheets
        excel_buffer = BytesIO()

        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            # Forecast sheet
            forecast_df.to_excel(writer, sheet_name="Forecast", index=False)

            # History sheet if requested
            if include_history:
                history_df = self._load_historical_data(db)
                history_df.to_excel(writer, sheet_name="History", index=False)

            # Metadata sheet
            metadata_df = self._create_metadata_sheet(db, forecasts)
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

        excel_content = excel_buffer.getvalue()
        logger.info(f"Exported forecast to Excel ({len(excel_content)} bytes)")
        return excel_content

    def export_forecast_to_json(
        self,
        db: DuckDBManager,
        include_history: bool = False,
        include_intervals: bool = True,
        include_all_models: bool = False,
    ) -> str:
        """Export forecast to JSON format.

        Args:
            db: DuckDB manager instance.
            include_history: Include historical data.
            include_intervals: Include confidence intervals.
            include_all_models: Include all models or just best.

        Returns:
            JSON string.

        Raises:
            ResourceNotFoundError: If forecast data not found.
        """
        logger.info("Exporting forecast to JSON")

        # Load forecast data
        forecasts = self._load_forecast_data(db)

        # Build JSON structure
        export_data = {
            "forecasts": [],
            "metadata": self._get_export_metadata(db),
        }

        # Add forecast data
        for forecast in forecasts:
            if not include_all_models and forecast.get(
                "model_name"
            ) != self._get_best_model(db):
                continue

            forecast_entry = {
                "model_name": forecast.get("model_name"),
                "dates": forecast.get("dates", []),
                "values": forecast.get("values", []),
            }

            if include_intervals:
                forecast_entry["lower_95"] = forecast.get("lower_95")
                forecast_entry["upper_95"] = forecast.get("upper_95")
                forecast_entry["lower_80"] = forecast.get("lower_80")
                forecast_entry["upper_80"] = forecast.get("upper_80")

            export_data["forecasts"].append(forecast_entry)

        # Add history if requested
        if include_history:
            history_df = self._load_historical_data(db)
            export_data["history"] = history_df.to_dict(orient="records")

        json_content = json.dumps(export_data, indent=2, default=str)
        logger.info(f"Exported forecast to JSON ({len(json_content)} bytes)")
        return json_content

    def export_model(
        self,
        db: DuckDBManager,
        model_name: str,
        format: str = "pickle",
    ) -> bytes:
        """Export trained model.

        Args:
            db: DuckDB manager instance.
            model_name: Name of the model to export.
            format: Export format ('pickle' or 'joblib').

        Returns:
            Model file as bytes.

        Raises:
            ResourceNotFoundError: If model file not found.
            DataValidationError: If format not supported.
        """
        logger.info(f"Exporting model: {model_name} (format: {format})")

        if format not in ["pickle", "joblib"]:
            raise DataValidationError(
                f"Unsupported export format: {format}. Use 'pickle' or 'joblib'.",
                details={"supported_formats": ["pickle", "joblib"]},
            )

        # Find model file
        # Priority 1: Exact match in root
        model_file = self.models_dir / f"{model_name}.pkl"

        if not model_file.exists():
            # Priority 2: Exact match in subdir
            model_file = self.models_dir / model_name / f"{model_name}.pkl"

        if not model_file.exists():
            # Priority 3: Glob match (timestamped or other suffix)
            model_files = list(self.models_dir.glob(f"{model_name}_*.pkl"))
            if model_files:
                # Sort by modification time to get latest
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                model_file = model_files[0]
            else:
                raise ResourceNotFoundError(
                    f"Model file not found: {model_name}",
                    details={
                        "model_name": model_name,
                        "models_dir": str(self.models_dir),
                    },
                )

        # Read model file
        with open(model_file, "rb") as f:
            model_bytes = f.read()

        logger.info(f"Exported model {model_name} ({len(model_bytes)} bytes)")
        return model_bytes

    def export_config(self, db: DuckDBManager) -> str:
        """Export complete project configuration to JSON.

        Args:
            db: DuckDB manager instance.

        Returns:
            JSON string with all configurations.

        Raises:
            ResourceNotFoundError: If configurations not found.
        """
        logger.info("Exporting project configuration")

        config_data = {
            "base_config": None,
            "preprocessing_config": None,
            "models_config": None,
            "forecast_config": None,
        }

        # Load base config
        if db.table_exists("config_base"):
            result = db.execute("SELECT config FROM config_base LIMIT 1")
            row = result.fetchone()
            if row:
                config_data["base_config"] = json.loads(row[0])

        # Load preprocessing config
        if db.table_exists("config_preprocessing"):
            result = db.execute("SELECT config FROM config_preprocessing LIMIT 1")
            row = result.fetchone()
            if row:
                config_data["preprocessing_config"] = json.loads(row[0])

        # Load models config
        if db.table_exists("config_models"):
            result = db.execute("SELECT config FROM config_models LIMIT 1")
            row = result.fetchone()
            if row:
                config_data["models_config"] = json.loads(row[0])

        # Check if any config was loaded
        if all(v is None for v in config_data.values()):
            raise ResourceNotFoundError(
                "No configuration found. Please configure the pipeline first.",
                details={"available_configs": list(config_data.keys())},
            )

        json_content = json.dumps(config_data, indent=2, default=str)
        logger.info(f"Exported configuration ({len(json_content)} bytes)")
        return json_content

    # Helper methods

    def _load_forecast_data(self, db: DuckDBManager) -> List[Dict]:
        """Load forecast data from database."""
        if not db.table_exists("forecast_results"):
            raise ResourceNotFoundError(
                "No forecast results found. Please generate forecast first.",
                details={"missing_table": "forecast_results"},
            )

        result = db.execute("SELECT forecasts FROM forecast_results LIMIT 1")
        row = result.fetchone()

        if not row:
            raise ResourceNotFoundError("No forecast results found")

        forecasts = json.loads(row[0])

        # Filter by selected models if config exists
        if db.table_exists("config_models"):
            try:
                result = db.execute("SELECT config FROM config_models LIMIT 1")
                row = result.fetchone()
                if row:
                    config = json.loads(row[0])
                    selected_models = set(config.get("selected_models", []))

                    if selected_models:
                        filtered_forecasts = []
                        for f in forecasts:
                            model_name = f.get("model_name", "")
                            # Check exact match or grouped format "model (group=val)"
                            base_name = model_name.split(" (")[0]
                            if (
                                base_name in selected_models
                                or model_name in selected_models
                            ):
                                filtered_forecasts.append(f)

                        if filtered_forecasts:
                            return filtered_forecasts
            except Exception as e:
                logger.warning(f"Failed to filter forecasts by selected models: {e}")

        return forecasts

    def _forecasts_to_dataframe(
        self,
        forecasts: List[Dict],
        db: Optional[DuckDBManager] = None,
        include_intervals: bool = True,
        include_all_models: bool = False,
    ) -> pd.DataFrame:
        """Convert forecast results to DataFrame."""
        rows = []

        for forecast in forecasts:
            if not include_all_models and forecast.get(
                "model_name"
            ) != self._get_best_model(db):
                continue

            model_name = forecast.get("model_name")
            dates = forecast.get("dates", [])
            values = forecast.get("values", [])
            lower_95 = forecast.get("lower_95", [None] * len(dates))
            upper_95 = forecast.get("upper_95", [None] * len(dates))
            lower_80 = forecast.get("lower_80", [None] * len(dates))
            upper_80 = forecast.get("upper_80", [None] * len(dates))

            for i in range(len(dates)):
                row = {
                    "date": dates[i],
                    "model": model_name,
                    "forecast": values[i],
                }

                if include_intervals:
                    row["lower_95"] = lower_95[i]
                    row["upper_95"] = upper_95[i]
                    row["lower_80"] = lower_80[i]
                    row["upper_80"] = upper_80[i]

                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def _load_historical_data(self, db: DuckDBManager) -> pd.DataFrame:
        """Load historical data from processed_data table."""
        if not db.table_exists("processed_data"):
            return pd.DataFrame()

        df = db.fetch_df("SELECT * FROM processed_data")

        # Get target column name from base config
        if db.table_exists("config_base"):
            result = db.execute("SELECT config FROM config_base LIMIT 1")
            row = result.fetchone()
            if row:
                base_config = json.loads(row[0])
                date_col = base_config.get("date_column", "date")
                target_col = base_config.get("target_column", "target")

                # Select only date and target columns
                if date_col in df.columns and target_col in df.columns:
                    df = df[[date_col, target_col]].copy()
                    df.columns = ["date", "actual"]
                    df["model"] = "historical"
                    df["forecast"] = None
                    return df

        return pd.DataFrame()

    def _create_metadata_sheet(
        self, db: DuckDBManager, forecasts: List[Dict]
    ) -> pd.DataFrame:
        """Create metadata sheet for Excel export."""
        metadata = []

        # Add forecast info
        metadata.append({"key": "Number of Models", "value": len(forecasts)})

        for forecast in forecasts:
            metadata.append(
                {
                    "key": f"Model: {forecast.get('model_name')}",
                    "value": f"Forecasted {len(forecast.get('dates', []))} points",
                }
            )

        # Add config info
        if db.table_exists("config_base"):
            result = db.execute("SELECT config FROM config_base LIMIT 1")
            row = result.fetchone()
            if row:
                config = json.loads(row[0])
                metadata.append(
                    {"key": "Target Column", "value": config.get("target_column")}
                )
                metadata.append({"key": "Frequency", "value": config.get("frequency")})

        return pd.DataFrame(metadata)

    def _get_export_metadata(self, db: DuckDBManager) -> Dict:
        """Get metadata for JSON export."""
        metadata = {
            "export_timestamp": pd.Timestamp.now().isoformat(),
            "application": "MLForcast",
        }

        # Add base config info
        if db.table_exists("config_base"):
            result = db.execute("SELECT config FROM config_base LIMIT 1")
            row = result.fetchone()
            if row:
                config = json.loads(row[0])
                metadata["target_column"] = config.get("target_column")
                metadata["frequency"] = config.get("frequency")

        return metadata

    def _get_best_model(self, db: Optional[DuckDBManager]) -> str:
        """Get best model name from evaluation results."""
        if db is None or not db.table_exists("evaluation_report"):
            return "unknown"

        try:
            result = db.execute("SELECT report FROM evaluation_report LIMIT 1")
            row = result.fetchone()
            if row:
                eval_data = json.loads(row[0])
                return eval_data.get("best_model", "unknown")
        except Exception:
            pass

        return "unknown"
