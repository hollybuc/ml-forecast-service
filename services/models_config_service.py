"""Models configuration service."""

from typing import Dict, Optional, Any
import json

from backend.core.logging import get_logger
from backend.core.exceptions import DataValidationError, ResourceNotFoundError
from backend.core.database import DuckDBManager
from backend.models import (
    BaseConfig,
    EDAReport,
    SplitConfig,
    ForecastConfig,
    ARIMAConfig,
    ProphetConfig,
    XGBoostConfig,
    LSTMConfig,
)
from backend.services.models_service import ModelsService

logger = get_logger(__name__)


class ModelsConfigService:
    """Service for managing models configuration."""

    def __init__(self, models_service: ModelsService):
        """Initialize service.

        Args:
            models_service: ModelsService instance for model validation.
        """
        self.models_service = models_service

    def validate_and_save_config(
        self,
        selected_models: list[str],
        split_config: SplitConfig,
        forecast_config: ForecastConfig,
        model_configs: Dict[str, Any],
        db: DuckDBManager,
    ) -> Dict:
        """Validate and save models configuration.

        Args:
            selected_models: List of selected model IDs.
            split_config: Data split configuration.
            forecast_config: Forecast configuration.
            model_configs: Dictionary with per-model configurations.
            db: DuckDB manager instance.

        Returns:
            Dictionary with validation results and saved configuration.

        Raises:
            ResourceNotFoundError: If prerequisites not found.
            DataValidationError: If validation fails.
        """
        logger.info(f"Validating configuration for {len(selected_models)} models")

        # 1. Load prerequisites
        base_config, eda_report = self._load_prerequisites(db)

        # 2. Validate selected models
        self._validate_selected_models(selected_models)

        # 3. Validate split configuration
        self._validate_split_config(split_config)

        # 4. Validate forecast configuration
        self._validate_forecast_config(forecast_config, eda_report)

        # 5. Validate model-specific configs
        validated_configs = self._validate_model_configs(
            selected_models, model_configs, base_config, eda_report, split_config
        )

        # 6. Save configuration to database
        config_data = {
            "selected_models": selected_models,
            "split_config": split_config.model_dump(),
            "forecast_config": forecast_config.model_dump(),
            "model_configs": validated_configs,
        }

        config_json = json.dumps(config_data)
        db.execute("CREATE TABLE IF NOT EXISTS config_models (config JSON)")
        db.execute("DELETE FROM config_models")
        db.execute("INSERT INTO config_models VALUES (?)", (config_json,))
        logger.info("Models configuration saved to database")

        return {
            "status": "success",
            "selected_models": selected_models,
            "split_config": split_config.model_dump(),
            "forecast_config": forecast_config.model_dump(),
            "model_configs": validated_configs,
            "validation_summary": self._create_validation_summary(
                selected_models, base_config, eda_report, split_config
            ),
        }

    def _load_prerequisites(
        self, db: DuckDBManager
    ) -> tuple[BaseConfig, EDAReport]:
        """Load base config and EDA report from database.

        Args:
            db: DuckDB manager instance.

        Returns:
            Tuple of (BaseConfig, EDAReport).

        Raises:
            ResourceNotFoundError: If prerequisites not found.
        """
        # Load base config
        if not db.table_exists("config_base"):
            raise ResourceNotFoundError(
                "No base configuration found. Please complete KROK 2 first.",
                details={"missing_table": "config_base"},
            )

        result = db.execute("SELECT config FROM config_base LIMIT 1")
        row = result.fetchone()
        if not row:
            raise ResourceNotFoundError("No base configuration found")

        base_config = BaseConfig.model_validate_json(row[0])

        # Load EDA report
        if not db.table_exists("eda_report"):
            raise ResourceNotFoundError(
                "No EDA report found. Please complete KROK 3 first.",
                details={"missing_table": "eda_report"},
            )

        result = db.execute("SELECT report FROM eda_report LIMIT 1")
        row = result.fetchone()
        if not row:
            raise ResourceNotFoundError("No EDA report found")

        eda_report = EDAReport.model_validate_json(row[0])

        logger.info("Prerequisites loaded successfully")
        return base_config, eda_report

    def _validate_selected_models(self, selected_models: list[str]):
        """Validate that selected models exist.

        Args:
            selected_models: List of model IDs.

        Raises:
            DataValidationError: If any model doesn't exist.
        """
        if not selected_models:
            raise DataValidationError(
                "No models selected. Please select at least one model.",
                details={"selected_models": selected_models},
            )

        available_model_ids = list(self.models_service.MODELS_REGISTRY.keys())

        for model_id in selected_models:
            if model_id not in available_model_ids:
                raise DataValidationError(
                    f"Invalid model ID: {model_id}",
                    details={
                        "invalid_model": model_id,
                        "available_models": available_model_ids,
                    },
                )

        logger.info(f"Selected models validated: {selected_models}")

    def _validate_split_config(self, split_config: SplitConfig):
        """Validate split configuration.

        Args:
            split_config: Split configuration.

        Raises:
            DataValidationError: If split config is invalid.
        """
        if split_config.method == "temporal":
            # Check that sizes sum to 1.0
            total = split_config.train_size + split_config.val_size + split_config.test_size
            if abs(total - 1.0) > 0.01:
                raise DataValidationError(
                    f"Train/val/test sizes must sum to 1.0, got {total:.2f}",
                    details={
                        "train_size": split_config.train_size,
                        "val_size": split_config.val_size,
                        "test_size": split_config.test_size,
                        "total": total,
                    },
                )

            # Check minimum sizes
            if split_config.train_size < 0.5:
                raise DataValidationError(
                    f"Train size too small: {split_config.train_size}. Minimum is 0.5",
                    details={"train_size": split_config.train_size},
                )

            if split_config.test_size < 0.1:
                raise DataValidationError(
                    f"Test size too small: {split_config.test_size}. Minimum is 0.1",
                    details={"test_size": split_config.test_size},
                )

        logger.info(f"Split config validated: {split_config.method}")

    def _validate_forecast_config(
        self, forecast_config: ForecastConfig, eda_report: EDAReport
    ):
        """Validate forecast configuration.

        Args:
            forecast_config: Forecast configuration.
            eda_report: EDA report with data info.

        Raises:
            DataValidationError: If forecast config is invalid.
        """
        # Check horizon is reasonable (not too large)
        n_obs = eda_report.n_observations
        max_horizon = int(n_obs * 0.3)  # Max 30% of data length

        if forecast_config.horizon > max_horizon:
            raise DataValidationError(
                f"Forecast horizon too large: {forecast_config.horizon}. "
                f"Maximum recommended is {max_horizon} (30% of {n_obs} observations)",
                details={
                    "horizon": forecast_config.horizon,
                    "max_recommended": max_horizon,
                    "n_observations": n_obs,
                },
            )

        # Validate confidence intervals
        for ci in forecast_config.confidence_intervals:
            if ci <= 0.0 or ci >= 1.0:
                raise DataValidationError(
                    f"Invalid confidence interval: {ci}. Must be between 0 and 1",
                    details={"confidence_interval": ci},
                )

        logger.info(f"Forecast config validated: horizon={forecast_config.horizon}")

    def _validate_model_configs(
        self,
        selected_models: list[str],
        model_configs: Dict[str, Any],
        base_config: BaseConfig,
        eda_report: EDAReport,
        split_config: SplitConfig,
    ) -> Dict[str, Any]:
        """Validate per-model configurations.

        Args:
            selected_models: List of selected model IDs.
            model_configs: Dictionary with model configs.
            base_config: Base configuration.
            eda_report: EDA report.
            split_config: Split configuration.

        Returns:
            Dictionary with validated model configs.

        Raises:
            DataValidationError: If any model config is invalid.
        """
        validated_configs = {}
        n_obs = eda_report.n_observations

        for model_id in selected_models:
            logger.info(f"Validating config for model: {model_id}")

            # Get model info
            model_info = self.models_service.MODELS_REGISTRY[model_id]
            requirements = model_info["requirements"]

            # 1. Check minimum observations
            min_obs = requirements.get("min_observations", 0)
            if n_obs < min_obs:
                raise DataValidationError(
                    f"Model '{model_id}' requires at least {min_obs} observations, "
                    f"but only {n_obs} are available",
                    details={
                        "model": model_id,
                        "required": min_obs,
                        "available": n_obs,
                    },
                )

            # 2. Check validation set requirement
            requires_validation = requirements.get("requires_validation", False)
            if requires_validation and split_config.val_size == 0:
                raise DataValidationError(
                    f"Model '{model_id}' requires a validation set, "
                    f"but val_size is 0",
                    details={
                        "model": model_id,
                        "val_size": split_config.val_size,
                    },
                )

            # 3. Check features requirement
            supports_features = requirements.get("supports_features", False)
            has_features = len(base_config.feature_columns) > 0
            if not supports_features and has_features:
                logger.warning(
                    f"Model '{model_id}' doesn't support features. "
                    f"Features will be ignored."
                )

            # 4. Check groups requirement
            supports_groups = requirements.get("supports_groups", False)
            has_groups = base_config.group_column is not None
            if not supports_groups and has_groups:
                logger.warning(
                    f"Model '{model_id}' doesn't support groups. "
                    f"Separate models will be trained per group."
                )

            # 5. Validate model-specific config
            config_key = f"{model_id}_config"
            if config_key in model_configs:
                config_data = model_configs[config_key]
                validated_config = self._validate_specific_config(
                    model_id, config_data
                )
                validated_configs[model_id] = validated_config
            else:
                # Use defaults
                validated_configs[model_id] = self._get_default_config(model_id)

        logger.info(f"All model configs validated: {list(validated_configs.keys())}")
        return validated_configs

    def _validate_specific_config(
        self, model_id: str, config_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate model-specific configuration.

        Args:
            model_id: Model identifier.
            config_data: Configuration data.

        Returns:
            Validated configuration dictionary.

        Raises:
            DataValidationError: If config is invalid.
        """
        try:
            if model_id == "arima" or model_id == "sarima":
                config = ARIMAConfig.model_validate(config_data)
                return config.model_dump()
            elif model_id == "prophet":
                config = ProphetConfig.model_validate(config_data)
                return config.model_dump()
            elif model_id == "xgboost" or model_id == "lightgbm":
                config = XGBoostConfig.model_validate(config_data)
                return config.model_dump()
            elif model_id == "lstm" or model_id == "gru":
                config = LSTMConfig.model_validate(config_data)
                return config.model_dump()
            else:
                # Baseline models (naive, seasonal_naive) don't need config
                return {}

        except Exception as e:
            raise DataValidationError(
                f"Invalid configuration for model '{model_id}': {str(e)}",
                details={"model": model_id, "error": str(e)},
            )

    def _get_default_config(self, model_id: str) -> Dict[str, Any]:
        """Get default configuration for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Default configuration dictionary.
        """
        if model_id == "arima" or model_id == "sarima":
            return ARIMAConfig().model_dump()
        elif model_id == "prophet":
            return ProphetConfig().model_dump()
        elif model_id == "xgboost" or model_id == "lightgbm":
            return XGBoostConfig().model_dump()
        elif model_id == "lstm" or model_id == "gru":
            return LSTMConfig().model_dump()
        else:
            return {}

    def _create_validation_summary(
        self,
        selected_models: list[str],
        base_config: BaseConfig,
        eda_report: EDAReport,
        split_config: SplitConfig,
    ) -> Dict[str, Any]:
        """Create validation summary.

        Args:
            selected_models: Selected model IDs.
            base_config: Base configuration.
            eda_report: EDA report.
            split_config: Split configuration.

        Returns:
            Validation summary dictionary.
        """
        n_obs = eda_report.n_observations
        n_train = int(n_obs * split_config.train_size)
        n_val = int(n_obs * split_config.val_size)
        n_test = int(n_obs * split_config.test_size)

        return {
            "n_models": len(selected_models),
            "models": selected_models,
            "data_size": {
                "total": n_obs,
                "train": n_train,
                "val": n_val,
                "test": n_test,
            },
            "has_features": len(base_config.feature_columns) > 0,
            "n_features": len(base_config.feature_columns),
            "has_groups": base_config.group_column is not None,
            "all_requirements_met": True,
        }


# Global service instance
_models_config_service: Optional[ModelsConfigService] = None


def get_models_config_service() -> ModelsConfigService:
    """Get models config service instance (for dependency injection).

    Returns:
        ModelsConfigService instance.
    """
    global _models_config_service
    if _models_config_service is None:
        from backend.services.models_service import models_service

        _models_config_service = ModelsConfigService(models_service)
    return _models_config_service
