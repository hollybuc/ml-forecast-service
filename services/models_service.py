"""Models selection and recommendation service."""

from typing import Dict, List
from backend.core.logging import get_logger
from backend.models import EDAReport, BaseConfig

logger = get_logger(__name__)


class ModelsService:
    """Service for model selection and recommendations."""

    # Model registry with descriptions and requirements
    MODELS_REGISTRY = {
        "naive": {
            "name": "Naive Baseline",
            "category": "baseline",
            "description": "Uses the last observed value as the forecast. Simple baseline model.",
            "pros": [
                "Extremely simple and fast",
                "No training required",
                "Good baseline for comparison",
            ],
            "cons": [
                "No trend or seasonality modeling",
                "Cannot use features",
                "Poor for complex patterns",
            ],
            "requirements": {
                "min_observations": 1,
                "supports_features": False,
                "supports_groups": False,
                "requires_validation": False,
            },
            "best_for": "Quick baseline, simple data",
        },
        "seasonal_naive": {
            "name": "Seasonal Naive",
            "category": "baseline",
            "description": "Uses the value from the same season in the previous cycle. Good seasonal baseline.",
            "pros": [
                "Simple and interpretable",
                "Captures seasonal patterns",
                "Fast execution",
            ],
            "cons": [
                "No trend modeling",
                "Cannot use features",
                "Requires regular seasonality",
            ],
            "requirements": {
                "min_observations": 2,
                "supports_features": False,
                "supports_groups": False,
                "requires_validation": False,
                "requires_seasonality": True,
            },
            "best_for": "Seasonal data, baseline comparison",
        },
        "arima": {
            "name": "ARIMA",
            "category": "statistical",
            "description": "AutoRegressive Integrated Moving Average. Classic statistical model for time series.",
            "pros": [
                "Well-established theory",
                "Handles trend and seasonality (SARIMA)",
                "Works well with small datasets",
                "Probabilistic forecasts",
            ],
            "cons": [
                "Cannot use external features",
                "Assumes linear relationships",
                "Slow for large datasets",
                "Requires stationarity (or differencing)",
            ],
            "requirements": {
                "min_observations": 30,
                "supports_features": False,
                "supports_groups": False,
                "requires_validation": False,
            },
            "best_for": "Small-medium datasets, univariate forecasting",
        },
        "prophet": {
            "name": "Prophet",
            "category": "statistical",
            "description": "Facebook's forecasting model. Handles seasonality, holidays, and external regressors.",
            "pros": [
                "Excellent seasonality handling",
                "Robust to missing data and outliers",
                "Can use external regressors",
                "Automatic holiday detection",
                "Interpretable components",
            ],
            "cons": [
                "Can be slow on large datasets",
                "May overfit with many regressors",
                "Less effective for short-term patterns",
            ],
            "requirements": {
                "min_observations": 50,
                "supports_features": True,
                "supports_groups": False,
                "requires_validation": False,
            },
            "best_for": "Data with strong seasonality, holidays impact, medium-large datasets",
        },
        "xgboost": {
            "name": "XGBoost",
            "category": "machine_learning",
            "description": "Gradient boosting model. Excellent for non-linear patterns and feature-rich data.",
            "pros": [
                "Handles non-linear relationships",
                "Can use many features",
                "Fast training and prediction",
                "Feature importance analysis",
                "Handles missing values",
            ],
            "cons": [
                "Requires feature engineering (lags, rolling)",
                "Needs validation set for tuning",
                "Less interpretable",
                "May need careful hyperparameter tuning",
            ],
            "requirements": {
                "min_observations": 100,
                "supports_features": True,
                "supports_groups": True,
                "requires_validation": True,
                "requires_feature_engineering": True,
            },
            "best_for": "Large datasets, many features, non-linear patterns",
        },
        "lightgbm": {
            "name": "LightGBM",
            "category": "machine_learning",
            "description": "Lightweight gradient boosting. Faster than XGBoost, similar performance.",
            "pros": [
                "Very fast training",
                "Handles large datasets efficiently",
                "Can use many features",
                "Good with categorical features",
            ],
            "cons": [
                "Requires feature engineering",
                "Needs validation set",
                "Can overfit on small datasets",
            ],
            "requirements": {
                "min_observations": 100,
                "supports_features": True,
                "supports_groups": True,
                "requires_validation": True,
                "requires_feature_engineering": True,
            },
            "best_for": "Large datasets, speed priority, many features",
        },
        "lstm": {
            "name": "LSTM",
            "category": "deep_learning",
            "description": "Long Short-Term Memory neural network. Deep learning for complex temporal patterns.",
            "pros": [
                "Captures complex temporal dependencies",
                "Can learn from long sequences",
                "Handles multivariate data",
                "Non-linear modeling",
            ],
            "cons": [
                "Requires large datasets (1000+ obs)",
                "Slow training",
                "Needs careful architecture design",
                "Prone to overfitting on small data",
                "Requires GPU for efficiency",
            ],
            "requirements": {
                "min_observations": 1000,
                "supports_features": True,
                "supports_groups": True,
                "requires_validation": True,
                "requires_scaling": True,
            },
            "best_for": "Very large datasets, complex patterns, multivariate forecasting",
        },
    }

    def __init__(self):
        """Initialize models service."""
        pass

    def get_available_models(self) -> List[Dict]:
        """Get list of all available models.

        Returns:
            List of model information dictionaries.
        """
        logger.info("Getting available models")

        models = []
        for model_id, model_info in self.MODELS_REGISTRY.items():
            models.append(
                {
                    "id": model_id,
                    "name": model_info["name"],
                    "category": model_info["category"],
                    "description": model_info["description"],
                    "pros": model_info["pros"],
                    "cons": model_info["cons"],
                    "requirements": model_info["requirements"],
                    "best_for": model_info["best_for"],
                }
            )

        logger.info(f"Returned {len(models)} available models")
        return models

    def get_model_categories(self) -> Dict[str, List[str]]:
        """Get models grouped by category.

        Returns:
            Dictionary with categories as keys and model IDs as values.
        """
        categories = {
            "baseline": [],
            "statistical": [],
            "machine_learning": [],
            "deep_learning": [],
        }

        for model_id, model_info in self.MODELS_REGISTRY.items():
            category = model_info["category"]
            if category in categories:
                categories[category].append(model_id)

        return categories

    def get_recommendations(
        self,
        eda_report: EDAReport,
        base_config: BaseConfig,
    ) -> Dict:
        """Generate model recommendations based on data characteristics.

        Args:
            eda_report: EDA report with data analysis.
            base_config: Base configuration.

        Returns:
            Dictionary with recommendations and scores.
        """
        logger.info("Generating model recommendations")

        n_obs = eda_report.n_observations
        has_seasonality = len(eda_report.seasonality_detected) > 0
        has_features = len(base_config.feature_columns) > 0
        has_groups = base_config.group_column is not None

        recommendations = []

        # Score each model
        for model_id, model_info in self.MODELS_REGISTRY.items():
            score, reasons = self._score_model(
                model_id,
                model_info,
                n_obs,
                has_seasonality,
                has_features,
                has_groups,
                eda_report,
            )

            recommendations.append(
                {
                    "model_id": model_id,
                    "model_name": model_info["name"],
                    "score": score,
                    "recommended": score >= 7.0,
                    "reasons": reasons,
                    "warnings": [],
                    "category": model_info["category"],
                }
            )

        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        # Add recommendations summary
        recommended_models = [r for r in recommendations if r["recommended"]]
        not_recommended = [r for r in recommendations if not r["recommended"]]

        result = {
            "recommended": recommended_models,
            "others": not_recommended,
            "summary": {
                "n_observations": n_obs,
                "has_seasonality": has_seasonality,
                "has_features": has_features,
                "has_groups": has_groups,
                "top_recommendation": recommended_models[0]["model_id"]
                if recommended_models
                else None,
            },
        }

        logger.info(f"Generated {len(recommended_models)} recommendations")
        return result

    def _score_model(
        self,
        model_id: str,
        model_info: Dict,
        n_obs: int,
        has_seasonality: bool,
        has_features: bool,
        has_groups: bool,
        eda_report: EDAReport,
    ) -> tuple[float, List[str]]:
        """Score a model based on data characteristics.

        Args:
            model_id: Model identifier.
            model_info: Model information.
            n_obs: Number of observations.
            has_seasonality: Whether data has seasonality.
            has_features: Whether features are available.
            has_groups: Whether groups are present.
            eda_report: EDA report.

        Returns:
            Tuple of (score, reasons).
        """
        score = 5.0  # Base score
        reasons = []

        requirements = model_info["requirements"]

        # Check minimum observations
        min_obs = requirements.get("min_observations", 0)
        if n_obs < min_obs:
            score -= 5.0
            reasons.append(f"❌ Not enough data (need {min_obs}+ obs, have {n_obs})")
            return score, reasons

        # Bonus for meeting minimum observations
        if n_obs >= min_obs:
            score += 1.0
            reasons.append(f"✅ Sufficient data ({n_obs} observations)")

        # Seasonality matching
        if has_seasonality:
            if model_id in ["seasonal_naive", "arima", "prophet"]:
                score += 2.0
                reasons.append(f"✅ Excellent for seasonal data")
            elif model_id in ["xgboost", "lightgbm", "lstm"]:
                score += 1.0
                reasons.append(f"✅ Can handle seasonality with features")

        # Features matching
        if has_features:
            if requirements.get("supports_features", False):
                score += 2.0
                reasons.append(f"✅ Can use {len(eda_report.recommendations)} available features")
            else:
                reasons.append(f"⚠️ Cannot use features")
        else:
            if not requirements.get("supports_features", False):
                score += 0.5
                reasons.append(f"✅ No features needed")

        # Groups matching
        if has_groups:
            if requirements.get("supports_groups", False):
                score += 1.5
                reasons.append(f"✅ Supports multiple groups")
            else:
                score -= 1.0
                reasons.append(f"⚠️ Doesn't support groups (need separate models)")

        # Data size bonuses
        if model_id in ["arima", "prophet"]:
            if 50 <= n_obs <= 500:
                score += 1.5
                reasons.append(f"✅ Optimal size for this model")
        elif model_id in ["xgboost", "lightgbm"]:
            if n_obs >= 200:
                score += 1.5
                reasons.append(f"✅ Good size for ML models")
        elif model_id == "lstm":
            if n_obs >= 1000:
                score += 2.0
                reasons.append(f"✅ Enough data for deep learning")
            elif n_obs < 500:
                score -= 2.0
                reasons.append(f"❌ Too little data for LSTM")

        # Stationarity considerations
        if not eda_report.is_stationary and model_id == "arima":
            score += 1.0
            reasons.append(f"✅ ARIMA handles non-stationary data with differencing")

        # Always recommend baselines for comparison
        if model_id in ["naive", "seasonal_naive"]:
            score += 1.0
            reasons.append(f"✅ Always useful as baseline")

        return score, reasons


# Global models service instance
models_service = ModelsService()


def get_models_service() -> ModelsService:
    """Get models service instance (for dependency injection).

    Returns:
        ModelsService instance.
    """
    return models_service
