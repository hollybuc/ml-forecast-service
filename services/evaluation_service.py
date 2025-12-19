"""Evaluation service for model comparison and metrics calculation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os
import logging
import traceback

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for evaluating and comparing trained models."""

    def evaluate_models(
        self,
        project_id: str,
        training_results: List[Dict[str, Any]],
        ranking_metric: str = 'mape',
    ) -> Dict[str, Any]:
        """Evaluate all trained models and create comparison report.

        Args:
            project_id: Project ID
            training_results: List of training result dictionaries with metrics
            ranking_metric: Metric to use for ranking ('mae', 'rmse', 'mape', 'smape', 'r2', 'medae', 'mase')

        Returns:
            Dictionary with evaluation results including rankings and recommendations
        """
        logger.info(f"Evaluating {len(training_results)} models for project {project_id} using {ranking_metric}")

        model_results = []

        for idx, training_result in enumerate(training_results):
            try:
                model_type = training_result.get('modelType', training_result.get('modelName', f'Model_{idx}'))
                model_path = training_result.get('modelPath')
                training_time = training_result.get('trainingTime', 0)
                metrics = training_result.get('metrics')

                logger.info(f"Processing model: {model_type} with metrics: {metrics}")

                # Check if model file exists
                if not model_path or not os.path.exists(model_path):
                    logger.warning(f"Model file not found: {model_path}, skipping")
                    continue

                # Check if metrics exist
                if not metrics or not isinstance(metrics, dict):
                    logger.warning(f"Model {model_type} has no valid metrics, skipping")
                    continue

                # Use the metrics that were calculated during training
                model_results.append({
                    'modelName': model_type,
                    'metrics': {
                        'mae': float(metrics.get('mae', 0)) if metrics.get('mae') is not None else 0.0,
                        'rmse': float(metrics.get('rmse', 0)) if metrics.get('rmse') is not None else 0.0,
                        'mape': float(metrics.get('mape', 0)) if metrics.get('mape') is not None else 0.0,
                        'smape': float(metrics.get('smape', 0)) if metrics.get('smape') is not None else 0.0,
                        'r2': float(metrics.get('r2', 0)) if metrics.get('r2') is not None else 0.0,
                        'medae': float(metrics.get('medae', 0)) if metrics.get('medae') is not None else 0.0,
                        'mase': float(metrics.get('mase', 0)) if metrics.get('mase') is not None else 0.0,
                    },
                    'trainingTime': float(training_time),
                    'predictionTime': 0,  # Not re-predicting, using stored metrics
                })

            except Exception as e:
                logger.error(f"Error processing model {model_type}: {e}")
                logger.error(traceback.format_exc())
                continue

        if not model_results:
            raise ValueError("No models could be evaluated. Please ensure models are trained successfully with valid metrics.")

        # Sort models by ranking metric
        reverse_sort = ranking_metric == 'r2'  # Higher is better for R2
        model_results.sort(
            key=lambda x: x['metrics'].get(ranking_metric, float('inf') if not reverse_sort else float('-inf')),
            reverse=reverse_sort
        )

        # Add rank to each model
        for idx, result in enumerate(model_results):
            result['rank'] = idx + 1

        # Calculate summary statistics
        best_model = model_results[0]['modelName'] if model_results else None
        best_model_metrics = model_results[0]['metrics'] if model_results else {}

        avg_metrics = {}
        for metric_name in ['mae', 'rmse', 'mape', 'smape', 'r2', 'medae', 'mase']:
            values = [
                r['metrics'].get(metric_name, 0)
                for r in model_results
                if r['metrics'].get(metric_name) is not None and r['metrics'].get(metric_name) != 0
            ]
            avg_metrics[metric_name] = float(sum(values) / len(values)) if values else 0.0

        recommendations = self._generate_recommendations(model_results, ranking_metric, best_model)

        summary = {
            'totalModels': len(model_results),
            'bestModelMetrics': best_model_metrics,
            'averageMetrics': avg_metrics,
            'recommendations': recommendations,
        }

        logger.info(f"Evaluation complete. Best model: {best_model} with {ranking_metric}={best_model_metrics.get(ranking_metric, 0):.4f}")

        return {
            'rankingMetric': ranking_metric,
            'bestModel': best_model,
            'modelResults': model_results,
            'summary': summary,
            'metadata': {
                'processingTime': 0,  # Will be set by worker
                'totalModelsEvaluated': len(model_results),
            }
        }

    def _generate_recommendations(
        self,
        model_results: List[Dict[str, Any]],
        ranking_metric: str,
        best_model: Optional[str] = None,
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        if not model_results:
            return ["No models available for evaluation."]

        if best_model:
            recommendations.append(
                f"The best performing model is '{best_model}' based on {ranking_metric.upper()}."
            )

        # Check if there's a clear winner or if models are close
        if len(model_results) > 1:
            best_metric = model_results[0]['metrics'].get(ranking_metric, 0)
            second_best_metric = model_results[1]['metrics'].get(ranking_metric, 0)

            if ranking_metric == 'r2':
                # Higher is better for R2
                diff_pct = abs(best_metric - second_best_metric) / max(abs(best_metric), 0.01) * 100
                if diff_pct < 5:
                    recommendations.append(
                        f"The top 2 models have similar performance (less than 5% difference). "
                        f"Consider other factors like training time or interpretability."
                    )
            else:
                # Lower is better for error metrics
                if second_best_metric > 0:
                    diff_pct = abs(best_metric - second_best_metric) / second_best_metric * 100
                    if diff_pct < 5:
                        recommendations.append(
                            f"The top 2 models have similar performance (less than 5% difference). "
                            f"Consider other factors like training time or interpretability."
                        )

        # Check overall performance
        best_mape = model_results[0]['metrics'].get('mape', 100)
        if best_mape < 5:
            recommendations.append("Excellent forecasting accuracy! MAPE is below 5%.")
        elif best_mape < 10:
            recommendations.append("Good forecasting accuracy. MAPE is below 10%.")
        elif best_mape > 20:
            recommendations.append(
                "MAPE is relatively high (>20%). Consider revisiting data preprocessing, "
                "feature engineering, or trying different model configurations."
            )

        # Check R2 score
        best_r2 = model_results[0]['metrics'].get('r2', 0)
        if best_r2 < 0.5:
            recommendations.append(
                "R² score is below 0.5, indicating the model may not be capturing the data patterns well. "
                "Consider feature engineering or trying different models."
            )
        elif best_r2 > 0.9:
            recommendations.append("Excellent R² score (>0.9)! The model explains most of the variance in the data.")

        return recommendations
