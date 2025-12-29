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
                training_result_id = training_result.get('id')
                metadata = training_result.get('metadata', {})

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
                    'trainingResultId': training_result_id,
                    'metadata': metadata,
                })

            except Exception as e:
                logger.error(f"Error processing model {model_type}: {e}")
                logger.error(traceback.format_exc())
                continue

        if not model_results:
            raise ValueError("No models could be evaluated. Please ensure models are trained successfully with valid metrics.")

        # Check if we have grouped models (models with group_value in metadata)
        has_grouped_models = any(
            result.get('metadata', {}).get('group_value') is not None 
            for result in model_results
        )

        if has_grouped_models:
            logger.info("Detected grouped models, aggregating by base model type")
            # Group models by base model type
            grouped_by_model = {}
            for result in model_results:
                base_model = result['modelName'].split('_')[0]  # Extract base model (e.g., "arima" from "arima_North")
                group_value = result.get('metadata', {}).get('group_value')
                
                if base_model not in grouped_by_model:
                    grouped_by_model[base_model] = {
                        'modelName': base_model,
                        'groupedResults': [],
                        'metadata': result.get('metadata', {}),
                        'trainingResultId': result.get('trainingResultId'),
                    }
                
                # Add this result to the grouped results
                grouped_by_model[base_model]['groupedResults'].append({
                    'group_value': group_value,
                    'metrics': result['metrics'],
                    'trainingTime': result.get('trainingTime', 0),
                    'metadata': result.get('metadata', {}),
                })
            
            # Calculate averaged metrics for each model type
            aggregated_results = []
            for base_model, group_data in grouped_by_model.items():
                # Calculate average metrics across all groups
                avg_metrics = {}
                for metric_name in ['mae', 'rmse', 'mape', 'smape', 'r2', 'medae', 'mase']:
                    values = [
                        gr['metrics'].get(metric_name, 0)
                        for gr in group_data['groupedResults']
                        if gr['metrics'].get(metric_name) is not None
                    ]
                    avg_metrics[metric_name] = float(sum(values) / len(values)) if values else 0.0
                
                aggregated_results.append({
                    'modelName': base_model,
                    'metrics': avg_metrics,
                    'trainingTime': sum(gr.get('trainingTime', 0) for gr in group_data['groupedResults']),
                    'predictionTime': 0,
                    'trainingResultId': group_data['trainingResultId'],
                    'metadata': {
                        **group_data['metadata'],
                        'groupedResults': group_data['groupedResults'],
                    }
                })
            
            model_results = aggregated_results
            logger.info(f"Aggregated {len(model_results)} model types from grouped results")

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
        # If we have grouped models, total models should count individuals
        total_models_count = 0
        total_training_time = 0.0
        for r in model_results:
            if 'groupedResults' in r.get('metadata', {}):
                count = len(r['metadata']['groupedResults'])
                total_models_count += count
                total_training_time += r.get('trainingTime', 0)
            else:
                total_models_count += 1
                total_training_time += r.get('trainingTime', 0)

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
            'totalModels': total_models_count,
            'bestModelMetrics': best_model_metrics,
            'averageMetrics': avg_metrics,
            'recommendations': recommendations,
            'totalTrainingTime': total_training_time,
        }

        logger.info(f"Evaluation complete. Best model: {best_model} with {ranking_metric}={best_model_metrics.get(ranking_metric, 0):.4f}")

        # Prepare top-level metadata
        report_metadata = {
            'processingTime': 0,  # Will be set by worker
            'totalModelsEvaluated': len(model_results),
        }

        # Promote common metadata from the first model result if it exists
        if model_results:
            # Check for dataSplits in the models metadata
            # EvaluationService.evaluate_models takes training_results which contains metadata
            # We already extracted metadata and put it into model_results[i]['metadata'] 
            # in evaluate_models loop
            first_metadata = model_results[0].get('metadata', {})
            for key in ['dataSplits', 'trainSize', 'valSize', 'testSize']:
                if key in first_metadata:
                    report_metadata[key] = first_metadata[key]

        return {
            'rankingMetric': ranking_metric,
            'bestModel': best_model,
            'modelResults': model_results,
            'summary': summary,
            'metadata': report_metadata
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
