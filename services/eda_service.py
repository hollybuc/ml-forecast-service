"""
EDA (Exploratory Data Analysis) Service

Performs comprehensive analysis of time series data including:
- Date range and frequency detection
- Missing data analysis
- Statistical summaries
- Trend detection
- Seasonality detection
- Stationarity tests
- Outlier detection
- Autocorrelation analysis
- Data quality alerts
- Model recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logger = logging.getLogger(__name__)


class EDAService:
    """Service for performing exploratory data analysis on time series data"""

    def analyze(
        self,
        dataset_path: str,
        date_column: str,
        target_column: str,
        feature_columns: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive EDA on the dataset

        Args:
            dataset_path: Path to the CSV file
            date_column: Name of the date column
            target_column: Name of the target variable column
            feature_columns: List of feature column names (optional)

        Returns:
            Dictionary with complete EDA results
        """
        logger.info(f"Starting EDA analysis for: {dataset_path}")

        # Load data
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Parse dates
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        # Convert target column to numeric (handles string values from CSV)
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

        # Extract time series
        ts = df[target_column].values
        dates = df[date_column]

        # Perform all analyses
        date_range = self._analyze_date_range(dates)
        missing_data = self._analyze_missing_data(df, date_column, target_column)
        statistics = self._calculate_statistics(ts)
        trend = self._detect_trend(ts)
        seasonality = self._detect_seasonality(ts, date_range['frequency'])
        stationarity = self._test_stationarity(ts)
        outliers = self._detect_outliers(ts)
        autocorrelation_data = self._calculate_autocorrelation(ts)
        chart_data = self._prepare_chart_data(df, date_column, target_column, ts)
        alerts = self._generate_alerts(
            missing_data, stationarity, outliers, statistics
        )
        recommendations = self._generate_recommendations(
            trend, seasonality, stationarity, len(ts)
        )

        logger.info("EDA analysis completed successfully")

        return {
            'dateRange': date_range,
            'missingData': missing_data,
            'statistics': statistics,
            'trend': trend,
            'seasonality': seasonality,
            'stationarity': stationarity,
            'outliers': outliers,
            'autocorrelation': autocorrelation_data,
            'chartData': chart_data,
            'alerts': alerts,
            'recommendations': recommendations,
        }

    def _analyze_date_range(self, dates: pd.Series) -> Dict[str, Any]:
        """Analyze date range and detect frequency"""
        start = dates.min()
        end = dates.max()
        total_days = (end - start).days

        # Detect frequency
        if len(dates) > 1:
            freq_counts = dates.diff().value_counts()
            most_common_diff = freq_counts.index[0]
            
            # Map to frequency string
            if most_common_diff.days == 1:
                frequency = 'D'  # Daily
            elif most_common_diff.days == 7:
                frequency = 'W'  # Weekly
            elif 28 <= most_common_diff.days <= 31:
                frequency = 'M'  # Monthly
            elif 89 <= most_common_diff.days <= 92:
                frequency = 'Q'  # Quarterly
            elif 364 <= most_common_diff.days <= 366:
                frequency = 'Y'  # Yearly
            else:
                frequency = 'irregular'
        else:
            frequency = 'unknown'

        return {
            'start': start.isoformat(),
            'end': end.isoformat(),
            'totalDays': int(total_days),
            'frequency': frequency,
        }

    def _analyze_missing_data(
        self, df: pd.DataFrame, date_column: str, target_column: str
    ) -> Dict[str, Any]:
        """Analyze missing dates and values"""
        # Missing values in target
        missing_values = df[target_column].isna().sum()
        missing_percentage = (missing_values / len(df)) * 100

        # Missing dates (gaps in time series)
        dates = df[date_column]
        if len(dates) > 1:
            expected_dates = pd.date_range(dates.min(), dates.max(), freq='D')
            missing_dates = len(expected_dates) - len(dates)
        else:
            missing_dates = 0

        return {
            'missingDates': int(missing_dates),
            'missingValues': int(missing_values),
            'missingPercentage': float(missing_percentage),
        }

    def _calculate_statistics(self, ts: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        # Remove NaN values for calculations
        ts_clean = ts[~np.isnan(ts)]

        return {
            'count': int(len(ts_clean)),
            'mean': float(np.mean(ts_clean)),
            'std': float(np.std(ts_clean)),
            'min': float(np.min(ts_clean)),
            'q25': float(np.percentile(ts_clean, 25)),
            'median': float(np.median(ts_clean)),
            'q75': float(np.percentile(ts_clean, 75)),
            'max': float(np.max(ts_clean)),
            'skewness': float(stats.skew(ts_clean)),
            'kurtosis': float(stats.kurtosis(ts_clean)),
        }

    def _detect_trend(self, ts: np.ndarray) -> Dict[str, Any]:
        """Detect trend using linear regression"""
        ts_clean = ts[~np.isnan(ts)]
        x = np.arange(len(ts_clean))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_clean)
        r_squared = r_value ** 2

        # Determine direction
        if abs(slope) < 0.01 * np.std(ts_clean):
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'

        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'rSquared': float(r_squared),
            'direction': direction,
        }

    def _detect_seasonality(
        self, ts: np.ndarray, frequency: str
    ) -> Dict[str, Any]:
        """Detect seasonality using seasonal decomposition"""
        ts_clean = ts[~np.isnan(ts)]

        # Need at least 2 full cycles for seasonal decomposition
        min_length = {'D': 14, 'W': 104, 'M': 24, 'Q': 8, 'Y': 4}
        period_map = {'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'Y': 1}

        if frequency not in period_map or len(ts_clean) < min_length.get(frequency, 24):
            return {
                'detected': False,
                'period': None,
                'strength': None,
            }

        try:
            period = period_map[frequency]
            decomposition = seasonal_decompose(
                ts_clean, model='additive', period=period, extrapolate_trend='freq'
            )

            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal)
            residual_var = np.var(decomposition.resid[~np.isnan(decomposition.resid)])
            strength = seasonal_var / (seasonal_var + residual_var)

            return {
                'detected': strength > 0.1,  # Threshold for detection
                'period': int(period),
                'strength': float(strength),
            }
        except Exception as e:
            logger.warning(f"Seasonality detection failed: {e}")
            return {
                'detected': False,
                'period': None,
                'strength': None,
            }

    def _test_stationarity(self, ts: np.ndarray) -> Dict[str, Any]:
        """Test for stationarity using ADF and KPSS tests"""
        ts_clean = ts[~np.isnan(ts)]

        # ADF test (null hypothesis: non-stationary)
        try:
            adf_result = adfuller(ts_clean, autolag='AIC')
            adf_statistic = float(adf_result[0])
            adf_pvalue = float(adf_result[1])
            adf_is_stationary = adf_pvalue < 0.05
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            adf_statistic = 0.0
            adf_pvalue = 1.0
            adf_is_stationary = False

        # KPSS test (null hypothesis: stationary)
        try:
            kpss_result = kpss(ts_clean, regression='c', nlags='auto')
            kpss_statistic = float(kpss_result[0])
            kpss_pvalue = float(kpss_result[1])
            kpss_is_stationary = kpss_pvalue > 0.05
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
            kpss_statistic = 0.0
            kpss_pvalue = 1.0
            kpss_is_stationary = True

        return {
            'adfTest': {
                'statistic': adf_statistic,
                'pValue': adf_pvalue,
                'isStationary': adf_is_stationary,
            },
            'kpssTest': {
                'statistic': kpss_statistic,
                'pValue': kpss_pvalue,
                'isStationary': kpss_is_stationary,
            },
        }

    def _detect_outliers(self, ts: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        ts_clean = ts[~np.isnan(ts)]

        q1 = np.percentile(ts_clean, 25)
        q3 = np.percentile(ts_clean, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (ts_clean < lower_bound) | (ts_clean > upper_bound)
        outlier_count = np.sum(outlier_mask)
        outlier_percentage = (outlier_count / len(ts_clean)) * 100
        outlier_indices = np.where(outlier_mask)[0].tolist()

        return {
            'count': int(outlier_count),
            'percentage': float(outlier_percentage),
            'method': 'IQR',
            'indices': outlier_indices[:100],  # Limit to first 100
        }

    def _calculate_autocorrelation(self, ts: np.ndarray) -> Dict[str, Any]:
        """Calculate ACF and PACF"""
        ts_clean = ts[~np.isnan(ts)]

        # Calculate ACF and PACF
        max_lags = min(40, len(ts_clean) // 2)
        
        try:
            acf_values = acf(ts_clean, nlags=max_lags, fft=True)
            pacf_values = pacf(ts_clean, nlags=max_lags, method='ywm')
            lags = list(range(max_lags + 1))

            return {
                'acf': [float(v) for v in acf_values],
                'pacf': [float(v) for v in pacf_values],
                'lags': lags,
            }
        except Exception as e:
            logger.warning(f"Autocorrelation calculation failed: {e}")
            return {
                'acf': [],
                'pacf': [],
                'lags': [],
            }

    def _prepare_chart_data(
        self, df: pd.DataFrame, date_column: str, target_column: str, ts: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare data for frontend charts"""
        dates = df[date_column].dt.strftime('%Y-%m-%d').tolist()
        values = df[target_column].fillna(0).tolist()

        # Time series chart data
        time_series = {
            'dates': dates,
            'values': values,
        }

        # Distribution (histogram)
        ts_clean = ts[~np.isnan(ts)]
        hist, bin_edges = np.histogram(ts_clean, bins=30)
        distribution = {
            'bins': bin_edges.tolist(),
            'counts': hist.tolist(),
        }

        # Seasonal decomposition (if possible)
        try:
            if len(ts_clean) >= 14:
                decomposition = seasonal_decompose(
                    ts_clean, model='additive', period=7, extrapolate_trend='freq'
                )
                seasonal_decomposition = {
                    'dates': dates[:len(decomposition.trend)],
                    'trend': [float(v) if not np.isnan(v) else 0 for v in decomposition.trend],
                    'seasonal': [float(v) if not np.isnan(v) else 0 for v in decomposition.seasonal],
                    'residual': [float(v) if not np.isnan(v) else 0 for v in decomposition.resid],
                }
            else:
                seasonal_decomposition = {
                    'dates': [],
                    'trend': [],
                    'seasonal': [],
                    'residual': [],
                }
        except Exception as e:
            logger.warning(f"Seasonal decomposition for chart failed: {e}")
            seasonal_decomposition = {
                'dates': [],
                'trend': [],
                'seasonal': [],
                'residual': [],
            }

        # Boxplot data (overall)
        q1 = float(np.percentile(ts_clean, 25))
        q3 = float(np.percentile(ts_clean, 75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ts_clean[(ts_clean < lower_bound) | (ts_clean > upper_bound)]

        boxplot = {
            'min': float(np.min(ts_clean)),
            'q1': q1,
            'median': float(np.median(ts_clean)),
            'q3': q3,
            'max': float(np.max(ts_clean)),
            'outliers': outliers.tolist()[:50],  # Limit outliers
        }

        # Boxplot by month
        boxplot_by_month = {}
        try:
            df_clean = df.dropna(subset=[target_column])
            df_clean['month'] = df_clean[date_column].dt.strftime('%b')
            
            for month in df_clean['month'].unique():
                month_data = df_clean[df_clean['month'] == month][target_column].values
                if len(month_data) > 0:
                    boxplot_by_month[month] = {
                        'min': float(np.min(month_data)),
                        'q1': float(np.percentile(month_data, 25)),
                        'median': float(np.median(month_data)),
                        'q3': float(np.percentile(month_data, 75)),
                        'max': float(np.max(month_data)),
                    }
        except Exception as e:
            logger.warning(f"Boxplot by month failed: {e}")
            boxplot_by_month = {}

        # ACF/PACF data
        acf_pacf = {}
        try:
            if len(ts_clean) >= 20:
                # Calculate ACF and PACF
                max_lags = min(40, len(ts_clean) // 2)
                acf_values = acf(ts_clean, nlags=max_lags, fft=True)
                pacf_values = pacf(ts_clean, nlags=max_lags)
                
                acf_pacf = {
                    'lags': list(range(max_lags + 1)),
                    'acf': acf_values.tolist(),
                    'pacf': pacf_values.tolist(),
                }
        except Exception as e:
            logger.warning(f"ACF/PACF calculation failed: {e}")
            acf_pacf = {}

        return {
            'timeSeries': time_series,
            'distribution': distribution,
            'seasonalDecomposition': seasonal_decomposition,
            'boxplot': boxplot,
            'boxplotByMonth': boxplot_by_month,
            'acfPacf': acf_pacf,
        }

    def _generate_alerts(
        self,
        missing_data: Dict,
        stationarity: Dict,
        outliers: Dict,
        statistics: Dict,
    ) -> List[Dict[str, str]]:
        """Generate data quality alerts"""
        alerts = []

        # Missing data alerts
        if missing_data['missingPercentage'] > 10:
            alerts.append({
                'level': 'error',
                'message': f"High percentage of missing values: {missing_data['missingPercentage']:.1f}%",
            })
        elif missing_data['missingPercentage'] > 5:
            alerts.append({
                'level': 'warning',
                'message': f"Missing values detected: {missing_data['missingPercentage']:.1f}%",
            })

        # Stationarity alerts
        if not stationarity['adfTest']['isStationary']:
            alerts.append({
                'level': 'info',
                'message': 'Time series is non-stationary (ADF test). Consider differencing.',
            })

        # Outlier alerts
        if outliers['percentage'] > 5:
            alerts.append({
                'level': 'warning',
                'message': f"High percentage of outliers: {outliers['percentage']:.1f}%",
            })

        # Data size alerts
        if statistics['count'] < 30:
            alerts.append({
                'level': 'warning',
                'message': 'Small dataset size may affect model performance.',
            })

        return alerts

    def _generate_recommendations(
        self,
        trend: Dict,
        seasonality: Dict,
        stationarity: Dict,
        data_length: int,
    ) -> List[Dict[str, Any]]:
        """Generate model recommendations based on data characteristics"""
        recommendations = []

        # Prophet - good for seasonal data with trends
        if seasonality['detected']:
            recommendations.append({
                'model': 'Prophet',
                'reason': 'Detected strong seasonality. Prophet handles seasonal patterns well.',
                'priority': 1,
            })

        # ARIMA - good for stationary or near-stationary data
        if stationarity['adfTest']['isStationary'] or not seasonality['detected']:
            recommendations.append({
                'model': 'ARIMA',
                'reason': 'Data shows stationarity. ARIMA is suitable for stationary time series.',
                'priority': 2,
            })

        # XGBoost/LightGBM - good for complex patterns
        if data_length > 100:
            recommendations.append({
                'model': 'XGBoost',
                'reason': 'Sufficient data for tree-based models. Good for complex patterns.',
                'priority': 3,
            })
            recommendations.append({
                'model': 'LightGBM',
                'reason': 'Fast training on larger datasets with good accuracy.',
                'priority': 4,
            })

        # Naive - baseline
        recommendations.append({
            'model': 'Naive',
            'reason': 'Simple baseline model for comparison.',
            'priority': 5,
        })

        # Seasonal Naive - if seasonality detected
        if seasonality['detected']:
            recommendations.append({
                'model': 'Seasonal Naive',
                'reason': 'Simple seasonal baseline for comparison.',
                'priority': 6,
            })

        return recommendations
