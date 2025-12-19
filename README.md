# ML Forecast Service

Python-based machine learning service for time series forecasting.

## Features

- **FastAPI** REST API for ML operations
- **Celery** for asynchronous task processing
- **Redis** for task queue and caching
- **PostgreSQL** for data persistence
- **Multiple ML models**: Prophet, ARIMA, XGBoost, LightGBM

## Architecture

```
┌─────────────────────┐
│  NestJS Backend     │
│  (ml-forecast-      │
│   backend)          │
└──────────┬──────────┘
           │ HTTP
           ▼
┌─────────────────────┐
│  FastAPI ML Service │
│  (Port 8001)        │
└──────────┬──────────┘
           │
           ├─────────► Redis (Celery Queue)
           │                    │
           │                    ▼
           │           ┌─────────────────┐
           │           │  Celery Worker  │
           │           │  (ML Tasks)     │
           │           └─────────────────┘
           │
           └─────────► PostgreSQL (ml_forecast schema)
```

## Setup

### Local Development

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Create `.env` file:**

```bash
cp env.example .env
# Edit .env with your configuration
```

3. **Run the service:**

```bash
uvicorn main:app --reload --port 8001
```

4. **Run Celery worker:**

```bash
celery -A celery_app worker --loglevel=info
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose -f ../docker-compose.ml-forecast.yml up ml-forecast-service ml-forecast-worker
```

## API Endpoints

### Health Check

```
GET /health
```

### Exploratory Data Analysis

```
POST /ml/eda
{
  "dataset_id": "uuid",
  "date_column": "date",
  "target_column": "value",
  "feature_columns": ["feature1", "feature2"],
  "group_column": "region",
  "frequency": "D"
}
```

### Start Training

```
POST /ml/training/start
{
  "project_id": "uuid",
  "dataset_id": "uuid",
  "model_type": "prophet",
  "parameters": {},
  "date_column": "date",
  "target_column": "value",
  "feature_columns": ["feature1"],
  "frequency": "D"
}
```

Returns:

```json
{
  "job_id": "job_uuid_prophet",
  "status": "queued",
  "message": "Training job queued successfully"
}
```

### Get Training Status

```
GET /ml/training/status/{job_id}
```

Returns:

```json
{
  "job_id": "job_uuid_prophet",
  "status": "in_progress",
  "progress": 50,
  "current_model": "prophet",
  "completed": 1,
  "total": 3
}
```

### Generate Forecast

```
POST /ml/forecast
{
  "project_id": "uuid",
  "model_id": "prophet_uuid",
  "horizon": 30,
  "parameters": {}
}
```

## Supported Models

### Statistical Models

- **Naive**: Simple baseline (last value)
- **Seasonal Naive**: Seasonal baseline
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA
- **Prophet**: Facebook's forecasting tool

### Machine Learning Models

- **XGBoost**: Gradient boosting
- **LightGBM**: Light gradient boosting
- **Random Forest**: Ensemble learning

## Development

### Adding a New Model

1. Create trainer in `trainers/` directory
2. Implement `train()` and `predict()` methods
3. Register model in `tasks.py`
4. Update API documentation

### Testing

```bash
pytest tests/
```

## Environment Variables

| Variable                | Description           | Default                    |
| ----------------------- | --------------------- | -------------------------- |
| `DB_HOST`               | PostgreSQL host       | `localhost`                |
| `DB_PORT`               | PostgreSQL port       | `5432`                     |
| `DB_NAME`               | Database name         | `wfm_db`                   |
| `DB_USERNAME`           | Database user         | `postgres`                 |
| `DB_PASSWORD`           | Database password     | `mysecretpassword`         |
| `DB_SCHEMA`             | Database schema       | `ml_forecast`              |
| `REDIS_HOST`            | Redis host            | `localhost`                |
| `REDIS_PORT`            | Redis port            | `6379`                     |
| `CELERY_BROKER_URL`     | Celery broker URL     | `redis://localhost:6379/0` |
| `CELERY_RESULT_BACKEND` | Celery result backend | `redis://localhost:6379/0` |
| `MODELS_PATH`           | Path to save models   | `/shared/models`           |
| `DATASETS_PATH`         | Path to save datasets | `/shared/datasets`         |

## License

MIT
