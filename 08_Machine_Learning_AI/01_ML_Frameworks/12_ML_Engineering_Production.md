# ML Engineering and Production

*Duration: 2-3 weeks*

This comprehensive guide covers the essential aspects of taking machine learning models from development to production, ensuring reliability, scalability, and maintainability.

---

## ML Pipeline Design

A well-designed ML pipeline automates the flow from raw data to production predictions, ensuring consistency and reproducibility.

### Pipeline Components Overview

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment → Monitoring
    ↑                                                                            ↓
    └──────────────────── Feedback Loop ←─────────────────────────────────────────┘
```

### 1. Data Preprocessing

**Key Considerations:**
- **Data validation**: Check schema, data types, and constraints
- **Missing value handling**: Imputation strategies
- **Outlier detection**: Statistical methods and domain knowledge
- **Data cleaning**: Remove duplicates, fix inconsistencies

**Example: Robust Data Preprocessing Pipeline**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        
    def validate_data(self, df):
        """Validate input data schema and quality"""
        required_columns = ['feature1', 'feature2', 'feature3']
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
            
        # Check data types
        expected_types = {'feature1': 'float64', 'feature2': 'int64'}
        for col, expected_type in expected_types.items():
            if df[col].dtype != expected_type:
                logging.warning(f"Column {col} has type {df[col].dtype}, expected {expected_type}")
        
        # Check for excessive missing values
        missing_threshold = 0.5
        for col in df.columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > missing_threshold:
                logging.warning(f"Column {col} has {missing_ratio:.2%} missing values")
                
        return True
    
    def handle_outliers(self, df, columns, method='iqr'):
        """Remove or cap outliers using IQR method"""
        df_clean = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                
        return df_clean
    
    def fit_transform(self, df):
        """Fit preprocessors and transform data"""
        self.validate_data(df)
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle numerical variables
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df_clean = self.handle_outliers(df, numerical_cols)
        
        # Impute missing values
        df_imputed = pd.DataFrame(
            self.imputer.fit_transform(df_clean),
            columns=df_clean.columns
        )
        
        # Scale features
        df_scaled = pd.DataFrame(
            self.scaler.fit_transform(df_imputed),
            columns=df_imputed.columns
        )
        
        return df_scaled
```

### 2. Feature Engineering

**Advanced Feature Engineering Techniques:**
```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineer:
    def __init__(self):
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.feature_selector = SelectKBest(score_func=f_classif, k=10)
        
    def create_time_features(self, df, datetime_col):
        """Extract time-based features"""
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        df['year'] = df[datetime_col].dt.year
        df['month'] = df[datetime_col].dt.month
        df['day'] = df[datetime_col].dt.day
        df['dayofweek'] = df[datetime_col].dt.dayofweek
        df['hour'] = df[datetime_col].dt.hour
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Cyclical encoding for circular features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def create_interaction_features(self, X):
        """Create polynomial and interaction features"""
        return self.poly_features.fit_transform(X)
    
    def select_features(self, X, y):
        """Select most important features"""
        return self.feature_selector.fit_transform(X, y)
```

### 3. Model Training with Experiment Tracking

**Example with MLflow:**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelTrainer:
    def __init__(self, experiment_name="ml_production_experiment"):
        mlflow.set_experiment(experiment_name)
        
    def train_model(self, X_train, y_train, X_val, y_val, model_params=None):
        """Train model with experiment tracking"""
        
        with mlflow.start_run():
            # Log parameters
            if model_params:
                mlflow.log_params(model_params)
            
            # Train model
            model = RandomForestClassifier(**model_params if model_params else {})
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            y_pred = model.predict(X_val)
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_score,
                "val_accuracy": val_score,
                "precision": precision,
                "recall": recall
            })
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
            return model, val_score
```

---

## Model Serving

### Deployment Architecture Patterns

```
                    Load Balancer
                          │
              ┌───────────┼───────────┐
              │           │           │
         Model Server  Model Server  Model Server
         (Instance A)  (Instance B)  (Instance C)
              │           │           │
              └───────────┼───────────┘
                          │
                   Model Storage
                   (S3/GCS/Azure)
```

### 1. TensorFlow Serving

**Setup and Configuration:**
```bash
# Install TensorFlow Serving
pip install tensorflow-serving-api

# Save model in SavedModel format
import tensorflow as tf

# Export model
model.save('/path/to/saved_model/1')

# Start TensorFlow Serving
tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=my_model \
  --model_base_path=/path/to/saved_model
```

**Client Code:**
```python
import requests
import json
import numpy as np

def predict_tensorflow_serving(data):
    """Make predictions using TensorFlow Serving"""
    url = 'http://localhost:8501/v1/models/my_model:predict'
    
    # Prepare data
    payload = {
        "signature_name": "serving_default",
        "instances": data.tolist()
    }
    
    # Make request
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        predictions = response.json()['predictions']
        return np.array(predictions)
    else:
        raise Exception(f"Prediction failed: {response.text}")

# Example usage
test_data = np.random.random((1, 10))  # 1 sample, 10 features
predictions = predict_tensorflow_serving(test_data)
```

### 2. Custom FastAPI Model Server

**Production-Ready FastAPI Server:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from typing import List
import time
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load("model.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Request/Response models
class PredictionRequest(BaseModel):
    features: List[float]
    model_version: str = "v1.0"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    inference_time_ms: float

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction"""
    start_time = time.time()
    
    try:
        # Validate input
        if len(request.features) != 10:  # Expected feature count
            raise HTTPException(status_code=400, detail="Expected 10 features")
        
        # Prepare data
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability (if available)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
        else:
            confidence = 1.0
        
        inference_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=confidence,
            model_version=request.model_version,
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    results = []
    
    for req in requests:
        try:
            result = await predict(req)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results}

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.pkl
    volumes:
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ml-api
```

---

## ML Monitoring

Comprehensive monitoring ensures model performance and system health in production.

### 1. Performance Metrics Monitoring

**Metrics Collection System:**
```python
import time
from dataclasses import dataclass
from typing import Dict, Any
import logging
from collections import defaultdict
import numpy as np

@dataclass
class PredictionMetrics:
    """Store prediction metrics"""
    timestamp: float
    model_version: str
    inference_time_ms: float
    prediction_value: float
    confidence_score: float
    input_features: Dict[str, Any]

class ModelMonitor:
    def __init__(self):
        self.metrics = []
        self.performance_stats = defaultdict(list)
        
    def log_prediction(self, metrics: PredictionMetrics):
        """Log individual prediction metrics"""
        self.metrics.append(metrics)
        self.performance_stats['inference_times'].append(metrics.inference_time_ms)
        self.performance_stats['confidences'].append(metrics.confidence_score)
        
        # Alert on slow predictions
        if metrics.inference_time_ms > 1000:  # 1 second threshold
            logging.warning(f"Slow prediction: {metrics.inference_time_ms}ms")
    
    def get_performance_summary(self, window_hours: int = 24):
        """Get performance summary for the last N hours"""
        cutoff_time = time.time() - (window_hours * 3600)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        inference_times = [m.inference_time_ms for m in recent_metrics]
        confidences = [m.confidence_score for m in recent_metrics]
        
        return {
            'total_predictions': len(recent_metrics),
            'avg_inference_time_ms': np.mean(inference_times),
            'p95_inference_time_ms': np.percentile(inference_times, 95),
            'avg_confidence': np.mean(confidences),
            'low_confidence_ratio': sum(1 for c in confidences if c < 0.7) / len(confidences)
        }

# Integration with FastAPI
monitor = ModelMonitor()

@app.post("/predict")
async def predict_with_monitoring(request: PredictionRequest):
    start_time = time.time()
    
    # ... prediction logic ...
    
    # Log metrics
    metrics = PredictionMetrics(
        timestamp=time.time(),
        model_version="v1.0",
        inference_time_ms=(time.time() - start_time) * 1000,
        prediction_value=prediction,
        confidence_score=confidence,
        input_features=request.dict()
    )
    monitor.log_prediction(metrics)
    
    return response

@app.get("/metrics")
async def get_metrics():
    """Endpoint for monitoring system to scrape metrics"""
    return monitor.get_performance_summary()
```

### 2. Data Drift Detection

**Statistical Drift Detection:**
```python
import numpy as np
from scipy import stats
from typing import List, Dict
import pandas as pd

class DriftDetector:
    def __init__(self, reference_data: np.ndarray):
        """Initialize with reference dataset"""
        self.reference_data = reference_data
        self.reference_stats = self._calculate_stats(reference_data)
        
    def _calculate_stats(self, data: np.ndarray) -> Dict:
        """Calculate statistical properties of data"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'percentiles': np.percentile(data, [25, 50, 75], axis=0)
        }
    
    def detect_distribution_drift(self, new_data: np.ndarray, 
                                 threshold: float = 0.05) -> Dict:
        """Detect drift using Kolmogorov-Smirnov test"""
        drift_results = {}
        
        for i in range(new_data.shape[1]):
            # KS test for each feature
            ks_statistic, p_value = stats.ks_2samp(
                self.reference_data[:, i], 
                new_data[:, i]
            )
            
            drift_results[f'feature_{i}'] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'drift_detected': p_value < threshold
            }
        
        return drift_results
    
    def detect_statistical_drift(self, new_data: np.ndarray,
                                threshold_std: float = 2.0) -> Dict:
        """Detect drift in statistical properties"""
        new_stats = self._calculate_stats(new_data)
        drift_results = {}
        
        for i in range(new_data.shape[1]):
            # Check if mean has shifted significantly
            mean_diff = abs(new_stats['mean'][i] - self.reference_stats['mean'][i])
            std_threshold = threshold_std * self.reference_stats['std'][i]
            
            drift_results[f'feature_{i}'] = {
                'mean_shift': mean_diff,
                'threshold': std_threshold,
                'drift_detected': mean_diff > std_threshold,
                'reference_mean': self.reference_stats['mean'][i],
                'new_mean': new_stats['mean'][i]
            }
        
        return drift_results

# Usage in production
drift_detector = DriftDetector(reference_training_data)

# Check for drift periodically
def check_drift_daily():
    recent_data = get_recent_predictions_data()  # Last 24h of data
    
    distribution_drift = drift_detector.detect_distribution_drift(recent_data)
    statistical_drift = drift_detector.detect_statistical_drift(recent_data)
    
    # Alert if drift detected
    features_with_drift = [
        f for f, result in distribution_drift.items() 
        if result['drift_detected']
    ]
    
    if features_with_drift:
        send_alert(f"Data drift detected in features: {features_with_drift}")
```

---

## ML DevOps

### 1. Continuous Training Pipeline

**Automated Retraining with Apache Airflow:**
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
import mlflow

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_continuous_training',
    default_args=default_args,
    description='Continuous ML model training pipeline',
    schedule_interval='@daily',
    catchup=False
)

def extract_new_data():
    """Extract new training data"""
    # Implementation depends on your data source
    pass

def validate_data():
    """Validate data quality"""
    # Data validation logic
    pass

def train_model():
    """Train new model version"""
    with mlflow.start_run():
        # Training logic
        pass

def evaluate_model():
    """Evaluate model performance"""
    # Compare with current production model
    pass

def deploy_model():
    """Deploy model if it's better than current"""
    # Deployment logic
    pass

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_new_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Set dependencies
extract_task >> validate_task >> train_task >> evaluate_task >> deploy_task
```

### 2. Model Versioning with MLflow

**Model Registry Management:**
```python
import mlflow
from mlflow.tracking import MlflowClient

class ModelVersionManager:
    def __init__(self):
        self.client = MlflowClient()
        
    def register_model(self, model_name: str, run_id: str):
        """Register a new model version"""
        model_uri = f"runs:/{run_id}/model"
        
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        return result.version
    
    def promote_to_production(self, model_name: str, version: str):
        """Promote model version to production"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
    def archive_old_versions(self, model_name: str, keep_latest: int = 3):
        """Archive old model versions"""
        versions = self.client.search_model_versions(
            filter_string=f"name='{model_name}'"
        )
        
        # Sort by version number (descending)
        versions.sort(key=lambda x: int(x.version), reverse=True)
        
        # Archive old versions
        for version in versions[keep_latest:]:
            if version.current_stage != "Production":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
    
    def get_production_model(self, model_name: str):
        """Get current production model"""
        latest_version = self.client.get_latest_versions(
            model_name, 
            stages=["Production"]
        )
        
        if latest_version:
            model_uri = f"models:/{model_name}/Production"
            return mlflow.sklearn.load_model(model_uri)
        else:
            raise ValueError("No production model found")
```

### 3. A/B Testing Framework

**A/B Testing for Model Deployment:**
```python
import random
from enum import Enum
from typing import Dict, Any
import logging

class ModelVariant(Enum):
    CONTROL = "control"
    TREATMENT = "treatment"

class ABTestManager:
    def __init__(self, treatment_ratio: float = 0.1):
        """
        Initialize A/B test manager
        
        Args:
            treatment_ratio: Percentage of traffic to send to treatment model
        """
        self.treatment_ratio = treatment_ratio
        self.control_model = None
        self.treatment_model = None
        self.results = {"control": [], "treatment": []}
        
    def load_models(self, control_model_path: str, treatment_model_path: str):
        """Load both model variants"""
        self.control_model = mlflow.sklearn.load_model(control_model_path)
        self.treatment_model = mlflow.sklearn.load_model(treatment_model_path)
        
    def get_variant(self, user_id: str = None) -> ModelVariant:
        """Determine which model variant to use"""
        if user_id:
            # Consistent assignment based on user ID
            hash_value = hash(user_id) % 100
            return ModelVariant.TREATMENT if hash_value < (self.treatment_ratio * 100) else ModelVariant.CONTROL
        else:
            # Random assignment
            return ModelVariant.TREATMENT if random.random() < self.treatment_ratio else ModelVariant.CONTROL
    
    def predict(self, features: np.ndarray, user_id: str = None) -> Dict[str, Any]:
        """Make prediction using appropriate model variant"""
        variant = self.get_variant(user_id)
        
        start_time = time.time()
        
        if variant == ModelVariant.TREATMENT:
            prediction = self.treatment_model.predict(features)[0]
            model_used = "treatment"
        else:
            prediction = self.control_model.predict(features)[0]
            model_used = "control"
        
        inference_time = time.time() - start_time
        
        result = {
            "prediction": prediction,
            "model_variant": model_used,
            "inference_time": inference_time,
            "user_id": user_id
        }
        
        # Log for analysis
        self.results[model_used].append(result)
        
        return result
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze A/B test results"""
        control_predictions = [r["prediction"] for r in self.results["control"]]
        treatment_predictions = [r["prediction"] for r in self.results["treatment"]]
        
        control_times = [r["inference_time"] for r in self.results["control"]]
        treatment_times = [r["inference_time"] for r in self.results["treatment"]]
        
        return {
            "control_count": len(control_predictions),
            "treatment_count": len(treatment_predictions),
            "control_avg_prediction": np.mean(control_predictions) if control_predictions else 0,
            "treatment_avg_prediction": np.mean(treatment_predictions) if treatment_predictions else 0,
            "control_avg_latency": np.mean(control_times) if control_times else 0,
            "treatment_avg_latency": np.mean(treatment_times) if treatment_times else 0
        }

# Integration with FastAPI
ab_test_manager = ABTestManager(treatment_ratio=0.1)

@app.post("/predict_ab")
async def predict_with_ab_test(request: PredictionRequest, user_id: str = None):
    """Prediction endpoint with A/B testing"""
    features = np.array(request.features).reshape(1, -1)
    result = ab_test_manager.predict(features, user_id)
    
    return {
        "prediction": result["prediction"],
        "model_variant": result["model_variant"],
        "inference_time_ms": result["inference_time"] * 1000
    }
```

### 4. Rollback Strategies

**Safe Deployment with Rollback:**
```python
import time
from typing import Optional

class SafeDeploymentManager:
    def __init__(self):
        self.current_model = None
        self.previous_model = None
        self.deployment_start_time = None
        self.error_count = 0
        self.error_threshold = 10
        self.monitoring_window = 300  # 5 minutes
        
    def deploy_new_model(self, new_model, model_version: str):
        """Deploy new model with rollback capability"""
        logging.info(f"Deploying model version {model_version}")
        
        # Store previous model for rollback
        self.previous_model = self.current_model
        self.current_model = new_model
        self.deployment_start_time = time.time()
        self.error_count = 0
        
        logging.info("New model deployed successfully")
        
    def should_rollback(self) -> bool:
        """Check if rollback is needed based on error rate"""
        if not self.deployment_start_time:
            return False
            
        # Check if we're still in monitoring window
        time_since_deployment = time.time() - self.deployment_start_time
        if time_since_deployment > self.monitoring_window:
            return False
            
        # Check error threshold
        return self.error_count >= self.error_threshold
    
    def record_error(self):
        """Record an error and check if rollback is needed"""
        self.error_count += 1
        logging.warning(f"Error recorded. Total errors: {self.error_count}")
        
        if self.should_rollback():
            self.rollback()
    
    def rollback(self):
        """Rollback to previous model"""
        if self.previous_model is None:
            logging.error("No previous model available for rollback")
            return False
            
        logging.warning("Rolling back to previous model due to high error rate")
        
        # Swap models
        self.current_model = self.previous_model
        self.previous_model = None
        self.deployment_start_time = None
        self.error_count = 0
        
        # Send alert
        send_alert("Model rollback performed due to high error rate")
        
        return True
    
    def predict_safely(self, features: np.ndarray):
        """Make prediction with error handling and rollback logic"""
        try:
            prediction = self.current_model.predict(features)
            return prediction
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            self.record_error()
            raise

# Integration example
deployment_manager = SafeDeploymentManager()

@app.post("/predict_safe")
async def predict_with_rollback(request: PredictionRequest):
    """Prediction endpoint with automatic rollback"""
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = deployment_manager.predict_safely(features)
        
        return {"prediction": float(prediction[0])}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
```

---

## Best Practices Summary

### ✅ DO:
- **Implement comprehensive monitoring** for both model performance and system health
- **Use A/B testing** for safe model deployments
- **Version control everything**: code, data, models, and configurations
- **Automate testing** at every stage of the pipeline
- **Set up alerting** for drift detection and performance degradation
- **Document model assumptions** and limitations
- **Use feature stores** for consistent feature engineering
- **Implement circuit breakers** for external dependencies

### ❌ DON'T:
- Deploy models without proper testing
- Ignore data drift and model degradation
- Skip monitoring and alerting
- Use different preprocessing logic for training and serving
- Deploy without rollback capabilities
- Hardcode configurations
- Forget to handle edge cases and errors
- Skip security considerations

### 🛠️ Tools and Technologies

**MLOps Platforms:**
- MLflow, Kubeflow, Weights & Biases, Neptune
- AWS SageMaker, Google Vertex AI, Azure ML

**Serving Frameworks:**
- TensorFlow Serving, TorchServe, ONNX Runtime
- FastAPI, Flask, Seldon Core, KServe

**Monitoring:**
- Prometheus + Grafana, DataDog, New Relic
- Evidently AI, WhyLabs, Arize

**Deployment:**
- Docker, Kubernetes, AWS ECS, Google Cloud Run
- Terraform, Ansible, GitLab CI/CD

---

## Learning Path and Next Steps

1. **Week 1**: Master ML pipeline design and experiment tracking
2. **Week 2**: Learn model serving and deployment patterns
3. **Week 3**: Implement monitoring, A/B testing, and MLOps practices

**Hands-on Projects:**
- Build an end-to-end ML pipeline with MLflow
- Deploy a model API with FastAPI and Docker
- Implement drift detection and automated retraining
- Set up A/B testing for model comparison

**Further Reading:**
- "Building Machine Learning Powered Applications" by Emmanuel Ameisen
- "Machine Learning Engineering" by Andriy Burkov
- "Designing Machine Learning Systems" by Chip Huyen
