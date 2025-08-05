# Deployment Architectures

*Duration: 3 weeks*

## Overview

Deploying NLP models in production requires robust, scalable, and efficient architectures. This section covers comprehensive deployment strategies including microservices, containerization, orchestration, monitoring, and scaling patterns for production-ready NLP services.

## 1. Model Serving Infrastructure

### FastAPI-based Model Serving

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import torch
import time
import logging
import json
import redis
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import asynccontextmanager
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total model requests', ['model', 'status'])
REQUEST_DURATION = Histogram('model_request_duration_seconds', 'Request duration')
ACTIVE_REQUESTS = Gauge('model_active_requests', 'Currently active requests')
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

class ModelManager:
    """Centralized model management for serving"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_configs = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Redis for caching (optional)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_available = True
            logger.info("Redis connection established")
        except:
            self.redis_available = False
            logger.warning("Redis not available, caching disabled")
    
    async def load_model(self, model_name: str, model_path: str):
        """Load model and tokenizer"""
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            model.eval()
            
            # Store in memory
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.model_configs[model_name] = {
                'path': model_path,
                'loaded_at': time.time(),
                'request_count': 0
            }
            
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def get_model(self, model_name: str):
        """Get model and tokenizer"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        return self.models[model_name], self.tokenizers[model_name]
    
    def get_cache_key(self, model_name: str, prompt: str, parameters: dict) -> str:
        """Generate cache key for request"""
        cache_data = {
            'model': model_name,
            'prompt': prompt,
            'parameters': parameters
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def generate_with_cache(self, model_name: str, prompt: str, 
                                parameters: dict) -> dict:
        """Generate with caching support"""
        
        # Check cache first
        if self.redis_available:
            cache_key = self.get_cache_key(model_name, prompt, parameters)
            cached_result = self.redis_client.get(cache_key)
            
            if cached_result:
                logger.info("Cache hit")
                return json.loads(cached_result)
        
        # Generate new result
        model, tokenizer = self.get_model(model_name)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=parameters.get('max_new_tokens', 50),
                temperature=parameters.get('temperature', 1.0),
                do_sample=parameters.get('do_sample', True),
                top_p=parameters.get('top_p', 0.9),
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode result
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        result = {
            'generated_text': generated_text,
            'input_length': inputs['input_ids'].shape[1],
            'output_length': outputs.shape[1],
            'model': model_name,
            'cached': False
        }
        
        # Cache result
        if self.redis_available:
            self.redis_client.setex(
                cache_key, 
                3600,  # 1 hour TTL
                json.dumps(result)
            )
        
        # Update model stats
        self.model_configs[model_name]['request_count'] += 1
        
        return result

# Global model manager
model_manager = ModelManager()

# Request/Response models
class GenerationRequest(BaseModel):
    model: str = Field(..., description="Model name to use")
    prompt: str = Field(..., description="Input prompt")
    max_new_tokens: int = Field(50, ge=1, le=500)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    do_sample: bool = Field(True)

class GenerationResponse(BaseModel):
    generated_text: str
    input_length: int
    output_length: int
    model: str
    cached: bool
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    uptime: float
    system_info: Dict[str, Any]

class BatchGenerationRequest(BaseModel):
    model: str
    prompts: List[str] = Field(..., min_items=1, max_items=10)
    max_new_tokens: int = Field(50, ge=1, le=500)
    temperature: float = Field(1.0, ge=0.1, le=2.0)

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    
    # Startup
    logger.info("Starting model serving application")
    
    # Load default models
    await model_manager.load_model("gpt2", "gpt2")
    
    yield
    
    # Shutdown
    logger.info("Shutting down model serving application")

# Create FastAPI app
app = FastAPI(
    title="NLP Model Serving API",
    description="Production-ready NLP model serving with caching and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting (simple in-memory implementation)
from collections import defaultdict, deque

class RateLimiter:
    def __init__(self, max_requests=100, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(deque)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests
        while client_requests and client_requests[0] < now - self.time_window:
            client_requests.popleft()
        
        # Check limit
        if len(client_requests) >= self.max_requests:
            return False
        
        # Add new request
        client_requests.append(now)
        return True

rate_limiter = RateLimiter()

# Middleware for metrics and rate limiting
@app.middleware("http")
async def add_process_time_header(request, call_next):
    client_ip = request.client.host
    
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Metrics
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        REQUEST_COUNT.labels(model="unknown", status="success").inc()
        return response
    except Exception as e:
        REQUEST_COUNT.labels(model="unknown", status="error").inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()
        REQUEST_DURATION.observe(time.time() - start_time)

# Update system metrics
async def update_system_metrics():
    """Background task to update system metrics"""
    while True:
        try:
            # CPU usage
            CPU_USAGE.set(psutil.cpu_percent())
            
            # GPU usage (if available)
            if torch.cuda.is_available():
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        GPU_MEMORY_USAGE.set(gpus[0].memoryUsed * 1024 * 1024)  # Convert to bytes
                except:
                    pass
            
            await asyncio.sleep(10)  # Update every 10 seconds
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            await asyncio.sleep(10)

# Start background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(update_system_metrics())

# API Endpoints
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using specified model"""
    
    start_time = time.time()
    
    try:
        # Validate model
        if request.model not in model_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
        
        # Generate text
        parameters = {
            'max_new_tokens': request.max_new_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'do_sample': request.do_sample
        }
        
        result = await model_manager.generate_with_cache(
            request.model, 
            request.prompt, 
            parameters
        )
        
        processing_time = time.time() - start_time
        
        REQUEST_COUNT.labels(model=request.model, status="success").inc()
        
        return GenerationResponse(
            **result,
            processing_time=processing_time
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(model=request.model, status="error").inc()
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_generate")
async def batch_generate(request: BatchGenerationRequest):
    """Generate text for multiple prompts"""
    
    try:
        if request.model not in model_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
        
        # Process batch
        model, tokenizer = model_manager.get_model(request.model)
        
        # Tokenize all prompts
        inputs = tokenizer(
            request.prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model_manager.device)
        
        # Generate for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode results
        results = []
        for i, output in enumerate(outputs):
            # Skip input tokens
            generated_tokens = output[inputs['input_ids'][i].shape[0]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            results.append({
                'prompt': request.prompts[i],
                'generated_text': generated_text,
                'input_length': inputs['input_ids'][i].shape[0],
                'output_length': output.shape[0]
            })
        
        REQUEST_COUNT.labels(model=request.model, status="batch_success").inc()
        
        return {
            'model': request.model,
            'results': results,
            'batch_size': len(request.prompts)
        }
        
    except Exception as e:
        REQUEST_COUNT.labels(model=request.model, status="batch_error").inc()
        logger.error(f"Batch generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    # System info
    memory = psutil.virtual_memory()
    system_info = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': memory.percent,
        'memory_available': memory.available,
        'disk_usage': psutil.disk_usage('/').percent
    }
    
    if torch.cuda.is_available():
        system_info['gpu_available'] = True
        system_info['gpu_count'] = torch.cuda.device_count()
        system_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
    else:
        system_info['gpu_available'] = False
    
    return HealthResponse(
        status="healthy",
        models_loaded=list(model_manager.models.keys()),
        uptime=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        system_info=system_info
    )

@app.get("/models")
async def list_models():
    """List available models"""
    
    models_info = {}
    for name, config in model_manager.model_configs.items():
        models_info[name] = {
            'path': config['path'],
            'loaded_at': config['loaded_at'],
            'request_count': config['request_count']
        }
    
    return {
        'models': models_info,
        'total_models': len(models_info)
    }

@app.post("/models/{model_name}/load")
async def load_model(model_name: str, model_path: str):
    """Load a new model"""
    
    try:
        await model_manager.load_model(model_name, model_path)
        return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    
    if not model_manager.redis_available:
        return {"error": "Cache not available"}
    
    try:
        info = model_manager.redis_client.info()
        return {
            'keys': model_manager.redis_client.dbsize(),
            'memory_usage': info.get('used_memory_human', 'unknown'),
            'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_misses', 0) + info.get('keyspace_hits', 0))
        }
    except Exception as e:
        return {"error": str(e)}

# Set start time
@app.on_event("startup")
async def set_start_time():
    app.state.start_time = time.time()

# Example client code
class ModelClient:
    """Client for the model serving API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        import httpx
        self.client = httpx.AsyncClient()
    
    async def generate(self, model: str, prompt: str, **kwargs):
        """Generate text"""
        
        request_data = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        
        response = await self.client.post(f"{self.base_url}/generate", json=request_data)
        response.raise_for_status()
        
        return response.json()
    
    async def batch_generate(self, model: str, prompts: List[str], **kwargs):
        """Generate text for multiple prompts"""
        
        request_data = {
            "model": model,
            "prompts": prompts,
            **kwargs
        }
        
        response = await self.client.post(f"{self.base_url}/batch_generate", json=request_data)
        response.raise_for_status()
        
        return response.json()
    
    async def health_check(self):
        """Check service health"""
        
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        
        return response.json()

# Example usage
async def demo_model_serving():
    """Demonstrate model serving"""
    
    print("Model Serving Demo")
    print("=" * 30)
    
    # Start server (would run this separately in production)
    # uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
    
    # Example client usage
    client = ModelClient()
    
    try:
        # Health check
        health = await client.health_check()
        print(f"Service status: {health['status']}")
        print(f"Models loaded: {health['models_loaded']}")
        
        # Single generation
        result = await client.generate(
            model="gpt2",
            prompt="The future of AI is",
            max_new_tokens=30
        )
        print(f"\nGenerated: {result['generated_text']}")
        print(f"Processing time: {result['processing_time']:.3f}s")
        
        # Batch generation
        batch_result = await client.batch_generate(
            model="gpt2",
            prompts=[
                "Climate change is",
                "Technology enables",
                "Innovation drives"
            ],
            max_new_tokens=20
        )
        
        print(f"\nBatch results:")
        for i, result in enumerate(batch_result['results']):
            print(f"  {i+1}. {result['generated_text']}")
        
    except Exception as e:
        print(f"Error: {e}")

# Run the demo
# asyncio.run(demo_model_serving())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
```

## 2. Containerization and Orchestration

### Docker Configuration

```dockerfile
# Dockerfile for NLP model serving
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
transformers==4.35.0
accelerate==0.24.0
redis==5.0.1
prometheus-client==0.19.0
psutil==5.9.6
GPUtil==1.4.0
httpx==0.25.2
pydantic==2.5.0
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlp-model-serving
  labels:
    app: nlp-model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nlp-model-serving
  template:
    metadata:
      labels:
        app: nlp-model-serving
    spec:
      containers:
      - name: nlp-service
        image: nlp-model-serving:latest
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nlp-model-service
spec:
  selector:
    app: nlp-model-serving
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
# HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nlp-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nlp-model-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Redis Cache Configuration

```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis-cache
  template:
    metadata:
      labels:
        app: redis-cache
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - --maxmemory
        - 2gb
        - --maxmemory-policy
        - allkeys-lru
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  selector:
    app: redis-cache
  ports:
  - protocol: TCP
    port: 6379
    targetPort: 6379
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

## 3. Advanced Deployment Patterns

### Blue-Green Deployment Strategy

```python
import asyncio
import aiohttp
import time
from typing import Dict, List

class BlueGreenDeploymentManager:
    """Manage blue-green deployments for model serving"""
    
    def __init__(self, blue_endpoints: List[str], green_endpoints: List[str]):
        self.blue_endpoints = blue_endpoints
        self.green_endpoints = green_endpoints
        self.active_environment = "blue"  # Start with blue
        self.traffic_split = {"blue": 100, "green": 0}
        
        self.health_check_interval = 30
        self.deployment_stats = {
            "blue": {"requests": 0, "errors": 0, "avg_latency": 0},
            "green": {"requests": 0, "errors": 0, "avg_latency": 0}
        }
    
    async def health_check(self, endpoint: str) -> bool:
        """Check if endpoint is healthy"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "healthy"
            return False
        except:
            return False
    
    async def check_environment_health(self, environment: str) -> Dict[str, bool]:
        """Check health of all endpoints in an environment"""
        
        endpoints = self.blue_endpoints if environment == "blue" else self.green_endpoints
        health_results = {}
        
        for endpoint in endpoints:
            health_results[endpoint] = await self.health_check(endpoint)
        
        return health_results
    
    async def gradual_traffic_shift(self, target_environment: str, 
                                  shift_duration: int = 300):
        """Gradually shift traffic to target environment"""
        
        print(f"Starting gradual traffic shift to {target_environment}")
        
        steps = 10
        step_duration = shift_duration / steps
        
        for step in range(1, steps + 1):
            # Calculate new traffic split
            target_percentage = (step / steps) * 100
            
            if target_environment == "blue":
                self.traffic_split = {
                    "blue": target_percentage,
                    "green": 100 - target_percentage
                }
            else:
                self.traffic_split = {
                    "blue": 100 - target_percentage,
                    "green": target_percentage
                }
            
            print(f"Traffic split: Blue {self.traffic_split['blue']:.0f}%, "
                  f"Green {self.traffic_split['green']:.0f}%")
            
            # Wait before next step
            await asyncio.sleep(step_duration)
            
            # Check health of target environment
            health = await self.check_environment_health(target_environment)
            unhealthy_endpoints = [ep for ep, healthy in health.items() if not healthy]
            
            if unhealthy_endpoints:
                print(f"Unhealthy endpoints detected: {unhealthy_endpoints}")
                print("Rolling back traffic shift")
                await self.rollback_traffic()
                return False
        
        # Complete the shift
        self.active_environment = target_environment
        print(f"Traffic shift to {target_environment} completed successfully")
        return True
    
    async def rollback_traffic(self):
        """Rollback traffic to previous environment"""
        
        previous_env = "green" if self.active_environment == "blue" else "blue"
        
        print(f"Rolling back traffic to {previous_env}")
        
        self.traffic_split = {
            previous_env: 100,
            self.active_environment: 0
        }
        
        self.active_environment = previous_env
    
    def route_request(self) -> str:
        """Route request based on current traffic split"""
        
        import random
        
        if random.randint(1, 100) <= self.traffic_split["blue"]:
            endpoints = self.blue_endpoints
            environment = "blue"
        else:
            endpoints = self.green_endpoints
            environment = "green"
        
        # Simple round-robin within environment
        endpoint = random.choice(endpoints)
        self.deployment_stats[environment]["requests"] += 1
        
        return endpoint
    
    async def deploy_new_version(self, model_path: str, target_environment: str):
        """Deploy new model version to target environment"""
        
        print(f"Deploying new version to {target_environment} environment")
        
        endpoints = (self.blue_endpoints if target_environment == "blue" 
                    else self.green_endpoints)
        
        # Update each endpoint in the target environment
        deployment_results = {}
        
        for endpoint in endpoints:
            try:
                async with aiohttp.ClientSession() as session:
                    # Load new model
                    load_data = {
                        "model_name": "new_model",
                        "model_path": model_path
                    }
                    
                    async with session.post(
                        f"{endpoint}/models/new_model/load",
                        json=load_data,
                        timeout=300
                    ) as response:
                        
                        if response.status == 200:
                            deployment_results[endpoint] = "success"
                            print(f"✓ Deployed to {endpoint}")
                        else:
                            deployment_results[endpoint] = "failed"
                            print(f"✗ Failed to deploy to {endpoint}")
            
            except Exception as e:
                deployment_results[endpoint] = f"error: {str(e)}"
                print(f"✗ Error deploying to {endpoint}: {e}")
        
        # Check if deployment was successful
        successful_deployments = sum(1 for result in deployment_results.values() 
                                   if result == "success")
        
        if successful_deployments == len(endpoints):
            print(f"All deployments to {target_environment} successful")
            return True
        else:
            print(f"Some deployments failed: {deployment_results}")
            return False
    
    async def canary_deployment(self, model_path: str, canary_percentage: int = 10):
        """Perform canary deployment"""
        
        print(f"Starting canary deployment with {canary_percentage}% traffic")
        
        # Deploy to inactive environment
        inactive_env = "green" if self.active_environment == "blue" else "blue"
        
        # Deploy new version
        if not await self.deploy_new_version(model_path, inactive_env):
            print("Canary deployment failed during model deployment")
            return False
        
        # Start with small traffic percentage
        original_split = self.traffic_split.copy()
        
        if inactive_env == "blue":
            self.traffic_split = {
                "blue": canary_percentage,
                "green": 100 - canary_percentage
            }
        else:
            self.traffic_split = {
                "blue": 100 - canary_percentage,
                "green": canary_percentage
            }
        
        print(f"Canary traffic split: Blue {self.traffic_split['blue']}%, "
              f"Green {self.traffic_split['green']}%")
        
        # Monitor for specified duration
        monitoring_duration = 300  # 5 minutes
        
        await asyncio.sleep(monitoring_duration)
        
        # Check canary environment health and metrics
        canary_health = await self.check_environment_health(inactive_env)
        unhealthy = [ep for ep, healthy in canary_health.items() if not healthy]
        
        if unhealthy:
            print(f"Canary environment unhealthy: {unhealthy}")
            print("Rolling back canary deployment")
            self.traffic_split = original_split
            return False
        
        # Check error rates and latency
        canary_stats = self.deployment_stats[inactive_env]
        if canary_stats["requests"] > 0:
            error_rate = canary_stats["errors"] / canary_stats["requests"]
            if error_rate > 0.05:  # 5% error threshold
                print(f"Canary error rate too high: {error_rate:.2%}")
                print("Rolling back canary deployment")
                self.traffic_split = original_split
                return False
        
        print("Canary deployment successful, proceeding with full deployment")
        return await self.gradual_traffic_shift(inactive_env)

# Example usage
async def deployment_demo():
    """Demonstrate blue-green and canary deployments"""
    
    print("Deployment Strategy Demo")
    print("=" * 30)
    
    # Mock endpoints
    blue_endpoints = ["http://blue-1:8000", "http://blue-2:8000"]
    green_endpoints = ["http://green-1:8000", "http://green-2:8000"]
    
    manager = BlueGreenDeploymentManager(blue_endpoints, green_endpoints)
    
    print(f"Active environment: {manager.active_environment}")
    print(f"Traffic split: {manager.traffic_split}")
    
    # Simulate canary deployment
    # success = await manager.canary_deployment("/models/new-gpt2", canary_percentage=20)
    # print(f"Canary deployment result: {success}")
    
    # Demonstrate request routing
    print("\nRequest routing simulation:")
    for i in range(10):
        endpoint = manager.route_request()
        print(f"Request {i+1} routed to: {endpoint}")

# asyncio.run(deployment_demo())
```

## 4. Monitoring and Observability

### Comprehensive Monitoring System

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import logging
import json
import time
from datetime import datetime
import asyncio
import aiohttp
from typing import Dict, List, Optional

class ModelServingMonitor:
    """Comprehensive monitoring for model serving"""
    
    def __init__(self):
        # Prometheus metrics
        self.request_counter = Counter(
            'model_requests_total',
            'Total requests processed',
            ['model', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'model_request_duration_seconds',
            'Request processing time',
            ['model', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
        )
        
        self.model_load_time = Histogram(
            'model_load_time_seconds',
            'Time to load models',
            ['model'],
            buckets=[1, 5, 10, 30, 60, 120, float('inf')]
        )
        
        self.active_requests = Gauge(
            'model_active_requests',
            'Currently processing requests',
            ['model', 'endpoint']
        )
        
        self.gpu_memory_usage = Gauge(
            'gpu_memory_usage_bytes',
            'GPU memory usage',
            ['gpu_id']
        )
        
        self.model_cache_hits = Counter(
            'model_cache_hits_total',
            'Cache hits',
            ['model']
        )
        
        self.model_cache_misses = Counter(
            'model_cache_misses_total',
            'Cache misses',
            ['model']
        )
        
        self.error_counter = Counter(
            'model_errors_total',
            'Total errors',
            ['model', 'error_type']
        )
        
        # Alerting thresholds
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5%
            'latency_p95': 5.0,  # 5 seconds
            'gpu_memory': 0.9,   # 90%
            'cache_hit_rate': 0.7  # 70%
        }
        
        # Alert history
        self.alerts = []
        
        # Performance data
        self.performance_data = {
            'requests_per_minute': [],
            'average_latency': [],
            'error_rates': [],
            'cache_hit_rates': []
        }
    
    def record_request(self, model: str, endpoint: str, status: str, 
                      duration: float, cached: bool = False):
        """Record request metrics"""
        
        # Update counters
        self.request_counter.labels(
            model=model, 
            endpoint=endpoint, 
            status=status
        ).inc()
        
        # Update duration
        self.request_duration.labels(
            model=model, 
            endpoint=endpoint
        ).observe(duration)
        
        # Update cache metrics
        if cached:
            self.model_cache_hits.labels(model=model).inc()
        else:
            self.model_cache_misses.labels(model=model).inc()
        
        # Record error if applicable
        if status == 'error':
            self.error_counter.labels(
                model=model, 
                error_type='generation_error'
            ).inc()
    
    def record_model_load(self, model: str, load_time: float):
        """Record model loading time"""
        
        self.model_load_time.labels(model=model).observe(load_time)
    
    def update_gpu_metrics(self):
        """Update GPU usage metrics"""
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpus):
                memory_used = gpu.memoryUsed * 1024 * 1024  # Convert to bytes
                self.gpu_memory_usage.labels(gpu_id=str(i)).set(memory_used)
        except ImportError:
            pass
        except Exception as e:
            print(f"Error updating GPU metrics: {e}")
    
    def check_alerts(self) -> List[Dict]:
        """Check for alert conditions"""
        
        alerts = []
        current_time = datetime.now()
        
        # Get recent metrics (last 5 minutes)
        try:
            # Error rate check
            total_requests = sum([
                metric.samples[0].value for metric in 
                prometheus_client.REGISTRY.collect() 
                if metric.name == 'model_requests_total'
            ])
            
            error_requests = sum([
                sample.value for metric in prometheus_client.REGISTRY.collect()
                if metric.name == 'model_requests_total'
                for sample in metric.samples
                if 'status="error"' in str(sample.labels)
            ])
            
            if total_requests > 0:
                error_rate = error_requests / total_requests
                if error_rate > self.alert_thresholds['error_rate']:
                    alerts.append({
                        'type': 'high_error_rate',
                        'severity': 'critical',
                        'message': f'Error rate {error_rate:.2%} exceeds threshold {self.alert_thresholds["error_rate"]:.2%}',
                        'timestamp': current_time.isoformat()
                    })
            
            # GPU memory check
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_used = torch.cuda.memory_allocated(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_ratio = memory_used / memory_total
                    
                    if memory_ratio > self.alert_thresholds['gpu_memory']:
                        alerts.append({
                            'type': 'high_gpu_memory',
                            'severity': 'warning',
                            'message': f'GPU {i} memory usage {memory_ratio:.1%} exceeds threshold',
                            'timestamp': current_time.isoformat()
                        })
            
            # Cache hit rate check
            cache_hits = sum([
                metric.samples[0].value for metric in 
                prometheus_client.REGISTRY.collect() 
                if metric.name == 'model_cache_hits_total'
            ])
            
            cache_misses = sum([
                metric.samples[0].value for metric in 
                prometheus_client.REGISTRY.collect() 
                if metric.name == 'model_cache_misses_total'
            ])
            
            if (cache_hits + cache_misses) > 0:
                hit_rate = cache_hits / (cache_hits + cache_misses)
                if hit_rate < self.alert_thresholds['cache_hit_rate']:
                    alerts.append({
                        'type': 'low_cache_hit_rate',
                        'severity': 'warning',
                        'message': f'Cache hit rate {hit_rate:.2%} below threshold',
                        'timestamp': current_time.isoformat()
                    })
        
        except Exception as e:
            alerts.append({
                'type': 'monitoring_error',
                'severity': 'warning',
                'message': f'Error checking alerts: {str(e)}',
                'timestamp': current_time.isoformat()
            })
        
        # Store alerts
        self.alerts.extend(alerts)
        
        return alerts
    
    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard"""
        
        try:
            # Collect current metrics
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'total_requests': 0,
                'active_requests': 0,
                'error_rate': 0,
                'average_latency': 0,
                'cache_hit_rate': 0,
                'gpu_memory_usage': {},
                'recent_alerts': self.alerts[-10:],  # Last 10 alerts
                'models': {}
            }
            
            # Extract metrics from Prometheus registry
            for family in prometheus_client.REGISTRY.collect():
                if family.name == 'model_requests_total':
                    dashboard_data['total_requests'] = sum(sample.value for sample in family.samples)
                
                elif family.name == 'model_active_requests':
                    dashboard_data['active_requests'] = sum(sample.value for sample in family.samples)
                
                elif family.name == 'gpu_memory_usage_bytes':
                    for sample in family.samples:
                        gpu_id = sample.labels.get('gpu_id', 'unknown')
                        dashboard_data['gpu_memory_usage'][gpu_id] = sample.value
            
            # Calculate derived metrics
            cache_hits = sum([
                sample.value for family in prometheus_client.REGISTRY.collect()
                if family.name == 'model_cache_hits_total'
                for sample in family.samples
            ])
            
            cache_misses = sum([
                sample.value for family in prometheus_client.REGISTRY.collect()
                if family.name == 'model_cache_misses_total'
                for sample in family.samples
            ])
            
            if (cache_hits + cache_misses) > 0:
                dashboard_data['cache_hit_rate'] = cache_hits / (cache_hits + cache_misses)
            
            return dashboard_data
        
        except Exception as e:
            return {
                'error': f'Failed to collect dashboard data: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def send_alert(self, alert: Dict, webhook_url: Optional[str] = None):
        """Send alert to external system"""
        
        if webhook_url:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=alert) as response:
                        if response.status == 200:
                            print(f"Alert sent successfully: {alert['type']}")
                        else:
                            print(f"Failed to send alert: {response.status}")
            except Exception as e:
                print(f"Error sending alert: {e}")
        else:
            # Log alert
            print(f"ALERT [{alert['severity']}]: {alert['message']}")
    
    async def continuous_monitoring(self, check_interval: int = 60):
        """Run continuous monitoring"""
        
        print("Starting continuous monitoring...")
        
        while True:
            try:
                # Update system metrics
                self.update_gpu_metrics()
                
                # Check for alerts
                new_alerts = self.check_alerts()
                
                # Send alerts
                for alert in new_alerts:
                    await self.send_alert(alert)
                
                # Log current status
                dashboard_data = self.get_dashboard_data()
                print(f"Monitoring - Requests: {dashboard_data['total_requests']}, "
                      f"Active: {dashboard_data['active_requests']}, "
                      f"Cache Hit Rate: {dashboard_data['cache_hit_rate']:.2%}")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(check_interval)

# Example integration with FastAPI
from fastapi import FastAPI

def add_monitoring_to_app(app: FastAPI, monitor: ModelServingMonitor):
    """Add monitoring endpoints to FastAPI app"""
    
    @app.get("/monitoring/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint"""
        return prometheus_client.generate_latest()
    
    @app.get("/monitoring/dashboard")
    async def dashboard_endpoint():
        """Dashboard data endpoint"""
        return monitor.get_dashboard_data()
    
    @app.get("/monitoring/alerts")
    async def alerts_endpoint():
        """Get recent alerts"""
        return {
            'alerts': monitor.alerts[-50:],  # Last 50 alerts
            'alert_count': len(monitor.alerts)
        }
    
    @app.get("/monitoring/health")
    async def detailed_health():
        """Detailed health check with metrics"""
        
        dashboard_data = monitor.get_dashboard_data()
        
        # Determine health status
        status = "healthy"
        issues = []
        
        if dashboard_data['error_rate'] > monitor.alert_thresholds['error_rate']:
            status = "degraded"
            issues.append(f"High error rate: {dashboard_data['error_rate']:.2%}")
        
        if any(alert['severity'] == 'critical' for alert in dashboard_data['recent_alerts']):
            status = "unhealthy"
            issues.append("Critical alerts present")
        
        return {
            'status': status,
            'issues': issues,
            'metrics': dashboard_data
        }

# Example usage
def monitoring_demo():
    """Demonstrate monitoring system"""
    
    print("Model Serving Monitoring Demo")
    print("=" * 40)
    
    monitor = ModelServingMonitor()
    
    # Simulate some requests
    for i in range(100):
        monitor.record_request(
            model="gpt2",
            endpoint="server-1",
            status="success" if i < 95 else "error",
            duration=0.5 + (i % 10) * 0.1,
            cached=(i % 3 == 0)
        )
    
    # Check alerts
    alerts = monitor.check_alerts()
    print(f"Generated {len(alerts)} alerts")
    
    for alert in alerts:
        print(f"  [{alert['severity']}] {alert['message']}")
    
    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print(f"\nDashboard Data:")
    print(f"  Total requests: {dashboard['total_requests']}")
    print(f"  Cache hit rate: {dashboard['cache_hit_rate']:.2%}")
    print(f"  Recent alerts: {len(dashboard['recent_alerts'])}")

# monitoring_demo()
```

## 5. Learning Objectives

By the end of this section, you should be able to:
- **Design** scalable model serving architectures
- **Implement** production-ready API services with FastAPI
- **Deploy** containerized applications with Docker and Kubernetes
- **Manage** blue-green and canary deployments
- **Monitor** system performance and set up alerting
- **Optimize** serving infrastructure for cost and performance

### Self-Assessment Checklist

□ Can design RESTful APIs for model serving  
□ Can containerize applications with Docker  
□ Can deploy to Kubernetes with proper scaling  
□ Can implement caching and rate limiting  
□ Can set up comprehensive monitoring  
□ Can manage deployment strategies  
□ Can troubleshoot production issues  

## 6. Practical Exercises

**Exercise 1: Complete Model Serving API**
```python
# TODO: Implement a full-featured model serving API
# Include authentication, rate limiting, caching, and monitoring
# Support multiple models and batch processing
```

**Exercise 2: Kubernetes Deployment**
```yaml
# TODO: Create complete Kubernetes manifests
# Include deployments, services, ingress, and autoscaling
# Set up monitoring and logging
```

**Exercise 3: Load Testing and Optimization**
```python
# TODO: Implement load testing suite
# Test different scaling configurations
# Optimize for latency and throughput
```

## 7. Study Materials

### Essential Resources
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Kubernetes Patterns**: https://kubernetes.io/docs/concepts/
- **Prometheus Monitoring**: https://prometheus.io/docs/
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/

### Production Deployment
- **Cloud Platforms**: AWS SageMaker, Google AI Platform, Azure ML
- **Container Orchestration**: Kubernetes, Docker Swarm
- **Service Mesh**: Istio, Linkerd
- **API Gateways**: Kong, Ambassador, Envoy

### Monitoring and Observability
```bash
# Essential monitoring stack
kubectl apply -f prometheus-operator.yaml
kubectl apply -f grafana.yaml
kubectl apply -f alertmanager.yaml
```