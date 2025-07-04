# Monitoring and Observability in Microservices

*Duration: 2-3 weeks*

## Overview

Monitoring and observability are critical for maintaining healthy microservices architectures. While **monitoring** tells you that something is wrong, **observability** helps you understand why it's wrong. This comprehensive guide covers the three pillars of observability: metrics, logs, and traces.

## Core Concepts

### The Three Pillars of Observability

#### 1. Metrics
Quantitative measurements over time that provide insights into system performance and health.

#### 2. Logs
Detailed records of events that occur within your system, providing context and debugging information.

#### 3. Traces
End-to-end request flow tracking across multiple services, showing how requests propagate through your system.

```
┌─────────────────────────────────────────────────────────────┐
│                    Observability Stack                     │
├─────────────────────────────────────────────────────────────┤
│  📊 Metrics     │  📝 Logs        │  🔍 Traces             │
│  - CPU usage    │  - Error logs   │  - Request flow        │
│  - Memory       │  - Access logs  │  - Latency breakdown   │
│  - Latency      │  - Debug info   │  - Service dependencies│
│  - Throughput   │  - Audit trails │  - Error propagation   │
└─────────────────────────────────────────────────────────────┘
```

## Distributed Tracing

### Understanding Distributed Tracing

Distributed tracing tracks requests as they flow through multiple microservices, creating a complete picture of how a single user request is processed across your entire system.

#### Key Concepts

**Trace**: A complete journey of a request through the system
**Span**: A single operation within a trace (e.g., database query, HTTP call)
**Trace ID**: Unique identifier for the entire trace
**Span ID**: Unique identifier for each span within a trace

#### OpenTelemetry Implementation

```python
# requirements.txt
"""
opentelemetry-api==1.20.0
opentelemetry-sdk==1.20.0
opentelemetry-instrumentation-flask==0.41b0
opentelemetry-instrumentation-requests==0.41b0
opentelemetry-instrumentation-sqlalchemy==0.41b0
opentelemetry-exporter-jaeger-thrift==1.20.0
opentelemetry-exporter-prometheus==1.12.0rc1
"""

# tracing_config.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

def configure_tracing(service_name: str):
    """Configure OpenTelemetry tracing with Jaeger exporter"""
    
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument common libraries
    FlaskInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()
    
    return tracer

# user_service.py
from flask import Flask, request, jsonify
import requests
import time
import random
from tracing_config import configure_tracing

app = Flask(__name__)
tracer = configure_tracing("user-service")

@app.route('/users/<user_id>')
def get_user(user_id):
    """Get user information with distributed tracing"""
    
    with tracer.start_as_current_span("get_user") as span:
        # Add custom attributes to span
        span.set_attribute("user.id", user_id)
        span.set_attribute("service.name", "user-service")
        
        try:
            # Simulate database lookup
            with tracer.start_as_current_span("database_query") as db_span:
                db_span.set_attribute("db.operation", "SELECT")
                db_span.set_attribute("db.table", "users")
                
                # Simulate DB latency
                time.sleep(random.uniform(0.01, 0.05))
                
                user_data = {
                    "id": user_id,
                    "name": f"User {user_id}",
                    "email": f"user{user_id}@example.com"
                }
            
            # Call order service to get user's orders
            with tracer.start_as_current_span("call_order_service") as order_span:
                order_span.set_attribute("http.method", "GET")
                order_span.set_attribute("http.url", f"http://order-service/orders/user/{user_id}")
                
                try:
                    orders_response = requests.get(
                        f"http://order-service:5001/orders/user/{user_id}",
                        timeout=5
                    )
                    user_data["orders"] = orders_response.json()
                    order_span.set_attribute("http.status_code", orders_response.status_code)
                    
                except requests.exceptions.RequestException as e:
                    order_span.set_attribute("error", True)
                    order_span.set_attribute("error.message", str(e))
                    user_data["orders"] = []
            
            span.set_attribute("user.orders_count", len(user_data.get("orders", [])))
            return jsonify(user_data)
            
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            return jsonify({"error": "Internal server error"}), 500

# order_service.py
from flask import Flask, jsonify
import time
import random
from tracing_config import configure_tracing

app = Flask(__name__)
tracer = configure_tracing("order-service")

@app.route('/orders/user/<user_id>')
def get_user_orders(user_id):
    """Get orders for a specific user"""
    
    with tracer.start_as_current_span("get_user_orders") as span:
        span.set_attribute("user.id", user_id)
        span.set_attribute("service.name", "order-service")
        
        # Simulate database query
        with tracer.start_as_current_span("orders_db_query") as db_span:
            db_span.set_attribute("db.operation", "SELECT")
            db_span.set_attribute("db.table", "orders")
            
            # Simulate variable DB performance
            query_time = random.uniform(0.02, 0.1)
            time.sleep(query_time)
            db_span.set_attribute("db.query_time", query_time)
            
            # Generate mock orders
            orders = [
                {"id": f"order_{i}", "amount": random.randint(10, 500)}
                for i in range(random.randint(0, 5))
            ]
        
        span.set_attribute("orders.count", len(orders))
        return jsonify(orders)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

#### Jaeger Setup with Docker

```yaml
# docker-compose.jaeger.yml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "6831:6831/udp"  # Jaeger agent
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - monitoring

  user-service:
    build: ./user-service
    ports:
      - "5000:5000"
    environment:
      - JAEGER_AGENT_HOST=jaeger
    depends_on:
      - jaeger
    networks:
      - monitoring

  order-service:
    build: ./order-service
    ports:
      - "5001:5001"
    environment:
      - JAEGER_AGENT_HOST=jaeger
    depends_on:
      - jaeger
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge
```

## Structured Logging

### Advanced Logging Strategies

Structured logging is essential for microservices as it allows for better searchability, filtering, and analysis of logs across distributed systems.

#### JSON Structured Logging

```python
# logging_config.py
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any
import traceback

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": getattr(record, 'service', 'unknown'),
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add trace context if available
        if hasattr(record, 'trace_id'):
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, 'span_id'):
            log_entry["span_id"] = record.span_id
        
        # Add custom fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stacktrace": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)

def setup_logging(service_name: str, log_level: str = "INFO"):
    """Configure structured logging for a microservice"""
    
    # Create custom logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Enhanced logging with context
class ContextualLogger:
    """Logger that maintains request context"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = setup_logging(service_name)
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set context for all subsequent log messages"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context"""
        self.context.clear()
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with current context"""
        extra_fields = {**self.context, **kwargs}
        
        # Create log record with extra fields
        record = self.logger.makeRecord(
            self.service_name, 
            getattr(logging, level.upper()),
            __file__, 
            0, 
            message, 
            (), 
            None
        )
        record.service = self.service_name
        record.extra_fields = extra_fields
        
        self.logger.handle(record)
    
    def info(self, message: str, **kwargs):
        self._log_with_context("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context("ERROR", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context("DEBUG", message, **kwargs)

# service_with_logging.py
from flask import Flask, request, g
import uuid
import time
from opentelemetry import trace
from logging_config import ContextualLogger

app = Flask(__name__)
logger = ContextualLogger("payment-service")

@app.before_request
def before_request():
    """Set up request context for logging"""
    # Generate or extract request ID
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    
    # Get current span for trace context
    current_span = trace.get_current_span()
    if current_span:
        span_context = current_span.get_span_context()
        trace_id = f"{span_context.trace_id:032x}"
        span_id = f"{span_context.span_id:016x}"
    else:
        trace_id = None
        span_id = None
    
    # Set logging context
    logger.set_context(
        request_id=request_id,
        trace_id=trace_id,
        span_id=span_id,
        method=request.method,
        path=request.path,
        user_agent=request.headers.get('User-Agent'),
        client_ip=request.remote_addr
    )
    
    # Store in Flask g for use in views
    g.request_id = request_id
    g.start_time = time.time()
    
    logger.info("Request started", 
                endpoint=request.endpoint,
                args=dict(request.args))

@app.after_request
def after_request(response):
    """Log request completion"""
    duration = time.time() - g.start_time
    
    logger.info("Request completed",
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
                content_length=response.content_length)
    
    # Clear context for next request
    logger.clear_context()
    return response

@app.route('/payments/<payment_id>')
def get_payment(payment_id):
    """Get payment information with comprehensive logging"""
    
    logger.info("Processing payment request", payment_id=payment_id)
    
    try:
        # Validate payment ID
        if not payment_id.isdigit():
            logger.warning("Invalid payment ID format", 
                          payment_id=payment_id,
                          error_type="validation_error")
            return {"error": "Invalid payment ID"}, 400
        
        # Simulate database lookup
        logger.debug("Querying database for payment", 
                    query_type="payment_lookup",
                    payment_id=payment_id)
        
        time.sleep(0.1)  # Simulate DB latency
        
        payment = {
            "id": payment_id,
            "amount": 150.00,
            "status": "completed",
            "currency": "USD"
        }
        
        logger.info("Payment retrieved successfully",
                   payment_id=payment_id,
                   amount=payment["amount"],
                   status=payment["status"])
        
        return payment
        
    except Exception as e:
        logger.error("Failed to retrieve payment",
                    payment_id=payment_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True)
        
        return {"error": "Internal server error"}, 500

@app.route('/payments', methods=['POST'])
def create_payment():
    """Create new payment with audit logging"""
    
    payment_data = request.get_json()
    
    logger.info("Creating new payment",
               amount=payment_data.get('amount'),
               currency=payment_data.get('currency'),
               user_id=payment_data.get('user_id'))
    
    try:
        # Validate payment data
        required_fields = ['amount', 'currency', 'user_id']
        missing_fields = [field for field in required_fields 
                         if field not in payment_data]
        
        if missing_fields:
            logger.warning("Payment creation failed - missing fields",
                          missing_fields=missing_fields,
                          provided_fields=list(payment_data.keys()))
            return {"error": "Missing required fields", 
                   "missing": missing_fields}, 400
        
        # Create payment (simulate)
        payment_id = str(uuid.uuid4())
        
        logger.info("Payment created successfully",
                   payment_id=payment_id,
                   amount=payment_data['amount'],
                   currency=payment_data['currency'],
                   user_id=payment_data['user_id'],
                   event_type="payment_created")
        
        return {"payment_id": payment_id, "status": "created"}, 201
        
    except Exception as e:
        logger.error("Payment creation failed",
                    payment_data=payment_data,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True)
        
        return {"error": "Failed to create payment"}, 500

if __name__ == '__main__':
    logger.info("Payment service starting", port=5002)
    app.run(host='0.0.0.0', port=5002)
```

#### Log Aggregation with ELK Stack

```yaml
# docker-compose.elk.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.10.2
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - elk

  logstash:
    image: docker.elastic.co/logstash/logstash:8.10.2
    volumes:
      - ./logstash/config:/usr/share/logstash/pipeline
    ports:
      - "5044:5044"
      - "9600:9600"
    environment:
      - "LS_JAVA_OPTS=-Xmx256m -Xms256m"
    depends_on:
      - elasticsearch
    networks:
      - elk

  kibana:
    image: docker.elastic.co/kibana/kibana:8.10.2
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - elk

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.10.2
    user: root
    volumes:
      - ./filebeat/config/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    depends_on:
      - logstash
    networks:
      - elk

volumes:
  elasticsearch_data:

networks:
  elk:
    driver: bridge
```

```yaml
# filebeat/config/filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
    - add_docker_metadata:
        host: "unix:///var/run/docker.sock"
    - decode_json_fields:
        fields: ["message"]
        target: ""
        overwrite_keys: true

output.logstash:
  hosts: ["logstash:5044"]

logging.level: info
```

```ruby
# logstash/config/logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  # Parse JSON log messages
  if [message] =~ /^{.*}$/ {
    json {
      source => "message"
    }
    
    # Parse timestamp
    if [timestamp] {
      date {
        match => [ "timestamp", "ISO8601" ]
      }
    }
    
    # Add custom fields
    mutate {
      add_field => { "environment" => "development" }
      add_field => { "log_type" => "microservice" }
    }
  }
  
  # Tag error logs
  if [level] == "ERROR" {
    mutate {
      add_tag => [ "error" ]
    }
  }
  
  # Tag slow requests (>1000ms)
  if [duration_ms] and [duration_ms] > 1000 {
    mutate {
      add_tag => [ "slow_request" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "microservices-logs-%{+YYYY.MM.dd}"
  }
}
```

## Metrics Collection

### Prometheus Integration

```python
# metrics_config.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.exposition import MetricsHandler
from flask import Response
import time
import functools

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code', 'service']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'service'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections',
    ['service']
)

DATABASE_OPERATIONS = Counter(
    'database_operations_total',
    'Total database operations',
    ['operation', 'table', 'status', 'service']
)

EXTERNAL_API_CALLS = Counter(
    'external_api_calls_total',
    'External API calls',
    ['service_name', 'endpoint', 'status_code', 'service']
)

QUEUE_SIZE = Gauge(
    'queue_size',
    'Current queue size',
    ['queue_name', 'service']
)

def track_metrics(service_name: str):
    """Decorator to track HTTP request metrics"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            method = request.method
            endpoint = request.endpoint or 'unknown'
            
            try:
                # Execute the request
                response = func(*args, **kwargs)
                
                # Determine status code
                if isinstance(response, tuple):
                    status_code = str(response[1])
                else:
                    status_code = '200'
                
                # Record metrics
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code,
                    service=service_name
                ).inc()
                
                REQUEST_DURATION.labels(
                    method=method,
                    endpoint=endpoint,
                    service=service_name
                ).observe(time.time() - start_time)
                
                return response
                
            except Exception as e:
                # Record error metrics
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code='500',
                    service=service_name
                ).inc()
                
                REQUEST_DURATION.labels(
                    method=method,
                    endpoint=endpoint,
                    service=service_name
                ).observe(time.time() - start_time)
                
                raise e
        
        return wrapper
    return decorator

# inventory_service.py
from flask import Flask, request, jsonify
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from metrics_config import *
import random
import time

app = Flask(__name__)
SERVICE_NAME = "inventory-service"

class DatabaseMetrics:
    """Context manager for database operation metrics"""
    
    def __init__(self, operation: str, table: str):
        self.operation = operation
        self.table = table
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else "success"
        DATABASE_OPERATIONS.labels(
            operation=self.operation,
            table=self.table,
            status=status,
            service=SERVICE_NAME
        ).inc()

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/inventory/<item_id>')
@track_metrics(SERVICE_NAME)
def get_inventory(item_id):
    """Get inventory information with metrics tracking"""
    
    # Track active connections
    ACTIVE_CONNECTIONS.labels(service=SERVICE_NAME).inc()
    
    try:
        # Simulate database query with metrics
        with DatabaseMetrics("SELECT", "inventory"):
            time.sleep(random.uniform(0.01, 0.1))  # Simulate DB latency
            
            inventory = {
                "item_id": item_id,
                "quantity": random.randint(0, 100),
                "location": "warehouse-1"
            }
        
        # Simulate external API call for pricing
        with REQUEST_DURATION.labels(
            method="GET",
            endpoint="pricing-api",
            service=SERVICE_NAME
        ).time():
            
            # Simulate external API call
            time.sleep(random.uniform(0.05, 0.2))
            
            if random.random() > 0.1:  # 90% success rate
                EXTERNAL_API_CALLS.labels(
                    service_name="pricing-service",
                    endpoint="/prices",
                    status_code="200",
                    service=SERVICE_NAME
                ).inc()
                inventory["price"] = random.uniform(10, 100)
            else:
                EXTERNAL_API_CALLS.labels(
                    service_name="pricing-service",
                    endpoint="/prices",
                    status_code="500",
                    service=SERVICE_NAME
                ).inc()
                inventory["price"] = None
        
        return jsonify(inventory)
        
    finally:
        ACTIVE_CONNECTIONS.labels(service=SERVICE_NAME).dec()

@app.route('/inventory/<item_id>/reserve', methods=['POST'])
@track_metrics(SERVICE_NAME)
def reserve_inventory(item_id):
    """Reserve inventory items"""
    
    data = request.get_json()
    quantity = data.get('quantity', 1)
    
    # Simulate queue processing
    current_queue_size = random.randint(0, 50)
    QUEUE_SIZE.labels(
        queue_name="reservation_queue",
        service=SERVICE_NAME
    ).set(current_queue_size)
    
    try:
        # Simulate reservation process
        with DatabaseMetrics("UPDATE", "inventory"):
            time.sleep(random.uniform(0.02, 0.08))
            
            # Simulate occasional failures
            if random.random() > 0.05:  # 95% success rate
                return jsonify({
                    "reservation_id": f"res_{item_id}_{int(time.time())}",
                    "status": "reserved",
                    "quantity": quantity
                })
            else:
                raise Exception("Insufficient inventory")
                
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": SERVICE_NAME,
        "timestamp": time.time()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
```

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'microservices'
    static_configs:
      - targets: 
          - 'user-service:5000'
          - 'order-service:5001'
          - 'payment-service:5002'
          - 'inventory-service:5003'
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

## Health Checking

### Comprehensive Health Checks

```python
# health_check.py
from flask import Flask, jsonify
import psutil
import time
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    details: Optional[Dict] = None
    duration_ms: Optional[float] = None

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.checks = {}
    
    def register_check(self, name: str, check_func, critical: bool = True):
        """Register a health check function"""
        self.checks[name] = {
            'func': check_func,
            'critical': critical
        }
    
    def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                details={"error": "Check not found"}
            )
        
        start_time = time.time()
        try:
            check_func = self.checks[name]['func']
            result = check_func()
            duration = (time.time() - start_time) * 1000
            
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                details = None
            elif isinstance(result, dict):
                status = HealthStatus(result.get('status', 'healthy'))
                details = result.get('details')
            else:
                status = HealthStatus.HEALTHY
                details = {"result": str(result)}
            
            return HealthCheck(
                name=name,
                status=status,
                details=details,
                duration_ms=round(duration, 2)
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e)},
                duration_ms=round(duration, 2)
            )
    
    def run_all_checks(self) -> Dict:
        """Run all registered health checks"""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, config in self.checks.items():
            check_result = self.run_check(name)
            results[name] = {
                "status": check_result.status.value,
                "duration_ms": check_result.duration_ms,
                "details": check_result.details
            }
            
            # Determine overall status
            if config['critical'] and check_result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif check_result.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "service": self.service_name,
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": results
        }

# Enhanced service with health checks
from sqlalchemy import create_engine, text
import redis

app = Flask(__name__)
health_checker = HealthChecker("order-service")

# Database connection for health checks
db_engine = create_engine("postgresql://user:pass@db:5432/orders")
redis_client = redis.Redis(host='redis', port=6379, db=0)

def check_database():
    """Check database connectivity and performance"""
    try:
        start_time = time.time()
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            query_time = (time.time() - start_time) * 1000
            
        if query_time > 1000:  # Slow query threshold
            return {
                "status": "degraded",
                "details": {
                    "query_time_ms": round(query_time, 2),
                    "message": "Database responding slowly"
                }
            }
        
        return {
            "status": "healthy",
            "details": {"query_time_ms": round(query_time, 2)}
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "details": {"error": str(e)}
        }

def check_redis():
    """Check Redis connectivity"""
    try:
        start_time = time.time()
        redis_client.ping()
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "details": {"response_time_ms": round(response_time, 2)}
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "details": {"error": str(e)}
        }

def check_memory():
    """Check memory usage"""
    memory = psutil.virtual_memory()
    usage_percent = memory.percent
    
    if usage_percent > 90:
        status = "unhealthy"
    elif usage_percent > 80:
        status = "degraded"
    else:
        status = "healthy"
    
    return {
        "status": status,
        "details": {
            "usage_percent": usage_percent,
            "available_mb": round(memory.available / 1024 / 1024, 2),
            "total_mb": round(memory.total / 1024 / 1024, 2)
        }
    }

def check_disk_space():
    """Check disk space"""
    disk = psutil.disk_usage('/')
    usage_percent = (disk.used / disk.total) * 100
    
    if usage_percent > 95:
        status = "unhealthy"
    elif usage_percent > 85:
        status = "degraded"
    else:
        status = "healthy"
    
    return {
        "status": status,
        "details": {
            "usage_percent": round(usage_percent, 2),
            "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
            "total_gb": round(disk.total / 1024 / 1024 / 1024, 2)
        }
    }

def check_external_services():
    """Check external service dependencies"""
    services = {
        "user-service": "http://user-service:5000/health",
        "payment-service": "http://payment-service:5002/health"
    }
    
    results = {}
    overall_status = "healthy"
    
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                results[service_name] = "healthy"
            else:
                results[service_name] = "unhealthy"
                overall_status = "degraded"
        except Exception as e:
            results[service_name] = "unhealthy"
            overall_status = "degraded"
    
    return {
        "status": overall_status,
        "details": results
    }

# Register health checks
health_checker.register_check("database", check_database, critical=True)
health_checker.register_check("redis", check_redis, critical=True)
health_checker.register_check("memory", check_memory, critical=False)
health_checker.register_check("disk", check_disk_space, critical=False)
health_checker.register_check("external_services", check_external_services, critical=False)

@app.route('/health')
def health():
    """Basic health check endpoint"""
    return jsonify({"status": "healthy", "service": "order-service"})

@app.route('/health/detailed')
def detailed_health():
    """Detailed health check with all diagnostics"""
    return jsonify(health_checker.run_all_checks())

@app.route('/health/live')
def liveness():
    """Kubernetes liveness probe - basic service availability"""
    # Only check critical components for liveness
    critical_checks = ["database", "redis"]
    
    for check_name in critical_checks:
        result = health_checker.run_check(check_name)
        if result.status == HealthStatus.UNHEALTHY:
            return jsonify({
                "status": "unhealthy",
                "failed_check": check_name
            }), 503
    
    return jsonify({"status": "healthy"})

@app.route('/health/ready')
def readiness():
    """Kubernetes readiness probe - ready to receive traffic"""
    all_results = health_checker.run_all_checks()
    
    if all_results["status"] == "unhealthy":
        return jsonify(all_results), 503
    
    return jsonify(all_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

## Alerting Systems

### Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@company.com'

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://webhook-service:8080/alerts'

- name: 'critical-alerts'
  email_configs:
  - to: 'oncall@company.com'
    subject: '🚨 CRITICAL Alert: {{ .GroupLabels.alertname }}'
    body: |
      Alert: {{ .GroupLabels.alertname }}
      Service: {{ .GroupLabels.service }}
      
      {{ range .Alerts }}
      Description: {{ .Annotations.description }}
      Summary: {{ .Annotations.summary }}
      {{ end }}
  
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts-critical'
    title: 'Critical Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'warning-alerts'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts-warning'
    title: 'Warning Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  equal: ['alertname', 'service']
```

### Alert Rules

```yaml
# alert_rules.yml
groups:
- name: microservice_alerts
  rules:
  
  # High Error Rate Alert
  - alert: HighErrorRate
    expr: |
      (
        sum(rate(http_requests_total{status_code=~"5.."}[5m])) by (service)
        /
        sum(rate(http_requests_total[5m])) by (service)
      ) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected for {{ $labels.service }}"
      description: "Error rate is {{ $value | humanizePercentage }} for service {{ $labels.service }}"
  
  # High Response Time Alert
  - alert: HighResponseTime
    expr: |
      histogram_quantile(0.95, 
        sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
      ) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time for {{ $labels.service }}"
      description: "95th percentile response time is {{ $value }}s for {{ $labels.service }}"
  
  # Service Down Alert
  - alert: ServiceDown
    expr: up{job="microservices"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.instance }} is down"
      description: "Service {{ $labels.instance }} has been down for more than 1 minute"
  
  # High Memory Usage Alert
  - alert: HighMemoryUsage
    expr: |
      (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage on {{ $labels.instance }}"
      description: "Memory usage is above 90% on {{ $labels.instance }}"
  
  # Database Connection Pool Alert
  - alert: DatabaseConnectionPoolExhausted
    expr: |
      database_connection_pool_active / database_connection_pool_max > 0.9
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Database connection pool nearly exhausted"
      description: "Connection pool usage is {{ $value | humanizePercentage }}"
  
  # External API High Latency
  - alert: ExternalAPIHighLatency
    expr: |
      histogram_quantile(0.95,
        sum(rate(external_api_request_duration_seconds_bucket[5m])) by (le, api_name)
      ) > 5.0
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "High latency for external API {{ $labels.api_name }}"
      description: "95th percentile latency is {{ $value }}s for {{ $labels.api_name }}"

- name: business_metrics_alerts
  rules:
  
  # Failed Payment Processing
  - alert: HighPaymentFailureRate
    expr: |
      (
        sum(rate(payment_transactions_total{status="failed"}[10m]))
        /
        sum(rate(payment_transactions_total[10m]))
      ) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High payment failure rate"
      description: "Payment failure rate is {{ $value | humanizePercentage }}"
  
  # Low Order Processing Rate
  - alert: LowOrderProcessingRate
    expr: |
      sum(rate(orders_processed_total[5m])) < 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low order processing rate"
      description: "Order processing rate is {{ $value }} orders/second"
```

### Custom Alert Webhook Service

```python
# alert_webhook.py
from flask import Flask, request, jsonify
import json
import requests
from datetime import datetime
from typing import Dict, List

app = Flask(__name__)

class AlertProcessor:
    """Process and route alerts to appropriate channels"""
    
    def __init__(self):
        self.integrations = {
            'slack': self.send_slack_alert,
            'email': self.send_email_alert,
            'pagerduty': self.send_pagerduty_alert,
            'teams': self.send_teams_alert
        }
    
    def process_alert(self, alert_data: Dict) -> Dict:
        """Process incoming alert and route to appropriate channels"""
        
        alerts = alert_data.get('alerts', [])
        processed_alerts = []
        
        for alert in alerts:
            # Extract alert information
            alert_info = {
                'name': alert.get('labels', {}).get('alertname', 'Unknown'),
                'service': alert.get('labels', {}).get('service', 'Unknown'),
                'severity': alert.get('labels', {}).get('severity', 'unknown'),
                'status': alert.get('status', 'unknown'),
                'description': alert.get('annotations', {}).get('description', ''),
                'summary': alert.get('annotations', {}).get('summary', ''),
                'timestamp': alert.get('startsAt', datetime.now().isoformat()),
                'labels': alert.get('labels', {}),
                'annotations': alert.get('annotations', {})
            }
            
            # Route alert based on severity and service
            channels = self.determine_channels(alert_info)
            
            # Send to each channel
            for channel in channels:
                try:
                    if channel in self.integrations:
                        self.integrations[channel](alert_info)
                except Exception as e:
                    print(f"Failed to send alert to {channel}: {e}")
            
            processed_alerts.append({
                'alert': alert_info,
                'channels': channels,
                'processed_at': datetime.now().isoformat()
            })
        
        return {
            'status': 'processed',
            'alerts_count': len(processed_alerts),
            'alerts': processed_alerts
        }
    
    def determine_channels(self, alert_info: Dict) -> List[str]:
        """Determine which channels to send alert to based on rules"""
        channels = []
        
        severity = alert_info['severity']
        service = alert_info['service']
        
        # Critical alerts go to multiple channels
        if severity == 'critical':
            channels.extend(['slack', 'email', 'pagerduty'])
        elif severity == 'warning':
            channels.append('slack')
        
        # Service-specific routing
        if service in ['payment-service', 'order-service']:
            channels.append('teams')  # Business critical services
        
        return list(set(channels))  # Remove duplicates
    
    def send_slack_alert(self, alert_info: Dict):
        """Send alert to Slack"""
        webhook_url = "YOUR_SLACK_WEBHOOK_URL"
        
        color = {
            'critical': 'danger',
            'warning': 'warning',
            'info': 'good'
        }.get(alert_info['severity'], 'good')
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"🚨 {alert_info['name']}",
                'fields': [
                    {
                        'title': 'Service',
                        'value': alert_info['service'],
                        'short': True
                    },
                    {
                        'title': 'Severity',
                        'value': alert_info['severity'].upper(),
                        'short': True
                    },
                    {
                        'title': 'Description',
                        'value': alert_info['description'],
                        'short': False
                    }
                ],
                'ts': int(datetime.now().timestamp())
            }]
        }
        
        response = requests.post(webhook_url, json=payload)
        return response.status_code == 200
    
    def send_email_alert(self, alert_info: Dict):
        """Send alert via email"""
        # Implementation would use SMTP or email service API
        print(f"EMAIL ALERT: {alert_info['summary']}")
        return True
    
    def send_pagerduty_alert(self, alert_info: Dict):
        """Send alert to PagerDuty"""
        # Implementation would use PagerDuty Events API v2
        print(f"PAGERDUTY ALERT: {alert_info['summary']}")
        return True
    
    def send_teams_alert(self, alert_info: Dict):
        """Send alert to Microsoft Teams"""
        # Implementation would use Teams webhook
        print(f"TEAMS ALERT: {alert_info['summary']}")
        return True

alert_processor = AlertProcessor()

@app.route('/alerts', methods=['POST'])
def handle_alerts():
    """Handle incoming alerts from Alertmanager"""
    try:
        alert_data = request.get_json()
        
        if not alert_data:
            return jsonify({'error': 'No alert data provided'}), 400
        
        # Process alerts
        result = alert_processor.process_alert(alert_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Observability Dashboard

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "Microservices Observability Dashboard",
    "tags": ["microservices", "observability"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Service Health Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"microservices\"}",
            "legendFormat": "{{ instance }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{ service }}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status_code=~\"5..\"}[5m])) by (service) / sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{ service }}"
          }
        ],
        "yAxes": [
          {
            "label": "Error Rate",
            "min": 0,
            "max": 1
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Response Time (95th percentile)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "{{ service }}"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### Custom Observability Service

```python
# observability_service.py
from flask import Flask, jsonify, request
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

app = Flask(__name__)

class ObservabilityAggregator:
    """Aggregate observability data from multiple sources"""
    
    def __init__(self):
        self.prometheus_url = "http://prometheus:9090"
        self.jaeger_url = "http://jaeger:16686"
        self.elasticsearch_url = "http://elasticsearch:9200"
    
    def get_service_overview(self) -> Dict[str, Any]:
        """Get comprehensive service overview"""
        overview = {
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'system_health': {},
            'alerts': []
        }
        
        # Get service metrics
        services = self.get_service_list()
        
        for service in services:
            overview['services'][service] = {
                'health': self.get_service_health(service),
                'metrics': self.get_service_metrics(service),
                'traces': self.get_service_traces(service),
                'logs': self.get_service_logs(service)
            }
        
        # Get system-wide metrics
        overview['system_health'] = self.get_system_metrics()
        
        # Get active alerts
        overview['alerts'] = self.get_active_alerts()
        
        return overview
    
    def get_service_list(self) -> List[str]:
        """Get list of monitored services"""
        query = "up{job='microservices'}"
        response = self.query_prometheus(query)
        
        services = set()
        if response and 'data' in response and 'result' in response['data']:
            for result in response['data']['result']:
                service = result['metric'].get('service')
                if service:
                    services.add(service)
        
        return list(services)
    
    def get_service_health(self, service: str) -> Dict[str, Any]:
        """Get health status for a specific service"""
        queries = {
            'up': f'up{{service="{service}"}}',
            'error_rate': f'sum(rate(http_requests_total{{service="{service}",status_code=~"5.."}}[5m])) / sum(rate(http_requests_total{{service="{service}"}}[5m]))',
            'response_time_p95': f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))',
            'cpu_usage': f'rate(process_cpu_seconds_total{{service="{service}"}}[5m])',
            'memory_usage': f'process_resident_memory_bytes{{service="{service}"}}'
        }
        
        health = {}
        for metric, query in queries.items():
            result = self.query_prometheus(query)
            if result and 'data' in result and 'result' in result['data']:
                if result['data']['result']:
                    health[metric] = float(result['data']['result'][0]['value'][1])
                else:
                    health[metric] = None
            else:
                health[metric] = None
        
        # Determine overall health status
        if health['up'] == 1:
            if health['error_rate'] and health['error_rate'] > 0.05:
                health['status'] = 'degraded'
            elif health['response_time_p95'] and health['response_time_p95'] > 1.0:
                health['status'] = 'degraded'
            else:
                health['status'] = 'healthy'
        else:
            health['status'] = 'unhealthy'
        
        return health
    
    def get_service_metrics(self, service: str) -> Dict[str, Any]:
        """Get detailed metrics for a service"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        metrics = {
            'request_rate': self.get_time_series(
                f'sum(rate(http_requests_total{{service="{service}"}}[5m]))',
                start_time, end_time
            ),
            'error_rate': self.get_time_series(
                f'sum(rate(http_requests_total{{service="{service}",status_code=~"5.."}}[5m]))',
                start_time, end_time
            ),
            'response_time': self.get_time_series(
                f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))',
                start_time, end_time
            )
        }
        
        return metrics
    
    def get_service_traces(self, service: str) -> Dict[str, Any]:
        """Get recent traces for a service"""
        # This would query Jaeger API
        # For demo, return mock data
        return {
            'recent_traces': [
                {
                    'trace_id': 'abc123',
                    'operation': 'GET /orders',
                    'duration_ms': 234,
                    'spans': 8,
                    'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat()
                }
            ],
            'slow_traces': [
                {
                    'trace_id': 'def456',
                    'operation': 'POST /payments',
                    'duration_ms': 1567,
                    'spans': 12,
                    'timestamp': (datetime.now() - timedelta(minutes=10)).isoformat()
                }
            ]
        }
    
    def get_service_logs(self, service: str) -> Dict[str, Any]:
        """Get recent logs for a service"""
        # This would query Elasticsearch
        # For demo, return mock data
        return {
            'error_logs': [
                {
                    'timestamp': (datetime.now() - timedelta(minutes=2)).isoformat(),
                    'level': 'ERROR',
                    'message': 'Database connection failed',
                    'trace_id': 'abc123'
                }
            ],
            'warning_logs': [
                {
                    'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                    'level': 'WARNING',
                    'message': 'High response time detected',
                    'trace_id': 'def456'
                }
            ]
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        queries = {
            'total_requests': 'sum(rate(http_requests_total[5m]))',
            'total_errors': 'sum(rate(http_requests_total{status_code=~"5.."}[5m]))',
            'avg_response_time': 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
            'active_services': 'count(up{job="microservices"} == 1)'
        }
        
        system_metrics = {}
        for metric, query in queries.items():
            result = self.query_prometheus(query)
            if result and 'data' in result and 'result' in result['data']:
                if result['data']['result']:
                    system_metrics[metric] = float(result['data']['result'][0]['value'][1])
                else:
                    system_metrics[metric] = 0
            else:
                system_metrics[metric] = 0
        
        return system_metrics
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/alerts")
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {}).get('alerts', [])
        except Exception as e:
            print(f"Error fetching alerts: {e}")
        
        return []
    
    def query_prometheus(self, query: str) -> Dict:
        """Execute Prometheus query"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query}
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
        
        return {}
    
    def get_time_series(self, query: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get time series data from Prometheus"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    'query': query,
                    'start': start_time.timestamp(),
                    'end': end_time.timestamp(),
                    'step': '60s'
                }
            )
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'result' in data['data']:
                    return data['data']['result']
        except Exception as e:
            print(f"Error querying time series: {e}")
        
        return []

observability = ObservabilityAggregator()

@app.route('/api/overview')
def get_overview():
    """Get comprehensive observability overview"""
    return jsonify(observability.get_service_overview())

@app.route('/api/services/<service_name>')
def get_service_details(service_name):
    """Get detailed information for a specific service"""
    details = {
        'service': service_name,
        'health': observability.get_service_health(service_name),
        'metrics': observability.get_service_metrics(service_name),
        'traces': observability.get_service_traces(service_name),
        'logs': observability.get_service_logs(service_name)
    }
    return jsonify(details)

@app.route('/api/alerts')
def get_alerts():
    """Get active alerts"""
    return jsonify(observability.get_active_alerts())

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)
```

## Complete Docker Compose Setup

```yaml
# docker-compose.observability.yml
version: '3.8'

services:
  # Application Services
  user-service:
    build: ./user-service
    ports:
      - "5000:5000"
    environment:
      - JAEGER_AGENT_HOST=jaeger
      - DATABASE_URL=postgresql://user:pass@postgres:5432/users
    depends_on:
      - postgres
      - jaeger
    networks:
      - microservices

  order-service:
    build: ./order-service
    ports:
      - "5001:5001"
    environment:
      - JAEGER_AGENT_HOST=jaeger
      - DATABASE_URL=postgresql://user:pass@postgres:5432/orders
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
      - jaeger
    networks:
      - microservices

  payment-service:
    build: ./payment-service
    ports:
      - "5002:5002"
    environment:
      - JAEGER_AGENT_HOST=jaeger
    depends_on:
      - jaeger
    networks:
      - microservices

  inventory-service:
    build: ./inventory-service
    ports:
      - "5003:5003"
    environment:
      - JAEGER_AGENT_HOST=jaeger
    depends_on:
      - jaeger
    networks:
      - microservices

  # Observability Stack
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alert_rules.yml:/etc/prometheus/alert_rules.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - microservices

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    networks:
      - microservices

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "6831:6831/udp"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - microservices

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    networks:
      - microservices

  # ELK Stack
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.10.2
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - microservices

  logstash:
    image: docker.elastic.co/logstash/logstash:8.10.2
    volumes:
      - ./logstash/config:/usr/share/logstash/pipeline
    ports:
      - "5044:5044"
    environment:
      - "LS_JAVA_OPTS=-Xmx256m -Xms256m"
    depends_on:
      - elasticsearch
    networks:
      - microservices

  kibana:
    image: docker.elastic.co/kibana/kibana:8.10.2
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - microservices

  # Supporting Services
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=microservices
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - microservices

  redis:
    image: redis:alpine
    networks:
      - microservices

  # Custom Services
  alert-webhook:
    build: ./alert-webhook
    ports:
      - "8080:8080"
    networks:
      - microservices

  observability-service:
    build: ./observability-service
    ports:
      - "8090:8090"
    environment:
      - PROMETHEUS_URL=http://prometheus:9090
      - JAEGER_URL=http://jaeger:16686
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    depends_on:
      - prometheus
      - jaeger
      - elasticsearch
    networks:
      - microservices

volumes:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
  postgres_data:

networks:
  microservices:
    driver: bridge
```

## Learning Objectives

By the end of this section, you should be able to:

### Core Understanding
- **Distinguish between monitoring and observability** and explain when to use each approach
- **Implement the three pillars** of observability (metrics, logs, traces) in microservices
- **Design comprehensive health checks** that provide meaningful insights into service status
- **Set up distributed tracing** to track requests across multiple services
- **Configure structured logging** for better searchability and analysis

### Technical Implementation
- **Deploy and configure Prometheus** for metrics collection and alerting
- **Implement OpenTelemetry instrumentation** for distributed tracing
- **Set up ELK stack** for centralized log aggregation and analysis
- **Create effective alerting rules** with appropriate thresholds and escalation
- **Build observability dashboards** using Grafana for data visualization

### Advanced Skills
- **Design correlation strategies** between metrics, logs, and traces
- **Implement custom metrics** for business-specific KPIs
- **Create intelligent alerting** that reduces noise and focuses on actionable issues
- **Optimize observability costs** while maintaining comprehensive coverage
- **Troubleshoot distributed systems** using observability data

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ **Set up basic monitoring** for a microservice with health checks  
□ **Implement distributed tracing** across multiple services  
□ **Create structured logs** with proper correlation IDs  
□ **Configure Prometheus metrics** with custom business metrics  
□ **Set up alerting rules** with appropriate thresholds  
□ **Build Grafana dashboards** that provide actionable insights  
□ **Correlate data** across metrics, logs, and traces for debugging  
□ **Implement SLIs/SLOs** (Service Level Indicators/Objectives)  
□ **Debug performance issues** using observability tools  
□ **Set up log aggregation** with ELK stack  

### Practical Exercises

**Exercise 1: Basic Service Monitoring**
```python
# TODO: Implement comprehensive monitoring for this e-commerce service
from flask import Flask

app = Flask(__name__)

@app.route('/products/<product_id>')
def get_product(product_id):
    # Add metrics collection
    # Add structured logging
    # Add distributed tracing
    # Add health checks
    pass

@app.route('/orders', methods=['POST'])
def create_order():
    # Implement monitoring for this critical business operation
    pass
```

**Exercise 2: Alert Rule Design**
```yaml
# TODO: Create alert rules for these scenarios:
# 1. High error rate (>5% for 2 minutes)
# 2. Slow response time (95th percentile >1s for 5 minutes)
# 3. Low order processing rate (<10 orders/minute for 5 minutes)
# 4. Database connection pool exhaustion (>90% usage)
# 5. External API failures (>10% failure rate)
```

**Exercise 3: Distributed Tracing Implementation**
```python
# TODO: Implement end-to-end tracing for this user registration flow:
# 1. User Service receives registration request
# 2. Validates user data
# 3. Calls Payment Service to set up billing
# 4. Calls Notification Service to send welcome email
# 5. Updates User Database
# 6. Returns success response

# Ensure each step is properly traced with:
# - Span creation and proper parent-child relationships
# - Error handling and span status
# - Custom attributes for debugging
# - Correlation with logs
```

## Study Materials

### Essential Reading
- **Primary:** "Observability Engineering" by Charity Majors, Liz Fong-Jones, George Miranda
- **Technical:** "Site Reliability Engineering" by Google - Chapters on Monitoring
- **Reference:** [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- **Best Practices:** [Prometheus Best Practices](https://prometheus.io/docs/practices/)

### Video Resources
- "Observability vs Monitoring" - Honeycomb.io talks
- "Distributed Tracing in Practice" - CNCF webinars
- "Building Effective Dashboards" - Grafana tutorials
- "Alerting Best Practices" - Prometheus community talks

### Hands-on Labs

**Lab 1: Three Pillars Implementation**
- Set up metrics collection with Prometheus
- Implement structured logging with ELK
- Add distributed tracing with Jaeger
- Create correlation between all three

**Lab 2: Production-Ready Monitoring**
- Design SLIs/SLOs for an e-commerce system
- Implement multi-tier alerting strategy
- Create runbooks for alert response
- Set up on-call rotation system

**Lab 3: Performance Investigation**
- Use observability tools to debug slow requests
- Identify bottlenecks using distributed traces
- Correlate metrics with deployment events
- Create capacity planning dashboards

### Recommended Tools and Technologies

**Metrics Collection:**
- Prometheus + Grafana (open source)
- DataDog, New Relic (commercial)
- CloudWatch, Azure Monitor (cloud native)

**Logging:**
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Fluentd + ElasticSearch
- Splunk (enterprise)
- Cloud logging services

**Distributed Tracing:**
- OpenTelemetry + Jaeger
- Zipkin
- AWS X-Ray, Google Cloud Trace

**All-in-One Solutions:**
- Honeycomb
- Datadog APM
- New Relic One
- Dynatrace

### Development Environment Setup

**Prerequisites:**
```bash
# Install Docker and Docker Compose
docker --version
docker-compose --version

# Install Python dependencies
pip install -r requirements.txt

# Install monitoring tools locally (optional)
# Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz

# Grafana
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
sudo apt-get update
sudo apt-get install grafana
```

**Quick Start Commands:**
```bash
# Start complete observability stack
docker-compose -f docker-compose.observability.yml up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f user-service

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686
# Kibana: http://localhost:5601

# Generate test traffic
curl http://localhost:5000/users/123
curl -X POST http://localhost:5001/orders -H "Content-Type: application/json" -d '{"user_id": "123", "items": ["item1"]}'

# Stop all services
docker-compose -f docker-compose.observability.yml down
```

### Key Performance Indicators (KPIs)

**Service Level Indicators (SLIs):**
- **Availability:** Percentage of successful requests (target: >99.9%)
- **Latency:** 95th percentile response time (target: <500ms)
- **Throughput:** Requests per second the service can handle
- **Error Rate:** Percentage of failed requests (target: <0.1%)

**Business Metrics:**
- **Order Processing Rate:** Orders processed per minute
- **Payment Success Rate:** Successful payments vs. total attempts
- **User Registration Rate:** New user registrations per hour
- **Cart Abandonment Rate:** Percentage of abandoned shopping carts

**Infrastructure Metrics:**
- **CPU Utilization:** Average CPU usage (target: <70%)
- **Memory Usage:** Memory consumption percentage
- **Disk I/O:** Read/write operations per second
- **Network Latency:** Inter-service communication delays

### Troubleshooting Guide

**Common Issues and Solutions:**

1. **High Memory Usage in Prometheus**
   ```yaml
   # Solution: Adjust retention and scrape intervals
   global:
     scrape_interval: 30s  # Increase from 15s
   
   storage:
     tsdb:
       retention.time: 15d  # Reduce from 30d
   ```

2. **Missing Traces in Jaeger**
   ```python
   # Check sampling configuration
   from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
   
   # Increase sampling rate for development
   sampler = TraceIdRatioBased(1.0)  # Sample 100% of traces
   ```

3. **Log Parsing Errors in Logstash**
   ```ruby
   # Add error handling in Logstash config
   filter {
     if "_grokparsefailure" in [tags] {
       mutate {
         add_field => { "logstash_parsing_error" => true }
       }
     }
   }
   ```

4. **Alertmanager Not Sending Notifications**
   ```yaml
   # Check webhook configuration
   receivers:
   - name: 'webhook'
     webhook_configs:
     - url: 'http://alert-webhook:8080/alerts'
       send_resolved: true
   ```

### Next Steps

After mastering monitoring and observability:

1. **Advanced Topics:**
   - Chaos engineering and observability
   - Machine learning for anomaly detection
   - Cost optimization for observability
   - Security monitoring and compliance

2. **Production Considerations:**
   - Data retention policies
   - GDPR compliance for logs
   - Multi-region monitoring setup
   - Disaster recovery for monitoring infrastructure
