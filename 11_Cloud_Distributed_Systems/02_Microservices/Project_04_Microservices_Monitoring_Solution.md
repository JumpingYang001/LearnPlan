# Project 4: Microservices Monitoring Solution

*Estimated Duration: 3-4 weeks*  
*Difficulty Level: Advanced*  
*Prerequisites: Docker, Kubernetes basics, Python/Node.js, distributed systems concepts*

## Project Overview

Build a comprehensive monitoring solution for microservices architecture that provides complete observability through the three pillars of observability: **Metrics**, **Logs**, and **Traces**. This project will implement distributed tracing, centralized log aggregation, real-time dashboards, and intelligent alerting mechanisms.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                             │
├─────────────────────────────────────────────────────────────────┤
│  Frontend: Grafana Dashboards + Custom Alert Manager UI        │
├─────────────────────────────────────────────────────────────────┤
│  Processing: Prometheus + Jaeger + ELK Stack + AlertManager     │
├─────────────────────────────────────────────────────────────────┤
│  Collection: Agents (Prometheus exporters, OpenTelemetry,      │
│              Fluent Bit, Custom collectors)                     │
├─────────────────────────────────────────────────────────────────┤
│  Services: User Service + Product Service + Order Service +     │
│           Payment Service + Notification Service                │
└─────────────────────────────────────────────────────────────────┘
```

## Learning Objectives

By completing this project, you will:
- ✅ **Implement distributed tracing** using OpenTelemetry and Jaeger
- ✅ **Build centralized logging** with ELK Stack (Elasticsearch, Logstash, Kibana)
- ✅ **Create metrics collection** using Prometheus and custom exporters
- ✅ **Design monitoring dashboards** with Grafana
- ✅ **Implement alerting systems** with AlertManager and custom notifications
- ✅ **Handle monitoring data at scale** with proper storage and retention policies
- ✅ **Debug distributed systems** using observability data
- ✅ **Apply SRE principles** for monitoring and reliability

## Project Structure

```
microservices-monitoring/
├── docker-compose.yml                 # Complete monitoring stack
├── kubernetes/                        # K8s deployment manifests
│   ├── monitoring-namespace.yaml
│   ├── prometheus/
│   ├── grafana/
│   ├── jaeger/
│   └── elk-stack/
├── services/                          # Microservices with monitoring
│   ├── user-service/
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   ├── monitoring.py
│   │   └── requirements.txt
│   ├── product-service/
│   ├── order-service/
│   ├── payment-service/
│   └── notification-service/
├── monitoring/                        # Monitoring configuration
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules/
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── provisioning/
│   ├── jaeger/
│   │   └── jaeger-config.yaml
│   └── elk/
│       ├── elasticsearch.yml
│       ├── logstash.conf
│       └── kibana.yml
├── scripts/                           # Automation scripts
│   ├── setup-monitoring.sh
│   ├── load-test.py
│   └── chaos-engineering.py
└── docs/                             # Documentation
    ├── setup-guide.md
    ├── dashboard-guide.md
    └── troubleshooting.md
```

## Phase 1: Foundation Setup

### 1.1 Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Monitoring Infrastructure
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # HTTP collector
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    container_name: logstash
    ports:
      - "5044:5044"
      - "5000:5000/tcp"
      - "5000:5000/udp"
      - "9600:9600"
    volumes:
      - ./monitoring/elk/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager:/etc/alertmanager

  # Sample Microservices
  user-service:
    build: ./services/user-service
    container_name: user-service
    ports:
      - "8001:8000"
    environment:
      - JAEGER_ENDPOINT=http://jaeger:14268/api/traces
      - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir

  product-service:
    build: ./services/product-service
    container_name: product-service
    ports:
      - "8002:8000"
    environment:
      - JAEGER_ENDPOINT=http://jaeger:14268/api/traces

  order-service:
    build: ./services/order-service
    container_name: order-service
    ports:
      - "8003:8000"
    environment:
      - JAEGER_ENDPOINT=http://jaeger:14268/api/traces

volumes:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
```

### 1.2 Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'user-service'
    static_configs:
      - targets: ['user-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'product-service'
    static_configs:
      - targets: ['product-service:8000']
    metrics_path: '/metrics'

  - job_name: 'order-service'
    static_configs:
      - targets: ['order-service:8000']
    metrics_path: '/metrics'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

## Phase 2: Instrumented Microservices

### 2.1 User Service with Complete Monitoring

```python
# services/user-service/app.py
from flask import Flask, request, jsonify
import logging
import time
import random
from monitoring import setup_monitoring, trace_request, log_structured, metrics

app = Flask(__name__)

# Setup monitoring components
tracer, logger, prom_metrics = setup_monitoring("user-service")

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "user-service"})

@app.route('/users', methods=['GET'])
@trace_request(tracer)
def get_users():
    with tracer.start_as_current_span("get_users") as span:
        start_time = time.time()
        
        try:
            # Simulate database query
            query_time = random.uniform(0.01, 0.1)
            time.sleep(query_time)
            
            users = [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
            ]
            
            # Add span attributes
            span.set_attribute("user.count", len(users))
            span.set_attribute("db.query_time", query_time)
            
            # Structured logging
            log_structured(logger, "info", "users_retrieved", {
                "user_count": len(users),
                "query_time_ms": query_time * 1000,
                "endpoint": "/users"
            })
            
            # Metrics
            prom_metrics['request_duration'].labels(
                method='GET', 
                endpoint='/users', 
                status='200'
            ).observe(time.time() - start_time)
            
            prom_metrics['request_count'].labels(
                method='GET', 
                endpoint='/users', 
                status='200'
            ).inc()
            
            return jsonify(users)
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            
            log_structured(logger, "error", "users_retrieval_failed", {
                "error": str(e),
                "endpoint": "/users"
            })
            
            prom_metrics['request_count'].labels(
                method='GET', 
                endpoint='/users', 
                status='500'
            ).inc()
            
            return jsonify({"error": "Internal server error"}), 500

@app.route('/users/<int:user_id>', methods=['GET'])
@trace_request(tracer)
def get_user(user_id):
    with tracer.start_as_current_span("get_user") as span:
        span.set_attribute("user.id", user_id)
        
        # Simulate user lookup
        if user_id <= 0:
            log_structured(logger, "warning", "invalid_user_id", {
                "user_id": user_id,
                "endpoint": f"/users/{user_id}"
            })
            return jsonify({"error": "Invalid user ID"}), 400
            
        if user_id > 100:
            log_structured(logger, "info", "user_not_found", {
                "user_id": user_id,
                "endpoint": f"/users/{user_id}"
            })
            return jsonify({"error": "User not found"}), 404
            
        user = {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}
        
        log_structured(logger, "info", "user_retrieved", {
            "user_id": user_id,
            "endpoint": f"/users/{user_id}"
        })
        
        return jsonify(user)

@app.route('/metrics')
def metrics_endpoint():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
```

### 2.2 Comprehensive Monitoring Module

```python
# services/user-service/monitoring.py
import logging
import json
import time
import functools
from typing import Dict, Any

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import os

class StructuredLogger:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter for structured logging
        handler = logging.StreamHandler()
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": time.time(),
                "level": record.levelname,
                "message": record.getMessage(),
                "service": getattr(record, 'service', 'unknown'),
                "trace_id": getattr(record, 'trace_id', None),
                "span_id": getattr(record, 'span_id', None)
            }
            
            # Add extra fields if present
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)
                
            return json.dumps(log_entry)

def setup_monitoring(service_name: str):
    """Setup all monitoring components for a service"""
    
    # 1. Distributed Tracing Setup
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=14268,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument Flask and Requests
    FlaskInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    
    # 2. Structured Logging Setup
    logger = StructuredLogger(service_name)
    
    # 3. Prometheus Metrics Setup
    registry = CollectorRegistry()
    
    metrics = {
        'request_count': Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=registry
        ),
        'request_duration': Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint', 'status'],
            registry=registry
        ),
        'active_connections': Gauge(
            'active_connections',
            'Active connections',
            registry=registry
        ),
        'business_metric': Counter(
            'business_operations_total',
            'Business operations',
            ['operation_type', 'status'],
            registry=registry
        )
    }
    
    return tracer, logger, metrics

def trace_request(tracer):
    """Decorator to trace function calls"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(func.__name__) as span:
                try:
                    # Add common span attributes
                    span.set_attribute("service.name", "user-service")
                    span.set_attribute("service.version", "1.0.0")
                    
                    if hasattr(func, '__self__'):
                        span.set_attribute("function.class", func.__self__.__class__.__name__)
                    span.set_attribute("function.name", func.__name__)
                    
                    result = func(*args, **kwargs)
                    span.set_attribute("function.result", "success")
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
                    
        return wrapper
    return decorator

def log_structured(logger: StructuredLogger, level: str, event: str, extra_fields: Dict[str, Any]):
    """Helper function for structured logging"""
    
    # Get current trace context
    current_span = trace.get_current_span()
    trace_id = None
    span_id = None
    
    if current_span.is_recording():
        span_context = current_span.get_span_context()
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')
    
    log_record = logging.LogRecord(
        name=logger.service_name,
        level=getattr(logging, level.upper()),
        pathname="",
        lineno=0,
        msg=event,
        args=(),
        exc_info=None
    )
    
    log_record.service = logger.service_name
    log_record.trace_id = trace_id
    log_record.span_id = span_id
    log_record.extra_fields = extra_fields
    
    logger.logger.handle(log_record)

class MonitoringMiddleware:
    """Custom middleware for additional monitoring"""
    
    def __init__(self, app, metrics):
        self.app = app
        self.metrics = metrics
        
    def __call__(self, environ, start_response):
        start_time = time.time()
        
        def new_start_response(status, response_headers, exc_info=None):
            duration = time.time() - start_time
            
            # Extract metrics from environ
            method = environ.get('REQUEST_METHOD', 'GET')
            path = environ.get('PATH_INFO', '/')
            status_code = status.split()[0]
            
            # Record metrics
            self.metrics['request_duration'].labels(
                method=method,
                endpoint=path,
                status=status_code
            ).observe(duration)
            
            return start_response(status, response_headers, exc_info)
        
        return self.app(environ, new_start_response)
```

### 2.3 Service Dockerfile with Monitoring

```dockerfile
# services/user-service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for Prometheus multiprocess mode
RUN mkdir -p /tmp/prometheus_multiproc_dir

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

```txt
# services/user-service/requirements.txt
Flask==2.3.3
requests==2.31.0
prometheus-client==0.17.1
opentelemetry-api==1.20.0
opentelemetry-sdk==1.20.0
opentelemetry-exporter-jaeger-thrift==1.20.0
opentelemetry-instrumentation-flask==0.41b0
opentelemetry-instrumentation-requests==0.41b0
gunicorn==21.2.0
```

## Phase 3: Advanced Monitoring Features

### 3.1 Custom Metrics and Business KPIs

```python
# services/order-service/business_metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time
from datetime import datetime, timedelta

class BusinessMetrics:
    def __init__(self):
        # E-commerce specific metrics
        self.orders_created = Counter(
            'orders_created_total',
            'Total orders created',
            ['product_category', 'payment_method', 'customer_segment']
        )
        
        self.order_value = Histogram(
            'order_value_dollars',
            'Order value in dollars',
            ['product_category', 'customer_segment'],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, float('inf')]
        )
        
        self.checkout_funnel = Counter(
            'checkout_funnel_total',
            'Checkout funnel steps',
            ['step', 'status']  # step: cart, shipping, payment, confirmation
        )
        
        self.inventory_levels = Gauge(
            'inventory_level',
            'Current inventory level',
            ['product_id', 'warehouse']
        )
        
        self.processing_time = Histogram(
            'order_processing_duration_seconds',
            'Time to process order',
            ['complexity'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')]
        )

    def record_order_created(self, order_data):
        """Record business metrics when order is created"""
        self.orders_created.labels(
            product_category=order_data.get('category', 'unknown'),
            payment_method=order_data.get('payment_method', 'unknown'),
            customer_segment=self._get_customer_segment(order_data.get('customer_id'))
        ).inc()
        
        self.order_value.labels(
            product_category=order_data.get('category', 'unknown'),
            customer_segment=self._get_customer_segment(order_data.get('customer_id'))
        ).observe(order_data.get('total_amount', 0))

    def record_checkout_step(self, step, status='completed'):
        """Record checkout funnel progression"""
        self.checkout_funnel.labels(step=step, status=status).inc()

    def _get_customer_segment(self, customer_id):
        # Simplified customer segmentation logic
        if not customer_id:
            return 'anonymous'
        elif customer_id % 10 < 3:
            return 'premium'
        elif customer_id % 10 < 7:
            return 'regular'
        else:
            return 'new'
```

### 3.2 Advanced Alerting Rules

```yaml
# monitoring/prometheus/rules/service-alerts.yml
groups:
  - name: service-health
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"
          description: "{{ $labels.instance }} has been down for more than 30 seconds"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on {{ $labels.instance }}"
          description: "Error rate is {{ $value | humanizePercentage }} for 2 minutes"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High latency on {{ $labels.instance }}"
          description: "95th percentile latency is {{ $value }}s"

  - name: business-metrics
    rules:
      - alert: LowOrderVolume
        expr: rate(orders_created_total[1h]) < 10
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low order volume detected"
          description: "Order creation rate has been below 10/hour for 15 minutes"

      - alert: HighValueOrdersDropped
        expr: rate(checkout_funnel_total{step="payment", status="failed"}[5m]) > 2
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High value orders being dropped at payment"
          description: "Payment failures exceed 2 per 5 minutes"

      - alert: InventoryLow
        expr: inventory_level < 10
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Low inventory for product {{ $labels.product_id }}"
          description: "Inventory level is {{ $value }} units"
```

### 3.3 AlertManager Configuration

```yaml
# monitoring/alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@company.com'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 5s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      routes:
        - match:
            alertname: ServiceDown
          receiver: 'service-down-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'default'
    email_configs:
      - to: 'team@company.com'
        subject: 'Alert: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Labels: {{ range .Labels.SortedPairs }}{{ .Name }}={{ .Value }} {{ end }}
          {{ end }}

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@company.com'
        subject: 'CRITICAL: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK'
        channel: '#alerts'
        color: 'danger'
        title: 'Critical Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}: {{ .Annotations.description }}{{ end }}'

  - name: 'service-down-alerts'
    pagerduty_configs:
      - routing_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'warning-alerts'
    email_configs:
      - to: 'team@company.com'
        subject: 'Warning: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## Phase 4: Dashboards and Visualization

### 4.1 Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "Microservices Overview",
    "tags": ["microservices", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Service Health Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "up",
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
                {
                  "color": "red",
                  "value": 0
                },
                {
                  "color": "green",
                  "value": 1
                }
              ]
            }
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{ instance }} - {{ method }}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "{{ instance }}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

### 4.2 Business Metrics Dashboard

```python
# scripts/generate-business-dashboard.py
import json

def create_business_dashboard():
    """Generate Grafana dashboard for business metrics"""
    
    dashboard = {
        "dashboard": {
            "title": "Business Metrics Dashboard",
            "panels": [
                {
                    "id": 1,
                    "title": "Orders per Hour",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(orders_created_total[1h]) * 3600",
                            "legendFormat": "Orders/hour"
                        }
                    ]
                },
                {
                    "id": 2,
                    "title": "Revenue per Hour",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rate(order_value_dollars_sum[1h]) * 3600",
                            "legendFormat": "Revenue/hour"
                        }
                    ]
                },
                {
                    "id": 3,
                    "title": "Checkout Funnel",
                    "type": "piechart",
                    "targets": [
                        {
                            "expr": "checkout_funnel_total",
                            "legendFormat": "{{ step }}"
                        }
                    ]
                },
                {
                    "id": 4,
                    "title": "Average Order Value",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(order_value_dollars_sum[5m]) / rate(order_value_dollars_count[5m])",
                            "legendFormat": "AOV"
                        }
                    ]
                }
            ]
        }
    }
    
    return json.dumps(dashboard, indent=2)

if __name__ == "__main__":
    print(create_business_dashboard())
```

## Phase 5: Advanced Features

### 5.1 Distributed Tracing with OpenTelemetry

```python
# services/order-service/trace_example.py
from opentelemetry import trace
from opentelemetry.propagate import extract
import requests
import time

tracer = trace.get_tracer(__name__)

class OrderProcessor:
    def __init__(self):
        self.user_service_url = "http://user-service:8000"
        self.product_service_url = "http://product-service:8000"
        self.payment_service_url = "http://payment-service:8000"

    def process_order(self, order_data):
        """Process order with complete tracing"""
        with tracer.start_as_current_span("process_order") as span:
            span.set_attribute("order.id", order_data['order_id'])
            span.set_attribute("order.value", order_data['total_amount'])
            span.set_attribute("customer.id", order_data['customer_id'])
            
            try:
                # Step 1: Validate customer
                customer = self._validate_customer(order_data['customer_id'])
                span.set_attribute("customer.segment", customer.get('segment'))
                
                # Step 2: Check inventory
                inventory_ok = self._check_inventory(order_data['items'])
                span.set_attribute("inventory.available", inventory_ok)
                
                if not inventory_ok:
                    span.set_attribute("order.status", "failed")
                    span.set_attribute("failure.reason", "insufficient_inventory")
                    raise Exception("Insufficient inventory")
                
                # Step 3: Process payment
                payment_result = self._process_payment(order_data)
                span.set_attribute("payment.status", payment_result['status'])
                span.set_attribute("payment.transaction_id", payment_result['transaction_id'])
                
                # Step 4: Update inventory
                self._update_inventory(order_data['items'])
                
                # Step 5: Send confirmation
                self._send_confirmation(order_data)
                
                span.set_attribute("order.status", "completed")
                span.add_event("Order processing completed successfully")
                
                return {"status": "success", "order_id": order_data['order_id']}
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.set_attribute("order.status", "failed")
                raise

    def _validate_customer(self, customer_id):
        """Validate customer with distributed tracing"""
        with tracer.start_as_current_span("validate_customer") as span:
            span.set_attribute("customer.id", customer_id)
            
            # Propagate trace context to downstream service
            headers = {}
            trace.get_current_span().inject(headers)
            
            response = requests.get(
                f"{self.user_service_url}/users/{customer_id}",
                headers=headers,
                timeout=5
            )
            
            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.url", response.url)
            
            if response.status_code == 200:
                customer_data = response.json()
                span.add_event("Customer validated successfully")
                return customer_data
            else:
                span.add_event("Customer validation failed")
                raise Exception(f"Customer validation failed: {response.status_code}")

    def _check_inventory(self, items):
        """Check inventory across multiple products"""
        with tracer.start_as_current_span("check_inventory") as span:
            span.set_attribute("items.count", len(items))
            
            all_available = True
            for item in items:
                with tracer.start_as_current_span("check_product_inventory") as product_span:
                    product_span.set_attribute("product.id", item['product_id'])
                    product_span.set_attribute("quantity.requested", item['quantity'])
                    
                    # Simulate inventory check
                    time.sleep(0.01)  # Simulate database query
                    available = item.get('available_quantity', 100) >= item['quantity']
                    
                    product_span.set_attribute("inventory.available", available)
                    if not available:
                        all_available = False
                        product_span.add_event("Insufficient inventory")
            
            span.set_attribute("inventory.all_available", all_available)
            return all_available

    def _process_payment(self, order_data):
        """Process payment with tracing"""
        with tracer.start_as_current_span("process_payment") as span:
            span.set_attribute("payment.amount", order_data['total_amount'])
            span.set_attribute("payment.method", order_data.get('payment_method', 'card'))
            
            headers = {}
            trace.get_current_span().inject(headers)
            
            payment_payload = {
                "amount": order_data['total_amount'],
                "currency": "USD",
                "payment_method": order_data.get('payment_method', 'card'),
                "customer_id": order_data['customer_id']
            }
            
            response = requests.post(
                f"{self.payment_service_url}/process",
                json=payment_payload,
                headers=headers,
                timeout=10
            )
            
            span.set_attribute("http.status_code", response.status_code)
            
            if response.status_code == 200:
                result = response.json()
                span.add_event("Payment processed successfully")
                return result
            else:
                span.add_event("Payment processing failed")
                raise Exception(f"Payment failed: {response.status_code}")
```

### 5.2 Log Aggregation with ELK Stack

```conf
# monitoring/elk/logstash.conf
input {
  beats {
    port => 5044
  }
  
  tcp {
    port => 5000
    codec => json_lines
  }
  
  udp {
    port => 5000
    codec => json_lines
  }
}

filter {
  # Parse JSON logs
  if [message] {
    json {
      source => "message"
    }
  }
  
  # Extract trace information
  if [trace_id] {
    mutate {
      add_field => { "[@metadata][trace_id]" => "%{trace_id}" }
    }
  }
  
  # Add environment information
  mutate {
    add_field => { "environment" => "production" }
    add_field => { "infrastructure" => "kubernetes" }
  }
  
  # Parse timestamps
  date {
    match => [ "timestamp", "UNIX" ]
  }
  
  # Categorize log levels
  if [level] == "ERROR" or [level] == "FATAL" {
    mutate {
      add_tag => [ "error" ]
    }
  }
  
  if [level] == "WARN" {
    mutate {
      add_tag => [ "warning" ]
    }
  }
  
  # Extract business events
  if [event] {
    mutate {
      add_field => { "business_event" => "%{event}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "microservices-logs-%{+YYYY.MM.dd}"
  }
  
  # Output errors to separate index
  if "error" in [tags] {
    elasticsearch {
      hosts => ["elasticsearch:9200"]
      index => "microservices-errors-%{+YYYY.MM.dd}"
    }
  }
  
  # Debug output
  stdout {
    codec => rubydebug
  }
}
```

### 5.3 Load Testing and Chaos Engineering

```python
# scripts/load-test.py
import asyncio
import aiohttp
import time
import random
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class LoadTestConfig:
    base_url: str
    concurrent_users: int
    duration_seconds: int
    endpoints: List[Dict[str, str]]

class LoadTester:
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = []
        self.session = None

    async def run_load_test(self):
        """Run load test with specified configuration"""
        connector = aiohttp.TCPConnector(limit=self.config.concurrent_users)
        self.session = aiohttp.ClientSession(connector=connector)
        
        try:
            print(f"Starting load test with {self.config.concurrent_users} concurrent users")
            print(f"Duration: {self.config.duration_seconds} seconds")
            print(f"Target: {self.config.base_url}")
            
            # Create tasks for concurrent users
            tasks = []
            for user_id in range(self.config.concurrent_users):
                task = asyncio.create_task(self._user_simulation(user_id))
                tasks.append(task)
            
            # Run for specified duration
            await asyncio.sleep(self.config.duration_seconds)
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self._print_results()
            
        finally:
            if self.session:
                await self.session.close()

    async def _user_simulation(self, user_id: int):
        """Simulate a single user's behavior"""
        try:
            while True:
                # Simulate user behavior patterns
                await self._simulate_user_journey(user_id)
                
                # Random delay between requests
                await asyncio.sleep(random.uniform(0.5, 2.0))
                
        except asyncio.CancelledError:
            pass

    async def _simulate_user_journey(self, user_id: int):
        """Simulate a typical user journey"""
        # Browse products
        await self._make_request("GET", "/users", user_id)
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # View specific product
        product_id = random.randint(1, 100)
        await self._make_request("GET", f"/products/{product_id}", user_id)
        await asyncio.sleep(random.uniform(0.2, 1.0))
        
        # Simulate order creation (10% of users)
        if random.random() < 0.1:
            order_data = {
                "customer_id": user_id,
                "items": [{"product_id": product_id, "quantity": 1}],
                "total_amount": random.uniform(10, 500)
            }
            await self._make_request("POST", "/orders", user_id, json_data=order_data)

    async def _make_request(self, method: str, endpoint: str, user_id: int, json_data=None):
        """Make HTTP request and record metrics"""
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}{endpoint}"
            
            if method == "GET":
                async with self.session.get(url) as response:
                    await response.text()
                    status = response.status
            elif method == "POST":
                async with self.session.post(url, json=json_data) as response:
                    await response.text()
                    status = response.status
            
            duration = time.time() - start_time
            
            self.results.append({
                "user_id": user_id,
                "method": method,
                "endpoint": endpoint,
                "status": status,
                "duration": duration,
                "timestamp": time.time()
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append({
                "user_id": user_id,
                "method": method,
                "endpoint": endpoint,
                "status": 0,
                "duration": duration,
                "error": str(e),
                "timestamp": time.time()
            })

    def _print_results(self):
        """Print load test results"""
        if not self.results:
            print("No results recorded")
            return
        
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if 200 <= r['status'] < 400])
        error_requests = total_requests - successful_requests
        
        durations = [r['duration'] for r in self.results]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        p95_duration = sorted_durations[int(len(sorted_durations) * 0.95)]
        p99_duration = sorted_durations[int(len(sorted_durations) * 0.99)]
        
        print("\n" + "="*50)
        print("LOAD TEST RESULTS")
        print("="*50)
        print(f"Total Requests: {total_requests}")
        print(f"Successful Requests: {successful_requests}")
        print(f"Error Requests: {error_requests}")
        print(f"Success Rate: {(successful_requests/total_requests)*100:.2f}%")
        print(f"\nLatency Metrics:")
        print(f"Average: {avg_duration:.3f}s")
        print(f"Min: {min_duration:.3f}s")
        print(f"Max: {max_duration:.3f}s")
        print(f"95th Percentile: {p95_duration:.3f}s")
        print(f"99th Percentile: {p99_duration:.3f}s")
        
        # RPS calculation
        test_duration = max([r['timestamp'] for r in self.results]) - min([r['timestamp'] for r in self.results])
        rps = total_requests / test_duration if test_duration > 0 else 0
        print(f"\nRequests per Second: {rps:.2f}")

# Usage example
async def main():
    config = LoadTestConfig(
        base_url="http://localhost:8001",
        concurrent_users=50,
        duration_seconds=60,
        endpoints=[
            {"method": "GET", "path": "/users"},
            {"method": "GET", "path": "/products"},
            {"method": "POST", "path": "/orders"}
        ]
    )
    
    tester = LoadTester(config)
    await tester.run_load_test()

if __name__ == "__main__":
    asyncio.run(main())
```

```python
# scripts/chaos-engineering.py
import docker
import time
import random
import requests
from typing import List, Dict
import threading
import logging

class ChaosEngineer:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.logger = logging.getLogger(__name__)
        self.experiments = []

    def network_partition(self, service_name: str, duration: int = 30):
        """Simulate network partition by blocking traffic"""
        container = self.docker_client.containers.get(service_name)
        
        print(f"🔥 Creating network partition for {service_name} for {duration}s")
        
        # Block all network traffic
        container.exec_run("iptables -A INPUT -j DROP")
        container.exec_run("iptables -A OUTPUT -j DROP")
        
        time.sleep(duration)
        
        # Restore network
        container.exec_run("iptables -F")
        print(f"✅ Network partition resolved for {service_name}")

    def cpu_stress(self, service_name: str, duration: int = 60):
        """Create CPU stress on service"""
        container = self.docker_client.containers.get(service_name)
        
        print(f"🔥 Creating CPU stress on {service_name} for {duration}s")
        
        # Start stress test
        stress_cmd = f"stress --cpu 2 --timeout {duration}s"
        container.exec_run(stress_cmd, detach=True)
        
        time.sleep(duration + 5)
        print(f"✅ CPU stress test completed for {service_name}")

    def memory_pressure(self, service_name: str, memory_mb: int = 100, duration: int = 60):
        """Create memory pressure"""
        container = self.docker_client.containers.get(service_name)
        
        print(f"🔥 Creating memory pressure on {service_name} ({memory_mb}MB for {duration}s)")
        
        stress_cmd = f"stress --vm 1 --vm-bytes {memory_mb}M --timeout {duration}s"
        container.exec_run(stress_cmd, detach=True)
        
        time.sleep(duration + 5)
        print(f"✅ Memory pressure test completed for {service_name}")

    def random_container_kill(self, services: List[str]):
        """Randomly kill and restart containers"""
        service = random.choice(services)
        container = self.docker_client.containers.get(service)
        
        print(f"🔥 Killing container {service}")
        container.kill()
        
        time.sleep(5)
        
        print(f"🔄 Restarting container {service}")
        container.restart()
        
        time.sleep(10)
        print(f"✅ Container {service} restarted")

    def simulate_slow_response(self, service_name: str, delay_ms: int = 2000, duration: int = 60):
        """Simulate slow responses using traffic control"""
        container = self.docker_client.containers.get(service_name)
        
        print(f"🔥 Adding {delay_ms}ms delay to {service_name} for {duration}s")
        
        # Add network delay
        container.exec_run(f"tc qdisc add dev eth0 root netem delay {delay_ms}ms")
        
        time.sleep(duration)
        
        # Remove delay
        container.exec_run("tc qdisc del dev eth0 root")
        print(f"✅ Network delay removed from {service_name}")

    def run_chaos_experiment(self, experiment_config: Dict):
        """Run a specific chaos experiment"""
        experiment_type = experiment_config['type']
        service = experiment_config['service']
        duration = experiment_config.get('duration', 60)
        
        if experiment_type == 'network_partition':
            self.network_partition(service, duration)
        elif experiment_type == 'cpu_stress':
            self.cpu_stress(service, duration)
        elif experiment_type == 'memory_pressure':
            memory_mb = experiment_config.get('memory_mb', 100)
            self.memory_pressure(service, memory_mb, duration)
        elif experiment_type == 'slow_response':
            delay_ms = experiment_config.get('delay_ms', 2000)
            self.simulate_slow_response(service, delay_ms, duration)
        elif experiment_type == 'container_kill':
            services = experiment_config.get('services', [service])
            self.random_container_kill(services)

    def monitor_system_during_chaos(self, monitoring_url: str, duration: int):
        """Monitor system metrics during chaos experiments"""
        start_time = time.time()
        metrics = []
        
        while time.time() - start_time < duration:
            try:
                # Check service health
                response = requests.get(f"{monitoring_url}/health", timeout=5)
                metric = {
                    'timestamp': time.time(),
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds()
                }
                metrics.append(metric)
                
            except Exception as e:
                metric = {
                    'timestamp': time.time(),
                    'error': str(e),
                    'status_code': 0,
                    'response_time': None
                }
                metrics.append(metric)
            
            time.sleep(5)
        
        return metrics

    def chaos_monkey(self, services: List[str], duration: int = 300):
        """Run random chaos experiments"""
        experiments = [
            {'type': 'cpu_stress', 'duration': 30},
            {'type': 'memory_pressure', 'duration': 45, 'memory_mb': 150},
            {'type': 'slow_response', 'duration': 60, 'delay_ms': 1500},
            {'type': 'container_kill'},
        ]
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Random interval between experiments
            wait_time = random.randint(30, 120)
            time.sleep(wait_time)
            
            # Choose random service and experiment
            service = random.choice(services)
            experiment = random.choice(experiments).copy()
            experiment['service'] = service
            
            print(f"🐒 Chaos Monkey: Running {experiment['type']} on {service}")
            
            try:
                self.run_chaos_experiment(experiment)
            except Exception as e:
                print(f"❌ Chaos experiment failed: {e}")

# Usage example
def main():
    chaos = ChaosEngineer()
    services = ['user-service', 'product-service', 'order-service']
    
    # Run specific experiments
    experiments = [
        {
            'type': 'cpu_stress',
            'service': 'user-service',
            'duration': 60
        },
        {
            'type': 'network_partition',
            'service': 'product-service',
            'duration': 30
        },
        {
            'type': 'memory_pressure',
            'service': 'order-service',
            'duration': 45,
            'memory_mb': 200
        }
    ]
    
    for experiment in experiments:
        chaos.run_chaos_experiment(experiment)
        time.sleep(30)  # Wait between experiments
    
    # Or run chaos monkey
    # chaos.chaos_monkey(services, duration=600)

if __name__ == "__main__":
    main()
```

## Deliverables and Assessment

### Required Deliverables

1. **Complete Monitoring Stack**
   - [ ] Docker Compose setup with all monitoring components
   - [ ] Prometheus configuration with custom metrics
   - [ ] Grafana dashboards (service health + business metrics)
   - [ ] Jaeger distributed tracing setup
   - [ ] ELK stack for log aggregation
   - [ ] AlertManager with multiple notification channels

2. **Instrumented Microservices**
   - [ ] At least 3 microservices with full monitoring
   - [ ] Custom business metrics implementation
   - [ ] Distributed tracing across service calls
   - [ ] Structured logging with correlation IDs
   - [ ] Health check endpoints

3. **Advanced Features**
   - [ ] Load testing framework and results
   - [ ] Chaos engineering experiments
   - [ ] SLA/SLO definitions and monitoring
   - [ ] Automated alerting with escalation rules
   - [ ] Performance benchmarks and optimization

4. **Documentation**
   - [ ] Setup and deployment guide
   - [ ] Dashboard usage instructions
   - [ ] Troubleshooting playbook
   - [ ] Performance tuning recommendations
   - [ ] Incident response procedures

### Assessment Criteria

- **Architecture Design (25%)**
  - Monitoring stack completeness
  - Scalability considerations
  - Security implementation

- **Implementation Quality (30%)**
  - Code quality and organization
  - Error handling and resilience
  - Performance optimization

- **Monitoring Effectiveness (25%)**
  - Metric coverage and usefulness
  - Dashboard design and usability
  - Alert accuracy and actionability

- **Documentation and Presentation (20%)**
  - Clarity and completeness
  - Real-world applicability
  - Demonstration of understanding

### Next Steps

After completing this project, consider:
- **Service Mesh Integration**: Implement Istio or Linkerd for advanced observability
- **ML-based Alerting**: Use machine learning for anomaly detection
- **Cost Optimization**: Implement monitoring cost controls and data retention policies
- **Multi-cluster Monitoring**: Extend to federated Prometheus setup
- **Custom Exporters**: Build domain-specific metrics exporters
