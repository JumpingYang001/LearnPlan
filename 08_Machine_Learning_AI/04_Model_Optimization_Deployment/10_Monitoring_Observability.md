# Monitoring and Observability

## Topics
- Metrics for ML systems
- Feature and prediction drift
- Logging and tracing for ML systems
- Comprehensive monitoring solutions

### Example: Prometheus Metrics with FastAPI
```python
from prometheus_client import start_http_server, Summary
import time

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

@REQUEST_TIME.time()
def process_request():
    time.sleep(1)

start_http_server(8000)
while True:
    process_request()
```
