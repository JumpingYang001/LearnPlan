# Resilience Patterns in Microservices

*Duration: 2-3 weeks*

## Introduction

Resilience patterns are essential design strategies that help microservices handle failures gracefully, maintain system stability, and provide a better user experience even when dependencies fail. In distributed systems, failures are inevitable - networks partition, services become unavailable, and resources get exhausted. Resilience patterns help us build systems that can adapt and recover from these failures.

### Why Resilience Patterns Matter

In a microservices architecture, a single service failure can cascade through the entire system, potentially bringing down multiple services. Resilience patterns help us:

- **Prevent cascade failures** from propagating through the system
- **Maintain service availability** even when dependencies fail
- **Provide graceful degradation** of functionality
- **Improve overall system stability** and user experience
- **Enable faster recovery** from failures

## Core Resilience Patterns

### 1. Circuit Breaker Pattern

The Circuit Breaker pattern prevents a microservice from repeatedly trying to call a failing service, which could lead to resource exhaustion and cascade failures.

#### How Circuit Breaker Works

The circuit breaker has three states:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Service is failing, requests are rejected immediately
- **HALF-OPEN**: Testing if service has recovered

```
    [CLOSED] ──failure threshold──> [OPEN]
        │                             │
        │<──success threshold─── [HALF-OPEN]
        │                             │
        └─────────────timeout─────────┘
```

#### Implementation Examples

**Python Implementation with Custom Circuit Breaker:**

```python
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps

class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func: Callable, *args, **kwargs) -> Any:
        with self.lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    print(f"Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        with self.lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                print(f"Circuit breaker transitioning to CLOSED")
    
    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                print(f"Circuit breaker transitioning to OPEN after {self.failure_count} failures")

# Usage Example
import requests
from requests.exceptions import RequestException

@CircuitBreaker(failure_threshold=3, timeout=30, expected_exception=RequestException)
def call_external_service(url: str):
    """Call external service with circuit breaker protection"""
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()

# Example usage with fallback
def get_user_data(user_id: str):
    try:
        return call_external_service(f"https://api.example.com/users/{user_id}")
    except CircuitBreakerError:
        print("Circuit breaker is open, returning cached data")
        return get_cached_user_data(user_id)
    except RequestException as e:
        print(f"Service call failed: {e}")
        return get_cached_user_data(user_id)

def get_cached_user_data(user_id: str):
    """Fallback implementation"""
    return {"id": user_id, "name": "Unknown", "cached": True}
```

**Advanced Circuit Breaker with Metrics:**

```python
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

@dataclass
class CircuitBreakerMetrics:
    total_requests: int = 0
    failed_requests: int = 0
    successful_requests: int = 0
    circuit_open_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

class AdvancedCircuitBreaker:
    def __init__(self, 
                 failure_threshold: int = 5,
                 failure_rate_threshold: float = 0.5,
                 timeout: int = 60,
                 monitoring_window: int = 300):  # 5 minutes
        self.failure_threshold = failure_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.timeout = timeout
        self.monitoring_window = monitoring_window
        
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.request_history: List[Dict] = []
        self.lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        async with self.lock:
            await self._update_state()
            
            if self.state == CircuitBreakerState.OPEN:
                self.metrics.circuit_open_count += 1
                raise CircuitBreakerError("Circuit breaker is OPEN")
        
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)
            await self._record_success(start_time)
            return result
        except Exception as e:
            await self._record_failure(start_time, e)
            raise e
    
    async def _update_state(self):
        now = datetime.now()
        
        # Clean old requests from history
        self.request_history = [
            req for req in self.request_history
            if now - req['timestamp'] <= timedelta(seconds=self.monitoring_window)
        ]
        
        # Update metrics
        self.metrics.total_requests = len(self.request_history)
        self.metrics.failed_requests = sum(1 for req in self.request_history if not req['success'])
        self.metrics.successful_requests = self.metrics.total_requests - self.metrics.failed_requests
        
        # State transitions
        if self.state == CircuitBreakerState.OPEN:
            if self.metrics.last_failure_time and \
               now - self.metrics.last_failure_time >= timedelta(seconds=self.timeout):
                self.state = CircuitBreakerState.HALF_OPEN
        
        elif self.state == CircuitBreakerState.CLOSED:
            if (self.metrics.failed_requests >= self.failure_threshold or 
                self.metrics.failure_rate >= self.failure_rate_threshold):
                self.state = CircuitBreakerState.OPEN
    
    async def _record_success(self, start_time: datetime):
        async with self.lock:
            self.request_history.append({
                'timestamp': start_time,
                'success': True,
                'duration': (datetime.now() - start_time).total_seconds()
            })
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
    
    async def _record_failure(self, start_time: datetime, error: Exception):
        async with self.lock:
            self.request_history.append({
                'timestamp': start_time,
                'success': False,
                'error': str(error),
                'duration': (datetime.now() - start_time).total_seconds()
            })
            self.metrics.last_failure_time = datetime.now()

# Usage with async HTTP client
async def fetch_user_data(session: aiohttp.ClientSession, user_id: str):
    async with session.get(f"https://api.example.com/users/{user_id}") as response:
        response.raise_for_status()
        return await response.json()

# Circuit breaker instance
user_service_breaker = AdvancedCircuitBreaker(
    failure_threshold=3,
    failure_rate_threshold=0.6,
    timeout=30
)

async def get_user_with_circuit_breaker(user_id: str):
    async with aiohttp.ClientSession() as session:
        try:
            return await user_service_breaker.call(fetch_user_data, session, user_id)
        except CircuitBreakerError:
            return {"error": "User service temporarily unavailable"}
```

### 2. Bulkhead Pattern

The Bulkhead pattern isolates critical resources to prevent failures in one part of the system from affecting other parts. Just like compartments in a ship prevent the entire vessel from sinking if one compartment is breached.

#### Types of Bulkheads

1. **Thread Pool Bulkheads**: Separate thread pools for different operations
2. **Connection Pool Bulkheads**: Separate connection pools for different services
3. **Resource Bulkheads**: Separate CPU, memory, or disk resources

#### Implementation Examples

**Thread Pool Bulkhead in Python:**

```python
import asyncio
import concurrent.futures
from typing import Dict, Any, Callable
from enum import Enum

class ServiceType(Enum):
    CRITICAL = "critical"
    NORMAL = "normal"
    BACKGROUND = "background"

class BulkheadExecutor:
    def __init__(self):
        # Separate thread pools for different service types
        self.executors = {
            ServiceType.CRITICAL: concurrent.futures.ThreadPoolExecutor(
                max_workers=10, thread_name_prefix="critical"
            ),
            ServiceType.NORMAL: concurrent.futures.ThreadPoolExecutor(
                max_workers=5, thread_name_prefix="normal"
            ),
            ServiceType.BACKGROUND: concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="background"
            )
        }
        
        # Queue size limits to prevent memory exhaustion
        self.queue_limits = {
            ServiceType.CRITICAL: 100,
            ServiceType.NORMAL: 50,
            ServiceType.BACKGROUND: 20
        }
        
        self.queue_sizes = {service_type: 0 for service_type in ServiceType}
    
    async def submit_task(self, 
                         service_type: ServiceType, 
                         func: Callable, 
                         *args, **kwargs) -> Any:
        """Submit task to appropriate bulkhead"""
        
        # Check queue limit
        if self.queue_sizes[service_type] >= self.queue_limits[service_type]:
            raise Exception(f"Queue limit reached for {service_type.value} service")
        
        executor = self.executors[service_type]
        loop = asyncio.get_event_loop()
        
        self.queue_sizes[service_type] += 1
        
        try:
            result = await loop.run_in_executor(executor, func, *args, **kwargs)
            return result
        finally:
            self.queue_sizes[service_type] -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics"""
        stats = {}
        for service_type, executor in self.executors.items():
            stats[service_type.value] = {
                "active_threads": executor._threads and len(executor._threads) or 0,
                "queue_size": self.queue_sizes[service_type],
                "queue_limit": self.queue_limits[service_type]
            }
        return stats
    
    def shutdown(self):
        """Shutdown all executors"""
        for executor in self.executors.values():
            executor.shutdown(wait=True)

# Usage example
bulkhead = BulkheadExecutor()

# Simulate different types of operations
def critical_operation(data):
    """Critical business operation - must not be affected by other operations"""
    time.sleep(0.1)  # Simulate quick critical work
    return f"Critical result for {data}"

def normal_operation(data):
    """Normal business operation"""
    time.sleep(0.5)  # Simulate normal work
    return f"Normal result for {data}"

def background_operation(data):
    """Background operation like analytics, cleanup, etc."""
    time.sleep(2.0)  # Simulate slow background work
    return f"Background result for {data}"

async def process_requests():
    """Process different types of requests using bulkheads"""
    tasks = []
    
    # Critical operations get dedicated resources
    for i in range(5):
        task = bulkhead.submit_task(
            ServiceType.CRITICAL, 
            critical_operation, 
            f"critical-{i}"
        )
        tasks.append(task)
    
    # Normal operations use separate pool
    for i in range(3):
        task = bulkhead.submit_task(
            ServiceType.NORMAL, 
            normal_operation, 
            f"normal-{i}"
        )
        tasks.append(task)
    
    # Background operations won't affect critical ones
    for i in range(2):
        task = bulkhead.submit_task(
            ServiceType.BACKGROUND, 
            background_operation, 
            f"background-{i}"
        )
        tasks.append(task)
    
    # Wait for all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Print stats
    print("Bulkhead Statistics:")
    for service_type, stats in bulkhead.get_stats().items():
        print(f"  {service_type}: {stats}")
    
    return results

# Run example
# asyncio.run(process_requests())
```

**Connection Pool Bulkhead:**

```python
import aiohttp
import asyncio
from typing import Dict, Optional
import aioredis

class ConnectionBulkhead:
    def __init__(self):
        # Separate connection pools for different services
        self.db_connector = None
        self.cache_connector = None
        self.external_api_connector = None
        
        # Connection limits per service type
        self.connection_limits = {
            'database': {'limit': 20, 'timeout': 30},
            'cache': {'limit': 10, 'timeout': 5},
            'external_api': {'limit': 5, 'timeout': 10}
        }
    
    async def initialize(self):
        """Initialize all connection pools"""
        
        # Database connection pool
        self.db_connector = aiohttp.TCPConnector(
            limit=self.connection_limits['database']['limit'],
            limit_per_host=10,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Cache connection pool (Redis)
        self.cache_pool = aioredis.ConnectionPool.from_url(
            "redis://localhost",
            max_connections=self.connection_limits['cache']['limit'],
            socket_timeout=self.connection_limits['cache']['timeout']
        )
        
        # External API connection pool
        self.external_api_connector = aiohttp.TCPConnector(
            limit=self.connection_limits['external_api']['limit'],
            limit_per_host=2,
            keepalive_timeout=10
        )
    
    async def get_database_session(self):
        """Get database session from dedicated pool"""
        session = aiohttp.ClientSession(
            connector=self.db_connector,
            timeout=aiohttp.ClientTimeout(
                total=self.connection_limits['database']['timeout']
            )
        )
        return session
    
    async def get_cache_connection(self):
        """Get cache connection from dedicated pool"""
        return aioredis.Redis(connection_pool=self.cache_pool)
    
    async def get_external_api_session(self):
        """Get external API session from dedicated pool"""
        session = aiohttp.ClientSession(
            connector=self.external_api_connector,
            timeout=aiohttp.ClientTimeout(
                total=self.connection_limits['external_api']['timeout']
            )
        )
        return session
    
    async def close(self):
        """Close all connection pools"""
        if self.db_connector:
            await self.db_connector.close()
        if self.cache_pool:
            await self.cache_pool.disconnect()
        if self.external_api_connector:
            await self.external_api_connector.close()

# Usage example
connection_bulkhead = ConnectionBulkhead()

async def fetch_user_data(user_id: str):
    """Fetch user data using isolated connection pools"""
    
    # Use database pool - won't be affected by external API issues
    db_session = await connection_bulkhead.get_database_session()
    try:
        async with db_session.get(f"http://db-service/users/{user_id}") as response:
            user_data = await response.json()
    finally:
        await db_session.close()
    
    # Use cache pool - separate from database pool
    cache = await connection_bulkhead.get_cache_connection()
    try:
        await cache.setex(f"user:{user_id}", 300, json.dumps(user_data))
    finally:
        await cache.close()
    
    # Use external API pool - isolated from internal services
    api_session = await connection_bulkhead.get_external_api_session()
    try:
        async with api_session.get(f"https://external-api.com/enrich/{user_id}") as response:
            if response.status == 200:
                enrichment_data = await response.json()
                user_data.update(enrichment_data)
    except Exception as e:
        # External API failure doesn't affect internal data
        print(f"External enrichment failed: {e}")
    finally:
        await api_session.close()
    
    return user_data
```

**CPU/Memory Resource Bulkhead using Docker:**

```yaml
# docker-compose.yml - Resource isolation using containers
version: '3.8'

services:
  critical-service:
    image: my-app:latest
    deploy:
      resources:
        limits:
          cpus: '2.0'      # Dedicated CPU cores
          memory: 2G       # Dedicated memory
        reservations:
          cpus: '1.0'      # Reserved minimum
          memory: 1G
    environment:
      - SERVICE_TYPE=critical
    
  normal-service:
    image: my-app:latest
    deploy:
      resources:
        limits:
          cpus: '1.0'      # Limited CPU
          memory: 1G       # Limited memory
        reservations:
          cpus: '0.5'
          memory: 512M
    environment:
      - SERVICE_TYPE=normal
    
  background-service:
    image: my-app:latest
    deploy:
      resources:
        limits:
          cpus: '0.5'      # Minimal CPU
          memory: 512M     # Minimal memory
        reservations:
          cpus: '0.1'
          memory: 256M
    environment:
      - SERVICE_TYPE=background
```

**Kubernetes Resource Bulkhead:**

```yaml
# critical-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: critical-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: critical-service
  template:
    metadata:
      labels:
        app: critical-service
    spec:
      containers:
      - name: app
        image: my-app:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: SERVICE_TYPE
          value: "critical"
      # Node affinity to ensure isolation
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node-type
                operator: In
                values: ["critical-workload"]

---
# background-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: background-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: background-service
  template:
    metadata:
      labels:
        app: background-service
    spec:
      containers:
      - name: app
        image: my-app:latest
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        env:
        - name: SERVICE_TYPE
          value: "background"
      # Allow scheduling on any available node
      tolerations:
      - key: "background-workload"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### 3. Rate Limiting Pattern

Rate limiting controls the rate of requests to prevent system overload and ensure fair resource usage among clients. It's essential for protecting services from traffic spikes and abuse.

#### Types of Rate Limiting

1. **Fixed Window**: Simple counter reset at fixed intervals
2. **Sliding Window**: More accurate, tracks requests over a moving time window
3. **Token Bucket**: Allows burst traffic up to a limit
4. **Leaky Bucket**: Smooths out traffic spikes

#### Implementation Examples

**Token Bucket Rate Limiter:**

```python
import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TokenBucket:
    capacity: int           # Maximum tokens
    tokens: float          # Current tokens
    fill_rate: float       # Tokens per second
    last_update: float     # Last update timestamp
    
    def __post_init__(self):
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        with self.lock:
            now = time.time()
            
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.last_update = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_tokens(self) -> float:
        """Get current token count"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            return self.tokens

class RateLimiter:
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()
    
    def create_bucket(self, 
                     key: str, 
                     capacity: int, 
                     fill_rate: float) -> TokenBucket:
        """Create a new token bucket for a key"""
        bucket = TokenBucket(
            capacity=capacity,
            tokens=capacity,
            fill_rate=fill_rate,
            last_update=time.time()
        )
        
        with self.lock:
            self.buckets[key] = bucket
        
        return bucket
    
    def is_allowed(self, 
                   key: str, 
                   capacity: int = 10, 
                   fill_rate: float = 1.0, 
                   tokens: int = 1) -> bool:
        """Check if request is allowed for the given key"""
        
        # Get or create bucket
        if key not in self.buckets:
            self.create_bucket(key, capacity, fill_rate)
        
        bucket = self.buckets[key]
        return bucket.consume(tokens)
    
    def get_bucket_status(self, key: str) -> Optional[Dict]:
        """Get status of a bucket"""
        if key not in self.buckets:
            return None
        
        bucket = self.buckets[key]
        return {
            "capacity": bucket.capacity,
            "current_tokens": bucket.get_tokens(),
            "fill_rate": bucket.fill_rate
        }

# Usage example
rate_limiter = RateLimiter()

def api_endpoint(user_id: str, request_data: dict):
    """API endpoint with rate limiting"""
    
    # Different rate limits for different users/tiers
    if user_id.startswith("premium_"):
        allowed = rate_limiter.is_allowed(
            key=f"user:{user_id}",
            capacity=100,    # 100 requests
            fill_rate=10,    # 10 requests per second
            tokens=1
        )
    else:
        allowed = rate_limiter.is_allowed(
            key=f"user:{user_id}",
            capacity=20,     # 20 requests
            fill_rate=2,     # 2 requests per second
            tokens=1
        )
    
    if not allowed:
        status = rate_limiter.get_bucket_status(f"user:{user_id}")
        return {
            "error": "Rate limit exceeded",
            "retry_after": max(0, (1 - status["current_tokens"]) / status["fill_rate"]),
            "status": status
        }
    
    # Process request
    return {"data": f"Processed request for {user_id}"}

# Test rate limiter
def test_rate_limiter():
    """Test the rate limiter behavior"""
    user_id = "premium_user_123"
    
    for i in range(15):
        result = api_endpoint(user_id, {"request": i})
        print(f"Request {i}: {result}")
        time.sleep(0.1)  # Small delay between requests

# Decorator for rate limiting
def rate_limit(capacity: int = 10, fill_rate: float = 1.0):
    """Decorator for rate limiting functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Use function name as key
            key = f"function:{func.__name__}"
            
            if not rate_limiter.is_allowed(key, capacity, fill_rate):
                raise Exception(f"Rate limit exceeded for {func.__name__}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(capacity=5, fill_rate=0.5)  # 5 requests, refill at 0.5/second
def expensive_operation(data):
    """Rate limited expensive operation"""
    time.sleep(1)  # Simulate expensive work
    return f"Processed: {data}"
```

**Sliding Window Rate Limiter:**

```python
import time
import threading
from collections import deque
from typing import Dict, List

class SlidingWindowRateLimiter:
    def __init__(self, window_size: int = 60):
        """
        window_size: Window size in seconds
        """
        self.window_size = window_size
        self.requests: Dict[str, deque] = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str, limit: int) -> bool:
        """Check if request is allowed within the sliding window"""
        now = time.time()
        
        with self.lock:
            # Initialize request queue for new keys
            if key not in self.requests:
                self.requests[key] = deque()
            
            request_times = self.requests[key]
            
            # Remove old requests outside the window
            while request_times and request_times[0] <= now - self.window_size:
                request_times.popleft()
            
            # Check if we're within the limit
            if len(request_times) < limit:
                request_times.append(now)
                return True
            
            return False
    
    def get_request_count(self, key: str) -> int:
        """Get current request count in the window"""
        now = time.time()
        
        with self.lock:
            if key not in self.requests:
                return 0
            
            request_times = self.requests[key]
            
            # Remove old requests
            while request_times and request_times[0] <= now - self.window_size:
                request_times.popleft()
            
            return len(request_times)

# Usage example
sliding_limiter = SlidingWindowRateLimiter(window_size=60)  # 1-minute window

def sliding_window_api(user_id: str):
    """API with sliding window rate limiting"""
    limit = 10  # 10 requests per minute
    
    if not sliding_limiter.is_allowed(f"user:{user_id}", limit):
        current_count = sliding_limiter.get_request_count(f"user:{user_id}")
        return {
            "error": "Rate limit exceeded",
            "current_requests": current_count,
            "limit": limit,
            "window_size": "60 seconds"
        }
    
    return {"data": "Request processed successfully"}
```

**Distributed Rate Limiter using Redis:**
```python
import aioredis
import asyncio
from typing import Dict, Any, Optional

class RedisRateLimiter:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, limit: int) -> bool:
        """Check if request is allowed for the given key"""
        current = await self.redis.get(key)
        
        if current is None:
            # Key does not exist, create it with expiration
            await self.redis.set(key, 1, ex=60)
            return True
        
        current = int(current)
        
        if current < limit:
            # Increment and allow request
            await self.redis.incr(key)
            return True
        
        return False

# Usage example
async def rate_limited_task(user_id: str):
    redis = await aioredis.from_url("redis://localhost")
    rate_limiter = RedisRateLimiter(redis)
    
    allowed = await rate_limiter.is_allowed(f"user:{user_id}", 100)
    if not allowed:
        return {"error": "Rate limit exceeded"}
    
    # Process the request
    return {"data": "Request processed"}
```

### 4. Retry Pattern with Exponential Backoff

The Retry pattern automatically retries failed operations with intelligent backoff strategies to handle transient failures without overwhelming the failing service.

#### Retry Strategies

1. **Linear Backoff**: Fixed delay between retries
2. **Exponential Backoff**: Exponentially increasing delay
3. **Exponential Backoff with Jitter**: Adds randomization to prevent thundering herd
4. **Circuit Breaker Integration**: Combines with circuit breaker for smart retries

#### Implementation Examples

**Advanced Retry Mechanism:**

```python
import asyncio
import random
import time
import logging
from typing import Callable, Any, Optional, Type, Union, List
from dataclasses import dataclass
from enum import Enum
import aiohttp

class RetryStrategy(Enum):
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0        # Base delay in seconds
    max_delay: float = 60.0        # Maximum delay in seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    jitter_factor: float = 0.1     # Jitter factor (0.0 to 1.0)
    backoff_multiplier: float = 2.0
    retryable_exceptions: tuple = (Exception,)

class RetryableError(Exception):
    """Exception that should trigger a retry"""
    pass

class NonRetryableError(Exception):
    """Exception that should NOT trigger a retry"""
    pass

class RetryHandler:
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            exponential_delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            jitter = exponential_delay * self.config.jitter_factor * random.random()
            delay = exponential_delay + jitter
        
        return min(delay, self.config.max_delay)
    
    async def execute_with_retry(self, 
                               func: Callable, 
                               *args, 
                               **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.logger.info(f"Attempt {attempt}/{self.config.max_attempts}")
                result = await func(*args, **kwargs)
                
                if attempt > 1:
                    self.logger.info(f"Operation succeeded on attempt {attempt}")
                
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts:
                    self.logger.error(f"All {self.config.max_attempts} attempts failed. Last error: {e}")
                    break
                
                delay = self.calculate_delay(attempt)
                self.logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f} seconds...")
                
                await asyncio.sleep(delay)
            
            except Exception as e:
                # Non-retryable exception
                self.logger.error(f"Non-retryable error: {e}")
                raise e
        
        raise last_exception

# Decorator for automatic retries
def retry(config: RetryConfig):
    """Decorator for automatic retry functionality"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(config)
            return await retry_handler.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator

# Usage examples
@retry(RetryConfig(
    max_attempts=5,
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    base_delay=0.5,
    max_delay=30.0,
    retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError)
))
async def fetch_user_data(user_id: str) -> dict:
    """Fetch user data with automatic retry"""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.example.com/users/{user_id}",
            timeout=aiohttp.ClientTimeout(total=5)
        ) as response:
            if response.status >= 500:
                # Server errors are retryable
                raise RetryableError(f"Server error: {response.status}")
            elif response.status >= 400:
                # Client errors are not retryable
                raise NonRetryableError(f"Client error: {response.status}")
            
            return await response.json()

# Manual retry with custom logic
class DatabaseRetryHandler(RetryHandler):
    """Specialized retry handler for database operations"""
    
    def __init__(self):
        config = RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=0.1,
            max_delay=5.0,
            retryable_exceptions=(ConnectionError, TimeoutError)
        )
        super().__init__(config)
    
    async def execute_query(self, query: str, params: dict = None):
        """Execute database query with retry logic"""
        
        async def db_operation():
            # Simulate database operation
            if random.random() < 0.3:  # 30% failure rate
                raise ConnectionError("Database connection failed")
            
            return {"result": f"Query executed: {query}"}
        
        return await self.execute_with_retry(db_operation)

# Test the retry mechanism
async def test_retry_patterns():
    """Test different retry patterns"""
    
    # Test exponential backoff with jitter
    print("Testing exponential backoff with jitter:")
    retry_handler = RetryHandler(RetryConfig(
        max_attempts=4,
        strategy=RetryStrategy.EXPONENTIAL_JITTER,
        base_delay=0.5,
        jitter_factor=0.2
    ))
    
    for attempt in range(1, 5):
        delay = retry_handler.calculate_delay(attempt)
        print(f"Attempt {attempt}: delay = {delay:.2f} seconds")
    
    # Test database retry
    print("\nTesting database retry:")
    db_handler = DatabaseRetryHandler()
    try:
        result = await db_handler.execute_query("SELECT * FROM users")
        print(f"Database query result: {result}")
    except Exception as e:
        print(f"Database query failed after all retries: {e}")

# asyncio.run(test_retry_patterns())
```

**Context-Aware Retry with Circuit Breaker Integration:**

```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Callable, Any

class SmartRetryHandler:
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_history: Dict[str, List] = {}
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=3,
                timeout=30
            )
        return self.circuit_breakers[service_name]
    
    def should_retry(self, 
                    service_name: str, 
                    exception: Exception, 
                    attempt: int) -> bool:
        """Intelligent retry decision based on context"""
        
        # Check circuit breaker state
        circuit_breaker = self.get_circuit_breaker(service_name)
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            return False
        
        # Check recent failure rate
        recent_failures = self.get_recent_failures(service_name)
        if len(recent_failures) > 10:  # Too many recent failures
            return False
        
        # Exception-specific retry logic
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return attempt <= 3
        elif isinstance(exception, aiohttp.ClientResponseError):
            if exception.status in [500, 502, 503, 504]:  # Server errors
                return attempt <= 2
            else:  # Client errors
                return False
        
        return attempt <= 1
    
    def get_recent_failures(self, service_name: str) -> List:
        """Get failures in the last 5 minutes"""
        if service_name not in self.retry_history:
            return []
        
        cutoff = datetime.now() - timedelta(minutes=5)
        return [
            failure for failure in self.retry_history[service_name]
            if failure['timestamp'] > cutoff
        ]
    
    def record_failure(self, service_name: str, exception: Exception):
        """Record failure for analysis"""
        if service_name not in self.retry_history:
            self.retry_history[service_name] = []
        
        self.retry_history[service_name].append({
            'timestamp': datetime.now(),
            'exception': str(exception),
            'type': type(exception).__name__
        })
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(hours=1)
        self.retry_history[service_name] = [
            failure for failure in self.retry_history[service_name]
            if failure['timestamp'] > cutoff
        ]
    
    async def execute_with_smart_retry(self, 
                                     service_name: str, 
                                     func: Callable, 
                                     *args, **kwargs) -> Any:
        """Execute with intelligent retry logic"""
        circuit_breaker = self.get_circuit_breaker(service_name)
        attempt = 0
        
        while True:
            attempt += 1
            
            try:
                # Try to execute through circuit breaker
                result = await circuit_breaker.call(func, *args, **kwargs)
                return result
                
            except CircuitBreakerError:
                raise Exception(f"Service {service_name} is currently unavailable")
            
            except Exception as e:
                self.record_failure(service_name, e)
                
                if not self.should_retry(service_name, e, attempt):
                    raise e
                
                # Calculate adaptive delay
                delay = self.calculate_adaptive_delay(service_name, attempt)
                await asyncio.sleep(delay)
    
    def calculate_adaptive_delay(self, service_name: str, attempt: int) -> float:
        """Calculate delay based on service health and attempt number"""
        base_delay = 0.5 * (2 ** (attempt - 1))  # Exponential backoff
        
        # Adjust based on recent failure rate
        recent_failures = self.get_recent_failures(service_name)
        failure_rate = len(recent_failures) / 60  # failures per minute
        
        # Increase delay if service is unhealthy
        health_multiplier = 1 + min(failure_rate * 0.1, 2.0)
        
        return min(base_delay * health_multiplier, 30.0)

# Usage example
smart_retry = SmartRetryHandler()

async def call_user_service(user_id: str):
    """Call user service with smart retry"""
    
    async def service_call():
        # Simulate service call
        if random.random() < 0.4:  # 40% failure rate
            raise aiohttp.ClientResponseError(
                request_info=None,
                history=None,
                status=503
            )
        return {"user_id": user_id, "name": "John Doe"}
    
    return await smart_retry.execute_with_smart_retry(
        "user-service", 
        service_call
    )
```

**Retry with Dead Letter Queue:**

```python
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import aioredis

class DeadLetterQueue:
    def __init__(self, redis_client: aiorededis.Redis):
        self.redis = redis_client
    
    async def send_to_dlq(self, 
                         operation_id: str, 
                         payload: Dict[Any, Any], 
                         error: str, 
                         retry_count: int):
        """Send failed operation to dead letter queue"""
        dlq_item = {
            "operation_id": operation_id,
            "payload": payload,
            "error": error,
            "retry_count": retry_count,
            "failed_at": datetime.now().isoformat(),
            "can_retry_after": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        await self.redis.lpush("dlq:failed_operations", json.dumps(dlq_item))
        print(f"Sent operation {operation_id} to DLQ after {retry_count} retries")
    
    async def process_dlq(self):
        """Process items from dead letter queue"""
        while True:
            try:
                item_json = await self.redis.brpop("dlq:failed_operations", timeout=10)
                if not item_json:
                    continue
                
                item = json.loads(item_json[1])
                
                # Check if item can be retried
                can_retry_after = datetime.fromisoformat(item["can_retry_after"])
                if datetime.now() < can_retry_after:
                    # Put back in queue for later
                    await self.redis.lpush("dlq:failed_operations", item_json[1])
                    await asyncio.sleep(60)  # Wait a minute
                    continue
                
                # Try to process the item again
                success = await self.retry_dlq_item(item)
                if not success:
                    # Update retry time and put back
                    item["can_retry_after"] = (datetime.now() + timedelta(hours=2)).isoformat()
                    await self.redis.lpush("dlq:failed_operations", json.dumps(item))
                
            except Exception as e:
                print(f"Error processing DLQ: {e}")
                await asyncio.sleep(10)
    
    async def retry_dlq_item(self, item: Dict) -> bool:
        """Retry processing a DLQ item"""
        try:
            # Implement your retry logic here
            print(f"Retrying operation {item['operation_id']}")
            # ... actual retry logic ...
            return True
        except Exception as e:
            print(f"DLQ retry failed: {e}")
            return False

class RetryWithDLQ:
    def __init__(self, redis_client: aiorededis.Redis):
        self.dlq = DeadLetterQueue(redis_client)
        self.retry_config = RetryConfig(max_attempts=3)
    
    async def execute_with_dlq(self, 
                              operation_id: str, 
                              func: Callable, 
                              payload: Dict[Any, Any]) -> Any:
        """Execute operation with DLQ fallback"""
        retry_handler = RetryHandler(self.retry_config)
        
        try:
            return await retry_handler.execute_with_retry(func, payload)
        except Exception as e:
            # All retries failed, send to DLQ
            await self.dlq.send_to_dlq(
                operation_id=operation_id,
                payload=payload,
                error=str(e),
                retry_count=self.retry_config.max_attempts
            )
            raise e
```

### 5. Timeout Pattern

Timeout patterns prevent operations from hanging indefinitely, ensuring system responsiveness and resource management. Different types of timeouts serve different purposes in distributed systems.

#### Types of Timeouts

1. **Connection Timeout**: Time to establish connection
2. **Request Timeout**: Time for complete request-response cycle
3. **Idle Timeout**: Time before closing idle connections
4. **Keep-Alive Timeout**: Time to keep connection alive

#### Implementation Examples

**Comprehensive Timeout Handler:**

```python
import asyncio
import aiohttp
import time
from typing import Any, Optional, Callable, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass

@dataclass
class TimeoutConfig:
    connect_timeout: float = 5.0      # Connection establishment
    request_timeout: float = 30.0     # Total request time
    read_timeout: float = 10.0        # Time between data reads
    pool_timeout: float = 1.0         # Time to get connection from pool

class TimeoutError(Exception):
    """Custom timeout error with context"""
    def __init__(self, message: str, timeout_type: str, duration: float):
        self.timeout_type = timeout_type
        self.duration = duration
        super().__init__(f"{timeout_type} timeout after {duration}s: {message}")

class TimeoutHandler:
    def __init__(self, config: TimeoutConfig):
        self.config = config
    
    @asynccontextmanager
    async def timeout_context(self, 
                             timeout: float, 
                             timeout_type: str = "operation"):
        """Context manager for timeout operations"""
        start_time = time.time()
        try:
            async with asyncio.timeout(timeout):
                yield
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            raise TimeoutError(
                f"Operation timed out",
                timeout_type,
                duration
            )
    
    async def execute_with_timeout(self, 
                                  func: Callable, 
                                  timeout: float,
                                  timeout_type: str = "operation",
                                  *args, **kwargs) -> Any:
        """Execute function with timeout"""
        async with self.timeout_context(timeout, timeout_type):
            return await func(*args, **kwargs)
    
    def create_http_session(self) -> aiohttp.ClientSession:
        """Create HTTP session with comprehensive timeouts"""
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=self.config.connect_timeout,
            sock_read=self.config.read_timeout,
            pool=self.config.pool_timeout
        )
        
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        return aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )

# Usage examples
timeout_handler = TimeoutHandler(TimeoutConfig(
    connect_timeout=3.0,
    request_timeout=15.0,
    read_timeout=5.0
))

async def fetch_with_timeouts(url: str) -> dict:
    """Fetch data with multiple timeout layers"""
    session = timeout_handler.create_http_session()
    
    try:
        # Overall operation timeout
        async with timeout_handler.timeout_context(20.0, "fetch_operation"):
            async with session.get(url) as response:
                response.raise_for_status()
                
                # Specific timeout for JSON parsing
                async with timeout_handler.timeout_context(2.0, "json_parsing"):
                    return await response.json()
    
    finally:
        await session.close()

# Timeout decorator
def timeout(seconds: float, timeout_type: str = "function"):
    """Decorator for function timeouts"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function {func.__name__} timed out",
                    timeout_type,
                    seconds
                )
        return wrapper
    return decorator

@timeout(10.0, "database_query")
async def complex_database_query(query: str):
    """Database query with timeout"""
    # Simulate complex query
    await asyncio.sleep(5)
    return f"Results for: {query}"
```

**Cascading Timeout Pattern:**

```python
class CascadingTimeoutHandler:
    """Handle timeouts that cascade through the call stack"""
    
    def __init__(self):
        self.timeout_stack = []
    
    @asynccontextmanager
    async def cascading_timeout(self, 
                               timeout: float, 
                               operation_name: str):
        """Timeout that respects parent timeouts"""
        
        # Calculate effective timeout based on parent timeouts
        parent_timeout = self.get_remaining_parent_timeout()
        effective_timeout = min(timeout, parent_timeout) if parent_timeout else timeout
        
        timeout_info = {
            'operation': operation_name,
            'requested_timeout': timeout,
            'effective_timeout': effective_timeout,
            'start_time': time.time()
        }
        
        self.timeout_stack.append(timeout_info)
        
        try:
            async with asyncio.timeout(effective_timeout):
                yield timeout_info
        except asyncio.TimeoutError:
            elapsed = time.time() - timeout_info['start_time']
            raise TimeoutError(
                f"Cascading timeout in {operation_name}",
                "cascading",
                elapsed
            )
        finally:
            self.timeout_stack.pop()
    
    def get_remaining_parent_timeout(self) -> Optional[float]:
        """Calculate remaining time from parent timeouts"""
        if not self.timeout_stack:
            return None
        
        current_time = time.time()
        min_remaining = float('inf')
        
        for timeout_info in self.timeout_stack:
            elapsed = current_time - timeout_info['start_time']
            remaining = timeout_info['effective_timeout'] - elapsed
            min_remaining = min(min_remaining, remaining)
        
        return max(0, min_remaining) if min_remaining != float('inf') else None
```

**Adaptive Timeout Pattern:**

```python
import statistics
from collections import deque
from typing import Dict, Deque

class AdaptiveTimeoutHandler:
    """Timeout handler that adapts based on historical performance"""
    
    def __init__(self, 
                 initial_timeout: float = 5.0,
                 min_timeout: float = 1.0,
                 max_timeout: float = 30.0,
                 history_size: int = 100):
        self.initial_timeout = initial_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.history_size = history_size
        
        # Track response times by operation
        self.response_times: Dict[str, Deque[float]] = {}
        self.current_timeouts: Dict[str, float] = {}
    
    def record_response_time(self, operation: str, response_time: float):
        """Record response time for adaptive timeout calculation"""
        if operation not in self.response_times:
            self.response_times[operation] = deque(maxlen=self.history_size)
        
        self.response_times[operation].append(response_time)
        self.update_timeout(operation)
    
    def update_timeout(self, operation: str):
        """Update timeout based on historical data"""
        times = list(self.response_times[operation])
        
        if len(times) < 5:  # Not enough data
            self.current_timeouts[operation] = self.initial_timeout
            return
        
        # Calculate percentile-based timeout (95th percentile + buffer)
        p95 = statistics.quantiles(times, n=20)[18]  # 95th percentile
        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        # Timeout = 95th percentile + 2 standard deviations + 20% buffer
        adaptive_timeout = p95 + (2 * std_dev) + (mean_time * 0.2)
        
        # Apply bounds
        adaptive_timeout = max(self.min_timeout, 
                             min(self.max_timeout, adaptive_timeout))
        
        self.current_timeouts[operation] = adaptive_timeout
    
    def get_timeout(self, operation: str) -> float:
        """Get current timeout for operation"""
        return self.current_timeouts.get(operation, self.initial_timeout)
    
    async def execute_with_adaptive_timeout(self, 
                                          operation: str,
                                          func: Callable,
                                          *args, **kwargs) -> Any:
        """Execute with adaptive timeout"""
        timeout = self.get_timeout(operation)
        start_time = time.time()
        
        try:
            async with asyncio.timeout(timeout):
                result = await func(*args, **kwargs)
                
                # Record successful response time
                response_time = time.time() - start_time
                self.record_response_time(operation, response_time)
                
                return result
        
        except asyncio.TimeoutError:
            # Record timeout as very slow response
            slow_response_time = time.time() - start_time
            self.record_response_time(operation, slow_response_time * 2)
            
            raise TimeoutError(
                f"Adaptive timeout for {operation}",
                "adaptive",
                timeout
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get timeout statistics"""
        stats = {}
        for operation, times in self.response_times.items():
            if times:
                stats[operation] = {
                    'current_timeout': self.get_timeout(operation),
                    'avg_response_time': statistics.mean(times),
                    'p95_response_time': statistics.quantiles(list(times), n=20)[18] if len(times) >= 5 else None,
                    'sample_count': len(times)
                }
        return stats

# Usage example
async def call_service_with_adaptive_timeout(service_name: str, data: dict):
    """Call service with adaptive timeout"""
    
    async def service_call():
        # Simulate variable response times
        base_time = random.uniform(0.5, 2.0)
        if random.random() < 0.1:  # 10% chance of slow response
            base_time *= 5
        
        await asyncio.sleep(base_time)
        return {"service": service_name, "response": data}
    
    return await adaptive_timeout.execute_with_adaptive_timeout(
        f"service_{service_name}",
        service_call
    )

# Test adaptive timeout
async def test_adaptive_timeout():
    """Test adaptive timeout behavior"""
    service_name = "user_service"
    
    # Make multiple calls to build history
    for i in range(20):
        try:
            result = await call_service_with_adaptive_timeout(service_name, {"id": i})
            print(f"Call {i}: Success")
        except TimeoutError as e:
            print(f"Call {i}: Timeout - {e}")
        
        # Print stats every 5 calls
        if (i + 1) % 5 == 0:
            stats = adaptive_timeout.get_stats()
            if f"service_{service_name}" in stats:
                service_stats = stats[f"service_{service_name}"]
                print(f"  Current timeout: {service_stats['current_timeout']:.2f}s")
                print(f"  Avg response: {service_stats['avg_response_time']:.2f}s")
```

**Distributed Timeout Coordination:**
```python
import aioredis
import json
from typing import Dict, Optional

class DistributedTimeoutCoordinator:
    """Coordinate timeouts across multiple services"""
    
    def __init__(self, redis_client: aioredis.Redis, service_name: str):
        self.redis = redis_client
        self.service_name = service_name
    
    async def start_distributed_operation(self, 
                                        operation_id: str, 
                                        total_timeout: float,
                                        participants: list) -> Dict:
        """Start a distributed operation with timeout coordination"""
        
        operation_data = {
            'operation_id': operation_id,
            'coordinator': self.service_name,
            'total_timeout': total_timeout,
            'start_time': time.time(),
            'participants': participants,
            'status': 'running'
        }
        
        # Store operation data in Redis
        await self.redis.setex(
            f"distributed_op:{operation_id}",
            int(total_timeout + 10),  # Extra time for cleanup
            json.dumps(operation_data)
        )
        
        # Set timeout alarm
        await self.redis.setex(
            f"timeout_alarm:{operation_id}",
            int(total_timeout),
            json.dumps({
                'operation_id': operation_id,
                'timeout_at': time.time() + total_timeout
            })
        )
        
        return operation_data
    
    async def check_operation_timeout(self, operation_id: str) -> Optional[Dict]:
        """Check if operation has timed out"""
        alarm_data = await self.redis.get(f"timeout_alarm:{operation_id}")
        
        if not alarm_data:
            # Timeout alarm expired - operation timed out
            op_data = await self.redis.get(f"distributed_op:{operation_id}")
            if op_data:
                operation = json.loads(op_data)
                if operation['status'] == 'running':
                    return {
                        'timed_out': True,
                        'operation': operation
                    }
        
        return {'timed_out': False}
    
    async def complete_operation(self, operation_id: str, success: bool = True):
        """Mark operation as completed"""
        op_data = await self.redis.get(f"distributed_op:{operation_id}")
        
        if op_data:
            operation = json.loads(op_data)
            operation['status'] = 'completed' if success else 'failed'
            operation['end_time'] = time.time()
            
            await self.redis.setex(
                f"distributed_op:{operation_id}",
                3600,  # Keep for 1 hour for debugging
                json.dumps(operation)
            )
            
            # Remove timeout alarm
            await self.redis.delete(f"timeout_alarm:{operation_id}")
    
    @asynccontextmanager
    async def distributed_timeout_context(self, 
                                        operation_id: str, 
                                        timeout: float,
                                        participants: list):
        """Context manager for distributed timeout operations"""
        
        operation = await self.start_distributed_operation(
            operation_id, timeout, participants
        )
        
        try:
            # Check for timeout periodically
            async def timeout_checker():
                while True:
                    await asyncio.sleep(1)
                    timeout_check = await self.check_operation_timeout(operation_id)
                    if timeout_check['timed_out']:
                        raise TimeoutError(
                            f"Distributed operation {operation_id} timed out",
                            "distributed",
                            timeout
                        )
            
            # Start timeout checker
            checker_task = asyncio.create_task(timeout_checker())
            
            try:
                yield operation
            finally:
                checker_task.cancel()
                try:
                    await checker_task
                except asyncio.CancelledError:
                    pass
            
            await self.complete_operation(operation_id, success=True)
            
        except Exception as e:
            await self.complete_operation(operation_id, success=False)
            raise e

# Usage example
async def distributed_microservice_operation():
    """Example of distributed operation with timeout coordination"""
    redis_client = aioredis.from_url("redis://localhost")
    coordinator = DistributedTimeoutCoordinator(redis_client, "order-service")
    
    operation_id = f"order_process_{int(time.time())}"
    participants = ["inventory-service", "payment-service", "shipping-service"]
    
    async with coordinator.distributed_timeout_context(
        operation_id, 30.0, participants
    ):
        # Simulate distributed operation
        print(f"Starting distributed operation {operation_id}")
        
        # Call multiple services
        await asyncio.gather(
            call_inventory_service(),
            call_payment_service(),
            call_shipping_service()
        )
        
        print(f"Distributed operation {operation_id} completed successfully")

async def call_inventory_service():
    await asyncio.sleep(2)
    return {"inventory": "reserved"}

async def call_payment_service():
    await asyncio.sleep(3)
    return {"payment": "processed"}

async def call_shipping_service():
    await asyncio.sleep(1)
    return {"shipping": "scheduled"}
```

## Learning Objectives

By the end of this section, you should be able to:

### Core Understanding
- **Explain the purpose and implementation** of each resilience pattern (Circuit Breaker, Bulkhead, Rate Limiting, Retry, Timeout)
- **Identify when to apply** specific resilience patterns based on failure scenarios
- **Understand the trade-offs** between different resilience strategies
- **Design resilient microservice architectures** that handle cascade failures gracefully

### Practical Implementation
- **Implement circuit breakers** with proper state management and failure detection
- **Design effective bulkheads** for resource isolation and failure containment
- **Configure rate limiting** algorithms for different traffic patterns and requirements
- **Build retry mechanisms** with exponential backoff and jitter to handle transient failures
- **Set appropriate timeouts** at different levels (connection, request, operation)
- **Combine multiple patterns** into a cohesive resilience strategy

### Operational Excellence
- **Monitor resilience patterns** with proper metrics and alerting
- **Debug resilience issues** using logs, metrics, and distributed tracing
- **Test resilience patterns** through chaos engineering and failure injection
- **Optimize performance** while maintaining resilience guarantees

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Explain how a circuit breaker prevents cascade failures  
□ Design bulkheads for different types of resources (threads, connections, CPU)  
□ Implement token bucket and sliding window rate limiters  
□ Configure retry strategies with proper backoff algorithms  
□ Set cascading timeouts that respect parent operation timeouts  
□ Combine all patterns in a resilient service client  
□ Monitor and alert on resilience pattern metrics  
□ Test resilience patterns using failure injection  

### Practical Exercises

**Exercise 1: Circuit Breaker Implementation**
```python
# TODO: Implement a circuit breaker that:
# 1. Tracks failure rate over a sliding window
# 2. Opens when failure rate exceeds 50% over 1 minute
# 3. Allows one test request in half-open state
# 4. Provides detailed metrics and state information

class AdvancedCircuitBreaker:
    def __init__(self, failure_rate_threshold=0.5, window_size=60):
        # Your implementation here
        pass
```

**Exercise 2: Multi-Level Rate Limiting**
```python
# TODO: Implement rate limiting with:
# 1. Per-user limits (100 requests/minute)
# 2. Per-endpoint limits (1000 requests/minute)
# 3. Global limits (10000 requests/minute)
# 4. Different limits for different user tiers

class MultiLevelRateLimiter:
    def __init__(self):
        # Your implementation here
        pass
    
    def is_allowed(self, user_id, endpoint, user_tier="basic"):
        # Your implementation here
        pass
```

**Exercise 3: Resilient Service Integration**
```python
# TODO: Create a resilient client that integrates:
# 1. Circuit breaker for failure detection
# 2. Retry with exponential backoff
# 3. Bulkhead for resource isolation
# 4. Rate limiting for traffic control
# 5. Comprehensive timeout handling
# 6. Fallback mechanisms for graceful degradation

class ResilientServiceClient:
    def __init__(self, config):
        # Your implementation here
        pass
```

## Study Materials

### Primary Resources
- **Books:**
  - "Microservices Patterns" by Chris Richardson - Chapters on Resilience
  - "Building Microservices" by Sam Newman - Resilience and Monitoring chapters
  - "Release It!" by Michael Nygard - Stability patterns

- **Online Documentation:**
  - [Netflix Hystrix Documentation](https://github.com/Netflix/Hystrix/wiki) - Circuit Breaker patterns
  - [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/) - Reliability pillar
  - [Google SRE Book](https://sre.google/sre-book/table-of-contents/) - Chapters on handling overload

### Video Resources
- "Microservices Resilience Patterns" - Microservices.io
- "Building Resilient Systems" - AWS re:Invent sessions
- "Chaos Engineering at Netflix" - Netflix Tech Blog videos

### Hands-on Labs

**Lab 1: Circuit Breaker Testing**
- Set up a simple HTTP service
- Implement circuit breaker protection
- Use fault injection to trigger circuit breaker states
- Monitor state transitions and recovery

**Lab 2: Rate Limiting Strategies**
- Implement different rate limiting algorithms
- Test with various traffic patterns
- Compare performance and accuracy
- Handle distributed rate limiting scenarios

**Lab 3: End-to-End Resilience**
- Build a multi-service application
- Implement all resilience patterns
- Perform chaos engineering experiments
- Monitor and optimize resilience behavior

### Tools and Libraries

**Python Libraries:**
```bash
pip install aiohttp asyncio aioredis
pip install pybreaker circuitbreaker
pip install tenacity retrying
pip install prometheus-client
```

**Monitoring Tools:**
- Prometheus + Grafana for metrics
- Jaeger for distributed tracing  
- ELK Stack for log analysis
- Chaos Monkey for failure injection

**Testing Tools:**
- Locust for load testing
- Pumba for container chaos testing
- Gremlin for infrastructure chaos testing

### Practice Questions

**Conceptual Questions:**
1. When would you choose bulkhead isolation over circuit breaker protection?
2. How do you prevent the "thundering herd" problem in retry mechanisms?
3. What are the trade-offs between different rate limiting algorithms?
4. How do cascading timeouts work in a multi-service call chain?
5. What metrics should you monitor for each resilience pattern?

**Scenario-Based Questions:**
6. Design resilience patterns for a payment processing service
7. How would you handle resilience in a read-heavy social media feed service?
8. What patterns would you use for a real-time chat application?
9. Design failure handling for a distributed file upload service
10. How would you implement resilience in a microservice mesh?

**Implementation Questions:**
```python
# Question 11: Fix the race condition in this circuit breaker
class BuggyCircuitBreaker:
    def __init__(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def call(self, func):
        if self.state == "OPEN":
            raise Exception("Circuit breaker open")
        
        try:
            result = func()
            self.failure_count = 0  # Race condition here!
            return result
        except Exception:
            self.failure_count += 1  # And here!
            if self.failure_count > 5:
                self.state = "OPEN"
            raise

# Question 12: Implement distributed rate limiting
# How would you implement rate limiting across multiple instances?

# Question 13: Design timeout strategy
# Design timeout configuration for this service call chain:
# API Gateway -> Order Service -> Payment Service -> Bank API
```

### Development Environment Setup

**Docker Compose for Testing:**
```yaml
version: '3.8'
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**Sample Prometheus Configuration:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'resilience-patterns'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```
