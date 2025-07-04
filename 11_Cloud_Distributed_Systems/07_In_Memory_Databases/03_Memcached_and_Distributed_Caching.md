# Memcached and Distributed Caching

*Duration: 2 weeks*

## Introduction

Memcached is a high-performance, distributed memory object caching system designed to speed up dynamic web applications by reducing database load. It's widely used by companies like Facebook, Twitter, and Wikipedia to handle millions of requests per second.

This comprehensive guide covers Memcached architecture, distributed caching strategies, and real-world implementation patterns for building scalable systems.

## Learning Objectives

By the end of this section, you should be able to:
- **Understand Memcached architecture** and internal mechanisms
- **Implement distributed caching strategies** using consistent hashing
- **Design cache invalidation patterns** for different use cases
- **Build fault-tolerant caching systems** with proper error handling
- **Optimize cache performance** and monitor cache metrics
- **Compare different caching solutions** and choose appropriate ones

## Memcached Architecture and Fundamentals

### What is Memcached?

Memcached is a simple, yet powerful distributed caching system that stores data as key-value pairs in RAM. It operates on a client-server model where:

- **Clients** (applications) store and retrieve data
- **Servers** (memcached instances) store data in memory
- **Protocol** is simple text-based or binary

### Core Architecture

```
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Client A   │    │  Client B   │    │  Client C   │
    │             │    │             │    │             │
    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │              Memcached Client Library              │
    │    (Handles hashing, server selection, etc.)       │
    └─────────────────────────┼─────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
    ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
    │ Memcached   │    │ Memcached   │    │ Memcached   │
    │ Server 1    │    │ Server 2    │    │ Server 3    │
    │ Port: 11211 │    │ Port: 11211 │    │ Port: 11211 │
    └─────────────┘    └─────────────┘    └─────────────┘
```

### Memory Management

Memcached uses a **slab allocator** for efficient memory management:

```
Memory Structure:
┌─────────────────────────────────────────────────────┐
│                 Memcached Memory                    │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Slab 1    │  │   Slab 2    │  │   Slab 3    │  │
│  │ (64 bytes)  │  │ (128 bytes) │  │ (256 bytes) │  │
│  │             │  │             │  │             │  │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │
│  │ │ Chunk 1 │ │  │ │ Chunk 1 │ │  │ │ Chunk 1 │ │  │
│  │ ├─────────┤ │  │ ├─────────┤ │  │ ├─────────┤ │  │
│  │ │ Chunk 2 │ │  │ │ Chunk 2 │ │  │ │ Chunk 2 │ │  │
│  │ ├─────────┤ │  │ ├─────────┤ │  │ ├─────────┤ │  │
│  │ │   ...   │ │  │ │   ...   │ │  │ │   ...   │ │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Installation and Setup

**Installing Memcached:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install memcached

# CentOS/RHEL
sudo yum install memcached

# macOS
brew install memcached

# Start Memcached
memcached -d -m 64 -p 11211 -u memcached
# -d: daemon mode
# -m: memory limit (64MB)
# -p: port
# -u: user
```

**Configuration File Example:**
```bash
# /etc/memcached.conf
-d                     # Run as daemon
-m 64                  # Memory limit in MB
-p 11211              # Port
-u memcached          # User
-l 127.0.0.1          # Listen address
-P /var/run/memcached/memcached.pid  # PID file
-vv                   # Very verbose logging
```

## Performance Optimization and Monitoring

### Key Performance Metrics

#### 1. Cache Hit Ratio Monitoring

```python
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import threading

class CacheMetricsCollector:
    def __init__(self, window_size: int = 3600):  # 1 hour window
        self.window_size = window_size
        self.operations = deque()
        self.lock = threading.Lock()
        
        # Metrics storage
        self.hit_count = 0
        self.miss_count = 0
        self.error_count = 0
        self.response_times = deque()
        
    def record_hit(self, response_time: float):
        """Record a cache hit"""
        with self.lock:
            timestamp = time.time()
            self.operations.append(('hit', timestamp, response_time))
            self.hit_count += 1
            self.response_times.append(response_time)
            self._cleanup_old_data(timestamp)
    
    def record_miss(self, response_time: float):
        """Record a cache miss"""
        with self.lock:
            timestamp = time.time()
            self.operations.append(('miss', timestamp, response_time))
            self.miss_count += 1
            self.response_times.append(response_time)
            self._cleanup_old_data(timestamp)
    
    def record_error(self, response_time: float):
        """Record a cache error"""
        with self.lock:
            timestamp = time.time()
            self.operations.append(('error', timestamp, response_time))
            self.error_count += 1
            self._cleanup_old_data(timestamp)
    
    def _cleanup_old_data(self, current_time: float):
        """Remove data outside the time window"""
        cutoff = current_time - self.window_size
        
        while self.operations and self.operations[0][1] < cutoff:
            op_type, _, _ = self.operations.popleft()
            if op_type == 'hit':
                self.hit_count -= 1
            elif op_type == 'miss':
                self.miss_count -= 1
            elif op_type == 'error':
                self.error_count -= 1
        
        # Cleanup response times
        while self.response_times and len(self.response_times) > 10000:
            self.response_times.popleft()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        with self.lock:
            total_ops = self.hit_count + self.miss_count + self.error_count
            
            if total_ops == 0:
                return {
                    'hit_ratio': 0.0,
                    'miss_ratio': 0.0,
                    'error_ratio': 0.0,
                    'avg_response_time': 0.0,
                    'p95_response_time': 0.0,
                    'operations_per_second': 0.0
                }
            
            # Calculate percentiles
            sorted_times = sorted(list(self.response_times))
            p95_index = int(0.95 * len(sorted_times)) if sorted_times else 0
            
            return {
                'hit_ratio': self.hit_count / total_ops,
                'miss_ratio': self.miss_count / total_ops,
                'error_ratio': self.error_count / total_ops,
                'avg_response_time': sum(sorted_times) / len(sorted_times) if sorted_times else 0,
                'p95_response_time': sorted_times[p95_index] if sorted_times else 0,
                'operations_per_second': total_ops / self.window_size,
                'total_operations': total_ops
            }

class MonitoredMemcachedClient:
    def __init__(self, servers: List[str]):
        self.client = memcache.Client(servers)
        self.metrics = CacheMetricsCollector()
    
    def get(self, key: str) -> Any:
        """Get with metrics collection"""
        start_time = time.time()
        
        try:
            value = self.client.get(key)
            response_time = time.time() - start_time
            
            if value is not None:
                self.metrics.record_hit(response_time)
            else:
                self.metrics.record_miss(response_time)
            
            return value
            
        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.record_error(response_time)
            raise e
    
    def set(self, key: str, value: Any, expiry: int = 3600) -> bool:
        """Set with error handling"""
        try:
            return self.client.set(key, value, time=expiry)
        except Exception as e:
            print(f"Cache set error for {key}: {e}")
            return False
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        metrics = self.metrics.get_metrics()
        
        return f"""
=== Cache Performance Report ===
Hit Ratio: {metrics['hit_ratio']:.2%}
Miss Ratio: {metrics['miss_ratio']:.2%}
Error Ratio: {metrics['error_ratio']:.2%}
Average Response Time: {metrics['avg_response_time']*1000:.2f}ms
95th Percentile Response Time: {metrics['p95_response_time']*1000:.2f}ms
Operations per Second: {metrics['operations_per_second']:.1f}
Total Operations: {metrics['total_operations']}
"""
```

#### 2. Memory and Connection Monitoring

```python
class MemcachedClusterMonitor:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.clients = {server: memcache.Client([server]) for server in servers}
    
    def get_cluster_health(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive cluster health information"""
        cluster_health = {}
        
        for server in self.servers:
            try:
                client = self.clients[server]
                stats = client.get_stats()[0][1]  # Get stats dictionary
                
                # Calculate derived metrics
                hit_ratio = float(stats.get('get_hits', 0)) / max(float(stats.get('cmd_get', 1)), 1)
                memory_usage = float(stats.get('bytes', 0)) / float(stats.get('limit_maxbytes', 1))
                connection_usage = float(stats.get('curr_connections', 0)) / float(stats.get('max_connections', 1024))
                
                cluster_health[server] = {
                    'status': 'healthy',
                    'uptime': int(stats.get('uptime', 0)),
                    'version': stats.get('version', 'unknown'),
                    
                    # Memory metrics
                    'memory_used_bytes': int(stats.get('bytes', 0)),
                    'memory_limit_bytes': int(stats.get('limit_maxbytes', 0)),
                    'memory_usage_percent': memory_usage * 100,
                    
                    # Connection metrics
                    'current_connections': int(stats.get('curr_connections', 0)),
                    'max_connections': int(stats.get('max_connections', 1024)),
                    'connection_usage_percent': connection_usage * 100,
                    'total_connections': int(stats.get('total_connections', 0)),
                    
                    # Cache metrics
                    'current_items': int(stats.get('curr_items', 0)),
                    'total_items': int(stats.get('total_items', 0)),
                    'cmd_get': int(stats.get('cmd_get', 0)),
                    'cmd_set': int(stats.get('cmd_set', 0)),
                    'get_hits': int(stats.get('get_hits', 0)),
                    'get_misses': int(stats.get('get_misses', 0)),
                    'hit_ratio_percent': hit_ratio * 100,
                    
                    # Network metrics
                    'bytes_read': int(stats.get('bytes_read', 0)),
                    'bytes_written': int(stats.get('bytes_written', 0)),
                    
                    # Eviction metrics
                    'evictions': int(stats.get('evictions', 0)),
                    'reclaimed': int(stats.get('reclaimed', 0))
                }
                
            except Exception as e:
                cluster_health[server] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return cluster_health
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get cluster-wide summary metrics"""
        health_data = self.get_cluster_health()
        healthy_servers = [s for s, data in health_data.items() if data.get('status') == 'healthy']
        
        if not healthy_servers:
            return {'status': 'critical', 'healthy_servers': 0, 'total_servers': len(self.servers)}
        
        # Aggregate metrics
        total_memory_used = sum(health_data[s].get('memory_used_bytes', 0) for s in healthy_servers)
        total_memory_limit = sum(health_data[s].get('memory_limit_bytes', 0) for s in healthy_servers)
        total_items = sum(health_data[s].get('current_items', 0) for s in healthy_servers)
        total_gets = sum(health_data[s].get('cmd_get', 0) for s in healthy_servers)
        total_hits = sum(health_data[s].get('get_hits', 0) for s in healthy_servers)
        total_connections = sum(health_data[s].get('current_connections', 0) for s in healthy_servers)
        
        cluster_hit_ratio = total_hits / max(total_gets, 1)
        cluster_memory_usage = total_memory_used / max(total_memory_limit, 1)
        
        # Determine cluster status
        if len(healthy_servers) < len(self.servers) * 0.5:
            status = 'critical'
        elif cluster_memory_usage > 0.9 or cluster_hit_ratio < 0.5:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'healthy_servers': len(healthy_servers),
            'total_servers': len(self.servers),
            'cluster_hit_ratio_percent': cluster_hit_ratio * 100,
            'cluster_memory_usage_percent': cluster_memory_usage * 100,
            'total_items': total_items,
            'total_connections': total_connections,
            'total_memory_used_mb': total_memory_used / (1024 * 1024),
            'total_memory_limit_mb': total_memory_limit / (1024 * 1024)
        }
    
    def generate_health_report(self) -> str:
        """Generate human-readable health report"""
        cluster_summary = self.get_cluster_summary()
        health_data = self.get_cluster_health()
        
        report = f"""
=== Memcached Cluster Health Report ===
Overall Status: {cluster_summary['status'].upper()}
Healthy Servers: {cluster_summary['healthy_servers']}/{cluster_summary['total_servers']}
Cluster Hit Ratio: {cluster_summary['cluster_hit_ratio_percent']:.1f}%
Cluster Memory Usage: {cluster_summary['cluster_memory_usage_percent']:.1f}%
Total Items: {cluster_summary['total_items']:,}
Total Connections: {cluster_summary['total_connections']}

=== Server Details ===
"""
        
        for server, data in health_data.items():
            if data.get('status') == 'healthy':
                report += f"""
Server: {server}
  Status: ✓ HEALTHY
  Uptime: {data['uptime']} seconds
  Memory: {data['memory_usage_percent']:.1f}% ({data['memory_used_bytes']/1024/1024:.1f}MB / {data['memory_limit_bytes']/1024/1024:.1f}MB)
  Items: {data['current_items']:,}
  Hit Ratio: {data['hit_ratio_percent']:.1f}%
  Connections: {data['current_connections']}/{data['max_connections']} ({data['connection_usage_percent']:.1f}%)
  Evictions: {data['evictions']}
"""
            else:
                report += f"""
Server: {server}
  Status: ✗ UNHEALTHY
  Error: {data.get('error', 'Unknown error')}
"""
        
        return report

# Usage example
def monitor_cluster():
    servers = ['10.0.1.10:11211', '10.0.1.11:11211', '10.0.1.12:11211']
    monitor = MemcachedClusterMonitor(servers)
    
    # Monitor performance
    monitored_client = MonitoredMemcachedClient(servers)
    
    # Simulate some operations
    for i in range(1000):
        key = f"test_key_{i % 100}"  # Some cache hits, some misses
        monitored_client.get(key)
        if i % 10 == 0:
            monitored_client.set(key, f"value_{i}")
    
    # Print reports
    print(monitored_client.get_performance_report())
    print(monitor.generate_health_report())
```

### Optimization Strategies

#### 1. Connection Pooling and Multiplexing

```python
import threading
from queue import Queue, Empty
from contextlib import contextmanager

class MemcachedConnectionPool:
    def __init__(self, servers: List[str], pool_size: int = 20, 
                 max_overflow: int = 10, timeout: int = 30):
        self.servers = servers
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        
        self.pool = Queue(maxsize=pool_size)
        self.overflow_connections = set()
        self.lock = threading.Lock()
        
        # Pre-populate pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self.pool.put(conn)
    
    def _create_connection(self):
        """Create a new memcached connection"""
        return memcache.Client(self.servers, 
                             socket_timeout=self.timeout,
                             server_max_key_length=250,
                             server_max_value_length=1024*1024)
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager"""
        conn = None
        is_overflow = False
        
        try:
            # Try to get from pool
            try:
                conn = self.pool.get_nowait()
            except Empty:
                # Pool is empty, check if we can create overflow connection
                with self.lock:
                    if len(self.overflow_connections) < self.max_overflow:
                        conn = self._create_connection()
                        self.overflow_connections.add(conn)
                        is_overflow = True
                    else:
                        # Wait for connection to become available
                        conn = self.pool.get(timeout=self.timeout)
            
            yield conn
            
        except Exception as e:
            # Connection error, create new one
            conn = self._create_connection()
            is_overflow = True
            yield conn
            
        finally:
            if conn:
                if is_overflow:
                    with self.lock:
                        self.overflow_connections.discard(conn)
                else:
                    # Return to pool
                    try:
                        self.pool.put_nowait(conn)
                    except:
                        # Pool is full, discard connection
                        pass

class OptimizedMemcachedClient:
    def __init__(self, servers: List[str], pool_size: int = 20):
        self.pool = MemcachedConnectionPool(servers, pool_size)
        self.serialization_cache = {}  # Cache serialized objects
    
    def get(self, key: str) -> Any:
        """Optimized get with connection pooling"""
        with self.pool.get_connection() as conn:
            return conn.get(key)
    
    def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get multiple keys efficiently"""
        with self.pool.get_connection() as conn:
            return conn.get_multi(keys)
    
    def set_multi(self, mapping: Dict[str, Any], time: int = 3600) -> List[str]:
        """Batch set multiple keys efficiently"""
        with self.pool.get_connection() as conn:
            return conn.set_multi(mapping, time=time)
    
    def delete_multi(self, keys: List[str]) -> bool:
        """Batch delete multiple keys"""
        with self.pool.get_connection() as conn:
            return conn.delete_multi(keys)
```

#### 2. Compression and Serialization Optimization

```python
import pickle
import gzip
import json
from typing import Any, Tuple

class CompressedMemcachedClient:
    def __init__(self, servers: List[str], compression_threshold: int = 1024):
        self.client = memcache.Client(servers)
        self.compression_threshold = compression_threshold
    
    def _serialize_and_compress(self, value: Any) -> Tuple[bytes, int]:
        """Serialize and optionally compress value"""
        # Serialize
        if isinstance(value, (str, int, float, bool)):
            # Simple types - use JSON
            serialized = json.dumps(value).encode('utf-8')
            flags = 1  # JSON flag
        else:
            # Complex types - use pickle
            serialized = pickle.dumps(value)
            flags = 2  # Pickle flag
        
        # Compress if large enough
        if len(serialized) > self.compression_threshold:
            compressed = gzip.compress(serialized)
            if len(compressed) < len(serialized):  # Only use if actually smaller
                return compressed, flags | 4  # Add compression flag
        
        return serialized, flags
    
    def _decompress_and_deserialize(self, data: bytes, flags: int) -> Any:
        """Decompress and deserialize value"""
        # Decompress if compressed
        if flags & 4:  # Compression flag
            data = gzip.decompress(data)
        
        # Deserialize
        if flags & 1:  # JSON flag
            return json.loads(data.decode('utf-8'))
        elif flags & 2:  # Pickle flag
            return pickle.loads(data)
        else:
            return data
    
    def set(self, key: str, value: Any, time: int = 3600) -> bool:
        """Set with compression"""
        data, flags = self._serialize_and_compress(value)
        return self.client.set(key, data, time=time)
    
    def get(self, key: str) -> Any:
        """Get with decompression"""
        result = self.client.get(key)
        if result is None:
            return None
        
        # In real implementation, flags would be stored with data
        # For demo, assume pickle + compression
        try:
            return self._decompress_and_deserialize(result, 6)  # Pickle + compression
        except:
            return result  # Fallback to raw data
```

### Advanced Caching Patterns

#### 1. Cache Warming

```python
import concurrent.futures
import time
from typing import Callable, List, Dict, Any

class CacheWarmer:
    def __init__(self, cache_client, max_workers: int = 10):
        self.cache = cache_client
        self.max_workers = max_workers
    
    def warm_cache_batch(self, keys_and_loaders: List[Tuple[str, Callable[[], Any]]], 
                        ttl: int = 3600) -> Dict[str, bool]:
        """Warm cache with multiple keys in parallel"""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_key = {
                executor.submit(self._warm_single_key, key, loader, ttl): key
                for key, loader in keys_and_loaders
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    print(f"Failed to warm cache for {key}: {e}")
                    results[key] = False
        
        return results
    
    def _warm_single_key(self, key: str, loader: Callable[[], Any], ttl: int) -> bool:
        """Warm a single cache key"""
        try:
            # Check if already cached
            if self.cache.get(key) is not None:
                return True
            
            # Load data
            data = loader()
            if data is not None:
                return self.cache.set(key, data, expiry=ttl)
            
        except Exception as e:
            print(f"Error warming cache for {key}: {e}")
        
        return False
    
    def warm_user_data(self, user_ids: List[int]) -> Dict[str, bool]:
        """Example: Warm user profile cache"""
        keys_and_loaders = [
            (f"user_profile:{user_id}", lambda uid=user_id: self._load_user_profile(uid))
            for user_id in user_ids
        ]
        
        return self.warm_cache_batch(keys_and_loaders, ttl=3600)
    
    def _load_user_profile(self, user_id: int) -> dict:
        """Simulate loading user profile from database"""
        time.sleep(0.1)  # Simulate DB latency
        return {
            'id': user_id,
            'name': f'User {user_id}',
            'email': f'user{user_id}@example.com'
        }
```

#### 2. Circuit Breaker Pattern

```python
import time
import threading
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker tripped
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise e
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class FaultTolerantCache:
    def __init__(self, cache_client, fallback_func: Callable = None):
        self.cache = cache_client
        self.fallback_func = fallback_func
        self.circuit_breaker = CircuitBreaker()
    
    def get(self, key: str) -> Any:
        """Get with circuit breaker and fallback"""
        try:
            return self.circuit_breaker.call(self.cache.get, key)
        except Exception as e:
            print(f"Cache get failed for {key}: {e}")
            
            if self.fallback_func:
                return self.fallback_func(key)
            
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set with circuit breaker"""
        try:
            return self.circuit_breaker.call(self.cache.set, key, value, expiry=ttl)
        except Exception as e:
            print(f"Cache set failed for {key}: {e}")
            return False
```

## Memcached Commands and Operations

### Basic Commands

Memcached supports both **text protocol** and **binary protocol**. Here are the fundamental operations:

#### Storage Commands

**SET Command:**
```bash
# Telnet to memcached
telnet localhost 11211

# Set a key-value pair
set mykey 0 3600 5
hello
STORED

# Syntax: set <key> <flags> <exptime> <bytes> [noreply]
# flags: 32-bit unsigned integer (client-specific)
# exptime: expiration time in seconds (0 = never expire)
# bytes: number of bytes in the data block
```

**ADD Command (only if key doesn't exist):**
```bash
add newkey 0 3600 4
test
STORED

add newkey 0 3600 4
fail
NOT_STORED
```

**REPLACE Command (only if key exists):**
```bash
replace mykey 0 3600 7
updated
STORED
```

#### Retrieval Commands

**GET Command:**
```bash
get mykey
VALUE mykey 0 7
updated
END

# Multiple keys
get key1 key2 key3
VALUE key1 0 5
value1
VALUE key2 0 5
value2
END
```

**GETS Command (with CAS value):**
```bash
gets mykey
VALUE mykey 0 7 123456  # 123456 is CAS value
updated
END
```

#### Update Commands

**INCR/DECR Commands:**
```bash
set counter 0 0 1
5
STORED

incr counter 3
8

decr counter 2
6
```

**CAS (Compare And Swap):**
```bash
cas mykey 0 3600 8 123456
modified
STORED
```

#### Deletion Commands

**DELETE Command:**
```bash
delete mykey
DELETED

delete nonexistent
NOT_FOUND
```

**FLUSH_ALL Command:**
```bash
flush_all
OK

# Flush with delay
flush_all 60  # Flush after 60 seconds
OK
```

### Python Client Implementation

**Basic Operations:**
```python
import memcache
import time
import json
from typing import Any, Optional, List, Dict

class MemcachedClient:
    def __init__(self, servers: List[str], debug: bool = False):
        """
        Initialize Memcached client
        
        Args:
            servers: List of memcached server addresses
            debug: Enable debug mode
        """
        self.mc = memcache.Client(servers, debug=debug)
        self.default_expiry = 3600  # 1 hour
    
    def set(self, key: str, value: Any, expiry: int = None) -> bool:
        """Set a key-value pair with optional expiry"""
        if expiry is None:
            expiry = self.default_expiry
            
        # Serialize complex objects
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        return self.mc.set(key, value, time=expiry)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key with default fallback"""
        value = self.mc.get(key)
        
        if value is None:
            return default
            
        # Try to deserialize JSON
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        
        return value
    
    def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys at once for better performance"""
        result = self.mc.get_multi(keys)
        
        # Deserialize JSON values
        for key, value in result.items():
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return result
    
    def add(self, key: str, value: Any, expiry: int = None) -> bool:
        """Add key only if it doesn't exist"""
        if expiry is None:
            expiry = self.default_expiry
            
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
            
        return self.mc.add(key, value, time=expiry)
    
    def replace(self, key: str, value: Any, expiry: int = None) -> bool:
        """Replace key only if it exists"""
        if expiry is None:
            expiry = self.default_expiry
            
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
            
        return self.mc.replace(key, value, time=expiry)
    
    def delete(self, key: str) -> bool:
        """Delete a key"""
        return self.mc.delete(key)
    
    def incr(self, key: str, delta: int = 1) -> Optional[int]:
        """Increment a numeric value"""
        return self.mc.incr(key, delta)
    
    def decr(self, key: str, delta: int = 1) -> Optional[int]:
        """Decrement a numeric value"""
        return self.mc.decr(key, delta)
    
    def flush_all(self) -> bool:
        """Clear all cached data"""
        return self.mc.flush_all()
    
    def get_stats(self) -> Dict[str, Dict[str, str]]:
        """Get server statistics"""
        return self.mc.get_stats()

# Usage Examples
if __name__ == "__main__":
    # Initialize client
    cache = MemcachedClient(['127.0.0.1:11211'])
    
    # Basic operations
    cache.set('user:1001', {'name': 'John', 'age': 30})
    user = cache.get('user:1001')
    print(f"User: {user}")
    
    # Counters
    cache.set('page_views', '1000')
    new_views = cache.incr('page_views', 1)
    print(f"Page views: {new_views}")
    
    # Multiple keys
    cache.set('key1', 'value1')
    cache.set('key2', 'value2')
    results = cache.get_multi(['key1', 'key2', 'key3'])
    print(f"Multiple results: {results}")
    
    # Statistics
    stats = cache.get_stats()
    for server, server_stats in stats.items():
        print(f"Server {server}:")
        print(f"  Current items: {server_stats.get('curr_items', 'N/A')}")
        print(f"  Total connections: {server_stats.get('total_connections', 'N/A')}")
        print(f"  Hit ratio: {float(server_stats.get('get_hits', 0)) / max(float(server_stats.get('cmd_get', 1)), 1):.2%}")
```

### Advanced Operations

**Atomic Operations with CAS:**
```python
import memcache
import time

def atomic_increment_with_cas(mc, key: str, increment: int = 1) -> bool:
    """
    Atomically increment a value using Compare-And-Swap
    """
    max_retries = 10
    
    for _ in range(max_retries):
        # Get current value with CAS token
        result = mc.gets(key)
        if result is None:
            # Key doesn't exist, try to add it
            if mc.add(key, str(increment)):
                return True
            continue
        
        try:
            current_value = int(result)
            new_value = current_value + increment
            
            # Attempt CAS operation
            if mc.cas(key, str(new_value)):
                return True
            # CAS failed, retry
            
        except (ValueError, TypeError):
            return False
        
        # Small delay before retry
        time.sleep(0.001)
    
    return False

# Usage
mc = memcache.Client(['127.0.0.1:11211'])
mc.set('atomic_counter', '100')

# Multiple threads can safely increment this counter
success = atomic_increment_with_cas(mc, 'atomic_counter', 5)
print(f"Atomic increment successful: {success}")
```

**Connection Pooling:**
```python
import memcache
import threading
from contextlib import contextmanager

class MemcachedPool:
    def __init__(self, servers, pool_size=10):
        self.servers = servers
        self.pool = []
        self.pool_lock = threading.Lock()
        
        # Pre-create connections
        for _ in range(pool_size):
            client = memcache.Client(servers)
            self.pool.append(client)
    
    @contextmanager
    def get_client(self):
        """Context manager for getting a client from pool"""
        client = None
        try:
            with self.pool_lock:
                if self.pool:
                    client = self.pool.pop()
                else:
                    # Pool exhausted, create new client
                    client = memcache.Client(self.servers)
            
            yield client
            
        finally:
            if client:
                with self.pool_lock:
                    self.pool.append(client)

# Usage
pool = MemcachedPool(['127.0.0.1:11211'], pool_size=5)

def worker_thread():
    with pool.get_client() as client:
        client.set('thread_data', f'Data from {threading.current_thread().name}')
        result = client.get('thread_data')
        print(f"Thread {threading.current_thread().name}: {result}")

# Create multiple threads
threads = []
for i in range(10):
    thread = threading.Thread(target=worker_thread, name=f'Worker-{i}')
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```

## Consistent Hashing and Sharding

### The Problem with Simple Hashing

When distributing data across multiple Memcached servers, a naive approach might use simple modulo hashing:

```python
# Problematic approach
def simple_hash(key: str, num_servers: int) -> int:
    return hash(key) % num_servers

servers = ['server1:11211', 'server2:11211', 'server3:11211']
server_index = simple_hash('user:1001', len(servers))
target_server = servers[server_index]
```

**Problems with Simple Hashing:**
1. **Adding/removing servers** causes massive cache invalidation
2. **Poor distribution** when servers are added/removed
3. **Hotspots** can develop on certain servers

### Consistent Hashing Solution

Consistent hashing solves these problems by:
- Minimizing redistribution when servers change
- Providing better load distribution
- Ensuring fault tolerance

#### How Consistent Hashing Works

```
Hash Ring (0 to 2^32-1):
                    0/2^32
                      |
              ┌───────┴───────┐
        S3  ←─┤               ├─→  S1
              │               │
              │               │
        S2  ←─┤               ├─→  S1
              │               │
              └───────┬───────┘
                      |
                    2^31

Keys are mapped to the ring, and assigned to the next clockwise server.
```

#### Implementation of Consistent Hashing

```python
import hashlib
import bisect
from typing import List, Dict, Optional, Any

class ConsistentHashRing:
    def __init__(self, servers: List[str], virtual_nodes: int = 150):
        """
        Initialize consistent hash ring
        
        Args:
            servers: List of server addresses
            virtual_nodes: Number of virtual nodes per physical server
        """
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.servers = set()
        
        for server in servers:
            self.add_server(server)
    
    def _hash(self, key: str) -> int:
        """Generate hash value for a key"""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    
    def add_server(self, server: str) -> None:
        """Add a server to the hash ring"""
        if server in self.servers:
            return
            
        self.servers.add(server)
        
        # Add virtual nodes for this server
        for i in range(self.virtual_nodes):
            virtual_key = f"{server}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = server
            bisect.insort(self.sorted_keys, hash_value)
    
    def remove_server(self, server: str) -> None:
        """Remove a server from the hash ring"""
        if server not in self.servers:
            return
            
        self.servers.remove(server)
        
        # Remove all virtual nodes for this server
        keys_to_remove = []
        for hash_value, srv in self.ring.items():
            if srv == server:
                keys_to_remove.append(hash_value)
        
        for key in keys_to_remove:
            del self.ring[key]
            self.sorted_keys.remove(key)
    
    def get_server(self, key: str) -> Optional[str]:
        """Get the server responsible for a given key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find the first server clockwise from the hash value
        index = bisect.bisect_right(self.sorted_keys, hash_value)
        if index == len(self.sorted_keys):
            index = 0  # Wrap around to the beginning
        
        return self.ring[self.sorted_keys[index]]
    
    def get_servers_for_key(self, key: str, count: int = 1) -> List[str]:
        """Get multiple servers for replication"""
        if not self.ring or count <= 0:
            return []
        
        hash_value = self._hash(key)
        servers = []
        servers_seen = set()
        
        # Start from the first server clockwise
        index = bisect.bisect_right(self.sorted_keys, hash_value)
        
        while len(servers) < count and len(servers_seen) < len(self.servers):
            if index >= len(self.sorted_keys):
                index = 0  # Wrap around
            
            server = self.ring[self.sorted_keys[index]]
            if server not in servers_seen:
                servers.append(server)
                servers_seen.add(server)
            
            index += 1
        
        return servers
    
    def get_distribution(self, keys: List[str]) -> Dict[str, int]:
        """Analyze key distribution across servers"""
        distribution = {server: 0 for server in self.servers}
        
        for key in keys:
            server = self.get_server(key)
            if server:
                distribution[server] += 1
        
        return distribution

# Example usage and testing
def test_consistent_hashing():
    # Initialize with 3 servers
    servers = ['memcached1:11211', 'memcached2:11211', 'memcached3:11211']
    hash_ring = ConsistentHashRing(servers)
    
    # Generate test keys
    test_keys = [f'user:{i}' for i in range(1000)]
    
    # Check initial distribution
    initial_distribution = hash_ring.get_distribution(test_keys)
    print("Initial distribution:")
    for server, count in initial_distribution.items():
        print(f"  {server}: {count} keys ({count/len(test_keys)*100:.1f}%)")
    
    # Add a new server
    hash_ring.add_server('memcached4:11211')
    
    # Check distribution after adding server
    new_distribution = hash_ring.get_distribution(test_keys)
    print("\nDistribution after adding server:")
    for server, count in new_distribution.items():
        print(f"  {server}: {count} keys ({count/len(test_keys)*100:.1f}%)")
    
    # Calculate redistribution percentage
    moved_keys = 0
    for key in test_keys:
        initial_server = None
        for srv in servers:
            temp_ring = ConsistentHashRing(servers)
            if temp_ring.get_server(key) == srv:
                initial_server = srv
                break
        
        new_server = hash_ring.get_server(key)
        if initial_server != new_server:
            moved_keys += 1
    
    print(f"\nKeys moved after adding server: {moved_keys}/{len(test_keys)} ({moved_keys/len(test_keys)*100:.1f}%)")

if __name__ == "__main__":
    test_consistent_hashing()
```

### Distributed Memcached Client

```python
import memcache
from typing import List, Dict, Any, Optional
import time
import random

class DistributedMemcachedClient:
    def __init__(self, servers: List[str], replication_factor: int = 2):
        """
        Distributed Memcached client with consistent hashing
        
        Args:
            servers: List of memcached server addresses
            replication_factor: Number of replicas per key
        """
        self.hash_ring = ConsistentHashRing(servers)
        self.replication_factor = replication_factor
        self.clients = {}
        
        # Create individual clients for each server
        for server in servers:
            self.clients[server] = memcache.Client([server])
    
    def _get_clients_for_key(self, key: str) -> List[memcache.Client]:
        """Get memcached clients for a key based on consistent hashing"""
        servers = self.hash_ring.get_servers_for_key(key, self.replication_factor)
        return [self.clients[server] for server in servers if server in self.clients]
    
    def set(self, key: str, value: Any, expiry: int = 3600) -> bool:
        """Set value with replication"""
        clients = self._get_clients_for_key(key)
        success_count = 0
        
        for client in clients:
            try:
                if client.set(key, value, time=expiry):
                    success_count += 1
            except Exception as e:
                print(f"Failed to set {key} on client: {e}")
        
        # Consider successful if majority of replicas succeeded
        return success_count > len(clients) // 2
    
    def get(self, key: str) -> Any:
        """Get value with fallback to replicas"""
        clients = self._get_clients_for_key(key)
        
        for client in clients:
            try:
                value = client.get(key)
                if value is not None:
                    return value
            except Exception as e:
                print(f"Failed to get {key} from client: {e}")
                continue
        
        return None
    
    def delete(self, key: str) -> bool:
        """Delete key from all replicas"""
        clients = self._get_clients_for_key(key)
        success_count = 0
        
        for client in clients:
            try:
                if client.delete(key):
                    success_count += 1
            except Exception as e:
                print(f"Failed to delete {key} from client: {e}")
        
        return success_count > 0
    
    def add_server(self, server: str) -> None:
        """Add a new server to the cluster"""
        self.hash_ring.add_server(server)
        self.clients[server] = memcache.Client([server])
    
    def remove_server(self, server: str) -> None:
        """Remove a server from the cluster"""
        self.hash_ring.remove_server(server)
        if server in self.clients:
            del self.clients[server]
    
    def get_cluster_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all servers"""
        stats = {}
        for server, client in self.clients.items():
            try:
                server_stats = client.get_stats()
                if server_stats:
                    stats[server] = server_stats[0]  # get_stats returns list of tuples
            except Exception as e:
                stats[server] = {'error': str(e)}
        
        return stats

# Usage example
def demo_distributed_cache():
    # Setup distributed cache
    servers = [
        '127.0.0.1:11211',
        '127.0.0.1:11212', 
        '127.0.0.1:11213'
    ]
    
    cache = DistributedMemcachedClient(servers, replication_factor=2)
    
    # Store some data
    cache.set('user:1001', {'name': 'Alice', 'age': 30})
    cache.set('user:1002', {'name': 'Bob', 'age': 25})
    cache.set('session:abc123', {'user_id': 1001, 'login_time': time.time()})
    
    # Retrieve data
    user = cache.get('user:1001')
    print(f"Retrieved user: {user}")
    
    # Add a new server (simulating scaling)
    cache.add_server('127.0.0.1:11214')
    
    # Data should still be accessible
    user = cache.get('user:1001')
    print(f"User after scaling: {user}")
    
    # Get cluster statistics
    stats = cache.get_cluster_stats()
    for server, server_stats in stats.items():
        if 'error' not in server_stats:
            print(f"Server {server}: {server_stats.get('curr_items', 0)} items")

if __name__ == "__main__":
    demo_distributed_cache()
```

### Sharding Strategies

#### Range-based Sharding
```python
class RangeShardedCache:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.clients = [memcache.Client([server]) for server in servers]
        self.num_shards = len(servers)
    
    def _get_shard(self, key: str) -> int:
        """Determine shard based on key range"""
        if key.startswith('user:'):
            user_id = int(key.split(':')[1])
            return user_id % self.num_shards
        elif key.startswith('session:'):
            # Hash-based for sessions
            return hash(key) % self.num_shards
        else:
            return 0  # Default shard
    
    def set(self, key: str, value: Any, expiry: int = 3600) -> bool:
        shard = self._get_shard(key)
        return self.clients[shard].set(key, value, time=expiry)
    
    def get(self, key: str) -> Any:
        shard = self._get_shard(key)
        return self.clients[shard].get(key)
```

#### Directory-based Sharding
```python
class DirectoryShardedCache:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.clients = {server: memcache.Client([server]) for server in servers}
        self.directory = {}  # key -> server mapping
        self.directory_client = memcache.Client([servers[0]])  # Directory stored on first server
    
    def _get_server_for_key(self, key: str) -> str:
        """Get server for key using directory lookup"""
        # Check directory cache first
        mapping = self.directory_client.get(f"dir:{key}")
        if mapping:
            return mapping
        
        # Assign to least loaded server (simplified)
        server = random.choice(self.servers)
        self.directory_client.set(f"dir:{key}", server)
        return server
    
    def set(self, key: str, value: Any, expiry: int = 3600) -> bool:
        server = self._get_server_for_key(key)
        return self.clients[server].set(key, value, time=expiry)
    
    def get(self, key: str) -> Any:
        server = self._get_server_for_key(key)
        return self.clients[server].get(key)
```

## Cache Invalidation Strategies

Cache invalidation is one of the hardest problems in computer science. Here are proven strategies for different scenarios:

### 1. Time-based Expiration (TTL)

The simplest strategy where cached data expires after a fixed time period.

```python
import time
from typing import Optional, Any, Callable

class TTLCache:
    def __init__(self, memcached_client):
        self.cache = memcached_client
        self.default_ttl = 3600  # 1 hour
    
    def get_with_ttl(self, key: str, ttl: int = None) -> Any:
        """Get value with custom TTL"""
        return self.cache.get(key)
    
    def set_with_ttl(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value with TTL"""
        if ttl is None:
            ttl = self.default_ttl
        return self.cache.set(key, value, expiry=ttl)
    
    def get_or_compute(self, key: str, compute_func: Callable[[], Any], 
                      ttl: int = None) -> Any:
        """Get from cache or compute and cache the result"""
        value = self.cache.get(key)
        if value is not None:
            return value
        
        # Cache miss - compute value
        computed_value = compute_func()
        if computed_value is not None:
            self.set_with_ttl(key, computed_value, ttl)
        
        return computed_value

# Usage example
def expensive_database_query(user_id: int) -> dict:
    """Simulate expensive database operation"""
    time.sleep(0.1)  # Simulate DB latency
    return {
        'id': user_id,
        'name': f'User {user_id}',
        'email': f'user{user_id}@example.com',
        'last_login': time.time()
    }

cache = TTLCache(memcache.Client(['127.0.0.1:11211']))

# Get user data with caching
user_data = cache.get_or_compute(
    f'user:1001',
    lambda: expensive_database_query(1001),
    ttl=1800  # Cache for 30 minutes
)
```

### 2. Write-through Caching

Data is written to both cache and database simultaneously.

```python
import logging
from typing import Any, Dict

class WriteThroughCache:
    def __init__(self, cache_client, database_client):
        self.cache = cache_client
        self.db = database_client
        self.logger = logging.getLogger(__name__)
    
    def write(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Write to both cache and database"""
        try:
            # Write to database first
            db_success = self._write_to_database(key, value)
            if not db_success:
                self.logger.error(f"Failed to write {key} to database")
                return False
            
            # Then write to cache
            cache_success = self.cache.set(key, value, expiry=ttl)
            if not cache_success:
                self.logger.warning(f"Failed to write {key} to cache")
                # Don't fail the operation if cache write fails
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in write-through for {key}: {e}")
            return False
    
    def read(self, key: str) -> Any:
        """Read from cache, fallback to database"""
        # Try cache first
        value = self.cache.get(key)
        
        if value is not None:
            return value
        
        # Cache miss - read from database
        value = self._read_from_database(key)
        if value is not None:
            # Populate cache
            self.cache.set(key, value)
        
        return value
    
    def _write_to_database(self, key: str, value: Any) -> bool:
        """Simulate database write"""
        # Implementation depends on your database
        try:
            # db.execute("INSERT INTO cache_data (key, value) VALUES (?, ?)", (key, value))
            return True
        except Exception:
            return False
    
    def _read_from_database(self, key: str) -> Any:
        """Simulate database read"""
        # Implementation depends on your database
        try:
            # result = db.execute("SELECT value FROM cache_data WHERE key = ?", (key,))
            # return result.fetchone() if result else None
            return None
        except Exception:
            return None
```

### 3. Write-behind (Write-back) Caching

Data is written to cache immediately and to database asynchronously.

```python
import threading
import queue
import time
from dataclasses import dataclass
from typing import Any

@dataclass
class WriteOperation:
    key: str
    value: Any
    timestamp: float
    operation_type: str  # 'set', 'delete'

class WriteBehindCache:
    def __init__(self, cache_client, database_client, 
                 batch_size: int = 100, flush_interval: int = 30):
        self.cache = cache_client
        self.db = database_client
        self.write_queue = queue.Queue()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.running = True
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._background_writer)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def write(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Write to cache immediately, queue for database"""
        # Write to cache first
        cache_success = self.cache.set(key, value, expiry=ttl)
        
        # Queue for async replication
        operation = WriteOperation(
            key=key,
            value=value,
            timestamp=time.time(),
            operation_type='set'
        )
        
        self.write_queue.put(operation)
        
        return cache_success
    
    def delete(self, key: str) -> bool:
        """Delete from cache immediately, queue for database"""
        cache_success = self.cache.delete(key)
        
        # Queue for database delete
        operation = WriteOperation(
            key=key,
            value=None,
            timestamp=time.time(),
            operation_type='delete'
        )
        self.write_queue.put(operation)
        
        return cache_success
    
    def read(self, key: str) -> Any:
        """Read from cache, fallback to database"""
        value = self.cache.get(key)
        if value is not None:
            return value
        
        # Cache miss - read from database
        value = self._read_from_database(key)
        if value is not None:
            self.cache.set(key, value)
        
        return value
    
    def _background_writer(self):
        """Background thread to flush writes to database"""
        batch = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Try to get operation with timeout
                operation = self.write_queue.get(timeout=1.0)
                batch.append(operation)
                
                # Flush if batch is full or enough time has passed
                if (len(batch) >= self.batch_size or 
                    time.time() - last_flush >= self.flush_interval):
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except queue.Empty:
                # Timeout - flush pending operations
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
    
    def _flush_batch(self, batch: list):
        """Flush a batch of operations to database"""
        if not batch:
            return
        
        try:
            # Group operations by type
            sets = [op for op in batch if op.operation_type == 'set']
            deletes = [op for op in batch if op.operation_type == 'delete']
            
            # Batch write to database
            if sets:
                self._batch_write_database(sets)
            
            if deletes:
                self._batch_delete_database(deletes)
                
        except Exception as e:
            logging.error(f"Failed to flush batch to database: {e}")
    
    def _start_batch_processor(self):
        """Start background thread for batch processing"""
        def batch_processor():
            while True:
                time.sleep(self.flush_interval)
                self._flush_score_batch()
        
        thread = threading.Thread(target=batch_processor)
        thread.daemon = True
        thread.start()
    
    def _invalidate_leaderboard_caches(self, game_id: int):
        """Invalidate all leaderboard caches for a game"""
        # Pattern-based invalidation
        patterns = [
            f"leaderboard:{game_id}:*",
            f"tournament:*:live"  # If game is part of tournament
        ]
        
        for pattern in patterns:
            self._invalidate_pattern(pattern)
    
    def _invalidate_player_caches(self, player_id: int, game_id: int):
        """Invalidate player-specific caches"""
        keys = [
            f"player_rank:{player_id}:{game_id}:global",
            f"player_rank:{player_id}:{game_id}:friends",
            f"player_stats:{player_id}:{game_id}"
        ]
        
        for key in keys:
            self.cache.delete(key)
    
    # Helper methods
    def _generate_leaderboard(self, game_id: int, board_type: str, limit: int) -> List[dict]:
        pass
    
    def _calculate_player_rank(self, player_id: int, game_id: int, board_type: str) -> dict:
        pass
    
    def _update_score_in_db(self, player_id: int, game_id: int, score: int) -> bool:
        pass
    
    def _batch_update_scores_in_db(self, batch: List[dict]):
        pass
    
    def _invalidate_pattern(self, pattern: str):
        pass
```

## Learning Resources and Exercises

### Recommended Reading

**Essential Books:**
- **"Scaling Memcache at Facebook"** - Facebook Engineering whitepaper
- **"High Performance Browser Networking"** by Ilya Grigorik - Chapter on caching
- **"Designing Data-Intensive Applications"** by Martin Kleppmann - Caching patterns
- **"Building Scalable Web Sites"** by Cal Henderson - Memcached in practice

**Online Resources:**
- [Memcached Official Documentation](https://memcached.org/about)
- [High Scalability - Memcached Articles](http://highscalability.com/blog/category/memcached)
- [AWS ElastiCache Documentation](https://aws.amazon.com/elasticache/)
- [Google Cloud Memorystore Documentation](https://cloud.google.com/memorystore)

**Research Papers:**
- "Scaling Memcache at Facebook" (Facebook, 2013)
- "TAO: Facebook's Distributed Data Store for the Social Graph" (Facebook, 2013)
- "Dynamo: Amazon's Highly Available Key-value Store" (Amazon, 2007)

### Hands-on Exercises

#### Exercise 1: Basic Memcached Setup and Operations

**Objective:** Set up a Memcached server and perform basic operations

**Tasks:**
1. Install Memcached on your local machine
2. Connect using telnet and perform manual operations
3. Implement a Python client with error handling
4. Measure and compare performance vs direct database access

**Solution Framework:**
```python
# TODO: Complete this implementation
import memcache
import time
import sqlite3

class MemcachedExercise:
    def __init__(self):
        self.mc = memcache.Client(['127.0.0.1:11211'])
        self.db = sqlite3.connect(':memory:')
        self._setup_database()
    
    def _setup_database(self):
        # Create test table
        self.db.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Insert test data
        for i in range(1000):
            self.db.execute(
                "INSERT INTO users (name, email, created_at) VALUES (?, ?, ?)",
                (f"User {i}", f"user{i}@example.com", time.time())
            )
        self.db.commit()
    
    def benchmark_cache_vs_db(self, user_id: int, iterations: int = 100):
        """Compare cache vs database performance"""
        # TODO: Implement performance comparison
        # 1. Time database-only lookups
        # 2. Time cache-enabled lookups
        # 3. Calculate and display performance improvement
        pass
    
    def test_cache_consistency(self):
        """Test cache invalidation and consistency"""
        # TODO: Implement consistency tests
        # 1. Update data in database
        # 2. Verify cache invalidation
        # 3. Test race conditions
        pass

# Your implementation here
exercise = MemcachedExercise()
exercise.benchmark_cache_vs_db(1)
exercise.test_cache_consistency()
```

#### Exercise 2: Implementing Consistent Hashing

**Objective:** Build a consistent hashing implementation from scratch

**Tasks:**
1. Implement basic consistent hashing
2. Add virtual nodes for better distribution
3. Test redistribution when servers are added/removed
4. Compare with simple modulo hashing

**Solution Framework:**
```python
class ConsistentHashingExercise:
    def __init__(self, servers: List[str], virtual_nodes: int = 150):
        # TODO: Implement consistent hashing ring
        pass
    
    def add_server(self, server: str):
        # TODO: Add server and measure redistribution
        pass
    
    def remove_server(self, server: str):
        # TODO: Remove server and measure redistribution
        pass
    
    def test_distribution_fairness(self, num_keys: int = 10000):
        """Test how evenly keys are distributed"""
        # TODO: Generate keys and measure distribution
        # Calculate standard deviation and max/min ratios
        pass
    
    def compare_with_modulo_hashing(self, num_keys: int = 10000):
        """Compare redistribution with simple modulo hashing"""
        # TODO: Compare both approaches when scaling
        pass

# Your implementation here
```

#### Exercise 3: Cache Invalidation Strategies

**Objective:** Implement and compare different invalidation strategies

**Tasks:**
1. Implement TTL-based invalidation
2. Build event-driven invalidation system
3. Create multi-level cache with proper invalidation
4. Test performance and consistency of each approach

**Solution Framework:**
```python
class CacheInvalidationExercise:
    def __init__(self):
        self.cache = memcache.Client(['127.0.0.1:11211'])
        self.event_listeners = {}
    
    def implement_ttl_cache(self):
        """Implement TTL-based caching with different strategies"""
        # TODO: Implement adaptive TTL based on data volatility
        pass
    
    def implement_event_driven_cache(self):
        """Implement event-driven invalidation"""
        # TODO: Create event system that invalidates related caches
        pass
    
    def implement_cache_tags(self):
        """Implement cache tagging for group invalidation"""
        # TODO: Tag related cache entries and invalidate by tags
        pass
    
    def test_invalidation_performance(self):
        """Measure performance impact of different strategies"""
        # TODO: Benchmark invalidation overhead
        pass

# Your implementation here
```

#### Exercise 4: Building a Production-Ready Cache Layer

**Objective:** Create a comprehensive caching solution with monitoring

**Tasks:**
1. Implement connection pooling and circuit breakers
2. Add comprehensive monitoring and metrics
3. Build auto-scaling capabilities
4. Create a fault-tolerant distributed cache

**Solution Framework:**
```python
class ProductionCacheSystem:
    def __init__(self, config: dict):
        # TODO: Initialize production-ready cache system
        # Include: connection pooling, monitoring, circuit breakers
        pass
    
    def implement_monitoring(self):
        """Add comprehensive monitoring"""
        # TODO: Track hit ratios, response times, error rates
        # Implement alerting for threshold breaches
        pass
    
    def implement_auto_scaling(self):
        """Add auto-scaling based on metrics"""
        # TODO: Scale cache cluster based on load
        pass
    
    def implement_fault_tolerance(self):
        """Add fault tolerance mechanisms"""
        # TODO: Implement failover, replica management, data recovery
        pass
    
    def load_test(self, concurrent_users: int = 100, duration: int = 300):
        """Perform comprehensive load testing"""
        # TODO: Simulate real-world load patterns
        pass

# Your implementation here
```

### Practice Projects

#### Project 1: Social Media Cache Architecture

**Description:** Build a caching layer for a social media application

**Requirements:**
- User profile caching with different TTLs based on activity
- News feed caching with real-time invalidation
- Friend relationship caching with consistency guarantees
- Trending topics cache with geographic distribution
- Handle 1M+ users with 10K+ concurrent requests

#### Project 2: E-commerce Product Catalog Cache

**Description:** Design caching for a large e-commerce platform

**Requirements:**
- Product catalog with inventory tracking
- Category browsing with faceted search caching
- Recommendation engine result caching
- Shopping cart persistence across sessions
- Handle flash sales and traffic spikes

#### Project 3: Gaming Leaderboard System

**Description:** Create a real-time gaming leaderboard with caching

**Requirements:**
- Real-time score updates with sub-second latency
- Multiple leaderboard types (global, friends, tournaments)
- Historical data with efficient querying
- Anti-cheat measures with cache validation
- Handle millions of score updates per minute

### Assessment Criteria

**Knowledge Assessment:**
- [ ] Understand Memcached architecture and memory management
- [ ] Explain consistent hashing and its benefits
- [ ] Design appropriate cache invalidation strategies
- [ ] Implement fault-tolerant distributed caching
- [ ] Optimize cache performance and monitor metrics
- [ ] Compare different caching solutions and patterns

**Practical Skills:**
- [ ] Set up and configure Memcached clusters
- [ ] Implement cache clients with proper error handling
- [ ] Design cache hierarchies for complex applications
- [ ] Troubleshoot cache-related performance issues
- [ ] Implement monitoring and alerting for cache systems
- [ ] Handle cache failures gracefully

**Production Readiness:**
- [ ] Understand operational aspects of cache management
- [ ] Implement proper security measures
- [ ] Design for scalability and high availability
- [ ] Create disaster recovery procedures
- [ ] Optimize costs while maintaining performance

## Conclusion

Memcached and distributed caching are fundamental technologies for building scalable, high-performance applications. This comprehensive guide has covered:

- **Architecture fundamentals** - Understanding how Memcached works internally
- **Distributed strategies** - Consistent hashing, sharding, and replication
- **Invalidation patterns** - Various approaches to cache consistency
- **Performance optimization** - Connection pooling, compression, monitoring
- **Real-world examples** - Production architectures from major companies
- **Best practices** - Operational excellence and reliability patterns

### Key Takeaways

1. **Caching is critical** for modern applications but must be designed carefully
2. **Consistent hashing** solves distribution problems elegantly
3. **Cache invalidation** is the hardest part - plan for it from the beginning
4. **Monitoring and observability** are essential for production systems
5. **Fault tolerance** must be built in - systems should work without cache
6. **Performance optimization** requires understanding your specific use case

### Next Steps

After mastering Memcached and distributed caching, consider exploring:

- **Redis** - More advanced data structures and features
- **CDN integration** - Edge caching strategies
- **Database caching** - Query result caching and optimization
- **Application-level caching** - Object caching and serialization strategies
- **Cache-aware architecture** - Designing systems with caching as a first-class citizen

The principles and patterns learned here apply to all caching technologies and will serve as a foundation for building scalable, high-performance distributed systems.
