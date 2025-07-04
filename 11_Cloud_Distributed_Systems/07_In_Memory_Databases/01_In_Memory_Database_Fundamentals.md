# In-Memory Database Fundamentals

*Duration: 2 weeks*

## Overview

In-memory databases (IMDBs) store data primarily in main memory (RAM) rather than on disk storage. This fundamental architectural difference enables significantly faster data access and processing, making them ideal for applications requiring low latency and high throughput. This module covers the core concepts, architectures, and implementation strategies of in-memory databases.

## Learning Objectives

By the end of this section, you should be able to:
- **Understand the architecture** and design principles of in-memory databases
- **Compare and contrast** in-memory vs disk-based database systems
- **Implement memory management techniques** for optimal performance
- **Design persistence strategies** including snapshots and write-ahead logs
- **Build a simple in-memory database** from scratch
- **Evaluate trade-offs** between performance, durability, and cost

## Architecture of In-Memory Databases

### Core Architectural Principles

In-memory databases are built on several key architectural principles that differentiate them from traditional disk-based systems:

#### 1. Memory-First Design
```
Traditional Database:        In-Memory Database:
┌─────────────────┐         ┌─────────────────┐
│   Application   │         │   Application   │
└─────────┬───────┘         └─────────┬───────┘
          │                           │
┌─────────▼───────┐         ┌─────────▼───────┐
│  Database       │         │  Database       │
│  Engine         │         │  Engine         │
│ ┌─────────────┐ │         │ ┌─────────────┐ │
│ │ Buffer Pool │ │         │ │Primary Store│ │
│ │(Limited RAM)│ │         │ │  (All RAM)  │ │
│ └─────────────┘ │         │ └─────────────┘ │
│ ┌─────────────┐ │         │ ┌─────────────┐ │
│ │Primary Store│ │         │ │Backup Store │ │
│ │   (Disk)    │ │         │ │   (Disk)    │ │
│ └─────────────┘ │         │ └─────────────┘ │
└─────────────────┘         └─────────────────┘
```

#### 2. Data Structure Optimization

In-memory databases use data structures optimized for RAM access patterns:

```python
# Example: Optimized in-memory data structures
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class Record:
    """Optimized record structure for in-memory storage"""
    key: str
    value: Any
    timestamp: datetime
    version: int = 1
    ttl: Optional[datetime] = None  # Time to live
    
    def is_expired(self) -> bool:
        return self.ttl and datetime.now() > self.ttl

class InMemoryIndex:
    """Hash-based index for O(1) lookups"""
    def __init__(self):
        self._index: Dict[str, Record] = {}
        self._lock = threading.RWLock()  # Read-write lock for concurrency
    
    def put(self, key: str, value: Any, ttl: Optional[datetime] = None) -> bool:
        with self._lock.write_lock():
            if key in self._index:
                # Update existing record
                self._index[key].value = value
                self._index[key].timestamp = datetime.now()
                self._index[key].version += 1
                self._index[key].ttl = ttl
            else:
                # Create new record
                self._index[key] = Record(key, value, datetime.now(), 1, ttl)
            return True
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock.read_lock():
            record = self._index.get(key)
            if record and not record.is_expired():
                return record.value
            elif record and record.is_expired():
                # Lazy expiration
                del self._index[key]
            return None
    
    def delete(self, key: str) -> bool:
        with self._lock.write_lock():
            return self._index.pop(key, None) is not None
    
    def size(self) -> int:
        with self._lock.read_lock():
            return len(self._index)
```

#### 3. Multi-Version Concurrency Control (MVCC)

```python
class MVCCDatabase:
    """Multi-Version Concurrency Control implementation"""
    
    def __init__(self):
        self._data: Dict[str, List[Record]] = {}
        self._transaction_counter = 0
        self._lock = threading.Lock()
    
    def begin_transaction(self) -> int:
        with self._lock:
            self._transaction_counter += 1
            return self._transaction_counter
    
    def read(self, key: str, transaction_id: int) -> Optional[Any]:
        """Read the latest version visible to the transaction"""
        versions = self._data.get(key, [])
        
        # Find the latest version created before this transaction
        for record in reversed(versions):
            if record.version <= transaction_id:
                return record.value
        
        return None
    
    def write(self, key: str, value: Any, transaction_id: int) -> bool:
        """Create a new version of the record"""
        new_record = Record(key, value, datetime.now(), transaction_id)
        
        if key not in self._data:
            self._data[key] = []
        
        self._data[key].append(new_record)
        
        # Keep only recent versions (cleanup old versions)
        self._cleanup_old_versions(key)
        return True
    
    def _cleanup_old_versions(self, key: str, keep_versions: int = 10):
        """Keep only the most recent versions"""
        if key in self._data and len(self._data[key]) > keep_versions:
            self._data[key] = self._data[key][-keep_versions:]
```

### Performance Characteristics

#### Memory Access Patterns
```python
import time
import random

def benchmark_access_patterns():
    """Demonstrate memory vs disk access patterns"""
    
    # Simulate in-memory access (nanoseconds)
    def memory_access():
        data = {}
        start = time.perf_counter()
        
        # Sequential access pattern
        for i in range(10000):
            data[f"key_{i}"] = f"value_{i}"
        
        # Random access pattern
        for _ in range(10000):
            key = f"key_{random.randint(0, 9999)}"
            _ = data.get(key)
        
        return time.perf_counter() - start
    
    # Simulate disk access (with caching)
    def disk_simulation():
        import sqlite3
        
        conn = sqlite3.connect(':memory:')  # In-memory SQLite for comparison
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE data (key TEXT PRIMARY KEY, value TEXT)')
        
        start = time.perf_counter()
        
        # Sequential writes
        for i in range(10000):
            cursor.execute('INSERT INTO data VALUES (?, ?)', 
                         (f"key_{i}", f"value_{i}"))
        conn.commit()
        
        # Random reads
        for _ in range(10000):
            key = f"key_{random.randint(0, 9999)}"
            cursor.execute('SELECT value FROM data WHERE key = ?', (key,))
            _ = cursor.fetchone()
        
        conn.close()
        return time.perf_counter() - start
    
    memory_time = memory_access()
    disk_time = disk_simulation()
    
    print(f"Memory access time: {memory_time:.4f} seconds")
    print(f"Disk simulation time: {disk_time:.4f} seconds")
    print(f"Speed improvement: {disk_time/memory_time:.2f}x faster")

# benchmark_access_patterns()
```

## In-Memory vs. Disk-Based Databases

### Comprehensive Comparison

| Aspect | In-Memory Database | Disk-Based Database |
|--------|-------------------|-------------------|
| **Performance** | Sub-millisecond latency | Milliseconds to seconds |
| **Throughput** | Millions of ops/sec | Thousands of ops/sec |
| **Data Volume** | Limited by RAM size | Limited by disk space |
| **Durability** | Requires additional mechanisms | Built-in durability |
| **Cost per GB** | $8-20/GB (RAM) | $0.03-0.10/GB (SSD/HDD) |
| **Power Consumption** | Higher (RAM always powered) | Lower (disk spins down) |
| **Scalability** | Vertical scaling costly | Horizontal scaling easier |
| **Recovery Time** | Fast (reload from snapshot) | Variable (depends on size) |

### Use Case Analysis

```python
class DatabaseSelector:
    """Helper class to choose between in-memory and disk-based solutions"""
    
    @staticmethod
    def analyze_requirements(
        data_size_gb: float,
        read_ops_per_sec: int,
        write_ops_per_sec: int,
        latency_requirement_ms: float,
        budget_per_month: float,
        durability_requirement: str  # 'high', 'medium', 'low'
    ) -> Dict[str, Any]:
        
        analysis = {
            'recommended_type': None,
            'reasoning': [],
            'considerations': []
        }
        
        # Performance analysis
        if latency_requirement_ms < 1:
            analysis['reasoning'].append("Sub-millisecond latency requires in-memory")
            score_inmemory = 10
        elif latency_requirement_ms < 10:
            analysis['reasoning'].append("Low latency favors in-memory")
            score_inmemory = 7
        else:
            score_inmemory = 3
        
        # Throughput analysis
        total_ops = read_ops_per_sec + write_ops_per_sec
        if total_ops > 100000:
            analysis['reasoning'].append("High throughput favors in-memory")
            score_inmemory += 5
        elif total_ops > 10000:
            score_inmemory += 2
        
        # Cost analysis
        ram_cost = data_size_gb * 15  # $15/GB RAM (rough estimate)
        disk_cost = data_size_gb * 0.05  # $0.05/GB SSD
        
        if ram_cost > budget_per_month:
            analysis['reasoning'].append("Budget constraints favor disk-based")
            score_inmemory -= 5
        
        # Durability analysis
        if durability_requirement == 'high':
            analysis['reasoning'].append("High durability needs favor disk-based")
            score_inmemory -= 3
        elif durability_requirement == 'medium':
            analysis['considerations'].append("Consider hybrid approach")
        
        # Final recommendation
        if score_inmemory >= 7:
            analysis['recommended_type'] = 'in-memory'
        elif score_inmemory >= 4:
            analysis['recommended_type'] = 'hybrid'
        else:
            analysis['recommended_type'] = 'disk-based'
        
        analysis['score'] = score_inmemory
        analysis['estimated_costs'] = {
            'in_memory_monthly': ram_cost,
            'disk_based_monthly': disk_cost
        }
        
        return analysis

# Example usage
requirements = DatabaseSelector.analyze_requirements(
    data_size_gb=50,
    read_ops_per_sec=50000,
    write_ops_per_sec=10000,
    latency_requirement_ms=0.5,
    budget_per_month=1000,
    durability_requirement='medium'
)

print(f"Recommended: {requirements['recommended_type']}")
print(f"Reasoning: {requirements['reasoning']}")
```

## Memory Management Techniques

### 1. Memory Pool Management

```python
import ctypes
from typing import List, Optional
import threading

class MemoryPool:
    """Fixed-size memory pool for efficient allocation"""
    
    def __init__(self, block_size: int, pool_size: int):
        self.block_size = block_size
        self.pool_size = pool_size
        
        # Allocate contiguous memory block
        self._memory = ctypes.create_string_buffer(block_size * pool_size)
        self._free_blocks: List[int] = list(range(pool_size))
        self._allocated_blocks: set = set()
        self._lock = threading.Lock()
    
    def allocate(self) -> Optional[int]:
        """Allocate a memory block, returns block index"""
        with self._lock:
            if not self._free_blocks:
                return None  # Pool exhausted
            
            block_index = self._free_blocks.pop()
            self._allocated_blocks.add(block_index)
            return block_index
    
    def deallocate(self, block_index: int) -> bool:
        """Deallocate a memory block"""
        with self._lock:
            if block_index not in self._allocated_blocks:
                return False
            
            self._allocated_blocks.remove(block_index)
            self._free_blocks.append(block_index)
            
            # Zero out the memory for security
            start_offset = block_index * self.block_size
            for i in range(self.block_size):
                self._memory[start_offset + i] = 0
            
            return True
    
    def get_memory_address(self, block_index: int) -> Optional[int]:
        """Get the memory address for a block"""
        if block_index not in self._allocated_blocks:
            return None
        
        return ctypes.addressof(self._memory) + (block_index * self.block_size)
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory pool statistics"""
        with self._lock:
            return {
                'total_blocks': self.pool_size,
                'allocated_blocks': len(self._allocated_blocks),
                'free_blocks': len(self._free_blocks),
                'memory_utilization': len(self._allocated_blocks) / self.pool_size
            }

# Example usage
pool = MemoryPool(block_size=1024, pool_size=1000)  # 1MB total
block_id = pool.allocate()
if block_id is not None:
    print(f"Allocated block {block_id}")
    print(f"Pool stats: {pool.get_stats()}")
    pool.deallocate(block_id)
```

### 2. Garbage Collection and Memory Compaction

```python
import gc
import weakref
from typing import Set, Dict, Any

class InMemoryGarbageCollector:
    """Custom garbage collector for in-memory database"""
    
    def __init__(self):
        self._live_objects: Set[weakref.ref] = set()
        self._collection_threshold = 1000
        self._allocations_since_gc = 0
    
    def register_object(self, obj: Any) -> weakref.ref:
        """Register an object for garbage collection tracking"""
        ref = weakref.ref(obj, self._object_finalized)
        self._live_objects.add(ref)
        self._allocations_since_gc += 1
        
        if self._allocations_since_gc >= self._collection_threshold:
            self.collect()
        
        return ref
    
    def _object_finalized(self, ref: weakref.ref):
        """Callback when an object is finalized"""
        self._live_objects.discard(ref)
    
    def collect(self) -> Dict[str, int]:
        """Perform garbage collection"""
        initial_count = len(self._live_objects)
        
        # Remove dead references
        dead_refs = [ref for ref in self._live_objects if ref() is None]
        for ref in dead_refs:
            self._live_objects.discard(ref)
        
        # Force Python GC
        collected = gc.collect()
        
        self._allocations_since_gc = 0
        
        return {
            'dead_references_removed': len(dead_refs),
            'python_objects_collected': collected,
            'live_objects_remaining': len(self._live_objects)
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        return {
            'live_objects': len(self._live_objects),
            'allocations_since_gc': self._allocations_since_gc,
            'gc_threshold': self._collection_threshold
        }

class CompactingDatabase:
    """Database with memory compaction support"""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._gc = InMemoryGarbageCollector()
        self._fragmentation_threshold = 0.3
    
    def put(self, key: str, value: Any):
        """Store data with GC tracking"""
        self._data[key] = value
        self._gc.register_object(value)
    
    def compact_memory(self) -> Dict[str, Any]:
        """Perform memory compaction"""
        # Collect garbage first
        gc_stats = self._gc.collect()
        
        # Rebuild data structures to reduce fragmentation
        old_data = self._data
        self._data = {}
        
        for key, value in old_data.items():
            if value is not None:  # Only keep non-null values
                self._data[key] = value
        
        return {
            'gc_stats': gc_stats,
            'keys_compacted': len(self._data),
            'fragmentation_reduced': True
        }
```

### 3. Cache Management and Eviction Policies

```python
from collections import OrderedDict
import heapq
from typing import Any, Optional
import time

class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache = OrderedDict()
        self._access_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            # Move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._access_count += 1
            return value
        return None
    
    def put(self, key: str, value: Any):
        if key in self._cache:
            # Update existing
            self._cache.pop(key)
        elif len(self._cache) >= self.capacity:
            # Remove least recently used
            self._cache.popitem(last=False)
        
        self._cache[key] = value
        self._access_count += 1
    
    def get_stats(self) -> Dict[str, int]:
        return {
            'size': len(self._cache),
            'capacity': self.capacity,
            'access_count': self._access_count,
            'utilization': len(self._cache) / self.capacity
        }

class LFUCache:
    """Least Frequently Used cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: Dict[str, Any] = {}
        self._frequencies: Dict[str, int] = {}
        self._freq_groups: Dict[int, set] = {}
        self._min_freq = 0
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        # Update frequency
        self._update_frequency(key)
        return self._cache[key]
    
    def put(self, key: str, value: Any):
        if self.capacity <= 0:
            return
        
        if key in self._cache:
            # Update existing
            self._cache[key] = value
            self._update_frequency(key)
            return
        
        if len(self._cache) >= self.capacity:
            # Evict least frequently used
            self._evict_lfu()
        
        # Add new item
        self._cache[key] = value
        self._frequencies[key] = 1
        self._freq_groups.setdefault(1, set()).add(key)
        self._min_freq = 1
    
    def _update_frequency(self, key: str):
        freq = self._frequencies[key]
        self._frequencies[key] = freq + 1
        
        # Move from old frequency group to new
        self._freq_groups[freq].remove(key)
        if not self._freq_groups[freq] and freq == self._min_freq:
            self._min_freq += 1
        
        self._freq_groups.setdefault(freq + 1, set()).add(key)
    
    def _evict_lfu(self):
        # Remove least frequently used item
        key = self._freq_groups[self._min_freq].pop()
        del self._cache[key]
        del self._frequencies[key]

class TTLCache:
    """Time-To-Live cache implementation"""
    
    def __init__(self, default_ttl: float = 3600):  # 1 hour default
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._default_ttl = default_ttl
        self._cleanup_heap: List[tuple] = []  # (expiry_time, key)
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        expiry_time = time.time() + (ttl or self._default_ttl)
        self._cache[key] = (value, expiry_time)
        heapq.heappush(self._cleanup_heap, (expiry_time, key))
    
    def get(self, key: str) -> Optional[Any]:
        self._cleanup_expired()
        
        if key in self._cache:
            value, expiry_time = self._cache[key]
            if time.time() < expiry_time:
                return value
            else:
                del self._cache[key]
        
        return None
    
    def _cleanup_expired(self):
        current_time = time.time()
        
        while self._cleanup_heap and self._cleanup_heap[0][0] <= current_time:
            expiry_time, key = heapq.heappop(self._cleanup_heap)
            if key in self._cache and self._cache[key][1] == expiry_time:
                del self._cache[key]
```

## Persistence Strategies

### 1. Snapshot-Based Persistence

```python
import json
import pickle
import gzip
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class SnapshotManager:
    """Manages database snapshots for persistence"""
    
    def __init__(self, database: Dict[str, Any], snapshot_dir: str = "snapshots"):
        self.database = database
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        
        self._snapshot_lock = threading.Lock()
        self._snapshot_interval = 300  # 5 minutes
        self._last_snapshot = time.time()
    
    def create_snapshot(self, compress: bool = True) -> str:
        """Create a point-in-time snapshot"""
        with self._snapshot_lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_file = self.snapshot_dir / f"snapshot_{timestamp}.db"
            
            # Create snapshot data
            snapshot_data = {
                'timestamp': timestamp,
                'data': dict(self.database),  # Deep copy
                'metadata': {
                    'version': '1.0',
                    'record_count': len(self.database),
                    'snapshot_type': 'full'
                }
            }
            
            if compress:
                snapshot_file = snapshot_file.with_suffix('.db.gz')
                with gzip.open(snapshot_file, 'wb') as f:
                    pickle.dump(snapshot_data, f)
            else:
                with open(snapshot_file, 'wb') as f:
                    pickle.dump(snapshot_data, f)
            
            self._last_snapshot = time.time()
            return str(snapshot_file)
    
    def restore_from_snapshot(self, snapshot_file: str) -> bool:
        """Restore database from snapshot"""
        try:
            snapshot_path = Path(snapshot_file)
            
            if snapshot_path.suffix == '.gz':
                with gzip.open(snapshot_path, 'rb') as f:
                    snapshot_data = pickle.load(f)
            else:
                with open(snapshot_path, 'rb') as f:
                    snapshot_data = pickle.load(f)
            
            # Restore data
            self.database.clear()
            self.database.update(snapshot_data['data'])
            
            print(f"Restored {len(self.database)} records from snapshot")
            print(f"Snapshot timestamp: {snapshot_data['timestamp']}")
            
            return True
            
        except Exception as e:
            print(f"Failed to restore snapshot: {e}")
            return False
    
    def auto_snapshot(self) -> Optional[str]:
        """Automatically create snapshot if interval elapsed"""
        if time.time() - self._last_snapshot >= self._snapshot_interval:
            return self.create_snapshot()
        return None
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots"""
        snapshots = []
        
        for snapshot_file in self.snapshot_dir.glob("snapshot_*.db*"):
            try:
                if snapshot_file.suffix == '.gz':
                    with gzip.open(snapshot_file, 'rb') as f:
                        metadata = pickle.load(f)['metadata']
                else:
                    with open(snapshot_file, 'rb') as f:
                        metadata = pickle.load(f)['metadata']
                
                snapshots.append({
                    'file': str(snapshot_file),
                    'size_mb': snapshot_file.stat().st_size / 1024 / 1024,
                    'created': datetime.fromtimestamp(snapshot_file.stat().st_mtime),
                    'record_count': metadata.get('record_count', 0)
                })
                
            except Exception as e:
                print(f"Error reading snapshot {snapshot_file}: {e}")
        
        return sorted(snapshots, key=lambda x: x['created'], reverse=True)

# Example usage
database = {}
snapshot_manager = SnapshotManager(database)

# Add some data
for i in range(10000):
    database[f"key_{i}"] = f"value_{i}"

# Create snapshot
snapshot_file = snapshot_manager.create_snapshot()
print(f"Created snapshot: {snapshot_file}")

# List snapshots
snapshots = snapshot_manager.list_snapshots()
for snap in snapshots:
    print(f"Snapshot: {snap['file']}, Records: {snap['record_count']}, Size: {snap['size_mb']:.2f}MB")
```

### 2. Write-Ahead Logging (WAL)

```python
import os
import threading
import struct
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

class WALOperation(Enum):
    INSERT = 1
    UPDATE = 2
    DELETE = 3
    TRANSACTION_BEGIN = 4
    TRANSACTION_COMMIT = 5
    TRANSACTION_ROLLBACK = 6

@dataclass
class WALEntry:
    """Write-Ahead Log entry"""
    operation: WALOperation
    transaction_id: int
    key: str
    value: Optional[Any] = None
    timestamp: float = 0.0
    checksum: int = 0

class WriteAheadLog:
    """Write-Ahead Logging implementation"""
    
    def __init__(self, log_file: str = "database.wal"):
        self.log_file = log_file
        self._log_lock = threading.Lock()
        self._transaction_counter = 0
        self._active_transactions: Dict[int, List[WALEntry]] = {}
        
        # Open log file for append
        self._log_handle = open(log_file, 'ab')
    
    def begin_transaction(self) -> int:
        """Begin a new transaction"""
        with self._log_lock:
            self._transaction_counter += 1
            transaction_id = self._transaction_counter
            self._active_transactions[transaction_id] = []
            
            # Log transaction begin
            entry = WALEntry(
                operation=WALOperation.TRANSACTION_BEGIN,
                transaction_id=transaction_id,
                key="",
                timestamp=time.time()
            )
            self._write_entry(entry)
            
            return transaction_id
    
    def log_operation(self, transaction_id: int, operation: WALOperation, 
                     key: str, value: Any = None):
        """Log a database operation"""
        entry = WALEntry(
            operation=operation,
            transaction_id=transaction_id,
            key=key,
            value=value,
            timestamp=time.time()
        )
        
        with self._log_lock:
            if transaction_id in self._active_transactions:
                self._active_transactions[transaction_id].append(entry)
                self._write_entry(entry)
    
    def commit_transaction(self, transaction_id: int):
        """Commit a transaction"""
        with self._log_lock:
            if transaction_id not in self._active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            # Log commit
            entry = WALEntry(
                operation=WALOperation.TRANSACTION_COMMIT,
                transaction_id=transaction_id,
                key="",
                timestamp=time.time()
            )
            self._write_entry(entry)
            
            # Force flush to disk
            self._log_handle.flush()
            os.fsync(self._log_handle.fileno())
            
            # Clean up
            del self._active_transactions[transaction_id]
    
    def rollback_transaction(self, transaction_id: int):
        """Rollback a transaction"""
        with self._log_lock:
            if transaction_id not in self._active_transactions:
                return
            
            # Log rollback
            entry = WALEntry(
                operation=WALOperation.TRANSACTION_ROLLBACK,
                transaction_id=transaction_id,
                key="",
                timestamp=time.time()
            )
            self._write_entry(entry)
            
            # Clean up
            del self._active_transactions[transaction_id]
    
    def _write_entry(self, entry: WALEntry):
        """Write entry to log file"""
        # Serialize entry (simplified)
        data = {
            'op': entry.operation.value,
            'txn': entry.transaction_id,
            'key': entry.key,
            'value': entry.value,
            'ts': entry.timestamp
        }
        
        serialized = json.dumps(data).encode('utf-8')
        entry_size = len(serialized)
        
        # Write entry size + entry data
        self._log_handle.write(struct.pack('I', entry_size))
        self._log_handle.write(serialized)
    
    def replay_log(self) -> Dict[str, Any]:
        """Replay log to rebuild database state"""
        database = {}
        
        with open(self.log_file, 'rb') as f:
            while True:
                try:
                    # Read entry size
                    size_data = f.read(4)
                    if len(size_data) < 4:
                        break
                    
                    entry_size = struct.unpack('I', size_data)[0]
                    
                    # Read entry data
                    entry_data = f.read(entry_size)
                    if len(entry_data) < entry_size:
                        break
                    
                    # Deserialize entry
                    data = json.loads(entry_data.decode('utf-8'))
                    operation = WALOperation(data['op'])
                    
                    # Apply operation
                    if operation == WALOperation.INSERT:
                        database[data['key']] = data['value']
                    elif operation == WALOperation.UPDATE:
                        database[data['key']] = data['value']
                    elif operation == WALOperation.DELETE:
                        database.pop(data['key'], None)
                    
                except Exception as e:
                    print(f"Error replaying log entry: {e}")
                    break
        
        return database
    
    def checkpoint(self, database: Dict[str, Any]):
        """Create checkpoint and truncate log"""
        # Create snapshot
        snapshot_manager = SnapshotManager(database)
        snapshot_file = snapshot_manager.create_snapshot()
        
        # Truncate log file
        with self._log_lock:
            self._log_handle.close()
            self._log_handle = open(self.log_file, 'wb')
        
        print(f"Checkpoint created: {snapshot_file}")
        print("WAL truncated")

# Example usage with WAL
class TransactionalDatabase:
    """Database with transaction support using WAL"""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._wal = WriteAheadLog()
    
    def begin_transaction(self) -> int:
        return self._wal.begin_transaction()
    
    def put(self, transaction_id: int, key: str, value: Any):
        self._wal.log_operation(transaction_id, WALOperation.INSERT, key, value)
        self._data[key] = value
    
    def delete(self, transaction_id: int, key: str):
        self._wal.log_operation(transaction_id, WALOperation.DELETE, key)
        self._data.pop(key, None)
    
    def commit(self, transaction_id: int):
        self._wal.commit_transaction(transaction_id)
    
    def rollback(self, transaction_id: int):
        self._wal.rollback_transaction(transaction_id)
        # Note: In a real implementation, you'd need to undo the changes
    
    def recover(self):
        """Recover database from WAL"""
        self._data = self._wal.replay_log()
        print(f"Recovered {len(self._data)} records from WAL")

# Example transaction
db = TransactionalDatabase()
txn = db.begin_transaction()

try:
    db.put(txn, "user:1", {"name": "Alice", "age": 30})
    db.put(txn, "user:2", {"name": "Bob", "age": 25})
    db.commit(txn)
    print("Transaction committed successfully")
except Exception as e:
    db.rollback(txn)
    print(f"Transaction rolled back: {e}")
```
## Practical Implementation: Complete In-Memory Database

Let's build a comprehensive in-memory database that demonstrates all the concepts covered:

```python
import threading
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"

@dataclass
class QueryResult:
    """Query execution result"""
    success: bool
    data: Optional[Any] = None
    affected_rows: int = 0
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None

@dataclass
class IndexEntry:
    """Index entry for fast lookups"""
    key: str
    row_id: str
    value: Any

class InMemoryDatabase:
    """Complete in-memory database implementation"""
    
    def __init__(self, name: str = "inmemory_db"):
        self.name = name
        self._tables: Dict[str, Dict[str, Any]] = {}
        self._indexes: Dict[str, Dict[str, List[IndexEntry]]] = {}
        self._schema: Dict[str, Dict[str, type]] = {}
        
        # Concurrency control
        self._table_locks: Dict[str, threading.RWLock] = {}
        self._global_lock = threading.Lock()
        
        # Memory management
        self._memory_pool = MemoryPool(block_size=4096, pool_size=10000)
        self._gc = InMemoryGarbageCollector()
        
        # Persistence
        self._wal = WriteAheadLog(f"{name}.wal")
        self._snapshot_manager = None
        
        # Statistics
        self._stats = {
            'queries_executed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_memory_used': 0
        }
        
        logger.info(f"In-memory database '{name}' initialized")
    
    def create_table(self, table_name: str, schema: Dict[str, type]) -> bool:
        """Create a new table with schema"""
        with self._global_lock:
            if table_name in self._tables:
                logger.error(f"Table '{table_name}' already exists")
                return False
            
            self._tables[table_name] = {}
            self._indexes[table_name] = {}
            self._schema[table_name] = schema
            self._table_locks[table_name] = threading.RWLock()
            
            # Initialize snapshot manager for this table
            if not self._snapshot_manager:
                self._snapshot_manager = SnapshotManager(self._tables)
            
            logger.info(f"Created table '{table_name}' with schema: {schema}")
            return True
    
    def create_index(self, table_name: str, column: str) -> bool:
        """Create an index on a column"""
        if table_name not in self._tables:
            logger.error(f"Table '{table_name}' does not exist")
            return False
        
        index_key = f"{table_name}.{column}"
        self._indexes[table_name][column] = {}
        
        # Build index for existing data
        with self._table_locks[table_name].read_lock():
            for row_id, row_data in self._tables[table_name].items():
                if column in row_data:
                    value = row_data[column]
                    if value not in self._indexes[table_name][column]:
                        self._indexes[table_name][column][value] = []
                    self._indexes[table_name][column][value].append(
                        IndexEntry(column, row_id, value)
                    )
        
        logger.info(f"Created index on {table_name}.{column}")
        return True
    
    def insert(self, table_name: str, data: Dict[str, Any]) -> QueryResult:
        """Insert data into table"""
        start_time = time.perf_counter()
        
        if table_name not in self._tables:
            return QueryResult(
                success=False,
                error_message=f"Table '{table_name}' does not exist"
            )
        
        # Validate schema
        schema = self._schema[table_name]
        for column, expected_type in schema.items():
            if column in data and not isinstance(data[column], expected_type):
                return QueryResult(
                    success=False,
                    error_message=f"Type mismatch for column '{column}'"
                )
        
        # Generate row ID
        row_id = hashlib.md5(
            f"{table_name}_{time.time()}_{len(self._tables[table_name])}".encode()
        ).hexdigest()[:16]
        
        # Begin transaction
        transaction_id = self._wal.begin_transaction()
        
        try:
            with self._table_locks[table_name].write_lock():
                # Insert data
                self._tables[table_name][row_id] = data.copy()
                
                # Update indexes
                for column, value in data.items():
                    if column in self._indexes[table_name]:
                        if value not in self._indexes[table_name][column]:
                            self._indexes[table_name][column][value] = []
                        self._indexes[table_name][column][value].append(
                            IndexEntry(column, row_id, value)
                        )
                
                # Log operation
                self._wal.log_operation(transaction_id, WALOperation.INSERT, row_id, data)
                self._wal.commit_transaction(transaction_id)
                
                self._stats['queries_executed'] += 1
                execution_time = (time.perf_counter() - start_time) * 1000
                
                return QueryResult(
                    success=True,
                    data={"row_id": row_id},
                    affected_rows=1,
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            self._wal.rollback_transaction(transaction_id)
            return QueryResult(
                success=False,
                error_message=str(e)
            )
    
    def select(self, table_name: str, where_clause: Optional[Dict[str, Any]] = None,
              limit: Optional[int] = None) -> QueryResult:
        """Select data from table"""
        start_time = time.perf_counter()
        
        if table_name not in self._tables:
            return QueryResult(
                success=False,
                error_message=f"Table '{table_name}' does not exist"
            )
        
        try:
            with self._table_locks[table_name].read_lock():
                results = []
                
                if where_clause:
                    # Try to use index
                    for column, value in where_clause.items():
                        if column in self._indexes[table_name]:
                            # Use index for fast lookup
                            index_entries = self._indexes[table_name][column].get(value, [])
                            for entry in index_entries:
                                row_data = self._tables[table_name][entry.row_id]
                                if self._matches_where_clause(row_data, where_clause):
                                    results.append({"row_id": entry.row_id, **row_data})
                            self._stats['cache_hits'] += 1
                            break
                    else:
                        # Full table scan
                        self._stats['cache_misses'] += 1
                        for row_id, row_data in self._tables[table_name].items():
                            if self._matches_where_clause(row_data, where_clause):
                                results.append({"row_id": row_id, **row_data})
                else:
                    # Select all
                    for row_id, row_data in self._tables[table_name].items():
                        results.append({"row_id": row_id, **row_data})
                
                # Apply limit
                if limit:
                    results = results[:limit]
                
                self._stats['queries_executed'] += 1
                execution_time = (time.perf_counter() - start_time) * 1000
                
                return QueryResult(
                    success=True,
                    data=results,
                    affected_rows=len(results),
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            return QueryResult(
                success=False,
                error_message=str(e)
            )
    
    def update(self, table_name: str, set_clause: Dict[str, Any],
              where_clause: Dict[str, Any]) -> QueryResult:
        """Update data in table"""
        start_time = time.perf_counter()
        
        if table_name not in self._tables:
            return QueryResult(
                success=False,
                error_message=f"Table '{table_name}' does not exist"
            )
        
        transaction_id = self._wal.begin_transaction()
        updated_count = 0
        
        try:
            with self._table_locks[table_name].write_lock():
                for row_id, row_data in self._tables[table_name].items():
                    if self._matches_where_clause(row_data, where_clause):
                        # Update row
                        old_data = row_data.copy()
                        row_data.update(set_clause)
                        
                        # Update indexes
                        self._update_indexes(table_name, row_id, old_data, row_data)
                        
                        # Log operation
                        self._wal.log_operation(transaction_id, WALOperation.UPDATE, 
                                              row_id, row_data)
                        updated_count += 1
                
                self._wal.commit_transaction(transaction_id)
                self._stats['queries_executed'] += 1
                execution_time = (time.perf_counter() - start_time) * 1000
                
                return QueryResult(
                    success=True,
                    affected_rows=updated_count,
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            self._wal.rollback_transaction(transaction_id)
            return QueryResult(
                success=False,
                error_message=str(e)
            )
    
    def delete(self, table_name: str, where_clause: Dict[str, Any]) -> QueryResult:
        """Delete data from table"""
        start_time = time.perf_counter()
        
        if table_name not in self._tables:
            return QueryResult(
                success=False,
                error_message=f"Table '{table_name}' does not exist"
            )
        
        transaction_id = self._wal.begin_transaction()
        deleted_count = 0
        
        try:
            with self._table_locks[table_name].write_lock():
                rows_to_delete = []
                
                for row_id, row_data in self._tables[table_name].items():
                    if self._matches_where_clause(row_data, where_clause):
                        rows_to_delete.append(row_id)
                
                for row_id in rows_to_delete:
                    row_data = self._tables[table_name][row_id]
                    
                    # Remove from indexes
                    self._remove_from_indexes(table_name, row_id, row_data)
                    
                    # Delete row
                    del self._tables[table_name][row_id]
                    
                    # Log operation
                    self._wal.log_operation(transaction_id, WALOperation.DELETE, row_id)
                    deleted_count += 1
                
                self._wal.commit_transaction(transaction_id)
                self._stats['queries_executed'] += 1
                execution_time = (time.perf_counter() - start_time) * 1000
                
                return QueryResult(
                    success=True,
                    affected_rows=deleted_count,
                    execution_time_ms=execution_time
                )
        
        except Exception as e:
            self._wal.rollback_transaction(transaction_id)
            return QueryResult(
                success=False,
                error_message=str(e)
            )
    
    def _matches_where_clause(self, row_data: Dict[str, Any], 
                             where_clause: Dict[str, Any]) -> bool:
        """Check if row matches where clause"""
        for column, value in where_clause.items():
            if column not in row_data or row_data[column] != value:
                return False
        return True
    
    def _update_indexes(self, table_name: str, row_id: str,
                       old_data: Dict[str, Any], new_data: Dict[str, Any]):
        """Update indexes after row modification"""
        for column in self._indexes[table_name]:
            old_value = old_data.get(column)
            new_value = new_data.get(column)
            
            if old_value != new_value:
                # Remove old index entry
                if old_value in self._indexes[table_name][column]:
                    entries = self._indexes[table_name][column][old_value]
                    self._indexes[table_name][column][old_value] = [
                        e for e in entries if e.row_id != row_id
                    ]
                
                # Add new index entry
                if new_value not in self._indexes[table_name][column]:
                    self._indexes[table_name][column][new_value] = []
                self._indexes[table_name][column][new_value].append(
                    IndexEntry(column, row_id, new_value)
                )
    
    def _remove_from_indexes(self, table_name: str, row_id: str, 
                           row_data: Dict[str, Any]):
        """Remove row from all indexes"""
        for column, value in row_data.items():
            if column in self._indexes[table_name]:
                if value in self._indexes[table_name][column]:
                    entries = self._indexes[table_name][column][value]
                    self._indexes[table_name][column][value] = [
                        e for e in entries if e.row_id != row_id
                    ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        total_rows = sum(len(table) for table in self._tables.values())
        total_tables = len(self._tables)
        
        return {
            **self._stats,
            'total_tables': total_tables,
            'total_rows': total_rows,
            'cache_hit_ratio': (
                self._stats['cache_hits'] / 
                (self._stats['cache_hits'] + self._stats['cache_misses'])
                if (self._stats['cache_hits'] + self._stats['cache_misses']) > 0 else 0
            ),
            'memory_pool_stats': self._memory_pool.get_stats(),
            'gc_stats': self._gc.get_memory_stats()
        }
    
    def create_snapshot(self) -> str:
        """Create database snapshot"""
        if self._snapshot_manager:
            return self._snapshot_manager.create_snapshot()
        else:
            raise RuntimeError("No snapshot manager available")
    
    def recover_from_wal(self):
        """Recover database from write-ahead log"""
        recovered_data = self._wal.replay_log()
        
        with self._global_lock:
            # Clear existing data
            self._tables.clear()
            self._indexes.clear()
            
            # Restore from WAL
            for table_name, table_data in recovered_data.items():
                self._tables[table_name] = table_data
                self._table_locks[table_name] = threading.RWLock()
        
        logger.info(f"Recovered {len(self._tables)} tables from WAL")

# Usage example
def demonstrate_inmemory_database():
    """Demonstrate the complete in-memory database"""
    
    # Create database
    db = InMemoryDatabase("demo_db")
    
    # Create table
    user_schema = {
        'name': str,
        'age': int,
        'email': str,
        'created_at': float
    }
    db.create_table('users', user_schema)
    
    # Create index
    db.create_index('users', 'email')
    
    # Insert data
    users_data = [
        {'name': 'Alice', 'age': 30, 'email': 'alice@example.com', 'created_at': time.time()},
        {'name': 'Bob', 'age': 25, 'email': 'bob@example.com', 'created_at': time.time()},
        {'name': 'Charlie', 'age': 35, 'email': 'charlie@example.com', 'created_at': time.time()},
    ]
    
    for user in users_data:
        result = db.insert('users', user)
        print(f"Insert result: {result.success}, Row ID: {result.data}")
    
    # Select all users
    result = db.select('users')
    print(f"\nAll users ({result.affected_rows} found):")
    for user in result.data:
        print(f"  {user}")
    
    # Select with where clause (using index)
    result = db.select('users', where_clause={'email': 'bob@example.com'})
    print(f"\nUser with email 'bob@example.com': {result.data}")
    
    # Update user
    result = db.update('users', 
                      set_clause={'age': 26}, 
                      where_clause={'email': 'bob@example.com'})
    print(f"\nUpdate result: {result.affected_rows} rows updated")
    
    # Delete user
    result = db.delete('users', where_clause={'name': 'Charlie'})
    print(f"\nDelete result: {result.affected_rows} rows deleted")
    
    # Show final state
    result = db.select('users')
    print(f"\nFinal users ({result.affected_rows} remaining):")
    for user in result.data:
        print(f"  {user}")
    
    # Show statistics
    stats = db.get_stats()
    print(f"\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create snapshot
    snapshot_file = db.create_snapshot()
    print(f"\nSnapshot created: {snapshot_file}")

if __name__ == "__main__":
    demonstrate_inmemory_database()
```

## Performance Benchmarking

```python
import random
import string
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

class DatabaseBenchmark:
    """Benchmark suite for in-memory database performance"""
    
    def __init__(self, db: InMemoryDatabase):
        self.db = db
        self.results = {}
    
    def generate_test_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate test data for benchmarking"""
        data = []
        for i in range(count):
            data.append({
                'id': i,
                'name': ''.join(random.choices(string.ascii_letters, k=10)),
                'email': f"user{i}@example.com",
                'age': random.randint(18, 80),
                'score': random.uniform(0, 100),
                'created_at': time.time() + i
            })
        return data
    
    def benchmark_inserts(self, record_count: int) -> Dict[str, float]:
        """Benchmark insert performance"""
        print(f"Benchmarking {record_count} inserts...")
        
        # Create table
        schema = {'id': int, 'name': str, 'email': str, 'age': int, 
                 'score': float, 'created_at': float}
        self.db.create_table('benchmark_table', schema)
        
        # Generate data
        test_data = self.generate_test_data(record_count)
        
        # Benchmark
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        for record in test_data:
            self.db.insert('benchmark_table', record)
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        return {
            'total_time': execution_time,
            'records_per_second': record_count / execution_time,
            'memory_used_mb': memory_used,
            'memory_per_record_kb': (memory_used * 1024) / record_count
        }
    
    def benchmark_selects(self, query_count: int) -> Dict[str, float]:
        """Benchmark select performance"""
        print(f"Benchmarking {query_count} selects...")
        
        start_time = time.perf_counter()
        
        for _ in range(query_count):
            # Random select
            random_id = random.randint(0, 9999)
            self.db.select('benchmark_table', where_clause={'id': random_id})
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        return {
            'total_time': execution_time,
            'queries_per_second': query_count / execution_time
        }
    
    def benchmark_concurrent_access(self, thread_count: int, 
                                  operations_per_thread: int) -> Dict[str, float]:
        """Benchmark concurrent access performance"""
        print(f"Benchmarking concurrent access: {thread_count} threads, "
              f"{operations_per_thread} ops/thread...")
        
        def worker_thread(thread_id: int):
            for i in range(operations_per_thread):
                # Mix of operations
                operation = random.choice(['select', 'insert', 'update'])
                
                if operation == 'select':
                    random_id = random.randint(0, 9999)
                    self.db.select('benchmark_table', where_clause={'id': random_id})
                
                elif operation == 'insert':
                    record = {
                        'id': thread_id * 10000 + i,
                        'name': f"user_{thread_id}_{i}",
                        'email': f"user{thread_id}_{i}@example.com",
                        'age': random.randint(18, 80),
                        'score': random.uniform(0, 100),
                        'created_at': time.time()
                    }
                    self.db.insert('benchmark_table', record)
                
                elif operation == 'update':
                    random_id = random.randint(0, 9999)
                    self.db.update('benchmark_table',
                                 set_clause={'score': random.uniform(0, 100)},
                                 where_clause={'id': random_id})
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = []
            for thread_id in range(thread_count):
                future = executor.submit(worker_thread, thread_id)
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                future.result()
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        total_operations = thread_count * operations_per_thread
        
        return {
            'total_time': execution_time,
            'operations_per_second': total_operations / execution_time,
            'threads': thread_count
        }
    
    def run_complete_benchmark(self):
        """Run complete benchmark suite"""
        print("=== In-Memory Database Benchmark Suite ===\n")
        
        # Insert benchmark
        insert_results = self.benchmark_inserts(10000)
        print(f"Insert Performance:")
        print(f"  Records/sec: {insert_results['records_per_second']:.2f}")
        print(f"  Memory/record: {insert_results['memory_per_record_kb']:.2f} KB")
        print(f"  Total memory: {insert_results['memory_used_mb']:.2f} MB\n")
        
        # Create index for better select performance
        self.db.create_index('benchmark_table', 'id')
        
        # Select benchmark
        select_results = self.benchmark_selects(5000)
        print(f"Select Performance:")
        print(f"  Queries/sec: {select_results['queries_per_second']:.2f}\n")
        
        # Concurrent access benchmark
        concurrent_results = self.benchmark_concurrent_access(10, 100)
        print(f"Concurrent Access Performance:")
        print(f"  Operations/sec: {concurrent_results['operations_per_second']:.2f}")
        print(f"  Threads: {concurrent_results['threads']}\n")
        
        # Database statistics
        stats = self.db.get_stats()
        print(f"Final Database Statistics:")
        print(f"  Total queries: {stats['queries_executed']}")
        print(f"  Cache hit ratio: {stats['cache_hit_ratio']:.2%}")
        print(f"  Total tables: {stats['total_tables']}")
        print(f"  Total rows: {stats['total_rows']}")

# Run benchmark
if __name__ == "__main__":
    db = InMemoryDatabase("benchmark_db")
    benchmark = DatabaseBenchmark(db)
    benchmark.run_complete_benchmark()
```

## Study Materials and Resources

### Recommended Reading

#### Books
- **"Database System Concepts" by Silberschatz, Galvin, and Gagne** - Chapters 11-12 (Storage and Indexing)
- **"Designing Data-Intensive Applications" by Martin Kleppmann** - Chapter 3 (Storage and Retrieval)
- **"High Performance MySQL" by Baron Schwartz** - Chapter 5 (Indexing for High Performance)
- **"Redis in Action" by Josiah L. Carlson** - Complete guide to Redis implementation

#### Research Papers
- "Main Memory Database Systems: An Overview" (Garcia-Molina & Salem, 1992)
- "The Case for RAMClouds: Scalable High-Performance Storage Entirely in DRAM" (Ousterhout et al., 2010)
- "SAP HANA Database: Data Management for Modern Business Applications" (Färber et al., 2012)

### Video Resources
- **"In-Memory Databases Explained"** - Database Engineering YouTube Channel
- **"Redis Deep Dive"** - Redis University Course
- **"SAP HANA Architecture"** - SAP Technology Overview
- **"Building High-Performance Data Systems"** - Strange Loop Conference Talks

### Hands-on Labs

#### Lab 1: Basic Implementation
```python
# TODO: Implement a simple key-value store with TTL support
class TTLKeyValueStore:
    def __init__(self):
        pass
    
    def put(self, key: str, value: Any, ttl_seconds: int = 3600):
        # Your implementation here
        pass
    
    def get(self, key: str) -> Optional[Any]:
        # Your implementation here
        pass
    
    def delete(self, key: str) -> bool:
        # Your implementation here
        pass
```
