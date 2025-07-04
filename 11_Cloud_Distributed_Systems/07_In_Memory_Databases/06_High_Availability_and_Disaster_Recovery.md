# High Availability and Disaster Recovery for In-Memory Databases

*Duration: 2-3 weeks*

## Description
Master comprehensive strategies for ensuring high availability, fault tolerance, and disaster recovery in in-memory database systems. Learn to design resilient architectures that can withstand failures while maintaining data consistency and minimizing downtime.

## Learning Objectives

By the end of this section, you should be able to:
- Design and implement various replication strategies for in-memory databases
- Configure clustering and automatic failover mechanisms
- Implement comprehensive backup and recovery procedures
- Build highly available in-memory database solutions
- Calculate and optimize system availability metrics
- Handle disaster recovery scenarios effectively
- Monitor and troubleshoot HA systems

## Core Concepts Overview

### High Availability vs Disaster Recovery

**High Availability (HA):**
- Minimizes downtime during planned and unplanned outages
- Typically measured in "nines" (99.9%, 99.99%, 99.999%)
- Focus on redundancy and automatic failover
- Usually handles single points of failure

**Disaster Recovery (DR):**
- Focuses on recovery from catastrophic events
- Includes data backup and restoration procedures
- May involve geographic distribution
- Typically measured in Recovery Time Objective (RTO) and Recovery Point Objective (RPO)

### Availability Metrics

| Availability | Downtime/Year | Downtime/Month | Use Case |
|--------------|---------------|----------------|----------|
| 99% | 3.65 days | 7.2 hours | Basic systems |
| 99.9% | 8.76 hours | 43.2 minutes | Standard business |
| 99.99% | 52.6 minutes | 4.32 minutes | Critical business |
| 99.999% | 5.26 minutes | 25.9 seconds | Mission critical |
| 99.9999% | 31.5 seconds | 2.59 seconds | Ultra critical |

## Replication Strategies

### 1. Master-Slave Replication

Master-slave replication is the most common pattern where one master node handles all writes, and one or more slave nodes replicate the data for read scalability and failover.

#### Redis Master-Slave Configuration

**Master Configuration (redis-master.conf):**
```bash
# Master Redis configuration
port 6379
bind 0.0.0.0
requirepass "master_password"

# Persistence settings
save 900 1
save 300 10
save 60 10000

# Replication settings
replica-serve-stale-data yes
replica-read-only yes
repl-diskless-sync no
repl-diskless-sync-delay 5

# Security
protected-mode yes
```

**Slave Configuration (redis-slave.conf):**
```bash
# Slave Redis configuration
port 6380
bind 0.0.0.0

# Master connection
replicaof 127.0.0.1 6379
masterauth "master_password"
requirepass "slave_password"

# Slave-specific settings
replica-serve-stale-data yes
replica-read-only yes
replica-priority 100

# Persistence (optional for slaves)
save ""
```

**Starting Master-Slave Setup:**
```bash
# Start master
redis-server /path/to/redis-master.conf

# Start slave
redis-server /path/to/redis-slave.conf

# Verify replication
redis-cli -p 6379 info replication
redis-cli -p 6380 info replication
```

**Python Implementation for Master-Slave Handling:**
```python
import redis
import time
import logging
from typing import Optional, List

class RedisHA:
    def __init__(self, master_config: dict, slave_configs: List[dict]):
        self.master_config = master_config
        self.slave_configs = slave_configs
        self.master_client = None
        self.slave_clients = []
        self.current_master = None
        self.setup_connections()
    
    def setup_connections(self):
        """Initialize connections to master and slaves"""
        try:
            # Connect to master
            self.master_client = redis.Redis(**self.master_config)
            self.master_client.ping()
            self.current_master = self.master_config
            logging.info("Connected to master successfully")
        except Exception as e:
            logging.error(f"Failed to connect to master: {e}")
        
        # Connect to slaves
        for slave_config in self.slave_configs:
            try:
                slave_client = redis.Redis(**slave_config)
                slave_client.ping()
                self.slave_clients.append(slave_client)
                logging.info(f"Connected to slave {slave_config['host']}:{slave_config['port']}")
            except Exception as e:
                logging.error(f"Failed to connect to slave: {e}")
    
    def write_data(self, key: str, value: str) -> bool:
        """Write data to master"""
        if not self.master_client:
            return False
        
        try:
            self.master_client.set(key, value)
            return True
        except Exception as e:
            logging.error(f"Write failed: {e}")
            return False
    
    def read_data(self, key: str, prefer_slave: bool = True) -> Optional[str]:
        """Read data with load balancing"""
        clients_to_try = []
        
        if prefer_slave and self.slave_clients:
            clients_to_try.extend(self.slave_clients)
        
        if self.master_client:
            clients_to_try.append(self.master_client)
        
        for client in clients_to_try:
            try:
                value = client.get(key)
                return value.decode('utf-8') if value else None
            except Exception as e:
                logging.error(f"Read failed from client: {e}")
                continue
        
        return None
    
    def check_replication_lag(self) -> dict:
        """Monitor replication lag"""
        if not self.master_client:
            return {}
        
        try:
            master_info = self.master_client.info('replication')
            lag_info = {}
            
            for i, slave_client in enumerate(self.slave_clients):
                try:
                    slave_info = slave_client.info('replication')
                    master_offset = master_info.get('master_repl_offset', 0)
                    slave_offset = slave_info.get('slave_repl_offset', 0)
                    lag = master_offset - slave_offset
                    lag_info[f'slave_{i}'] = {
                        'lag_bytes': lag,
                        'master_offset': master_offset,
                        'slave_offset': slave_offset
                    }
                except Exception as e:
                    lag_info[f'slave_{i}'] = {'error': str(e)}
            
            return lag_info
        except Exception as e:
            logging.error(f"Failed to check replication lag: {e}")
            return {}

# Usage example
if __name__ == "__main__":
    master_config = {
        'host': 'localhost',
        'port': 6379,
        'password': 'master_password',
        'decode_responses': False
    }
    
    slave_configs = [
        {
            'host': 'localhost',
            'port': 6380,
            'password': 'slave_password',
            'decode_responses': False
        }
    ]
    
    redis_ha = RedisHA(master_config, slave_configs)
    
    # Test write/read
    redis_ha.write_data("test_key", "test_value")
    value = redis_ha.read_data("test_key")
    print(f"Retrieved value: {value}")
    
    # Check replication lag
    lag_info = redis_ha.check_replication_lag()
    print(f"Replication lag: {lag_info}")
```

### 2. Master-Master Replication

Master-master replication allows multiple nodes to accept writes, providing better write scalability but requiring conflict resolution.

#### Redis Cluster Configuration

**Setting up Redis Cluster:**
```bash
# Create cluster nodes (6 nodes for HA - 3 masters, 3 slaves)
mkdir -p /opt/redis-cluster/{7000,7001,7002,7003,7004,7005}

# Node configuration template (redis-7000.conf)
port 7000
cluster-enabled yes
cluster-config-file nodes-7000.conf
cluster-node-timeout 5000
appendonly yes
bind 0.0.0.0

# Start all nodes
for port in {7000..7005}; do
    redis-server /opt/redis-cluster/$port/redis-$port.conf &
done

# Create cluster
redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
    127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
    --cluster-replicas 1
```

**Python Redis Cluster Client:**
```python
import rediscluster
import random
import time
from typing import Dict, Any

class RedisClusterHA:
    def __init__(self, startup_nodes: list):
        self.startup_nodes = startup_nodes
        self.client = None
        self.connect()
    
    def connect(self):
        """Connect to Redis cluster"""
        try:
            self.client = rediscluster.RedisCluster(
                startup_nodes=self.startup_nodes,
                decode_responses=True,
                skip_full_coverage_check=True
            )
            # Test connection
            self.client.ping()
            print("Connected to Redis cluster successfully")
        except Exception as e:
            print(f"Failed to connect to cluster: {e}")
            raise
    
    def write_with_retry(self, key: str, value: Any, max_retries: int = 3) -> bool:
        """Write data with automatic retry on failure"""
        for attempt in range(max_retries):
            try:
                self.client.set(key, value)
                return True
            except Exception as e:
                print(f"Write attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    return False
        return False
    
    def read_with_retry(self, key: str, max_retries: int = 3) -> Any:
        """Read data with automatic retry on failure"""
        for attempt in range(max_retries):
            try:
                return self.client.get(key)
            except Exception as e:
                print(f"Read attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
        return None
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster health information"""
        try:
            nodes = self.client.cluster_nodes()
            cluster_info = {
                'total_nodes': len(nodes),
                'master_nodes': 0,
                'slave_nodes': 0,
                'failed_nodes': 0,
                'nodes_detail': []
            }
            
            for node_id, node_info in nodes.items():
                if 'master' in node_info['flags']:
                    cluster_info['master_nodes'] += 1
                elif 'slave' in node_info['flags']:
                    cluster_info['slave_nodes'] += 1
                
                if 'fail' in node_info['flags']:
                    cluster_info['failed_nodes'] += 1
                
                cluster_info['nodes_detail'].append({
                    'id': node_id,
                    'host': node_info['host'],
                    'port': node_info['port'],
                    'flags': node_info['flags'],
                    'slots': node_info.get('slots', [])
                })
            
            return cluster_info
        except Exception as e:
            return {'error': str(e)}
    
    def monitor_cluster_health(self, interval: int = 10):
        """Monitor cluster health continuously"""
        while True:
            try:
                info = self.get_cluster_info()
                print(f"\n=== Cluster Health Report ===")
                print(f"Total nodes: {info.get('total_nodes', 0)}")
                print(f"Master nodes: {info.get('master_nodes', 0)}")
                print(f"Slave nodes: {info.get('slave_nodes', 0)}")
                print(f"Failed nodes: {info.get('failed_nodes', 0)}")
                
                if info.get('failed_nodes', 0) > 0:
                    print("⚠️  WARNING: Some nodes are failing!")
                
                time.sleep(interval)
            except KeyboardInterrupt:
                print("Health monitoring stopped")
                break
            except Exception as e:
                print(f"Health check failed: {e}")
                time.sleep(interval)

# Usage example
startup_nodes = [
    {"host": "127.0.0.1", "port": "7000"},
    {"host": "127.0.0.1", "port": "7001"},
    {"host": "127.0.0.1", "port": "7002"}
]

cluster = RedisClusterHA(startup_nodes)
cluster.write_with_retry("test:key", "test_value")
value = cluster.read_with_retry("test:key")
print(f"Value: {value}")

# Monitor cluster
# cluster.monitor_cluster_health()
```

### 3. Chain Replication

Chain replication provides strong consistency by organizing nodes in a chain where writes flow from head to tail.

#### Chain Replication Implementation

```python
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class NodeRole(Enum):
    HEAD = "head"
    MIDDLE = "middle"
    TAIL = "tail"

@dataclass
class ChainNode:
    node_id: str
    host: str
    port: int
    role: NodeRole
    next_node: Optional['ChainNode'] = None
    prev_node: Optional['ChainNode'] = None

class ChainReplicationNode:
    def __init__(self, node: ChainNode):
        self.node = node
        self.data_store: Dict[str, Any] = {}
        self.version_vector: Dict[str, int] = {}
        self.pending_writes: Dict[str, Any] = {}
    
    async def handle_write(self, key: str, value: Any, request_id: str) -> bool:
        """Handle write request based on node role"""
        if self.node.role == NodeRole.HEAD:
            # Head node receives client writes
            return await self._propagate_write(key, value, request_id)
        else:
            # Only head should receive client writes
            return False
    
    async def _propagate_write(self, key: str, value: Any, request_id: str) -> bool:
        """Propagate write down the chain"""
        try:
            # Store in pending writes
            self.pending_writes[request_id] = {'key': key, 'value': value}
            
            if self.node.next_node:
                # Forward to next node
                success = await self._send_to_next_node(key, value, request_id)
                if not success:
                    del self.pending_writes[request_id]
                    return False
            else:
                # This is the tail node, commit the write
                await self._commit_write(key, value, request_id)
            
            return True
        except Exception as e:
            print(f"Write propagation failed: {e}")
            return False
    
    async def _send_to_next_node(self, key: str, value: Any, request_id: str) -> bool:
        """Send write to next node in chain"""
        if not self.node.next_node:
            return True
        
        try:
            url = f"http://{self.node.next_node.host}:{self.node.next_node.port}/write"
            payload = {
                'key': key,
                'value': value,
                'request_id': request_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Failed to send to next node: {e}")
            return False
    
    async def _commit_write(self, key: str, value: Any, request_id: str):
        """Commit write and send acknowledgment back up the chain"""
        # Apply write to local store
        self.data_store[key] = value
        self.version_vector[key] = self.version_vector.get(key, 0) + 1
        
        # Send ACK back up the chain
        await self._send_ack_upstream(request_id)
    
    async def _send_ack_upstream(self, request_id: str):
        """Send acknowledgment upstream"""
        if self.node.prev_node:
            try:
                url = f"http://{self.node.prev_node.host}:{self.node.prev_node.port}/ack"
                payload = {'request_id': request_id}
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        pass  # ACK sent
            except Exception as e:
                print(f"Failed to send ACK upstream: {e}")
        else:
            # This is the head node, notify client
            print(f"Write {request_id} committed successfully")
    
    async def handle_read(self, key: str) -> Optional[Any]:
        """Handle read request (only tail serves reads for consistency)"""
        if self.node.role == NodeRole.TAIL:
            return self.data_store.get(key)
        else:
            # Forward read to tail
            if self.node.role == NodeRole.HEAD:
                tail_node = self._find_tail()
                if tail_node:
                    return await self._forward_read_to_tail(key, tail_node)
        return None
    
    def _find_tail(self) -> Optional[ChainNode]:
        """Find tail node from head"""
        current = self.node
        while current.next_node:
            current = current.next_node
        return current if current.role == NodeRole.TAIL else None
    
    async def _forward_read_to_tail(self, key: str, tail_node: ChainNode) -> Optional[Any]:
        """Forward read request to tail node"""
        try:
            url = f"http://{tail_node.host}:{tail_node.port}/read"
            payload = {'key': key}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('value')
        except Exception as e:
            print(f"Failed to forward read to tail: {e}")
        return None

# Chain setup example
async def setup_chain_replication():
    """Setup a 3-node chain replication system"""
    
    # Create nodes
    head_node = ChainNode("node1", "localhost", 8001, NodeRole.HEAD)
    middle_node = ChainNode("node2", "localhost", 8002, NodeRole.MIDDLE)
    tail_node = ChainNode("node3", "localhost", 8003, NodeRole.TAIL)
    
    # Link nodes
    head_node.next_node = middle_node
    middle_node.prev_node = head_node
    middle_node.next_node = tail_node
    tail_node.prev_node = middle_node
    
    # Create replication nodes
    head_repl = ChainReplicationNode(head_node)
    middle_repl = ChainReplicationNode(middle_node)
    tail_repl = ChainReplicationNode(tail_node)
    
    return head_repl, middle_repl, tail_repl

# Usage example
async def test_chain_replication():
    head, middle, tail = await setup_chain_replication()
    
    # Write data (only through head)
    success = await head.handle_write("user:1", {"name": "John", "age": 30}, "req_001")
    print(f"Write success: {success}")
    
    # Read data (from tail for consistency)
    value = await tail.handle_read("user:1")
    print(f"Read value: {value}")

# Run test
# asyncio.run(test_chain_replication())
```

## Clustering and Failover

### 1. Redis Sentinel for Automatic Failover

Redis Sentinel provides high availability for Redis through monitoring, notification, and automatic failover.

#### Sentinel Configuration

**Sentinel Configuration (sentinel.conf):**
```bash
# Sentinel configuration
port 26379
bind 0.0.0.0

# Monitor master
sentinel monitor mymaster 127.0.0.1 6379 2
sentinel auth-pass mymaster master_password
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 10000

# Notification scripts
sentinel notification-script mymaster /path/to/notify.sh
sentinel client-reconfig-script mymaster /path/to/reconfig.sh

# Logging
logfile "/var/log/redis/sentinel.log"
loglevel notice
```

**Starting Sentinel Cluster:**
```bash
# Start 3 sentinel instances for quorum
redis-sentinel /etc/redis/sentinel-26379.conf &
redis-sentinel /etc/redis/sentinel-26380.conf &
redis-sentinel /etc/redis/sentinel-26381.conf &

# Check sentinel status
redis-cli -p 26379 sentinel masters
redis-cli -p 26379 sentinel slaves mymaster
redis-cli -p 26379 sentinel sentinels mymaster
```

**Python Sentinel Client with Automatic Failover:**
```python
import redis
import redis.sentinel
import time
import threading
import logging
from typing import List, Optional, Dict, Any

class RedisSentinelHA:
    def __init__(self, sentinel_hosts: List[tuple], master_name: str, password: str = None):
        self.sentinel_hosts = sentinel_hosts
        self.master_name = master_name
        self.password = password
        self.sentinel = None
        self.master_client = None
        self.slave_clients = []
        self.failover_callbacks = []
        self.monitoring = False
        self.setup_sentinel()
    
    def setup_sentinel(self):
        """Initialize Sentinel connection"""
        try:
            self.sentinel = redis.sentinel.Sentinel(
                self.sentinel_hosts,
                password=self.password,
                socket_timeout=0.1
            )
            
            # Get initial master connection
            self.master_client = self.sentinel.master_for(
                self.master_name,
                socket_timeout=0.1,
                password=self.password,
                db=0
            )
            
            # Get slave connections
            self.slave_clients = [
                self.sentinel.slave_for(
                    self.master_name,
                    socket_timeout=0.1,
                    password=self.password,
                    db=0
                )
            ]
            
            logging.info("Sentinel setup completed successfully")
            
        except Exception as e:
            logging.error(f"Failed to setup Sentinel: {e}")
            raise
    
    def write_data(self, key: str, value: Any) -> bool:
        """Write data to master with failover handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.master_client:
                    self._refresh_master_connection()
                
                self.master_client.set(key, value)
                return True
                
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logging.warning(f"Master write failed (attempt {retry_count + 1}): {e}")
                retry_count += 1
                
                if retry_count < max_retries:
                    # Try to refresh master connection
                    self._refresh_master_connection()
                    time.sleep(0.1 * retry_count)  # Exponential backoff
                
            except Exception as e:
                logging.error(f"Unexpected error during write: {e}")
                return False
        
        logging.error(f"Failed to write after {max_retries} attempts")
        return False
    
    def read_data(self, key: str, prefer_slave: bool = True) -> Optional[str]:
        """Read data with slave preference for load balancing"""
        clients_to_try = []
        
        # Build priority list of clients to try
        if prefer_slave and self.slave_clients:
            clients_to_try.extend(self.slave_clients)
        
        if self.master_client:
            clients_to_try.append(self.master_client)
        
        for client in clients_to_try:
            try:
                value = client.get(key)
                return value.decode('utf-8') if value else None
            except Exception as e:
                logging.warning(f"Read failed from client: {e}")
                continue
        
        logging.error("Failed to read from any available client")
        return None
    
    def _refresh_master_connection(self):
        """Refresh master connection after failover"""
        try:
            self.master_client = self.sentinel.master_for(
                self.master_name,
                socket_timeout=0.1,
                password=self.password,
                db=0
            )
            logging.info("Master connection refreshed")
        except Exception as e:
            logging.error(f"Failed to refresh master connection: {e}")
            self.master_client = None
    
    def get_master_info(self) -> Dict[str, Any]:
        """Get current master information"""
        try:
            master_info = self.sentinel.sentinel_master(self.master_name)
            return {
                'host': master_info['ip'],
                'port': master_info['port'],
                'flags': master_info['flags'],
                'last_ping_sent': master_info['last-ping-sent'],
                'last_ok_ping_reply': master_info['last-ok-ping-reply'],
                'down_after_milliseconds': master_info['down-after-milliseconds'],
                'info_refresh': master_info['info-refresh'],
                'role_reported': master_info['role-reported'],
                'role_reported_time': master_info['role-reported-time'],
                'failover_timeout': master_info['failover-timeout'],
                'parallel_syncs': master_info['parallel-syncs'],
                'num_slaves': master_info['num-slaves'],
                'num_other_sentinels': master_info['num-other-sentinels']
            }
        except Exception as e:
            logging.error(f"Failed to get master info: {e}")
            return {}
    
    def get_slaves_info(self) -> List[Dict[str, Any]]:
        """Get information about all slaves"""
        try:
            slaves_info = self.sentinel.sentinel_slaves(self.master_name)
            return [
                {
                    'host': slave['ip'],
                    'port': slave['port'],
                    'flags': slave['flags'],
                    'master_link_status': slave['master-link-status'],
                    'slave_priority': slave['slave-priority'],
                    'slave_repl_offset': slave['slave-repl-offset']
                }
                for slave in slaves_info
            ]
        except Exception as e:
            logging.error(f"Failed to get slaves info: {e}")
            return []
    
    def add_failover_callback(self, callback):
        """Add callback function to be called on failover"""
        self.failover_callbacks.append(callback)
    
    def start_monitoring(self, check_interval: int = 5):
        """Start monitoring for failover events"""
        def monitor():
            last_master_info = self.get_master_info()
            
            while self.monitoring:
                try:
                    current_master_info = self.get_master_info()
                    
                    # Check if master has changed
                    if (last_master_info and current_master_info and
                        (last_master_info['host'] != current_master_info['host'] or
                         last_master_info['port'] != current_master_info['port'])):
                        
                        logging.info(f"Failover detected! New master: {current_master_info['host']}:{current_master_info['port']}")
                        
                        # Refresh connections
                        self._refresh_master_connection()
                        
                        # Call failover callbacks
                        for callback in self.failover_callbacks:
                            try:
                                callback(last_master_info, current_master_info)
                            except Exception as e:
                                logging.error(f"Failover callback failed: {e}")
                    
                    last_master_info = current_master_info
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
                    time.sleep(check_interval)
        
        self.monitoring = True
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logging.info("Failover monitoring started")
    
    def stop_monitoring(self):
        """Stop failover monitoring"""
        self.monitoring = False
        logging.info("Failover monitoring stopped")
    
    def force_failover(self) -> bool:
        """Manually trigger failover for testing"""
        try:
            result = self.sentinel.sentinel_failover(self.master_name)
            logging.info(f"Manual failover triggered: {result}")
            return True
        except Exception as e:
            logging.error(f"Failed to trigger manual failover: {e}")
            return False

# Failover notification callback example
def on_failover(old_master: Dict, new_master: Dict):
    """Example callback function for failover events"""
    print(f"🚨 FAILOVER ALERT:")
    print(f"   Old Master: {old_master['host']}:{old_master['port']}")
    print(f"   New Master: {new_master['host']}:{new_master['port']}")
    
    # Here you could:
    # - Send alerts to monitoring systems
    # - Update load balancer configurations
    # - Log to audit systems
    # - Update application configurations

# Usage example
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Sentinel configuration
    sentinel_hosts = [
        ('localhost', 26379),
        ('localhost', 26380),
        ('localhost', 26381)
    ]
    
    # Create HA client
    redis_ha = RedisSentinelHA(
        sentinel_hosts=sentinel_hosts,
        master_name='mymaster',
        password='master_password'
    )
    
    # Add failover callback
    redis_ha.add_failover_callback(on_failover)
    
    # Start monitoring
    redis_ha.start_monitoring()
    
    # Test operations
    success = redis_ha.write_data("test:failover", "test_value")
    print(f"Write success: {success}")
    
    value = redis_ha.read_data("test:failover")
    print(f"Read value: {value}")
    
    # Get cluster status
    master_info = redis_ha.get_master_info()
    print(f"Master info: {master_info}")
    
    slaves_info = redis_ha.get_slaves_info()
    print(f"Slaves info: {slaves_info}")
    
    # Keep running to monitor failovers
    try:
        while True:
            time.sleep(10)
            print("System running... (Ctrl+C to stop)")
    except KeyboardInterrupt:
        redis_ha.stop_monitoring()
        print("Monitoring stopped")
```

## Highly Available In-Memory Solutions

### 1. Multi-Region Deployment Architecture

Building globally distributed, highly available in-memory database systems requires careful consideration of network latency, data consistency, and disaster recovery across geographic regions.

#### Global Redis Deployment with Cross-Region Replication

```python
import redis
import redis.sentinel
import asyncio
import aioredis
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class Region(Enum):
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    ASIA_PACIFIC = "ap-southeast-1"

@dataclass
class RegionConfig:
    region: Region
    redis_hosts: List[Tuple[str, int]]
    sentinel_hosts: List[Tuple[str, int]]
    master_name: str
    latency_weight: float = 1.0
    priority: int = 1

class GlobalRedisCluster:
    def __init__(self, regions: List[RegionConfig]):
        self.regions = {config.region: config for config in regions}
        self.region_clients = {}
        self.region_sentinels = {}
        self.primary_region = None
        self.setup_connections()
    
    def setup_connections(self):
        """Setup connections to all regions"""
        for region, config in self.regions.items():
            try:
                # Setup Sentinel for this region
                sentinel = redis.sentinel.Sentinel(
                    config.sentinel_hosts,
                    socket_timeout=0.5
                )
                
                # Get master client
                master_client = sentinel.master_for(
                    config.master_name,
                    socket_timeout=0.5
                )
                
                self.region_clients[region] = {
                    'master': master_client,
                    'sentinel': sentinel
                }
                
                # Set primary region (highest priority)
                if (self.primary_region is None or 
                    config.priority > self.regions[self.primary_region].priority):
                    self.primary_region = region
                
                logging.info(f"Connected to region {region.value}")
                
            except Exception as e:
                logging.error(f"Failed to connect to region {region.value}: {e}")
    
    def write_globally(self, key: str, value: Any, consistency_level: str = 'eventual') -> bool:
        """Write data with global replication"""
        if consistency_level == 'strong':
            return self._write_strong_consistency(key, value)
        else:
            return self._write_eventual_consistency(key, value)
    
    def _write_strong_consistency(self, key: str, value: Any) -> bool:
        """Write with strong consistency (synchronous replication)"""
        success_count = 0
        total_regions = len(self.region_clients)
        
        # Write to all regions synchronously
        for region, clients in self.region_clients.items():
            try:
                clients['master'].set(key, value)
                success_count += 1
                logging.debug(f"Write successful to region {region.value}")
            except Exception as e:
                logging.error(f"Write failed to region {region.value}: {e}")
        
        # Consider successful if majority of regions succeeded
        return success_count > (total_regions // 2)
    
    def _write_eventual_consistency(self, key: str, value: Any) -> bool:
        """Write with eventual consistency (asynchronous replication)"""
        # Write to primary region first
        try:
            primary_client = self.region_clients[self.primary_region]['master']
            primary_client.set(key, value)
            
            # Asynchronously replicate to other regions
            asyncio.create_task(self._async_replicate(key, value))
            
            return True
            
        except Exception as e:
            logging.error(f"Write failed to primary region {self.primary_region.value}: {e}")
            
            # Try backup regions
            for region, clients in self.region_clients.items():
                if region != self.primary_region:
                    try:
                        clients['master'].set(key, value)
                        logging.info(f"Failover write successful to region {region.value}")
                        return True
                    except Exception as backup_e:
                        logging.error(f"Backup write failed to region {region.value}: {backup_e}")
            
            return False
    
    async def _async_replicate(self, key: str, value: Any):
        """Asynchronously replicate to secondary regions"""
        tasks = []
        
        for region, clients in self.region_clients.items():
            if region != self.primary_region:
                task = asyncio.create_task(
                    self._replicate_to_region(region, key, value)
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _replicate_to_region(self, region: Region, key: str, value: Any):
        """Replicate data to specific region"""
        try:
            # Use aioredis for async operations
            config = self.regions[region]
            redis_url = f"redis://{config.redis_hosts[0][0]}:{config.redis_hosts[0][1]}"
            
            async with aioredis.from_url(redis_url) as redis_client:
                await redis_client.set(key, value)
                logging.debug(f"Async replication successful to region {region.value}")
                
        except Exception as e:
            logging.error(f"Async replication failed to region {region.value}: {e}")
    
    def read_with_locality(self, key: str, preferred_region: Optional[Region] = None) -> Optional[str]:
        """Read with region locality preference"""
        regions_to_try = []
        
        # Add preferred region first
        if preferred_region and preferred_region in self.region_clients:
            regions_to_try.append(preferred_region)
        
        # Add primary region
        if self.primary_region not in regions_to_try:
            regions_to_try.append(self.primary_region)
        
        # Add other regions sorted by priority
        for region in sorted(self.regions.keys(), 
                           key=lambda r: self.regions[r].priority, 
                           reverse=True):
            if region not in regions_to_try:
                regions_to_try.append(region)
        
        # Try reading from regions in order
        for region in regions_to_try:
            try:
                clients = self.region_clients.get(region)
                if clients:
                    # Try slave first for read load balancing
                    try:
                        slave_client = clients['sentinel'].slave_for(
                            self.regions[region].master_name
                        )
                        value = slave_client.get(key)
                        if value:
                            return value.decode('utf-8')
                    except:
                        pass
                    
                    # Fallback to master
                    value = clients['master'].get(key)
                    if value:
                        return value.decode('utf-8')
                        
            except Exception as e:
                logging.error(f"Read failed from region {region.value}: {e}")
        
        return None
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get status of all regions"""
        status = {
            'primary_region': self.primary_region.value if self.primary_region else None,
            'regions': {},
            'total_regions': len(self.regions),
            'healthy_regions': 0
        }
        
        for region, clients in self.region_clients.items():
            region_status = {
                'healthy': False,
                'latency_ms': None,
                'master_info': {},
                'slaves_count': 0
            }
            
            try:
                # Measure latency
                start_time = time.time()
                clients['master'].ping()
                latency = (time.time() - start_time) * 1000
                
                region_status.update({
                    'healthy': True,
                    'latency_ms': round(latency, 2)
                })
                
                # Get master info
                master_info = clients['sentinel'].master_for(
                    self.regions[region].master_name
                ).info('replication')
                
                region_status['master_info'] = {
                    'role': master_info.get('role'),
                    'connected_slaves': master_info.get('connected_slaves', 0)
                }
                region_status['slaves_count'] = master_info.get('connected_slaves', 0)
                
                status['healthy_regions'] += 1
                
            except Exception as e:
                region_status['error'] = str(e)
            
            status['regions'][region.value] = region_status
        
        return status
    
    def handle_region_failure(self, failed_region: Region) -> bool:
        """Handle region failure and failover"""
        if failed_region not in self.region_clients:
            return False
        
        try:
            logging.warning(f"Handling failure for region {failed_region.value}")
            
            # Remove failed region from active clients
            del self.region_clients[failed_region]
            
            # If primary region failed, promote another region
            if self.primary_region == failed_region:
                new_primary = self._select_new_primary()
                if new_primary:
                    self.primary_region = new_primary
                    logging.info(f"Promoted region {new_primary.value} to primary")
                else:
                    logging.error("No healthy region available for promotion")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Region failover failed: {e}")
            return False
    
    def _select_new_primary(self) -> Optional[Region]:
        """Select new primary region based on priority and health"""
        candidates = []
        
        for region, clients in self.region_clients.items():
            try:
                # Test region health
                clients['master'].ping()
                priority = self.regions[region].priority
                candidates.append((region, priority))
            except:
                continue
        
        if candidates:
            # Sort by priority (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
```

## Practical Exercises and Labs

### Exercise 1: Redis Master-Slave Setup with Automatic Failover

**Objective:** Configure a Redis master-slave cluster with Sentinel for automatic failover.

**Setup Instructions:**
```bash
# 1. Create Redis configuration files
mkdir -p /tmp/redis-ha/{master,slave1,slave2,sentinel1,sentinel2,sentinel3}

# 2. Master configuration
cat > /tmp/redis-ha/master/redis.conf << EOF
port 6379
bind 0.0.0.0
requirepass mypassword
save 900 1
save 300 10
EOF

# 3. Slave configurations
cat > /tmp/redis-ha/slave1/redis.conf << EOF
port 6380
bind 0.0.0.0
replicaof 127.0.0.1 6379
masterauth mypassword
requirepass mypassword
EOF

cat > /tmp/redis-ha/slave2/redis.conf << EOF
port 6381
bind 0.0.0.0
replicaof 127.0.0.1 6379
masterauth mypassword
requirepass mypassword
EOF

# 4. Sentinel configurations
for port in 26379 26380 26381; do
cat > /tmp/redis-ha/sentinel$((port-26378))/sentinel.conf << EOF
port $port
bind 0.0.0.0
sentinel monitor mymaster 127.0.0.1 6379 2
sentinel auth-pass mymaster mypassword
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 10000
sentinel parallel-syncs mymaster 1
EOF
done

# 5. Start all instances
redis-server /tmp/redis-ha/master/redis.conf &
redis-server /tmp/redis-ha/slave1/redis.conf &
redis-server /tmp/redis-ha/slave2/redis.conf &
redis-sentinel /tmp/redis-ha/sentinel1/sentinel.conf &
redis-sentinel /tmp/redis-ha/sentinel2/sentinel.conf &
redis-sentinel /tmp/redis-ha/sentinel3/sentinel.conf &
```

**Testing Tasks:**
1. Verify replication is working
2. Stop the master and observe failover
3. Measure failover time
4. Test application connectivity during failover

**Expected Results:**
- Automatic failover within 10 seconds
- No data loss during failover
- Application reconnects to new master

### Exercise 2: Multi-Region Redis Deployment

**Objective:** Deploy Redis across multiple simulated regions with cross-region replication.

**Setup Requirements:**
- 3 Redis clusters (simulating different regions)
- Cross-region replication setup
- Load balancing with region awareness

**Implementation Steps:**
```python
# Complete the global Redis setup from the examples above
# Test with:
# 1. Write to US-East region
# 2. Read from EU-West region
# 3. Simulate region failure
# 4. Verify data consistency
```

**Validation Criteria:**
- Data consistency across regions
- Proper failover handling
- Acceptable latency for cross-region operations

### Exercise 3: Backup and Recovery Testing

**Objective:** Implement and test comprehensive backup and recovery procedures.

**Tasks:**
1. **Setup automated backups:**
   ```bash
   # Create backup script
   cat > /tmp/backup-test.py << 'EOF'
   from redis_backup_manager import RedisBackupManager, BackupConfig
   
   config = BackupConfig(
       redis_host='localhost',
       redis_port=6379,
       backup_dir='/tmp/redis-backups',
       retention_days=7
   )
   
   backup_manager = RedisBackupManager(config)
   
   # Test all backup types
   rdb_result = backup_manager.create_rdb_backup()
   aof_result = backup_manager.create_aof_backup()
   logical_result = backup_manager.create_logical_backup()
   
   print(f"Backups completed: RDB={rdb_result['status']}, AOF={aof_result['status']}, Logical={logical_result['status']}")
   EOF
   
   python /tmp/backup-test.py
   ```

2. **Test restoration procedures:**
   - Restore from RDB backup
   - Restore from logical backup
   - Verify data integrity after restoration

3. **Measure recovery times:**
   - RDB restoration time
   - Logical restoration time
   - Application startup time after recovery

### Exercise 4: Disaster Recovery Simulation

**Objective:** Simulate various disaster scenarios and test recovery procedures.

**Disaster Scenarios to Test:**
