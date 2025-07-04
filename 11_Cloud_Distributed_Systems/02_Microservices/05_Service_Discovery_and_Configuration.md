# Service Discovery and Configuration

*Duration: 2 weeks*

## Overview

In microservices architecture, services need to discover and communicate with each other dynamically. This module covers the patterns, technologies, and best practices for service discovery, configuration management, and service mesh technologies that enable scalable and resilient distributed systems.

## Learning Objectives

By the end of this section, you should be able to:
- **Understand service discovery patterns** and implement different discovery mechanisms
- **Design and implement service registries** using various technologies
- **Manage dynamic configuration** across distributed services
- **Implement service mesh solutions** for traffic management and security
- **Handle service health checking** and fault tolerance
- **Design configuration strategies** that support different environments
- **Troubleshoot service discovery issues** in production environments

## Core Concepts

### What is Service Discovery?

Service discovery is the mechanism by which services in a distributed system locate and communicate with each other. Instead of hardcoding service endpoints, applications dynamically discover available services through a registry.

#### Traditional vs. Microservices Communication

**Traditional Monolith:**
```
┌─────────────────┐
│   Monolith      │
│ ┌─────┐ ┌─────┐ │
│ │ UI  │ │ BL  │ │
│ └─────┘ └─────┘ │
│ ┌─────────────┐ │
│ │  Database   │ │
│ └─────────────┘ │
└─────────────────┘
```

**Microservices with Service Discovery:**
```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│   UI Service │    │ Service Registry│    │ Auth Service │
│              │    │                 │    │              │
│ 1. Lookup    │───▶│ 1. Register     │◀───│ 2. Register  │
│ 2. Call      │    │ 2. Health Check │    │              │
└──────────────┘    │ 3. Discovery    │    └──────────────┘
       │            └─────────────────┘           ▲
       │                                          │
       ▼                                          │
┌──────────────┐                        ┌──────────────┐
│Order Service │                        │Product Svc   │
│              │                        │              │
│ 3. Register  │                        │ 4. Register  │
└──────────────┘                        └──────────────┘
```

### Service Discovery Patterns

#### 1. Client-Side Discovery Pattern

The client is responsible for determining the locations of available service instances and load balancing between them.

**Architecture:**
```
┌─────────────┐     ┌─────────────────┐
│   Client    │────▶│ Service Registry│
│             │     │                 │
│ 1. Query    │     │ - Service A: IP1│
│ 2. Load     │     │ - Service A: IP2│
│    Balance  │     │ - Service B: IP3│
│ 3. Call     │     └─────────────────┘
└─────────────┘
       │
       ▼
┌─────────────┐   ┌─────────────┐
│ Service A   │   │ Service A   │
│ Instance 1  │   │ Instance 2  │
└─────────────┘   └─────────────┘
```

**Example Implementation with Consul:**

```python
import consul
import requests
import random
from typing import List, Dict

class ConsulServiceDiscovery:
    def __init__(self, consul_host='localhost', consul_port=8500):
        self.consul = consul.Consul(host=consul_host, port=consul_port)
    
    def register_service(self, name: str, service_id: str, address: str, port: int, health_check_url: str = None):
        """Register a service with Consul"""
        check = None
        if health_check_url:
            check = consul.Check.http(health_check_url, interval="10s")
        
        self.consul.agent.service.register(
            name=name,
            service_id=service_id,
            address=address,
            port=port,
            check=check
        )
        print(f"Service {name} registered with ID {service_id}")
    
    def discover_service(self, service_name: str) -> List[Dict]:
        """Discover healthy instances of a service"""
        _, services = self.consul.health.service(service_name, passing=True)
        
        instances = []
        for service in services:
            instances.append({
                'address': service['Service']['Address'],
                'port': service['Service']['Port'],
                'service_id': service['Service']['ID']
            })
        
        return instances
    
    def call_service(self, service_name: str, endpoint: str, method='GET', **kwargs):
        """Discover service and make a call with client-side load balancing"""
        instances = self.discover_service(service_name)
        
        if not instances:
            raise Exception(f"No healthy instances found for service {service_name}")
        
        # Simple random load balancing
        instance = random.choice(instances)
        url = f"http://{instance['address']}:{instance['port']}{endpoint}"
        
        try:
            response = requests.request(method, url, **kwargs)
            return response
        except Exception as e:
            print(f"Failed to call {service_name} at {url}: {e}")
            raise

# Usage Example
if __name__ == "__main__":
    # Initialize service discovery
    discovery = ConsulServiceDiscovery()
    
    # Register this service
    discovery.register_service(
        name="order-service",
        service_id="order-service-1",
        address="127.0.0.1",
        port=5000,
        health_check_url="http://127.0.0.1:5000/health"
    )
    
    # Discover and call another service
    try:
        response = discovery.call_service("product-service", "/api/products")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Service call failed: {e}")
```

**Flask Service Example:**
```python
from flask import Flask, jsonify
import threading
import time

app = Flask(__name__)
discovery = ConsulServiceDiscovery()

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/api/orders')
def get_orders():
    return jsonify({"orders": ["order1", "order2", "order3"]})

@app.route('/api/orders/<order_id>')
def get_order(order_id):
    # Example of calling another service
    try:
        response = discovery.call_service("product-service", f"/api/products/{order_id}")
        product_data = response.json()
        
        return jsonify({
            "order_id": order_id,
            "product": product_data,
            "status": "confirmed"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def register_service():
    """Register service on startup"""
    discovery.register_service(
        name="order-service",
        service_id="order-service-1",
        address="127.0.0.1",
        port=5000,
        health_check_url="http://127.0.0.1:5000/health"
    )

if __name__ == '__main__':
    # Register service in background
    registration_thread = threading.Thread(target=register_service)
    registration_thread.start()
    
    app.run(host='0.0.0.0', port=5000)
```

#### 2. Server-Side Discovery Pattern

A load balancer or API gateway handles service discovery and routing.

**Architecture:**
```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client    │────▶│  Load Balancer  │────▶│ Service Registry│
│             │     │   (API Gateway) │     │                 │
│             │     │                 │     │ - Service A: IP1│
│             │     │ 1. Route        │     │ - Service A: IP2│
│             │     │ 2. Load Balance │     │ - Service B: IP3│
│             │     │ 3. Health Check │     └─────────────────┘
└─────────────┘     └─────────────────┘
                             │
                             ▼
                    ┌─────────────┐   ┌─────────────┐
                    │ Service A   │   │ Service A   │
                    │ Instance 1  │   │ Instance 2  │
                    └─────────────┘   └─────────────┘
```

**NGINX Configuration Example:**
```nginx
upstream product_service {
    # Dynamic upstream populated by consul-template
    server 192.168.1.10:5001;
    server 192.168.1.11:5001;
    server 192.168.1.12:5001;
}

upstream order_service {
    server 192.168.1.20:5002;
    server 192.168.1.21:5002;
}

server {
    listen 80;
    
    location /api/products/ {
        proxy_pass http://product_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }
    
    location /api/orders/ {
        proxy_pass http://order_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Consul-Template for Dynamic Configuration:**
```bash
#!/bin/bash
# consul-template-config.hcl

template {
  source = "/etc/nginx/upstream.conf.ctmpl"
  destination = "/etc/nginx/conf.d/upstream.conf"
  command = "nginx -s reload"
}
```

```nginx
# upstream.conf.ctmpl
{{range services}}
upstream {{.Name}} {
  {{range service .Name}}
  server {{.Address}}:{{.Port}};
  {{end}}
}
{{end}}
```

#### 3. Service Mesh Pattern

A dedicated infrastructure layer handles service-to-service communication.

**Istio Service Mesh Example:**

```yaml
# service-registry.yaml
apiVersion: v1
kind: Service
metadata:
  name: product-service
  labels:
    app: product-service
spec:
  ports:
  - port: 80
    targetPort: 5001
    name: http
  selector:
    app: product-service
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: product-service
  template:
    metadata:
      labels:
        app: product-service
    spec:
      containers:
      - name: product-service
        image: product-service:latest
        ports:
        - containerPort: 5001
```

```yaml
# destination-rule.yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: product-service
spec:
  host: product-service
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

```yaml
# virtual-service.yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: product-service
spec:
  hosts:
  - product-service
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: product-service
        subset: v2
      weight: 100
  - route:
    - destination:
        host: product-service
        subset: v1
      weight: 90
    - destination:
        host: product-service
        subset: v2
      weight: 10
```

## Service Registry Technologies

### 1. Consul by HashiCorp

Consul provides service discovery, health checking, and key-value storage with strong consistency guarantees.

#### Setting up Consul

**Docker Compose Setup:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  consul:
    image: consul:latest
    command: agent -dev -client 0.0.0.0 -ui
    ports:
      - "8500:8500"
      - "8600:8600/udp"
    environment:
      - CONSUL_BIND_INTERFACE=eth0
    volumes:
      - consul_data:/consul/data

  product-service:
    build: ./product-service
    ports:
      - "5001:5001"
    depends_on:
      - consul
    environment:
      - CONSUL_HOST=consul
      - CONSUL_PORT=8500

volumes:
  consul_data:
```

#### Advanced Consul Configuration

```python
import consul
import json
from typing import Dict, Any, Optional

class AdvancedConsulClient:
    def __init__(self, host='localhost', port=8500):
        self.consul = consul.Consul(host=host, port=port)
    
    def register_service_with_metadata(self, 
                                     name: str, 
                                     service_id: str, 
                                     address: str, 
                                     port: int,
                                     tags: list = None,
                                     meta: Dict[str, str] = None,
                                     health_checks: list = None):
        """Register service with comprehensive metadata and health checks"""
        
        checks = []
        if health_checks:
            for check in health_checks:
                if check['type'] == 'http':
                    checks.append(consul.Check.http(
                        check['url'], 
                        interval=check.get('interval', '10s'),
                        timeout=check.get('timeout', '3s')
                    ))
                elif check['type'] == 'tcp':
                    checks.append(consul.Check.tcp(
                        f"{address}:{port}",
                        interval=check.get('interval', '10s'),
                        timeout=check.get('timeout', '3s')
                    ))
        
        self.consul.agent.service.register(
            name=name,
            service_id=service_id,
            address=address,
            port=port,
            tags=tags or [],
            meta=meta or {},
            check=checks[0] if checks else None
        )
    
    def discover_services_with_filter(self, 
                                    service_name: str, 
                                    tags: list = None,
                                    meta_filter: Dict[str, str] = None) -> list:
        """Discover services with advanced filtering"""
        
        _, services = self.consul.health.service(service_name, passing=True)
        
        filtered_services = []
        for service in services:
            service_info = service['Service']
            
            # Filter by tags
            if tags:
                if not all(tag in service_info.get('Tags', []) for tag in tags):
                    continue
            
            # Filter by metadata
            if meta_filter:
                service_meta = service_info.get('Meta', {})
                if not all(service_meta.get(k) == v for k, v in meta_filter.items()):
                    continue
            
            filtered_services.append({
                'id': service_info['ID'],
                'address': service_info['Address'],
                'port': service_info['Port'],
                'tags': service_info.get('Tags', []),
                'meta': service_info.get('Meta', {}),
                'health': 'passing'
            })
        
        return filtered_services
    
    def setup_key_value_config(self, prefix: str, config: Dict[str, Any]):
        """Store configuration in Consul KV store"""
        for key, value in config.items():
            full_key = f"{prefix}/{key}"
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            self.consul.kv.put(full_key, str(value))
    
    def get_config(self, prefix: str) -> Dict[str, str]:
        """Retrieve configuration from Consul KV store"""
        _, data = self.consul.kv.get(prefix, recurse=True)
        
        config = {}
        if data:
            for item in data:
                key = item['Key'].replace(f"{prefix}/", "")
                value = item['Value'].decode('utf-8') if item['Value'] else ""
                
                # Try to parse as JSON
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
                
                config[key] = value
        
        return config

# Usage Example
consul_client = AdvancedConsulClient()

# Register service with metadata
consul_client.register_service_with_metadata(
    name="product-service",
    service_id="product-service-v1-001",
    address="192.168.1.10",
    port=5001,
    tags=["v1", "production", "api"],
    meta={
        "version": "1.2.3",
        "environment": "production",
        "region": "us-west-2"
    },
    health_checks=[
        {
            "type": "http",
            "url": "http://192.168.1.10:5001/health",
            "interval": "10s",
            "timeout": "3s"
        },
        {
            "type": "tcp",
            "interval": "30s"
        }
    ]
)

# Discover services with filters
production_services = consul_client.discover_services_with_filter(
    service_name="product-service",
    tags=["production"],
    meta_filter={"environment": "production"}
)

print(f"Found {len(production_services)} production services")
```

### 2. Eureka (Netflix OSS)

Netflix's service registry designed for AWS cloud environments.

#### Eureka Server Configuration

```yaml
# application.yml for Eureka Server
server:
  port: 8761

eureka:
  instance:
    hostname: localhost
  client:
    registerWithEureka: false
    fetchRegistry: false
    serviceUrl:
      defaultZone: http://${eureka.instance.hostname}:${server.port}/eureka/
  server:
    enableSelfPreservation: false
    evictionIntervalTimerInMs: 4000
```

```java
// EurekaServerApplication.java
@SpringBootApplication
@EnableEurekaServer
public class EurekaServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(EurekaServerApplication.class, args);
    }
}
```

#### Eureka Client Configuration

```yaml
# application.yml for Eureka Client
spring:
  application:
    name: product-service

eureka:
  client:
    serviceUrl:
      defaultZone: http://localhost:8761/eureka/
    fetchRegistry: true
    registerWithEureka: true
  instance:
    preferIpAddress: true
    leaseRenewalIntervalInSeconds: 10
    leaseExpirationDurationInSeconds: 30
```

```java
// ProductServiceApplication.java
@SpringBootApplication
@EnableEurekaClient
@RestController
public class ProductServiceApplication {
    
    @Autowired
    private DiscoveryClient discoveryClient;
    
    @GetMapping("/products")
    public List<Product> getProducts() {
        return productService.getAllProducts();
    }
    
    @GetMapping("/discover/{serviceName}")
    public List<ServiceInstance> discover(@PathVariable String serviceName) {
        return discoveryClient.getInstances(serviceName);
    }
    
    public static void main(String[] args) {
        SpringApplication.run(ProductServiceApplication.class, args);
    }
}
```

#### Service-to-Service Communication with Eureka

```java
@Component
public class OrderServiceClient {
    
    @Autowired
    private DiscoveryClient discoveryClient;
    
    @Autowired
    private RestTemplate restTemplate;
    
    public Order getOrderFromOrderService(String orderId) {
        List<ServiceInstance> instances = discoveryClient.getInstances("order-service");
        
        if (instances.isEmpty()) {
            throw new RuntimeException("No instances of order-service available");
        }
        
        // Simple load balancing - choose first available
        ServiceInstance instance = instances.get(0);
        String url = String.format("http://%s:%d/orders/%s", 
                                 instance.getHost(), 
                                 instance.getPort(), 
                                 orderId);
        
        return restTemplate.getForObject(url, Order.class);
    }
    
    @Bean
    @LoadBalanced  // Enable client-side load balancing
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

### 3. etcd (Kubernetes Native)

Distributed key-value store used by Kubernetes for service discovery.

#### etcd Service Discovery Implementation

```python
import etcd3
import json
import threading
import time
from typing import Dict, List, Callable

class EtcdServiceRegistry:
    def __init__(self, host='localhost', port=2379):
        self.client = etcd3.client(host=host, port=port)
        self.service_prefix = "/services/"
        self.config_prefix = "/config/"
        self.watchers = {}
    
    def register_service(self, 
                        service_name: str, 
                        instance_id: str, 
                        address: str, 
                        port: int, 
                        metadata: Dict = None,
                        ttl: int = 30):
        """Register service with TTL for health checking"""
        
        service_data = {
            "address": address,
            "port": port,
            "metadata": metadata or {},
            "registered_at": time.time()
        }
        
        key = f"{self.service_prefix}{service_name}/{instance_id}"
        
        # Create lease for TTL
        lease = self.client.lease(ttl)
        
        # Register service
        self.client.put(key, json.dumps(service_data), lease=lease)
        
        # Keep lease alive
        def refresh_lease():
            while True:
                try:
                    lease.refresh()
                    time.sleep(ttl // 3)  # Refresh at 1/3 of TTL
                except Exception as e:
                    print(f"Failed to refresh lease for {service_name}: {e}")
                    break
        
        lease_thread = threading.Thread(target=refresh_lease)
        lease_thread.daemon = True
        lease_thread.start()
        
        return lease
    
    def discover_services(self, service_name: str) -> List[Dict]:
        """Discover all instances of a service"""
        prefix = f"{self.service_prefix}{service_name}/"
        
        services = []
        for value, metadata in self.client.get_prefix(prefix):
            if value:
                try:
                    service_data = json.loads(value.decode('utf-8'))
                    service_data['instance_id'] = metadata.key.decode('utf-8').split('/')[-1]
                    services.append(service_data)
                except json.JSONDecodeError:
                    continue
        
        return services
    
    def watch_service(self, service_name: str, callback: Callable):
        """Watch for changes in service instances"""
        prefix = f"{self.service_prefix}{service_name}/"
        
        def watch_callback(event):
            if hasattr(event, 'events'):
                for e in event.events:
                    if isinstance(e, etcd3.events.PutEvent):
                        service_data = json.loads(e.value.decode('utf-8'))
                        callback('PUT', service_data)
                    elif isinstance(e, etcd3.events.DeleteEvent):
                        callback('DELETE', {'key': e.key.decode('utf-8')})
        
        watch_id = self.client.add_watch_prefix_callback(prefix, watch_callback)
        self.watchers[service_name] = watch_id
        
        return watch_id
    
    def put_config(self, config_key: str, config_value: Dict):
        """Store configuration"""
        key = f"{self.config_prefix}{config_key}"
        self.client.put(key, json.dumps(config_value))
    
    def get_config(self, config_key: str) -> Dict:
        """Retrieve configuration"""
        key = f"{self.config_prefix}{config_key}"
        value, _ = self.client.get(key)
        
        if value:
            return json.loads(value.decode('utf-8'))
        return {}
    
    def watch_config(self, config_key: str, callback: Callable):
        """Watch configuration changes"""
        key = f"/config/{config_key}"
        
        def config_callback(event):
            if hasattr(event, 'events'):
                for e in event.events:
                    if isinstance(e, etcd3.events.PutEvent):
                        config_data = json.loads(e.value.decode('utf-8'))
                        callback(config_data)
        
        return self.client.add_watch_callback(key, config_callback)

# Usage Example
registry = EtcdServiceRegistry()

# Register service
lease = registry.register_service(
    service_name="product-service",
    instance_id="instance-001",
    address="192.168.1.10",
    port=5001,
    metadata={"version": "1.0", "environment": "production"},
    ttl=30
)

# Service discovery with monitoring
def service_change_handler(event_type, data):
    print(f"Service change: {event_type} - {data}")

registry.watch_service("product-service", service_change_handler)

# Configuration management
app_config = {
    "database": {
        "host": "db.example.com",
        "port": 5432,
        "name": "products"
    },
    "features": {
        "enable_caching": True,
        "max_connections": 100
    }
}

registry.put_config("product-service/config", app_config)

def config_change_handler(new_config):
    print(f"Configuration updated: {new_config}")
    # Reload application configuration

registry.watch_config("product-service/config", config_change_handler)
```

### 4. Apache Zookeeper

Centralized service for maintaining configuration information and distributed synchronization.

#### Zookeeper Service Discovery

```python
from kazoo.client import KazooClient
from kazoo.recipe.watchers import ChildrenWatch, DataWatch
import json
import threading
import time

class ZookeeperServiceRegistry:
    def __init__(self, hosts='localhost:2181'):
        self.zk = KazooClient(hosts=hosts)
        self.zk.start()
        
        # Ensure base paths exist
        self.zk.ensure_path("/services")
        self.zk.ensure_path("/config")
    
    def register_service(self, 
                        service_name: str, 
                        instance_id: str, 
                        address: str, 
                        port: int, 
                        metadata: dict = None):
        """Register an ephemeral service node"""
        
        service_path = f"/services/{service_name}"
        self.zk.ensure_path(service_path)
        
        instance_data = {
            "address": address,
            "port": port,
            "metadata": metadata or {},
            "registered_at": time.time()
        }
        
        instance_path = f"{service_path}/{instance_id}"
        
        # Create ephemeral node (will be deleted when client disconnects)
        self.zk.create(
            instance_path, 
            json.dumps(instance_data).encode('utf-8'),
            ephemeral=True,
            makepath=True
        )
        
        print(f"Service {service_name}/{instance_id} registered")
        return instance_path
    
    def discover_services(self, service_name: str) -> list:
        """Discover all instances of a service"""
        service_path = f"/services/{service_name}"
        
        try:
            children = self.zk.get_children(service_path)
            services = []
            
            for child in children:
                child_path = f"{service_path}/{child}"
                data, stat = self.zk.get(child_path)
                
                if data:
                    service_info = json.loads(data.decode('utf-8'))
                    service_info['instance_id'] = child
                    service_info['path'] = child_path
                    services.append(service_info)
            
            return services
            
        except Exception as e:
            print(f"Error discovering services: {e}")
            return []
    
    def watch_services(self, service_name: str, callback):
        """Watch for service changes"""
        service_path = f"/services/{service_name}"
        self.zk.ensure_path(service_path)
        
        def children_callback(children):
            services = []
            for child in children:
                child_path = f"{service_path}/{child}"
                try:
                    data, stat = self.zk.get(child_path)
                    service_info = json.loads(data.decode('utf-8'))
                    service_info['instance_id'] = child
                    services.append(service_info)
                except Exception as e:
                    print(f"Error reading child {child}: {e}")
            
            callback(services)
        
        return ChildrenWatch(self.zk, service_path, children_callback)
    
    def set_config(self, config_path: str, config_data: dict):
        """Set configuration data"""
        full_path = f"/config/{config_path}"
        self.zk.ensure_path(full_path)
        
        self.zk.set(full_path, json.dumps(config_data).encode('utf-8'))
    
    def get_config(self, config_path: str) -> dict:
        """Get configuration data"""
        full_path = f"/config/{config_path}"
        
        try:
            data, stat = self.zk.get(full_path)
            return json.loads(data.decode('utf-8'))
        except Exception:
            return {}
    
    def watch_config(self, config_path: str, callback):
        """Watch configuration changes"""
        full_path = f"/config/{config_path}"
        self.zk.ensure_path(full_path)
        
        def data_callback(data, stat):
            if data:
                config = json.loads(data.decode('utf-8'))
                callback(config)
        
        return DataWatch(self.zk, full_path, data_callback)

# Example usage
zk_registry = ZookeeperServiceRegistry()

# Register service
zk_registry.register_service(
    service_name="product-service",
    instance_id="instance-001",
    address="192.168.1.10",
    port=5001,
    metadata={"version": "1.0", "datacenter": "us-west"}
)

# Watch for service changes
def on_service_change(services):
    print(f"Service instances changed: {len(services)} instances")
    for service in services:
        print(f"  - {service['instance_id']}: {service['address']}:{service['port']}")

watcher = zk_registry.watch_services("product-service", on_service_change)

# Configuration management
config = {
    "database_url": "postgresql://localhost:5432/products",
    "redis_url": "redis://localhost:6379",
    "log_level": "INFO"
}

zk_registry.set_config("product-service", config)

def on_config_change(new_config):
    print(f"Configuration changed: {new_config}")
    # Reload application with new config

config_watcher = zk_registry.watch_config("product-service", on_config_change)
```

## Health Checking and Fault Tolerance

Robust health checking and fault tolerance mechanisms are essential for maintaining service reliability in distributed systems.

### Health Check Patterns

#### 1. Basic Health Checks

```python
from flask import Flask, jsonify
import psutil
import time
import threading
from typing import Dict, Any

app = Flask(__name__)

class HealthChecker:
    def __init__(self):
        self.start_time = time.time()
        self.health_status = {
            'status': 'healthy',
            'checks': {},
            'last_updated': time.time()
        }
        self.dependencies = {}
        
        # Start background health monitoring
        self.start_health_monitoring()
    
    def add_dependency(self, name: str, check_func, critical: bool = True):
        """Add a dependency health check"""
        self.dependencies[name] = {
            'check_func': check_func,
            'critical': critical,
            'last_check': 0,
            'status': 'unknown'
        }
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Simulate database check
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'products'),
                user=os.getenv('DB_USER', 'user'),
                password=os.getenv('DB_PASSWORD', 'password'),
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            cursor.fetchone()
            cursor.close()
            conn.close()
            
            return {
                'status': 'healthy',
                'response_time': 0.05,
                'details': 'Database connection successful'
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'Database connection failed'
            }
    
    def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            import redis
            r = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                db=0,
                socket_timeout=5
            )
            
            start_time = time.time()
            r.ping()
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'details': 'Redis connection successful'
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'Redis connection failed'
            }
    
    def check_external_service_health(self, service_url: str) -> Dict[str, Any]:
        """Check external service health"""
        try:
            import requests
            
            start_time = time.time()
            response = requests.get(f"{service_url}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time': response_time,
                    'details': f'External service {service_url} is healthy'
                }
            else:
                return {
                    'status': 'unhealthy',
                    'status_code': response.status_code,
                    'details': f'External service returned {response.status_code}'
                }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': f'Failed to reach external service {service_url}'
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Define thresholds
            cpu_threshold = 80
            memory_threshold = 85
            disk_threshold = 90
            
            issues = []
            if cpu_percent > cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > memory_threshold:
                issues.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > disk_threshold:
                issues.append(f"High disk usage: {disk.percent}%")
            
            status = 'unhealthy' if issues else 'healthy'
            
            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'issues': issues,
                'details': 'System resource check completed'
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'details': 'Failed to check system resources'
            }
    
    def perform_health_checks(self) -> Dict[str, Any]:
        """Perform all health checks"""
        checks = {}
        overall_status = 'healthy'
        
        for name, dependency in self.dependencies.items():
            try:
                check_result = dependency['check_func']()
                checks[name] = check_result
                dependency['status'] = check_result['status']
                dependency['last_check'] = time.time()
                
                # If critical dependency is unhealthy, mark overall as unhealthy
                if dependency['critical'] and check_result['status'] != 'healthy':
                    overall_status = 'unhealthy'
            
            except Exception as e:
                checks[name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'details': f'Health check failed for {name}'
                }
                
                if dependency['critical']:
                    overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time,
            'checks': checks
        }
    
    def start_health_monitoring(self):
        """Start background health monitoring"""
        def monitor():
            while True:
                try:
                    self.health_status = self.perform_health_checks()
                except Exception as e:
                    print(f"Error in health monitoring: {e}")
                
                time.sleep(30)  # Check every 30 seconds
        
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

# Initialize health checker
health_checker = HealthChecker()

# Add dependency checks
health_checker.add_dependency('database', health_checker.check_database_health, critical=True)
health_checker.add_dependency('redis', health_checker.check_redis_health, critical=False)
health_checker.add_dependency('system', health_checker.check_system_resources, critical=True)
health_checker.add_dependency('inventory_service', 
                             lambda: health_checker.check_external_service_health('http://inventory-service:5002'),
                             critical=False)

@app.route('/health')
def basic_health():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'service': 'product-service'
    })

@app.route('/health/detailed')
def detailed_health():
    """Detailed health check with dependencies"""
    return jsonify(health_checker.health_status)

@app.route('/health/live')
def liveness_probe():
    """Kubernetes liveness probe"""
    # Simple check - is the application responding?
    return jsonify({
        'status': 'alive',
        'timestamp': time.time()
    })

@app.route('/health/ready')
def readiness_probe():
    """Kubernetes readiness probe"""
    # Check if service is ready to handle traffic
    health_status = health_checker.health_status
    
    # Consider ready if critical dependencies are healthy
    critical_healthy = all(
        check.get('status') == 'healthy'
        for name, check in health_status['checks'].items()
        if health_checker.dependencies.get(name, {}).get('critical', True)
    )
    
    status_code = 200 if critical_healthy else 503
    
    return jsonify({
        'status': 'ready' if critical_healthy else 'not_ready',
        'timestamp': time.time(),
        'critical_checks': {
            name: check
            for name, check in health_status['checks'].items()
            if health_checker.dependencies.get(name, {}).get('critical', True)
        }
    }), status_code
```

#### 2. Kubernetes Health Check Configuration

```yaml
# deployment-with-health-checks.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: product-service
  template:
    metadata:
      labels:
        app: product-service
    spec:
      containers:
      - name: product-service
        image: product-service:latest
        ports:
        - containerPort: 5001
        
        # Liveness probe - restart container if failing
        livenessProbe:
          httpGet:
            path: /health/live
            port: 5001
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        
        # Readiness probe - remove from load balancer if failing
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 5001
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        
        # Startup probe - give more time for initial startup
        startupProbe:
          httpGet:
            path: /health/live
            port: 5001
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 30  # Allow up to 5 minutes for startup
        
        env:
        - name: DB_HOST
          value: "postgres-service"
        - name: REDIS_HOST
          value: "redis-service"
        
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### Circuit Breaker Pattern

#### 1. Python Circuit Breaker Implementation

```python
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open" # Testing if service is back

class CircuitBreaker:
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception,
                 name: str = "circuit_breaker"):
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    print(f"Circuit breaker {self.name}: Attempting reset (HALF_OPEN)")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            print(f"Circuit breaker {self.name}: Reset successful (CLOSED)")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                print(f"Circuit breaker {self.name}: Opened due to {self.failure_count} failures")
    
    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout
        }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass
```

## Study Materials and Best Practices

### Recommended Reading
- **Primary:** "Building Microservices" by Sam Newman - Chapters 4-6 (Service Discovery & Configuration)
- **Architecture:** "Microservices Patterns" by Chris Richardson - Discovery and Configuration patterns
- **Service Mesh:** "Istio in Action" by Christian Posta and Rinor Maloku
- **Fault Tolerance:** "Release It!" by Michael Nygard - Stability patterns

### Hands-on Labs
1. **Lab 1:** Set up Consul cluster with service registration
2. **Lab 2:** Implement feature flags with real-time configuration updates
3. **Lab 3:** Deploy Istio service mesh with traffic management
4. **Lab 4:** Build resilient service client with circuit breakers

### Practice Exercises

**Exercise 1: Service Registry Implementation**
```python
# TODO: Implement a service registry client that:
# 1. Registers services with health checks
# 2. Discovers services with load balancing
# 3. Handles service failures gracefully

class ServiceRegistry:
    def __init__(self, registry_url):
        # Your implementation here
        pass
    
    def register(self, service_name, address, port, health_check_url):
        # Your implementation here
        pass
    
    def discover(self, service_name):
        # Your implementation here
        pass
```

**Exercise 2: Circuit Breaker Testing**
```python
# TODO: Test circuit breaker behavior:
# 1. Normal operation (CLOSED state)
# 2. Failure scenarios (OPEN state)
# 3. Recovery testing (HALF_OPEN state)

def test_circuit_breaker():
    # Your test implementation here
    pass
```

### Best Practices Summary

✅ **DO:**
- Implement comprehensive health checks for all dependencies
- Use circuit breakers for external service calls
- Implement graceful degradation with fallbacks
- Monitor service discovery performance and reliability
- Use feature flags for safe deployment strategies
- Secure service-to-service communication with mTLS
- Implement proper retry logic with exponential backoff

❌ **DON'T:**
- Hardcode service endpoints
- Skip health check endpoints
- Ignore circuit breaker states in monitoring
- Store sensitive configuration in plain text
- Deploy configuration changes without validation
- Overload service registries with frequent registrations
- Ignore service mesh security policies

### Next Steps
1. Implement service discovery in your microservices
2. Set up configuration management with environment-specific settings
3. Deploy and configure a service mesh
4. Implement comprehensive health checking and fault tolerance
5. Monitor and alert on service discovery metrics
