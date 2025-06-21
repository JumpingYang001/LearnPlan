# Kubernetes Fundamentals

## Container Orchestration Concepts

Container orchestration is the automated management of containerized applications across a cluster of machines. As organizations scale their containerized workloads, manual management becomes impractical and error-prone. Kubernetes provides a robust platform for automating deployment, scaling, and operations of application containers.

### Why Container Orchestration?

Modern applications often consist of multiple microservices running in containers. Without orchestration, developers and operators face challenges such as:

- **Service Discovery**: How do containers find and communicate with each other?
- **Load Balancing**: How is traffic distributed across multiple container instances?
- **Scaling**: How do we automatically scale applications based on demand?
- **Health Management**: How do we detect and replace failed containers?
- **Configuration Management**: How do we manage application configuration across environments?
- **Storage Management**: How do we handle persistent data in ephemeral containers?

Kubernetes addresses these challenges through declarative configuration and automated control loops that continuously work to maintain the desired state of your applications.

### Kubernetes vs. Other Orchestrators

Kubernetes has become the de facto standard for container orchestration due to its:

- **Extensibility**: Rich API and plugin ecosystem
- **Community**: Large, active open-source community
- **Portability**: Runs on any infrastructure (cloud, on-premises, hybrid)
- **Maturity**: Battle-tested in production environments
- **Ecosystem**: Extensive tooling and integration capabilities

Compared to alternatives like Docker Swarm (simpler but less feature-rich) or Apache Mesos (more complex, focused on resource management), Kubernetes provides the best balance of features, flexibility, and community support.

### Cloud-Native Principles

Kubernetes embodies cloud-native principles that enable applications to take full advantage of cloud computing:

1. **Containerization**: Applications are packaged with their dependencies
2. **Microservices**: Applications are decomposed into loosely coupled services
3. **DevOps**: Development and operations teams collaborate closely
4. **Continuous Delivery**: Automated deployment pipelines
5. **Declarative APIs**: Infrastructure and applications are defined as code

## Core Kubernetes Resources

Understanding Kubernetes resources is fundamental to working with the platform. These resources represent the desired state of your applications and infrastructure.

### Pods - The Atomic Unit

A Pod is the smallest deployable unit in Kubernetes. It represents a group of one or more containers that:
- Share the same network namespace (IP address and port space)
- Share storage volumes
- Are scheduled together on the same node
- Live and die together

Most Pods contain only one container, but multi-container pods are used for tightly coupled processes that need to share resources.

### Basic Pod Definition

The following example shows a simple Pod definition with resource management and health checks:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-world
  labels:
    app: hello-world
spec:
  containers:
  - name: hello-container
    image: nginx:latest
    ports:
    - containerPort: 80
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
```

**Key Components Explained:**
- **apiVersion**: Specifies the Kubernetes API version for this resource
- **kind**: Defines the type of Kubernetes resource (Pod in this case)
- **metadata**: Contains identifying information like name and labels
- **spec**: Defines the desired state of the Pod
- **containers**: Lists the containers that should run in this Pod
- **resources**: Specifies CPU and memory requests and limits for resource management

### ReplicaSets - Ensuring Availability

ReplicaSets ensure that a specified number of pod replicas are running at any given time. They provide basic scaling and high availability for stateless applications.

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: nginx-replicaset
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
```

**ReplicaSet Features:**
- **replicas**: Specifies the desired number of pod instances
- **selector**: Defines which pods are managed by this ReplicaSet
- **template**: Provides the pod template used to create new pods
- **Automatic Recovery**: Replaces failed pods automatically
- **Scaling**: Can be scaled up or down by changing the replica count

While ReplicaSets provide basic replication, they are typically managed by higher-level controllers like Deployments.

### Deployments - Production-Ready Workload Management

Deployments provide declarative updates for Pods and ReplicaSets. They are the recommended way to manage stateless applications in production, offering features like rolling updates, rollbacks, and deployment strategies.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app-deployment
  labels:
    app: web-app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web-app
        image: nginx:1.21
        ports:
        - containerPort: 80
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Deployment Features:**
- **Rolling Updates**: Updates pods gradually without downtime
- **Rollback**: Can revert to previous versions quickly
- **Scaling**: Easily scale up or down by changing replica count
- **Health Checks**: Liveness and readiness probes ensure healthy pods
- **Deployment Strategies**: Supports different update strategies (RollingUpdate, Recreate)

**Health Probes Explained:**
- **Liveness Probe**: Determines if a container is running and healthy
- **Readiness Probe**: Determines if a container is ready to serve traffic
- **Startup Probe**: Provides additional time for slow-starting containers

### Services - Network Abstraction and Load Balancing

Services provide a stable network endpoint for accessing pods, abstracting away the dynamic nature of pod IP addresses and providing load balancing across pod replicas.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
spec:
  selector:
    app: web-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: web-app-nodeport
spec:
  selector:
    app: web-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
      nodePort: 30080
  type: NodePort
```

**Service Types:**
- **ClusterIP**: Default type, provides internal cluster access only
- **NodePort**: Exposes service on each node's IP at a static port
- **LoadBalancer**: Creates an external load balancer (cloud provider dependent)
- **ExternalName**: Maps service to a DNS name for external services

**Key Concepts:**
- **Selector**: Determines which pods receive traffic from the service
- **Port**: The port that the service exposes
- **TargetPort**: The port on the pods that receives traffic
- **Load Balancing**: Automatically distributes traffic across healthy pods

### Configuration Management - ConfigMaps and Secrets

ConfigMaps and Secrets separate configuration from application code, following the twelve-factor app principle of storing config in the environment.
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgresql://localhost:5432/mydb"
  log_level: "info"
  feature_flags: |
    feature_a=true
    feature_b=false
---
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  # Base64 encoded values
  username: YWRtaW4=  # admin
  password: MWYyZDFlMmU2N2Rm  # 1f2d1e2e67df
stringData:
  # Plain text values (will be base64 encoded automatically)
  api_key: "sk-1234567890abcdef"
```

**ConfigMap Features:**
- **Non-sensitive Configuration**: Store configuration data like database URLs, feature flags
- **Multiple Data Types**: Support for key-value pairs, files, and multi-line strings
- **Usage Patterns**: Can be mounted as volumes or used as environment variables
- **Dynamic Updates**: Changes can be reflected in running pods (with proper configuration)

**Secret Features:**
- **Sensitive Data**: Designed for storing passwords, tokens, keys, and certificates
- **Base64 Encoding**: Data is base64 encoded (not encrypted by default)
- **Secret Types**: Supports different types like Opaque, TLS, Docker registry secrets
- **Security**: Can be encrypted at rest with proper etcd configuration

## Basic kubectl Operations

kubectl is the command-line tool for interacting with Kubernetes clusters. Mastering basic kubectl commands is essential for day-to-day Kubernetes operations.
```bash
# Create resources
kubectl create -f pod.yaml
kubectl apply -f deployment.yaml

# Get resources
kubectl get pods
kubectl get deployments
kubectl get services

# Describe resources
kubectl describe pod hello-world
kubectl describe deployment web-app-deployment

# View logs
kubectl logs hello-world
kubectl logs -f deployment/web-app-deployment

# Execute commands in pods
kubectl exec -it hello-world -- /bin/bash
kubectl exec hello-world -- ps aux

# Port forwarding
kubectl port-forward pod/hello-world 8080:80
kubectl port-forward service/web-app-service 8080:80

# Delete resources
kubectl delete pod hello-world
kubectl delete -f deployment.yaml
```

**Command Categories:**
- **Resource Management**: create, apply, delete for managing resource lifecycle
- **Information Gathering**: get, describe, logs for inspecting cluster state
- **Troubleshooting**: exec, port-forward for debugging and testing
- **Watching**: Use -w flag with get commands to watch for changes
- **Output Formats**: Use -o yaml, -o json, -o wide for different output formats

## Advanced Patterns

### Multi-Container Pods

Multi-container pods are useful for tightly coupled processes that need to share resources like storage and network. Common patterns include sidecar, adapter, and ambassador patterns.
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-container-pod
spec:
  containers:
  - name: web-server
    image: nginx:1.21
    ports:
    - containerPort: 80
    volumeMounts:
    - name: shared-data
      mountPath: /usr/share/nginx/html
  - name: content-puller
    image: alpine/git
    command: ["/bin/sh"]
    args:
      - -c
      - |
        while true; do
          git clone https://github.com/example/website.git /tmp/repo
          cp -r /tmp/repo/* /shared/
          rm -rf /tmp/repo
          sleep 3600
        done
    volumeMounts:
    - name: shared-data
      mountPath: /shared
  volumes:
  - name: shared-data
    emptyDir: {}
```

**Multi-Container Patterns:**
- **Sidecar Pattern**: Helper container that enhances the main container (logging, monitoring)
- **Adapter Pattern**: Container that transforms the main container's output
- **Ambassador Pattern**: Container that proxies connections for the main container
- **Shared Resources**: Containers in the same pod share network and storage volumes
- **Lifecycle**: All containers in a pod are scheduled together and share the same lifecycle

This example demonstrates the sidecar pattern where a content-puller container updates static content that the web server serves, both sharing a common volume for data exchange.

## Key Takeaways

1. **Declarative Configuration**: Kubernetes uses YAML manifests to describe desired state
2. **Controllers**: Kubernetes controllers continuously work to maintain desired state
3. **Abstraction Layers**: Services abstract network access, while controllers abstract workload management
4. **Resource Management**: Proper resource requests and limits are crucial for cluster stability
5. **Health Checks**: Liveness and readiness probes ensure application reliability
6. **Configuration Separation**: Use ConfigMaps and Secrets to separate configuration from code

Understanding these fundamentals provides the foundation for more advanced Kubernetes concepts like networking, storage, security, and operations.

## Next Section

Continue to [Cluster Architecture](02_Cluster_Architecture.md) to learn about Kubernetes cluster components and architecture patterns.
