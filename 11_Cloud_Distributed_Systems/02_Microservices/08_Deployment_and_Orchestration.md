# Deployment and Orchestration

*Duration: 2-3 weeks*

## Overview

Deployment and orchestration are critical aspects of modern microservices architecture. This section covers containerization with Docker, orchestration with Kubernetes, CI/CD pipelines, and automated deployment strategies. You'll learn to package, deploy, and manage microservices at scale.

## Key Concepts to Master

### 1. Containerization Fundamentals

#### What is Containerization?
Containerization packages applications and their dependencies into lightweight, portable containers that can run consistently across different environments.

#### Benefits of Containerization
- **Consistency**: Same environment from development to production
- **Portability**: Run anywhere that supports containers
- **Isolation**: Applications don't interfere with each other
- **Efficiency**: Lightweight compared to virtual machines
- **Scalability**: Easy to scale up/down

#### Container vs Virtual Machine Comparison

```
Virtual Machines                    Containers
┌─────────────────────────────┐    ┌─────────────────────────────┐
│        Application          │    │        Application          │
│ ┌─────────────────────────┐ │    │ ┌─────────────────────────┐ │
│ │      App Code           │ │    │ │      App Code           │ │
│ │      Libraries          │ │    │ │      Libraries          │ │
│ └─────────────────────────┘ │    │ └─────────────────────────┘ │
│        Guest OS             │    │      Container Runtime      │
│ ┌─────────────────────────┐ │    │ ┌─────────────────────────┐ │
│ │      Kernel             │ │    │ │      Shared Kernel      │ │
│ └─────────────────────────┘ │    │ └─────────────────────────┘ │
│        Hypervisor           │    │        Host OS              │
│        Host OS              │    │        Hardware             │
│        Hardware             │    └─────────────────────────────┘
└─────────────────────────────┘
```

### 2. Docker Deep Dive

#### Docker Architecture
- **Docker Engine**: Core runtime
- **Docker Images**: Read-only templates
- **Docker Containers**: Running instances of images
- **Docker Registry**: Storage for images (Docker Hub, private registries)

#### Dockerfile Best Practices

**Basic Microservice Dockerfile:**
```dockerfile
# Multi-stage build for smaller production images
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files first (better layer caching)
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src/ ./src/

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodeuser -u 1001

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder --chown=nodeuser:nodejs /app/dist ./dist
COPY --from=builder --chown=nodeuser:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodeuser:nodejs /app/package*.json ./

# Switch to non-root user
USER nodeuser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start the application
CMD ["node", "dist/index.js"]
```

**Python Flask Microservice with Best Practices:**
```dockerfile
# Use specific version for reproducibility
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

#### Docker Compose for Multi-Service Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  # API Gateway
  api-gateway:
    build: 
      context: ./api-gateway
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - NODE_ENV=development
      - USER_SERVICE_URL=http://user-service:3001
      - ORDER_SERVICE_URL=http://order-service:3002
    depends_on:
      - user-service
      - order-service
    networks:
      - microservices-network

  # User Service
  user-service:
    build: 
      context: ./user-service
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://user:password@user-db:5432/userdb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - user-db
      - redis
    networks:
      - microservices-network

  # Order Service
  order-service:
    build: 
      context: ./order-service
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=mongodb://order-db:27017/orderdb
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on:
      - order-db
      - rabbitmq
    networks:
      - microservices-network

  # Databases
  user-db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=userdb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - user-db-data:/var/lib/postgresql/data
    networks:
      - microservices-network

  order-db:
    image: mongo:6.0-alpine
    volumes:
      - order-db-data:/data/db
    networks:
      - microservices-network

  # Message Queue
  rabbitmq:
    image: rabbitmq:3.11-management-alpine
    ports:
      - "15672:15672"  # Management UI
    environment:
      - RABBITMQ_DEFAULT_USER=admin
      - RABBITMQ_DEFAULT_PASS=password
    networks:
      - microservices-network

  # Cache
  redis:
    image: redis:7-alpine
    networks:
      - microservices-network

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - microservices-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - microservices-network

volumes:
  user-db-data:
  order-db-data:
  grafana-data:

networks:
  microservices-network:
    driver: bridge
```

### 3. Kubernetes Orchestration

#### Kubernetes Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                       │
│                                                                 │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │   Master Node    │              │   Worker Node    │         │
│  │                  │              │                  │         │
│  │  ┌─────────────┐ │              │  ┌─────────────┐ │         │
│  │  │ API Server  │ │              │  │   Kubelet   │ │         │
│  │  └─────────────┘ │              │  └─────────────┘ │         │
│  │  ┌─────────────┐ │              │  ┌─────────────┐ │         │
│  │  │   etcd      │ │              │  │ Kube Proxy  │ │         │
│  │  └─────────────┘ │              │  └─────────────┘ │         │
│  │  ┌─────────────┐ │              │  ┌─────────────┐ │         │
│  │  │ Scheduler   │ │              │  │ Container   │ │         │
│  │  └─────────────┘ │              │  └─────────────┘ │         │
│  │  ┌─────────────┐ │              │                  │         │
│  │  │ Controller  │ │              │     ┌──────┐     │         │
│  │  │ Manager     │ │              │     │ Pod  │     │         │
│  │  └─────────────┘ │              │     └──────┘     │         │
│  └──────────────────┘              │     ┌──────┐     │         │
│                                    └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

#### Core Kubernetes Resources

**1. Namespace (Environment Isolation):**
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: microservices-prod
  labels:
    environment: production
    team: backend
---
apiVersion: v1
kind: Namespace
metadata:
  name: microservices-dev
  labels:
    environment: development
    team: backend
```

**2. ConfigMap and Secrets:**
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: microservices-prod
data:
  database.host: "prod-db.example.com"
  database.port: "5432"
  api.timeout: "30s"
  log.level: "info"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: microservices-prod
type: Opaque
data:
  database.password: cGFzc3dvcmQxMjM=  # base64 encoded
  api.key: YWJjZGVmZ2hpams=
  jwt.secret: c3VwZXJzZWNyZXRrZXk=
```

**3. Deployment with Advanced Features:**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
  namespace: microservices-prod
  labels:
    app: user-service
    version: v1.2.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero-downtime deployment
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
        version: v1.2.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "3001"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: user-service-sa
      containers:
      - name: user-service
        image: your-registry.com/user-service:v1.2.0
        ports:
        - containerPort: 3001
          name: http
        env:
        - name: NODE_ENV
          value: "production"
        - name: DATABASE_HOST
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database.host
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database.password
        
        # Resource limits and requests
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 3001
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        # Graceful shutdown
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
        
        # Security context
        securityContext:
          runAsNonRoot: true
          runAsUser: 1001
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
```

**4. Service and Ingress:**
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: user-service
  namespace: microservices-prod
  labels:
    app: user-service
spec:
  selector:
    app: user-service
  ports:
  - port: 80
    targetPort: 3001
    protocol: TCP
    name: http
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: microservices-ingress
  namespace: microservices-prod
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls-secret
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /users
        pathType: Prefix
        backend:
          service:
            name: user-service
            port:
              number: 80
      - path: /orders
        pathType: Prefix
        backend:
          service:
            name: order-service
            port:
              number: 80
```

**5. Horizontal Pod Autoscaler:**
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: user-service-hpa
  namespace: microservices-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: user-service
  minReplicas: 3
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

## Learning Objectives

By the end of this section, you should be able to:

### Core Containerization Skills
- **Create production-ready Dockerfiles** with multi-stage builds, security best practices, and optimization
- **Compose multi-service applications** using Docker Compose for local development
- **Understand container networking** and volume management
- **Implement container security** including non-root users, read-only filesystems, and vulnerability scanning

### Kubernetes Orchestration Mastery
- **Deploy applications to Kubernetes** using Deployments, Services, ConfigMaps, and Secrets
- **Configure autoscaling** with HPA and VPA for optimal resource utilization
- **Implement service discovery** and load balancing within the cluster
- **Manage application lifecycle** including rolling updates, rollbacks, and health checks
- **Set up ingress controllers** for external access and SSL termination

### CI/CD Pipeline Development
- **Design automated pipelines** for testing, building, and deployment
- **Implement GitOps workflows** with proper branching strategies
- **Configure security scanning** and vulnerability assessment in pipelines
- **Set up multi-environment deployments** (dev, staging, production)

### Advanced Deployment Strategies
- **Implement blue-green deployments** for zero-downtime releases
- **Configure canary releases** with traffic splitting and automated rollback
- **Use feature flags** for progressive feature rollouts
- **Handle database migrations** in microservices environments

### Monitoring and Observability
- **Set up comprehensive monitoring** with Prometheus and Grafana
- **Implement distributed tracing** with Jaeger or Zipkin
- **Configure alerting rules** and incident response workflows
- **Monitor application and infrastructure metrics**

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can:

□ Write a multi-stage Dockerfile that produces a secure, minimal production image  
□ Deploy a multi-service application to Kubernetes with proper resource limits  
□ Set up a complete CI/CD pipeline that includes testing, security scanning, and deployment  
□ Configure horizontal pod autoscaling based on CPU and memory metrics  
□ Implement a canary deployment strategy with automated rollback  
□ Set up monitoring dashboards that track key application and infrastructure metrics  
□ Troubleshoot common deployment issues using kubectl and logs  
□ Implement network policies to secure inter-service communication  
□ Configure service mesh for advanced traffic management  
□ Handle rolling updates and rollbacks in production environments  

### Hands-on Exercises

#### Exercise 1: Production Dockerfile
Create a production-ready Dockerfile for a Node.js microservice:

```dockerfile
# TODO: Complete this Dockerfile with best practices
FROM node:18-alpine AS builder

# Your implementation here

FROM node:18-alpine AS production

# Your implementation here
```

**Requirements:**
- Multi-stage build
- Non-root user
- Health check
- Minimal attack surface
- Proper caching layers

#### Exercise 2: Kubernetes Deployment
Deploy a complete microservices stack to Kubernetes:

```yaml
# TODO: Create deployment, service, configmap, and secret
# For a user service that connects to PostgreSQL
```

**Requirements:**
- Resource limits and requests
- Liveness and readiness probes
- ConfigMap for configuration
- Secret for sensitive data
- HPA configuration

#### Exercise 3: CI/CD Pipeline
Create a complete GitLab CI/CD pipeline:

```yaml
# TODO: Implement stages for:
# - Testing (unit, integration, security)
# - Building Docker image
# - Deploying to dev/staging/prod
# - Rolling back on failure
```

**Requirements:**
- Parallel testing stages
- Docker layer caching
- Security scanning
- Environment-specific deployments
- Manual approval for production

#### Exercise 4: Monitoring Setup
Set up monitoring for your microservices:

```yaml
# TODO: Configure Prometheus, Grafana, and alerts
# Monitor: request rate, error rate, latency, resource usage
```

**Requirements:**
- Service discovery
- Custom metrics
- Alert rules
- Dashboard creation
- Log aggregation

### Practical Projects

#### Project 1: E-commerce Microservices Platform
Build and deploy a complete e-commerce platform with:
- User service (authentication, profiles)
- Product service (catalog, inventory)
- Order service (shopping cart, checkout)
- Payment service (payment processing)
- Notification service (emails, SMS)

**Technical Requirements:**
- Docker containers for all services
- Kubernetes deployment with proper networking
- CI/CD pipeline with multiple environments
- Service mesh for traffic management
- Comprehensive monitoring and alerting

#### Project 2: Chat Application with Real-time Features
Deploy a scalable chat application:
- WebSocket service for real-time messaging
- User management service
- Message persistence service
- File upload service
- Push notification service

**Technical Requirements:**
- Horizontal autoscaling based on connections
- Blue-green deployment strategy
- Canary releases for new features
- Distributed tracing
- Performance monitoring

#### Project 3: IoT Data Processing Pipeline
Create an IoT data processing system:
- Data ingestion service
- Stream processing service
- Data storage service
- Analytics service
- Dashboard service

**Technical Requirements:**
- Event-driven architecture
- High-throughput message processing
- Auto-scaling based on queue depth
- Disaster recovery setup
- Cost optimization strategies

## Best Practices Summary

### Docker Best Practices
✅ **DO:**
- Use multi-stage builds to reduce image size
- Run containers as non-root users
- Use specific base image tags, not `latest`
- Implement health checks
- Use `.dockerignore` to exclude unnecessary files
- Scan images for vulnerabilities regularly
- Use minimal base images (alpine, distroless)

❌ **DON'T:**
- Store secrets in Docker images
- Run as root user in production
- Install unnecessary packages
- Use mutable tags in production
- Include development dependencies in production images

### Kubernetes Best Practices
✅ **DO:**
- Set resource requests and limits
- Use namespaces for environment isolation
- Implement proper health checks
- Use ConfigMaps and Secrets for configuration
- Apply security contexts and pod security standards
- Use network policies for micro-segmentation
- Monitor resource usage and set up alerts

❌ **DON'T:**
- Deploy without resource limits
- Store sensitive data in plain text
- Use privileged containers
- Ignore security scanning results
- Deploy without proper testing

### CI/CD Best Practices
✅ **DO:**
- Implement automated testing at multiple levels
- Use immutable artifacts (Docker images)
- Implement security scanning in pipelines
- Use GitOps for deployment management
- Implement proper secret management
- Set up monitoring and alerting for pipelines
- Use feature flags for gradual rollouts

❌ **DON'T:**
- Deploy untested code
- Store secrets in pipeline scripts
- Skip security scanning
- Deploy directly to production without staging
- Ignore test failures

### Security Best Practices
✅ **DO:**
- Scan container images for vulnerabilities
- Use minimal base images
- Implement network policies
- Use service accounts with minimal permissions
- Encrypt secrets at rest and in transit
- Implement admission controllers
- Regular security audits

❌ **DON'T:**
- Run containers as root
- Use default service accounts
- Allow unrestricted network access
- Store secrets in plain text
- Skip security updates

## Troubleshooting Guide

### Common Docker Issues

**Issue: Container exits immediately**
```bash
# Debug container startup
docker run --rm -it your-image /bin/sh

# Check logs
docker logs container-id

# Inspect image
docker inspect your-image
```

**Issue: Large image size**
```bash
# Analyze image layers
docker history your-image

# Use dive for detailed analysis
dive your-image
```

### Common Kubernetes Issues

**Issue: Pod stuck in Pending state**
```bash
# Check pod events
kubectl describe pod pod-name

# Check node resources
kubectl top nodes

# Check node conditions
kubectl get nodes -o wide
```

**Issue: Service not reachable**
```bash
# Check service endpoints
kubectl get endpoints service-name

# Test service from within cluster
kubectl run test-pod --image=busybox --rm -it -- nslookup service-name

# Check network policies
kubectl get networkpolicies
```

**Issue: Pod crashes or restarts**
```bash
# Check pod logs
kubectl logs pod-name --previous

# Check resource usage
kubectl top pod pod-name

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

### Performance Optimization

**Container Optimization:**
- Use multi-stage builds
- Optimize layer caching
- Use .dockerignore effectively
- Choose appropriate base images

**Kubernetes Optimization:**
- Set appropriate resource requests/limits
- Use horizontal pod autoscaling
- Implement pod disruption budgets
- Optimize scheduling with node affinity

**Application Optimization:**
- Implement proper health checks
- Use graceful shutdown handling
- Optimize startup time
- Implement circuit breakers

## Study Materials

### Essential Reading
- **Primary:** "Kubernetes in Action" by Marko Lukša
- **Docker:** "Docker Deep Dive" by Nigel Poulton
- **CI/CD:** "Continuous Delivery" by Jez Humble and David Farley
- **Monitoring:** "Prometheus: Up & Running" by Brian Brazil

### Online Resources
- [Kubernetes Official Documentation](https://kubernetes.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [CNCF Cloud Native Landscape](https://landscape.cncf.io/)
- [12-Factor App Methodology](https://12factor.net/)

### Video Courses
- "Docker and Kubernetes: The Complete Guide" (Udemy)
- "Kubernetes for Developers" (Linux Foundation)
- "GitLab CI/CD Tutorial" (GitLab University)

### Hands-on Labs
- [Katacoda Kubernetes Scenarios](https://www.katacoda.com/courses/kubernetes)
- [Play with Docker](https://labs.play-with-docker.com/)
- [Kubernetes the Hard Way](https://github.com/kelseyhightower/kubernetes-the-hard-way)

### Certification Preparation
- **CKA (Certified Kubernetes Administrator)**
- **CKAD (Certified Kubernetes Application Developer)**
- **Docker Certified Associate**

### Tools to Learn
- **Container Tools:** Docker, Podman, Buildah
- **Orchestration:** Kubernetes, Docker Swarm
- **CI/CD:** GitLab CI, GitHub Actions, Jenkins, Tekton
- **Service Mesh:** Istio, Linkerd, Consul Connect
- **Monitoring:** Prometheus, Grafana, Jaeger, Fluentd
- **Security:** Twistlock, Aqua Security, Falco

## Next Steps

After mastering deployment and orchestration, consider exploring:

1. **Advanced Kubernetes Topics**
   - Custom Resource Definitions (CRDs)
   - Operators and controllers
   - Multi-cluster management

2. **Service Mesh Deep Dive**
   - Advanced traffic management
   - Security policies
   - Multi-cluster service mesh

3. **GitOps and Infrastructure as Code**
   - ArgoCD, Flux
   - Terraform, Pulumi
   - Configuration management

4. **Cloud-Native Security**
   - Zero-trust networking
   - Container runtime security
   - Compliance and governance

5. **Observability and Reliability**
   - SRE practices
   - Chaos engineering
   - Advanced monitoring patterns

---

