# Project 2: Microservices Platform

## Overview
Design a comprehensive platform for microservices deployment with CI/CD pipelines, service mesh integration, and developer self-service capabilities.

## Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                    Developer Self-Service Portal                │
│                     (Kubernetes Dashboard +                     │
│                      Custom Web Interface)                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                        CI/CD Pipeline                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │   GitLab    │  │   Jenkins   │  │   ArgoCD    │  │ Tekton  ││
│  │     CI      │  │   Pipeline  │  │  (GitOps)   │  │  Tasks  ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   Service Mesh (Istio)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Gateway   │  │ Virtual Svc │  │ Destination │              │
│  │             │  │             │  │    Rules    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  Kubernetes Workloads                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │ User Service│  │Order Service│  │Pay  Service │  │Inventory││
│  │             │  │             │  │             │  │ Service ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Data Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
│  │ PostgreSQL  │  │    Redis    │  │  MongoDB    │  │  Kafka  ││
│  │             │  │             │  │             │  │         ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Platform Foundation

#### Namespace Structure
```yaml
# Platform namespaces
apiVersion: v1
kind: Namespace
metadata:
  name: platform-system
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
apiVersion: v1
kind: Namespace
metadata:
  name: istio-system
  labels:
    istio-injection: disabled
---
apiVersion: v1
kind: Namespace
metadata:
  name: cicd
  labels:
    pod-security.kubernetes.io/enforce: baseline
    istio-injection: enabled
---
apiVersion: v1
kind: Namespace
metadata:
  name: microservices-dev
  labels:
    pod-security.kubernetes.io/enforce: baseline
    istio-injection: enabled
    environment: development
---
apiVersion: v1
kind: Namespace
metadata:
  name: microservices-staging
  labels:
    pod-security.kubernetes.io/enforce: restricted
    istio-injection: enabled
    environment: staging
---
apiVersion: v1
kind: Namespace
metadata:
  name: microservices-prod
  labels:
    pod-security.kubernetes.io/enforce: restricted
    istio-injection: enabled
    environment: production
```

#### Platform Resource Quotas
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: platform-quota
  namespace: microservices-dev
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
    pods: "50"
    services: "10"
    secrets: "20"
    configmaps: "20"
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: platform-quota
  namespace: microservices-staging
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "20"
    pods: "100"
    services: "20"
    secrets: "30"
    configmaps: "30"
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: platform-quota
  namespace: microservices-prod
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 100Gi
    limits.cpu: "100"
    limits.memory: 200Gi
    persistentvolumeclaims: "50"
    pods: "200"
    services: "50"
    secrets: "50"
    configmaps: "50"
```

### 2. Service Mesh Implementation (Istio)

#### Istio Installation
```bash
#!/bin/bash
# install-istio.sh

set -e

ISTIO_VERSION="1.18.0"

# Download and install Istio
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=${ISTIO_VERSION} sh -
cd istio-${ISTIO_VERSION}
export PATH=$PWD/bin:$PATH

# Install Istio with configuration
istioctl install --set values.pilot.traceSampling=1.0 --set values.global.meshID=mesh1 --set values.global.multiCluster.clusterName=cluster1 --set values.global.network=network1 -y

# Enable Istio injection for microservices namespaces
kubectl label namespace microservices-dev istio-injection=enabled
kubectl label namespace microservices-staging istio-injection=enabled
kubectl label namespace microservices-prod istio-injection=enabled

echo "Istio installation completed"
```

#### Istio Gateway Configuration
```yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: microservices-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*.microservices.local"
    tls:
      httpsRedirect: true
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: microservices-tls
    hosts:
    - "*.microservices.local"
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: microservices-routing
  namespace: microservices-prod
spec:
  hosts:
  - "api.microservices.local"
  gateways:
  - istio-system/microservices-gateway
  http:
  - match:
    - uri:
        prefix: /api/v1/users
    route:
    - destination:
        host: user-service
        port:
          number: 8080
  - match:
    - uri:
        prefix: /api/v1/orders
    route:
    - destination:
        host: order-service
        port:
          number: 8080
  - match:
    - uri:
        prefix: /api/v1/payments
    route:
    - destination:
        host: payment-service
        port:
          number: 8080
  - match:
    - uri:
        prefix: /api/v1/inventory
    route:
    - destination:
        host: inventory-service
        port:
          number: 8080
```

#### Service Mesh Security Policies
```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: microservices-prod
spec:
  mtls:
    mode: STRICT
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-microservices
  namespace: microservices-prod
spec:
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/microservices-prod/sa/user-service"]
    to:
    - operation:
        methods: ["GET", "POST"]
    when:
    - key: request.headers[authorization]
      values: ["Bearer *"]
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: microservices-prod
spec:
  action: DENY
  rules:
  - from:
    - source:
        notPrincipals: ["cluster.local/ns/microservices-prod/sa/*"]
```

### 3. CI/CD Pipeline Implementation

#### GitLab CI Configuration
```yaml
# .gitlab-ci.yml
stages:
  - validate
  - build
  - test
  - security-scan
  - deploy-dev
  - integration-test
  - deploy-staging
  - deploy-prod

variables:
  DOCKER_REGISTRY: "registry.gitlab.com/company/microservices"
  KUBE_NAMESPACE_DEV: "microservices-dev"
  KUBE_NAMESPACE_STAGING: "microservices-staging"
  KUBE_NAMESPACE_PROD: "microservices-prod"

before_script:
  - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY

validate:
  stage: validate
  image: alpine/helm:latest
  script:
    - helm lint ./helm/microservice
    - kubectl --dry-run=client apply -f k8s/
  only:
    - merge_requests
    - main

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $DOCKER_REGISTRY/$CI_PROJECT_NAME:$CI_COMMIT_SHA .
    - docker tag $DOCKER_REGISTRY/$CI_PROJECT_NAME:$CI_COMMIT_SHA $DOCKER_REGISTRY/$CI_PROJECT_NAME:latest
    - docker push $DOCKER_REGISTRY/$CI_PROJECT_NAME:$CI_COMMIT_SHA
    - docker push $DOCKER_REGISTRY/$CI_PROJECT_NAME:latest

test:
  stage: test
  image: node:16
  script:
    - npm install
    - npm run test:unit
    - npm run test:integration
  coverage: '/Coverage: \d+\.\d+%/'
  artifacts:
    reports:
      junit: junit.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

security-scan:
  stage: security-scan
  image: aquasec/trivy:latest
  script:
    - trivy image --exit-code 1 --severity HIGH,CRITICAL $DOCKER_REGISTRY/$CI_PROJECT_NAME:$CI_COMMIT_SHA
  allow_failure: false

deploy-dev:
  stage: deploy-dev
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_DEV
    - helm upgrade --install $CI_PROJECT_NAME ./helm/microservice 
        --namespace $KUBE_NAMESPACE_DEV
        --set image.repository=$DOCKER_REGISTRY/$CI_PROJECT_NAME
        --set image.tag=$CI_COMMIT_SHA
        --set environment=development
        --wait
  environment:
    name: development
    url: https://$CI_PROJECT_NAME-dev.microservices.local
  only:
    - main

integration-test:
  stage: integration-test
  image: postman/newman:latest
  script:
    - newman run tests/integration/postman_collection.json 
        --environment tests/integration/dev_environment.json
        --reporters cli,junit
        --reporter-junit-export newman-results.xml
  artifacts:
    reports:
      junit: newman-results.xml
  dependencies:
    - deploy-dev

deploy-staging:
  stage: deploy-staging
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_STAGING
    - helm upgrade --install $CI_PROJECT_NAME ./helm/microservice 
        --namespace $KUBE_NAMESPACE_STAGING
        --set image.repository=$DOCKER_REGISTRY/$CI_PROJECT_NAME
        --set image.tag=$CI_COMMIT_SHA
        --set environment=staging
        --set replicaCount=2
        --wait
  environment:
    name: staging
    url: https://$CI_PROJECT_NAME-staging.microservices.local
  when: manual
  only:
    - main

deploy-prod:
  stage: deploy-prod
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_PROD
    - helm upgrade --install $CI_PROJECT_NAME ./helm/microservice 
        --namespace $KUBE_NAMESPACE_PROD
        --set image.repository=$DOCKER_REGISTRY/$CI_PROJECT_NAME
        --set image.tag=$CI_COMMIT_SHA
        --set environment=production
        --set replicaCount=3
        --set resources.requests.cpu=500m
        --set resources.requests.memory=512Mi
        --set resources.limits.cpu=1000m
        --set resources.limits.memory=1Gi
        --wait
  environment:
    name: production
    url: https://$CI_PROJECT_NAME.microservices.local
  when: manual
  only:
    - main
```

#### ArgoCD GitOps Configuration
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: user-service
  namespace: argocd
spec:
  project: microservices
  source:
    repoURL: https://gitlab.com/company/microservices/user-service
    targetRevision: main
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: microservices-prod
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
  revisionHistoryLimit: 10
---
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: microservices
  namespace: argocd
spec:
  description: Microservices platform project
  sourceRepos:
  - 'https://gitlab.com/company/microservices/*'
  destinations:
  - namespace: 'microservices-*'
    server: https://kubernetes.default.svc
  clusterResourceWhitelist:
  - group: ''
    kind: Namespace
  namespaceResourceWhitelist:
  - group: 'apps'
    kind: Deployment
  - group: ''
    kind: Service
  - group: ''
    kind: ConfigMap
  - group: ''
    kind: Secret
  - group: 'networking.k8s.io'
    kind: Ingress
  - group: 'networking.istio.io'
    kind: VirtualService
  - group: 'networking.istio.io'
    kind: DestinationRule
  roles:
  - name: admin
    description: Admin access to microservices
    policies:
    - p, proj:microservices:admin, applications, *, microservices/*, allow
    - p, proj:microservices:admin, repositories, *, *, allow
    groups:
    - company:microservices-admins
  - name: developer
    description: Developer access to microservices
    policies:
    - p, proj:microservices:developer, applications, get, microservices/*, allow
    - p, proj:microservices:developer, applications, sync, microservices/*, allow
    groups:
    - company:developers
```

### 4. Microservice Template

#### Helm Chart Template
```yaml
# helm/microservice/Chart.yaml
apiVersion: v2
name: microservice
description: A Helm chart for microservices
type: application
version: 0.1.0
appVersion: "1.0.0"

---
# helm/microservice/values.yaml
replicaCount: 1

image:
  repository: nginx
  pullPolicy: IfNotPresent
  tag: ""

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  fsGroup: 2000
  runAsNonRoot: true
  runAsUser: 1000

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000

service:
  type: ClusterIP
  port: 8080
  targetPort: 8080

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: chart-example.local
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 100
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}

# Microservice specific configurations
environment: development

config:
  database:
    host: "postgresql"
    port: 5432
    name: "microservice_db"
  redis:
    host: "redis"
    port: 6379
  kafka:
    brokers: "kafka:9092"

secrets:
  database:
    username: "user"
    password: "password"
  jwt:
    secret: "jwt-secret"

monitoring:
  enabled: true
  port: 9090
  path: /metrics

healthcheck:
  enabled: true
  liveness:
    path: /health/live
    port: 8080
    initialDelaySeconds: 30
    periodSeconds: 10
  readiness:
    path: /health/ready
    port: 8080
    initialDelaySeconds: 5
    periodSeconds: 5

istio:
  enabled: true
  gateway: "istio-system/microservices-gateway"
```

#### Microservice Deployment Template
```yaml
# helm/microservice/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "microservice.fullname" . }}
  labels:
    {{- include "microservice.labels" . | nindent 4 }}
    version: {{ .Values.image.tag | default .Chart.AppVersion }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "microservice.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
        {{- with .Values.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      labels:
        {{- include "microservice.selectorLabels" . | nindent 8 }}
        version: {{ .Values.image.tag | default .Chart.AppVersion }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "microservice.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
            {{- if .Values.monitoring.enabled }}
            - name: metrics
              containerPort: {{ .Values.monitoring.port }}
              protocol: TCP
            {{- end }}
          env:
            - name: ENVIRONMENT
              value: {{ .Values.environment }}
            - name: SERVICE_NAME
              value: {{ include "microservice.fullname" . }}
            - name: DATABASE_HOST
              valueFrom:
                configMapKeyRef:
                  name: {{ include "microservice.fullname" . }}-config
                  key: database-host
            - name: DATABASE_PORT
              valueFrom:
                configMapKeyRef:
                  name: {{ include "microservice.fullname" . }}-config
                  key: database-port
            - name: DATABASE_NAME
              valueFrom:
                configMapKeyRef:
                  name: {{ include "microservice.fullname" . }}-config
                  key: database-name
            - name: DATABASE_USERNAME
              valueFrom:
                secretKeyRef:
                  name: {{ include "microservice.fullname" . }}-secrets
                  key: database-username
            - name: DATABASE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {{ include "microservice.fullname" . }}-secrets
                  key: database-password
            - name: JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: {{ include "microservice.fullname" . }}-secrets
                  key: jwt-secret
          {{- if .Values.healthcheck.enabled }}
          livenessProbe:
            httpGet:
              path: {{ .Values.healthcheck.liveness.path }}
              port: {{ .Values.healthcheck.liveness.port }}
            initialDelaySeconds: {{ .Values.healthcheck.liveness.initialDelaySeconds }}
            periodSeconds: {{ .Values.healthcheck.liveness.periodSeconds }}
          readinessProbe:
            httpGet:
              path: {{ .Values.healthcheck.readiness.path }}
              port: {{ .Values.healthcheck.readiness.port }}
            initialDelaySeconds: {{ .Values.healthcheck.readiness.initialDelaySeconds }}
            periodSeconds: {{ .Values.healthcheck.readiness.periodSeconds }}
          {{- end }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /app/cache
      volumes:
        - name: tmp
          emptyDir: {}
        - name: cache
          emptyDir: {}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

### 5. Sample Microservices

#### User Service Implementation
```typescript
// user-service/src/app.ts
import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import prometheus from 'prom-client';
import { Pool } from 'pg';

const app = express();
const port = process.env.PORT || 8080;

// Metrics
const register = new prometheus.Registry();
const httpRequestsTotal = new prometheus.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'],
});
const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route'],
});

register.registerMetric(httpRequestsTotal);
register.registerMetric(httpRequestDuration);

// Database connection
const pool = new Pool({
  host: process.env.DATABASE_HOST,
  port: parseInt(process.env.DATABASE_PORT || '5432'),
  database: process.env.DATABASE_NAME,
  user: process.env.DATABASE_USERNAME,
  password: process.env.DATABASE_PASSWORD,
});

app.use(express.json());

// Middleware for metrics
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestsTotal.labels(req.method, req.route?.path || req.path, res.statusCode.toString()).inc();
    httpRequestDuration.labels(req.method, req.route?.path || req.path).observe(duration);
  });
  next();
});

// Health checks
app.get('/health/live', (req, res) => {
  res.status(200).json({ status: 'alive', timestamp: new Date().toISOString() });
});

app.get('/health/ready', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.status(200).json({ status: 'ready', timestamp: new Date().toISOString() });
  } catch (error) {
    res.status(503).json({ status: 'not ready', error: error.message });
  }
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

// User routes
app.get('/api/v1/users', async (req, res) => {
  try {
    const result = await pool.query('SELECT id, username, email, created_at FROM users');
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/v1/users/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const result = await pool.query('SELECT id, username, email, created_at FROM users WHERE id = $1', [id]);
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json(result.rows[0]);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/v1/users', async (req, res) => {
  try {
    const { username, email, password } = req.body;
    const result = await pool.query(
      'INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3) RETURNING id, username, email, created_at',
      [username, email, password] // In production, hash the password
    );
    res.status(201).json(result.rows[0]);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`User service listening on port ${port}`);
});
```

#### User Service Dockerfile
```dockerfile
# user-service/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine AS runtime

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

WORKDIR /app

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --chown=nextjs:nodejs . .

# Security: remove unnecessary packages
RUN apk del curl wget

USER nextjs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node healthcheck.js

CMD ["node", "dist/app.js"]
```

### 6. Developer Self-Service Platform

#### Platform API
```typescript
// platform-api/src/app.ts
import express from 'express';
import { K8sApi } from './k8s-api';
import { GitLabAPI } from './gitlab-api';
import { HelmClient } from './helm-client';

const app = express();
const k8s = new K8sApi();
const gitlab = new GitLabAPI();
const helm = new HelmClient();

app.use(express.json());

// Create new microservice
app.post('/api/v1/microservices', async (req, res) => {
  try {
    const { name, template, namespace } = req.body;
    
    // 1. Create GitLab project from template
    const project = await gitlab.createProject({
      name,
      template_project_id: template,
      namespace_id: namespace
    });
    
    // 2. Create Kubernetes namespace if not exists
    await k8s.createNamespace(name);
    
    // 3. Create initial Helm deployment
    await helm.install(name, {
      chart: './helm/microservice',
      namespace: name,
      values: {
        image: {
          repository: project.container_registry_url,
          tag: 'latest'
        }
      }
    });
    
    res.status(201).json({
      message: 'Microservice created successfully',
      project: project,
      namespace: name
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get microservice status
app.get('/api/v1/microservices/:name', async (req, res) => {
  try {
    const { name } = req.params;
    
    const deployment = await k8s.getDeployment(name, name);
    const service = await k8s.getService(name, name);
    const pods = await k8s.getPods(name, { app: name });
    
    res.json({
      name,
      deployment: {
        replicas: deployment.status.replicas,
        readyReplicas: deployment.status.readyReplicas,
        updatedReplicas: deployment.status.updatedReplicas
      },
      service: {
        clusterIP: service.spec.clusterIP,
        ports: service.spec.ports
      },
      pods: pods.items.map(pod => ({
        name: pod.metadata.name,
        status: pod.status.phase,
        ready: pod.status.conditions?.find(c => c.type === 'Ready')?.status === 'True'
      }))
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Scale microservice
app.put('/api/v1/microservices/:name/scale', async (req, res) => {
  try {
    const { name } = req.params;
    const { replicas } = req.body;
    
    await k8s.scaleDeployment(name, name, replicas);
    
    res.json({
      message: `Microservice ${name} scaled to ${replicas} replicas`
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get microservice logs
app.get('/api/v1/microservices/:name/logs', async (req, res) => {
  try {
    const { name } = req.params;
    const { lines = 100 } = req.query;
    
    const logs = await k8s.getPodLogs(name, name, { tailLines: lines });
    
    res.json({ logs });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => {
  console.log('Platform API listening on port 3000');
});
```

### 7. Monitoring and Observability

#### Service Monitor for Microservices
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: microservices-monitor
  namespace: monitoring
  labels:
    app: microservices
spec:
  selector:
    matchLabels:
      monitoring: enabled
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
  namespaceSelector:
    matchNames:
    - microservices-dev
    - microservices-staging
    - microservices-prod
```

#### Grafana Dashboard for Microservices
```json
{
  "dashboard": {
    "id": null,
    "title": "Microservices Platform",
    "tags": ["microservices", "platform"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "HTTP Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ]
      },
      {
        "id": 2,
        "title": "HTTP Request Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "{{service}} 95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Pod Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(container_memory_usage_bytes{namespace=~\"microservices-.*\"}) by (pod)",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Pod CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total{namespace=~\"microservices-.*\"}[5m])) by (pod)",
            "legendFormat": "{{pod}}"
          }
        ]
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

This comprehensive microservices platform provides:

1. **Service Mesh Integration**: Complete Istio setup with security policies
2. **CI/CD Pipelines**: GitLab CI with ArgoCD GitOps workflow
3. **Developer Self-Service**: API and web interface for microservice management
4. **Template-Based Development**: Helm charts for consistent microservice deployment
5. **Monitoring & Observability**: Prometheus metrics and Grafana dashboards
6. **Security**: Pod Security Standards, Network Policies, and mTLS
7. **Scalability**: Auto-scaling and resource management

The platform enables developers to quickly deploy, manage, and monitor microservices while maintaining consistency and security across environments.
