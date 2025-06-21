# Kubernetes Security Architecture

Kubernetes security is built on a defense-in-depth approach that provides multiple layers of protection for containerized applications. Understanding Kubernetes security architecture is crucial for deploying secure, compliant, and resilient applications in production environments.

## Security Overview

Kubernetes security encompasses several key areas:

### Core Security Principles

1. **Least Privilege**: Grant minimum necessary permissions
2. **Defense in Depth**: Multiple security layers and controls
3. **Zero Trust**: Never trust, always verify
4. **Immutable Infrastructure**: Treat containers as immutable
5. **Security by Default**: Secure configurations out of the box

### Security Boundaries

**Cluster Level**: Control plane security, node security, and cluster-wide policies
**Namespace Level**: Multi-tenant isolation and resource boundaries
**Pod Level**: Container security contexts and runtime protection
**Network Level**: Network policies and service mesh security

## RBAC (Role-Based Access Control)

Role-Based Access Control (RBAC) is the primary mechanism for controlling access to Kubernetes resources. RBAC uses roles and bindings to define who can perform what actions on which resources.

### RBAC Components

- **Role/ClusterRole**: Defines permissions (what can be done)
- **RoleBinding/ClusterRoleBinding**: Assigns roles to subjects (who can do it)
- **Subjects**: Users, groups, or service accounts

### Cluster Roles and Bindings

Cluster roles define permissions across the entire cluster or for cluster-scoped resources.

```yaml
# Cluster-wide read-only role
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-reader
  annotations:
    description: "Read-only access to most cluster resources"
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "statefulsets", "daemonsets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["networking.k8s.io"]
  resources: ["networkpolicies", "ingresses"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["pods", "nodes"]
  verbs: ["get", "list"]
---
# Bind cluster role to users and groups
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cluster-reader-binding
subjects:
- kind: User
  name: john.doe@company.com
  apiGroup: rbac.authorization.k8s.io
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
- kind: Group
  name: sre-team
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: cluster-reader
  apiGroup: rbac.authorization.k8s.io
```

**Key Features Explained:**
- **apiGroups**: Specifies which API groups the rule applies to
- **resources**: Lists the Kubernetes resources covered by the rule
- **verbs**: Defines the allowed actions (get, list, watch, create, update, patch, delete)
- **resourceNames**: Restricts access to specific named resources (optional)

### Namespace-Specific Roles

Namespace roles provide fine-grained permissions within specific namespaces.

```yaml
# Namespace-specific developer role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: development
  name: developer
  annotations:
    description: "Full access to development namespace resources"
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "persistentvolumeclaims"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list", "watch"]  # Read-only for secrets
- apiGroups: [""]
  resources: ["pods/exec", "pods/portforward", "pods/log"]
  verbs: ["create", "get"]
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
# Production environment role with restricted permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: production-deployer
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "update", "patch"]  # No create/delete in prod
- apiGroups: [""]
  resources: ["services", "configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
---
# Bind role to developers in development namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: developer-binding
  namespace: development
subjects:
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
- kind: User
  name: alice@company.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: developer
  apiGroup: rbac.authorization.k8s.io
```

### Service Account Configuration

Service accounts provide identity for processes running in pods and enable fine-grained access control for applications.

```yaml
# Service account for applications
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
  namespace: production
  annotations:
    description: "Service account for production applications"
automountServiceAccountToken: false  # Disable automatic token mounting for security
---
# Role for the service account with minimal permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: app-role
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["app-secrets", "database-secrets"]  # Specific secret access
  verbs: ["get"]
- apiGroups: [""]
  resources: ["services"]
  verbs: ["get"]
- apiGroups: [""]
  resources: ["endpoints"]
  verbs: ["get", "list", "watch"]
---
# Bind role to service account
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-role-binding
  namespace: production
subjects:
- kind: ServiceAccount
  name: app-service-account
  namespace: production
roleRef:
  kind: Role
  name: app-role
  apiGroup: rbac.authorization.k8s.io
---
# Pod using the service account
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  namespace: production
spec:
  serviceAccountName: app-service-account
  automountServiceAccountToken: true  # Explicitly mount token when needed
  containers:
  - name: app
    image: myapp:latest
    env:
    - name: KUBERNETES_SERVICE_ACCOUNT
      value: app-service-account
    volumeMounts:
    - name: service-account-token
      mountPath: /var/run/secrets/kubernetes.io/serviceaccount
      readOnly: true
  volumes:
  - name: service-account-token
    projected:
      sources:
      - serviceAccountToken:
          path: token
          expirationSeconds: 3600  # Token expires after 1 hour
      - configMap:
          name: kube-root-ca.crt
          items:
          - key: ca.crt
            path: ca.crt
      - downwardAPI:
          items:
          - path: namespace
            fieldRef:
              fieldPath: metadata.namespace
```

## Pod Security Standards

Pod Security Standards replace the deprecated PodSecurityPolicy and provide three security profiles: Privileged, Baseline, and Restricted.

### Security Profiles

**Privileged**: Unrestricted policy (no restrictions)
**Baseline**: Minimally restrictive policy (prevents known privilege escalations)
**Restricted**: Heavily restricted policy (follows current Pod hardening best practices)

### Restricted Pod Security Context

The restricted profile implements the most stringent security requirements.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
  namespace: restricted-namespace
  labels:
    app: secure-app
spec:
  securityContext:
    runAsNonRoot: true              # Pod must run as non-root user
    runAsUser: 1000                 # Specific non-root user ID
    runAsGroup: 3000                # Specific group ID
    fsGroup: 2000                   # File system group for volumes
    fsGroupChangePolicy: "OnRootMismatch"  # Change ownership only when needed
    seccompProfile:
      type: RuntimeDefault          # Use default seccomp profile
    supplementalGroups: [4000]      # Additional groups
  containers:
  - name: app
    image: nginx:1.21-alpine
    securityContext:
      allowPrivilegeEscalation: false    # Prevent privilege escalation
      readOnlyRootFilesystem: true       # Read-only root filesystem
      runAsNonRoot: true                 # Container must run as non-root
      runAsUser: 1000                    # Non-root user ID
      runAsGroup: 3000                   # Non-root group ID
      capabilities:
        drop:
        - ALL                            # Drop all capabilities
        add:
        - NET_BIND_SERVICE               # Only add necessary capabilities
      seccompProfile:
        type: RuntimeDefault             # Use default seccomp profile
    ports:
    - containerPort: 8080
      name: http
    resources:
      requests:
        memory: "128Mi"
        cpu: "100m"
      limits:
        memory: "256Mi"
        cpu: "200m"
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: var-cache
      mountPath: /var/cache/nginx
    - name: var-run
      mountPath: /var/run
    - name: config
      mountPath: /etc/nginx/conf.d
      readOnly: true
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
  volumes:
  - name: tmp
    emptyDir:
      sizeLimit: 1Gi
  - name: var-cache
    emptyDir:
      sizeLimit: 1Gi
  - name: var-run
    emptyDir:
      sizeLimit: 100Mi
  - name: config
    configMap:
      name: nginx-config
      defaultMode: 0444  # Read-only for owner, group, and others
```

### Namespace-Level Pod Security Standards

```yaml
# Enforce restricted security profile at namespace level
apiVersion: v1
kind: Namespace
metadata:
  name: secure-namespace
  labels:
    pod-security.kubernetes.io/enforce: restricted      # Enforce restricted profile
    pod-security.kubernetes.io/audit: restricted        # Audit against restricted profile  
    pod-security.kubernetes.io/warn: restricted         # Warn if violating restricted profile
    pod-security.kubernetes.io/enforce-version: latest  # Use latest policy version
  annotations:
    pod-security.kubernetes.io/enforce-policy: "restricted security policy enforced"
---
# Development namespace with baseline security
apiVersion: v1
kind: Namespace
metadata:
  name: development
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/audit: restricted        # Audit against stricter profile
    pod-security.kubernetes.io/warn: restricted         # Warn about security improvements
---
# Privileged namespace for system components
apiVersion: v1
kind: Namespace
metadata:
  name: system-privileged
  labels:
    pod-security.kubernetes.io/enforce: privileged
    pod-security.kubernetes.io/audit: baseline
    pod-security.kubernetes.io/warn: baseline
```

### Security Context Constraints (OpenShift)

For OpenShift environments, Security Context Constraints provide additional security controls.

```yaml
apiVersion: security.openshift.io/v1
kind: SecurityContextConstraints
metadata:
  name: restricted-v2
  annotations:
    description: "Highly restrictive SCC for production workloads"
allowHostDirVolumePlugin: false      # Disallow host directory volumes
allowHostIPC: false                  # Disallow host IPC namespace
allowHostNetwork: false              # Disallow host network namespace
allowHostPID: false                  # Disallow host PID namespace
allowHostPorts: false                # Disallow host port mapping
allowPrivilegedContainer: false      # Disallow privileged containers
allowedCapabilities: []              # No additional capabilities allowed
defaultAddCapabilities: []           # No default capabilities added
requiredDropCapabilities:            # Must drop all capabilities
- ALL
runAsUser:
  type: MustRunAsRange               # Must run in specified user range
  uidRangeMin: 1000
  uidRangeMax: 65535
seLinuxContext:
  type: MustRunAs                    # Must use SELinux context
fsGroup:
  type: MustRunAs                    # Must specify file system group
  ranges:
  - min: 1000
    max: 65535
supplementalGroups:
  type: MustRunAs
  ranges:
  - min: 1000
    max: 65535
volumes:                             # Allowed volume types
- configMap
- downwardAPI
- emptyDir
- persistentVolumeClaim
- projected
- secret
users:                               # Users allowed to use this SCC
- system:serviceaccount:production:restricted-app
groups:                              # Groups allowed to use this SCC
- system:authenticated
priority: 10                         # SCC selection priority
```

## Secret Management

Kubernetes secrets provide a mechanism to store and manage sensitive information such as passwords, OAuth tokens, and SSH keys.

### Types of Secrets

**Opaque**: Default secret type for arbitrary data
**kubernetes.io/service-account-token**: Service account tokens
**kubernetes.io/dockercfg**: Docker registry credentials
**kubernetes.io/tls**: TLS certificates and keys

### Basic Secret Creation

```yaml
# Opaque secret with multiple data types
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: production
  annotations:
    description: "Application configuration secrets"
type: Opaque
data:
  # Base64 encoded values
  username: YWRtaW4=              # 'admin'
  password: MWYyZDFlMmU2N2Rm      # secure password
  api-key: YWJjZGVmZ2hpams=       # API key
stringData:
  # Plain text values (automatically base64 encoded)
  database-url: "postgresql://user:pass@db.example.com:5432/myapp"
  config.yaml: |
    database:
      host: db.example.com
      port: 5432
      ssl_mode: require
    api:
      timeout: 30s
      retries: 3
    features:
      enable_metrics: true
      enable_tracing: true
---
# TLS secret for HTTPS
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
  namespace: production
  annotations:
    cert-manager.io/issuer: "letsencrypt-prod"
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi...         # Base64 encoded certificate
  tls.key: LS0tLS1CRUdJTi...         # Base64 encoded private key
---
# Docker registry authentication secret
apiVersion: v1
kind: Secret
metadata:
  name: docker-registry-secret
  namespace: production
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: ewogICJhdXRocyI6IHsKICAgICJyZWdpc3RyeS5leGFtcGxlLmNvbSI6IHsKICAgICAgInVzZXJuYW1lIjogInVzZXIiLAogICAgICAicGFzc3dvcmQiOiAicGFzcyIsCiAgICAgICJhdXRoIjogImRYTmxjanB3WVhOeiIKICAgIH0KICB9Cn0=
```

### External Secrets Operator

External Secrets Operator enables integration with external secret management systems like AWS Secrets Manager, HashiCorp Vault, and Azure Key Vault.

```yaml
# External secret using AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: app-secrets-external
  namespace: production
  annotations:
    description: "External secret from AWS Secrets Manager"
spec:
  refreshInterval: 5m              # Refresh every 5 minutes
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: app-secrets
    creationPolicy: Owner          # ESO owns the secret
    template:
      type: Opaque
      metadata:
        labels:
          managed-by: external-secrets-operator
      data:
        username: "{{ .username }}"
        password: "{{ .password }}"
        database-url: "postgresql://{{ .username }}:{{ .password }}@{{ .host }}:{{ .port }}/{{ .database }}"
        redis-url: "redis://{{ .redis_password }}@{{ .redis_host }}:{{ .redis_port }}"
  data:
  - secretKey: username
    remoteRef:
      key: prod/myapp/db
      property: username
  - secretKey: password
    remoteRef:
      key: prod/myapp/db
      property: password
  - secretKey: host
    remoteRef:
      key: prod/myapp/db
      property: host
  - secretKey: port
    remoteRef:
      key: prod/myapp/db
      property: port
  - secretKey: database
    remoteRef:
      key: prod/myapp/db
      property: database
  - secretKey: redis_password
    remoteRef:
      key: prod/myapp/redis
      property: password
  - secretKey: redis_host
    remoteRef:
      key: prod/myapp/redis
      property: host
  - secretKey: redis_port
    remoteRef:
      key: prod/myapp/redis
      property: port
---
# Secret store configuration for AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: production
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        secretRef:
          accessKeyID:
            name: aws-credentials
            key: access-key-id
          secretAccessKey:
            name: aws-credentials
            key: secret-access-key
---
# Cluster-wide secret store for shared secrets
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: vault-cluster-store
spec:
  provider:
    vault:
      server: "https://vault.company.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "external-secrets-operator"
          serviceAccountRef:
            name: external-secrets-sa
            namespace: external-secrets-system
```

### Sealed Secrets

Sealed Secrets allow storing encrypted secrets in Git repositories safely.

```yaml
# Sealed secret (encrypted secret that can be stored in Git)
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: app-secrets-sealed
  namespace: production
  annotations:
    sealedsecrets.bitnami.com/cluster-wide: "false"
    sealedsecrets.bitnami.com/namespace-wide: "false"
spec:
  encryptedData:
    username: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQAx...
    password: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQAx...
    api-key: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEQBx...
  template:
    metadata:
      name: app-secrets
      namespace: production
      labels:
        app: myapp
    type: Opaque
```

**Creating Sealed Secrets:**
```bash
# Install kubeseal client
curl -OL https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/kubeseal-linux-amd64
sudo install -m 755 kubeseal-linux-amd64 /usr/local/bin/kubeseal

# Create a sealed secret
echo -n mypassword | kubectl create secret generic mysecret --dry-run=client --from-file=password=/dev/stdin -o yaml | kubeseal -o yaml > mysealedsecret.yaml
```

## Network Security

Network security in Kubernetes involves controlling traffic flow between pods, namespaces, and external systems using network policies.

### Network Policy Fundamentals

Network policies are Kubernetes resources that control traffic flow at the IP address or port level. They act as a firewall for pods.

### Default Deny Policies

```yaml
# Default deny all ingress and egress traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
  annotations:
    description: "Default deny all traffic - explicit allow required"
spec:
  podSelector: {}                   # Applies to all pods in namespace
  policyTypes:
  - Ingress                        # Block all incoming traffic
  - Egress                         # Block all outgoing traffic
---
# Default deny ingress only (allow egress)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: development
spec:
  podSelector: {}
  policyTypes:
  - Ingress                        # Only block incoming traffic
```

### Micro-segmentation Policies

```yaml
# Allow frontend to backend communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-to-backend
  namespace: production
  annotations:
    description: "Allow frontend pods to communicate with backend on port 8080"
spec:
  podSelector:
    matchLabels:
      tier: backend                # Target backend pods
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: frontend           # Allow from frontend pods
    - namespaceSelector:
        matchLabels:
          name: api-gateway        # Allow from API gateway namespace
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 8443                   # HTTPS port
---
# Allow backend to database communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-to-database
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: database
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: backend
    ports:
    - protocol: TCP
      port: 5432                   # PostgreSQL
    - protocol: TCP
      port: 3306                   # MySQL
    - protocol: TCP
      port: 27017                  # MongoDB
---
# Cross-namespace communication policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cross-namespace-api
  namespace: api
spec:
  podSelector:
    matchLabels:
      app: api-server
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          environment: production  # Allow from production namespaces
    - namespaceSelector:
        matchLabels:
          environment: staging     # Allow from staging namespaces
    - podSelector:
        matchLabels:
          role: api-client
    ports:
    - protocol: TCP
      port: 80
    - protocol: TCP
      port: 443
```

### Egress Policies

```yaml
# Allow egress for DNS and external APIs
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns-and-external
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  # Allow DNS resolution
  - to: []
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
  # Allow HTTPS to external services
  - to: []
    ports:
    - protocol: TCP
      port: 443
  # Allow specific external service
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system       # Allow to kube-system namespace
    ports:
    - protocol: TCP
      port: 443
  # Allow communication to specific external CIDR
  - to:
    - ipBlock:
        cidr: 10.0.0.0/8
        except:
        - 10.0.1.0/24            # Except this subnet
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
---
# Strict egress policy for sensitive workloads
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restricted-egress
  namespace: production
spec:
  podSelector:
    matchLabels:
      security: high             # Apply to high-security pods
  policyTypes:
  - Egress
  egress:
  # Only allow DNS
  - to: []
    ports:
    - protocol: UDP
      port: 53
  # Only allow communication within namespace
  - to:
    - podSelector: {}            # All pods in same namespace
    ports:
    - protocol: TCP
      port: 8080
  # Allow communication to specific external service by IP
  - to:
    - ipBlock:
        cidr: 203.0.113.0/24     # Specific external service
    ports:
    - protocol: TCP
      port: 443
```

## Security Scanning and Compliance

### Container Image Scanning

```yaml
# Trivy security scanner job for comprehensive scanning
apiVersion: batch/v1
kind: Job
metadata:
  name: security-scan-comprehensive
  namespace: security
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trivy-scanner
        image: aquasec/trivy:latest
        command:
        - sh
        - -c
        - |
          set -e
          echo "Starting comprehensive security scan..."
          
          # Create report directory
          mkdir -p /reports
          
          # Scan container images for vulnerabilities
          echo "Scanning container images..."
          trivy image --format json --output /reports/image-vulnerabilities.json nginx:latest
          trivy image --format json --output /reports/app-image-scan.json myapp:latest
          
          # Scan Kubernetes manifests for misconfigurations
          echo "Scanning Kubernetes manifests..."
          trivy fs --format json --output /reports/manifest-scan.json /manifests/
          
          # Scan for configuration issues
          echo "Scanning configurations..."
          trivy config --format json --output /reports/config-scan.json /manifests/
          
          # Generate summary report
          echo "Generating summary..."
          cat > /reports/scan-summary.txt << EOF
          Security Scan Summary - $(date)
          ================================
          Image Vulnerabilities: $(cat /reports/image-vulnerabilities.json | jq '.Results[].Vulnerabilities | length')
          Configuration Issues: $(cat /reports/config-scan.json | jq '.Results[].Misconfigurations | length')
          Manifest Issues: $(cat /reports/manifest-scan.json | jq '.Results[].Misconfigurations | length')
          EOF
          
          echo "Security scan completed successfully"
        env:
        - name: TRIVY_CACHE_DIR
          value: /tmp/trivy-cache
        - name: TRIVY_DB_REPOSITORY
          value: ghcr.io/aquasecurity/trivy-db
        volumeMounts:
        - name: reports
          mountPath: /reports
        - name: manifests
          mountPath: /manifests
        - name: cache
          mountPath: /tmp/trivy-cache
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: reports
        persistentVolumeClaim:
          claimName: scan-reports-pvc
      - name: manifests
        configMap:
          name: k8s-manifests
      - name: cache
        emptyDir:
          sizeLimit: 1Gi
---
# CronJob for regular security scanning
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scheduled-security-scan
  namespace: security
spec:
  schedule: "0 2 * * *"           # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: scanner
            image: aquasec/trivy:latest
            command:
            - sh
            - -c
            - |
              # Daily security scan script
              trivy image --exit-code 1 --severity HIGH,CRITICAL myapp:latest
              trivy k8s --format json --output /reports/k8s-scan-$(date +%Y%m%d).json cluster
            volumeMounts:
            - name: reports
              mountPath: /reports
          volumes:
          - name: reports
            persistentVolumeClaim:
              claimName: scan-reports-pvc
```

### Falco Runtime Security

Falco provides runtime security monitoring and threat detection for Kubernetes.

```yaml
# Falco DaemonSet for runtime security monitoring
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: falco
  namespace: falco-system
  labels:
    app: falco
spec:
  selector:
    matchLabels:
      app: falco
  template:
    metadata:
      labels:
        app: falco
    spec:
      serviceAccountName: falco
      hostNetwork: true
      hostPID: true
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
        operator: Exists
      - effect: NoSchedule
        key: node-role.kubernetes.io/control-plane
        operator: Exists
      containers:
      - name: falco
        image: falcosecurity/falco:0.35.1
        securityContext:
          privileged: true
        args:
        - /usr/bin/falco
        - --cri=/run/containerd/containerd.sock
        - --k8s-api=https://kubernetes.default.svc.cluster.local
        - --k8s-api-cert=/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        - --k8s-api-token-file=/var/run/secrets/kubernetes.io/serviceaccount/token
        - --verbose
        env:
        - name: FALCO_GRPC_ENABLED
          value: "true"
        - name: FALCO_GRPC_BIND_ADDRESS
          value: "0.0.0.0:5060"
        ports:
        - name: grpc
          containerPort: 5060
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: boot
          mountPath: /host/boot
          readOnly: true
        - name: lib-modules
          mountPath: /host/lib/modules
          readOnly: true
        - name: usr
          mountPath: /host/usr
          readOnly: true
        - name: etc
          mountPath: /host/etc
          readOnly: true
        - name: containerd-sock
          mountPath: /run/containerd/containerd.sock
        - name: falco-config
          mountPath: /etc/falco
        - name: falco-rules
          mountPath: /etc/falco/rules.d
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8765
          initialDelaySeconds: 60
          timeoutSeconds: 5
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: boot
        hostPath:
          path: /boot
      - name: lib-modules
        hostPath:
          path: /lib/modules
      - name: usr
        hostPath:
          path: /usr
      - name: etc
        hostPath:
          path: /etc
      - name: containerd-sock
        hostPath:
          path: /run/containerd/containerd.sock
      - name: falco-config
        configMap:
          name: falco-config
      - name: falco-rules
        configMap:
          name: falco-rules
---
# Custom Falco rules ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: falco-rules
  namespace: falco-system
data:
  custom_rules.yaml: |
    - rule: Sensitive File Access
      desc: Detect access to sensitive files
      condition: >
        open_read and
        (fd.filename startswith /etc/passwd or
         fd.filename startswith /etc/shadow or
         fd.filename startswith /etc/ssh/ or
         fd.filename contains id_rsa or
         fd.filename contains id_dsa)
      output: >
        Sensitive file accessed (user=%user.name command=%proc.cmdline
        file=%fd.name container_id=%container.id image=%container.image.repository)
      priority: WARNING
      tags: [filesystem, sensitive]
    
    - rule: Kubernetes Secret Access
      desc: Detect access to Kubernetes secrets from containers
      condition: >
        open_read and
        fd.filename startswith /var/run/secrets/kubernetes.io/serviceaccount and
        not proc.name in (pause, kubelet, kube-proxy)
      output: >
        Kubernetes secret accessed (user=%user.name command=%proc.cmdline
        file=%fd.name container_id=%container.id image=%container.image.repository)
      priority: WARNING
      tags: [kubernetes, secrets]
```

## Admission Controllers

Admission controllers provide policy enforcement and mutation capabilities during the API request processing.

### Validating Admission Webhook

```yaml
# Validating admission webhook for security policies
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionWebhook
metadata:
  name: security-policy-validator
  annotations:
    description: "Validates pods against security policies"
webhooks:
- name: pod-security.company.com
  clientConfig:
    service:
      name: security-webhook-service
      namespace: security-system
      path: /validate-pods
    caBundle: LS0tLS1CRUdJTi...    # Base64 encoded CA certificate
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  - operations: ["CREATE", "UPDATE"]
    apiGroups: ["apps"]
    apiVersions: ["v1"]
    resources: ["deployments", "replicasets", "daemonsets", "statefulsets"]
  namespaceSelector:
    matchLabels:
      security-validation: enabled
  admissionReviewVersions: ["v1", "v1beta1"]
  sideEffects: None
  failurePolicy: Fail              # Fail closed for security
  timeoutSeconds: 10
---
# Security webhook service
apiVersion: v1
kind: Service
metadata:
  name: security-webhook-service
  namespace: security-system
spec:
  selector:
    app: security-webhook
  ports:
  - port: 443
    targetPort: 8443
    protocol: TCP
---
# Security webhook deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-webhook
  namespace: security-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: security-webhook
  template:
    metadata:
      labels:
        app: security-webhook
    spec:
      serviceAccountName: security-webhook
      containers:
      - name: webhook
        image: security-webhook:latest
        ports:
        - containerPort: 8443
          name: https
        env:
        - name: TLS_CERT_FILE
          value: /etc/certs/tls.crt
        - name: TLS_KEY_FILE
          value: /etc/certs/tls.key
        - name: LOG_LEVEL
          value: info
        volumeMounts:
        - name: certs
          mountPath: /etc/certs
          readOnly: true
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      volumes:
      - name: certs
        secret:
          secretName: security-webhook-certs
```

### Mutating Admission Webhook

```yaml
# Mutating admission webhook for security enhancement
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingAdmissionWebhook
metadata:
  name: security-enhancer
webhooks:
- name: security-enhancer.company.com
  clientConfig:
    service:
      name: security-enhancer-service
      namespace: security-system
      path: /mutate
    caBundle: LS0tLS1CRUdJTi...
  rules:
  - operations: ["CREATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  - operations: ["CREATE"]
    apiGroups: ["apps"]
    apiVersions: ["v1"]
    resources: ["deployments"]
  namespaceSelector:
    matchLabels:
      security-enhancement: enabled
  admissionReviewVersions: ["v1"]
  sideEffects: None
  failurePolicy: Fail
  reinvocationPolicy: Never        # Don't re-invoke after mutation
---
# Example of what the webhook might inject
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-enhancement-policy
  namespace: security-system
data:
  policy.yaml: |
    mutations:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
      resources:
        requests:
          memory: "64Mi"
          cpu: "50m"
        limits:
          memory: "128Mi"
          cpu: "100m"
      annotations:
        security.company.com/enhanced: "true"
        security.company.com/policy-version: "v1.0"
```

## Security Monitoring and Auditing

### Comprehensive Security Monitoring Script

```bash
#!/bin/bash
# Kubernetes security monitoring and auditing script

echo "=== Kubernetes Security Audit Report ==="
echo "Generated on: $(date)"
echo "Cluster: $(kubectl config current-context)"
echo

# Function to check and report security issues
check_security_issue() {
    local check_name="$1"
    local command="$2"
    local description="$3"
    
    echo "Checking: $check_name"
    echo "Description: $description"
    
    result=$(eval "$command" 2>/dev/null)
    if [[ -n "$result" ]]; then
        echo "⚠️  ISSUES FOUND:"
        echo "$result"
    else
        echo "✅ No issues found"
    fi
    echo "---"
}

# 1. Check for privileged pods
check_security_issue \
    "Privileged Pods" \
    "kubectl get pods --all-namespaces -o jsonpath='{range .items[?(@.spec.securityContext.privileged==true)]}{.metadata.namespace}{\"\\t\"}{.metadata.name}{\"\\n\"}{end}'" \
    "Pods running with privileged access"

# 2. Check for pods running as root
check_security_issue \
    "Root User Pods" \
    "kubectl get pods --all-namespaces -o jsonpath='{range .items[?(@.spec.securityContext.runAsUser==0)]}{.metadata.namespace}{\"\\t\"}{.metadata.name}{\"\\n\"}{end}'" \
    "Pods running as root user (UID 0)"

# 3. Check for pods with host network
check_security_issue \
    "Host Network Pods" \
    "kubectl get pods --all-namespaces -o jsonpath='{range .items[?(@.spec.hostNetwork==true)]}{.metadata.namespace}{\"\\t\"}{.metadata.name}{\"\\n\"}{end}'" \
    "Pods using host network namespace"

# 4. Check for pods with host PID
check_security_issue \
    "Host PID Pods" \
    "kubectl get pods --all-namespaces -o jsonpath='{range .items[?(@.spec.hostPID==true)]}{.metadata.namespace}{\"\\t\"}{.metadata.name}{\"\\n\"}{end}'" \
    "Pods using host PID namespace"

# 5. Check for containers with excessive capabilities
check_security_issue \
    "Excessive Capabilities" \
    "kubectl get pods --all-namespaces -o json | jq -r '.items[] | select(.spec.containers[]?.securityContext.capabilities.add[]? == \"SYS_ADMIN\" or .spec.containers[]?.securityContext.capabilities.add[]? == \"NET_ADMIN\") | \"\\(.metadata.namespace)\\t\\(.metadata.name)\"'" \
    "Containers with excessive Linux capabilities"

# 6. Check RBAC overprivileged service accounts
echo "Checking: Overprivileged Service Accounts"
echo "Description: Service accounts with cluster-admin or admin privileges"
kubectl get clusterrolebindings -o json | jq -r '
  .items[] | 
  select(.roleRef.name == "cluster-admin" or .roleRef.name == "admin") | 
  .subjects[]? | 
  select(.kind == "ServiceAccount") | 
  "\(.namespace // "cluster-wide")\t\(.name)\tBound to: \(.kind)/\(.name)"
' 2>/dev/null || echo "✅ No overprivileged service accounts found"
echo "---"

# 7. Check for default service accounts with secrets
check_security_issue \
    "Default Service Accounts with Secrets" \
    "kubectl get serviceaccounts --all-namespaces -o json | jq -r '.items[] | select(.metadata.name == \"default\" and (.secrets | length) > 0) | \"\\(.metadata.namespace)\\t\\(.metadata.name)\"'" \
    "Default service accounts with mounted secrets"

# 8. Check network policies
echo "Checking: Network Policy Coverage"
echo "Description: Namespaces without network policies"
all_namespaces=$(kubectl get namespaces -o jsonpath='{.items[*].metadata.name}')
for ns in $all_namespaces; do
    if [[ "$ns" != "kube-system" && "$ns" != "kube-public" && "$ns" != "kube-node-lease" ]]; then
        policies=$(kubectl get networkpolicies -n "$ns" --no-headers 2>/dev/null | wc -l)
        if [[ $policies -eq 0 ]]; then
            echo "⚠️  Namespace '$ns' has no network policies"
        fi
    fi
done
echo "---"

# 9. Check for exposed services
check_security_issue \
    "Exposed Services" \
    "kubectl get services --all-namespaces -o wide | grep -E '(LoadBalancer|NodePort)' | grep -v 'kube-system'" \
    "Services exposed outside the cluster"

# 10. Check Pod Security Standards
echo "Checking: Pod Security Standards"
echo "Description: Namespaces without Pod Security Standards labels"
kubectl get namespaces -o json | jq -r '
  .items[] | 
  select(.metadata.labels["pod-security.kubernetes.io/enforce"] == null) | 
  select(.metadata.name != "kube-system" and .metadata.name != "kube-public" and .metadata.name != "kube-node-lease") |
  .metadata.name
' | while read ns; do
    echo "⚠️  Namespace '$ns' missing Pod Security Standards labels"
done
echo "---"

# 11. Check for secrets in environment variables
echo "Checking: Secrets in Environment Variables"
echo "Description: Pods with potential secrets in environment variables"
kubectl get pods --all-namespaces -o json | jq -r '
  .items[] | 
  .spec.containers[] | 
  select(.env[]?.name | test("(?i)(password|secret|key|token)")) |
  "\(.env[] | select(.name | test("(?i)(password|secret|key|token)")) | .name)"
' 2>/dev/null | sort -u | while read secret_env; do
    echo "⚠️  Found potential secret in environment variable: $secret_env"
done
echo "---"

# 12. Check image pull policies
echo "Checking: Image Pull Policies"
echo "Description: Containers without explicit pull policies"
kubectl get pods --all-namespaces -o json | jq -r '
  .items[] | 
  select(.spec.containers[] | .imagePullPolicy == null or .imagePullPolicy == "" or .imagePullPolicy == "Always") |
  "\(.metadata.namespace)\t\(.metadata.name)"
' 2>/dev/null | head -10 | while read line; do
    echo "⚠️  $line"
done
echo "---"

# Summary
echo "=== Security Audit Summary ==="
echo "Audit completed at: $(date)"
echo "Review all identified issues and implement necessary security measures."
echo "Consider running this audit regularly as part of your security monitoring."
echo
```

This comprehensive Kubernetes Security Architecture guide provides the foundation for implementing robust security controls in Kubernetes environments. Regular security audits, proper RBAC configuration, network policies, and continuous monitoring are essential for maintaining a secure container platform.

## Next Section

Continue to [Advanced Networking](08_Advanced_Networking.md) to learn about ingress controllers, load balancing patterns, and advanced networking concepts.
