# Kubernetes Networking

Kubernetes networking is one of the most complex aspects of the platform, providing the foundation for pod-to-pod communication, service discovery, and external access. Understanding Kubernetes networking is crucial for building scalable, secure, and performant applications.

## Kubernetes Network Model Fundamentals

Kubernetes implements a flat network model with the following key principles:

### Core Networking Requirements

1. **All pods can communicate with all other pods** without Network Address Translation (NAT)
2. **All nodes can communicate with all pods** (and vice versa) without NAT
3. **The IP that a pod sees itself as** is the same IP that others see it as

This model simplifies networking complexity and provides a consistent environment across different infrastructure providers.

### Network Layers in Kubernetes

**Pod Network**: Every pod gets its own IP address from the cluster's pod CIDR range. Containers within a pod share the same network namespace, including IP address and port space.

**Service Network**: Services provide stable endpoints for accessing pods through virtual IP addresses (ClusterIPs) that are managed by kube-proxy.

**Node Network**: The underlying network that connects Kubernetes nodes, typically using the infrastructure provider's networking.

## Container Network Interface (CNI)

The Container Network Interface (CNI) is a specification and set of libraries for configuring network interfaces in Linux containers. Kubernetes uses CNI plugins to implement its networking model.

### CNI Plugin Architecture

CNI plugins are responsible for:
- Assigning IP addresses to pods
- Setting up network interfaces
- Configuring routing rules
- Implementing network policies
- Managing network security

### Popular CNI Plugin Implementations

#### Calico - Policy-Rich Networking

Calico provides both networking and network policy for Kubernetes. It uses BGP routing and can operate in multiple modes.

**Calico Installation Manifest:**

```yaml
# Calico CNI configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: calico-config
  namespace: kube-system
data:
  # Cluster-wide Calico configuration
  cni_network_config: |-
    {
      "name": "k8s-pod-network",
      "cniVersion": "0.3.1",
      "plugins": [
        {
          "type": "calico",
          "log_level": "info",
          "log_file_path": "/var/log/calico/cni/cni.log",
          "datastore_type": "kubernetes",
          "nodename": "__KUBERNETES_NODE_NAME__",
          "mtu": "__CNI_MTU__",
          "ipam": {
            "type": "calico-ipam"
          },
          "policy": {
            "type": "k8s"
          },
          "kubernetes": {
            "kubeconfig": "__KUBECONFIG_FILEPATH__"
          }
        },
        {
          "type": "portmap",
          "snat": true,
          "capabilities": {"portMappings": true}
        }
      ]
    }
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: calico-node
  namespace: kube-system
  labels:
    k8s-app: calico-node
spec:
  selector:
    matchLabels:
      k8s-app: calico-node
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  template:
    metadata:
      labels:
        k8s-app: calico-node
    spec:
      nodeSelector:
        kubernetes.io/os: linux
      hostNetwork: true
      tolerations:
      - effect: NoSchedule
        operator: Exists
      - key: CriticalAddonsOnly
        operator: Exists
      - effect: NoExecute
        operator: Exists
      serviceAccountName: calico-node
      terminationGracePeriodSeconds: 0
      priorityClassName: system-node-critical
      containers:
      - name: calico-node
        image: calico/node:v3.26.1
        env:
        - name: DATASTORE_TYPE
          value: "kubernetes"
        - name: WAIT_FOR_DATASTORE
          value: "true"
        - name: NODENAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        - name: CALICO_NETWORKING_BACKEND
          valueFrom:
            configMapKeyRef:
              name: calico-config
              key: calico_backend
        - name: CLUSTER_TYPE
          value: "k8s,bgp"
        - name: IP
          value: "autodetect"
        - name: CALICO_IPV4POOL_IPIP
          value: "Always"
        - name: CALICO_IPV4POOL_VXLAN
          value: "Never"
        - name: FELIX_IPINIPMTU
          valueFrom:
            configMapKeyRef:
              name: calico-config
              key: veth_mtu
        - name: FELIX_VXLANMTU
          valueFrom:
            configMapKeyRef:
              name: calico-config
              key: veth_mtu
        - name: FELIX_WIREGUARDMTU
          valueFrom:
            configMapKeyRef:
              name: calico-config
              key: veth_mtu
        securityContext:
          privileged: true
        resources:
          requests:
            cpu: 250m
        volumeMounts:
        - mountPath: /lib/modules
          name: lib-modules
          readOnly: true
        - mountPath: /run/xtables.lock
          name: xtables-lock
          readOnly: false
        - mountPath: /var/run/calico
          name: var-run-calico
          readOnly: false
        - mountPath: /var/lib/calico
          name: var-lib-calico
          readOnly: false
        - name: policysync
          mountPath: /var/run/nodeagent
      volumes:
      - name: lib-modules
        hostPath:
          path: /lib/modules
      - name: var-run-calico
        hostPath:
          path: /var/run/calico
      - name: var-lib-calico
        hostPath:
          path: /var/lib/calico
      - name: xtables-lock
        hostPath:
          path: /run/xtables.lock
          type: FileOrCreate
      - name: policysync
        hostPath:
          path: /var/run/nodeagent
          type: DirectoryOrCreate
```

#### Flannel - Simple Overlay Network

Flannel creates a simple overlay network that provides each node with a subnet for pod networking.

**Flannel Configuration:**

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: kube-flannel-cfg
  namespace: kube-system
  labels:
    tier: node
    app: flannel
data:
  cni-conf.json: |
    {
      "name": "cbr0",
      "cniVersion": "0.3.1",
      "plugins": [
        {
          "type": "flannel",
          "delegate": {
            "hairpinMode": true,
            "isDefaultGateway": true
          }
        },
        {
          "type": "portmap",
          "capabilities": {
            "portMappings": true
          }
        }
      ]
    }
  net-conf.json: |
    {
      "Network": "10.244.0.0/16",
      "Backend": {
        "Type": "vxlan"
      }
    }
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: kube-flannel-ds
  namespace: kube-system
  labels:
    tier: node
    app: flannel
spec:
  selector:
    matchLabels:
      app: flannel
  template:
    metadata:
      labels:
        tier: node
        app: flannel
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/os
                operator: In
                values:
                - linux
      hostNetwork: true
      priorityClassName: system-node-critical
      tolerations:
      - operator: Exists
        effect: NoSchedule
      serviceAccountName: flannel
      containers:
      - name: kube-flannel
        image: quay.io/coreos/flannel:v0.22.0
        command:
        - /opt/bin/flanneld
        args:
        - --ip-masq
        - --kube-subnet-mgr
        resources:
          requests:
            cpu: "100m"
            memory: "50Mi"
          limits:
            cpu: "100m"
            memory: "50Mi"
        securityContext:
          privileged: false
          capabilities:
            add: ["NET_ADMIN", "NET_RAW"]
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        volumeMounts:
        - name: run
          mountPath: /run/flannel
        - name: flannel-cfg
          mountPath: /etc/kube-flannel/
      volumes:
      - name: run
        hostPath:
          path: /run/flannel
      - name: cni
        hostPath:
          path: /etc/cni/net.d
      - name: flannel-cfg
        configMap:
          name: kube-flannel-cfg
```

### Popular CNI Solutions Comparison

**Calico**: 
- **Strengths**: Advanced network policies, BGP routing, high performance
- **Use Cases**: Enterprise environments requiring fine-grained security controls
- **Routing**: BGP-based routing with optional VXLAN overlay

**Flannel**:
- **Strengths**: Simple setup, reliable, minimal resource overhead
- **Use Cases**: Development environments, simple production setups
- **Routing**: VXLAN overlay network by default

**Cilium**:
- **Strengths**: eBPF-based, advanced observability, API-aware security
- **Use Cases**: Cloud-native applications requiring L7 policies
- **Routing**: eBPF datapath with optional BGP

## CNI Configuration Deep Dive

### Calico CNI Setup - Production Configuration

Calico is a popular CNI solution that provides both networking and network security for Kubernetes. It uses BGP for routing and supports advanced network policies.

```yaml
# Calico CNI configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: calico-config
  namespace: kube-system
data:
  calico_backend: "bird"
  cluster_type: "k8s,bgp"
  cni_network_config: |-
    {
      "name": "k8s-pod-network",
      "cniVersion": "0.3.1",
      "plugins": [
        {
          "type": "calico",
          "log_level": "info",
          "datastore_type": "kubernetes",
          "nodename": "__KUBERNETES_NODE_NAME__",
          "mtu": __CNI_MTU__,
          "ipam": {
              "type": "calico-ipam"
          },
          "policy": {
              "type": "k8s"
          },
          "kubernetes": {
              "kubeconfig": "__KUBECONFIG_FILEPATH__"
          }
        },
        {
          "type": "portmap",
          "snat": true,
          "capabilities": {"portMappings": true}
        },
        {
          "type": "bandwidth",
          "capabilities": {"bandwidth": true}
        }
      ]
    }
  typha_service_name: "calico-typha"
```

**Key Configuration Parameters:**
- **calico_backend: "bird"**: Uses BIRD BGP daemon for routing
- **datastore_type: "kubernetes"**: Stores Calico configuration in Kubernetes etcd
- **ipam.type: "calico-ipam"**: Uses Calico's IP address management
- **policy.type: "k8s"**: Enables Kubernetes network policy support
- **portmap plugin**: Enables port mapping for services
- **bandwidth plugin**: Enables bandwidth limiting capabilities

### Calico Node Configuration
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: calico-node
  namespace: kube-system
spec:
  selector:
    matchLabels:
      k8s-app: calico-node
  template:
    metadata:
      labels:
        k8s-app: calico-node
    spec:
      nodeSelector:
        kubernetes.io/os: linux
      hostNetwork: true
      tolerations:
      - effect: NoSchedule
        operator: Exists
      - key: CriticalAddonsOnly
        operator: Exists
      - effect: NoExecute
        operator: Exists
      serviceAccountName: calico-node
      terminationGracePeriodSeconds: 0
      priorityClassName: system-node-critical
      containers:
      - name: calico-node
        image: calico/node:v3.26.0
        envFrom:
        - configMapRef:
            name: kubernetes-services-endpoint
            optional: true
        env:
        - name: DATASTORE_TYPE
          value: "kubernetes"
        - name: CALICO_NETWORKING_BACKEND
          valueFrom:
            configMapKeyRef:
              name: calico-config
              key: calico_backend
        - name: CLUSTER_TYPE
          valueFrom:
            configMapKeyRef:
              name: calico-config
              key: cluster_type
        - name: IP
          value: "autodetect"
        - name: CALICO_IPV4POOL_IPIP
          value: "Always"
        - name: FELIX_IPINIPMTU
          valueFrom:
            configMapKeyRef:
              name: calico-config
              key: veth_mtu
        - name: CALICO_DISABLE_FILE_LOGGING
          value: "true"
        - name: FELIX_DEFAULTENDPOINTTOHOSTACTION
          value: "ACCEPT"
        - name: FELIX_IPV6SUPPORT
          value: "false"
        - name: FELIX_LOGSEVERITYSCREEN
          value: "info"
        - name: FELIX_HEALTHENABLED
          value: "true"
        securityContext:
          privileged: true
        resources:
          requests:
            cpu: 250m
```

### Flannel CNI Configuration
```yaml
# Flannel CNI DaemonSet
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: kube-flannel-ds
  namespace: kube-system
  labels:
    tier: node
    app: flannel
spec:
  selector:
    matchLabels:
      app: flannel
  template:
    metadata:
      labels:
        tier: node
        app: flannel
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/os
                operator: In
                values:
                - linux
      hostNetwork: true
      priorityClassName: system-node-critical
      tolerations:
      - operator: Exists
        effect: NoSchedule
      serviceAccountName: flannel
      initContainers:
      - name: install-cni-plugin
        image: rancher/mirrored-flannelcni-flannel-cni-plugin:v1.1.0
        command:
        - cp
        args:
        - -f
        - /flannel
        - /opt/cni/bin/flannel
        volumeMounts:
        - name: cni-plugin
          mountPath: /opt/cni/bin
      - name: install-cni
        image: rancher/mirrored-flannelcni-flannel:v0.19.2
        command:
        - cp
        args:
        - -f
        - /etc/kube-flannel/cni-conf.json
        - /etc/cni/net.d/10-flannel.conflist
        volumeMounts:
        - name: cni
          mountPath: /etc/cni/net.d
        - name: flannel-cfg
          mountPath: /etc/kube-flannel/
      containers:
      - name: kube-flannel
        image: rancher/mirrored-flannelcni-flannel:v0.19.2
        command:
        - /opt/bin/flanneld
        args:
        - --ip-masq
        - --kube-subnet-mgr
        resources:
          requests:
            cpu: "100m"
            memory: "50Mi"
          limits:
            cpu: "100m"
            memory: "50Mi"
        securityContext:
          privileged: false
          capabilities:
            add: ["NET_ADMIN", "NET_RAW"]
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        volumeMounts:
        - name: run
          mountPath: /run/flannel
        - name: flannel-cfg
          mountPath: /etc/kube-flannel/
        - name: xtables-lock
          mountPath: /run/xtables.lock
      volumes:
      - name: run
        hostPath:
          path: /run/flannel
      - name: cni-plugin
        hostPath:
          path: /opt/cni/bin
      - name: cni
        hostPath:
          path: /etc/cni/net.d
      - name: flannel-cfg
        configMap:
          name: kube-flannel-cfg
      - name: xtables-lock
        hostPath:
          path: /run/xtables.lock
          type: FileOrCreate
```

## Network Policies

### Default Deny Network Policy
```yaml
# Default deny all ingress traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
---
# Default deny all egress traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-egress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Egress
```

### Application-Specific Network Policy
```yaml
# Allow web app to communicate with database
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: web-app-netpol
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: web-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    - namespaceSelector:
        matchLabels:
          name: ingress-system
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  - to: []  # Allow DNS resolution
    ports:
    - protocol: UDP
      port: 53
```

### Namespace Isolation Policy
```yaml
# Isolate namespace traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: namespace-isolation
  namespace: secure-apps
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: secure-apps
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: secure-apps
  - to: []  # Allow DNS
    ports:
    - protocol: UDP
      port: 53
  - to: []  # Allow external HTTPS
    ports:
    - protocol: TCP
      port: 443
```

## Service Types and Load Balancing

### LoadBalancer Service with Annotations
```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-app-lb
  annotations:
    # AWS Load Balancer Controller annotations
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "tcp"
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:us-east-1:123456789:certificate/abc123"
    service.beta.kubernetes.io/aws-load-balancer-ssl-ports: "https"
    # Azure annotations
    service.beta.kubernetes.io/azure-load-balancer-internal: "true"
    service.beta.kubernetes.io/azure-load-balancer-internal-subnet: "subnet-1"
    # GCP annotations
    cloud.google.com/load-balancer-type: "Internal"
spec:
  type: LoadBalancer
  selector:
    app: web-app
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: https
    port: 443
    targetPort: 8080
    protocol: TCP
```

### ExternalName Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-database
  namespace: production
spec:
  type: ExternalName
  externalName: database.example.com
  ports:
  - port: 5432
    targetPort: 5432
    protocol: TCP
```

### Headless Service for StatefulSet
```yaml
apiVersion: v1
kind: Service
metadata:
  name: mysql-headless
  labels:
    app: mysql
spec:
  ports:
  - port: 3306
    name: mysql
  clusterIP: None  # Headless service
  selector:
    app: mysql
```

## Ingress Controllers

### NGINX Ingress Controller
```yaml
# NGINX Ingress Configuration
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-app-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - example.com
    - www.example.com
    secretName: example-tls
  rules:
  - host: example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
  - host: www.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```

### Traefik Ingress Configuration
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: traefik-ingress
  annotations:
    traefik.ingress.kubernetes.io/router.entrypoints: websecure
    traefik.ingress.kubernetes.io/router.tls: "true"
    traefik.ingress.kubernetes.io/router.middlewares: default-auth@kubernetescrd
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-service
            port:
              number: 80
  tls:
  - secretName: app-tls
    hosts:
    - app.example.com
```

## DNS Configuration

### CoreDNS Custom Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns-custom
  namespace: kube-system
data:
  example.server: |
    example.com:53 {
        errors
        cache 30
        forward . 8.8.8.8 8.8.4.4
    }
  custom.override: |
    hosts {
        192.168.1.100 custom.example.com
        fallthrough
    }
```

### Network Troubleshooting Tools
```yaml
# Network debugging pod
apiVersion: v1
kind: Pod
metadata:
  name: netshoot
spec:
  containers:
  - name: netshoot
    image: nicolaka/netshoot
    command: ["/bin/bash"]
    args: ["-c", "while true; do ping localhost; sleep 60;done"]
    securityContext:
      capabilities:
        add: ["NET_ADMIN"]
  restartPolicy: Always
```

### Network Testing Script
```bash
#!/bin/bash
# Kubernetes network connectivity test

echo "=== Kubernetes Network Connectivity Test ==="

# Test DNS resolution
echo "1. Testing DNS resolution..."
nslookup kubernetes.default.svc.cluster.local
nslookup google.com

# Test service connectivity
echo -e "\n2. Testing service connectivity..."
kubectl run test-pod --image=busybox --rm -it --restart=Never -- /bin/sh -c "
  echo 'Testing cluster DNS...'
  nslookup kubernetes.default.svc.cluster.local
  
  echo 'Testing external connectivity...'
  wget -qO- http://httpbin.org/ip
  
  echo 'Testing internal service connectivity...'
  # Replace with your service name
  wget -qO- http://web-app-service/health
"

# Test network policies
echo -e "\n3. Testing network policies..."
kubectl run policy-test --image=busybox --rm -it --restart=Never -- /bin/sh -c "
  # Test blocked connection
  timeout 5 wget -qO- http://blocked-service || echo 'Connection blocked as expected'
  
  # Test allowed connection
  wget -qO- http://allowed-service || echo 'Connection failed - check policy'
"

echo -e "\n=== Network Test Complete ==="
```

## Next Section

Continue to [Storage](04_Storage.md) to learn about Kubernetes persistent storage, volumes, and storage classes.
