# Kubernetes Cluster Architecture

Kubernetes cluster architecture consists of two main components: the control plane (master) and worker nodes. Understanding this architecture is crucial for deploying, managing, and troubleshooting Kubernetes clusters in production environments.

## Cluster Architecture Overview

A Kubernetes cluster is a set of machines (physical or virtual) that work together to run containerized applications. The cluster architecture follows a master-worker pattern where:

- **Control Plane (Master)**: Manages the cluster state and makes decisions about scheduling
- **Worker Nodes**: Run the actual application workloads
- **Communication**: Secure API-based communication between all components

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Control Plane                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ API Server  │  │ Scheduler   │  │ Controller  │              │
│  │             │  │             │  │  Manager    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                           │                                     │
│                    ┌─────────────┐                              │
│                    │    etcd     │                              │
│                    │  (Storage)  │                              │
│                    └─────────────┘                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │ (Secure Communication)
┌─────────────────────────▼───────────────────────────────────────┐
│                    Worker Nodes                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────────────┐│
│  │       Node 1            │  │        Node 2                   ││
│  │  ┌─────────────────────┐│  │  ┌─────────────────────────────┐││
│  │  │      kubelet        ││  │  │       kubelet               │││
│  │  └─────────────────────┘│  │  └─────────────────────────────┘││
│  │  ┌─────────────────────┐│  │  ┌─────────────────────────────┐││
│  │  │    kube-proxy       ││  │  │     kube-proxy              │││
│  │  └─────────────────────┘│  │  └─────────────────────────────┘││
│  │  ┌─────────────────────┐│  │  ┌─────────────────────────────┐││
│  │  │Container Runtime    ││  │  │  Container Runtime          │││
│  │  │   (Docker/CRI-O)    ││  │  │    (Docker/CRI-O)           │││
│  │  └─────────────────────┘│  │  └─────────────────────────────┘││
│  └─────────────────────────┘  └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Control Plane Components

The control plane manages the cluster state and makes global decisions about the cluster. In production environments, control plane components are typically distributed across multiple machines for high availability.

### kube-apiserver

The API server is the central management component that exposes the Kubernetes API. It serves as the frontend for the Kubernetes control plane and is the only component that directly communicates with etcd.

**Key Responsibilities:**
- Validates and processes API requests
- Authenticates and authorizes users
- Implements admission control
- Stores resource definitions in etcd
- Serves the Kubernetes API

**Critical Configuration Parameters:**

```yaml
# kube-apiserver configuration example
apiVersion: v1
kind: Pod
metadata:
  name: kube-apiserver
  namespace: kube-system
spec:
  containers:
  - name: kube-apiserver
    image: k8s.gcr.io/kube-apiserver:v1.28.0
    command:
    - kube-apiserver
    - --advertise-address=192.168.1.100          # IP address for API server
    - --allow-privileged=true                    # Allow privileged containers
    - --authorization-mode=Node,RBAC             # Authorization methods
    - --client-ca-file=/etc/kubernetes/pki/ca.crt
    - --enable-admission-plugins=NodeRestriction # Security plugins
    - --enable-bootstrap-token-auth=true
    - --etcd-cafile=/etc/kubernetes/pki/etcd/ca.crt
    - --etcd-certfile=/etc/kubernetes/pki/apiserver-etcd-client.crt
    - --etcd-keyfile=/etc/kubernetes/pki/apiserver-etcd-client.key
    - --etcd-servers=https://127.0.0.1:2379      # etcd endpoints
    - --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt
    - --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
    - --runtime-config=api/all=true              # Enable all API groups
    - --service-account-issuer=https://kubernetes.default.svc.cluster.local
    - --service-account-key-file=/etc/kubernetes/pki/sa.pub
    - --service-account-signing-key-file=/etc/kubernetes/pki/sa.key
    - --service-cluster-ip-range=10.96.0.0/12    # Service IP range
    - --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
    - --tls-private-key-file=/etc/kubernetes/pki/apiserver.key
    ports:
    - containerPort: 6443
      hostPort: 6443
      name: https
    - containerPort: 8080
      hostPort: 8080
      name: local
    volumeMounts:
    - name: ca-certs
      mountPath: /etc/ssl/certs
      readOnly: true
    - name: etc-ca-certificates
      mountPath: /etc/ca-certificates
      readOnly: true
    - name: k8s-certs
      mountPath: /etc/kubernetes/pki
      readOnly: true
    - name: usr-local-share-ca-certificates
      mountPath: /usr/local/share/ca-certificates
      readOnly: true
    - name: usr-share-ca-certificates
      mountPath: /usr/share/ca-certificates
      readOnly: true
  volumes:
  - name: ca-certs
    hostPath:
      path: /etc/ssl/certs
      type: DirectoryOrCreate
  - name: etc-ca-certificates
    hostPath:
      path: /etc/ca-certificates
      type: DirectoryOrCreate
  - name: k8s-certs
    hostPath:
      path: /etc/kubernetes/pki
      type: DirectoryOrCreate
  - name: usr-local-share-ca-certificates
    hostPath:
      path: /usr/local/share/ca-certificates
      type: DirectoryOrCreate
  - name: usr-share-ca-certificates
    hostPath:
      path: /usr/share/ca-certificates
      type: DirectoryOrCreate
```

### etcd - The Cluster Database

etcd is a consistent, distributed key-value store that serves as Kubernetes' backing store for all cluster data. It stores the configuration data, state, and metadata for the entire cluster.

**Key Characteristics:**
- Strong consistency using Raft consensus algorithm
- High availability through clustering
- Watch capabilities for real-time updates
- Secure communication with TLS encryption
**Production etcd Configuration:**

```yaml
# etcd cluster configuration
apiVersion: v1
kind: Pod
metadata:
  name: etcd
  namespace: kube-system
spec:
  containers:
  - name: etcd
    image: k8s.gcr.io/etcd:3.5.9-0
    command:
    - etcd
    - --advertise-client-urls=https://192.168.1.100:2379
    - --cert-file=/etc/kubernetes/pki/etcd/server.crt
    - --client-cert-auth=true
    - --data-dir=/var/lib/etcd
    - --initial-advertise-peer-urls=https://192.168.1.100:2380
    - --initial-cluster=master1=https://192.168.1.100:2380,master2=https://192.168.1.101:2380,master3=https://192.168.1.102:2380
    - --initial-cluster-state=new
    - --initial-cluster-token=etcd-cluster-token
    - --key-file=/etc/kubernetes/pki/etcd/server.key
    - --listen-client-urls=https://127.0.0.1:2379,https://192.168.1.100:2379
    - --listen-metrics-urls=http://127.0.0.1:2381
    - --listen-peer-urls=https://192.168.1.100:2380
    - --name=master1
    - --peer-cert-file=/etc/kubernetes/pki/etcd/peer.crt
    - --peer-client-cert-auth=true
    - --peer-key-file=/etc/kubernetes/pki/etcd/peer.key
    - --peer-trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
    - --snapshot-count=10000
    - --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
    ports:
    - containerPort: 2379
      name: client
    - containerPort: 2380
      name: peer
    - containerPort: 2381
      name: metrics
    volumeMounts:
    - name: etcd-certs
      mountPath: /etc/kubernetes/pki/etcd
      readOnly: true
    - name: etcd-data
      mountPath: /var/lib/etcd
    livenessProbe:
      httpGet:
        path: /health
        port: 2381
        scheme: HTTP
      initialDelaySeconds: 15
      timeoutSeconds: 15
    resources:
      requests:
        cpu: 100m
        memory: 100Mi
      limits:
        cpu: 200m
        memory: 300Mi
  volumes:
  - name: etcd-certs
    hostPath:
      path: /etc/kubernetes/pki/etcd
      type: DirectoryOrCreate
  - name: etcd-data
    hostPath:
      path: /var/lib/etcd
      type: DirectoryOrCreate
```

**etcd Best Practices:**
- Always run etcd in a cluster with odd number of members (3, 5, or 7)
- Use dedicated storage with low latency (SSD recommended)
- Regular automated backups with retention policies
- Monitor etcd performance and resource usage
- Secure etcd with TLS certificates
- Keep etcd version compatible with Kubernetes version

### kube-scheduler

The scheduler is responsible for selecting which node should run each newly created pod. It makes scheduling decisions based on resource requirements, hardware/software constraints, affinity specifications, and workload requirements.

**Scheduling Process:**
1. **Filtering**: Eliminate nodes that don't meet pod requirements
2. **Scoring**: Rank remaining nodes based on priorities
3. **Selection**: Choose the highest-scoring node
4. **Binding**: Assign the pod to the selected node

**kube-scheduler Configuration:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kube-scheduler
  namespace: kube-system
spec:
  containers:
  - name: kube-scheduler
    image: k8s.gcr.io/kube-scheduler:v1.28.0
    command:
    - kube-scheduler
    - --authentication-kubeconfig=/etc/kubernetes/scheduler.conf
    - --authorization-kubeconfig=/etc/kubernetes/scheduler.conf
    - --bind-address=127.0.0.1
    - --kubeconfig=/etc/kubernetes/scheduler.conf
    - --leader-elect=true
    - --secure-port=10259
    - --v=2
    ports:
    - containerPort: 10259
      name: https
    resources:
      requests:
        cpu: 100m
        memory: 100Mi
    volumeMounts:
    - name: kubeconfig
      mountPath: /etc/kubernetes/scheduler.conf
      readOnly: true
    - name: ca-certs
      mountPath: /etc/ssl/certs
      readOnly: true
  volumes:
  - name: kubeconfig
    hostPath:
      path: /etc/kubernetes/scheduler.conf
      type: FileOrCreate
  - name: ca-certs
    hostPath:
      path: /etc/ssl/certs
      type: DirectoryOrCreate
```

**Scheduler Policies and Profiles:**

The scheduler can be customized with policies and profiles to implement specific scheduling behaviors:

```yaml
apiVersion: kubescheduler.config.k8s.io/v1beta3
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: default-scheduler
  plugins:
    score:
      enabled:
      - name: NodeResourcesFit
      - name: NodeAffinity
      - name: PodTopologySpread
      disabled:
      - name: NodeResourcesLeastAllocated
  pluginConfig:
  - name: NodeResourcesFit
    args:
      scoringStrategy:
        type: LeastAllocated
        resources:
        - name: cpu
          weight: 1
        - name: memory
          weight: 1
```

### kube-controller-manager

The controller manager runs controllers that regulate the state of the cluster. Each controller watches the shared state through the API server and makes changes attempting to move the current state towards the desired state.

**Key Controllers:**
- **Node Controller**: Manages node lifecycle and health
- **Replication Controller**: Ensures correct number of replicas
- **Endpoints Controller**: Manages service endpoints
- **Service Account Controller**: Creates default service accounts
- **Deployment Controller**: Manages deployments and replica sets
- **StatefulSet Controller**: Manages stateful applications
- **DaemonSet Controller**: Ensures pods run on selected nodes

**kube-controller-manager Configuration:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kube-controller-manager
  namespace: kube-system
spec:
  containers:
  - name: kube-controller-manager
    image: k8s.gcr.io/kube-controller-manager:v1.28.0
    command:
    - kube-controller-manager
    - --authentication-kubeconfig=/etc/kubernetes/controller-manager.conf
    - --authorization-kubeconfig=/etc/kubernetes/controller-manager.conf
    - --bind-address=127.0.0.1
    - --client-ca-file=/etc/kubernetes/pki/ca.crt
    - --cluster-name=kubernetes
    - --cluster-signing-cert-file=/etc/kubernetes/pki/ca.crt
    - --cluster-signing-key-file=/etc/kubernetes/pki/ca.key
    - --controllers=*,bootstrapsigner,tokencleaner
    - --kubeconfig=/etc/kubernetes/controller-manager.conf
    - --leader-elect=true
    - --requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt
    - --root-ca-file=/etc/kubernetes/pki/ca.crt
    - --service-account-private-key-file=/etc/kubernetes/pki/sa.key
    - --service-cluster-ip-range=10.96.0.0/12
    - --use-service-account-credentials=true
    - --v=2
    ports:
    - containerPort: 10257
      name: https
    resources:
      requests:
        cpu: 200m
        memory: 200Mi
    volumeMounts:
    - name: ca-certs
      mountPath: /etc/ssl/certs
      readOnly: true
    - name: etc-ca-certificates
      mountPath: /etc/ca-certificates
      readOnly: true
    - name: flexvolume-dir
      mountPath: /usr/libexec/kubernetes/kubelet-plugins/volume/exec
    - name: k8s-certs
      mountPath: /etc/kubernetes/pki
      readOnly: true
    - name: kubeconfig
      mountPath: /etc/kubernetes/controller-manager.conf
      readOnly: true
    - name: usr-local-share-ca-certificates
      mountPath: /usr/local/share/ca-certificates
      readOnly: true
    - name: usr-share-ca-certificates
      mountPath: /usr/share/ca-certificates
      readOnly: true
  volumes:
  - name: ca-certs
    hostPath:
      path: /etc/ssl/certs
      type: DirectoryOrCreate
  - name: etc-ca-certificates
    hostPath:
      path: /etc/ca-certificates
      type: DirectoryOrCreate
  - name: flexvolume-dir
    hostPath:
      path: /usr/libexec/kubernetes/kubelet-plugins/volume/exec
      type: DirectoryOrCreate
  - name: k8s-certs
    hostPath:
      path: /etc/kubernetes/pki
      type: DirectoryOrCreate
  - name: kubeconfig
    hostPath:
      path: /etc/kubernetes/controller-manager.conf
      type: FileOrCreate
  - name: usr-local-share-ca-certificates
    hostPath:
      path: /usr/local/share/ca-certificates
      type: DirectoryOrCreate
  - name: usr-share-ca-certificates
    hostPath:
      path: /usr/share/ca-certificates
      type: DirectoryOrCreate
```

### cloud-controller-manager

The cloud controller manager integrates with cloud provider APIs to manage cloud-specific resources like load balancers, storage volumes, and nodes. It decouples cloud-specific logic from the core Kubernetes components.

**Cloud-Specific Controllers:**
- **Node Controller**: Updates node objects with cloud provider metadata
- **Route Controller**: Configures routes in the cloud network
- **Service Controller**: Creates/updates/deletes cloud load balancers
- **Volume Controller**: Creates/attaches/mounts cloud volumes

**Example cloud-controller-manager for AWS:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: cloud-controller-manager
  namespace: kube-system
spec:
  containers:
  - name: cloud-controller-manager
    image: k8s.gcr.io/cloud-controller-manager:v1.28.0
    command:
    - /usr/local/bin/cloud-controller-manager
    - --allocate-node-cidrs=true
    - --authentication-kubeconfig=/etc/kubernetes/controller-manager.conf
    - --authorization-kubeconfig=/etc/kubernetes/controller-manager.conf
    - --bind-address=127.0.0.1
    - --cloud-provider=aws
    - --cluster-cidr=10.244.0.0/16
    - --cluster-name=kubernetes
    - --configure-cloud-routes=false
    - --controllers=*,-nodeipam,-route
    - --kubeconfig=/etc/kubernetes/controller-manager.conf
    - --leader-elect=true
    - --secure-port=10258
    - --use-service-account-credentials=true
    - --v=2
    env:
    - name: AWS_REGION
      value: us-west-2
    ports:
    - containerPort: 10258
      name: https
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
    volumeMounts:
    - name: kubeconfig
      mountPath: /etc/kubernetes/controller-manager.conf
      readOnly: true
    - name: ca-certs
      mountPath: /etc/ssl/certs
      readOnly: true
  volumes:
  - name: kubeconfig
    hostPath:
      path: /etc/kubernetes/controller-manager.conf
      type: FileOrCreate
  - name: ca-certs
    hostPath:
      path: /etc/ssl/certs
      type: DirectoryOrCreate
```

## Node Components

Worker nodes run the containerized applications and provide the runtime environment. Each node contains the necessary components to run pods and is managed by the control plane.

### kubelet - The Node Agent

The kubelet is the primary node agent that communicates with the control plane. It's responsible for pod lifecycle management on its node.

**Key Responsibilities:**
- Registers the node with the API server
- Monitors pod specifications from the API server
- Manages pod and container lifecycle
- Reports node and pod status back to the control plane
- Implements container health checks
- Manages storage volumes and secrets

**kubelet Configuration:**

```yaml
# kubelet configuration file (/var/lib/kubelet/config.yaml)
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
address: 0.0.0.0
port: 10250
authentication:
  anonymous:
    enabled: false
  webhook:
    enabled: true
  x509:
    clientCAFile: /etc/kubernetes/pki/ca.crt
authorization:
  mode: Webhook
cgroupDriver: systemd
clusterDNS:
- 10.96.0.10
clusterDomain: cluster.local
containerRuntimeEndpoint: unix:///var/run/containerd/containerd.sock
cpuManagerPolicy: none
evictionHard:
  imagefs.available: 15%
  memory.available: 100Mi
  nodefs.available: 10%
  nodefs.inodesFree: 5%
evictionPressureTransitionPeriod: 5m0s
failSwapOn: false
fileCheckFrequency: 20s
healthzBindAddress: 127.0.0.1
healthzPort: 10248
httpCheckFrequency: 20s
imageGCHighThresholdPercent: 85
imageGCLowThresholdPercent: 80
kubeAPIBurst: 10
kubeAPIQPS: 5
makeIPTablesUtilChains: true
maxOpenFiles: 1000000
maxPods: 110
nodeStatusUpdateFrequency: 10s
readOnlyPort: 0
registryBurst: 10
registryPullQPS: 5
resolvConf: /run/systemd/resolve/resolv.conf
rotateCertificates: true
runtimeRequestTimeout: 2m0s
serializeImagePulls: false
serverTLSBootstrap: true
staticPodPath: /etc/kubernetes/manifests
streamingConnectionIdleTimeout: 4h0m0s
syncFrequency: 1m0s
volumeStatsAggPeriod: 1m0s
```

**kubelet System Service:**

```systemd
# /etc/systemd/system/kubelet.service
[Unit]
Description=kubelet: The Kubernetes Node Agent
Documentation=https://kubernetes.io/docs/home/
Wants=network-online.target
After=network-online.target

[Service]
ExecStart=/usr/bin/kubelet \
  --config=/var/lib/kubelet/config.yaml \
  --kubeconfig=/etc/kubernetes/kubelet.conf \
  --bootstrap-kubeconfig=/etc/kubernetes/bootstrap-kubelet.conf \
  --container-runtime=remote \
  --container-runtime-endpoint=unix:///var/run/containerd/containerd.sock \
  --fail-swap-on=false \
  --node-ip=192.168.1.100 \
  --v=2
Restart=always
StartLimitInterval=0
RestartSec=10

[Install]
WantedBy=multi-user.target
```
kind: Pod
metadata:
  name: etcd
  namespace: kube-system
spec:
  containers:
  - name: etcd
    image: k8s.gcr.io/etcd:3.5.9-0
    command:
    - etcd
    - --advertise-client-urls=https://192.168.1.100:2379
    - --client-cert-auth=true
    - --data-dir=/var/lib/etcd
    - --initial-advertise-peer-urls=https://192.168.1.100:2380
    - --initial-cluster=master=https://192.168.1.100:2380
    - --key-file=/etc/kubernetes/pki/etcd/server.key
    - --listen-client-urls=https://127.0.0.1:2379,https://192.168.1.100:2379
    - --listen-metrics-urls=http://127.0.0.1:2381
    - --listen-peer-urls=https://192.168.1.100:2380
    - --name=master
    - --peer-cert-file=/etc/kubernetes/pki/etcd/peer.crt
    - --peer-client-cert-auth=true
    - --peer-key-file=/etc/kubernetes/pki/etcd/peer.key
    - --peer-trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
    - --snapshot-count=10000
    - --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
    ports:
    - containerPort: 2379
      hostPort: 2379
      name: client
    - containerPort: 2380
      hostPort: 2380
      name: peer
```

### Scheduler Configuration
```yaml
# kube-scheduler configuration
apiVersion: kubescheduler.config.k8s.io/v1beta3
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: default-scheduler
  plugins:
    score:
      enabled:
      - name: NodeResourcesFit
      - name: NodeAffinity
      - name: PodTopologySpread
      - name: TaintToleration
    filter:
      enabled:
      - name: NodeResourcesFit
      - name: NodeAffinity
      - name: PodTopologySpread
      - name: TaintToleration
  pluginConfig:
  - name: NodeResourcesFit
    args:
      scoringStrategy:
        type: LeastAllocated
```

## Node Components

### Kubelet Configuration
```yaml
# kubelet configuration
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
address: 0.0.0.0
port: 10250
readOnlyPort: 10255
cgroupDriver: systemd
clusterDNS:
- 10.96.0.10
clusterDomain: cluster.local
failSwapOn: false
authentication:
  anonymous:
    enabled: false
  webhook:
    enabled: true
authorization:
  mode: Webhook
serverTLSBootstrap: true
```

### Kube-proxy Configuration
```yaml
# kube-proxy configuration
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
bindAddress: 0.0.0.0
healthzBindAddress: 0.0.0.0:10256
metricsBindAddress: 127.0.0.1:10249
enableProfiling: false
clusterCIDR: 10.244.0.0/16
hostnameOverride: ""
clientConnection:
  kubeconfig: /var/lib/kube-proxy/kubeconfig.conf
mode: "iptables"
portRange: ""
iptables:
  masqueradeAll: false
  masqueradeBit: 14
  minSyncPeriod: 0s
  syncPeriod: 30s
ipvs:
  strictARP: false
  tcpTimeout: 0s
  tcpFinTimeout: 0s
  udpTimeout: 0s
nodePortAddresses: []
```

## High Availability Setup

### Multi-Master Configuration
```yaml
# HAProxy configuration for HA masters
global
    log stdout local0
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

defaults
    mode http
    log global
    option httplog
    option dontlognull
    option log-health-checks
    option forwardfor except 127.0.0.0/8
    option redispatch
    retries 3
    timeout http-request 10s
    timeout queue 20s
    timeout connect 5s
    timeout client 20s
    timeout server 20s
    timeout http-keep-alive 10s
    timeout check 10s

frontend kubernetes-apiserver
    bind *:6443
    mode tcp
    option tcplog
    timeout client 300s
    default_backend kubernetes-apiserver

backend kubernetes-apiserver
    mode tcp
    option tcplog
    option tcp-check
    timeout server 300s
    balance roundrobin
    default-server inter 10s downinter 5s rise 2 fall 2 slowstart 60s maxconn 250 maxqueue 256 weight 100
    server k8s-master-1 192.168.1.101:6443 check
    server k8s-master-2 192.168.1.102:6443 check
    server k8s-master-3 192.168.1.103:6443 check
```

### etcd Cluster Setup
```bash
#!/bin/bash
# etcd cluster initialization script

# Node 1
export ETCD_NAME=etcd-1
export ETCD_IP=192.168.1.101
export ETCD_CLUSTER="etcd-1=https://192.168.1.101:2380,etcd-2=https://192.168.1.102:2380,etcd-3=https://192.168.1.103:2380"

etcd --name ${ETCD_NAME} \
  --data-dir /var/lib/etcd \
  --listen-client-urls https://0.0.0.0:2379 \
  --advertise-client-urls https://${ETCD_IP}:2379 \
  --listen-peer-urls https://0.0.0.0:2380 \
  --initial-advertise-peer-urls https://${ETCD_IP}:2380 \
  --initial-cluster ${ETCD_CLUSTER} \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster-state new \
  --cert-file=/etc/etcd/server.crt \
  --key-file=/etc/etcd/server.key \
  --trusted-ca-file=/etc/etcd/ca.crt \
  --peer-cert-file=/etc/etcd/peer.crt \
  --peer-key-file=/etc/etcd/peer.key \
  --peer-trusted-ca-file=/etc/etcd/ca.crt \
  --peer-client-cert-auth \
  --client-cert-auth
```

## Cluster Health Check Script
```bash
#!/bin/bash
# Kubernetes cluster health check script

echo "=== Kubernetes Cluster Health Check ==="

# Check API server health
echo "1. Checking API Server..."
kubectl cluster-info
if [ $? -eq 0 ]; then
    echo "✓ API Server is healthy"
else
    echo "✗ API Server is not responding"
fi

# Check node status
echo -e "\n2. Checking Node Status..."
kubectl get nodes
READY_NODES=$(kubectl get nodes --no-headers | grep " Ready " | wc -l)
TOTAL_NODES=$(kubectl get nodes --no-headers | wc -l)
echo "Ready Nodes: $READY_NODES/$TOTAL_NODES"

# Check system pods
echo -e "\n3. Checking System Pods..."
kubectl get pods -n kube-system
RUNNING_PODS=$(kubectl get pods -n kube-system --no-headers | grep "Running" | wc -l)
TOTAL_PODS=$(kubectl get pods -n kube-system --no-headers | wc -l)
echo "Running System Pods: $RUNNING_PODS/$TOTAL_PODS"

# Check etcd health
echo -e "\n4. Checking etcd Health..."
kubectl exec -n kube-system etcd-master -- etcdctl \
  --endpoints https://127.0.0.1:2379 \
  --cert /etc/kubernetes/pki/etcd/server.crt \
  --key /etc/kubernetes/pki/etcd/server.key \
  --cacert /etc/kubernetes/pki/etcd/ca.crt \
  endpoint health

# Check component status
echo -e "\n5. Checking Component Status..."
kubectl get componentstatuses

echo -e "\n=== Health Check Complete ==="
```

### kube-proxy - Network Proxy

The kube-proxy runs on each node and maintains network rules that allow communication to pods from inside or outside the cluster. It implements the Kubernetes service abstraction by maintaining network rules and connection forwarding.

**Key Functions:**
- Implements service load balancing
- Maintains network rules for service connectivity
- Handles connection forwarding and load distribution
- Supports multiple proxy modes (iptables, IPVS, userspace)

**kube-proxy Configuration:**

```yaml
# kube-proxy configuration
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
bindAddress: 0.0.0.0
clientConnection:
  acceptContentTypes: ""
  burst: 10
  contentType: application/vnd.kubernetes.protobuf
  kubeconfig: /var/lib/kube-proxy/kubeconfig.conf
  qps: 5
clusterCIDR: 10.244.0.0/16
configSyncPeriod: 15m0s
conntrack:
  max: null
  maxPerCore: 32768
  min: 131072
  tcpCloseWaitTimeout: 1h0m0s
  tcpEstablishedTimeout: 24h0m0s
enableProfiling: false
featureGates: {}
healthzBindAddress: 0.0.0.0:10256
hostnameOverride: ""
iptables:
  masqueradeAll: false
  masqueradeBit: 14
  minSyncPeriod: 0s
  syncPeriod: 30s
ipvs:
  excludeCIDRs: null
  minSyncPeriod: 0s
  scheduler: ""
  strictARP: false
  syncPeriod: 30s
  tcpFinTimeout: 0s
  tcpTimeout: 0s
  udpTimeout: 0s
metricsBindAddress: 127.0.0.1:10249
mode: "iptables"
nodePortAddresses: null
oomScoreAdj: -999
portRange: ""
udpIdleTimeout: 250ms
winkernel:
  enableDSR: false
  networkName: ""
  sourceVip: ""
```

**kube-proxy as DaemonSet:**

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: kube-proxy
  namespace: kube-system
  labels:
    k8s-app: kube-proxy
spec:
  selector:
    matchLabels:
      k8s-app: kube-proxy
  template:
    metadata:
      labels:
        k8s-app: kube-proxy
    spec:
      hostNetwork: true
      tolerations:
      - key: node-role.kubernetes.io/master
        operator: Exists
        effect: NoSchedule
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule
      serviceAccountName: kube-proxy
      containers:
      - name: kube-proxy
        image: k8s.gcr.io/kube-proxy:v1.28.0
        command:
        - /usr/local/bin/kube-proxy
        - --config=/var/lib/kube-proxy/config.conf
        - --hostname-override=$(NODE_NAME)
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        securityContext:
          privileged: true
        volumeMounts:
        - name: kube-proxy
          mountPath: /var/lib/kube-proxy
        - name: xtables-lock
          mountPath: /run/xtables.lock
        - name: lib-modules
          mountPath: /lib/modules
          readOnly: true
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
      volumes:
      - name: kube-proxy
        configMap:
          name: kube-proxy
      - name: xtables-lock
        hostPath:
          path: /run/xtables.lock
          type: FileOrCreate
      - name: lib-modules
        hostPath:
          path: /lib/modules
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
```

### Container Runtime

The container runtime is software responsible for running containers. Kubernetes supports multiple container runtimes through the Container Runtime Interface (CRI).

**Supported Runtimes:**
- **containerd**: Lightweight, high-performance container runtime
- **CRI-O**: Lightweight alternative specifically for Kubernetes
- **Docker Engine**: Traditional container runtime (deprecated in Kubernetes 1.20+)

**containerd Configuration:**

```toml
# /etc/containerd/config.toml
version = 2

[plugins."io.containerd.grpc.v1.cri"]
  sandbox_image = "k8s.gcr.io/pause:3.9"
  
  [plugins."io.containerd.grpc.v1.cri".containerd]
    snapshotter = "overlayfs"
    default_runtime_name = "runc"
    
    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]
      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
        runtime_type = "io.containerd.runc.v2"
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
          SystemdCgroup = true
          
  [plugins."io.containerd.grpc.v1.cri".cni]
    bin_dir = "/opt/cni/bin"
    conf_dir = "/etc/cni/net.d"
    max_conf_num = 1
    conf_template = ""
    
  [plugins."io.containerd.grpc.v1.cri".registry]
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors]
      [plugins."io.containerd.grpc.v1.cri".registry.mirrors."docker.io"]
        endpoint = ["https://registry-1.docker.io"]
```

**containerd System Service:**

```systemd
# /etc/systemd/system/containerd.service
[Unit]
Description=containerd container runtime
Documentation=https://containerd.io
After=network.target local-fs.target
Before=docker.service

[Service]
ExecStartPre=-/sbin/modprobe overlay
ExecStart=/usr/bin/containerd
Type=notify
Delegate=yes
KillMode=process
Restart=always
RestartSec=5
LimitNOFILE=1048576
LimitNPROC=1048576
LimitCORE=infinity
TasksMax=infinity
OOMScoreAdjust=-999

[Install]
WantedBy=multi-user.target
```

## Cluster Communication

Understanding how Kubernetes components communicate is crucial for troubleshooting and security configuration.

### API Server Communication Patterns

The API server serves as the central hub for all cluster communication:

**Client-to-API Server:**
- All cluster components communicate through the API server
- Uses HTTPS with TLS encryption
- Authentication and authorization at the API level
- RESTful API with standard HTTP methods

**API Server-to-etcd:**
- Direct connection to etcd cluster
- Uses mutual TLS authentication
- Only the API server communicates with etcd
- Connection pooling for performance

**API Server-to-Kubelet:**
- Secure HTTPS connection to kubelet
- Used for log retrieval, port forwarding, and exec
- Kubelet certificates for authentication
- Webhook authorization for API calls

### Node-to-Control Plane Communication

**Kubelet-to-API Server:**
- Kubelet authenticates using client certificates or bootstrap tokens
- Registers node information and reports status
- Receives pod specifications and reports pod status
- Implements watch connections for real-time updates

**kube-proxy-to-API Server:**
- Watches service and endpoint changes
- Updates local proxy rules based on service definitions
- Uses service account tokens for authentication

### Secure Communication Channels

**TLS Everywhere:**
- All component-to-component communication uses TLS
- Certificate-based authentication
- Mutual TLS for high-security environments

**Certificate Management:**
- PKI infrastructure for certificate generation
- Automatic certificate rotation
- Certificate authority (CA) management

```yaml
# Certificate configuration example
apiVersion: v1
kind: Secret
metadata:
  name: kubernetes-ca
  namespace: kube-system
type: Opaque
data:
  ca.crt: LS0tLS1CRUdJTi... # Base64 encoded CA certificate
  ca.key: LS0tLS1CRUdJTi... # Base64 encoded CA private key
```

## High Availability Architecture

Production Kubernetes clusters require high availability to ensure continuous operation and fault tolerance.

### Control Plane Redundancy

**Multi-Master Setup:**
- Multiple API server instances behind a load balancer
- Distributed etcd cluster across multiple nodes
- Active-passive scheduler and controller-manager instances
- Shared storage for critical components

**Load Balancer Configuration:**
```yaml
# HAProxy configuration for Kubernetes API
global
    maxconn 4096
    log stdout local0
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    
frontend kubernetes_frontend
    bind *:6443
    mode tcp
    default_backend kubernetes_backend
    
backend kubernetes_backend
    mode tcp
    balance roundrobin
    option tcp-check
    server master1 192.168.1.100:6443 check
    server master2 192.168.1.101:6443 check
    server master3 192.168.1.102:6443 check
```

### etcd Clustering

**etcd High Availability:**
- Minimum 3 nodes for production
- Odd number of nodes to avoid split-brain
- Distributed across failure domains
- Regular backups and disaster recovery procedures

**etcd Cluster Health Check:**
```bash
# Check etcd cluster health
etcdctl --endpoints=https://192.168.1.100:2379,https://192.168.1.101:2379,https://192.168.1.102:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  endpoint health

# Check etcd cluster members
etcdctl --endpoints=https://192.168.1.100:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  member list
```

### Multi-Zone Deployments

**Zone-Aware Scheduling:**
```yaml
# Node labels for zone awareness
apiVersion: v1
kind: Node
metadata:
  name: worker-node-1
  labels:
    topology.kubernetes.io/zone: us-west-2a
    topology.kubernetes.io/region: us-west-2
    node-role.kubernetes.io/worker: ""
spec:
  # ...existing code...
```

**Pod Anti-Affinity for HA:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - webapp
              topologyKey: topology.kubernetes.io/zone
      containers:
      - name: webapp
        image: nginx:1.21
        ports:
        - containerPort: 80
```

### Failure Domains

**Cluster Topology:**
- Control plane distributed across multiple zones/regions
- Worker nodes spread across failure domains
- Network redundancy between zones
- Storage replication across zones

**Disaster Recovery Planning:**
- Regular etcd backups
- Infrastructure as code for cluster recreation
- Data backup and recovery procedures
- Runbook for disaster scenarios

## Monitoring and Troubleshooting

### Cluster Component Health

**Component Status Monitoring:**
```bash
# Check component status
kubectl get componentstatuses

# Check node status
kubectl get nodes -o wide

# Check pod status in kube-system namespace
kubectl get pods -n kube-system

# Check cluster events
kubectl get events --sort-by=.metadata.creationTimestamp
```

### Log Analysis

**Control Plane Logs:**
```bash
# API server logs
sudo journalctl -u kube-apiserver -f

# etcd logs
sudo journalctl -u etcd -f

# Scheduler logs
sudo journalctl -u kube-scheduler -f

# Controller manager logs
sudo journalctl -u kube-controller-manager -f
```

**Node Component Logs:**
```bash
# kubelet logs
sudo journalctl -u kubelet -f

# Container runtime logs
sudo journalctl -u containerd -f

# kube-proxy logs
kubectl logs -n kube-system -l k8s-app=kube-proxy
```

### Performance Monitoring

**Resource Utilization:**
```bash
# Node resource usage
kubectl top nodes

# Pod resource usage
kubectl top pods --all-namespaces

# Control plane component metrics
kubectl get --raw /metrics | grep apiserver
```

This comprehensive understanding of Kubernetes cluster architecture provides the foundation for deploying, managing, and troubleshooting production-grade Kubernetes environments. The architecture's modular design allows for flexibility in deployment while maintaining consistency across different infrastructure providers.

## Next Section

Continue to [Networking](03_Networking.md) to learn about Kubernetes networking concepts, CNI plugins, and network policies.
