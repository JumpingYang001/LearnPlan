# Kubernetes Storage

Kubernetes storage provides persistent data management for containerized applications. Understanding storage concepts is crucial for running stateful applications, databases, and any workload that requires data persistence beyond the container lifecycle.

## Kubernetes Storage Concepts

### Volumes

Volumes provide a way for containers to access persistent storage. Unlike ephemeral container filesystems, volumes persist across container restarts and can be shared between containers in a pod.

### Volume Types

Kubernetes supports multiple volume types:
- **emptyDir**: Temporary directory that exists for the pod's lifetime
- **hostPath**: Mounts a directory from the host node's filesystem
- **configMap/secret**: Mounts configuration data or secrets as files
- **persistentVolumeClaim**: Claims persistent storage from the cluster
- **Cloud provider volumes**: AWS EBS, Azure Disk, GCP Persistent Disk
- **Network storage**: NFS, Ceph, GlusterFS

### PersistentVolumes (PV)

PersistentVolumes are cluster-wide storage resources provisioned by administrators or dynamically created by storage classes. They have a lifecycle independent of pods.

### PersistentVolumeClaims (PVC) 

PersistentVolumeClaims are requests for storage by users. They specify size, access modes, and storage class requirements. PVCs bind to available PVs that meet their requirements.

### StorageClasses

StorageClasses provide a way to describe different types of storage available in the cluster. They enable dynamic provisioning of PersistentVolumes with specific characteristics.

## Storage Classes

Storage classes define the properties and provisioning details for different types of storage. They enable dynamic provisioning and allow users to request storage with specific performance and availability characteristics.

### AWS EBS Storage Class

Amazon Elastic Block Store provides persistent block storage for EC2 instances.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ebs-gp3
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3                    # General Purpose SSD v3
  iops: "3000"                # Input/Output Operations Per Second
  throughput: "125"           # Throughput in MiB/s
  encrypted: "true"           # Enable encryption at rest
  kmsKeyId: "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
volumeBindingMode: WaitForFirstConsumer    # Delay binding until pod is scheduled
allowVolumeExpansion: true    # Allow volume expansion
reclaimPolicy: Delete         # Delete PV when PVC is deleted
mountOptions:
- debug
- noatime                     # Improve performance by not updating access times
```

**Key Parameters Explained:**
- **type**: EBS volume type (gp3, gp2, io1, io2, st1, sc1)
- **iops**: Baseline IOPS for gp3 and provisioned IOPS for io1/io2
- **throughput**: Baseline throughput for gp3 volumes
- **encrypted**: Enable EBS encryption
- **volumeBindingMode**: Controls when volume binding occurs
- **allowVolumeExpansion**: Enables online volume expansion

### Azure Disk Storage Class

Azure Managed Disks provide persistent storage for Azure VMs.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: azure-disk-premium
  annotations:
    storageclass.kubernetes.io/is-default-class: "false"
provisioner: disk.csi.azure.com
parameters:
  skuName: Premium_LRS        # Premium SSD with local redundancy
  location: eastus            # Azure region
  resourceGroup: myResourceGroup
  cachingMode: ReadOnly       # Host caching mode
  diskEncryptionSetID: "/subscriptions/subscription-id/resourceGroups/rg/providers/Microsoft.Compute/diskEncryptionSets/des"
  networkAccessPolicy: AllowAll
  diskAccessID: "/subscriptions/subscription-id/resourceGroups/rg/providers/Microsoft.Compute/diskAccesses/diskAccess"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
reclaimPolicy: Delete
```

**Azure Disk SKU Options:**
- **Standard_LRS**: Standard HDD with local redundancy
- **Premium_LRS**: Premium SSD with local redundancy  
- **StandardSSD_LRS**: Standard SSD with local redundancy
- **UltraSSD_LRS**: Ultra SSD with local redundancy (highest performance)

### GCP Persistent Disk Storage Class

Google Cloud Persistent Disks provide durable block storage for Google Compute Engine.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gcp-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "false"
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-ssd                # Persistent disk type
  zones: us-central1-a,us-central1-b,us-central1-c    # Allowed zones
  replication-type: regional-pd    # Regional persistent disk for HA
  provisioned-iops-on-create: "1000"    # Provisioned IOPS
  provisioned-throughput-on-create: "50"    # Provisioned throughput (MB/s)
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
reclaimPolicy: Delete
```

**GCP Disk Types:**
- **pd-standard**: Standard persistent disk (HDD)
- **pd-ssd**: SSD persistent disk
- **pd-balanced**: Balanced persistent disk (cost-effective SSD)
- **pd-extreme**: Extreme persistent disk (highest IOPS)
  replication-type: regional-pd
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
reclaimPolicy: Delete
```

### Local Storage Class

Local storage provides high-performance storage using node-local storage devices.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete
```

### NFS Storage Class

Network File System storage for shared access across multiple pods.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs-storage
provisioner: nfs.csi.k8s.io
parameters:
  server: nfs-server.example.com
  share: /nfs/share
  # Mount options for NFS
mountOptions:
  - nfsvers=4.1
  - rsize=1048576
  - wsize=1048576
  - hard
  - intr
volumeBindingMode: Immediate
allowVolumeExpansion: false
reclaimPolicy: Retain
```

## PersistentVolumes and PersistentVolumeClaims

### Static Provisioning

In static provisioning, administrators pre-create PersistentVolumes that can be claimed by users.

#### Creating a PersistentVolume

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-database
  labels:
    type: database
    performance: high
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteOnce        # Single node read-write
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ebs-gp3
  awsElasticBlockStore:
    volumeID: vol-0123456789abcdef0
    fsType: ext4
  mountOptions:
    - noatime
    - nodiratime
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: topology.kubernetes.io/zone
          operator: In
          values:
          - us-west-2a
          - us-west-2b
```

#### Creating a PersistentVolumeClaim

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: database-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: ebs-gp3
  selector:
    matchLabels:
      type: database
      performance: high
```

### Dynamic Provisioning

Dynamic provisioning automatically creates PVs when PVCs are created, based on StorageClass specifications.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: app-data-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: ebs-gp3    # References the StorageClass
```

### Access Modes

PersistentVolumes support different access modes:

- **ReadWriteOnce (RWO)**: Volume can be mounted read-write by a single node
- **ReadOnlyMany (ROX)**: Volume can be mounted read-only by many nodes  
- **ReadWriteMany (RWX)**: Volume can be mounted read-write by many nodes
- **ReadWriteOncePod (RWOP)**: Volume can be mounted read-write by a single pod

```yaml
# Example showing different access modes
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-storage-pvc
spec:
  accessModes:
    - ReadWriteMany      # Required for shared access
  resources:
    requests:
      storage: 200Gi
  storageClassName: nfs-storage
```

## Using Storage in Pods and Deployments

### Basic Volume Mount

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: database-pod
spec:
  containers:
  - name: database
    image: postgres:14
    env:
    - name: POSTGRES_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: password
    - name: PGDATA
      value: /var/lib/postgresql/data/pgdata
    ports:
    - containerPort: 5432
    volumeMounts:
    - name: database-storage
      mountPath: /var/lib/postgresql/data
    resources:
      requests:
        memory: "1Gi"
        cpu: "500m"
      limits:
        memory: "2Gi"
        cpu: "1000m"
  volumes:
  - name: database-storage
    persistentVolumeClaim:
      claimName: database-pvc
```

### StatefulSet with Volume Claims

StatefulSets provide stable, unique network identifiers and persistent storage for each pod.

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web-app
spec:
  serviceName: web-service
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web
        image: nginx:1.21
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: web-storage
          mountPath: /usr/share/nginx/html
        - name: config-storage
          mountPath: /etc/nginx/conf.d
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
  volumeClaimTemplates:
  - metadata:
      name: web-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: ebs-gp3
      resources:
        requests:
          storage: 10Gi
  - metadata:
      name: config-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: ebs-gp3
      resources:
        requests:
          storage: 1Gi
```

### Multi-Container Pod with Shared Storage

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-with-sidecar
spec:
  containers:
  - name: web-server
    image: nginx:1.21
    ports:
    - containerPort: 80
    volumeMounts:
    - name: shared-data
      mountPath: /usr/share/nginx/html
    - name: config-volume
      mountPath: /etc/nginx/conf.d
  - name: content-updater
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "while true; do echo $(date) > /shared/index.html; sleep 60; done"]
    volumeMounts:
    - name: shared-data
      mountPath: /shared
  - name: log-collector
    image: fluent/fluent-bit:1.9
    volumeMounts:
    - name: shared-data
      mountPath: /logs
      readOnly: true
    - name: fluent-bit-config
      mountPath: /fluent-bit/etc
  volumes:
  - name: shared-data
    persistentVolumeClaim:
      claimName: web-data-pvc
  - name: config-volume
    configMap:
      name: nginx-config
  - name: fluent-bit-config
    configMap:
      name: fluent-bit-config
```

## CSI Driver Examples

### AWS EFS CSI Driver
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-12345678
  directoryPerms: "0755"
  gidRangeStart: "1000"
  gidRangeEnd: "2000"
  basePath: "/dynamic_provisioning"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: efs-claim
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: efs-sc
  resources:
    requests:
      storage: 5Gi
```

### Longhorn Distributed Storage
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: longhorn
provisioner: driver.longhorn.io
allowVolumeExpansion: true
parameters:
  numberOfReplicas: "3"
  staleReplicaTimeout: "2880"
  baseImage: "longhornio/longhorn-engine:v1.4.0"
  fromBackup: ""
  fsType: "ext4"
  dataLocality: "disabled"
```

### Rook Ceph Storage
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: rook-ceph-block
provisioner: rook-ceph.rbd.csi.ceph.com
parameters:
  clusterID: rook-ceph
  pool: replicapool
  imageFormat: "2"
  imageFeatures: layering
  csi.storage.k8s.io/provisioner-secret-name: rook-csi-rbd-provisioner
  csi.storage.k8s.io/provisioner-secret-namespace: rook-ceph
  csi.storage.k8s.io/controller-expand-secret-name: rook-csi-rbd-provisioner
  csi.storage.k8s.io/controller-expand-secret-namespace: rook-ceph
  csi.storage.k8s.io/node-stage-secret-name: rook-csi-rbd-node
  csi.storage.k8s.io/node-stage-secret-namespace: rook-ceph
  csi.storage.k8s.io/fstype: ext4
allowVolumeExpansion: true
reclaimPolicy: Delete
```

## Volume Types and Configurations

### EmptyDir Volume
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: emptydir-pod
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: cache-volume
      mountPath: /cache
  - name: sidecar
    image: busybox
    command: ['sh', '-c', 'while true; do echo hello > /cache/hello.txt; sleep 30; done']
    volumeMounts:
    - name: cache-volume
      mountPath: /cache
  volumes:
  - name: cache-volume
    emptyDir:
      sizeLimit: 1Gi
      medium: Memory  # Use RAM for faster access
```

### HostPath Volume
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hostpath-pod
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: host-volume
      mountPath: /host-data
  volumes:
  - name: host-volume
    hostPath:
      path: /var/lib/myapp
      type: DirectoryOrCreate
```

### ConfigMap and Secret Volumes
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: config-pod
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config
    - name: secret-volume
      mountPath: /etc/secrets
      readOnly: true
  volumes:
  - name: config-volume
    configMap:
      name: app-config
      defaultMode: 0644
  - name: secret-volume
    secret:
      secretName: app-secrets
      defaultMode: 0600
```

## Backup and Restore

### Velero Backup Configuration
```yaml
apiVersion: velero.io/v1
kind: Backup
metadata:
  name: mysql-backup
spec:
  includedNamespaces:
  - production
  includedResources:
  - persistentvolumeclaims
  - persistentvolumes
  - pods
  - secrets
  - configmaps
  labelSelector:
    matchLabels:
      app: mysql
  snapshotVolumes: true
  ttl: 720h0m0s  # 30 days
```

### Backup Script
```bash
#!/bin/bash
# Kubernetes storage backup script

NAMESPACE="production"
BACKUP_DIR="/backup/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

echo "=== Kubernetes Storage Backup ==="

# Backup PVCs
echo "1. Backing up PersistentVolumeClaims..."
kubectl get pvc -n $NAMESPACE -o yaml > $BACKUP_DIR/pvcs.yaml

# Backup PVs
echo "2. Backing up PersistentVolumes..."
kubectl get pv -o yaml > $BACKUP_DIR/pvs.yaml

# Backup StatefulSets
echo "3. Backing up StatefulSets..."
kubectl get statefulsets -n $NAMESPACE -o yaml > $BACKUP_DIR/statefulsets.yaml

# Create volume snapshots if supported
echo "4. Creating volume snapshots..."
kubectl get pvc -n $NAMESPACE --no-headers | while read pvc rest; do
  cat <<EOF | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: ${pvc}-snapshot-$(date +%Y%m%d)
  namespace: $NAMESPACE
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: $pvc
EOF
done

echo "=== Backup Complete ==="
echo "Backup saved to: $BACKUP_DIR"
```

### Storage Monitoring
```yaml
# Storage metrics monitoring
apiVersion: v1
kind: Service
metadata:
  name: storage-metrics
  labels:
    app: storage-exporter
spec:
  ports:
  - port: 8080
    name: metrics
  selector:
    app: storage-exporter
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: storage-exporter
spec:
  selector:
    matchLabels:
      app: storage-exporter
  template:
    metadata:
      labels:
        app: storage-exporter
    spec:
      containers:
      - name: storage-exporter
        image: prom/node-exporter:latest
        ports:
        - containerPort: 8080
          name: metrics
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        - name: rootfs
          mountPath: /rootfs
          readOnly: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      - name: rootfs
        hostPath:
          path: /
```

## Storage Performance Optimization

### I/O Performance Tuning

```yaml
# High-performance storage configuration
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: high-performance-ssd
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "16000"              # Maximum IOPS for gp3
  throughput: "1000"         # Maximum throughput
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
mountOptions:
- noatime                    # Don't update access times
- nodiratime                 # Don't update directory access times
- discard                    # Enable TRIM for SSDs
- barrier=0                  # Disable write barriers for performance
```

### Resource Quotas for Storage

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: storage-quota
  namespace: development
spec:
  hard:
    requests.storage: 100Gi                    # Total storage requests
    persistentvolumeclaims: 10                 # Maximum number of PVCs
    ebs-gp3.storageclass.storage.k8s.io/requests.storage: 50Gi    # Per storage class limit
    ebs-gp3.storageclass.storage.k8s.io/persistentvolumeclaims: 5
```

## Storage Monitoring and Troubleshooting

### Storage Metrics

```yaml
# ServiceMonitor for storage metrics
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: csi-driver-metrics
spec:
  selector:
    matchLabels:
      app: ebs-csi-controller
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Storage Health Checks

```bash
#!/bin/bash
# Storage health check script

echo "=== Kubernetes Storage Health Check ==="

# Check PVC status
echo "1. Checking PVC status..."
kubectl get pvc --all-namespaces -o wide

# Check PV status
echo -e "\n2. Checking PV status..."
kubectl get pv -o wide

# Check StorageClass
echo -e "\n3. Checking StorageClasses..."
kubectl get storageclass

# Check CSI drivers
echo -e "\n4. Checking CSI drivers..."
kubectl get csidriver

# Check for storage-related events
echo -e "\n5. Checking storage-related events..."
kubectl get events --all-namespaces --field-selector reason=FailedMount,reason=FailedAttachVolume

# Check node storage capacity
echo -e "\n6. Checking node storage capacity..."
kubectl describe nodes | grep -A 5 "Allocated resources"

echo -e "\n=== Storage Health Check Complete ==="
```

### Common Storage Issues and Solutions

#### 1. Pod Stuck in Pending State

```bash
# Diagnosis
kubectl describe pod <pod-name>
kubectl describe pvc <pvc-name>

# Common causes and solutions:
# - Insufficient storage capacity in the cluster
# - StorageClass not found or misconfigured
# - Volume zone mismatch with pod scheduling
# - Storage quota exceeded
```

#### 2. Volume Mount Failures

```bash
# Check for mount errors
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# Check CSI driver logs
kubectl logs -n kube-system -l app=ebs-csi-controller
kubectl logs -n kube-system -l app=ebs-csi-node
```

#### 3. Performance Issues

```yaml
# Pod with performance monitoring
apiVersion: v1
kind: Pod
metadata:
  name: storage-perf-test
spec:
  containers:
  - name: fio
    image: ljishen/fio
    command: ["fio"]
    args: 
    - "--name=randwrite"
    - "--ioengine=libaio"
    - "--iodepth=32"
    - "--rw=randwrite"
    - "--bs=4k"
    - "--direct=1"
    - "--size=1G"
    - "--numjobs=4"
    - "--runtime=60"
    - "--group_reporting"
    - "--filename=/data/test"
    volumeMounts:
    - name: test-volume
      mountPath: /data
    resources:
      requests:
        memory: "1Gi"
        cpu: "500m"
  volumes:
  - name: test-volume
    persistentVolumeClaim:
      claimName: performance-test-pvc
```

This comprehensive guide to Kubernetes storage covers all aspects from basic concepts to advanced performance optimization and troubleshooting. Understanding these storage patterns and configurations is essential for running production workloads that require persistent data.

## Next Section

Continue to [Workload Management](05_Workload_Management.md) to learn about deployment strategies, scaling, and workload controllers.
