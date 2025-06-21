# Project 05: Kubernetes-based Data Platform

## Project Overview

This project involves building a comprehensive data platform on Kubernetes that can handle various data workloads including batch processing, stream processing, data storage, and analytics. You'll learn to deploy and manage stateful services, implement robust backup and recovery systems, create auto-scaling mechanisms, and establish secure data access patterns.

## Architecture Diagram

The Kubernetes-based Data Platform implements a comprehensive multi-layered architecture that handles the complete data lifecycle from ingestion to analytics. The platform leverages Kubernetes' orchestration capabilities to provide scalable, reliable, and secure data services.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    Kubernetes Control Plane                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ API Server  │  │   etcd      │  │ Scheduler   │  │ Controller  │  │  Admission  │                  │
│  │             │  │             │  │             │  │  Manager    │  │ Controllers │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────────────────────────────────────────────────┐
│                                   Data Governance Layer                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   Apache    │  │   Vault     │  │   Network   │  │   RBAC      │  │  Audit      │                  │
│  │   Ranger    │  │   Secret    │  │  Policies   │  │ Policies    │  │  Logging    │                  │
│  │(Access Ctrl)│  │   Mgmt      │  │             │  │             │  │             │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────────────────────────────────────────────────┐
│                                   Data Analytics Layer                                                │
│                                                                                                        │
│  ┌───────────────────────────────┐  ┌───────────────────────────────┐  ┌───────────────────────────┐   │
│  │        Query Engines         │  │       Visualization           │  │      Notebooks           │   │
│  │                               │  │                               │  │                           │   │
│  │  ┌─────────────────────────┐  │  │  ┌─────────────────────────┐  │  │  ┌─────────────────────┐ │   │
│  │  │        Trino            │  │  │  │   Apache Superset       │  │  │  │    JupyterHub       │ │   │
│  │  │   (Distributed SQL)     │  │  │  │                         │  │  │  │                     │ │   │
│  │  │                         │  │  │  │  ┌─────────┐            │  │  │  │  ┌─────────┐        │ │   │
│  │  │ ┌─────────┐ ┌─────────┐ │  │  │  │  │Dashboard│            │  │  │  │  │ Jupyter │        │ │   │
│  │  │ │Coordin. │ │Workers  │ │  │  │  │  │  └─────────┘            │  │  │  │  └─────────┘        │ │   │
│  │  │ └─────────┘ └─────────┘ │  │  │  │  ┌─────────┐            │  │  │  │  ┌─────────┐        │ │   │
│  │  │ ┌─────────┐ ┌─────────┐ │  │  │  │ │Checkpnt │ │ State   │ │  │  │  │ ┌─────────┐        │ │   │
│  │  │ │ History │ │ Metrics │ │  │  │  │ │Storage  │ │ Backend │ │  │  │  │ └─────────┘        │ │   │
│  │  │ │ Server  │ │ Export  │ │  │  │  │ └─────────┘ └─────────┘ │  │  │  │ ┌─────────┐        │ │   │
│  │  │ └─────────┘ └─────────┘ │  │  │  └─────────────────────────┘  │  │  │ │Worker   │        │ │   │
│  │  └─────────────────────────┘  │  └───────────────────────────────┘  │  │ │ Pods    │        │ │   │
│  └───────────────────────────────┘                                     │  │ └─────────┘        │ │   │
│                                                                        │  └─────────────────────┘ │   │
│                                                                        └───────────────────────────┘   │
└─────────────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────────────────────────────────────────────────┐
│                                    Data Storage Layer                                                 │
│                                                                                                        │
│  ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────┐│
│  │   Relational DBs        │ │   Document Stores       │ │   Wide Column Stores    │ │   Object Store  ││
│  │                         │ │                         │ │                         │ │                 ││
│  │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │ ┌─────────────┐ ││
│  │ │    PostgreSQL       │ │ │ │      MongoDB        │ │ │ │     Cassandra       │ │ │ │    MinIO    │ ││
│  │ │     Cluster         │ │ │ │     Replica Set     │ │ │ │      Cluster        │ │ │ │   Cluster   │ ││
│  │ │                     │ │ │ │                     │ │ │ │                     │ │ │ │             │ ││
│  │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────────┐ │ ││
│  │ │ │Prim.│ │Replica  │ │ │ │ │ │Prim.│ │Replica  │ │ │ │ │ │Seed │ │ Node    │ │ │ │ │ │ Node 1  │ │ ││
│  │ │ └─────┘ └─────────┘ │ │ │ │ └─────┘ └─────────┘ │ │ │ │ └─────┘ └─────────┘ │ │ │ │ └─────────┘ │ ││
│  │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────────┐ │ ││
│  │ │ │Conn │ │ Backup  │ │ │ │ │ │Shard│ │ Agent   │ │ │ │ │ │Repl │ │ Backup  │ │ │ │ │ │ Node 2  │ │ ││
│  │ │ │Pool │ │ Agent   │ │ │ │ │ │Mgr  │ │ Agent   │ │ │ │ │ │Fact │ │ Agent   │ │ │ │ │ └─────────┘ │ ││
│  │ │ └─────┘ └─────────┘ │ │ │ │ └─────┘ └─────────┘ │ │ │ │ └─────┘ └─────────┘ │ │ │ │ ┌─────────┐ │ ││
│  │ └─────────────────────┘ │ │ └─────────────────────┘ │ │ └─────────────────────┘ │ │ │ │ Node 3  │ │ ││
│  └─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘ │ │ └─────────┘ │ ││
│                                                                                      │ │ ┌─────────┐ │ ││
│  ┌─────────────────────────┐ ┌─────────────────────────┐                            │ │ │ Node 4  │ │ ││
│  │      Caching Layer      │ │    Message Queues       │                            │ │ └─────────┘ │ ││
│  │                         │ │                         │                            │ └─────────────┘ ││
│  │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │                            │                 ││
│  │ │       Redis         │ │ │ │       Kafka         │ │                            │                 ││
│  │ │      Cluster        │ │ │ │      Cluster        │ │                            │                 ││
│  │ │                     │ │ │ │                     │ │                            │                 ││
│  │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │                            │                 ││
│  │ │ │Mstr │ │ Replica │ │ │ │ │ │Broker│ │Zookeeper│ │ │                            │                 ││
│  │ │ └─────┘ └─────────┘ │ │ │ │ └─────┘ └─────────┘ │ │                            │                 ││
│  │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │                            │                 ││
│  │ │ │Sntnl│ │ Metrics │ │ │ │ │ │Topic│ │ Metrics │ │ │                            │                 ││
│  │ │ └─────┘ └─────────┘ │ │ │ │ │Mgmt │ └─────────┘ │ │                            │                 ││
│  │ └─────────────────────┘ │ │ │ └─────┘             │ │                            │                 ││
│  └─────────────────────────┘ └─────────────────────────┘                            │                 ││
└─────────────────────────────────────────────────────────────────────────────────────┘                 ││
                                          │                                                                ││
┌─────────────────────────────────────────▼───────────────────────────────────────────────────────────────┐
│                                  Operations Layer                                                     │
│                                                                                                        │
│  ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────┐│
│  │       Monitoring        │ │      Backup & DR        │ │      Auto Scaling       │ │   Networking    ││
│  │                         │ │                         │ │                         │ │                 ││
│  │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │ ┌─────────────┐ ││
│  │ │    Prometheus       │ │ │ │       Velero        │ │ │ │        HPA          │ │ │ │   Ingress   │ ││
│  │ │                     │ │ │ │                     │ │ │ │                     │ │ │ │ Controllers │ ││
│  │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │ │ └─────────────┘ ││
│  │ │ │Srv  │ │ Alert   │ │ │ │ │ │Ctrl │ │ Storage │ │ │ │ │ │Ctrl │ │ Metrics │ │ │ │ ┌─────────────┐ ││
│  │ │ │Mon  │ │ Manager │ │ │ │ │ │     │ │ Plugin  │ │ │ │ │ │     │ │ Server  │ │ │ │ │   Service   │ ││
│  │ │ └─────┘ └─────────┘ │ │ │ │ └─────┘ └─────────┘ │ │ │ │ └─────┘ └─────────┘ │ │ │ │    Mesh     │ ││
│  │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │ │ │ ┌─────┐ ┌─────────┐ │ │ │ │   (Istio)   │ ││
│  │ │ │TSDB │ │ Grafana │ │ │ │ │ │Schd │ │ Cross   │ │ │ │ │ │VPA  │ │ Cluster │ │ │ │ └─────────────┘ ││
│  │ │ └─────┘ └─────────┘ │ │ │ │ │     │ │ Region  │ │ │ │ │ │     │ │Autoscale│ │ │ │ ┌─────────────┐ ││
│  │ └─────────────────────┘ │ │ │ └─────┘ └─────────┘ │ │ │ └─────┘ └─────────┘ │ │ │ │   Network   │ ││
│  └─────────────────────────┘ │ └─────────────────────┘ │ └─────────────────────┘ │ │ │  Policies   │ ││
│                              └─────────────────────────┘                         │ │ └─────────────┘ ││
│  ┌─────────────────────────┐ ┌─────────────────────────┐                         │ └─────────────────┘│
│  │       Security          │ │      Resource Mgmt      │                         │                   │
│  │                         │ │                         │                         │                   │
│  │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │                         │                   │
│  │ │      Falco          │ │ │ │   Resource Quotas   │ │                         │                   │
│  │ │   (Runtime Sec)     │ │ │ │                     │ │                         │                   │
│  │ │                     │ │ │ │ ┌─────┐ ┌─────────┐ │ │                         │                   │
│  │ │ ┌─────┐ ┌─────────┐ │ │ │ │ │CPU  │ │ Memory  │ │ │                         │                   │
│  │ │ │Rules│ │ Alerts  │ │ │ │ │ └─────┘ └─────────┘ │ │                         │                   │
│  │ │ └─────┘ └─────────┘ │ │ │ │ ┌─────┐ ┌─────────┐ │ │                         │                   │
│  │ │ ┌─────┐ ┌─────────┐ │ │ │ │ │Strg │ │ Network │ │ │                         │                   │
│  │ │ │Logs │ │ Metrics │ │ │ │ │ └─────┘ └─────────┘ │ │                         │                   │
│  │ │ └─────┘ └─────────┘ │ │ │ └─────────────────────┘ │                         │                   │
│  │ └─────────────────────┘ │ └─────────────────────────┘                         │                   │
│  └─────────────────────────┘                                                     │                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Data Flow:
1. Data Ingestion → Kafka → Stream Processing (Flink) → Storage Layer
2. Batch Processing → Spark → Reads from Storage → Processes → Writes to Storage
3. Analytics → Trino → Queries across all storage systems → Results to Visualization
4. Workflow Orchestration → Airflow → Manages ETL pipelines and data workflows
5. Backup → Velero → Snapshots storage and config → Cross-region replication
6. Monitoring → Prometheus → Collects metrics → Grafana visualization → Alerting
```


## Learning Objectives

By completing this project, you will:
- Deploy and manage stateful data services on Kubernetes
- Implement comprehensive backup and disaster recovery strategies
- Create intelligent scaling mechanisms for data workloads
- Establish secure data access patterns and governance
- Build data pipelines using cloud-native technologies
- Implement monitoring and observability for data platforms
- Design multi-tenant data isolation and resource management

## Project Scope

### Data Platform Components

The platform will include:
1. **Data Storage Layer**: Distributed databases, object storage, and caching
2. **Data Processing Layer**: Batch and stream processing engines
3. **Data Analytics Layer**: OLAP engines and query interfaces
4. **Data Governance Layer**: Security, access control, and compliance
5. **Operations Layer**: Monitoring, backup, and disaster recovery

## Phase 1: Platform Architecture and Planning (1 week)

### 1.1 Architecture Design

**High-Level Architecture:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: data-platform-architecture
data:
  architecture.yaml: |
    data_platform:
      storage_layer:
        - name: postgresql-cluster
          type: relational_database
          purpose: transactional_data
          ha: true
          backup: automated
        - name: mongodb-cluster
          type: document_database
          purpose: semi_structured_data
          ha: true
          backup: automated
        - name: cassandra-cluster
          type: wide_column_database
          purpose: time_series_data
          ha: true
          backup: automated
        - name: redis-cluster
          type: cache
          purpose: session_cache
          ha: true
          backup: periodic
        - name: minio-cluster
          type: object_storage
          purpose: data_lake
          ha: true
          backup: cross_region
      
      processing_layer:
        - name: apache-spark
          type: batch_processing
          auto_scaling: true
          resource_management: dynamic
        - name: apache-flink
          type: stream_processing
          auto_scaling: true
          resource_management: dynamic
        - name: apache-airflow
          type: workflow_orchestration
          ha: true
      
      analytics_layer:
        - name: apache-superset
          type: visualization
          ha: true
        - name: jupyter-hub
          type: notebooks
          multi_tenant: true
        - name: trino
          type: query_engine
          auto_scaling: true
      
      governance_layer:
        - name: apache-ranger
          type: access_control
          integration: all_services
        - name: vault
          type: secret_management
          ha: true
```

### 1.2 Infrastructure Requirements

**Cluster Specifications:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-requirements
data:
  requirements.yaml: |
    cluster_config:
      node_pools:
        - name: control-plane
          node_count: 3
          machine_type: n1-standard-4
          disk_size: 100GB
          labels:
            role: control-plane
        
        - name: data-storage
          node_count: 6
          machine_type: n1-standard-8
          disk_size: 500GB
          local_ssd: true
          labels:
            role: data-storage
            workload: stateful
        
        - name: data-processing
          node_count: 3
          machine_type: n1-highmem-8
          disk_size: 200GB
          auto_scaling:
            min_nodes: 3
            max_nodes: 20
          labels:
            role: data-processing
            workload: compute
        
        - name: analytics
          node_count: 2
          machine_type: n1-standard-4
          disk_size: 100GB
          labels:
            role: analytics
            workload: interactive
      
      storage_classes:
        - name: fast-ssd
          provisioner: kubernetes.io/gce-pd
          parameters:
            type: pd-ssd
          reclaim_policy: Retain
        
        - name: bulk-storage
          provisioner: kubernetes.io/gce-pd
          parameters:
            type: pd-standard
          reclaim_policy: Retain
        
        - name: local-storage
          provisioner: kubernetes.io/no-provisioner
          volumeBindingMode: WaitForFirstConsumer
```

### 1.3 Security and Compliance Framework

**Security Configuration:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-framework
data:
  security-policies.yaml: |
    security_framework:
      network_policies:
        - name: data-storage-isolation
          applies_to: data-storage
          ingress:
            - from: data-processing
            - from: analytics
          egress:
            - to: backup-storage
        
        - name: processing-isolation
          applies_to: data-processing
          ingress:
            - from: api-gateway
          egress:
            - to: data-storage
            - to: object-storage
      
      pod_security_standards:
        level: restricted
        exceptions:
          - namespace: data-platform
            workloads: ["cassandra", "mongodb"]
            violations: ["runAsRoot"]
      
      rbac:
        service_accounts:
          - name: data-engineer
            permissions: ["read", "write", "execute"]
            resources: ["processing", "storage"]
          - name: data-scientist
            permissions: ["read", "execute"]
            resources: ["analytics", "notebooks"]
          - name: data-admin
            permissions: ["admin"]
            resources: ["all"]
      
      encryption:
        at_rest: true
        in_transit: true
        key_management: vault
```

## Phase 2: Data Storage Layer Implementation (2 weeks)

### 2.1 PostgreSQL High Availability Cluster

**PostgreSQL Cluster with Patroni:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql-cluster
  namespace: data-platform
spec:
  serviceName: postgresql-cluster
  replicas: 3
  selector:
    matchLabels:
      app: postgresql
      role: cluster
  template:
    metadata:
      labels:
        app: postgresql
        role: cluster
    spec:
      serviceAccountName: postgresql-cluster
      securityContext:
        fsGroup: 999
      containers:
      - name: postgresql
        image: postgres:14.9
        ports:
        - containerPort: 5432
          name: postgres
        - containerPort: 8008
          name: patroni
        env:
        - name: PGUSER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-credentials
              key: password
        - name: PATRONI_KUBERNETES_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: PATRONI_KUBERNETES_LABELS
          value: '{app: postgresql, role: cluster}'
        - name: PATRONI_SUPERUSER_USERNAME
          value: postgres
        - name: PATRONI_SUPERUSER_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-credentials
              key: password
        - name: PATRONI_REPLICATION_USERNAME
          value: replicator
        - name: PATRONI_REPLICATION_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-credentials
              key: replication-password
        - name: PATRONI_SCOPE
          value: postgresql-cluster
        - name: PATRONI_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: PATRONI_POSTGRESQL_DATA_DIR
          value: /var/lib/postgresql/data/pgdata
        - name: PATRONI_POSTGRESQL_PGPASS
          value: /tmp/pgpass0
        - name: PATRONI_POSTGRESQL_LISTEN
          value: '0.0.0.0:5432'
        - name: PATRONI_RESTAPI_LISTEN
          value: '0.0.0.0:8008'
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql
        livenessProbe:
          httpGet:
            path: /liveness
            port: 8008
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readiness
            port: 8008
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
      - name: pgbouncer
        image: pgbouncer/pgbouncer:1.18.0
        ports:
        - containerPort: 6432
          name: pgbouncer
        env:
        - name: DATABASES_HOST
          value: "127.0.0.1"
        - name: DATABASES_PORT
          value: "5432"
        - name: DATABASES_USER
          value: postgres
        - name: DATABASES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-credentials
              key: password
        volumeMounts:
        - name: pgbouncer-config
          mountPath: /etc/pgbouncer
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
      volumes:
      - name: postgres-config
        configMap:
          name: postgresql-config
      - name: pgbouncer-config
        configMap:
          name: pgbouncer-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgresql-cluster
  namespace: data-platform
spec:
  ports:
  - port: 5432
    targetPort: 6432
    name: postgres
  - port: 8008
    targetPort: 8008
    name: patroni
  selector:
    app: postgresql
    role: cluster
---
apiVersion: v1
kind: Service
metadata:
  name: postgresql-cluster-primary
  namespace: data-platform
  annotations:
    service.alpha.kubernetes.io/tolerate-unready-endpoints: "true"
spec:
  ports:
  - port: 5432
    targetPort: 6432
    name: postgres
  selector:
    app: postgresql
    role: cluster
    master: "true"
```

### 2.2 MongoDB Replica Set

**MongoDB Cluster Configuration:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mongodb-cluster
  namespace: data-platform
spec:
  serviceName: mongodb-cluster
  replicas: 3
  selector:
    matchLabels:
      app: mongodb
  template:
    metadata:
      labels:
        app: mongodb
    spec:
      terminationGracePeriodSeconds: 30
      containers:
      - name: mongodb
        image: mongo:6.0.8
        command:
        - mongod
        - --replSet=rs0
        - --bind_ip_all
        - --auth
        - --keyFile=/etc/mongodb/keyfile
        ports:
        - containerPort: 27017
          name: mongodb
        env:
        - name: MONGO_INITDB_ROOT_USERNAME
          value: admin
        - name: MONGO_INITDB_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mongodb-credentials
              key: password
        volumeMounts:
        - name: mongodb-data
          mountPath: /data/db
        - name: mongodb-config
          mountPath: /etc/mongodb
        - name: mongodb-keyfile
          mountPath: /etc/mongodb-keyfile
        livenessProbe:
          exec:
            command:
            - mongo
            - --eval
            - "db.adminCommand('ping')"
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - mongo
            - --eval
            - "db.adminCommand('ping')"
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      - name: mongodb-exporter
        image: percona/mongodb_exporter:0.39.0
        ports:
        - containerPort: 9216
          name: metrics
        env:
        - name: MONGODB_URI
          value: "mongodb://admin:$(MONGO_PASSWORD)@localhost:27017/?authSource=admin"
        - name: MONGO_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mongodb-credentials
              key: password
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
      volumes:
      - name: mongodb-config
        configMap:
          name: mongodb-config
      - name: mongodb-keyfile
        secret:
          secretName: mongodb-keyfile
          defaultMode: 0600
  volumeClaimTemplates:
  - metadata:
      name: mongodb-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 50Gi
```

### 2.3 Apache Cassandra Cluster

**Cassandra StatefulSet:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: cassandra-cluster
  namespace: data-platform
spec:
  serviceName: cassandra
  replicas: 3
  selector:
    matchLabels:
      app: cassandra
  template:
    metadata:
      labels:
        app: cassandra
    spec:
      containers:
      - name: cassandra
        image: cassandra:4.1.1
        ports:
        - containerPort: 7000
          name: intra-node
        - containerPort: 7001
          name: tls-intra-node
        - containerPort: 7199
          name: jmx
        - containerPort: 9042
          name: cql
        - containerPort: 9160
          name: thrift
        env:
        - name: CASSANDRA_SEEDS
          value: "cassandra-cluster-0.cassandra.data-platform.svc.cluster.local"
        - name: CASSANDRA_CLUSTER_NAME
          value: "DataPlatformCluster"
        - name: CASSANDRA_DC
          value: "DC1"
        - name: CASSANDRA_RACK
          value: "Rack1"
        - name: CASSANDRA_ENDPOINT_SNITCH
          value: "GossipingPropertyFileSnitch"
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        volumeMounts:
        - name: cassandra-data
          mountPath: /var/lib/cassandra
        - name: cassandra-config
          mountPath: /etc/cassandra
        livenessProbe:
          exec:
            command:
            - /bin/bash
            - -c
            - "nodetool status"
          initialDelaySeconds: 90
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - /bin/bash
            - -c
            - "nodetool status | grep $POD_IP | grep UN"
          initialDelaySeconds: 60
          periodSeconds: 10
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
      volumes:
      - name: cassandra-config
        configMap:
          name: cassandra-config
  volumeClaimTemplates:
  - metadata:
      name: cassandra-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

### 2.4 MinIO Object Storage

**MinIO Distributed Setup:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: minio-cluster
  namespace: data-platform
spec:
  serviceName: minio
  replicas: 4
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        image: minio/minio:RELEASE.2023-06-19T19-52-50Z
        command:
        - /bin/bash
        - -c
        args:
        - minio server http://minio-cluster-{0...3}.minio.data-platform.svc.cluster.local/data --console-address ":9001"
        ports:
        - containerPort: 9000
          name: api
        - containerPort: 9001
          name: console
        env:
        - name: MINIO_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: minio-credentials
              key: access-key
        - name: MINIO_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: minio-credentials
              key: secret-key
        - name: MINIO_PROMETHEUS_AUTH_TYPE
          value: "public"
        volumeMounts:
        - name: minio-data
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /minio/health/live
            port: 9000
          initialDelaySeconds: 120
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /minio/health/ready
            port: 9000
          initialDelaySeconds: 120
          periodSeconds: 20
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: minio-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: bulk-storage
      resources:
        requests:
          storage: 500Gi
---
apiVersion: v1
kind: Service
metadata:
  name: minio
  namespace: data-platform
spec:
  ports:
  - port: 9000
    targetPort: 9000
    name: api
  - port: 9001
    targetPort: 9001
    name: console
  selector:
    app: minio
```

## Phase 3: Data Processing Layer (2 weeks)

### 3.1 Apache Spark on Kubernetes

**Spark Operator Installation:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spark-operator
  namespace: data-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spark-operator
  template:
    metadata:
      labels:
        app: spark-operator
    spec:
      serviceAccountName: spark-operator
      containers:
      - name: spark-operator
        image: gcr.io/spark-operator/spark-operator:v1beta2-1.3.8-3.1.1
        args:
        - -v=2
        - -namespace=data-platform
        - -enable-batch-scheduler=true
        - -enable-resource-quota-enforcement=true
        - -enable-metrics=true
        - -metrics-bind-address=0.0.0.0:10254
        ports:
        - containerPort: 10254
          name: metrics
        resources:
          requests:
            memory: "512Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "200m"
```

**Spark Application Template:**
```yaml
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: data-processing-job
  namespace: data-platform
spec:
  type: Scala
  mode: cluster
  image: "spark:3.4.0"
  imagePullPolicy: Always
  mainClass: com.example.DataProcessingJob
  mainApplicationFile: "s3a://data-platform/jars/data-processing-job.jar"
  sparkVersion: "3.4.0"
  restartPolicy:
    type: OnFailure
    onFailureRetries: 3
    onFailureRetryInterval: 10
    onSubmissionFailureRetries: 5
    onSubmissionFailureRetryInterval: 20
  driver:
    cores: 1
    coreLimit: "1200m"
    memory: "2g"
    serviceAccount: spark-driver
    labels:
      version: 3.4.0
    env:
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: minio-credentials
          key: access-key
    - name: AWS_SECRET_ACCESS_KEY
      valueFrom:
        secretKeyRef:
          name: minio-credentials
          key: secret-key
    - name: S3_ENDPOINT
      value: "http://minio.data-platform.svc.cluster.local:9000"
  executor:
    cores: 2
    instances: 3
    memory: "4g"
    serviceAccount: spark-executor
    labels:
      version: 3.4.0
  dynamicAllocation:
    enabled: true
    initialExecutors: 2
    minExecutors: 1
    maxExecutors: 10
  sparkConf:
    "spark.sql.adaptive.enabled": "true"
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem"
    "spark.hadoop.fs.s3a.endpoint": "http://minio.data-platform.svc.cluster.local:9000"
    "spark.hadoop.fs.s3a.path.style.access": "true"
    "spark.hadoop.fs.s3a.connection.ssl.enabled": "false"
  monitoring:
    enabled: true
    metricsProperties: |
      *.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet
      *.sink.prometheusServlet.path=/metrics/prometheus
      master.sink.prometheusServlet.path=/metrics/master/prometheus
      applications.sink.prometheusServlet.path=/metrics/applications/prometheus
```

### 3.2 Apache Flink for Stream Processing

**Flink Cluster Setup:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flink-jobmanager
  namespace: data-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flink
      component: jobmanager
  template:
    metadata:
      labels:
        app: flink
        component: jobmanager
    spec:
      containers:
      - name: jobmanager
        image: flink:1.17.1-scala_2.12
        args: ["jobmanager"]
        ports:
        - containerPort: 6123
          name: rpc
        - containerPort: 6124
          name: blob-server
        - containerPort: 8081
          name: webui
        livenessProbe:
          httpGet:
            path: /
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        env:
        - name: JOB_MANAGER_RPC_ADDRESS
          value: "flink-jobmanager"
        - name: JOB_MANAGER_MEMORY_PROCESS_SIZE
          value: "2048m"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        volumeMounts:
        - name: flink-config
          mountPath: /opt/flink/conf
      volumes:
      - name: flink-config
        configMap:
          name: flink-config
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flink-taskmanager
  namespace: data-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: flink
      component: taskmanager
  template:
    metadata:
      labels:
        app: flink
        component: taskmanager
    spec:
      containers:
      - name: taskmanager
        image: flink:1.17.1-scala_2.12
        args: ["taskmanager"]
        ports:
        - containerPort: 6122
          name: rpc
        - containerPort: 6125
          name: query-state
        livenessProbe:
          tcpSocket:
            port: 6122
          initialDelaySeconds: 30
          periodSeconds: 10
        env:
        - name: JOB_MANAGER_RPC_ADDRESS
          value: "flink-jobmanager"
        - name: TASK_MANAGER_MEMORY_PROCESS_SIZE
          value: "4096m"
        - name: TASK_MANAGER_NUMBER_OF_TASK_SLOTS
          value: "4"
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
        volumeMounts:
        - name: flink-config
          mountPath: /opt/flink/conf
      volumes:
      - name: flink-config
        configMap:
          name: flink-config
```

### 3.3 Apache Airflow for Workflow Orchestration

**Airflow Helm Values:**
```yaml
# airflow-values.yaml
executor: "KubernetesExecutor"

postgresql:
  enabled: true
  persistence:
    enabled: true
    size: 20Gi
    storageClass: fast-ssd

redis:
  enabled: true
  persistence:
    enabled: true
    size: 10Gi
    storageClass: fast-ssd

webserver:
  replicas: 2
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"
  
  service:
    type: ClusterIP
    port: 8080

scheduler:
  replicas: 2
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

workers:
  enabled: false  # Using KubernetesExecutor

config:
  core:
    dags_are_paused_at_creation: 'False'
    load_examples: 'False'
    max_active_runs_per_dag: 16
    parallelism: 32
    max_active_tasks_per_dag: 16
  
  webserver:
    expose_config: 'True'
  
  kubernetes:
    namespace: data-platform
    worker_container_repository: apache/airflow
    worker_container_tag: 2.6.1
    delete_worker_pods: 'True'
    delete_worker_pods_on_success: 'True'
    
dags:
  persistence:
    enabled: true
    size: 10Gi
    storageClassName: fast-ssd
    accessMode: ReadWriteMany
  
  gitSync:
    enabled: true
    repo: https://github.com/example/data-platform-dags.git
    branch: main
    subPath: "dags"
    wait: 60

logs:
  persistence:
    enabled: true
    size: 20Gi
    storageClassName: bulk-storage

extraSecrets:
  minio-connection:
    data: |
      AIRFLOW_CONN_MINIO_DEFAULT: s3://admin:minioadmin@minio.data-platform.svc.cluster.local:9000
  
  postgres-connection:
    data: |
      AIRFLOW_CONN_POSTGRES_DEFAULT: postgresql://postgres:password@postgresql-cluster.data-platform.svc.cluster.local:5432/dataplatform
```

## Phase 4: Backup and Disaster Recovery (1 week)

### 4.1 Velero Backup Solution

**Velero Installation Configuration:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: velero-backup-config
  namespace: velero
data:
  backup-strategy.yaml: |
    backup_strategies:
      databases:
        frequency: "daily"
        retention: "30d"
        backup_type: "full"
        pre_hooks:
          - name: postgres-dump
            container: postgresql
            command: 
              - /bin/bash
              - -c
              - "pg_dumpall -h localhost -U postgres > /tmp/backup.sql"
          - name: mongodb-dump
            container: mongodb
            command:
              - /bin/bash
              - -c
              - "mongodump --host localhost --out /tmp/backup"
        post_hooks:
          - name: cleanup-temp
            container: backup-agent
            command:
              - /bin/bash
              - -c
              - "rm -rf /tmp/backup*"
      
      persistent_volumes:
        frequency: "hourly"
        retention: "7d"
        snapshot: true
      
      applications:
        frequency: "daily"
        retention: "14d"
        include_namespaces:
          - data-platform
          - monitoring
        exclude_resources:
          - events
          - pods
          - replicasets
```

**Backup Scripts and Jobs:**
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: data-platform
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup-agent
            image: postgres:14.9
            command:
            - /bin/bash
            - -c
            - |
              set -e
              
              # PostgreSQL Backup
              echo "Starting PostgreSQL backup..."
              PGPASSWORD=$POSTGRES_PASSWORD pg_dumpall -h postgresql-cluster -U postgres > /backup/postgres-$(date +%Y%m%d_%H%M%S).sql
              
              # MongoDB Backup
              echo "Starting MongoDB backup..."
              mongodump --host mongodb-cluster --out /backup/mongodb-$(date +%Y%m%d_%H%M%S)
              
              # Upload to S3
              echo "Uploading backups to S3..."
              aws s3 sync /backup/ s3://data-platform-backups/$(date +%Y-%m-%d)/
              
              # Cleanup old local backups
              find /backup -name "*.sql" -mtime +1 -delete
              find /backup -name "mongodb-*" -mtime +1 -exec rm -rf {} \;
              
              echo "Backup completed successfully"
            env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgresql-credentials
                  key: password
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: access-key
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: secret-key
            - name: S3_ENDPOINT
              value: "http://minio.data-platform.svc.cluster.local:9000"
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
            resources:
              requests:
                memory: "512Mi"
                cpu: "200m"
              limits:
                memory: "1Gi"
                cpu: "500m"
          volumes:
          - name: backup-storage
            emptyDir:
              sizeLimit: 50Gi
          restartPolicy: OnFailure
```

### 4.2 Cross-Region Disaster Recovery

**Disaster Recovery Automation:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: disaster-recovery-plan
  namespace: data-platform
data:
  dr-runbook.yaml: |
    disaster_recovery:
      rto: "4 hours"  # Recovery Time Objective
      rpo: "1 hour"   # Recovery Point Objective
      
      backup_locations:
        primary: "s3://data-platform-backups-us-east-1"
        secondary: "s3://data-platform-backups-us-west-2"
        tertiary: "gcs://data-platform-backups-europe"
      
      recovery_procedures:
        database_recovery:
          postgresql:
            - restore_from_backup: "latest_full"
            - apply_wal_files: "continuous"
            - verify_data_integrity: true
            - update_connection_strings: true
          
          mongodb:
            - restore_replica_set: "from_backup"
            - reconfigure_sharding: true
            - verify_replication: true
          
          cassandra:
            - restore_snapshots: "all_nodes"
            - rebuild_secondary_indexes: true
            - run_repair: "full_cluster"
        
        application_recovery:
          - restore_kubernetes_resources: true
          - update_service_endpoints: true
          - verify_application_health: true
          - run_smoke_tests: true
      
      monitoring_during_recovery:
        - track_recovery_progress: true
        - monitor_resource_usage: true
        - alert_on_failures: true
        - document_recovery_actions: true
```

## Phase 5: Auto-scaling and Resource Management (1 week)

### 5.1 Horizontal Pod Autoscaler (HPA) Configuration

**Custom Metrics HPA:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: spark-executor-hpa
  namespace: data-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: spark-executor-pool
  minReplicas: 2
  maxReplicas: 50
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
  - type: Pods
    pods:
      metric:
        name: spark_tasks_pending
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 10
        periodSeconds: 15
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: flink-taskmanager-hpa
  namespace: data-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: flink-taskmanager
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
  - type: Pods
    pods:
      metric:
        name: flink_job_backpressure
      target:
        type: AverageValue
        averageValue: "0.7"
```

### 5.2 Vertical Pod Autoscaler (VPA)

**VPA Configuration for Data Services:**
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: postgresql-vpa
  namespace: data-platform
spec:
  targetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: postgresql-cluster
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: postgresql
      maxAllowed:
        cpu: "4"
        memory: 8Gi
      minAllowed:
        cpu: 500m
        memory: 1Gi
      controlledResources: ["cpu", "memory"]
    - containerName: pgbouncer
      maxAllowed:
        cpu: 500m
        memory: 512Mi
      minAllowed:
        cpu: 100m
        memory: 64Mi
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: mongodb-vpa
  namespace: data-platform
spec:
  targetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: mongodb-cluster
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: mongodb
      maxAllowed:
        cpu: "4"
        memory: 8Gi
      minAllowed:
        cpu: 500m
        memory: 1Gi
      controlledResources: ["cpu", "memory"]
```

### 5.3 Cluster Autoscaler Configuration

**Node Pool Auto-scaling:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
  namespace: kube-system
data:
  nodes.yaml: |
    node_groups:
      data_processing:
        min_size: 2
        max_size: 50
        desired_capacity: 5
        instance_types: ["n1-highmem-8", "n1-highmem-16"]
        scaling_policies:
          scale_up_delay: "60s"
          scale_down_delay: "300s"
          scale_down_utilization_threshold: 0.6
        labels:
          workload: "data-processing"
          auto-scaling: "enabled"
        taints:
          - key: "workload"
            value: "data-processing"
            effect: "NoSchedule"
      
      data_storage:
        min_size: 3
        max_size: 10
        desired_capacity: 6
        instance_types: ["n1-standard-8"]
        scaling_policies:
          scale_up_delay: "120s"
          scale_down_delay: "600s"
          scale_down_utilization_threshold: 0.7
        labels:
          workload: "data-storage"
        taints:
          - key: "workload"
            value: "data-storage"
            effect: "NoSchedule"
```

## Phase 6: Data Access Patterns and Security (1 week)

### 6.1 Data Access Layer with Trino

**Trino Distributed Query Engine:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trino-coordinator
  namespace: data-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trino
      component: coordinator
  template:
    metadata:
      labels:
        app: trino
        component: coordinator
    spec:
      containers:
      - name: trino
        image: trinodb/trino:420
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: TRINO_ENVIRONMENT
          value: "production"
        volumeMounts:
        - name: trino-config
          mountPath: /etc/trino
        - name: trino-catalogs
          mountPath: /etc/trino/catalog
        livenessProbe:
          httpGet:
            path: /v1/info
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v1/info
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 5
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
      volumes:
      - name: trino-config
        configMap:
          name: trino-coordinator-config
      - name: trino-catalogs
        configMap:
          name: trino-catalogs
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trino-worker
  namespace: data-platform
spec:
  replicas: 5
  selector:
    matchLabels:
      app: trino
      component: worker
  template:
    metadata:
      labels:
        app: trino
        component: worker
    spec:
      containers:
      - name: trino
        image: trinodb/trino:420
        env:
        - name: TRINO_ENVIRONMENT
          value: "production"
        volumeMounts:
        - name: trino-config
          mountPath: /etc/trino
        - name: trino-catalogs
          mountPath: /etc/trino/catalog
        resources:
          requests:
            memory: "8Gi"
            cpu: "2000m"
          limits:
            memory: "16Gi"
            cpu: "4000m"
      volumes:
      - name: trino-config
        configMap:
          name: trino-worker-config
      - name: trino-catalogs
        configMap:
          name: trino-catalogs
```

**Trino Catalog Configuration:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trino-catalogs
  namespace: data-platform
data:
  postgresql.properties: |
    connector.name=postgresql
    connection-url=jdbc:postgresql://postgresql-cluster:5432/dataplatform
    connection-user=trino
    connection-password=${ENV:POSTGRES_PASSWORD}
    
  mongodb.properties: |
    connector.name=mongodb
    mongodb.connection-url=mongodb://mongodb-cluster:27017
    mongodb.credentials=admin:${ENV:MONGO_PASSWORD}@admin
    
  cassandra.properties: |
    connector.name=cassandra
    cassandra.contact-points=cassandra-cluster-0.cassandra,cassandra-cluster-1.cassandra,cassandra-cluster-2.cassandra
    cassandra.load-policy.use-dc-aware=true
    cassandra.load-policy.dc-aware.local-dc=DC1
    
  hive.properties: |
    connector.name=hive-hadoop2
    hive.metastore.uri=thrift://hive-metastore:9083
    hive.s3.endpoint=http://minio:9000
    hive.s3.path-style-access=true
    hive.s3.ssl.enabled=false
    hive.s3.aws-access-key=${ENV:S3_ACCESS_KEY}
    hive.s3.aws-secret-key=${ENV:S3_SECRET_KEY}
```

### 6.2 Apache Ranger for Data Governance

**Ranger Admin Setup:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ranger-admin
  namespace: data-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ranger-admin
  template:
    metadata:
      labels:
        app: ranger-admin
    spec:
      containers:
      - name: ranger-admin
        image: apache/ranger:2.4.0
        ports:
        - containerPort: 6080
          name: http
        env:
        - name: RANGER_DB_TYPE
          value: "postgres"
        - name: RANGER_DB_HOST
          value: "postgresql-cluster"
        - name: RANGER_DB_NAME
          value: "ranger"
        - name: RANGER_DB_USER
          value: "ranger"
        - name: RANGER_DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ranger-credentials
              key: db-password
        volumeMounts:
        - name: ranger-config
          mountPath: /opt/ranger-admin/conf
        livenessProbe:
          httpGet:
            path: /login.jsp
            port: 6080
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /login.jsp
            port: 6080
          initialDelaySeconds: 60
          periodSeconds: 10
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
      volumes:
      - name: ranger-config
        configMap:
          name: ranger-admin-config
```

### 6.3 Data Access Patterns Implementation

**Data Access Service:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-access-api
  namespace: data-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: data-access-api
  template:
    metadata:
      labels:
        app: data-access-api
    spec:
      containers:
      - name: api-server
        image: data-platform/access-api:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: TRINO_ENDPOINT
          value: "http://trino-coordinator:8080"
        - name: POSTGRES_CONNECTION
          value: "postgresql://postgresql-cluster:5432/dataplatform"
        - name: MONGO_CONNECTION
          value: "mongodb://mongodb-cluster:27017"
        - name: CASSANDRA_ENDPOINTS
          value: "cassandra-cluster-0.cassandra:9042,cassandra-cluster-1.cassandra:9042,cassandra-cluster-2.cassandra:9042"
        - name: RANGER_ENDPOINT
          value: "http://ranger-admin:6080"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: jwt-secret
        volumeMounts:
        - name: api-config
          mountPath: /app/config
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
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: api-config
        configMap:
          name: data-access-api-config
```

## Phase 7: Monitoring and Observability (1 week)

### 7.1 Comprehensive Monitoring Stack

**Prometheus Configuration for Data Platform:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-data-platform
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
    - "/etc/prometheus/rules/*.yml"
    
    scrape_configs:
    # PostgreSQL monitoring
    - job_name: 'postgresql'
      static_configs:
      - targets: ['postgresql-cluster.data-platform:9187']
      metrics_path: /metrics
      scrape_interval: 30s
    
    # MongoDB monitoring
    - job_name: 'mongodb'
      static_configs:
      - targets: ['mongodb-cluster.data-platform:9216']
      metrics_path: /metrics
      scrape_interval: 30s
    
    # Cassandra monitoring
    - job_name: 'cassandra'
      static_configs:
      - targets: ['cassandra-cluster.data-platform:7070']
      metrics_path: /metrics
      scrape_interval: 30s
    
    # MinIO monitoring
    - job_name: 'minio'
      static_configs:
      - targets: ['minio.data-platform:9000']
      metrics_path: /minio/v2/metrics/cluster
      scrape_interval: 30s
    
    # Spark monitoring
    - job_name: 'spark'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - data-platform
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_spark_role]
        action: keep
        regex: driver|executor
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
    
    # Flink monitoring
    - job_name: 'flink'
      static_configs:
      - targets: ['flink-jobmanager.data-platform:9249']
      - targets: ['flink-taskmanager.data-platform:9249']
      metrics_path: /metrics
      scrape_interval: 15s
    
    # Trino monitoring
    - job_name: 'trino'
      static_configs:
      - targets: ['trino-coordinator.data-platform:8080']
      metrics_path: /v1/jmx/mbean
      params:
        pattern: ['trino.execution:*']
      scrape_interval: 30s
    
    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager.monitoring:9093
  
  data-platform-rules.yml: |
    groups:
    - name: data-platform.rules
      rules:
      # Database availability rules
      - alert: PostgreSQLDown
        expr: up{job="postgresql"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL instance is down"
          description: "PostgreSQL instance has been down for more than 2 minutes"
      
      - alert: MongoDBDown
        expr: up{job="mongodb"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "MongoDB instance is down"
          description: "MongoDB instance has been down for more than 2 minutes"
      
      # Storage rules
      - alert: HighDiskUsage
        expr: (node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High disk usage detected"
          description: "Disk usage is above 85% on {{ $labels.instance }}"
      
      # Processing rules
      - alert: SparkJobFailure
        expr: increase(spark_driver_DAGScheduler_job_allJobs_failed[5m]) > 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Spark job failure detected"
          description: "{{ $value }} Spark jobs have failed in the last 5 minutes"
      
      - alert: FlinkJobLatencyHigh
        expr: flink_jobmanager_job_latency_p99 > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High Flink job latency"
          description: "Flink job latency P99 is {{ $value }}ms"
```

### 7.2 Grafana Dashboards

**Data Platform Dashboard Configuration:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: monitoring
data:
  data-platform-overview.json: |
    {
      "dashboard": {
        "id": null,
        "title": "Data Platform Overview",
        "tags": ["data-platform"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "Database Health",
            "type": "stat",
            "targets": [
              {
                "expr": "up{job=~\"postgresql|mongodb|cassandra\"}",
                "legendFormat": "{{ job }}"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
          },
          {
            "id": 2,
            "title": "Processing Jobs",
            "type": "graph",
            "targets": [
              {
                "expr": "spark_driver_DAGScheduler_stage_runningStages",
                "legendFormat": "Spark Running Stages"
              },
              {
                "expr": "flink_jobmanager_numRunningJobs",
                "legendFormat": "Flink Running Jobs"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
          },
          {
            "id": 3,
            "title": "Storage Usage",
            "type": "graph",
            "targets": [
              {
                "expr": "sum(pg_stat_database_size) by (datname)",
                "legendFormat": "PostgreSQL - {{ datname }}"
              },
              {
                "expr": "mongodb_ss_wt_cache_bytes_currently_in_the_cache",
                "legendFormat": "MongoDB Cache"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
          },
          {
            "id": 4,
            "title": "Query Performance",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(pg_stat_database_tup_returned[5m])",
                "legendFormat": "PostgreSQL Tuples/sec"
              },
              {
                "expr": "trino_execution_QueryManager_RunningQueries",
                "legendFormat": "Trino Running Queries"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
          }
        ],
        "time": {"from": "now-1h", "to": "now"},
        "refresh": "30s"
      }
    }
```

### Key Architectural Components

**Data Storage Layer:**
- **PostgreSQL Cluster**: ACID-compliant transactional data with high availability through Patroni
- **MongoDB Replica Set**: Document storage for semi-structured data with automatic failover
- **Cassandra Cluster**: Distributed wide-column store for time-series and high-volume data
- **MinIO Cluster**: S3-compatible object storage for data lake and backup storage
- **Redis Cluster**: High-performance caching and session storage
- **Apache Kafka**: Distributed event streaming platform for real-time data pipelines

**Data Processing Layer:**
- **Apache Spark**: Unified analytics engine for large-scale batch processing
- **Apache Flink**: Stream processing framework for real-time data processing
- **Apache Airflow**: Workflow orchestration platform for complex data pipelines

**Data Analytics Layer:**
- **Trino**: Distributed SQL query engine for interactive analytics across multiple data sources
- **Apache Superset**: Modern data visualization and exploration platform
- **JupyterHub**: Multi-user notebook environment for data science and analytics

**Operations Layer:**
- **Monitoring**: Prometheus and Grafana for comprehensive observability
- **Backup & DR**: Velero for automated backup and disaster recovery
- **Auto-scaling**: HPA, VPA, and Cluster Autoscaler for dynamic resource management
- **Security**: Falco for runtime security monitoring and threat detection

**Data Governance Layer:**
- **Apache Ranger**: Comprehensive security framework for data access control
- **HashiCorp Vault**: Secrets management and encryption key management
- **Network Policies**: Kubernetes-native network segmentation and security
- **RBAC**: Role-based access control for platform resources
- **Audit Logging**: Comprehensive audit trail for compliance and security

### Data Flow Patterns

**Batch Processing Pipeline:**
1. Data ingestion from external sources into object storage (MinIO)
2. Airflow orchestrates Spark jobs for data transformation and analysis
3. Processed data stored in appropriate databases (PostgreSQL, MongoDB, Cassandra)
4. Results available for analytics through Trino query engine

**Stream Processing Pipeline:**
1. Real-time data streams into Kafka topics
2. Flink processes streaming data with low latency
3. Results written to databases and caching layer (Redis)
4. Real-time dashboards updated through Superset

**Analytics and Reporting:**
1. Trino federates queries across all data sources
2. JupyterHub provides interactive data science environment
3. Superset delivers self-service analytics and visualization
4. API layer provides programmatic access to data services

**Operational Workflows:**
1. Automated backup schedules ensure data protection
2. Monitoring systems track performance and health
3. Auto-scaling responds to workload demands
4. Security policies enforce access controls and compliance
