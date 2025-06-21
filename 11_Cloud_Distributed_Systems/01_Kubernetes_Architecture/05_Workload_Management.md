# Kubernetes Workload Management

Workload management in Kubernetes involves understanding different deployment patterns, managing application lifecycle, and implementing strategies for updates, scaling, and high availability. This comprehensive guide covers the essential workload controllers and deployment strategies.

## Deployment Strategies

Deployment strategies determine how applications are updated and rolled out in production environments. Choosing the right strategy depends on factors such as downtime tolerance, resource availability, and rollback requirements.

### Rolling Update Deployment

Rolling updates gradually replace old instances with new ones, ensuring zero downtime during deployments.

**Key Benefits:**
- Zero downtime deployments
- Gradual rollout reduces risk
- Automatic rollback on failure
- Resource efficient

**Rolling Update Configuration:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rolling-update-app
  labels:
    app: rolling-update-app
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Can create 2 extra pods during update
      maxUnavailable: 1  # Max 1 pod can be unavailable during update
  selector:
    matchLabels:
      app: rolling-update-app
  template:
    metadata:
      labels:
        app: rolling-update-app
        version: v2
    spec:
      containers:
      - name: app
        image: nginx:1.21
        ports:
        - containerPort: 80
          name: http
        readinessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        env:
        - name: VERSION
          value: "v2.0.0"
        - name: ENVIRONMENT
          value: "production"
```

**Key Parameters Explained:**
- **maxSurge**: Maximum number of pods that can be created above the desired number
- **maxUnavailable**: Maximum number of pods that can be unavailable during the update
- **readinessProbe**: Determines when a pod is ready to receive traffic
- **livenessProbe**: Determines when to restart a container

### Blue-Green Deployment

Blue-green deployment involves running two identical production environments, switching traffic between them during updates.

**Blue Environment (Current):**

```yaml
# Blue deployment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-blue
  labels:
    app: webapp
    version: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
      version: blue
  template:
    metadata:
      labels:
        app: webapp
        version: blue
    spec:
      containers:
      - name: webapp
        image: myapp:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: VERSION
          value: "blue"
        - name: COLOR
          value: "blue"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
# Service pointing to blue deployment
apiVersion: v1
kind: Service
metadata:
  name: webapp-service
spec:
  selector:
    app: webapp
    version: blue    # Points to blue deployment
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

**Green Environment (New):**

```yaml
# Green deployment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-green
  labels:
    app: webapp
    version: green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
      version: green
  template:
    metadata:
      labels:
        app: webapp
        version: green
    spec:
      containers:
      - name: webapp
        image: myapp:v2.0.0    # New version
        ports:
        - containerPort: 8080
        env:
        - name: VERSION
          value: "green"
        - name: COLOR
          value: "green"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

**Traffic Switching:**

```bash
# Switch traffic to green deployment
kubectl patch service webapp-service -p '{"spec":{"selector":{"version":"green"}}}'

# Rollback to blue deployment if needed
kubectl patch service webapp-service -p '{"spec":{"selector":{"version":"blue"}}}'

# Clean up old deployment after successful switch
kubectl delete deployment webapp-blue
```

### Canary Deployment

Canary deployment gradually rolls out changes to a small subset of users before full deployment.

**Main Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-stable
  labels:
    app: webapp
    version: stable
spec:
  replicas: 8    # 80% of traffic
  selector:
    matchLabels:
      app: webapp
      version: stable
  template:
    metadata:
      labels:
        app: webapp
        version: stable
    spec:
      containers:
      - name: webapp
        image: myapp:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: VERSION
          value: "stable"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
# Canary deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-canary
  labels:
    app: webapp
    version: canary
spec:
  replicas: 2    # 20% of traffic
  selector:
    matchLabels:
      app: webapp
      version: canary
  template:
    metadata:
      labels:
        app: webapp
        version: canary
    spec:
      containers:
      - name: webapp
        image: myapp:v2.0.0    # New version
        ports:
        - containerPort: 8080
        env:
        - name: VERSION
          value: "canary"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
# Service distributing traffic between stable and canary
apiVersion: v1
kind: Service
metadata:
  name: webapp-service
spec:
  selector:
    app: webapp    # Selects both stable and canary pods
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

**Advanced Canary with Ingress:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: webapp-canary-ingress
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "20"    # 20% traffic to canary
    nginx.ingress.kubernetes.io/canary-by-header: "X-Canary"
    nginx.ingress.kubernetes.io/canary-by-header-value: "always"
spec:
  ingressClassName: nginx
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: webapp-canary-service
            port:
              number: 80
```

### A/B Testing Deployment

A/B testing involves running multiple versions simultaneously to compare performance and user engagement.

```yaml
# A/B Testing with Feature Flags
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-variant-a
  labels:
    app: webapp
    variant: a
spec:
  replicas: 5
  selector:
    matchLabels:
      app: webapp
      variant: a
  template:
    metadata:
      labels:
        app: webapp
        variant: a
    spec:
      containers:
      - name: webapp
        image: myapp:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: VARIANT
          value: "A"
        - name: FEATURE_FLAG_NEW_UI
          value: "false"
        - name: FEATURE_FLAG_ANALYTICS
          value: "true"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp-variant-b
  labels:
    app: webapp
    variant: b
spec:
  replicas: 5
  selector:
    matchLabels:
      app: webapp
      variant: b
  template:
    metadata:
      labels:
        app: webapp
        variant: b
    spec:
      containers:
      - name: webapp
        image: myapp:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: VARIANT
          value: "B"
        - name: FEATURE_FLAG_NEW_UI
          value: "true"     # New UI enabled for variant B
        - name: FEATURE_FLAG_ANALYTICS
          value: "true"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
```

## StatefulSets

StatefulSets manage stateful applications that require stable network identities, persistent storage, and ordered deployment/scaling.

### Basic StatefulSet Configuration

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web-app
  labels:
    app: web-app
spec:
  serviceName: "web-service"    # Headless service name
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
        env:
        - name: MY_POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: MY_POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: MY_POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
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
  podManagementPolicy: OrderedReady    # Pods are created in order
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
---
# Headless service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: web-service
  labels:
    app: web-app
spec:
  ports:
  - port: 80
    name: web
  clusterIP: None    # Headless service
  selector:
    app: web-app
```

### Database StatefulSet Example

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      initContainers:
      - name: init-mysql
        image: mysql:8.0
        command:
        - bash
        - "-c"
        - |
          set -ex
          # Generate mysql server-id from pod ordinal index.
          [[ `hostname` =~ -([0-9]+)$ ]] || exit 1
          ordinal=${BASH_REMATCH[1]}
          echo [mysqld] > /mnt/conf.d/server-id.cnf
          # Add an offset to avoid reserved server-id=0 value.
          echo server-id=$((100 + $ordinal)) >> /mnt/conf.d/server-id.cnf
          # Copy appropriate conf.d files from config-map to emptyDir.
          if [[ $ordinal -eq 0 ]]; then
            cp /mnt/config-map/primary.cnf /mnt/conf.d/
          else
            cp /mnt/config-map/replica.cnf /mnt/conf.d/
          fi
        volumeMounts:
        - name: conf
          mountPath: /mnt/conf.d
        - name: config-map
          mountPath: /mnt/config-map
      - name: clone-mysql
        image: gcr.io/google-samples/xtrabackup:1.0
        command:
        - bash
        - "-c"
        - |
          set -ex
          # Skip the clone if data already exists.
          [[ -d /var/lib/mysql/mysql ]] && exit 0
          # Skip the clone on primary (ordinal index 0).
          [[ `hostname` =~ -([0-9]+)$ ]] || exit 1
          ordinal=${BASH_REMATCH[1]}
          [[ $ordinal -eq 0 ]] && exit 0
          # Clone data from previous peer.
          ncat --recv-only mysql-$(($ordinal-1)).mysql 3307 | xbstream -x -C /var/lib/mysql
          # Prepare the backup.
          xtrabackup --prepare --target-dir=/var/lib/mysql
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
          subPath: mysql
        - name: conf
          mountPath: /etc/mysql/conf.d
      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ALLOW_EMPTY_PASSWORD
          value: "1"
        ports:
        - name: mysql
          containerPort: 3306
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
          subPath: mysql
        - name: conf
          mountPath: /etc/mysql/conf.d
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
        livenessProbe:
          exec:
            command: ["mysqladmin", "ping"]
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          exec:
            command: ["mysql", "-h", "127.0.0.1", "-e", "SELECT 1"]
          initialDelaySeconds: 5
          periodSeconds: 2
          timeoutSeconds: 1
      - name: xtrabackup
        image: gcr.io/google-samples/xtrabackup:1.0
        ports:
        - name: xtrabackup
          containerPort: 3307
        command:
        - bash
        - "-c"
        - |
          set -ex
          cd /var/lib/mysql
          # Determine binlog position of cloned data, if any.
          if [[ -f xtrabackup_slave_info && "x$(<xtrabackup_slave_info)" != "x" ]]; then
            # XtraBackup already generated a partial "CHANGE MASTER TO" query
            cat xtrabackup_slave_info | sed -E 's/;$//g' > change_master_to.sql.in
            # Ignore xtrabackup_binlog_info in this case (for new primary).
            rm -f xtrabackup_slave_info xtrabackup_binlog_info
          elif [[ -f xtrabackup_binlog_info ]]; then
            # We're cloning directly from primary. Parse binlog position.
            [[ `cat xtrabackup_binlog_info` =~ ^(.*?)[[:space:]]+(.*?)$ ]] || exit 1
            rm -f xtrabackup_binlog_info xtrabackup_slave_info
            echo "CHANGE MASTER TO MASTER_LOG_FILE='${BASH_REMATCH[1]}',\
                  MASTER_LOG_POS=${BASH_REMATCH[2]}" > change_master_to.sql.in
          fi
          # Check if we need to complete a clone by starting replication.
          if [[ -f change_master_to.sql.in ]]; then
            echo "Waiting for mysqld to be ready (accepting connections)"
            until mysql -h 127.0.0.1 -e "SELECT 1"; do sleep 1; done
            echo "Initializing replication from clone position"
            mysql -h 127.0.0.1 \
                  -e "$(<change_master_to.sql.in), \
                          MASTER_HOST='mysql-0.mysql', \
                          MASTER_USER='root', \
                          MASTER_PASSWORD='', \
                          MASTER_CONNECT_RETRY=10; \
                        START SLAVE;" || exit 1
            # In case of container restart, attempt this at-most-once.
            mv change_master_to.sql.in change_master_to.sql.orig
          fi
          # Start a server to send backups to newly added replicas.
          exec ncat --listen --keep-open --send-only --max-conns=1 3307 -c \
            "xtrabackup --backup --slave-info --stream=xbstream --host=127.0.0.1 --user=root"
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
          subPath: mysql
        - name: conf
          mountPath: /etc/mysql/conf.d
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
      volumes:
      - name: conf
        emptyDir: {}
      - name: config-map
        configMap:
          name: mysql
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: ebs-gp3
      resources:
        requests:
          storage: 20Gi
```

## DaemonSets

DaemonSets ensure that a copy of a pod runs on all (or some) nodes in the cluster.

### Basic DaemonSet Configuration

```yaml
metadata:
  name: app-blue
  labels:
    app: myapp
    version: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: blue
  template:
    metadata:
      labels:
        app: myapp
        version: blue
    spec:
      containers:
      - name: app
        image: myapp:v1.0
        ports:
        - containerPort: 8080
---
# Green deployment (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-green
  labels:
    app: myapp
    version: green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: green
  template:
    metadata:
      labels:
        app: myapp
        version: green
    spec:
      containers:
      - name: app
        image: myapp:v2.0
        ports:
        - containerPort: 8080
---
# Service pointing to blue (switch to green when ready)
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
    version: blue  # Change to 'green' for cutover
  ports:
  - port: 80
    targetPort: 8080
```

### Canary Deployment with Istio
```yaml
# Main deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: myapp
      version: stable
  template:
    metadata:
      labels:
        app: myapp
        version: stable
    spec:
      containers:
      - name: app
        image: myapp:v1.0
        ports:
        - containerPort: 8080
---
# Canary deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: myapp
      version: canary
  template:
    metadata:
      labels:
        app: myapp
        version: canary
    spec:
      containers:
      - name: app
        image: myapp:v2.0
        ports:
        - containerPort: 8080
---
# Istio Virtual Service for traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: myapp
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: myapp-service
        subset: canary
  - route:
    - destination:
        host: myapp-service
        subset: stable
      weight: 90
    - destination:
        host: myapp-service
        subset: canary
      weight: 10
---
# Destination Rule
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: myapp
spec:
  host: myapp-service
  subsets:
  - name: stable
    labels:
      version: stable
  - name: canary
    labels:
      version: canary
```

## StatefulSets

### MySQL Master-Slave StatefulSet
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      initContainers:
      - name: init-mysql
        image: mysql:8.0
        command:
        - bash
        - "-c"
        - |
          set -ex
          # Generate mysql server-id from pod ordinal index.
          [[ `hostname` =~ -([0-9]+)$ ]] || exit 1
          ordinal=${BASH_REMATCH[1]}
          echo [mysqld] > /mnt/conf.d/server-id.cnf
          # Add an offset to avoid reserved server-id=0 value.
          echo server-id=$((100 + $ordinal)) >> /mnt/conf.d/server-id.cnf
          # Copy appropriate conf.d files from config-map to emptyDir.
          if [[ $ordinal -eq 0 ]]; then
            cp /mnt/config-map/primary.cnf /mnt/conf.d/
          else
            cp /mnt/config-map/replica.cnf /mnt/conf.d/
          fi
        volumeMounts:
        - name: conf
          mountPath: /mnt/conf.d
        - name: config-map
          mountPath: /mnt/config-map
      - name: clone-mysql
        image: percona/percona-xtrabackup:8.0
        command:
        - bash
        - "-c"
        - |
          set -ex
          # Skip the clone if data already exists.
          [[ -d /var/lib/mysql/mysql ]] && exit 0
          # Skip the clone on primary (ordinal index 0).
          [[ `hostname` =~ -([0-9]+)$ ]] || exit 1
          ordinal=${BASH_REMATCH[1]}
          [[ $ordinal -eq 0 ]] && exit 0
          # Clone data from previous peer.
          ncat --recv-only mysql-$(($ordinal-1)).mysql 3307 | xbstream -x -C /var/lib/mysql
          # Prepare the backup.
          xtrabackup --prepare --target-dir=/var/lib/mysql
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
          subPath: mysql
        - name: conf
          mountPath: /etc/mysql/conf.d
      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ALLOW_EMPTY_PASSWORD
          value: "1"
        ports:
        - name: mysql
          containerPort: 3306
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
          subPath: mysql
        - name: conf
          mountPath: /etc/mysql/conf.d
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
        livenessProbe:
          exec:
            command: ["mysqladmin", "ping"]
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          exec:
            command: ["mysql", "-h", "127.0.0.1", "-e", "SELECT 1"]
          initialDelaySeconds: 5
          periodSeconds: 2
          timeoutSeconds: 1
      - name: xtrabackup
        image: percona/percona-xtrabackup:8.0
        ports:
        - name: xtrabackup
          containerPort: 3307
        command:
        - bash
        - "-c"
        - |
          set -ex
          cd /var/lib/mysql
          
          # Determine binlog position of cloned data, if any.
          if [[ -f xtrabackup_slave_info ]]; then
            # XtraBackup already generated a partial "CHANGE MASTER TO" query
            # because the --slave-info flag was set. (Need to remove the tailing semicolon!)
            mv xtrabackup_slave_info change_master_to.sql.in
            # Ignore xtrabackup_binlog_info in this case (it's useless).
            rm -f xtrabackup_binlog_info
          elif [[ -f xtrabackup_binlog_info ]]; then
            # We're cloning directly from primary. Parse binlog position.
            [[ `cat xtrabackup_binlog_info` =~ ^(.*?)[[:space:]]+(.*?)$ ]] || exit 1
            rm xtrabackup_binlog_info
            echo "CHANGE MASTER TO MASTER_LOG_FILE='${BASH_REMATCH[1]}',\
                  MASTER_LOG_POS=${BASH_REMATCH[2]}" > change_master_to.sql.in
          fi
          
          # Check if we need to complete a clone by starting replication.
          if [[ -f change_master_to.sql.in ]]; then
            echo "Waiting for mysqld to be ready (accepting connections)"
            until mysql -h 127.0.0.1 -e "SELECT 1"; do sleep 1; done
            
            echo "Initializing replication from clone position"
            # In case of container restart, attempt this at-most-once.
            mv change_master_to.sql.in change_master_to.sql.orig
            mysql -h 127.0.0.1 <<EOF
          $(<change_master_to.sql.orig),
            MASTER_HOST='mysql-0.mysql',
            MASTER_USER='root',
            MASTER_PASSWORD='',
            MASTER_CONNECT_RETRY=10;
          START SLAVE;
          EOF
          fi
          
          # Start a server to send backups to newly added slaves.
          exec ncat --listen --keep-open --send-only --max-conns=1 3307 -c \
            "xtrabackup --backup --slave-info --stream=xbstream --host=127.0.0.1 --user=root"
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
          subPath: mysql
        - name: conf
          mountPath: /etc/mysql/conf.d
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
      volumes:
      - name: conf
        emptyDir: {}
      - name: config-map
        configMap:
          name: mysql
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

### Redis Cluster StatefulSet
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        command:
        - redis-server
        args:
        - /conf/redis.conf
        - --cluster-enabled
        - "yes"
        - --cluster-config-file
        - /data/nodes.conf
        - --cluster-node-timeout
        - "5000"
        - --appendonly
        - "yes"
        - --protected-mode
        - "no"
        readinessProbe:
          exec:
            command:
            - sh
            - -c
            - "redis-cli ping"
          initialDelaySeconds: 15
          timeoutSeconds: 5
        livenessProbe:
          exec:
            command:
            - sh
            - -c
            - "redis-cli ping"
          initialDelaySeconds: 20
          periodSeconds: 3
        volumeMounts:
        - name: conf
          mountPath: /conf
          readOnly: false
        - name: data
          mountPath: /data
          readOnly: false
      volumes:
      - name: conf
        configMap:
          name: redis-cluster-config
          defaultMode: 0755
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 1Gi
```

## DaemonSets

### Log Collector DaemonSet
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
  labels:
    k8s-app: fluentd-logging
spec:
  selector:
    matchLabels:
      name: fluentd
  template:
    metadata:
      labels:
        name: fluentd
        k8s-app: fluentd-logging
    spec:
      tolerations:
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule
      - key: node-role.kubernetes.io/master
        operator: Exists
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1.16-debian-elasticsearch7-1
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch-logging"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        - name: FLUENT_ELASTICSEARCH_SCHEME
          value: "http"
        - name: FLUENT_UID
          value: "0"
        - name: FLUENT_CONF
          value: fluent.conf
        resources:
          limits:
            memory: 200Mi
          requests:
            cpu: 100m
            memory: 200Mi
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: fluentd-config
          mountPath: /fluentd/etc
      terminationGracePeriodSeconds: 30
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: fluentd-config
        configMap:
          name: fluentd-config
```

### Node Monitoring DaemonSet
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      hostPID: true
      hostIPC: true
      hostNetwork: true
      containers:
      - name: node-exporter
        image: quay.io/prometheus/node-exporter:latest
        ports:
        - containerPort: 9100
          hostPort: 9100
          name: scrape
        args:
        - --path.procfs=/host/proc
        - --path.sysfs=/host/sys
        - --path.rootfs=/host/root
        - --collector.filesystem.ignored-mount-points
        - ^/(dev|proc|sys|var/lib/docker/.+)($|/)
        - --collector.filesystem.ignored-fs-types
        - ^(autofs|binfmt_misc|cgroup|configfs|debugfs|devpts|devtmpfs|fusectl|hugetlbfs|mqueue|overlay|proc|procfs|pstore|rpc_pipefs|securityfs|sysfs|tracefs)$
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        - name: root
          mountPath: /host/root
          mountPropagation: HostToContainer
          readOnly: true
      tolerations:
      - effect: NoSchedule
        operator: Exists
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      - name: root
        hostPath:
          path: /
```

## Jobs and CronJobs

### Database Migration Job
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 3
  activeDeadlineSeconds: 600
  template:
    metadata:
      labels:
        app: db-migration
    spec:
      restartPolicy: Never
      containers:
      - name: migration
        image: migrate/migrate:latest
        command:
        - migrate
        args:
        - -path
        - /migrations
        - -database
        - $(DATABASE_URL)
        - up
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        volumeMounts:
        - name: migrations
          mountPath: /migrations
      volumes:
      - name: migrations
        configMap:
          name: db-migrations
```

### Backup CronJob
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      completions: 1
      parallelism: 1
      backoffLimit: 2
      activeDeadlineSeconds: 3600
      template:
        metadata:
          labels:
            app: database-backup
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: postgres:14
            command:
            - /bin/bash
            - -c
            - |
              BACKUP_FILE="/backup/backup-$(date +%Y%m%d-%H%M%S).sql"
              pg_dump $DATABASE_URL > $BACKUP_FILE
              # Upload to S3 (example)
              aws s3 cp $BACKUP_FILE s3://my-backups/database/
              # Cleanup old local backups
              find /backup -name "*.sql" -mtime +7 -delete
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret-key
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
```

### Log Cleanup CronJob
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: log-cleanup
spec:
  schedule: "0 0 * * 0"  # Weekly on Sunday
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          hostPID: true
          containers:
          - name: log-cleanup
            image: alpine:latest
            command:
            - /bin/sh
            - -c
            - |
              # Cleanup old log files
              find /var/log -name "*.log" -mtime +30 -delete
              find /var/log -name "*.gz" -mtime +90 -delete
              
              # Cleanup Docker logs
              docker system prune -f --filter "until=168h"
              
              # Cleanup old container logs
              find /var/lib/docker/containers -name "*.log" -mtime +30 -delete
            securityContext:
              privileged: true
            volumeMounts:
            - name: varlog
              mountPath: /var/log
            - name: dockersock
              mountPath: /var/run/docker.sock
          volumes:
          - name: varlog
            hostPath:
              path: /var/log
          - name: dockersock
            hostPath:
              path: /var/run/docker.sock
  successfulJobsHistoryLimit: 2
  failedJobsHistoryLimit: 1
```

## Workload Testing Script
```bash
#!/bin/bash
# Kubernetes workload testing script

echo "=== Kubernetes Workload Testing ==="

# Test deployment rollout status
echo "1. Testing deployment rollout..."
kubectl rollout status deployment/my-app --timeout=300s

# Test pod readiness
echo "2. Testing pod readiness..."
kubectl wait --for=condition=ready pod -l app=my-app --timeout=300s

# Test service connectivity
echo "3. Testing service connectivity..."
kubectl run test-pod --image=busybox --rm -it --restart=Never -- /bin/sh -c "
  wget -qO- http://my-app-service/health || echo 'Service not accessible'
"

# Test horizontal scaling
echo "4. Testing horizontal scaling..."
kubectl scale deployment my-app --replicas=5
kubectl wait --for=condition=ready pod -l app=my-app --timeout=300s
echo "Current replicas: $(kubectl get deployment my-app -o jsonpath='{.status.readyReplicas}')"

# Test rolling update
echo "5. Testing rolling update..."
kubectl set image deployment/my-app container=my-app:v2
kubectl rollout status deployment/my-app --timeout=300s

# Test job completion
echo "6. Testing job completion..."
kubectl create job test-job --image=busybox -- /bin/sh -c "echo 'Job completed successfully'"
kubectl wait --for=condition=complete job/test-job --timeout=300s

echo "=== Workload Testing Complete ==="
```

## Next Section

Continue to [Resource Management](06_Resource_Management.md) to learn about managing Kubernetes resources efficiently.
