# Project 04: Custom Kubernetes Operator

## Project Overview

This project focuses on building a custom Kubernetes operator that extends the Kubernetes API to manage application-specific resources. You'll learn the operator pattern, implement custom resource definitions (CRDs), and create controllers that reconcile desired state with actual state.


## Architecture Diagram

The Custom Kubernetes Operator follows a controller pattern that extends the Kubernetes API to manage application-specific resources. The architecture demonstrates how operators integrate with the Kubernetes control plane to provide automated lifecycle management.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Kubernetes Control Plane                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ API Server  │  │   etcd      │  │ Scheduler   │  │ Controller  │            │
│  │   (CRDs)    │  │(State Store)│  │             │  │  Manager    │            │
│  └──────┬──────┘  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────┼───────────────────────────────────────────────────────────────────────┘
          │ REST API Calls
          │
┌─────────▼───────────────────────────────────────────────────────────────────────┐
│                           Database Operator Components                          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         Custom Controllers                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │ Database    │  │   Backup    │  │ Monitoring  │  │    User     │    │   │
│  │  │ Controller  │  │ Controller  │  │ Controller  │  │ Controller  │    │   │
│  │  │             │  │             │  │             │  │             │    │   │
│  │  │ Reconciles  │  │ Manages     │  │ Sets up     │  │ Manages DB  │    │   │
│  │  │ Database    │  │ Backup      │  │ Prometheus  │  │ Users &     │    │   │
│  │  │ Resources   │  │ Schedules   │  │ & Grafana   │  │ Permissions │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      Custom Resource Definitions                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │  Database   │  │ DatabaseUser│  │DatabaseBackup│ │DatabaseMonitor│   │   │
│  │  │    CRD      │  │    CRD      │  │    CRD      │  │    CRD      │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────────────────────┘
                          │ Creates & Manages
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Managed Kubernetes Resources                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                            Database Instances                           │   │
│  │                                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │   │
│  │  │ PostgreSQL  │  │   MySQL     │  │  MongoDB    │                    │   │
│  │  │             │  │             │  │             │                    │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                    │   │
│  │  │ │StatefulS│ │  │ │StatefulS│ │  │ │StatefulS│ │                    │   │
│  │  │ │   et    │ │  │ │   et    │ │  │ │   et    │ │                    │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                    │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                    │   │
│  │  │ │ Service │ │  │ │ Service │ │  │ │ Service │ │                    │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                    │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                    │   │
│  │  │ │ PVC     │ │  │ │ PVC     │ │  │ │ PVC     │ │                    │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                    │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                    │   │
│  │  │ │ConfigMap│ │  │ │ConfigMap│ │  │ │ConfigMap│ │                    │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                    │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                    │   │
│  │  │ │ Secret  │ │  │ │ Secret  │ │  │ │ Secret  │ │                    │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Supporting Services                               │   │
│  │                                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │   Backup    │  │ Monitoring  │  │  Network    │  │   Storage   │    │   │
│  │  │   CronJob   │  │  Services   │  │  Policies   │  │   Classes   │    │   │
│  │  │             │  │             │  │             │  │             │    │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │    │   │
│  │  │ │Velero   │ │  │ │Prometheus│ │  │ │ Calico  │ │  │ │   CSI   │ │    │   │
│  │  │ │Schedule │ │  │ │ServiceMon│ │  │ │ Policies│ │  │ │ Drivers │ │    │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │    │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │             │  │             │    │   │
│  │  │ │S3 Bucket│ │  │ │ Grafana │ │  │             │  │             │    │   │
│  │  │ │Storage  │ │  │ │Dashboard│ │  │             │  │             │    │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │             │  │             │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘

Control Flow:
1. User creates Database custom resource → API Server validates & stores in etcd
2. Database Controller watches for Database resources → Reconciliation loop triggered
3. Controller creates StatefulSet, Service, ConfigMap, Secret, PVC
4. Backup Controller creates CronJob for automated backups
5. Monitoring Controller sets up Prometheus ServiceMonitor and Grafana dashboard
6. User Controller manages database users and permissions
7. Status updates flow back through controllers to the Database resource status
```

## Learning Objectives

By completing this project, you will:
- Understand the Kubernetes operator pattern and its benefits
- Design and implement Custom Resource Definitions (CRDs)
- Build a controller that implements reconciliation logic
- Handle complex resource lifecycle management
- Implement proper error handling and status reporting
- Package and distribute a Kubernetes operator

## Project Scope

### Custom Resource: Database Application

We'll create an operator that manages a simplified database application with the following features:
- Automatic database deployment and configuration
- Backup scheduling and management
- User and permission management
- Monitoring and alerting integration
- Scaling and resource management

## Phase 1: Design and Planning (1 week)

### 1.1 Operator Design

**Custom Resource Definition Design:**
```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              engine:
                type: string
                enum: ["postgresql", "mysql", "mongodb"]
              version:
                type: string
              replicas:
                type: integer
                minimum: 1
                maximum: 10
              storage:
                type: object
                properties:
                  size:
                    type: string
                  storageClass:
                    type: string
              backup:
                type: object
                properties:
                  enabled:
                    type: boolean
                  schedule:
                    type: string
                  retention:
                    type: string
              monitoring:
                type: object
                properties:
                  enabled:
                    type: boolean
                  alerting:
                    type: boolean
          status:
            type: object
            properties:
              phase:
                type: string
                enum: ["Pending", "Running", "Failed", "Scaling"]
              conditions:
                type: array
                items:
                  type: object
                  properties:
                    type:
                      type: string
                    status:
                      type: string
                    reason:
                      type: string
                    message:
                      type: string
              endpoints:
                type: object
                properties:
                  primary:
                    type: string
                  readonly:
                    type: string
  scope: Namespaced
  names:
    plural: databases
    singular: database
    kind: Database
```

### 1.2 Architecture Planning

**Controller Components:**
1. **Database Controller**: Main reconciliation logic
2. **Backup Controller**: Manages backup schedules and retention
3. **Monitoring Controller**: Sets up monitoring and alerting
4. **User Controller**: Manages database users and permissions

**External Dependencies:**
- Storage CSI drivers
- Backup solution (Velero, custom scripts)
- Monitoring stack (Prometheus, Grafana)
- Secret management

### 1.3 Development Environment Setup

**Tools and Frameworks:**
```bash
# Install Operator SDK
curl -LO https://github.com/operator-framework/operator-sdk/releases/download/v1.34.1/operator-sdk_linux_amd64
chmod +x operator-sdk_linux_amd64
sudo mv operator-sdk_linux_amd64 /usr/local/bin/operator-sdk

# Verify installation
operator-sdk version

# Install Kubebuilder (alternative framework)
curl -L -o kubebuilder https://go.kubebuilder.io/dl/latest/$(go env GOOS)/$(go env GOARCH)
chmod +x kubebuilder
sudo mv kubebuilder /usr/local/bin/

# Set up development cluster
kind create cluster --name operator-dev
kubectl cluster-info --context kind-operator-dev
```

## Phase 2: Custom Resource Definition Implementation (1 week)

### 2.1 CRD Creation and Validation

**Enhanced CRD with OpenAPI Schema:**
```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
  annotations:
    controller-gen.kubebuilder.io/version: v0.13.0
spec:
  group: example.com
  names:
    kind: Database
    listKind: DatabaseList
    plural: databases
    singular: database
  scope: Namespaced
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        description: Database is the Schema for the databases API
        type: object
        properties:
          apiVersion:
            description: 'APIVersion defines the versioned schema'
            type: string
          kind:
            description: 'Kind is a string value representing the REST resource'
            type: string
          metadata:
            type: object
          spec:
            description: DatabaseSpec defines the desired state of Database
            type: object
            required:
            - engine
            - version
            properties:
              engine:
                description: Database engine type
                type: string
                enum:
                - postgresql
                - mysql
                - mongodb
              version:
                description: Database version
                type: string
                pattern: '^[0-9]+\.[0-9]+(\.[0-9]+)?$'
              replicas:
                description: Number of database replicas
                type: integer
                minimum: 1
                maximum: 10
                default: 1
              storage:
                description: Storage configuration
                type: object
                required:
                - size
                properties:
                  size:
                    description: Storage size
                    type: string
                    pattern: '^[0-9]+[KMGT]i?$'
                  storageClass:
                    description: Storage class name
                    type: string
              backup:
                description: Backup configuration
                type: object
                properties:
                  enabled:
                    type: boolean
                    default: false
                  schedule:
                    description: Cron schedule for backups
                    type: string
                    pattern: '^(@(annually|yearly|monthly|weekly|daily|hourly|reboot))|(@every (\d+(ns|us|µs|ms|s|m|h))+)|((((\d+,)+\d+|(\d+(\/|-)\d+)|\d+|\*) ?){5,7})$'
                  retention:
                    description: Backup retention period
                    type: string
                  destination:
                    description: Backup destination
                    type: object
                    properties:
                      s3:
                        type: object
                        properties:
                          bucket:
                            type: string
                          region:
                            type: string
          status:
            description: DatabaseStatus defines the observed state of Database
            type: object
            properties:
              phase:
                description: Current phase of the database
                type: string
                enum:
                - Pending
                - Creating
                - Running
                - Scaling
                - Updating
                - Failed
                - Deleting
              conditions:
                description: Conditions represent the latest available observations
                type: array
                items:
                  description: DatabaseCondition describes the state of a database
                  type: object
                  required:
                  - status
                  - type
                  properties:
                    lastTransitionTime:
                      description: Last time the condition transitioned
                      type: string
                      format: date-time
                    message:
                      description: Human readable message
                      type: string
                    reason:
                      description: Unique, one-word, CamelCase reason
                      type: string
                    status:
                      description: Status of the condition
                      type: string
                      enum:
                      - "True"
                      - "False"
                      - Unknown
                    type:
                      description: Type of database condition
                      type: string
              endpoints:
                description: Database connection endpoints
                type: object
                properties:
                  primary:
                    description: Primary database endpoint
                    type: string
                  readonly:
                    description: Read-only replica endpoints
                    type: array
                    items:
                      type: string
              observedGeneration:
                description: Most recent generation observed by the controller
                type: integer
    subresources:
      status: {}
      scale:
        specReplicasPath: .spec.replicas
        statusReplicasPath: .status.replicas
    additionalPrinterColumns:
    - name: Engine
      type: string
      description: Database engine
      jsonPath: .spec.engine
    - name: Version
      type: string
      description: Database version
      jsonPath: .spec.version
    - name: Replicas
      type: integer
      description: Number of replicas
      jsonPath: .spec.replicas
    - name: Phase
      type: string
      description: Current phase
      jsonPath: .status.phase
    - name: Age
      type: date
      jsonPath: .metadata.creationTimestamp
```

### 2.2 Go Types and Code Generation

**Database Type Definition (Go):**
```go
package v1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DatabaseSpec defines the desired state of Database
type DatabaseSpec struct {
    // Engine specifies the database engine type
    // +kubebuilder:validation:Enum=postgresql;mysql;mongodb
    Engine string `json:"engine"`
    
    // Version specifies the database version
    // +kubebuilder:validation:Pattern=`^[0-9]+\.[0-9]+(\.[0-9]+)?$`
    Version string `json:"version"`
    
    // Replicas specifies the number of database replicas
    // +kubebuilder:validation:Minimum=1
    // +kubebuilder:validation:Maximum=10
    // +kubebuilder:default=1
    Replicas int32 `json:"replicas,omitempty"`
    
    // Storage defines storage configuration
    Storage StorageSpec `json:"storage"`
    
    // Backup defines backup configuration
    // +optional
    Backup *BackupSpec `json:"backup,omitempty"`
    
    // Monitoring defines monitoring configuration
    // +optional
    Monitoring *MonitoringSpec `json:"monitoring,omitempty"`
}

type StorageSpec struct {
    // Size specifies the storage size
    // +kubebuilder:validation:Pattern=`^[0-9]+[KMGT]i?$`
    Size string `json:"size"`
    
    // StorageClass specifies the storage class
    // +optional
    StorageClass string `json:"storageClass,omitempty"`
}

type BackupSpec struct {
    // Enabled specifies if backups are enabled
    // +kubebuilder:default=false
    Enabled bool `json:"enabled,omitempty"`
    
    // Schedule specifies the backup schedule in cron format
    // +optional
    Schedule string `json:"schedule,omitempty"`
    
    // Retention specifies backup retention period
    // +optional
    Retention string `json:"retention,omitempty"`
    
    // Destination specifies backup destination
    // +optional
    Destination *BackupDestination `json:"destination,omitempty"`
}

type BackupDestination struct {
    // S3 specifies S3 backup configuration
    // +optional
    S3 *S3BackupConfig `json:"s3,omitempty"`
}

type S3BackupConfig struct {
    Bucket string `json:"bucket"`
    Region string `json:"region"`
}

type MonitoringSpec struct {
    // Enabled specifies if monitoring is enabled
    // +kubebuilder:default=false
    Enabled bool `json:"enabled,omitempty"`
    
    // Alerting specifies if alerting is enabled
    // +kubebuilder:default=false
    Alerting bool `json:"alerting,omitempty"`
}

// DatabaseStatus defines the observed state of Database
type DatabaseStatus struct {
    // Phase represents the current phase of the database
    // +kubebuilder:validation:Enum=Pending;Creating;Running;Scaling;Updating;Failed;Deleting
    Phase string `json:"phase,omitempty"`
    
    // Conditions represent the latest available observations
    // +optional
    Conditions []DatabaseCondition `json:"conditions,omitempty"`
    
    // Endpoints contain database connection information
    // +optional
    Endpoints *DatabaseEndpoints `json:"endpoints,omitempty"`
    
    // ObservedGeneration reflects the generation most recently observed
    // +optional
    ObservedGeneration int64 `json:"observedGeneration,omitempty"`
    
    // Replicas is the current number of replicas
    // +optional
    Replicas int32 `json:"replicas,omitempty"`
}

type DatabaseCondition struct {
    // Type of database condition
    Type string `json:"type"`
    
    // Status of the condition
    // +kubebuilder:validation:Enum=True;False;Unknown
    Status metav1.ConditionStatus `json:"status"`
    
    // Last time the condition transitioned
    // +optional
    LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`
    
    // Unique, one-word, CamelCase reason for the condition's last transition
    // +optional
    Reason string `json:"reason,omitempty"`
    
    // Human-readable message indicating details about last transition
    // +optional
    Message string `json:"message,omitempty"`
}

type DatabaseEndpoints struct {
    // Primary database endpoint
    // +optional
    Primary string `json:"primary,omitempty"`
    
    // Read-only replica endpoints
    // +optional
    ReadOnly []string `json:"readonly,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas
//+kubebuilder:printcolumn:name="Engine",type="string",JSONPath=".spec.engine"
//+kubebuilder:printcolumn:name="Version",type="string",JSONPath=".spec.version"
//+kubebuilder:printcolumn:name="Replicas",type="integer",JSONPath=".spec.replicas"
//+kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
//+kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// Database is the Schema for the databases API
type Database struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   DatabaseSpec   `json:"spec,omitempty"`
    Status DatabaseStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// DatabaseList contains a list of Database
type DatabaseList struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ListMeta `json:"metadata,omitempty"`
    Items           []Database `json:"items"`
}

func init() {
    SchemeBuilder.Register(&Database{}, &DatabaseList{})
}
```

## Phase 3: Controller Implementation (2 weeks)

### 3.1 Main Controller Logic

**Database Controller Implementation:**
```go
package controllers

import (
    "context"
    "fmt"
    "time"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/api/errors"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
    "sigs.k8s.io/controller-runtime/pkg/log"

    databasev1 "github.com/example/database-operator/api/v1"
)

const (
    DatabaseFinalizerName = "database.example.com/finalizer"
)

// DatabaseReconciler reconciles a Database object
type DatabaseReconciler struct {
    client.Client
    Scheme *runtime.Scheme
}

//+kubebuilder:rbac:groups=example.com,resources=databases,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=example.com,resources=databases/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=example.com,resources=databases/finalizers,verbs=update
//+kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop
func (r *DatabaseReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    // Fetch the Database instance
    database := &databasev1.Database{}
    err := r.Get(ctx, req.NamespacedName, database)
    if err != nil {
        if errors.IsNotFound(err) {
            log.Info("Database resource not found. Ignoring since object must be deleted")
            return ctrl.Result{}, nil
        }
        log.Error(err, "Failed to get Database")
        return ctrl.Result{}, err
    }

    // Handle deletion
    if database.DeletionTimestamp != nil {
        return r.handleDeletion(ctx, database)
    }

    // Add finalizer if not present
    if !controllerutil.ContainsFinalizer(database, DatabaseFinalizerName) {
        controllerutil.AddFinalizer(database, DatabaseFinalizerName)
        return ctrl.Result{}, r.Update(ctx, database)
    }

    // Update status
    originalStatus := database.Status.DeepCopy()
    
    // Reconcile database components
    result, err := r.reconcileDatabase(ctx, database)
    
    // Update status if changed
    if !r.statusEqual(originalStatus, &database.Status) {
        if updateErr := r.Status().Update(ctx, database); updateErr != nil {
            log.Error(updateErr, "Failed to update Database status")
            return ctrl.Result{}, updateErr
        }
    }

    return result, err
}

func (r *DatabaseReconciler) reconcileDatabase(ctx context.Context, database *databasev1.Database) (ctrl.Result, error) {
    log := log.FromContext(ctx)
    
    // Set initial status
    if database.Status.Phase == "" {
        database.Status.Phase = "Pending"
        r.updateCondition(database, "Initializing", metav1.ConditionTrue, "ReconciliationStarted", "Starting database reconciliation")
    }

    // Reconcile ConfigMap
    if err := r.reconcileConfigMap(ctx, database); err != nil {
        log.Error(err, "Failed to reconcile ConfigMap")
        database.Status.Phase = "Failed"
        r.updateCondition(database, "Ready", metav1.ConditionFalse, "ConfigMapError", err.Error())
        return ctrl.Result{RequeueAfter: time.Minute * 5}, err
    }

    // Reconcile Secret
    if err := r.reconcileSecret(ctx, database); err != nil {
        log.Error(err, "Failed to reconcile Secret")
        database.Status.Phase = "Failed"
        r.updateCondition(database, "Ready", metav1.ConditionFalse, "SecretError", err.Error())
        return ctrl.Result{RequeueAfter: time.Minute * 5}, err
    }

    // Reconcile StatefulSet
    if err := r.reconcileStatefulSet(ctx, database); err != nil {
        log.Error(err, "Failed to reconcile StatefulSet")
        database.Status.Phase = "Failed"
        r.updateCondition(database, "Ready", metav1.ConditionFalse, "StatefulSetError", err.Error())
        return ctrl.Result{RequeueAfter: time.Minute * 5}, err
    }

    // Reconcile Service
    if err := r.reconcileService(ctx, database); err != nil {
        log.Error(err, "Failed to reconcile Service")
        database.Status.Phase = "Failed"
        r.updateCondition(database, "Ready", metav1.ConditionFalse, "ServiceError", err.Error())
        return ctrl.Result{RequeueAfter: time.Minute * 5}, err
    }

    // Check if database is ready
    if r.isDatabaseReady(ctx, database) {
        database.Status.Phase = "Running"
        r.updateCondition(database, "Ready", metav1.ConditionTrue, "DatabaseReady", "Database is ready and serving traffic")
        
        // Update endpoints
        r.updateEndpoints(ctx, database)
    } else {
        database.Status.Phase = "Creating"
        r.updateCondition(database, "Ready", metav1.ConditionFalse, "DatabaseNotReady", "Database is not ready yet")
        return ctrl.Result{RequeueAfter: time.Second * 30}, nil
    }

    return ctrl.Result{RequeueAfter: time.Minute * 5}, nil
}

func (r *DatabaseReconciler) reconcileConfigMap(ctx context.Context, database *databasev1.Database) error {
    configMap := &corev1.ConfigMap{
        ObjectMeta: metav1.ObjectMeta{
            Name:      database.Name + "-config",
            Namespace: database.Namespace,
        },
    }

    op, err := controllerutil.CreateOrUpdate(ctx, r.Client, configMap, func() error {
        configMap.Data = r.generateConfigMapData(database)
        return controllerutil.SetControllerReference(database, configMap, r.Scheme)
    })

    if err != nil {
        return err
    }

    if op != controllerutil.OperationResultNone {
        log.FromContext(ctx).Info("ConfigMap reconciled", "operation", op)
    }

    return nil
}

func (r *DatabaseReconciler) generateConfigMapData(database *databasev1.Database) map[string]string {
    data := make(map[string]string)
    
    switch database.Spec.Engine {
    case "postgresql":
        data["postgresql.conf"] = fmt.Sprintf(`
# PostgreSQL configuration
listen_addresses = '*'
port = 5432
max_connections = 100
shared_buffers = 128MB
effective_cache_size = 4GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
`)
    case "mysql":
        data["my.cnf"] = fmt.Sprintf(`
[mysqld]
bind-address = 0.0.0.0
port = 3306
max_connections = 100
innodb_buffer_pool_size = 128M
innodb_log_file_size = 64M
innodb_flush_log_at_trx_commit = 1
`)
    case "mongodb":
        data["mongod.conf"] = fmt.Sprintf(`
net:
  port: 27017
  bindIp: 0.0.0.0
storage:
  dbPath: /data/db
  wiredTiger:
    engineConfig:
      cacheSizeGB: 0.25
`)
    }
    
    return data
}

// SetupWithManager sets up the controller with the Manager
func (r *DatabaseReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&databasev1.Database{}).
        Owns(&appsv1.StatefulSet{}).
        Owns(&corev1.Service{}).
        Owns(&corev1.ConfigMap{}).
        Owns(&corev1.Secret{}).
        Complete(r)
}
```

### 3.2 StatefulSet Management

**StatefulSet Reconciliation Logic:**
```go
func (r *DatabaseReconciler) reconcileStatefulSet(ctx context.Context, database *databasev1.Database) error {
    statefulSet := &appsv1.StatefulSet{
        ObjectMeta: metav1.ObjectMeta{
            Name:      database.Name,
            Namespace: database.Namespace,
        },
    }

    op, err := controllerutil.CreateOrUpdate(ctx, r.Client, statefulSet, func() error {
        statefulSet.Spec = r.generateStatefulSetSpec(database)
        return controllerutil.SetControllerReference(database, statefulSet, r.Scheme)
    })

    if err != nil {
        return err
    }

    if op != controllerutil.OperationResultNone {
        log.FromContext(ctx).Info("StatefulSet reconciled", "operation", op)
    }

    return nil
}

func (r *DatabaseReconciler) generateStatefulSetSpec(database *databasev1.Database) appsv1.StatefulSetSpec {
    replicas := database.Spec.Replicas
    
    labels := map[string]string{
        "app":     database.Name,
        "engine":  database.Spec.Engine,
        "version": database.Spec.Version,
    }

    var containers []corev1.Container
    var ports []corev1.ContainerPort
    
    switch database.Spec.Engine {
    case "postgresql":
        ports = []corev1.ContainerPort{{ContainerPort: 5432, Name: "postgres"}}
        containers = []corev1.Container{{
            Name:  "postgres",
            Image: fmt.Sprintf("postgres:%s", database.Spec.Version),
            Ports: ports,
            Env: []corev1.EnvVar{
                {Name: "POSTGRES_DB", Value: database.Name},
                {Name: "POSTGRES_USER", ValueFrom: &corev1.EnvVarSource{
                    SecretKeyRef: &corev1.SecretKeySelector{
                        LocalObjectReference: corev1.LocalObjectReference{
                            Name: database.Name + "-secret",
                        },
                        Key: "username",
                    },
                }},
                {Name: "POSTGRES_PASSWORD", ValueFrom: &corev1.EnvVarSource{
                    SecretKeyRef: &corev1.SecretKeySelector{
                        LocalObjectReference: corev1.LocalObjectReference{
                            Name: database.Name + "-secret",
                        },
                        Key: "password",
                    },
                }},
                {Name: "PGDATA", Value: "/var/lib/postgresql/data/pgdata"},
            },
            VolumeMounts: []corev1.VolumeMount{
                {Name: "data", MountPath: "/var/lib/postgresql/data"},
                {Name: "config", MountPath: "/etc/postgresql"},
            },
            ReadinessProbe: &corev1.Probe{
                ProbeHandler: corev1.ProbeHandler{
                    Exec: &corev1.ExecAction{
                        Command: []string{"pg_isready", "-U", "postgres", "-d", database.Name},
                    },
                },
                InitialDelaySeconds: 10,
                PeriodSeconds:       5,
                TimeoutSeconds:      3,
            },
            LivenessProbe: &corev1.Probe{
                ProbeHandler: corev1.ProbeHandler{
                    Exec: &corev1.ExecAction{
                        Command: []string{"pg_isready", "-U", "postgres", "-d", database.Name},
                    },
                },
                InitialDelaySeconds: 30,
                PeriodSeconds:       10,
                TimeoutSeconds:      5,
            },
        }}
    // Add cases for MySQL and MongoDB...
    }

    return appsv1.StatefulSetSpec{
        Replicas:    &replicas,
        ServiceName: database.Name + "-headless",
        Selector: &metav1.LabelSelector{
            MatchLabels: labels,
        },
        Template: corev1.PodTemplateSpec{
            ObjectMeta: metav1.ObjectMeta{
                Labels: labels,
            },
            Spec: corev1.PodSpec{
                Containers: containers,
                Volumes: []corev1.Volume{
                    {
                        Name: "config",
                        VolumeSource: corev1.VolumeSource{
                            ConfigMap: &corev1.ConfigMapVolumeSource{
                                LocalObjectReference: corev1.LocalObjectReference{
                                    Name: database.Name + "-config",
                                },
                            },
                        },
                    },
                },
            },
        },
        VolumeClaimTemplates: []corev1.PersistentVolumeClaim{
            {
                ObjectMeta: metav1.ObjectMeta{
                    Name: "data",
                },
                Spec: corev1.PersistentVolumeClaimSpec{
                    AccessModes: []corev1.PersistentVolumeAccessMode{
                        corev1.ReadWriteOnce,
                    },
                    Resources: corev1.ResourceRequirements{
                        Requests: corev1.ResourceList{
                            corev1.ResourceStorage: resource.MustParse(database.Spec.Storage.Size),
                        },
                    },
                    StorageClassName: &database.Spec.Storage.StorageClass,
                },
            },
        },
    }
}
```

## Phase 4: Testing and Validation (1 week)

### 4.1 Unit Testing

**Controller Unit Tests:**
```go
package controllers

import (
    "context"
    "time"

    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"
    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"

    databasev1 "github.com/example/database-operator/api/v1"
)

var _ = Describe("Database Controller", func() {
    Context("When creating a Database", func() {
        It("Should create associated resources", func() {
            ctx := context.Background()
            
            database := &databasev1.Database{
                ObjectMeta: metav1.ObjectMeta{
                    Name:      "test-postgres",
                    Namespace: "default",
                },
                Spec: databasev1.DatabaseSpec{
                    Engine:   "postgresql",
                    Version:  "13.8",
                    Replicas: 1,
                    Storage: databasev1.StorageSpec{
                        Size: "1Gi",
                    },
                },
            }
            
            Expect(k8sClient.Create(ctx, database)).Should(Succeed())
            
            // Verify StatefulSet creation
            statefulSet := &appsv1.StatefulSet{}
            Eventually(func() error {
                return k8sClient.Get(ctx, types.NamespacedName{
                    Name:      "test-postgres",
                    Namespace: "default",
                }, statefulSet)
            }, time.Second*10, time.Millisecond*250).Should(Succeed())
            
            Expect(statefulSet.Spec.Replicas).Should(Equal(int32Ptr(1)))
            Expect(statefulSet.Spec.Template.Spec.Containers[0].Image).Should(Equal("postgres:13.8"))
            
            // Verify ConfigMap creation
            configMap := &corev1.ConfigMap{}
            Eventually(func() error {
                return k8sClient.Get(ctx, types.NamespacedName{
                    Name:      "test-postgres-config",
                    Namespace: "default",
                }, configMap)
            }, time.Second*10, time.Millisecond*250).Should(Succeed())
            
            Expect(configMap.Data).Should(HaveKey("postgresql.conf"))
        })
        
        It("Should update status correctly", func() {
            ctx := context.Background()
            
            database := &databasev1.Database{}
            Eventually(func() error {
                return k8sClient.Get(ctx, types.NamespacedName{
                    Name:      "test-postgres",
                    Namespace: "default",
                }, database)
            }, time.Second*10, time.Millisecond*250).Should(Succeed())
            
            Eventually(func() string {
                k8sClient.Get(ctx, types.NamespacedName{
                    Name:      "test-postgres",
                    Namespace: "default",
                }, database)
                return database.Status.Phase
            }, time.Second*30, time.Second).Should(Equal("Creating"))
        })
    })
    
    Context("When scaling a Database", func() {
        It("Should update StatefulSet replicas", func() {
            ctx := context.Background()
            
            database := &databasev1.Database{}
            Expect(k8sClient.Get(ctx, types.NamespacedName{
                Name:      "test-postgres",
                Namespace: "default",
            }, database)).Should(Succeed())
            
            // Update replicas
            database.Spec.Replicas = 3
            Expect(k8sClient.Update(ctx, database)).Should(Succeed())
            
            // Verify StatefulSet is updated
            statefulSet := &appsv1.StatefulSet{}
            Eventually(func() int32 {
                k8sClient.Get(ctx, types.NamespacedName{
                    Name:      "test-postgres",
                    Namespace: "default",
                }, statefulSet)
                return *statefulSet.Spec.Replicas
            }, time.Second*10, time.Millisecond*250).Should(Equal(int32(3)))
        })
    })
})

func int32Ptr(i int32) *int32 {
    return &i
}
```

### 4.2 Integration Testing

**End-to-End Test Suite:**
```go
package e2e

import (
    "context"
    "fmt"
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"
    "k8s.io/apimachinery/pkg/util/wait"
    "sigs.k8s.io/controller-runtime/pkg/client"

    databasev1 "github.com/example/database-operator/api/v1"
)

func TestDatabaseLifecycle(t *testing.T) {
    ctx := context.Background()
    
    // Create database
    database := &databasev1.Database{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "e2e-test-db",
            Namespace: "default",
        },
        Spec: databasev1.DatabaseSpec{
            Engine:   "postgresql",
            Version:  "13.8",
            Replicas: 2,
            Storage: databasev1.StorageSpec{
                Size: "2Gi",
            },
            Backup: &databasev1.BackupSpec{
                Enabled:  true,
                Schedule: "0 2 * * *",
                Retention: "7d",
            },
        },
    }
    
    err := k8sClient.Create(ctx, database)
    require.NoError(t, err)
    
    // Wait for database to be ready
    err = wait.PollImmediate(time.Second*5, time.Minute*5, func() (bool, error) {
        if err := k8sClient.Get(ctx, types.NamespacedName{
            Name:      "e2e-test-db",
            Namespace: "default",
        }, database); err != nil {
            return false, err
        }
        return database.Status.Phase == "Running", nil
    })
    require.NoError(t, err)
    
    // Verify StatefulSet
    statefulSet := &appsv1.StatefulSet{}
    err = k8sClient.Get(ctx, types.NamespacedName{
        Name:      "e2e-test-db",
        Namespace: "default",
    }, statefulSet)
    require.NoError(t, err)
    assert.Equal(t, int32(2), *statefulSet.Spec.Replicas)
    
    // Test scaling
    database.Spec.Replicas = 3
    err = k8sClient.Update(ctx, database)
    require.NoError(t, err)
    
    // Wait for scaling to complete
    err = wait.PollImmediate(time.Second*5, time.Minute*2, func() (bool, error) {
        if err := k8sClient.Get(ctx, types.NamespacedName{
            Name:      "e2e-test-db",
            Namespace: "default",
        }, statefulSet); err != nil {
            return false, err
        }
        return *statefulSet.Spec.Replicas == 3 && statefulSet.Status.ReadyReplicas == 3, nil
    })
    require.NoError(t, err)
    
    // Test backup configuration
    cronJob := &batchv1.CronJob{}
    err = k8sClient.Get(ctx, types.NamespacedName{
        Name:      "e2e-test-db-backup",
        Namespace: "default",
    }, cronJob)
    require.NoError(t, err)
    assert.Equal(t, "0 2 * * *", cronJob.Spec.Schedule)
    
    // Clean up
    err = k8sClient.Delete(ctx, database)
    require.NoError(t, err)
}
```

## Phase 5: Packaging and Distribution (1 week)

### 5.1 Helm Chart Creation

**Helm Chart Structure:**
```
charts/database-operator/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── rbac.yaml
│   ├── crd.yaml
│   └── service.yaml
└── crds/
    └── databases.example.com.yaml
```

**Chart.yaml:**
```yaml
apiVersion: v2
name: database-operator
description: A Kubernetes operator for managing database applications
type: application
version: 0.1.0
appVersion: "0.1.0"
keywords:
  - database
  - operator
  - postgresql
  - mysql
  - mongodb
home: https://github.com/example/database-operator
sources:
  - https://github.com/example/database-operator
maintainers:
  - name: Your Name
    email: your.email@example.com
```

**values.yaml:**
```yaml
# Default values for database-operator

replicaCount: 1

image:
  repository: database-operator
  pullPolicy: IfNotPresent
  tag: "v0.1.0"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  runAsNonRoot: true

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL

service:
  type: ClusterIP
  port: 8443

resources:
  limits:
    cpu: 500m
    memory: 128Mi
  requests:
    cpu: 10m
    memory: 64Mi

nodeSelector: {}

tolerations: []

affinity: {}

# Webhook configuration
webhook:
  enabled: true
  certManager:
    enabled: true

# Metrics configuration
metrics:
  enabled: true
  serviceMonitor:
    enabled: false
```

### 5.2 OLM (Operator Lifecycle Manager) Integration

**ClusterServiceVersion (CSV):**
```yaml
apiVersion: operators.coreos.com/v1alpha1
kind: ClusterServiceVersion
metadata:
  name: database-operator.v0.1.0
  namespace: operators
  annotations:
    alm-examples: |-
      [
        {
          "apiVersion": "example.com/v1",
          "kind": "Database",
          "metadata": {
            "name": "postgres-sample"
          },
          "spec": {
            "engine": "postgresql",
            "version": "13.8",
            "replicas": 2,
            "storage": {
              "size": "10Gi"
            }
          }
        }
      ]
    capabilities: Deep Insights
    categories: Database
    certified: "false"
    createdAt: "2025-06-21T00:00:00Z"
    description: Kubernetes operator for managing database applications
    containerImage: database-operator:v0.1.0
    support: Community
    repository: https://github.com/example/database-operator
spec:
  displayName: Database Operator
  description: |
    The Database Operator provides a declarative way to manage database 
    applications on Kubernetes. It supports PostgreSQL, MySQL, and MongoDB 
    with automated backup, monitoring, and scaling capabilities.

    ## Features
    * Automated database deployment and configuration
    * Built-in backup and restore capabilities
    * Horizontal scaling support
    * Monitoring and alerting integration
    * Multi-engine support (PostgreSQL, MySQL, MongoDB)

  keywords:
    - database
    - postgresql
    - mysql
    - mongodb
    - backup
    - monitoring
  version: 0.1.0
  maturity: alpha
  minKubeVersion: 1.20.0
  maintainers:
    - name: Your Name
      email: your.email@example.com
  provider:
    name: Example Corp
  labels:
    alm-owner-enterprise-app: database-operator
    alm-status-descriptors: database-operator.v0.1.0
  selector:
    matchLabels:
      alm-owner-enterprise-app: database-operator
  links:
    - name: Documentation
      url: https://github.com/example/database-operator/blob/main/README.md
    - name: Source Code
      url: https://github.com/example/database-operator
  icon:
    - base64data: >-
        iVBORw0KGgoAAAANSUhEUgAAAF8AAAAfCAYAAAC+K3J7AAAACXBIWXMAAAsSAAALEgHS3X78...
      mediatype: image/png
  customresourcedefinitions:
    owned:
      - name: databases.example.com
        version: v1
        kind: Database
        displayName: Database
        description: A database instance managed by the operator
        resources:
          - version: v1
            kind: StatefulSet
          - version: v1
            kind: Service
          - version: v1
            kind: ConfigMap
          - version: v1
            kind: Secret
        specDescriptors:
          - description: Database engine type
            displayName: Engine
            path: engine
            x-descriptors:
              - 'urn:alm:descriptor:com.tectonic.ui:select:postgresql'
              - 'urn:alm:descriptor:com.tectonic.ui:select:mysql'
              - 'urn:alm:descriptor:com.tectonic.ui:select:mongodb'
          - description: Database version
            displayName: Version
            path: version
            x-descriptors:
              - 'urn:alm:descriptor:com.tectonic.ui:text'
          - description: Number of replicas
            displayName: Replicas
            path: replicas
            x-descriptors:
              - 'urn:alm:descriptor:com.tectonic.ui:podCount'
        statusDescriptors:
          - description: Current phase of the database
            displayName: Phase
            path: phase
            x-descriptors:
              - 'urn:alm:descriptor:io.kubernetes.phase'
          - description: Database endpoints
            displayName: Endpoints
            path: endpoints
            x-descriptors:
              - 'urn:alm:descriptor:org.w3:link'
  install:
    strategy: deployment
    spec:
      permissions:
        - serviceAccountName: database-operator
          rules:
            - apiGroups: [""]
              resources: ["configmaps", "secrets", "services", "persistentvolumeclaims"]
              verbs: ["*"]
            - apiGroups: ["apps"]
              resources: ["statefulsets"]
              verbs: ["*"]
            - apiGroups: ["example.com"]
              resources: ["databases"]
              verbs: ["*"]
      deployments:
        - name: database-operator
          spec:
            replicas: 1
            selector:
              matchLabels:
                name: database-operator
            template:
              metadata:
                labels:
                  name: database-operator
              spec:
                serviceAccountName: database-operator
                containers:
                  - name: manager
                    image: database-operator:v0.1.0
                    command:
                      - /manager
                    args:
                      - --leader-elect
                    env:
                      - name: WATCH_NAMESPACE
                        valueFrom:
                          fieldRef:
                            fieldPath: metadata.annotations['olm.targetNamespaces']
                    resources:
                      limits:
                        cpu: 500m
                        memory: 128Mi
                      requests:
                        cpu: 10m
                        memory: 64Mi
```


### Key Architectural Components

**Control Plane Integration:**
- **Custom Resource Definitions (CRDs)**: Extend Kubernetes API with database-specific resources
- **Controller Pattern**: Implement reconciliation loops that maintain desired state
- **API Server Integration**: Leverage Kubernetes validation, authorization, and admission control

**Operator Components:**
- **Database Controller**: Core component managing database lifecycle
- **Backup Controller**: Automated backup scheduling and management
- **Monitoring Controller**: Observability and alerting setup
- **User Controller**: Database user and permission management

**Managed Resources:**
- **StatefulSets**: Provide stable network identity and persistent storage for databases
- **Services**: Enable network access and load balancing
- **ConfigMaps/Secrets**: Manage configuration and sensitive data
- **PersistentVolumeClaims**: Handle storage requirements

**Operational Integration:**
- **Backup Systems**: Integration with Velero and cloud storage
- **Monitoring Stack**: Prometheus metrics and Grafana dashboards
- **Network Policies**: Security and traffic control
- **Storage Classes**: Dynamic provisioning and storage tiers

## Learning Outcomes and Best Practices

### Key Concepts Mastered

1. **Operator Pattern Understanding**
   - Kubernetes API extension mechanisms
   - Custom Resource Definitions (CRDs)
   - Controller reconciliation loops
   - Finalizer patterns for cleanup

2. **Advanced Kubernetes Programming**
   - client-go library usage
   - controller-runtime framework
   - Webhook development and validation
   - RBAC and security considerations

3. **Production Readiness**
   - Comprehensive testing strategies
   - Observability and monitoring integration
   - Error handling and recovery
   - Packaging and distribution

### Best Practices Implemented

- **Declarative API Design**: Well-structured CRDs with proper validation
- **Reconciliation Logic**: Idempotent operations with proper error handling
- **Status Management**: Comprehensive status reporting with conditions
- **Resource Management**: Proper owner references and garbage collection
- **Security**: Minimal RBAC permissions and secure defaults
- **Observability**: Metrics, logging, and tracing integration
- **Testing**: Unit, integration, and end-to-end test coverage

This project provides comprehensive hands-on experience with Kubernetes operators, from basic concepts to production-ready implementation and distribution.
