# Project 3: Multi-tenant Kubernetes Environment

## Overview
Design and implement a secure, scalable multi-tenant Kubernetes environment with proper isolation mechanisms, resource quotas, monitoring per tenant, and automated tenant onboarding processes.

## Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                     Control Plane                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ API Server  │  │ Scheduler   │  │ Controller  │             │
│  │             │  │             │  │  Manager    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                 Tenant Management                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Tenant    │  │   RBAC      │  │  Resource   │             │
│  │ Controller  │  │ Controller  │  │   Quotas    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  Tenant Isolation                               │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│ │   Tenant A  │ │   Tenant B  │ │   Tenant C  │ │   Tenant D  ││
│ │             │ │             │ │             │ │             ││
│ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ ││
│ │ │   App   │ │ │ │   App   │ │ │ │   App   │ │ │ │   App   │ ││
│ │ │ Workload│ │ │ │ Workload│ │ │ │ Workload│ │ │ │ Workload│ ││
│ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ ││
│ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ ││
│ │ │ Storage │ │ │ │ Storage │ │ │ │ Storage │ │ │ │ Storage │ ││
│ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ ││
│ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ ││
│ │ │ Network │ │ │ │ Network │ │ │ │ Network │ │ │ │ Network │ ││
│ │ │ Policy  │ │ │ │ Policy  │ │ │ │ Policy  │ │ │ │ Policy  ││
│ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ ││
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Tenant CRD and Controller

#### Tenant Custom Resource Definition
```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: tenants.multitenancy.io
spec:
  group: multitenancy.io
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
              displayName:
                type: string
              owner:
                type: string
              contact:
                type: string
              resourceQuota:
                type: object
                properties:
                  cpu:
                    type: string
                  memory:
                    type: string
                  storage:
                    type: string
                  pods:
                    type: integer
                  services:
                    type: integer
                  persistentVolumeClaims:
                    type: integer
              networkPolicy:
                type: object
                properties:
                  isolation:
                    type: string
                    enum: ["strict", "relaxed", "none"]
                  allowedNamespaces:
                    type: array
                    items:
                      type: string
              storageClasses:
                type: array
                items:
                  type: string
              nodeSelector:
                type: object
                additionalProperties:
                  type: string
          status:
            type: object
            properties:
              phase:
                type: string
                enum: ["Pending", "Active", "Terminating", "Failed"]
              namespaces:
                type: array
                items:
                  type: string
              message:
                type: string
              lastUpdated:
                type: string
                format: date-time
    additionalPrinterColumns:
    - name: Display Name
      type: string
      jsonPath: .spec.displayName
    - name: Owner
      type: string
      jsonPath: .spec.owner
    - name: Phase
      type: string
      jsonPath: .status.phase
    - name: Age
      type: date
      jsonPath: .metadata.creationTimestamp
  scope: Cluster
  names:
    plural: tenants
    singular: tenant
    kind: Tenant
```

#### Tenant Controller Implementation
```go
// tenant-controller/main.go
package main

import (
    "context"
    "fmt"
    "time"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    rbacv1 "k8s.io/api/rbac/v1"
    networkingv1 "k8s.io/api/networking/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/client-go/kubernetes"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/log"

    multitenancyv1 "github.com/company/tenant-controller/api/v1"
)

type TenantReconciler struct {
    client.Client
    Scheme *runtime.Scheme
    Clientset kubernetes.Interface
}

func (r *TenantReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    logger := log.FromContext(ctx)
    logger.Info("Reconciling Tenant", "tenant", req.Name)

    // Fetch the Tenant instance
    tenant := &multitenancyv1.Tenant{}
    if err := r.Get(ctx, req.NamespacedName, tenant); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    // Handle deletion
    if tenant.DeletionTimestamp != nil {
        return r.handleDeletion(ctx, tenant)
    }

    // Add finalizer if not present
    if !containsString(tenant.Finalizers, "tenant.multitenancy.io/finalizer") {
        tenant.Finalizers = append(tenant.Finalizers, "tenant.multitenancy.io/finalizer")
        if err := r.Update(ctx, tenant); err != nil {
            return ctrl.Result{}, err
        }
    }

    // Create or update tenant resources
    if err := r.reconcileTenant(ctx, tenant); err != nil {
        tenant.Status.Phase = "Failed"
        tenant.Status.Message = err.Error()
        r.Status().Update(ctx, tenant)
        return ctrl.Result{RequeueAfter: time.Minute * 5}, err
    }

    // Update status
    tenant.Status.Phase = "Active"
    tenant.Status.LastUpdated = metav1.Time{Time: time.Now()}
    if err := r.Status().Update(ctx, tenant); err != nil {
        return ctrl.Result{}, err
    }

    return ctrl.Result{RequeueAfter: time.Minute * 10}, nil
}

func (r *TenantReconciler) reconcileTenant(ctx context.Context, tenant *multitenancyv1.Tenant) error {
    // Create namespaces
    if err := r.createNamespaces(ctx, tenant); err != nil {
        return fmt.Errorf("failed to create namespaces: %w", err)
    }

    // Create RBAC
    if err := r.createRBAC(ctx, tenant); err != nil {
        return fmt.Errorf("failed to create RBAC: %w", err)
    }

    // Create resource quotas
    if err := r.createResourceQuotas(ctx, tenant); err != nil {
        return fmt.Errorf("failed to create resource quotas: %w", err)
    }

    // Create network policies
    if err := r.createNetworkPolicies(ctx, tenant); err != nil {
        return fmt.Errorf("failed to create network policies: %w", err)
    }

    return nil
}

func (r *TenantReconciler) createNamespaces(ctx context.Context, tenant *multitenancyv1.Tenant) error {
    namespaces := []string{
        fmt.Sprintf("%s-dev", tenant.Name),
        fmt.Sprintf("%s-staging", tenant.Name),
        fmt.Sprintf("%s-prod", tenant.Name),
    }

    for _, nsName := range namespaces {
        ns := &corev1.Namespace{
            ObjectMeta: metav1.ObjectMeta{
                Name: nsName,
                Labels: map[string]string{
                    "tenant":                  tenant.Name,
                    "multitenancy.io/tenant":  tenant.Name,
                    "pod-security.kubernetes.io/enforce": "restricted",
                    "pod-security.kubernetes.io/audit":   "restricted",
                    "pod-security.kubernetes.io/warn":    "restricted",
                },
                Annotations: map[string]string{
                    "tenant.multitenancy.io/owner":   tenant.Spec.Owner,
                    "tenant.multitenancy.io/contact": tenant.Spec.Contact,
                },
            },
        }

        if err := ctrl.SetControllerReference(tenant, ns, r.Scheme); err != nil {
            return err
        }

        if err := r.Create(ctx, ns); err != nil && !errors.IsAlreadyExists(err) {
            return err
        }
    }

    tenant.Status.Namespaces = namespaces
    return nil
}

func (r *TenantReconciler) createRBAC(ctx context.Context, tenant *multitenancyv1.Tenant) error {
    // Create Role for tenant admin
    role := &rbacv1.Role{
        ObjectMeta: metav1.ObjectMeta{
            Name:      fmt.Sprintf("%s-admin", tenant.Name),
            Namespace: fmt.Sprintf("%s-dev", tenant.Name),
        },
        Rules: []rbacv1.PolicyRule{
            {
                APIGroups: [""],
                Resources: ["pods", "services", "configmaps", "secrets", "persistentvolumeclaims"],
                Verbs:     ["get", "list", "watch", "create", "update", "patch", "delete"],
            },
            {
                APIGroups: ["apps"],
                Resources: ["deployments", "replicasets", "statefulsets", "daemonsets"],
                Verbs:     ["get", "list", "watch", "create", "update", "patch", "delete"],
            },
            {
                APIGroups: ["networking.k8s.io"],
                Resources: ["ingresses", "networkpolicies"],
                Verbs:     ["get", "list", "watch", "create", "update", "patch", "delete"],
            },
        },
    }

    if err := ctrl.SetControllerReference(tenant, role, r.Scheme); err != nil {
        return err
    }

    if err := r.Create(ctx, role); err != nil && !errors.IsAlreadyExists(err) {
        return err
    }

    // Create RoleBinding
    roleBinding := &rbacv1.RoleBinding{
        ObjectMeta: metav1.ObjectMeta{
            Name:      fmt.Sprintf("%s-admin-binding", tenant.Name),
            Namespace: fmt.Sprintf("%s-dev", tenant.Name),
        },
        Subjects: []rbacv1.Subject{
            {
                Kind: "User",
                Name: tenant.Spec.Owner,
            },
        },
        RoleRef: rbacv1.RoleRef{
            Kind:     "Role",
            Name:     fmt.Sprintf("%s-admin", tenant.Name),
            APIGroup: "rbac.authorization.k8s.io",
        },
    }

    if err := ctrl.SetControllerReference(tenant, roleBinding, r.Scheme); err != nil {
        return err
    }

    if err := r.Create(ctx, roleBinding); err != nil && !errors.IsAlreadyExists(err) {
        return err
    }

    return nil
}

func (r *TenantReconciler) createResourceQuotas(ctx context.Context, tenant *multitenancyv1.Tenant) error {
    for _, nsName := range tenant.Status.Namespaces {
        quota := &corev1.ResourceQuota{
            ObjectMeta: metav1.ObjectMeta{
                Name:      fmt.Sprintf("%s-quota", tenant.Name),
                Namespace: nsName,
            },
            Spec: corev1.ResourceQuotaSpec{
                Hard: corev1.ResourceList{
                    "requests.cpu":               resource.MustParse(tenant.Spec.ResourceQuota.CPU),
                    "requests.memory":            resource.MustParse(tenant.Spec.ResourceQuota.Memory),
                    "requests.storage":           resource.MustParse(tenant.Spec.ResourceQuota.Storage),
                    "pods":                       resource.MustParse(fmt.Sprintf("%d", tenant.Spec.ResourceQuota.Pods)),
                    "services":                   resource.MustParse(fmt.Sprintf("%d", tenant.Spec.ResourceQuota.Services)),
                    "persistentvolumeclaims":     resource.MustParse(fmt.Sprintf("%d", tenant.Spec.ResourceQuota.PersistentVolumeClaims)),
                },
            },
        }

        if err := ctrl.SetControllerReference(tenant, quota, r.Scheme); err != nil {
            return err
        }

        if err := r.Create(ctx, quota); err != nil && !errors.IsAlreadyExists(err) {
            return err
        }
    }

    return nil
}

func (r *TenantReconciler) createNetworkPolicies(ctx context.Context, tenant *multitenancyv1.Tenant) error {
    for _, nsName := range tenant.Status.Namespaces {
        // Default deny all policy
        denyAll := &networkingv1.NetworkPolicy{
            ObjectMeta: metav1.ObjectMeta{
                Name:      "deny-all",
                Namespace: nsName,
            },
            Spec: networkingv1.NetworkPolicySpec{
                PodSelector: metav1.LabelSelector{},
                PolicyTypes: []networkingv1.PolicyType{
                    networkingv1.PolicyTypeIngress,
                    networkingv1.PolicyTypeEgress,
                },
            },
        }

        if err := ctrl.SetControllerReference(tenant, denyAll, r.Scheme); err != nil {
            return err
        }

        if err := r.Create(ctx, denyAll); err != nil && !errors.IsAlreadyExists(err) {
            return err
        }

        // Allow within namespace policy
        allowWithinNS := &networkingv1.NetworkPolicy{
            ObjectMeta: metav1.ObjectMeta{
                Name:      "allow-within-namespace",
                Namespace: nsName,
            },
            Spec: networkingv1.NetworkPolicySpec{
                PodSelector: metav1.LabelSelector{},
                PolicyTypes: []networkingv1.PolicyType{
                    networkingv1.PolicyTypeIngress,
                    networkingv1.PolicyTypeEgress,
                },
                Ingress: []networkingv1.NetworkPolicyIngressRule{
                    {
                        From: []networkingv1.NetworkPolicyPeer{
                            {
                                NamespaceSelector: &metav1.LabelSelector{
                                    MatchLabels: map[string]string{
                                        "name": nsName,
                                    },
                                },
                            },
                        },
                    },
                },
                Egress: []networkingv1.NetworkPolicyEgressRule{
                    {
                        To: []networkingv1.NetworkPolicyPeer{
                            {
                                NamespaceSelector: &metav1.LabelSelector{
                                    MatchLabels: map[string]string{
                                        "name": nsName,
                                    },
                                },
                            },
                        },
                    },
                    // Allow DNS
                    {
                        To: []networkingv1.NetworkPolicyPeer{},
                        Ports: []networkingv1.NetworkPolicyPort{
                            {
                                Protocol: &[]corev1.Protocol{corev1.ProtocolUDP}[0],
                                Port:     &intstr.IntOrString{IntVal: 53},
                            },
                        },
                    },
                },
            },
        }

        if err := ctrl.SetControllerReference(tenant, allowWithinNS, r.Scheme); err != nil {
            return err
        }

        if err := r.Create(ctx, allowWithinNS); err != nil && !errors.IsAlreadyExists(err) {
            return err
        }
    }

    return nil
}

func containsString(slice []string, s string) bool {
    for _, item := range slice {
        if item == s {
            return true
        }
    }
    return false
}
```

### 2. Tenant Onboarding Automation

#### Tenant Onboarding Script
```bash
#!/bin/bash
# tenant-onboarding.sh - Automated tenant onboarding

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENANT_NAME=""
DISPLAY_NAME=""
OWNER=""
CONTACT=""
CPU_QUOTA="4"
MEMORY_QUOTA="8Gi"
STORAGE_QUOTA="50Gi"
PODS_QUOTA="20"
SERVICES_QUOTA="10"
PVC_QUOTA="10"
NETWORK_ISOLATION="strict"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -n, --name          Tenant name (required)
    -d, --display-name  Display name (required)
    -o, --owner         Owner email (required)
    -c, --contact       Contact email (required)
    --cpu-quota         CPU quota (default: 4)
    --memory-quota      Memory quota (default: 8Gi)
    --storage-quota     Storage quota (default: 50Gi)
    --pods-quota        Pods quota (default: 20)
    --services-quota    Services quota (default: 10)
    --pvc-quota         PVC quota (default: 10)
    --network-isolation Network isolation level: strict|relaxed|none (default: strict)
    -h, --help          Show this help message

Example:
    $0 -n acme-corp -d "ACME Corporation" -o john.doe@acme.com -c support@acme.com
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--name)
                TENANT_NAME="$2"
                shift 2
                ;;
            -d|--display-name)
                DISPLAY_NAME="$2"
                shift 2
                ;;
            -o|--owner)
                OWNER="$2"
                shift 2
                ;;
            -c|--contact)
                CONTACT="$2"
                shift 2
                ;;
            --cpu-quota)
                CPU_QUOTA="$2"
                shift 2
                ;;
            --memory-quota)
                MEMORY_QUOTA="$2"
                shift 2
                ;;
            --storage-quota)
                STORAGE_QUOTA="$2"
                shift 2
                ;;
            --pods-quota)
                PODS_QUOTA="$2"
                shift 2
                ;;
            --services-quota)
                SERVICES_QUOTA="$2"
                shift 2
                ;;
            --pvc-quota)
                PVC_QUOTA="$2"
                shift 2
                ;;
            --network-isolation)
                NETWORK_ISOLATION="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

validate_input() {
    if [[ -z "$TENANT_NAME" ]]; then
        error "Tenant name is required"
        exit 1
    fi

    if [[ -z "$DISPLAY_NAME" ]]; then
        error "Display name is required"
        exit 1
    fi

    if [[ -z "$OWNER" ]]; then
        error "Owner email is required"
        exit 1
    fi

    if [[ -z "$CONTACT" ]]; then
        error "Contact email is required"
        exit 1
    fi

    # Validate tenant name format
    if [[ ! "$TENANT_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
        error "Tenant name must be a valid DNS-1123 label"
        exit 1
    fi

    # Validate email formats
    if [[ ! "$OWNER" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
        error "Invalid owner email format"
        exit 1
    fi

    if [[ ! "$CONTACT" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
        error "Invalid contact email format"
        exit 1
    fi

    # Validate network isolation
    if [[ ! "$NETWORK_ISOLATION" =~ ^(strict|relaxed|none)$ ]]; then
        error "Network isolation must be one of: strict, relaxed, none"
        exit 1
    fi
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check if tenant already exists
    if kubectl get tenant "$TENANT_NAME" &> /dev/null; then
        error "Tenant '$TENANT_NAME' already exists"
        exit 1
    fi

    # Check tenant controller is running
    if ! kubectl get pods -n tenant-system -l app=tenant-controller | grep -q Running; then
        error "Tenant controller is not running"
        exit 1
    fi

    log "Prerequisites check completed"
}

create_tenant_manifest() {
    log "Creating tenant manifest..."

    cat > "/tmp/${TENANT_NAME}-tenant.yaml" << EOF
apiVersion: multitenancy.io/v1
kind: Tenant
metadata:
  name: ${TENANT_NAME}
  labels:
    tenant.multitenancy.io/created-by: "onboarding-script"
  annotations:
    tenant.multitenancy.io/created-at: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    tenant.multitenancy.io/onboarded-by: "$(whoami)"
spec:
  displayName: "${DISPLAY_NAME}"
  owner: "${OWNER}"
  contact: "${CONTACT}"
  resourceQuota:
    cpu: "${CPU_QUOTA}"
    memory: "${MEMORY_QUOTA}"
    storage: "${STORAGE_QUOTA}"
    pods: ${PODS_QUOTA}
    services: ${SERVICES_QUOTA}
    persistentVolumeClaims: ${PVC_QUOTA}
  networkPolicy:
    isolation: "${NETWORK_ISOLATION}"
    allowedNamespaces: []
  storageClasses:
    - "standard"
    - "fast"
  nodeSelector: {}
EOF

    log "Tenant manifest created: /tmp/${TENANT_NAME}-tenant.yaml"
}

create_tenant() {
    log "Creating tenant '${TENANT_NAME}'..."

    if kubectl apply -f "/tmp/${TENANT_NAME}-tenant.yaml"; then
        log "Tenant '${TENANT_NAME}' created successfully"
    else
        error "Failed to create tenant '${TENANT_NAME}'"
        exit 1
    fi

    # Wait for tenant to be ready
    log "Waiting for tenant to be ready..."
    timeout=300
    elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        phase=$(kubectl get tenant "$TENANT_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
        if [[ "$phase" == "Active" ]]; then
            log "Tenant '${TENANT_NAME}' is active"
            break
        elif [[ "$phase" == "Failed" ]]; then
            error "Tenant '${TENANT_NAME}' failed to initialize"
            kubectl get tenant "$TENANT_NAME" -o yaml
            exit 1
        fi
        
        sleep 10
        elapsed=$((elapsed + 10))
        log "Waiting... (${elapsed}s/${timeout}s) - Status: ${phase}"
    done

    if [[ $elapsed -ge $timeout ]]; then
        error "Timeout waiting for tenant to be ready"
        exit 1
    fi
}

setup_monitoring() {
    log "Setting up monitoring for tenant '${TENANT_NAME}'..."

    # Create monitoring configuration
    cat > "/tmp/${TENANT_NAME}-monitoring.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${TENANT_NAME}-grafana-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
    tenant: ${TENANT_NAME}
data:
  ${TENANT_NAME}-dashboard.json: |
    {
      "dashboard": {
        "id": null,
        "title": "${DISPLAY_NAME} - Tenant Dashboard",
        "tags": ["tenant", "${TENANT_NAME}"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "Resource Usage",
            "type": "stat",
            "targets": [
              {
                "expr": "sum(kube_pod_container_resource_requests{namespace=~\"${TENANT_NAME}-.*\", resource=\"cpu\"}) / sum(kube_resourcequota{namespace=~\"${TENANT_NAME}-.*\", resource=\"requests.cpu\"}) * 100",
                "legendFormat": "CPU Usage %"
              },
              {
                "expr": "sum(kube_pod_container_resource_requests{namespace=~\"${TENANT_NAME}-.*\", resource=\"memory\"}) / sum(kube_resourcequota{namespace=~\"${TENANT_NAME}-.*\", resource=\"requests.memory\"}) * 100",
                "legendFormat": "Memory Usage %"
              }
            ]
          },
          {
            "id": 2,
            "title": "Pod Count",
            "type": "stat",
            "targets": [
              {
                "expr": "sum(kube_pod_info{namespace=~\"${TENANT_NAME}-.*\"})",
                "legendFormat": "Running Pods"
              }
            ]
          }
        ]
      }
    }
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ${TENANT_NAME}-workloads
  namespace: monitoring
  labels:
    tenant: ${TENANT_NAME}
spec:
  selector:
    matchLabels:
      tenant: ${TENANT_NAME}
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
  namespaceSelector:
    matchNames:
    - ${TENANT_NAME}-dev
    - ${TENANT_NAME}-staging
    - ${TENANT_NAME}-prod
EOF

    if kubectl apply -f "/tmp/${TENANT_NAME}-monitoring.yaml"; then
        log "Monitoring setup completed for tenant '${TENANT_NAME}'"
    else
        warn "Failed to set up monitoring for tenant '${TENANT_NAME}'"
    fi
}

create_access_documentation() {
    log "Creating access documentation..."

    cat > "/tmp/${TENANT_NAME}-access-guide.md" << EOF
# Tenant Access Guide: ${DISPLAY_NAME}

## Overview
Your tenant \`${TENANT_NAME}\` has been successfully created with the following resources:

## Namespaces
- \`${TENANT_NAME}-dev\` - Development environment
- \`${TENANT_NAME}-staging\` - Staging environment
- \`${TENANT_NAME}-prod\` - Production environment

## Resource Quotas
- CPU: ${CPU_QUOTA}
- Memory: ${MEMORY_QUOTA}
- Storage: ${STORAGE_QUOTA}
- Pods: ${PODS_QUOTA}
- Services: ${SERVICES_QUOTA}
- Persistent Volume Claims: ${PVC_QUOTA}

## Access Instructions

### kubectl Configuration
1. Ensure you have the appropriate kubeconfig file
2. Set your context to access your namespaces:
   \`\`\`bash
   kubectl config set-context ${TENANT_NAME}-dev --cluster=<cluster> --user=<user> --namespace=${TENANT_NAME}-dev
   kubectl config use-context ${TENANT_NAME}-dev
   \`\`\`

### Available Commands
You have admin access within your namespaces. You can:
- Create and manage deployments, services, configmaps, secrets
- View logs and exec into pods
- Manage network policies within your namespaces
- Create persistent volume claims using available storage classes

### Monitoring and Logging
- Grafana Dashboard: Available in the monitoring namespace
- Logs: Available through your cluster's logging solution

## Support
- Owner: ${OWNER}
- Contact: ${CONTACT}
- Platform Team: platform-team@company.com

## Next Steps
1. Deploy your applications to the \`${TENANT_NAME}-dev\` namespace
2. Test your applications in the \`${TENANT_NAME}-staging\` namespace
3. Deploy to production in the \`${TENANT_NAME}-prod\` namespace

For more information, see the platform documentation.
EOF

    log "Access documentation created: /tmp/${TENANT_NAME}-access-guide.md"
}

send_notification() {
    log "Sending notification email..."

    # This would typically integrate with your email system
    # For now, we'll create a notification file
    cat > "/tmp/${TENANT_NAME}-notification.txt" << EOF
Subject: Kubernetes Tenant Created - ${DISPLAY_NAME}

Hello ${OWNER},

Your Kubernetes tenant "${DISPLAY_NAME}" (${TENANT_NAME}) has been successfully created.

Tenant Details:
- Name: ${TENANT_NAME}
- Display Name: ${DISPLAY_NAME}
- Owner: ${OWNER}
- Contact: ${CONTACT}

Resources Allocated:
- CPU: ${CPU_QUOTA}
- Memory: ${MEMORY_QUOTA}
- Storage: ${STORAGE_QUOTA}
- Pods: ${PODS_QUOTA}
- Services: ${SERVICES_QUOTA}
- PVCs: ${PVC_QUOTA}

Namespaces Created:
- ${TENANT_NAME}-dev
- ${TENANT_NAME}-staging
- ${TENANT_NAME}-prod

Please see the attached access guide for detailed information on how to use your tenant.

Best regards,
Platform Team
EOF

    log "Notification prepared: /tmp/${TENANT_NAME}-notification.txt"
}

cleanup() {
    log "Cleaning up temporary files..."
    rm -f "/tmp/${TENANT_NAME}-tenant.yaml"
    rm -f "/tmp/${TENANT_NAME}-monitoring.yaml"
}

main() {
    log "Starting tenant onboarding process..."

    parse_args "$@"
    validate_input
    check_prerequisites
    create_tenant_manifest
    create_tenant
    setup_monitoring
    create_access_documentation
    send_notification
    cleanup

    log "Tenant onboarding completed successfully!"
    log "Tenant: ${TENANT_NAME} (${DISPLAY_NAME})"
    log "Owner: ${OWNER}"
    log "Contact: ${CONTACT}"
    log "Access guide: /tmp/${TENANT_NAME}-access-guide.md"
}

# Handle script interruption
trap 'error "Script interrupted"; cleanup; exit 1' INT TERM

main "$@"
```

### 3. Advanced Isolation Mechanisms

#### Hierarchical Namespaces Configuration
```yaml
apiVersion: hnc.x-k8s.io/v1alpha2
kind: HierarchyConfiguration
metadata:
  name: hierarchy
  namespace: tenant-acme-corp
spec:
  parent: tenant-root
---
apiVersion: hnc.x-k8s.io/v1alpha2
kind: HierarchyConfiguration
metadata:
  name: hierarchy
  namespace: acme-corp-dev
spec:
  parent: tenant-acme-corp
---
apiVersion: hnc.x-k8s.io/v1alpha2
kind: HierarchyConfiguration
metadata:
  name: hierarchy
  namespace: acme-corp-staging
spec:
  parent: tenant-acme-corp
---
apiVersion: hnc.x-k8s.io/v1alpha2
kind: HierarchyConfiguration
metadata:
  name: hierarchy
  namespace: acme-corp-prod
spec:
  parent: tenant-acme-corp
```

#### Pod Security Standards per Tenant
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: acme-corp-prod
  labels:
    tenant: acme-corp
    environment: production
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
apiVersion: v1
kind: Namespace
metadata:
  name: acme-corp-dev
  labels:
    tenant: acme-corp
    environment: development
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

#### Gatekeeper Policies for Multi-tenancy
```yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: tenantresourceconstraints
spec:
  crd:
    spec:
      names:
        kind: TenantResourceConstraints
      validation:
        properties:
          allowedStorageClasses:
            type: array
            items:
              type: string
          maxReplicas:
            type: integer
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package tenantresourceconstraints

        violation[{"msg": msg}] {
          input.review.kind.kind == "Deployment"
          input.review.object.spec.replicas > input.parameters.maxReplicas
          msg := sprintf("Deployment %v has %v replicas, maximum allowed is %v", [input.review.object.metadata.name, input.review.object.spec.replicas, input.parameters.maxReplicas])
        }

        violation[{"msg": msg}] {
          input.review.kind.kind == "PersistentVolumeClaim"
          not input.review.object.spec.storageClassName in input.parameters.allowedStorageClasses
          msg := sprintf("PVC %v uses storage class %v, allowed classes are %v", [input.review.object.metadata.name, input.review.object.spec.storageClassName, input.parameters.allowedStorageClasses])
        }
---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: TenantResourceConstraints
metadata:
  name: acme-corp-constraints
spec:
  match:
    kinds:
      - apiGroups: ["apps"]
        kinds: ["Deployment"]
      - apiGroups: [""]
        kinds: ["PersistentVolumeClaim"]
    namespaces: ["acme-corp-dev", "acme-corp-staging", "acme-corp-prod"]
  parameters:
    allowedStorageClasses: ["standard", "fast"]
    maxReplicas: 10
```

### 4. Tenant Monitoring and Cost Tracking

#### Tenant Resource Usage Monitoring
```yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: tenant-metrics
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: kube-state-metrics
  endpoints:
  - port: http-metrics
    interval: 30s
    path: /metrics
    relabelings:
    - sourceLabels: [__meta_kubernetes_namespace]
      regex: '(.*-dev|.*-staging|.*-prod)'
      targetLabel: tenant_namespace
    - sourceLabels: [tenant_namespace]
      regex: '(.*)-.*'
      targetLabel: tenant_name
      replacement: '${1}'
```

#### Cost Allocation Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Multi-tenant Cost Analysis",
    "panels": [
      {
        "title": "CPU Cost by Tenant",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (tenant) (label_replace(kube_pod_container_resource_requests{resource='cpu'}, 'tenant', '$1', 'namespace', '(.*)-.*')) * 0.05",
            "legendFormat": "{{tenant}}"
          }
        ]
      },
      {
        "title": "Memory Cost by Tenant",
        "type": "piechart", 
        "targets": [
          {
            "expr": "sum by (tenant) (label_replace(kube_pod_container_resource_requests{resource='memory'}, 'tenant', '$1', 'namespace', '(.*)-.*')) / 1024 / 1024 / 1024 * 0.01",
            "legendFormat": "{{tenant}}"
          }
        ]
      },
      {
        "title": "Storage Cost by Tenant",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (tenant) (label_replace(kube_persistentvolumeclaim_resource_requests_storage_bytes, 'tenant', '$1', 'namespace', '(.*)-.*')) / 1024 / 1024 / 1024 * 0.001",
            "legendFormat": "{{tenant}}"
          }
        ]
      }
    ]
  }
}
```

### 5. Tenant Self-Service Portal

#### Frontend Web Interface
```typescript
// tenant-portal/src/components/TenantDashboard.tsx
import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Button } from 'antd';

interface TenantData {
  name: string;
  displayName: string;
  resourceUsage: {
    cpu: { used: number; total: number };
    memory: { used: number; total: number };
    pods: { used: number; total: number };
    storage: { used: number; total: number };
  };
  namespaces: string[];
  applications: Array<{
    name: string;
    namespace: string;
    status: string;
    replicas: number;
  }>;
}

const TenantDashboard: React.FC<{ tenantName: string }> = ({ tenantName }) => {
  const [tenantData, setTenantData] = useState<TenantData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTenantData();
  }, [tenantName]);

  const fetchTenantData = async () => {
    try {
      const response = await fetch(`/api/v1/tenants/${tenantName}`);
      const data = await response.json();
      setTenantData(data);
    } catch (error) {
      console.error('Failed to fetch tenant data:', error);
    } finally {
      setLoading(false);
    }
  };

  const createApplication = async (appName: string, namespace: string) => {
    try {
      await fetch(`/api/v1/tenants/${tenantName}/applications`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: appName, namespace }),
      });
      fetchTenantData(); // Refresh data
    } catch (error) {
      console.error('Failed to create application:', error);
    }
  };

  if (loading || !tenantData) {
    return <div>Loading...</div>;
  }

  const { resourceUsage, namespaces, applications } = tenantData;

  const columns = [
    {
      title: 'Application',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Namespace',
      dataIndex: 'namespace',
      key: 'namespace',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <span style={{ color: status === 'Running' ? 'green' : 'orange' }}>
          {status}
        </span>
      ),
    },
    {
      title: 'Replicas',
      dataIndex: 'replicas',
      key: 'replicas',
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: any) => (
        <Button size="small" onClick={() => window.open(`/apps/${record.name}`)}>
          Manage
        </Button>
      ),
    },
  ];

  return (
    <div>
      <h1>{tenantData.displayName} Dashboard</h1>
      
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic title="CPU Usage" value={resourceUsage.cpu.used} suffix={`/ ${resourceUsage.cpu.total}`} />
            <Progress 
              percent={Math.round((resourceUsage.cpu.used / resourceUsage.cpu.total) * 100)} 
              status={resourceUsage.cpu.used / resourceUsage.cpu.total > 0.8 ? 'exception' : 'normal'}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="Memory Usage" value={`${resourceUsage.memory.used}Gi`} suffix={`/ ${resourceUsage.memory.total}Gi`} />
            <Progress 
              percent={Math.round((resourceUsage.memory.used / resourceUsage.memory.total) * 100)}
              status={resourceUsage.memory.used / resourceUsage.memory.total > 0.8 ? 'exception' : 'normal'}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="Pods" value={resourceUsage.pods.used} suffix={`/ ${resourceUsage.pods.total}`} />
            <Progress 
              percent={Math.round((resourceUsage.pods.used / resourceUsage.pods.total) * 100)}
              status={resourceUsage.pods.used / resourceUsage.pods.total > 0.8 ? 'exception' : 'normal'}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="Storage" value={`${resourceUsage.storage.used}Gi`} suffix={`/ ${resourceUsage.storage.total}Gi`} />
            <Progress 
              percent={Math.round((resourceUsage.storage.used / resourceUsage.storage.total) * 100)}
              status={resourceUsage.storage.used / resourceUsage.storage.total > 0.8 ? 'exception' : 'normal'}
            />
          </Card>
        </Col>
      </Row>

      <Card title="Applications" style={{ marginBottom: 16 }}>
        <Table 
          columns={columns} 
          dataSource={applications} 
          rowKey="name"
          pagination={false}
        />
      </Card>

      <Card title="Namespaces">
        <Row gutter={16}>
          {namespaces.map(ns => (
            <Col span={8} key={ns}>
              <Card size="small" title={ns}>
                <Button 
                  type="primary" 
                  size="small"
                  onClick={() => window.open(`/kubectl/${ns}`)}
                >
                  Access
                </Button>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>
    </div>
  );
};

export default TenantDashboard;
```

This comprehensive multi-tenant Kubernetes implementation provides:

1. **Custom Tenant CRD**: Complete tenant lifecycle management
2. **Automated Onboarding**: Script-based tenant provisioning
3. **Isolation Mechanisms**: Network policies, resource quotas, RBAC
4. **Advanced Security**: Pod Security Standards, Gatekeeper policies
5. **Cost Tracking**: Resource usage monitoring and cost allocation
6. **Self-Service Portal**: Web interface for tenant management
7. **Hierarchical Namespaces**: Advanced namespace organization

The platform ensures complete tenant isolation while providing comprehensive management and monitoring capabilities.
