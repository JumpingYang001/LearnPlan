# Resource Management

*Part of: Kubernetes Architecture Learning Path*
*Estimated Duration: 1 week*

## Overview

This section covers comprehensive resource management in Kubernetes, including resource requests and limits, autoscaling mechanisms, and cluster resource optimization. You'll learn how to effectively manage compute resources and implement automatic scaling strategies.

## Learning Objectives

By the end of this section, you should be able to:
- Configure resource requests and limits for optimal resource utilization
- Implement horizontal and vertical pod autoscaling
- Set up cluster autoscaling for dynamic node management
- Optimize resource usage and implement proper QoS policies

## Topics Covered

### 1. Resource Requests and Limits

#### CPU and Memory Specification
- **Resource Units**
  - CPU measurement (cores, millicores)
  - Memory measurement (bytes, Ki, Mi, Gi)
  - Resource request vs limit concepts
  - Overcommitment strategies

- **Container Resource Configuration**
  ```yaml
  resources:
    requests:
      memory: "64Mi"
      cpu: "250m"
    limits:
      memory: "128Mi"
      cpu: "500m"
  ```

- **Resource Calculation**
  - Node capacity assessment
  - Resource allocation strategies
  - Bin packing algorithms
  - Resource fragmentation

#### QoS Classes
- **Quality of Service Classes**
  - **Guaranteed**: Requests equal limits
  - **Burstable**: Requests less than limits
  - **BestEffort**: No requests or limits set

- **QoS Impact on Scheduling**
  - Pod prioritization
  - Eviction order
  - Resource contention handling
  - Node pressure responses

- **QoS Configuration Best Practices**
  - Critical workload guarantees
  - Development environment settings
  - Batch job configurations
  - Resource monitoring alignment

#### Resource Quotas
- **Namespace-level Quotas**
  - Compute resource quotas
  - Storage resource quotas
  - Object count quotas
  - Extended resource quotas

- **ResourceQuota Configuration**
  ```yaml
  apiVersion: v1
  kind: ResourceQuota
  metadata:
    name: compute-quota
  spec:
    hard:
      requests.cpu: "4"
      requests.memory: 8Gi
      limits.cpu: "8"
      limits.memory: 16Gi
      pods: "10"
  ```

- **Quota Scopes and Selectors**
  - Priority class scopes
  - Cross-namespace quotas
  - Hierarchical quotas
  - Quota enforcement

#### LimitRanges
- **Container-level Limits**
  - Default requests and limits
  - Minimum and maximum constraints
  - Request-to-limit ratios
  - Storage capacity limits

- **LimitRange Configuration**
  ```yaml
  apiVersion: v1
  kind: LimitRange
  metadata:
    name: resource-constraints
  spec:
    limits:
    - default:
        cpu: "500m"
        memory: "512Mi"
      defaultRequest:
        cpu: "100m"
        memory: "128Mi"
      type: Container
  ```

### 2. Horizontal Pod Autoscaling (HPA)

#### HPA Architecture
- **HPA Controller Components**
  - Metrics server integration
  - Controller manager role
  - Scaling decision logic
  - Target reference objects

- **HPA API Versions**
  - autoscaling/v1 (CPU-based)
  - autoscaling/v2 (multi-metric)
  - Custom resource support
  - Behavior configuration

#### Metrics-based Scaling
- **CPU Utilization Scaling**
  - Target CPU percentage
  - Current utilization calculation
  - Scaling threshold management
  - CPU metric collection

- **Memory Utilization Scaling**
  - Memory-based scaling policies
  - Memory pressure indicators
  - OOM (Out of Memory) prevention
  - Memory leak detection

- **Request-per-Second Scaling**
  - Application-level metrics
  - Load-based scaling
  - Traffic pattern analysis
  - Peak load handling

#### Custom Metrics
- **Custom Metrics API**
  - Metrics server adaptation
  - External metrics integration
  - Prometheus adapter setup
  - Custom metric collection

- **External Metrics**
  - Cloud provider metrics
  - Application performance metrics
  - Business metrics scaling
  - Queue depth scaling

- **Multiple Metrics Configuration**
  ```yaml
  apiVersion: autoscaling/v2
  kind: HorizontalPodAutoscaler
  metadata:
    name: multi-metric-hpa
  spec:
    scaleTargetRef:
      apiVersion: apps/v1
      kind: Deployment
      name: webapp
    minReplicas: 2
    maxReplicas: 10
    metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  ```

#### Scaling Behavior
- **Scale-up Policies**
  - Scaling velocity limits
  - Stabilization windows
  - Scale-up percentage/count
  - Scaling cooldown periods

- **Scale-down Policies**
  - Graceful scale-down
  - Minimum scale-down interval
  - Scale-down stabilization
  - Pod disruption budgets

### 3. Vertical Pod Autoscaling (VPA)

#### VPA Components
- **VPA Architecture**
  - Recommender component
  - Updater component
  - Admission controller webhook
  - History storage

- **VPA Modes**
  - "Off" mode (recommendations only)
  - "Initial" mode (at pod creation)
  - "Auto" mode (automatic updates)
  - "Recreate" mode (pod recreation)

#### Recommendation Modes
- **Resource Recommendations**
  - Historical usage analysis
  - Recommendation algorithms
  - Confidence intervals
  - Resource optimization

- **Recommendation Quality**
  - Data collection period
  - Workload pattern analysis
  - Seasonal adjustment
  - Outlier handling

#### Integration with HPA
- **HPA-VPA Interaction**
  - Coordination mechanisms
  - Metric conflicts resolution
  - Scaling decision priorities
  - Combined scaling strategies

- **Best Practices**
  - When to use VPA vs HPA
  - Multi-dimensional scaling
  - Resource efficiency optimization
  - Cost optimization strategies

#### Resource Efficiency
- **Right-sizing Workloads**
  - Over-provisioning reduction
  - Resource waste elimination
  - Cost optimization
  - Performance maintenance

### 4. Cluster Autoscaling

#### Node Pool Scaling
- **Cluster Autoscaler Architecture**
  - Node group management
  - Scaling decision logic
  - Cloud provider integration
  - Node lifecycle management

- **Node Pool Configuration**
  - Minimum and maximum node counts
  - Instance types and sizes
  - Spot/preemptible instances
  - Multi-zone deployments

#### Scaling Triggers
- **Scale-up Triggers**
  - Unschedulable pods
  - Resource pressure
  - Pending pod analysis
  - Node utilization thresholds

- **Scale-down Triggers**
  - Node underutilization
  - Empty node detection
  - Graceful node termination
  - Pod rescheduling

#### Scaling Policies
- **Scale-up Policies**
  - New node provisioning
  - Node selection algorithms
  - Batch scaling strategies
  - Scaling velocity limits

- **Scale-down Policies**
  - Node drain procedures
  - Pod disruption budgets
  - Critical pod protection
  - Scale-down delays

#### Provider-specific Implementations
- **AWS Auto Scaling Groups**
  - Launch templates
  - Mixed instance policies
  - Spot fleet integration
  - Availability zone balancing

- **Google Cloud Node Pools**
  - Preemptible instances
  - Node auto-upgrade
  - Node auto-repair
  - Surge upgrades

- **Azure Virtual Machine Scale Sets**
  - Low-priority VMs
  - Proximity placement groups
  - Accelerated networking
  - Spot instances

## Hands-on Labs

### Lab 1: Resource Quotas and Limits
- Configure namespace resource quotas
- Set up LimitRanges for containers
- Test resource enforcement
- Monitor resource usage

### Lab 2: Horizontal Pod Autoscaling
- Deploy metrics server
- Configure CPU-based HPA
- Set up custom metrics HPA
- Test scaling behavior

### Lab 3: Vertical Pod Autoscaling
- Install VPA components
- Configure VPA recommendations
- Test automatic resource updates
- Analyze resource optimization

### Lab 4: Cluster Autoscaling
- Configure cluster autoscaler
- Test node scaling triggers
- Implement multi-zone scaling
- Monitor cluster resource usage

## Best Practices

### Resource Configuration
- Always set resource requests for production workloads
- Use appropriate QoS classes for different workload types
- Implement resource quotas for multi-tenant environments
- Monitor resource utilization continuously

### Autoscaling Strategy
- Start with HPA for stateless applications
- Use VPA for right-sizing workloads
- Combine HPA and VPA carefully
- Configure appropriate scaling policies

### Cluster Management
- Enable cluster autoscaling for dynamic workloads
- Use multiple node pools for different workload types
- Configure appropriate scaling limits
- Monitor cluster capacity and costs

### Cost Optimization
- Right-size resources based on actual usage
- Use spot/preemptible instances where appropriate
- Implement resource efficiency monitoring
- Regular resource utilization reviews

## Troubleshooting Guide

### Common Issues
1. **Pods not scaling**
   - Check HPA configuration
   - Verify metrics server status
   - Review resource requests
   - Check scaling policies

2. **Resource quota exceeded**
   - Review quota configuration
   - Check resource usage
   - Verify namespace limits
   - Update quota if needed

3. **Nodes not scaling**
   - Check cluster autoscaler logs
   - Verify node pool configuration
   - Review IAM permissions
   - Check cloud provider limits

4. **VPA not updating resources**
   - Check VPA mode configuration
   - Verify recommendation quality
   - Review update policies
   - Check admission controller

## Assessment

### Knowledge Check
- Configure appropriate resource requests and limits
- Implement effective autoscaling strategies
- Troubleshoot resource management issues
- Optimize cluster resource utilization

### Practical Tasks
- Set up comprehensive resource management
- Configure multi-metric autoscaling
- Implement cluster-wide resource policies
- Optimize resource costs

## Resources

### Documentation
- [Kubernetes Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Vertical Pod Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler)
- [Cluster Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler)

### Tools
- Metrics Server
- Prometheus Adapter
- VPA Components
- Cluster Autoscaler

### Monitoring
- Resource utilization metrics
- Autoscaling events
- Cost monitoring
- Performance metrics

## Next Section

Continue to [Security Architecture](07_Security_Architecture.md) to learn about Kubernetes security concepts, RBAC, and security best practices.
