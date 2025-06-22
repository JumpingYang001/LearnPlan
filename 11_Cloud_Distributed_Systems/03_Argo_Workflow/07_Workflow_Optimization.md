# Workflow Optimization

## Resource Allocation and Requests
- Setting resource requests and limits
- Example usage

## Parallelism and Synchronization
- Controlling parallelism
- Synchronizing steps

## Workflow and Pod Garbage Collection
- Cleaning up completed workflows and pods
- Example configuration

## Example
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: optimize-example-
spec:
  entrypoint: optimize
  templates:
  - name: optimize
    container:
      image: busybox
      command: [sh, -c]
      args: ["echo optimizing"]
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
```
