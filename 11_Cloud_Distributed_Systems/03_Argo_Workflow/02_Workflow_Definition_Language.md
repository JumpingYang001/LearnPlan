# Workflow Definition Language

## YAML-based Workflow Definitions
- Structure of Argo Workflow YAML
- Example of a simple workflow YAML

## Templates, Steps, and DAG
- Defining templates and steps
- Using DAG for complex dependencies

## Parameters and Arguments
- Passing parameters to templates
- Example usage

## Example
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: hello-world-
spec:
  entrypoint: whalesay
  templates:
  - name: whalesay
    container:
      image: docker/whalesay
      command: [cowsay]
      args: ["hello world"]
```
