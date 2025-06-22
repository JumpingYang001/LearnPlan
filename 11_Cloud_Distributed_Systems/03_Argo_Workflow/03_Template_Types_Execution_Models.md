# Template Types and Execution Models

## Container, Script, and Resource Templates
- How to use different template types
- Example of each template type

## DAG Templates
- Defining complex dependencies
- Example DAG template

## Suspend Templates and Human Interaction
- Suspending workflows for manual intervention
- Example usage

## Example
```yaml
# Example of a script template
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: script-example-
spec:
  entrypoint: run-script
  templates:
  - name: run-script
    script:
      image: python:3.8
      command: [python]
      source: |
        print("Hello from script!")
```
