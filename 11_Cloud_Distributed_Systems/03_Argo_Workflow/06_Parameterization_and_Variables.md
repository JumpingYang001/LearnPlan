# Parameterization and Variables

## Workflow and Template Parameters
- Defining parameters at workflow and template level
- Example usage

## Expression Evaluation
- Using expressions in parameters
- Example of variable substitution

## Variable Manipulation
- Manipulating variables in steps

## Example
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: param-example-
spec:
  entrypoint: print-message
  arguments:
    parameters:
    - name: message
      value: "Hello, Argo!"
  templates:
  - name: print-message
    inputs:
      parameters:
      - name: message
    container:
      image: alpine:3.7
      command: [echo]
      args: ["{{inputs.parameters.message}}"]
```
