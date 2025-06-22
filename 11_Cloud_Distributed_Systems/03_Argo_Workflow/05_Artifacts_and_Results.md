# Artifacts and Results

## Input and Output Artifacts
- How to define and use artifacts
- Example of passing artifacts between steps

## Artifact Repositories
- Supported artifact repositories (S3, GCS, etc.)
- Configuration examples

## Result Passing
- Passing results between steps
- Example usage

## Example
```yaml
# Example of artifact passing
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: artifact-example-
spec:
  entrypoint: artifact-example
  templates:
  - name: artifact-example
    steps:
    - - name: generate-artifact
        template: generate
      - name: consume-artifact
        template: consume
        arguments:
          artifacts:
          - name: result
            from: "{{steps.generate-artifact.outputs.artifacts.result}}"
  - name: generate
    script:
      image: python:3.8
      command: [python]
      source: |
        with open('/tmp/result.txt', 'w') as f:
          f.write('artifact data')
      outputs:
        artifacts:
        - name: result
          path: /tmp/result.txt
  - name: consume
    script:
      image: python:3.8
      command: [python]
      source: |
        with open('/tmp/result.txt') as f:
          print(f.read())
      inputs:
        artifacts:
        - name: result
          path: /tmp/result.txt
```
