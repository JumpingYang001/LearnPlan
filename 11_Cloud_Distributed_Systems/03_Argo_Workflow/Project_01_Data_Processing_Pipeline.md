# Project: Data Processing Pipeline

## Objective
Build a scalable data processing workflow using Argo Workflows.

## Steps
1. Ingest data files as input artifacts.
2. Process data in parallel steps (e.g., ETL, transformation).
3. Store processed data as output artifacts.

## Example Workflow
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: data-processing-
spec:
  entrypoint: process-data
  templates:
  - name: process-data
    steps:
    - - name: ingest
        template: ingest-data
      - name: transform
        template: transform-data
        arguments:
          artifacts:
          - name: input
            from: "{{steps.ingest.outputs.artifacts.data}}"
  - name: ingest-data
    script:
      image: python:3.8
      command: [python]
      source: |
        with open('/tmp/data.txt', 'w') as f:
          f.write('raw data')
      outputs:
        artifacts:
        - name: data
          path: /tmp/data.txt
  - name: transform-data
    script:
      image: python:3.8
      command: [python]
      source: |
        with open('/tmp/data.txt') as f:
          data = f.read().upper()
        with open('/tmp/processed.txt', 'w') as f:
          f.write(data)
      inputs:
        artifacts:
        - name: input
          path: /tmp/data.txt
      outputs:
        artifacts:
        - name: processed
          path: /tmp/processed.txt
```
