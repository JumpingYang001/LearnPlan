# Project: Scientific Computing Workflow

## Objective
Build workflows for scientific computations with checkpointing, resume, and result visualization.

## Steps
1. Run computation with checkpointing.
2. Resume from checkpoint if needed.
3. Visualize results.

## Example Workflow
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: scientific-compute-
spec:
  entrypoint: compute-pipeline
  templates:
  - name: compute-pipeline
    steps:
    - - name: compute
        template: compute
    - - name: checkpoint
        template: checkpoint
    - - name: resume
        template: resume
        when: "{{steps.checkpoint.outputs.result}} == 'resume'"
    - - name: visualize
        template: visualize
  - name: compute
    script:
      image: python:3.8
      command: [python]
      source: |
        print('Running computation')
        with open('/tmp/checkpoint.txt', 'w') as f:
          f.write('resume')
  - name: checkpoint
    script:
      image: python:3.8
      command: [python]
      source: |
        print('Checkpointing')
        with open('/tmp/checkpoint.txt') as f:
          result = f.read()
        with open('/tmp/checkpoint-result.txt', 'w') as f:
          f.write(result)
      outputs:
        result:
          name: result
          valueFrom:
            path: /tmp/checkpoint-result.txt
  - name: resume
    script:
      image: python:3.8
      command: [python]
      source: |
        print('Resuming computation')
  - name: visualize
    script:
      image: python:3.8
      command: [python]
      source: |
        print('Visualizing results')
```
