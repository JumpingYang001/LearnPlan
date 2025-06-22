# Project: Machine Learning Training Pipeline

## Objective
Develop a workflow for ML model training with hyperparameter tuning and artifact management.

## Steps
1. Prepare training data.
2. Run parallel training jobs with different hyperparameters.
3. Store trained models as artifacts.

## Example Workflow
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: ml-training-
spec:
  entrypoint: train-models
  templates:
  - name: train-models
    steps:
    - - name: prepare-data
        template: prepare-data
    - - name: train
        template: train-model
        arguments:
          parameters:
          - name: lr
            value: "{{item}}"
        withItems:
        - 0.01
        - 0.1
        - 1.0
  - name: prepare-data
    script:
      image: python:3.8
      command: [python]
      source: |
        # Prepare data
        print('Data prepared')
  - name: train-model
    inputs:
      parameters:
      - name: lr
    script:
      image: python:3.8
      command: [python]
      source: |
        lr = {{inputs.parameters.lr}}
        print(f'Training with lr={lr}')
        # Save model
        with open('/tmp/model.txt', 'w') as f:
          f.write(f'model with lr={lr}')
      outputs:
        artifacts:
        - name: model
          path: /tmp/model.txt
```
