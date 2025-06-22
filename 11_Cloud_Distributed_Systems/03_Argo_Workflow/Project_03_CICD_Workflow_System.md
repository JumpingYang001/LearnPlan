# Project: CI/CD Workflow System

## Objective
Build a complete CI/CD system using Argo Workflows.

## Steps
1. Build application.
2. Run tests.
3. Deploy to environment.
4. Implement approval gates and rollback.

## Example Workflow
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: cicd-
spec:
  entrypoint: cicd-pipeline
  templates:
  - name: cicd-pipeline
    steps:
    - - name: build
        template: build
    - - name: test
        template: test
    - - name: deploy
        template: deploy
    - - name: approval
        template: approval-gate
    - - name: rollback
        template: rollback
        when: "{{steps.test.outputs.result}} == 'fail'"
  - name: build
    script:
      image: node:14
      command: [npm]
      args: [install]
  - name: test
    script:
      image: node:14
      command: [npm]
      args: [test]
      outputs:
        result:
          name: result
          valueFrom:
            path: /tmp/test-result.txt
  - name: deploy
    script:
      image: node:14
      command: [npm]
      args: [run, deploy]
  - name: approval-gate
    suspend: {}
  - name: rollback
    script:
      image: node:14
      command: [npm]
      args: [run, rollback]
```
