# Workflow Control Flow

## Conditional Execution
- Using `when` expressions
- Example of conditional step

## Loops and Recursion
- Using `withItems` for loops
- Example of looping steps

## Error Handling and Retries
- Specifying retries for steps
- Example of error handling

## Example
```yaml
# Example of a step with retry
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: retry-example-
spec:
  entrypoint: retry-step
  templates:
  - name: retry-step
    steps:
    - - name: hello
        template: whalesay
        retryStrategy:
          limit: 3
  - name: whalesay
    container:
      image: docker/whalesay
      command: [cowsay]
      args: ["hello"]
```
