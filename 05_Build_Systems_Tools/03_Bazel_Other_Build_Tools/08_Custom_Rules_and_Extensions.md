# Custom Rules and Extensions

## Overview
Learn to create custom Bazel rules using Starlark, and how to use aspects and providers.

## Example: Custom Rule (Starlark)
```python
# hello_rule.bzl
def _hello_rule_impl(ctx):
    ctx.actions.write(
        output=ctx.outputs.out,
        content="Hello from custom rule!"
    )
hello_rule = rule(
    implementation = _hello_rule_impl,
    outputs = {"out": "%{name}.txt"},
)
```
```python
# BUILD file
load(":hello_rule.bzl", "hello_rule")
hello_rule(
    name = "custom_hello",
)
```
