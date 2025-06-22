# Parameterization

## Description
Shows how to use @pytest.mark.parametrize for multiple test cases and custom parameter generators.

## Example
```python
import pytest

@pytest.mark.parametrize("a,b,result", [(1, 2, 3), (2, 3, 5)])
def test_add(a, b, result):
    assert a + b == result
```
