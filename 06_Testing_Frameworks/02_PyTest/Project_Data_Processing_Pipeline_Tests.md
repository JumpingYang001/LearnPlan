# Project: Data Processing Pipeline Tests

## Description
Create tests for a data pipeline, use property-based testing, and parameterize with different data sets.

## Example
```python
import pytest
from hypothesis import given
import hypothesis.strategies as st

def process(data):
    return [x * 2 for x in data]

@given(st.lists(st.integers()))
def test_process_property(data):
    result = process(data)
    assert all(isinstance(x, int) for x in result)

@pytest.mark.parametrize("input_data,expected", [([1,2], [2,4]), ([0], [0])])
def test_process_examples(input_data, expected):
    assert process(input_data) == expected
```
